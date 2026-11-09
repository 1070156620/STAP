"""Source-only evidence gating and complete retrospective comparisons.

No target outcomes are supplied to candidate scoring or policy selection.
Previously inspected courses are explicitly exploratory, not fresh confirmation.
Existing models/results/paper files are never overwritten. No timing experiment.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from support_selection import quality
from support_selection.data import FEATURES, ROOT, sha256
from support_selection.experiment import descriptor, metrics, purge_students, select_top, validate_frame


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, compression="gzip" if path.name.endswith(".gz") else None,
                 float_format="%.17g")
    tmp.replace(path)


def dataset_frame(path):
    df = pd.read_csv(path, dtype={k: str for k in ("record_id", "student_id", "course", "cohort")},
                     parse_dates=["start", "label_available"])
    validate_frame(df)
    if df.dataset.nunique() != 1 or df.cutoff.nunique() != 1:
        raise ValueError("One dataset and one cutoff per input file are required")
    return df


def delta_method(delta):
    return "evidence_gate_delta_" + f"{100 * delta:g}".replace(".", "p") + "pp"


def select_family(evidence, budget, family):
    _, utilities, tags, names = evidence
    rows = [i for i, tag in enumerate(tags) if np.isclose(tag["budget"], budget)]
    allowed = [n for n in names if (family == "all" or
                (family == "rule" and n.startswith("rule_")) or
                (family == "model" and n.startswith(("lr_", "hgb_"))) or
                (family == "lr" and n.startswith("lr_")) or
                (family == "hgb" and n.startswith("hgb_")))]
    if not rows or not allowed:
        raise ValueError("Insufficient source evidence for family selection")
    mean = utilities[rows].mean(axis=0)
    # Preserve candidate-pool insertion order for deterministic ties.
    return max(allowed, key=lambda name: mean[names.index(name)])


def course_recall(frame, scores, budget):
    values = []
    for _, cohort in frame.groupby("cohort", sort=True):
        if not cohort.y.sum():
            continue
        ix = frame.index.get_indexer(cohort.index)
        chosen = select_top(np.asarray(scores)[ix], cohort.record_id, budget)
        values.append(float(cohort.y.to_numpy()[chosen].sum() / cohort.y.sum()))
    return float(np.mean(values)) if values else None


def lower_source_quantile(gains, protocol, seed):
    values = np.asarray(gains, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < protocol["minimum_source_validation_courses"]:
        return {"courses": int(len(values)), "mean_gain": None, "lower_quantile": None}
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(protocol["source_bootstrap"], len(values)), replace=True).mean(axis=1)
    return {"courses": int(len(values)), "mean_gain": float(values.mean()),
            "lower_quantile": float(np.quantile(means, protocol["source_lower_quantile"]))}


def evidence_gate(rule_name, model_name, stats, delta):
    lower = stats["lower_quantile"]
    switch = lower is not None and lower > delta + 1e-12
    return model_name if switch else rule_name


def source_crossfit(source, cols, cfg, protocol, cache):
    """Hold out a middle source course before selecting either candidate family.

    Candidate identities and utility-regressor fitting for middle validation are
    obtained from regenerated inner evidence excluding that validation course.
    The same source folds tune the archived adaptive-selector alpha; all of this
    remains inside the outer training set.
    """
    gains = {b: [] for b in cfg["budgets"]}
    alpha_scores = {a: [] for a in cfg["selector_alphas"]}
    log = []
    for course, fit, val in quality.source_folds(source, cfg, middle=True):
        if course in set(fit.course) or set(fit.student_id) & set(val.student_id):
            raise ValueError("Middle-split course/student leakage")
        evidence = quality.build_evidence(fit, cols, cfg, cache)
        if not len(evidence[0]):
            raise ValueError("Empty regenerated source evidence; no silent fold skipping")
        # Target labels are deliberately removed at the scoring boundary.
        score, _ = quality.score_pool(fit, val.drop(columns="y"), cols, cfg, cache)
        queries = {b: descriptor(fit, val.drop(columns="y"), b, cols) for b in cfg["budgets"]}
        choices = []
        for b in cfg["budgets"]:
            rule = select_family(evidence, b, "rule")
            model = select_family(evidence, b, "model")
            r0, r1 = course_recall(val, score[rule], b), course_recall(val, score[model], b)
            gain = None if r0 is None or r1 is None else r1 - r0
            if gain is not None:
                gains[b].append(gain)
            choices.append({"budget": b, "rule": rule, "model": model, "gain": gain})
        # Match the archived joint-capacity adaptive-alpha selection criterion.
        for alpha in alpha_scores:
            vals = []
            for b in cfg["budgets"]:
                selected = quality.select_names(evidence, queries[b], b, cfg, alpha=alpha)["chosen"]
                for _, cohort in val.groupby("cohort", sort=True):
                    if not cohort.y.sum():
                        continue
                    ix = val.index.get_indexer(cohort.index)
                    mask = select_top(score[selected][ix], cohort.record_id, b)
                    vals.append(float(cohort.y.to_numpy()[mask].sum() / cohort.y.sum()))
            if vals:
                alpha_scores[alpha].append(float(np.mean(vals)))
        log.append({"validation_course": str(course), "training_courses": sorted(fit.course.unique().tolist()),
                    "evidence_courses": sorted({str(t["course"]) for t in evidence[2]}),
                    "student_overlap": 0, "family_choices": choices})
    means = {a: float(np.mean(v)) for a, v in alpha_scores.items() if v}
    alpha = max(means, key=lambda a: (means[a], a)) if means else max(cfg["selector_alphas"])
    stats = {b: lower_source_quantile(gains[b], protocol, cfg["seed"]) for b in cfg["budgets"]}
    return stats, alpha, {"middle_folds": log, "alpha_scores": alpha_scores,
                          "chosen_alpha": alpha, "gain_unit": "recall fraction, not percentage points"}


def evaluate_fold(train, target, cols, cfg, protocol, cache):
    if set(train.student_id) & set(target.student_id) or set(train.course) & set(target.course):
        raise ValueError("Outer course/student leakage")
    if len(train) < cfg["min_train"] or len(target) < cfg["min_test"] or train.y.nunique() < 2:
        raise ValueError("Ineligible outer task; do not silently exclude it")
    stats, alpha, audit = source_crossfit(train, cols, cfg, protocol, cache)
    evidence = quality.build_evidence(train, cols, cfg, cache)
    target_x = target.drop(columns="y")
    scores, _ = quality.score_pool(train, target_x, cols, cfg, cache)
    scores["fixed_blend"] = (rankdata(scores["lr_C1"]) + rankdata(scores["rule_inactivity"])) / (2 * len(target))
    pred = target[["record_id", "student_id", "course", "cohort", "y"]].copy()
    for name, values in scores.items():
        pred["score__" + name] = values
    metrics_rows, decisions, diagnostic = [], [], []
    for b in cfg["budgets"]:
        adaptive = quality.select_names(evidence, descriptor(train, target_x, b, cols), b, cfg, alpha=alpha)
        rule, model = select_family(evidence, b, "rule"), select_family(evidence, b, "model")
        mapping = {"tuned_rule": rule, "tuned_lr": select_family(evidence, b, "lr"),
                   "tuned_hgb": select_family(evidence, b, "hgb"),
                   "source_fixed": select_family(evidence, b, "all"),
                   "adaptive_selector": adaptive["chosen"], "fixed_blend": "fixed_blend"}
        mapping.update({delta_method(d): evidence_gate(rule, model, stats[b], d) for d in protocol["delta_grid"]})
        selections = {name: np.zeros(len(target), dtype=bool) for name in mapping}
        for _, cohort in target.groupby("cohort", sort=True):
            ix = target.index.get_indexer(cohort.index)
            fixed_mask = select_top(scores[mapping["source_fixed"]][ix], cohort.record_id, b)
            rule_mask = select_top(scores[rule][ix], cohort.record_id, b)
            adaptive_mask = select_top(scores[adaptive["chosen"]][ix], cohort.record_id, b)
            for name, candidate in mapping.items():
                mask = select_top(scores[candidate][ix], cohort.record_id, b)
                if mask.sum() != fixed_mask.sum():
                    raise ValueError("Unequal recipient capacity")
                selections[name][ix] = mask
                m = metrics(cohort.y, scores[candidate][ix], cohort.record_id, b)
                row = {"budget": b, "cohort": str(cohort.cohort.iloc[0]), "method": name,
                       "candidate": candidate, **{k: m[k] for k in ["n", "positives", "selected", "tp", "fp", "fn", "recall", "precision"]}}
                for label, baseline in [("fixed", fixed_mask), ("rule", rule_mask), ("adaptive", adaptive_mask)]:
                    row["jaccard_vs_" + label] = float(np.sum(mask & baseline) / np.sum(mask | baseline))
                    row["net_tp_vs_" + label] = int(cohort.y.to_numpy()[mask].sum() - cohort.y.to_numpy()[baseline].sum())
                    row["extra_per_1000_vs_" + label] = 1000 * row["net_tp_vs_" + label] / len(cohort)
                metrics_rows.append(row)
        for name, mask in selections.items():
            pred[f"selected__{b:g}__{name}"] = mask.astype(np.uint8)
        candidate_recalls = {name: course_recall(target, scores[name], b) for name in evidence[3]}
        valid = {name: r for name, r in candidate_recalls.items() if r is not None}
        oracle_name = max(valid, key=valid.get) if valid else None
        fixed_r = candidate_recalls[mapping["source_fixed"]]
        adaptive_r = candidate_recalls[adaptive["chosen"]]
        diagnostic.append({"budget": b, "oracle_candidate_tie_broken": oracle_name,
                           "oracle_recall": valid.get(oracle_name), "source_fixed_recall": fixed_r,
                           "adaptive_recall": adaptive_r,
                           "pool_headroom_pp": None if oracle_name is None else 100 * (valid[oracle_name] - fixed_r),
                           "realized_adaptive_pp": None if adaptive_r is None else 100 * (adaptive_r - fixed_r),
                           "interpretation": "Retrospective target-label diagnostic; never a selectable policy"})
        decisions.append({"budget": b, "rule": rule, "source_selected_model": model,
                          "source_gain": stats[b], "mapping": mapping,
                          "target_labels_used_for_selection": False})
    audit.update({"outer_student_overlap": 0, "target_labels_sent_to_score_pool": False,
                  "source_courses": sorted(train.course.unique().tolist()),
                  "target_courses": sorted(target.course.unique().tolist())})
    return pd.DataFrame(metrics_rows), pred, decisions, audit, diagnostic


def experiment_config(protocol, name, seed):
    info = protocol["datasets"][name]
    cfg = {**protocol["base"], "seed": seed, "threads": protocol["threads"],
           "budgets": protocol["budgets"], "max_outer": None}
    for key in ("source_course_counts", "max_middle_courses"):
        cfg[key] = info[key]
    return cfg


def build_manifest(protocol, config_path, names):
    paths = [Path(__file__), Path(config_path), ROOT / "support_selection/quality.py",
             ROOT / "support_selection/experiment.py", ROOT / "support_selection/data.py"]
    for name in names:
        paths.append(ROOT / protocol["datasets"][name]["processed"])
        paths.append(ROOT / protocol["datasets"][name]["archived_primary"] / "candidate_headroom_diagnostic.csv")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing inputs: " + str(missing))
    return {"protocol": protocol, "datasets": names,
            "hashes": {str(p.relative_to(ROOT)): sha256(p) for p in paths},
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "pandas": pd.__version__, "sklearn": sklearn.__version__}}


def lock_output(out, manifest):
    out = Path(out).resolve()
    if not out.is_relative_to((ROOT / "outputs").resolve()) or out == (ROOT / "outputs").resolve():
        raise ValueError("Output must be a new subdirectory under this project's outputs")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol_manifest.json"
    if path.exists():
        if read_json(path) != manifest:
            raise ValueError("Input/code/config/environment differs from frozen run. Use a new OutputRoot.")
    else:
        if any(out.iterdir()):
            raise ValueError("Refusing to adopt a nonempty unrelated output directory")
        write_json(path, manifest)
        write_json(out / "registration.json", {"saved_at_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "Exploratory retrospective protocol lock before this run; not prospective preregistration"})
    return out


def headroom_decomposition(protocol, names, out):
    parts = []
    for name in names:
        p = ROOT / protocol["datasets"][name]["archived_primary"] / "candidate_headroom_diagnostic.csv"
        d = pd.read_csv(p, dtype={"test_course": str})
        if d.duplicated(["test_course", "budget"]).any() or set(d.budget) != {0.2}:
            raise ValueError("Expected one original primary row per target course")
        d["pool_headroom_pp"] = 100 * (d.oracle_course - d.global_validation)
        d["realized_adaptive_pp"] = 100 * (d.selector - d.global_validation)
        d["unrealized_vs_adaptive_pp"] = 100 * (d.oracle_course - d.selector)
        if (d.pool_headroom_pp < -1e-8).any():
            raise ValueError("Candidate-pool reference below Source-fixed")
        d["source_file"] = str(p.relative_to(ROOT))
        parts.append(d)
    courses = pd.concat(parts, ignore_index=True)
    write_csv(out / "diagnostics/headroom_by_course.csv", courses)
    summary = []
    for name, q in courses.groupby("dataset", sort=True):
        for delta in protocol["delta_grid"]:
            summary.append({"dataset": name, "courses": len(q), "margin_pp": 100 * delta,
                "mean_headroom_pp": q.pool_headroom_pp.mean(),
                "median_headroom_pp": q.pool_headroom_pp.median(),
                "mean_realized_adaptive_pp": q.realized_adaptive_pp.mean(),
                "courses_headroom_above_margin": int((q.pool_headroom_pp > 100 * delta + 1e-10).sum()),
                "courses_adaptive_above_margin": int((q.realized_adaptive_pp > 100 * delta + 1e-10).sum()),
                "note": "Target-label-dependent retrospective description; no model-selection thresholds fitted here"})
    write_csv(out / "diagnostics/headroom_summary.csv", pd.DataFrame(summary))
    print("HEADROOM DONE:", out / "diagnostics", flush=True)


def task_id(name, seed, course, nsource, draw):
    digest = hashlib.sha256(str(course).encode()).hexdigest()[:14]
    return f"{name}_{seed}_{nsource}_{draw}_{digest}"


def check_done(out, marker):
    if not marker.exists():
        return False
    value = read_json(marker)
    for relative, digest in value["files"].items():
        p = out / relative
        if not p.exists() or sha256(p) != digest:
            raise ValueError("Completed-fold output changed: " + str(p))
    return True


def run(protocol, names, out):
    expected = completed = reused = 0
    cache = ROOT / ".cache/quality_scores"
    with threadpool_limits(limits=protocol["threads"]):
        for name in names:
            df = dataset_frame(ROOT / protocol["datasets"][name]["processed"])
            df, cols = quality.transform_features(df, protocol["feature_mode"])
            for seed in protocol["seeds"]:
                cfg = experiment_config(protocol, name, seed)
                for course, nsource, draw, train, test in quality.plan_runs(df, cfg):
                    expected += 1
                    uid = task_id(name, seed, course, nsource, draw)
                    marker = out / "folds" / (uid + ".complete.json")
                    if check_done(out, marker):
                        reused += 1
                        completed += 1
                        print(f"REUSE {completed}: {uid}", flush=True)
                        continue
                    print(f"START {expected}: dataset={name} seed={seed} target={course} sources={nsource}", flush=True)
                    try:
                        frame, pred, choices, audit, diagnostic = evaluate_fold(train, test, cols, cfg, protocol, cache)
                        tag = {"dataset": name, "seed": seed, "test_course": str(course),
                               "source_courses": nsource, "draw": draw, "cutoff": int(df.cutoff.iloc[0])}
                        for key, value in tag.items():
                            frame[key] = value
                        files = {"metrics": out / "folds" / (uid + ".metrics.csv"),
                                 "predictions": out / "predictions" / (uid + ".csv.gz"),
                                 "decisions": out / "folds" / (uid + ".decisions.json")}
                        write_csv(files["metrics"], frame)
                        write_csv(files["predictions"], pred)
                        write_json(files["decisions"], {**tag, "choices": choices, "audit": audit,
                                                        "retrospective_diagnostic": diagnostic})
                        write_json(marker, {**tag, "complete": True,
                                   "files": {str(p.relative_to(out)): sha256(p) for p in files.values()}})
                    except Exception as exc:
                        write_json(out / "errors" / (uid + ".json"), {"task": uid, "error": str(exc),
                                   "status": "Run stopped; completed tasks retained, no silent omission"})
                        raise
                    completed += 1
                    write_json(out / "progress.json", {"completed": completed, "reused": reused,
                               "latest": tag, "status": "running"})
                    print(f"DONE {completed}: {uid}", flush=True)
    if completed != expected or not expected:
        raise ValueError("Outer-task coverage mismatch")
    write_json(out / "run_complete.json", {"complete": True, "expected_tasks": expected,
               "completed_tasks": completed, "reused_tasks": reused,
               "scope": protocol["study_status"], "performance_filtering": False})


def bootstrap_interval(values, count, seed):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return None, None
    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(count, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(boot, [0.025, 0.975]))


def summarize(protocol, out):
    markers = sorted((out / "folds").glob("*.complete.json"))
    if not markers:
        raise ValueError("No completed tasks to summarize")
    parts, switch_rows = [], []
    for marker in markers:
        check_done(out, marker)
        uid = marker.name.removesuffix(".complete.json")
        parts.append(pd.read_csv(out / "folds" / (uid + ".metrics.csv"), dtype={"test_course": str}))
        detail = read_json(out / "folds" / (uid + ".decisions.json"))
        for choice in detail["choices"]:
            for delta in protocol["delta_grid"]:
                method = delta_method(delta)
                switch_rows.append({k: detail[k] for k in ["dataset", "seed", "test_course"]} | {
                    "budget": choice["budget"], "method": method,
                    "switch_to_model": choice["mapping"][method].startswith(("lr_", "hgb_")),
                    "source_validation_courses": choice["source_gain"]["courses"],
                    "source_mean_gain": choice["source_gain"]["mean_gain"],
                    "source_lower_quantile": choice["source_gain"]["lower_quantile"]})
    metrics_df = pd.concat(parts, ignore_index=True)
    if metrics_df.duplicated(["dataset", "seed", "test_course", "budget", "cohort", "method"]).any():
        raise ValueError("Duplicate cohort metrics")
    keys = ["dataset", "seed", "test_course", "budget", "method"]
    mean_cols = ["recall", "precision", "jaccard_vs_fixed", "jaccard_vs_rule", "jaccard_vs_adaptive",
                 "extra_per_1000_vs_fixed", "extra_per_1000_vs_rule", "extra_per_1000_vs_adaptive"]
    course = metrics_df.groupby(keys, dropna=False)[mean_cols].mean().reset_index()
    counts = metrics_df.groupby(keys)[["n", "positives", "selected", "tp", "net_tp_vs_fixed",
                                       "net_tp_vs_rule", "net_tp_vs_adaptive"]].sum().reset_index()
    course = course.merge(counts, on=keys, validate="one_to_one")
    write_csv(out / "analysis/course_metrics.csv", course)
    seed_means = course.groupby(["dataset", "seed", "budget", "method"])[mean_cols].mean().reset_index()
    write_csv(out / "analysis/seed_summary.csv", seed_means)
    mean_sd = seed_means.groupby(["dataset", "budget", "method"])[["recall", "precision"]].agg(["mean", "std"])
    mean_sd.columns = ["_".join(c) for c in mean_sd.columns]
    write_csv(out / "analysis/seed_mean_sd.csv", mean_sd.reset_index())
    recipient_counts = course.groupby(["dataset", "seed", "budget", "method"])[[
        "n", "selected", "tp", "net_tp_vs_fixed", "net_tp_vs_rule", "net_tp_vs_adaptive"]].sum().reset_index()
    write_csv(out / "analysis/recipient_effects_per_seed.csv", recipient_counts.merge(
        seed_means.drop(columns=["recall", "precision"]), on=["dataset", "seed", "budget", "method"],
        validate="one_to_one"))
    write_csv(out / "analysis/switch_decisions.csv", pd.DataFrame(switch_rows))
    comparisons = []
    for (dataset, seed, budget), q in course.groupby(["dataset", "seed", "budget"]):
        pivot = q.pivot(index="test_course", columns="method", values="recall")
        for method in [delta_method(d) for d in protocol["delta_grid"]]:
            for comparator in ["source_fixed", "tuned_rule", "adaptive_selector"]:
                difference = 100 * (pivot[method] - pivot[comparator])
                for target_course, value in difference.items():
                    comparisons.append({"dataset": dataset, "seed": seed, "budget": budget,
                                        "test_course": target_course, "method": method,
                                        "comparator": comparator, "delta_pp": value})
    paired = pd.DataFrame(comparisons)
    write_csv(out / "analysis/paired_course_differences.csv", paired)
    # Average seeds within each target course before bootstrapping courses.
    grouped = paired.groupby(["dataset", "budget", "method", "comparator", "test_course"]).delta_pp.mean().reset_index()
    summary = []
    for (dataset, budget, method, comparator), q in grouped.groupby(["dataset", "budget", "method", "comparator"]):
        raw = q.delta_pp.to_numpy()
        v = raw[np.isfinite(raw)]
        lo, hi = bootstrap_interval(v, protocol["report_bootstrap"], protocol["seeds"][0])
        summary.append({"dataset": dataset, "budget": budget, "method": method, "comparator": comparator,
                        "courses": len(v), "courses_without_defined_recall": int(len(raw) - len(v)),
                        "mean_delta_pp": float(v.mean()) if len(v) else None, "ci_low_pp": lo, "ci_high_pp": hi,
                        "higher": int((v > 1e-10).sum()), "tied": int((np.abs(v) <= 1e-10).sum()),
                        "lower": int((v < -1e-10).sum()),
                        "is_primary_comparison": bool(np.isclose(budget, protocol["primary_budget"]) and
                          method == delta_method(protocol["primary_delta"]) and comparator == "source_fixed")})
    result = pd.DataFrame(summary)
    write_csv(out / "analysis/all_comparisons.csv", result)
    write_csv(out / "analysis/primary_comparison.csv", result.loc[result.is_primary_comparison])
    complete = read_json(out / "run_complete.json") if (out / "run_complete.json").exists() else {"complete": False}
    write_json(out / "analysis/report_status.json", {"complete": complete["complete"], "folds": len(markers),
               "all_comparisons_retained": True, "seed_repetitions_are_not_independent_courses": True,
               "interval_scope": "Conditional paired course bootstrap of seed-averaged differences; no multiplicity adjustment",
               "study_status": protocol["study_status"]})
    print("SUMMARY", "COMPLETE" if complete["complete"] else "PARTIAL", out / "analysis", flush=True)
    print(result.loc[result.is_primary_comparison, ["dataset", "mean_delta_pp", "ci_low_pp", "ci_high_pp", "higher", "tied", "lower"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["plan", "headroom", "run", "summarize", "all"], default="plan")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/decision_principle.json")
    parser.add_argument("--datasets", nargs="+", choices=["oulad", "kdd2015", "mooccubex"])
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/decision_principle_v1")
    args = parser.parse_args()
    protocol = read_json(args.config)
    names = args.datasets or list(protocol["datasets"])
    if len(names) != len(set(names)):
        raise ValueError("Repeated dataset selection")
    manifest = build_manifest(protocol, args.config, names)
    if args.stage == "plan":
        courses = {n: int(dataset_frame(ROOT / protocol["datasets"][n]["processed"]).course.nunique()) for n in names}
        print(json.dumps({"courses": courses, "seeds": protocol["seeds"],
              "outer_tasks": sum(courses.values()) * len(protocol["seeds"]),
              "capacities_per_task": protocol["budgets"], "delta_grid_pp": [100 * d for d in protocol["delta_grid"]],
              "primary_delta_pp": 100 * protocol["primary_delta"], "scope": protocol["study_status"],
              "out": str(args.out), "training_started": False}, ensure_ascii=False, indent=2))
        return
    out = lock_output(args.out, manifest)
    if args.stage in ("headroom", "all"):
        headroom_decomposition(protocol, names, out)
    if args.stage in ("run", "all"):
        run(protocol, names, out)
    if args.stage in ("summarize", "all"):
        summarize(protocol, out)


if __name__ == "__main__":
    main()
