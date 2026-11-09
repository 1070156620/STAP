"""Chunked feature engineering. Outcomes are never used to compute features."""
from pathlib import Path
import hashlib
import json
import zipfile
import urllib.request
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
KEYS = ["code_module", "code_presentation", "id_student"]
FEATURES = ["log_clicks", "log_events", "log_active_days", "log_recent_clicks",
            "recent_fraction", "inactivity_fraction"]

def enrollment_key(df):
    return df.code_module.astype(str)+'|'+df.code_presentation.astype(str)+'|'+df.id_student.astype(str)

def dump_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str,
                               allow_nan=False), encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def locate(root, name):
    hits = list(Path(root).rglob(name))
    if len(hits) != 1:
        raise ValueError(f"Expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]

def download_oulad():
    raw = ROOT / "data/raw/oulad"; raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "oulad.zip"
    if not archive.exists():
        tmp = archive.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        tmp.replace(archive)
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            dest = (raw / member.filename).resolve()
            if not dest.is_relative_to(raw.resolve()): raise ValueError("Unsafe ZIP entry")
        z.extractall(raw)
    dump_json(raw / "download_manifest.json", {"url": URL, "sha256": sha256(archive),
              "license": "CC BY 4.0", "dataset_doi": "10.24432/C5KK69"})
    return raw

def add_activity_features(base, accum, cutoff):
    """accum rows: index record_id, clicks/events/active/recent/last."""
    a = accum.reindex(base.record_id).copy()
    for col in ["clicks", "events", "active", "recent"]: a[col] = a[col].fillna(0)
    a["last"] = a["last"].fillna(-1)
    base = base.copy()
    base["log_clicks"] = np.log1p(a.clicks.to_numpy())
    base["log_events"] = np.log1p(a.events.to_numpy())
    base["log_active_days"] = np.log1p(a.active.to_numpy())
    base["log_recent_clicks"] = np.log1p(a.recent.to_numpy())
    base["recent_fraction"] = a.recent.to_numpy()/np.maximum(a.clicks.to_numpy(), 1)
    base["inactivity_fraction"] = (cutoff-a["last"].to_numpy())/(cutoff+1)
    base["cutoff"] = cutoff
    return base

def prepare_oulad(raw, cutoffs, outcome="withdrawal", chunksize=150000):
    info = pd.read_csv(locate(raw, "studentInfo.csv"), dtype={k:str for k in KEYS})
    reg = pd.read_csv(locate(raw, "studentRegistration.csv"), dtype={k:str for k in KEYS},na_values=['?'])
    for c in ['date_registration','date_unregistration']:
        reg[c]=pd.to_numeric(reg[c],errors='raise')
    if info.duplicated(KEYS).any() or reg.duplicated(KEYS).any():
        raise ValueError("Duplicate enrollment keys; inspect source, do not silently drop")
    base = info.merge(reg[KEYS+["date_registration", "date_unregistration"]],
                      on=KEYS, validate="one_to_one")
    if len(base) != len(info): raise ValueError("Missing registration join")
    base["record_id"] = enrollment_key(base)
    base["student_id"] = base.id_student
    base["course"] = base.code_module
    base["cohort"] = base.code_module + "|" + base.code_presentation
    def when(p):
        return pd.Timestamp(year=int(p[:4]), month={"B":2,"J":10}[p[-1]], day=1)
    base["start"] = base.code_presentation.map(when)
    courses = pd.read_csv(locate(raw,"courses.csv"),dtype={"code_module":str,"code_presentation":str})
    base=base.merge(courses[["code_module","code_presentation","module_presentation_length"]],
                    on=["code_module","code_presentation"],validate="many_to_one")
    if base.module_presentation_length.isna().any():raise ValueError("Missing course duration")
    # Only start MONTH is documented. A 31-day buffer bounds uncertain start-day placement.
    # Train final-outcome labels only once the full course has certainly finished.
    base['label_available']=base.start+pd.to_timedelta(base.module_presentation_length+31,unit='D')
    if outcome == "withdrawal": base["y"] = (base.final_result == "Withdrawn").astype(int)
    elif outcome == "unsuccessful": base["y"] = base.final_result.isin(["Withdrawn","Fail"]).astype(int)
    else: raise ValueError("Unknown outcome")
    base["dataset"] = "oulad"; base["outcome"] = outcome
    sums = {c: [] for c in cutoffs}; daily = {c: [] for c in cutoffs}
    observed_rows = 0
    for block in pd.read_csv(locate(raw,"studentVle.csv"), dtype={k:str for k in KEYS}, chunksize=chunksize):
        observed_rows += len(block)
        block["record_id"] = enrollment_key(block)
        if (block.sum_click < 0).any(): raise ValueError("Negative clicks")
        for c in cutoffs:
            q = block.loc[block.date.between(0,c),["record_id","date","sum_click"]].copy()
            q["recent"] = q.sum_click.where(q.date > c-7,0)
            q["events"] = 1
            sums[c].append(q.groupby("record_id").agg(clicks=("sum_click","sum"),
                          events=("events","sum"), recent=("recent","sum"), last=("date","max")))
            # Deduplicate days across chunk boundaries after concatenation.
            daily[c].append(q[["record_id","date"]].drop_duplicates())
        if observed_rows % (chunksize*10)==0:print(f'Processed {observed_rows:,} raw VLE rows',flush=True)
    reports=[]
    for c in cutoffs:
        totals = pd.concat(sums[c]).groupby(level=0).agg({"clicks":"sum","events":"sum","recent":"sum","last":"max"})
        days = pd.concat(daily[c]).drop_duplicates().groupby("record_id").size()
        totals["active"] = days
        registered = base.date_registration.notna() & (base.date_registration <= c)
        still_at_risk = base.date_unregistration.isna() | (base.date_unregistration > c)
        keep = registered & still_at_risk
        f = add_activity_features(base.loc[keep], totals, c)
        # Availability of assessment scores at submission is not guaranteed; omit all scores by default.
        f = f[["dataset","outcome","record_id","student_id","course","cohort","start","label_available","cutoff","y"]+FEATURES]
        dest=ROOT/f"data/processed/oulad_{outcome}_{c}.csv.gz";dest.parent.mkdir(parents=True,exist_ok=True)
        f.to_csv(dest,index=False)
        reports.append({"cutoff":c,"input_enrollments":len(base),"retained":len(f),
                        "registration_missing_or_late":int((~registered).sum()),
                        "already_withdrawn":int((~still_at_risk).sum()),"positive":int(f.y.sum()),
                        "path":str(dest),"sha256":sha256(dest)})
    dump_json(ROOT/f"data/processed/oulad_{outcome}_audit.json",
              {"outcome":outcome,"log_rows":observed_rows,"cohorts":reports,
              "source_hashes":{n:sha256(locate(raw,n)) for n in ["studentInfo.csv","studentRegistration.csv","studentVle.csv","courses.csv"]},
               "feature_window":"0 <= course day <= cutoff; final_result and withdrawal dates only determine labels/eligibility",
              "no_assessment_score_availability_assumption":True,
              "calendar_assumption":"B=Feb 1 lower bound, J=Oct 1 lower bound; final labels mature at lower-bound start + recorded duration + 31 days"})
    return reports

def prepare_kdd(raw, cutoffs, chunksize=150000):
    """Official labelled TRAIN partition only. Never consume released test labels."""
    enroll = pd.read_csv(locate(raw,"enrollment_train.csv"),dtype=str)
    labels = pd.read_csv(locate(raw,"truth_train.csv"),header=None,names=["enrollment_id","y"],dtype=str)
    # Accept documented optional header, but validate binary labels.
    if labels.iloc[0].enrollment_id == "enrollment_id": labels=labels.iloc[1:].copy()
    if not labels.y.isin(["0","1"]).all(): raise ValueError("KDD labels must be 0/1")
    labels.y=labels.y.astype(int)
    dates=pd.read_csv(locate(raw,"date.csv"),dtype=str)
    dates=dates.rename(columns={"from":"start","to":"end"})
    if not {"course_id","start","end"}.issubset(dates):
        raise ValueError("date.csv must have course_id, from, to (or start,end); do not infer starts from future logs")
    dates.start=pd.to_datetime(dates.start);dates.end=pd.to_datetime(dates.end)
    if enroll.enrollment_id.duplicated().any() or labels.enrollment_id.duplicated().any() or dates.course_id.duplicated().any():
        raise ValueError("Duplicate KDD identifiers")
    base=enroll.merge(labels,on="enrollment_id",validate="one_to_one").merge(dates,on="course_id",validate="many_to_one")
    if len(base)!=len(enroll):raise ValueError("KDD labels/date join lost enrollments")
    base["record_id"]=base.enrollment_id;base["student_id"]=base.username
    base["course"]=base.course_id;base["cohort"]=base.course_id;base["dataset"]="kdd2015"
    base["outcome"]="competition_dropout"
    base['label_available']=base.end+pd.Timedelta(days=11)
    lookup=base.set_index("record_id").start
    sums={c:[] for c in cutoffs};daily={c:[] for c in cutoffs}
    for q in pd.read_csv(locate(raw,"log_train.csv"),dtype={"enrollment_id":str},chunksize=chunksize):
        q["record_id"]=q.enrollment_id
        # Calendar course start is mandatory, supplied separately and not inferred from full history.
        q["date"]=(pd.to_datetime(q.time).dt.normalize()-q.record_id.map(lookup)).dt.days
        for c in cutoffs:
            w=q.loc[q.date.between(0,c),["record_id","date"]].copy()
            w["clicks"]=1;w["recent"]=(w.date > c-7).astype(int)
            sums[c].append(w.groupby("record_id").agg(clicks=("clicks","sum"),events=("clicks","sum"),recent=("recent","sum"),last=("date","max")))
            daily[c].append(w[["record_id","date"]].drop_duplicates())
    reports=[]
    for c in cutoffs:
        totals=pd.concat(sums[c]).groupby(level=0).agg({"clicks":"sum","events":"sum","recent":"sum","last":"max"})
        totals["active"]=pd.concat(daily[c]).drop_duplicates().groupby("record_id").size()
        # Ongoing courses only; exact dropout time is unavailable in this dataset.
        keep=(base.end-base.start).dt.days > c
        f=add_activity_features(base.loc[keep],totals,c)
        f=f[["dataset","outcome","record_id","student_id","course","cohort","start","label_available","cutoff","y"]+FEATURES]
        dest=ROOT/f"data/processed/kdd2015_competition_dropout_{c}.csv.gz";dest.parent.mkdir(parents=True,exist_ok=True);f.to_csv(dest,index=False)
        reports.append({"cutoff":c,"retained":len(f),"path":str(dest),"sha256":sha256(dest)})
    dump_json(ROOT/"data/processed/kdd2015_audit.json",{"cohorts":reports,
              "source_hashes":{n:sha256(locate(raw,n)) for n in ["enrollment_train.csv","truth_train.csv","date.csv","log_train.csv"]},
              "warning":"External methodological replication, NOT frozen-model transfer; KDD competition dropout differs from OULAD withdrawal. No exact dropout times or registration times; enrollment eligibility at cutoff cannot be fully established. Events and OULAD clicks are different measures. Never pool endpoint estimates."})
    return reports
