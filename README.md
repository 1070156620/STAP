# STAP

Source-Tuned Adaptive Prioritization for Capacity-Constrained Educational Support.

STAP selects a ranking policy and its selection-evidence mode using nested source-course validation, then generates a top-k potential support-recipient list at a fixed capacity. Future target outcomes are reserved for evaluation. This repository contains implementation and configurations, not datasets or reported experiment outputs.

## Installation

Python 3.10+ is specified by the project. The original experiments used the versions recorded in `requirements-lock.txt`; version changes can change results and score-cache keys.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

For the recorded environment, install `requirements-lock.txt` instead of the version-range requirements, on a compatible Python version.

## Data

Obtain datasets from their providers and follow their terms; no dataset is distributed here.

- OULAD: https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset
- KDD Cup 2015: https://moocdata.cn/challenges/kddcup-2015
- Optional exploratory MOOCCubeX: https://github.com/THU-KEG/MOOCCubeX

Expected raw KDD files under `data/raw/kdd2015/` (subdirectories supported): `enrollment_train.csv`, `truth_train.csv`, `log_train.csv`, `date.csv`. Competition submission templates and unknown test labels are not ground truth.

```powershell
python -m support_selection --help
python -m support_selection download
python -m support_selection prepare --dataset oulad --config configs/full.json
python -m support_selection prepare --dataset kdd2015 --config configs/kdd_external.json
```

## Source-based configuration search

Prepare the processed data referenced in `configs/decision_principle.json` before training. The following commands select the two core datasets explicitly:

```powershell
python selector_optimization.py --stage plan --datasets oulad kdd2015
python selector_optimization.py --stage all --datasets oulad kdd2015
```

The plan stage does not replace preparation or training. Dataset paths in configurations refer to locally generated inputs. No training was rerun during repository preparation.

## Implementation map

| File | Role |
|---|---|
| `support_selection/data.py` | Dataset preparation and feature construction |
| `support_selection/experiment.py` | Splits, descriptors and capacity-constrained selection |
| `support_selection/quality.py` | Candidate fitting, nested validation and utility estimation |
| `selector_optimization.py` | Source-only configuration search and frozen outer-task evaluation |
| `selector_replacement_suite.py` | Complete experiment-family orchestration |
| `metric_refinement.py` | Final source recall/AP selection policy and expanded metrics |
| `published_baselines.py` | Random forest and linear SVM comparator implementation |
| `stap_course_stability.py` | Course-level and cross-seed recipient-list analysis |
| `stap_mechanism_checks.py`, `kdd_matched_ablation.py` | Internal component analyses |

`metric_refinement.py` deliberately requires previously generated continuous score caches and replacement-suite records; it stops if those inputs are absent. Later analysis scripts also require locally generated outputs. These artifacts are excluded from this source-only repository. Follow each command's `--help` and its configuration dependencies rather than running every script on an empty directory. The complete replacement suite includes optional MOOCCubeX tasks.

## Paper protocol

The primary tasks are OULAD subsequent formal withdrawal and KDD2015 competition-defined dropout. Source tuning uses Recall@20%, with AP breaking source-recall ties in the final metric-refinement stage. The configuration grid has 142 entries. Five recorded seeds are 20261028–20261032; the manuscript's displayed primary run is seed 20261032 and repeated-run results are reported separately. Recall, precision, TP, AP, Lift and recipient-set overlap describe offline prioritization, not delivered intervention effects.

## Repository scope

Dataset files, prediction caches, logs, results, unit-test scripts, manuscript-editing helpers, LaTeX files and local virtual environments are intentionally excluded. Source modules are copied without changing their algorithmic implementation; `SOURCE_MANIFEST.json` records source hashes. No open-source license is assigned here: the author should choose a license before granting reuse permissions. The repository has not been uploaded to GitHub.
