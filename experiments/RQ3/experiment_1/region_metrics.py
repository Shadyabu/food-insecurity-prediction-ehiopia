"""RQ3 per-region (zone / livelihood-zone) predictive-performance scoring,
using the best RQ2 model: the 4-way averaging ensemble XGB+RF+LSTM+TabICL
(default test window), which CLAUDE.md Sec 2 documents as "the best all-zones
number in this project to date" (mean weighted F1 0.7218) and the only
combination confirmed (by exhaustive 26-combination search, RQ2 Experiment 5)
to be the global optimum on both the default and extended test windows.

Pooling choice (documented, per CLAUDE.md Sec 4's "comment the statistical
choices" convention): the ensemble's test set is 9 target months x 7 lead
times per zone = 63 rows/zone. A single lead for a single zone only has 9
test months -- far too few for a stable weighted F1. Metrics here therefore
pool every (target month, lead) row for a zone (or livelihood-zone cluster)
into one F1/balanced-accuracy pair, matching the one-completeness-score-per-
zone granularity from completeness.py so the two can be correlated 1:1. This
is coarser than the project's usual "mean F1 across leads" convention
(build_all_combinations.py's overall_summary()) -- reported as a limitation,
not hidden -- but is the only pooling that survives the small-N problem here.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

REPO_ROOT = Path(__file__).resolve().parents[3]
PREDICTIONS_PATH = (
    REPO_ROOT / "experiments" / "RQ2" / "experiment_5" / "all_combinations"
    / "predictions" / "XGB_RF_LSTM_TabICL.parquet"
)
MIN_TEST_OBS = 10  # minimum pooled test rows for a region's F1 to be reported, not flagged unstable


def to_ipc_class(x):
    """Round continuous ipc_continuous predictions to the 1-5 IPC phase scale,
    same convention as build_all_combinations.py's to_ipc_class() -- the
    ensemble's members are regressors (XGB/RF/LSTM) or already-discrete
    classifiers (TabICLv2) averaged together as continuous values.
    """
    return np.clip(np.round(x), 1, 5).astype(int)


def to_ipc_class_3plus(x):
    """1/2/3+ (Crisis-or-worse) collapse -- added 2026-09-02 for the
    2020-2024-window RQ3 rebuild, same round-then-clip logic as
    to_ipc_class(), clipped at 3 instead of 5. See
    experiments/RQ2/experiment_5/score_1_2_3plus_coverage.py's identical
    function for the project-wide convention this matches.
    """
    return np.clip(np.round(x), 1, 3).astype(int)


def load_predictions(predictions_path=None, discretize=to_ipc_class):
    """predictions_path/discretize default to the original 2020-2022,
    1-5-scale behavior -- passing either is purely additive, existing
    callers (run_rq3_analysis.py) are unaffected."""
    path = predictions_path if predictions_path is not None else PREDICTIONS_PATH
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df["y_true"] = discretize(df["observed"])
    df["y_pred"] = discretize(df["prediction"])
    return df


def _score_group(sub):
    n = len(sub)
    n_unique_true = sub["y_true"].nunique()
    # A zone whose true class never varies across its pooled test rows scores
    # a trivial, near-1.0 weighted F1 regardless of model skill (constant-
    # target zones checked empirically: mean F1 0.99 vs. 0.57 for zones with
    # >1 true class, CLAUDE.md's report.md-style verify-before-trust
    # convention) -- flagged per-region so it can be excluded in a robustness
    # re-run rather than silently inflating/diluting the completeness-F1
    # correlation with zones where F1 isn't really measuring predictive skill.
    if n < MIN_TEST_OBS:
        return {
            "n_test_obs": n,
            "n_unique_true_classes": n_unique_true,
            "trivial_constant_target": n_unique_true <= 1,
            "excluded": True,
            "f1_weighted": np.nan,
            "balanced_accuracy": np.nan,
        }
    return {
        "n_test_obs": n,
        "n_unique_true_classes": n_unique_true,
        "trivial_constant_target": n_unique_true <= 1,
        "excluded": False,
        "f1_weighted": f1_score(sub["y_true"], sub["y_pred"], average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(sub["y_true"], sub["y_pred"]),
    }


def score_by_zone(pred_df):
    rows = []
    for zone_code, sub in pred_df.groupby("zone_code"):
        row = {"zone_code": zone_code}
        row.update(_score_group(sub))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("zone_code").reset_index(drop=True)
    return out


def score_by_lhz(pred_df):
    rows = []
    for lhz, sub in pred_df.groupby("lhz"):
        row = {"lhz": lhz}
        row.update(_score_group(sub))
        rows.append(row)
    out = pd.DataFrame(rows).sort_values("lhz").reset_index(drop=True)
    return out
