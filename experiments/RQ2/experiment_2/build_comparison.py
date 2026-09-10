"""RQ2 Experiment 2: three-way architecture comparison (XGBoost, LSTM,
TabICLv2) on the Ethiopia dataset. Extends experiment_1's own XGBoost-vs-LSTM
comparison (experiments/RQ2/experiment_1/build_comparison.py) with the
TabICLv2 arm built in this folder, rather than editing that script in place --
experiment_1 is its own already-reported deliverable, and this mirrors the
existing pattern of one experiment's script reading a prior experiment's
already-computed results (e.g. this repo's own RQ2 Experiment 1 reading RQ1
Experiment 2's outputs).

Pools predictions from 7 already-run result folders (4 XGBoost runs, 2 LSTM
runs -- the same 6 experiment_1 uses -- plus this folder's 1 TabICLv2 run)
into one long table, computes weighted-F1/accuracy (CLAUDE.md Sec 7's primary
metric, nearest-integer IPC phase discretization -- same method as
experiments/RQ1/experiment_2/build_classification_metrics.py and
experiment_1's own build_comparison.py, so numbers are directly comparable to
both reports), and writes a single 3-way comparison report.

Caveat carried over from run_tabicl_ethiopia.py's own docstring: TabICLv2's
`observed`/`prediction` are already on the discrete 1-5 IPC-phase scale
(it is a classifier, fit on ipc_phase_20pct), not the continuous
ipc_continuous scale XGBoost/LSTM regress on. to_ipc_class() below is a
no-op for TabICL rows (already integer), so accuracy/f1_weighted stay
directly comparable across all three architectures -- but TabICL's mae/r2
columns are computed on that discretized scale, not the continuous one, and
are therefore NOT comparable to XGBoost/LSTM's mae/r2 in this same table.
Read only accuracy/f1_weighted across architectures; mae/r2 within an
architecture only.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]

SOURCES = [
    ("XGBoost", "level", "default", "../../RQ1/experiment_2/unhcr_level/predictions.parquet", None),
    ("XGBoost", "level", "extended", "../../RQ1/experiment_2/rq2_level_test2020to2024/predictions.parquet", None),
    ("XGBoost", "delta", "default", "../../RQ1/experiment_2/rq2_delta_default/predictions.parquet", None),
    ("XGBoost", "delta", "extended", "../../RQ1/experiment_2/rq2_delta_test2020to2024/predictions.parquet", None),
    ("LSTM", "level", None, "../experiment_1/lstm_level/predictions.parquet", "window"),
    ("LSTM", "delta", None, "../experiment_1/lstm_delta/predictions.parquet", "window"),
    ("TabICLv2", "classification", None, "tabicl_level/predictions.parquet", "window"),
]


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_all():
    frames = []
    for model, target_mode, fixed_window, path, window_col in SOURCES:
        df = pd.read_parquet(path)
        df = df[["zone_code", "lhz", "lead", "observed", "prediction"] +
                ([window_col] if window_col else [])].copy()
        if window_col:
            df = df.rename(columns={window_col: "window"})
        else:
            df["window"] = fixed_window
        df["model"] = model
        df["target_mode"] = target_mode
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def score_group(g):
    y_true = to_ipc_class(g["observed"])
    y_pred = to_ipc_class(g["prediction"])
    return pd.Series({
        "n": len(g),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mae": mean_absolute_error(g["observed"], g["prediction"]),
        "r2": r2_score(g["observed"], g["prediction"]) if g["observed"].nunique() > 1 else np.nan,
    })


def main():
    df = load_all()

    per_lead = (df.groupby(["model", "target_mode", "window", "lead"])
                  .apply(score_group, include_groups=False).reset_index())
    per_cluster = (df.groupby(["model", "target_mode", "window", "lhz", "lead"])
                     .apply(score_group, include_groups=False).reset_index())
    per_lead.to_csv("comparison_metrics_per_lead.csv", index=False)
    per_cluster.to_csv("comparison_metrics_per_cluster.csv", index=False)

    summary = (per_lead.groupby(["model", "target_mode", "window"])
               .agg(mean_accuracy=("accuracy", "mean"), mean_f1_weighted=("f1_weighted", "mean"),
                    mean_mae=("mae", "mean"), mean_r2=("r2", "mean"))
               .reset_index())
    summary_cluster = (per_cluster.groupby(["model", "target_mode", "window", "lhz"])
                        .agg(mean_accuracy=("accuracy", "mean"), mean_f1_weighted=("f1_weighted", "mean"),
                             mean_mae=("mae", "mean"), mean_r2=("r2", "mean"))
                        .reset_index())
    summary.to_csv("comparison_summary.csv", index=False)
    summary_cluster.to_csv("comparison_summary_per_cluster.csv", index=False)

    print(summary.round(4).to_string(index=False))
    print()
    print(summary_cluster.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
