"""RQ2 Experiment 1: XGBoost vs LSTM architecture comparison on the Ethiopia
dataset. Pools predictions from 6 already-run result folders (4 XGBoost runs
via ../../RQ1/experiment_2/run_model_ethiopia.py, 2 LSTM runs via
run_lstm_ethiopia.py -- level and delta target mode, each scored on both the
default Busker-parity test window (2020-01..2022-12) and an extended window
(2020-01..2024-12, same trained model)) into one long table, computes
weighted-F1/accuracy (CLAUDE.md Sec 7's primary metric, nearest-integer IPC
phase discretization -- same method as
experiments/RQ1/experiment_2/build_classification_metrics.py, so numbers are
directly comparable to that report), and writes a single comparison report.
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
    ("LSTM", "level", None, "lstm_level/predictions.parquet", "window"),
    ("LSTM", "delta", None, "lstm_delta/predictions.parquet", "window"),
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
