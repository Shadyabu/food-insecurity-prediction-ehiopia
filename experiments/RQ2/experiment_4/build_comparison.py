"""RQ2 Experiment 4: five-way architecture comparison (XGBoost, LSTM,
TabICLv2, RandomForest, T-GCN) on the Ethiopia dataset. Extends
experiments/RQ2/experiment_2/build_comparison.py's 3-way table with this
folder's 4 RandomForest runs (level/delta x default/extended window,
mirroring the XGBoost arm exactly) plus T-GCN's own already-scored result
(experiments/RQ2/experiment_3), rather than editing the experiment_2 script
in place -- same "read a prior experiment's already-computed results"
pattern this project already uses (RQ2 Experiment 1 reading RQ1 Experiment
2's outputs; RQ2 Experiment 2 reading Experiment 1's).

T-GCN caveat, carried over from experiments/RQ2/experiment_3/report.md: its
predictions are 0-indexed discrete class labels (`observed`/`prediction`
already 0..4 for IPC phase 1..5), a single pooled default-window model per
lead, no delta-target variant, and no per-cluster livelihood-zone models
(node_livelihood_zone is a column in its own predictions.parquet, not
`lhz` -- renamed here for join compatibility). Its own report already
established it does not beat XGBoost or LSTM; included here so the
complete RQ2 picture is in one table, not to imply new comparability beyond
what discretized accuracy/weighted-F1 already supports across every
architecture in this project (same to_ipc_class() discretization as
experiment_2's build_comparison.py).
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
    ("TabICLv2", "classification", None, "../experiment_2/tabicl_level/predictions.parquet", "window"),
    ("RandomForest", "level", "default", "rf_level_default/predictions.parquet", None),
    ("RandomForest", "level", "extended", "rf_level_extended/predictions.parquet", None),
    ("RandomForest", "delta", "default", "rf_delta_default/predictions.parquet", None),
    ("RandomForest", "delta", "extended", "rf_delta_extended/predictions.parquet", None),
]

TGCN_PATH = "../experiment_3/tgcn_ce/predictions.parquet"


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_tabular_sources():
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


def load_tgcn():
    """T-GCN predictions (experiments/RQ2/experiment_3/tgcn_ce/predictions.parquet)
    already carry true_class/pred_class as 0-indexed discrete classes (0..4
    == IPC phase 1..5, per that experiment's evaluate.py) and lhz/window
    columns matching this table's own naming already -- only the class
    offset and column subset/rename (true_class/pred_class ->
    observed/prediction) need handling here."""
    df = pd.read_parquet(TGCN_PATH)
    df = df[df["window"] == "default"].copy()
    df = df.rename(columns={"true_class": "observed", "pred_class": "prediction"})
    df = df[["zone_code", "lhz", "lead", "observed", "prediction"]].copy()
    df["observed"] = df["observed"] + 1
    df["prediction"] = df["prediction"] + 1
    df["window"] = "default"
    df["model"] = "T-GCN"
    df["target_mode"] = "classification"
    return df


def load_all():
    frames = [load_tabular_sources()]
    try:
        frames.append(load_tgcn())
    except FileNotFoundError:
        print(f"warning: {TGCN_PATH} not found, skipping T-GCN in this comparison")
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

    print(summary.sort_values("mean_f1_weighted", ascending=False).round(4).to_string(index=False))
    print()
    print(summary_cluster.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
