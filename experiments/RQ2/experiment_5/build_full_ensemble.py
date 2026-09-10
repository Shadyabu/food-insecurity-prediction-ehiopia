"""Extension of build_ensemble.py: average ALL FIVE RQ2 architectures
(not just XGBoost+RandomForest) into one ensemble, then run leave-one-out
ablations -- drop each architecture in turn and re-score the remaining
4-way average -- to see which members help vs. hurt the blend.

Each architecture's single best/default-established variant is used, all
on the shared default test window (2020-2022, 828 rows/lead, confirmed
identical (time, zone_code, lead) key set across all five prediction
files before any averaging happens):
  - XGBoost:      unhcr_delta_noclimate   (RQ1/experiment_2 section 7.4 best)
  - RandomForest: rf_delta_noclimate      (RQ2/experiment_4, RF's own best)
  - LSTM:         lstm_delta              (RQ2/experiment_1, delta-target rescue)
  - TabICLv2:     tabicl_level            (RQ2/experiment_2, only variant built)
  - T-GCN:        tgcn_ce                 (RQ2/experiment_3, best of the paper's
                                            own hyperparameter ablation, CE head)

TabICLv2 and T-GCN are natively classifiers (integer IPC class, not a
continuous score) -- their "prediction" is already a class 1-5 (T-GCN's
0-indexed pred_class + 1). Averaged directly against the three regressors'
continuous predictions, exactly like build_ensemble.py already treats
XGBoost/RandomForest's continuous output before to_ipc_class() rounds it --
same discretize-at-the-end convention, extended to 5 members instead of 2.

This directly contradicts build_ensemble.py's own stated reason for
excluding these three ("no proven edge anywhere, and different feature
representations would need re-running before a like-for-like blend is
possible") -- run here anyway per explicit project-owner request, to
empirically test whether that concern actually costs performance. The
leave-one-out ablation is exactly the tool to answer that.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]

MEMBERS = {
    "XGBoost": "../../RQ1/experiment_2/unhcr_delta_noclimate/predictions.parquet",
    "RandomForest": "../experiment_4/rf_delta_noclimate/predictions.parquet",
    "LSTM": "../experiment_1/lstm_delta/predictions.parquet",
    "TabICLv2": "../experiment_2/tabicl_level/predictions.parquet",
    "T-GCN": "../experiment_3/tgcn_ce/predictions.parquet",
}


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_member(name, path):
    df = pd.read_parquet(path)
    if "window" in df.columns:
        df = df[df["window"] == "default"].copy()
    df["time"] = pd.to_datetime(df["time"])

    if name == "T-GCN":
        # 0-indexed class -> same 1-5 IPC scale as every other member.
        df["observed"] = df["true_class"] + 1
        df["prediction"] = df["pred_class"] + 1
        df["base1_preds"] = np.nan  # persistence baseline not carried in this file
    key = ["time", "zone_code", "lead"]
    return df.set_index(key).sort_index()[["lhz", "observed", "prediction", "base1_preds"]]


def score_subset(y_true, y_pred, observed_raw=None, pred_raw=None):
    out = {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if observed_raw is not None and pred_raw is not None:
        out["mae"] = mean_absolute_error(observed_raw, pred_raw)
        out["r2"] = r2_score(observed_raw, pred_raw) if observed_raw.nunique() > 1 else np.nan
    return out


def build_table(df, mask, label):
    sub = df[mask]
    rows = []
    for lead in LEADS:
        s = sub[sub["lead"] == lead]
        if len(s) == 0:
            continue
        y_true = to_ipc_class(s["observed"])
        y_pred = to_ipc_class(s["prediction"])
        r = score_subset(y_true, y_pred, s["observed"], s["prediction"])
        r["lead"] = lead
        rows.append(r)
    out = pd.DataFrame(rows)
    out.insert(0, "subset", label)
    return out


def summarize(df, name):
    print(f"\n=== {name} ===")
    whole = build_table(df, pd.Series(True, index=df.index), "all")
    print(f"  all zones:     mean F1={whole['f1_weighted'].mean():.4f}  mean R2={whole['r2'].mean():.4f}")
    per_cluster = {}
    for cluster in CLUSTERS:
        sub = build_table(df, df["lhz"] == cluster, cluster)
        per_cluster[cluster] = sub["f1_weighted"].mean()
        print(f"  {cluster:13} mean F1={sub['f1_weighted'].mean():.4f}  mean R2={sub['r2'].mean():.4f}")
    return whole, whole["f1_weighted"].mean(), per_cluster


def main():
    members = {name: load_member(name, path) for name, path in MEMBERS.items()}

    ref_key = members["XGBoost"].index
    for name, df in members.items():
        assert set(df.index) == set(ref_key), f"key mismatch: {name}"
        members[name] = df.loc[ref_key]

    # Ground truth is always taken from XGBoost's "observed" (raw ipc_continuous,
    # naive-rounded to a class at scoring time, per to_ipc_class -- the convention
    # every prior RQ1/RQ2 experiment in this project uses). Checked directly:
    # RandomForest/LSTM/T-GCN's own "observed" columns match this exactly after
    # rounding (0 mismatches). TabICLv2 does NOT match after rounding (17.9% of
    # rows) -- it was deliberately fit on ipc_phase_20pct (the >=20%-population-share
    # class rule, CLAUDE.md Sec 11), a genuinely different discretization of the
    # same underlying month, not a data bug -- so its own "observed" is discarded
    # here in favor of the shared XGBoost-derived ground truth, and only its
    # "prediction" column is used in the ensemble.
    observed = members["XGBoost"]["observed"]
    for name in ("RandomForest", "LSTM"):
        assert np.allclose(members[name]["observed"], observed), f"observed mismatch: {name}"
    # T-GCN's "observed" is already the discretized class (int), not the raw
    # continuous ipc_continuous value -- compare at the class level instead.
    assert (members["T-GCN"]["observed"] == to_ipc_class(observed)).all(), "observed class mismatch: T-GCN"

    lhz = members["XGBoost"]["lhz"]
    persistence_pred = members["XGBoost"]["base1_preds"]  # persistence is identical input across XGB/RF/LSTM/TabICL

    pred_matrix = pd.DataFrame({name: df["prediction"] for name, df in members.items()})

    results_summary = []

    def make_frame(pred_series):
        f = pd.DataFrame({"lhz": lhz, "observed": observed, "prediction": pred_series})
        return f.reset_index()

    # Individual members
    for name in MEMBERS:
        _, mean_f1, per_cluster = summarize(make_frame(pred_matrix[name]), f"{name} (single)")
        row = {"config": name, "mean_f1_all": mean_f1}
        row.update({f"mean_f1_{c}": v for c, v in per_cluster.items()})
        results_summary.append(row)

    # Persistence reference
    _, mean_f1, per_cluster = summarize(make_frame(persistence_pred), "Persistence (reference)")
    row = {"config": "Persistence", "mean_f1_all": mean_f1}
    row.update({f"mean_f1_{c}": v for c, v in per_cluster.items()})
    results_summary.append(row)

    # Full 5-way ensemble
    full_pred = pred_matrix.mean(axis=1)
    full_frame = make_frame(full_pred)
    _, mean_f1, per_cluster = summarize(full_frame, "Full 5-way ensemble (all members averaged)")
    row = {"config": "Full-5-way", "mean_f1_all": mean_f1}
    row.update({f"mean_f1_{c}": v for c, v in per_cluster.items()})
    results_summary.append(row)
    full5_mean_f1 = mean_f1
    full_frame.to_parquet("full_ensemble_predictions.parquet", index=False)

    # Leave-one-out ablations
    for dropped in MEMBERS:
        remaining = [m for m in MEMBERS if m != dropped]
        loo_pred = pred_matrix[remaining].mean(axis=1)
        loo_frame = make_frame(loo_pred)
        _, mean_f1, per_cluster = summarize(loo_frame, f"Leave-one-out: drop {dropped} (4-way avg of {remaining})")
        row = {"config": f"LOO_drop_{dropped}", "mean_f1_all": mean_f1}
        row.update({f"mean_f1_{c}": v for c, v in per_cluster.items()})
        row["delta_vs_full5_all"] = mean_f1 - full5_mean_f1
        results_summary.append(row)
        loo_frame.to_parquet(f"loo_drop_{dropped}_predictions.parquet", index=False)

    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv("full_ensemble_summary.csv", index=False)

    print("\n\n=== SUMMARY (mean weighted F1 across 7 leads) ===")
    print(summary_df.to_string(index=False))
    print("\nwrote full_ensemble_predictions.parquet, loo_drop_*_predictions.parquet, full_ensemble_summary.csv")


if __name__ == "__main__":
    main()
