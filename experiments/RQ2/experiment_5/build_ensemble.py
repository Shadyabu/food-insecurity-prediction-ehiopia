"""RQ2 Experiment 5 (supplementary): averaging-blend ensemble of the two
strongest RQ2 architectures on their best-established backbone (delta
target, climate cluster excluded -- report.md section 7.4's documented
best XGBoost recipe).

Design rationale (see significance_test.py, run first): a livelihood-zone-
conditional routing ensemble (route pastoral to LSTM, everything else to a
tree model) was the original idea, but the zone-level bootstrap showed the
LSTM pastoral "edge" motivating it is not statistically distinguishable
from noise (p~0.29, 95% CI [-0.018, +0.049]) while LSTM's agropastoral
weakness IS real (p~0.000). TabICLv2 showed no significant edge over
either tree model anywhere tested. RandomForest's only *significant* edge
over XGBoost (in the original full-feature delta comparison) was in
crop_farming (p~0.016) -- everywhere else, not distinguishable from noise.

Given that, routing by architecture isn't evidence-backed. What IS
evidence-backed: RandomForest and XGBoost are statistically tied almost
everywhere and never significantly worse than each other except RF's one
proven crop_farming edge -- exactly the situation where an unweighted
average of two decorrelated, comparably-accurate models (bagging vs.
boosting -- different bias-variance profiles, different sampling) is the
textbook case for variance reduction, more defensible than hand-picking a
winner per zone from noisy per-cluster deltas. This blends
unhcr_delta_noclimate (XGBoost) and rf_delta_noclimate (RandomForest) --
both fit on the identical feature set/split/target framing, differing only
in architecture -- by averaging their continuous `prediction` column
before discretizing to IPC class. LSTM/TabICLv2/T-GCN are excluded: no
proven edge anywhere, and LSTM/TabICLv2 use different feature
representations (raw sequence / top-80 subset) that would need re-running
on the no-climate backbone before a like-for-like blend is even possible.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]

XGB_PATH = "../../RQ1/experiment_2/unhcr_delta_noclimate/predictions.parquet"
RF_PATH = "../experiment_4/rf_delta_noclimate/predictions.parquet"


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load(path):
    df = pd.read_parquet(path)
    return df[["time", "zone_code", "lhz", "lead", "observed", "prediction", "base1_preds"]].copy()


def score_subset(df, lead):
    sub = df[df["lead"] == lead]
    if len(sub) == 0:
        return None
    y_true = to_ipc_class(sub["observed"])
    y_pred = to_ipc_class(sub["prediction"])
    return {
        "lead": lead, "n": len(sub),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "mae": mean_absolute_error(sub["observed"], sub["prediction"]),
        "r2": r2_score(sub["observed"], sub["prediction"]) if sub["observed"].nunique() > 1 else np.nan,
    }


def build_table(df, mask, label):
    sub = df[mask]
    rows = [score_subset(sub, lead) for lead in LEADS]
    rows = [r for r in rows if r is not None]
    out = pd.DataFrame(rows)
    out.insert(0, "subset", label)
    return out


def summarize(df, name):
    print(f"\n=== {name} ===")
    whole = build_table(df, pd.Series(True, index=df.index), "all")
    print(f"  all zones:     mean F1={whole['f1_weighted'].mean():.4f}  mean R2={whole['r2'].mean():.4f}")
    for cluster in CLUSTERS:
        sub = build_table(df, df["lhz"] == cluster, cluster)
        print(f"  {cluster:13} mean F1={sub['f1_weighted'].mean():.4f}  mean R2={sub['r2'].mean():.4f}")
    return whole


def persistence_table(df, name):
    tmp = df.copy()
    tmp["prediction"] = tmp["base1_preds"]
    print(f"\n=== {name} (persistence baseline, same rows) ===")
    whole = build_table(tmp, pd.Series(True, index=tmp.index), "all")
    print(f"  all zones:     mean F1={whole['f1_weighted'].mean():.4f}")
    for cluster in CLUSTERS:
        sub = build_table(tmp, tmp["lhz"] == cluster, cluster)
        print(f"  {cluster:13} mean F1={sub['f1_weighted'].mean():.4f}")


def main():
    xgb = load(XGB_PATH)
    rf = load(RF_PATH)

    key = ["time", "zone_code", "lead"]
    xgb_i = xgb.set_index(key).sort_index()
    rf_i = rf.set_index(key).sort_index()
    assert set(xgb_i.index) == set(rf_i.index), "key mismatch between XGBoost and RF predictions"
    rf_i = rf_i.loc[xgb_i.index]
    assert np.allclose(xgb_i["observed"], rf_i["observed"]), "observed mismatch"

    blend = xgb_i[["lhz", "observed", "base1_preds"]].copy()
    blend["prediction"] = (xgb_i["prediction"] + rf_i["prediction"]) / 2.0
    blend = blend.reset_index()

    summarize(xgb, "XGBoost-delta-noclimate (single, current best)")
    summarize(rf, "RandomForest-delta-noclimate (single)")
    blend_summary = summarize(blend, "Blend: 50/50 average(XGBoost, RandomForest)")
    persistence_table(blend, "reference")

    blend.to_parquet("blend_predictions.parquet", index=False)
    blend_summary.to_csv("blend_metrics_per_lead.csv", index=False)
    print("\nwrote blend_predictions.parquet, blend_metrics_per_lead.csv")


if __name__ == "__main__":
    main()
