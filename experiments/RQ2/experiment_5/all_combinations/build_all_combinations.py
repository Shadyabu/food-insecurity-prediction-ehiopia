"""Exhaustive ensemble search over the 5 RQ2 architectures: every
combination of size 2-5 (10 + 10 + 5 + 1 = 26 combinations), each scored
as an unweighted average of its members' `prediction` columns -- same
averaging + round-to-class-at-scoring convention as
`build_full_ensemble.py`/`build_ensemble.py`, generalized from "2-way" /
"5-way" / "leave-one-out-4-way" to every possible subset.

"Best performing" = ranked by mean weighted F1 across the 7 lead times,
all zones pooled (mean_f1_all) -- the same ranking metric every other
model/ensemble comparison in this project uses (report.md's "mean
weighted F1" headline number). Per-livelihood-zone-cluster breakdown is
saved for every combination, not just used for ranking.

Ground truth and member alignment follow build_full_ensemble.py exactly:
observed is XGBoost's `observed` (ipc_continuous, rounded at scoring);
TabICLv2's own `observed` (fit on ipc_phase_20pct, CLAUDE.md Sec 11) is
discarded in favor of the shared value, only its `prediction` is used;
T-GCN's `observed`(=true_class+1) is checked to match after rounding.
"""

import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]
N_TOP = 5

MEMBERS = {
    "XGB": "../../../RQ1/experiment_2/unhcr_delta_noclimate/predictions.parquet",
    "RF": "../../experiment_4/rf_delta_noclimate/predictions.parquet",
    "LSTM": "../../experiment_1/lstm_delta/predictions.parquet",
    "TabICL": "../../experiment_2/tabicl_level/predictions.parquet",
    "TGCN": "../../experiment_3/tgcn_ce/predictions.parquet",
}


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_member(name, rel_path):
    path = os.path.join(SCRIPT_DIR, rel_path)
    df = pd.read_parquet(path)
    if "window" in df.columns:
        df = df[df["window"] == "default"].copy()
    df["time"] = pd.to_datetime(df["time"])

    if name == "TGCN":
        df["observed"] = df["true_class"] + 1
        df["prediction"] = df["pred_class"] + 1
        df["base1_preds"] = np.nan
    key = ["time", "zone_code", "lead"]
    return df.set_index(key).sort_index()[["lhz", "observed", "prediction", "base1_preds"]]


def load_all_members():
    members = {name: load_member(name, path) for name, path in MEMBERS.items()}
    ref_key = members["XGB"].index
    for name, df in members.items():
        assert set(df.index) == set(ref_key), f"key mismatch: {name}"
        members[name] = df.loc[ref_key]

    observed = members["XGB"]["observed"]
    for name in ("RF", "LSTM"):
        assert np.allclose(members[name]["observed"], observed), f"observed mismatch: {name}"
    assert (members["TGCN"]["observed"] == to_ipc_class(observed)).all(), "observed class mismatch: TGCN"

    lhz = members["XGB"]["lhz"]
    persistence_pred = members["XGB"]["base1_preds"]
    pred_matrix = pd.DataFrame({name: df["prediction"] for name, df in members.items()})
    return pred_matrix, observed, lhz, persistence_pred


def detailed_metrics(observed, lhz, prediction, label):
    frame = pd.DataFrame({"lhz": lhz, "observed": observed, "prediction": prediction})
    frame = frame.reset_index()  # time, zone_code, lead columns back
    rows = []
    for subset_name, mask in [("all", pd.Series(True, index=frame.index))] + \
                              [(c, frame["lhz"] == c) for c in CLUSTERS]:
        sub_all = frame[mask]
        for lead in LEADS:
            sub = sub_all[sub_all["lead"] == lead]
            if len(sub) == 0:
                continue
            y_true = to_ipc_class(sub["observed"])
            y_pred = to_ipc_class(sub["prediction"])
            rows.append({
                "combo": label, "subset": subset_name, "lead": lead, "n": len(sub),
                "accuracy": accuracy_score(y_true, y_pred),
                "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
                "mae": mean_absolute_error(sub["observed"], sub["prediction"]),
                "r2": r2_score(sub["observed"], sub["prediction"]) if sub["observed"].nunique() > 1 else np.nan,
            })
    return pd.DataFrame(rows)


def overall_summary(detail_df):
    out = {}
    for subset_name in ["all"] + CLUSTERS:
        sub = detail_df[detail_df["subset"] == subset_name]
        out[f"mean_f1_{subset_name}"] = sub["f1_weighted"].mean()
        out[f"mean_r2_{subset_name}"] = sub["r2"].mean()
    return out


def main():
    pred_matrix, observed, lhz, persistence_pred = load_all_members()
    names = list(MEMBERS.keys())

    combos = []
    for size in range(2, 6):
        combos.extend(itertools.combinations(names, size))
    assert len(combos) == 26, f"expected 26 combinations, got {len(combos)}"

    summary_rows = []
    detail_by_combo = {}
    for combo in combos:
        label = "+".join(combo)
        combo_pred = pred_matrix[list(combo)].mean(axis=1)
        detail = detailed_metrics(observed, lhz, combo_pred, label)
        detail_by_combo[label] = detail
        summ = overall_summary(detail)
        summ["combo"] = label
        summ["size"] = len(combo)
        summary_rows.append(summ)

    summary_df = pd.DataFrame(summary_rows).sort_values("mean_f1_all", ascending=False).reset_index(drop=True)
    cols = ["combo", "size", "mean_f1_all", "mean_f1_pastoral", "mean_f1_agropastoral",
            "mean_f1_crop_farming", "mean_r2_all", "mean_r2_pastoral", "mean_r2_agropastoral",
            "mean_r2_crop_farming"]
    summary_df = summary_df[cols]
    summary_df.to_csv(os.path.join(SCRIPT_DIR, "combos_summary.csv"), index=False)

    print(f"=== All {len(combos)} combinations, ranked by mean_f1_all ===")
    print(summary_df.to_string(index=False))

    top5_labels = summary_df.head(N_TOP)["combo"].tolist()
    print(f"\n=== Top {N_TOP}: {top5_labels} ===")

    top5_detail = pd.concat([detail_by_combo[label] for label in top5_labels], axis=0, ignore_index=True)
    top5_detail.to_csv(os.path.join(SCRIPT_DIR, "top5_detailed_metrics.csv"), index=False)

    # Persistence reference, same 28-row (7 lead x 4 subset) breakdown
    persistence_detail = detailed_metrics(observed, lhz, persistence_pred, "Persistence")
    persistence_detail.to_csv(os.path.join(SCRIPT_DIR, "persistence_detailed_metrics.csv"), index=False)

    # Save top-5 predictions
    for label in top5_labels:
        combo = tuple(label.split("+"))
        combo_pred = pred_matrix[list(combo)].mean(axis=1)
        frame = pd.DataFrame({"lhz": lhz, "observed": observed, "prediction": combo_pred}).reset_index()
        safe_name = label.replace("+", "_")
        frame.to_parquet(os.path.join(SCRIPT_DIR, "predictions", f"{safe_name}.parquet"), index=False)

    print(f"\nwrote combos_summary.csv ({len(summary_df)} rows), "
          f"top5_detailed_metrics.csv ({len(top5_detail)} rows), "
          f"persistence_detailed_metrics.csv ({len(persistence_detail)} rows), "
          f"predictions/{{combo}}.parquet x {N_TOP}")


if __name__ == "__main__":
    main()
