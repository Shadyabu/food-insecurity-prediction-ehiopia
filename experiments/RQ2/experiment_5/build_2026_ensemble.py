"""Diagnostic-only script (2026-09-02): does the 4-way ensemble (XGBoost +
RandomForest + LSTM + TabICLv2, the RQ2 Experiment 5 champion combo) also
suffer the ACLED-raw-data-ceiling problem diagnosed in
experiments/RQ1/experiment_2/report.md section 13, when scored on 2026
specifically? Not part of the standing ensemble build (build_full_ensemble.py)
-- a one-off comparison, same averaging convention (ground truth from
XGBoost's own "observed", raw predictions averaged then discretized once).

Two ensembles are built: BASELINE (each member's already-established best
recipe -- XGBoost/RandomForest: delta target + climate excluded;
LSTM: delta target; TabICLv2: level/class-native -- conflict features
still present in every member that can see them) and NOCONFLICT (same,
but ACLED+UNHCR also excluded from every member that can see them).
T-GCN is excluded, per the standing 2026-08-25 decision.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]

VARIANTS = {
    "baseline": {
        "XGBoost": "../../RQ1/experiment_2/rq5ens_xgb_2026_baseline/predictions.parquet",
        "RandomForest": "rq5ens_rf_2026_baseline/predictions.parquet",
        "LSTM": "rq5ens_lstm_2026_baseline/predictions.parquet",
        "TabICLv2": "rq5ens_tabicl_2026_baseline/predictions.parquet",
    },
    "noconflict": {
        "XGBoost": "../../RQ1/experiment_2/rq5ens_xgb_2026_noconflict/predictions.parquet",
        "RandomForest": "rq5ens_rf_2026_noconflict/predictions.parquet",
        "LSTM": "rq5ens_lstm_2026_noconflict/predictions.parquet",
        "TabICLv2": "rq5ens_tabicl_2026_noconflict/predictions.parquet",
    },
}


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_member(name, path):
    df = pd.read_parquet(path)
    if "window" in df.columns:
        df = df[df["window"] == "extended"].copy()
    df["time"] = pd.to_datetime(df["time"])
    key = ["time", "zone_code", "lead"]
    return df.set_index(key).sort_index()[["lhz", "observed", "prediction", "base1_preds"]]


def score(df):
    y_true = to_ipc_class(df["observed"])
    y_pred = to_ipc_class(df["prediction"])
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def build_and_score(paths, label):
    members = {name: load_member(name, path) for name, path in paths.items()}
    ref_key = members["XGBoost"].index
    for name, df in members.items():
        assert set(df.index) == set(ref_key), f"{label}: key mismatch: {name} ({len(df.index)} vs {len(ref_key)})"
        members[name] = df.loc[ref_key]

    observed = members["XGBoost"]["observed"]
    for name in ("RandomForest", "LSTM"):
        assert np.allclose(members[name]["observed"], observed), f"{label}: observed mismatch: {name}"
    lhz = members["XGBoost"]["lhz"]
    persistence_pred = members["XGBoost"]["base1_preds"]
    pred_matrix = pd.DataFrame({name: df["prediction"] for name, df in members.items()})
    ensemble_pred = pred_matrix.mean(axis=1)

    def make_frame(pred_series):
        return pd.DataFrame({"lhz": lhz, "observed": observed, "prediction": pred_series}).reset_index()

    print(f"\n=== {label} ===")
    rows = []
    for name in list(paths) + ["Persistence", "Ensemble(4-way)"]:
        pred = {**{n: pred_matrix[n] for n in paths}, "Persistence": persistence_pred,
                "Ensemble(4-way)": ensemble_pred}[name]
        frame = make_frame(pred)
        f1s, r2s = [], []
        for lead in LEADS:
            sub = frame[frame["lead"] == lead]
            f1s.append(score(sub))
            r2s.append(r2_score(sub["observed"], sub["prediction"]) if sub["observed"].nunique() > 1 else np.nan)
        mean_f1 = np.mean(f1s)
        mean_r2 = np.nanmean(r2s)
        print(f"  {name:18s} mean F1={mean_f1:.4f}  mean R2={mean_r2:.4f}")
        rows.append({"config": name, "variant": label, "mean_f1": mean_f1, "mean_r2": mean_r2})
        for cluster in CLUSTERS:
            csub = frame[frame["lhz"] == cluster]
            cf1s = [score(csub[csub["lead"] == lead]) for lead in LEADS if len(csub[csub["lead"] == lead]) > 0]
            print(f"      {cluster:13s} mean F1={np.mean(cf1s):.4f}")
    return pd.DataFrame(rows)


def main():
    all_rows = []
    for label, paths in VARIANTS.items():
        all_rows.append(build_and_score(paths, label))
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv("ensemble_2026_summary.csv", index=False)
    print("\nwrote ensemble_2026_summary.csv")


if __name__ == "__main__":
    main()
