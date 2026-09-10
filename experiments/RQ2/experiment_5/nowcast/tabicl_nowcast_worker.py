"""Champion-ensemble nowcast, part 3/3 -- TabICLv2 member, run as a
SEPARATE PROCESS (never imports xgboost), same documented reason as
experiments/RQ4/tabicl_worker.py.

TabICLv2 has no persisted "trained model" the way XGBoost/RandomForest do
-- it's an in-context-learning classifier that is fit (fast; no gradient
descent) and predicted in the same call, every time it's used, including
in every other script in this project that touches it. This worker does
the same: build train_x/train_y from the real historical training rows
(same top-N-feature selection as run_tabicl_ethiopia.py's champion
recipe), fit fresh (deterministic, random_state=42), predict directly on
this nowcast's synthetic test row (from experiments/RQ5/nowcast/
nowcast_feature_rows.csv, the same architecture-agnostic raw feature rows
the XGBoost/RandomForest members reuse).
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RQ5_NOWCAST_DIR = REPO_ROOT / "experiments" / "RQ5" / "nowcast"
NOWCAST_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ2" / "experiment_2"))

MODEL_TABLES_DIR = REPO_ROOT / "data" / "processed" / "model_tables"
IMPORTANCE_PATH = REPO_ROOT / "experiments" / "RQ1" / "experiment_2" / "unhcr_level" / "feature_importance.csv"
NOWCAST_FEATURE_ROWS_PATH = RQ5_NOWCAST_DIR / "nowcast_feature_rows.csv"
TOP_N_FEATURES = 80

LEADS = [0, 1, 2, 3, 4, 8, 12]
ORIGIN_MONTH = pd.Timestamp("2026-06-01")


def run_lead_nowcast(t, tabicl_cls, lead, synthetic_rows):
    df = t.load_lead_table(str(MODEL_TABLES_DIR), lead)
    train_df = df[(df["observed"] == True) & df[t.TARGET_COL].notna() & (df["split"] == "train")].copy()

    synth = synthetic_rows[synthetic_rows["lead"] == lead].drop(columns=["lead", "target_month"]).copy()
    missing_in_synth = set(train_df.columns) - set(synth.columns)
    missing_in_train = set(synth.columns) - set(train_df.columns)
    if missing_in_synth:
        raise ValueError(f"lead {lead}: synthetic row missing columns: {sorted(missing_in_synth)}")
    if missing_in_train:
        synth = synth.drop(columns=sorted(missing_in_train))
    synth = synth[train_df.columns]
    for col in train_df.columns:
        if train_df[col].dtype != synth[col].dtype:
            try:
                synth[col] = synth[col].astype(train_df[col].dtype)
            except (ValueError, TypeError):
                pass

    combined = pd.concat([train_df, synth], ignore_index=True)
    combined["time"] = pd.to_datetime(combined["month"])
    combined = combined.sort_values(["time", "zone_code"]).reset_index(drop=True)

    meta = combined[["time", "zone_code", "zone_name", "split", "dominant_livelihood_zone"]].copy()
    labels = t.to_class(combined[t.TARGET_COL])

    features_frame = t.build_feature_frame(combined)
    ranking = t.load_importance_ranking(str(IMPORTANCE_PATH), lead)
    selected = t.select_features(features_frame, ranking, TOP_N_FEATURES)
    X = features_frame[selected]

    train_mask = (meta["split"] == "train").to_numpy()
    test_mask = ~train_mask  # exactly the synthetic row(s) -- their split is "beyond", never "train"
    assert test_mask.sum() == len(synth), f"lead {lead}: expected {len(synth)} test rows, got {test_mask.sum()}"

    train_x = X[train_mask].reset_index(drop=True)
    train_y = labels[train_mask]
    test_x = X[test_mask].reset_index(drop=True)
    test_meta = meta[test_mask].reset_index(drop=True)

    clf = tabicl_cls(random_state=42, device="cpu")
    clf.fit(train_x, train_y)
    preds = clf.predict(test_x)

    return test_meta, preds, len(selected)


def main():
    import run_tabicl_ethiopia as t
    from tabicl import TabICLClassifier

    synthetic_rows = pd.read_csv(NOWCAST_FEATURE_ROWS_PATH, low_memory=False)

    records = []
    for lead in LEADS:
        test_meta, preds, n_feat = run_lead_nowcast(t, TabICLClassifier, lead, synthetic_rows)
        target_month = (ORIGIN_MONTH + pd.DateOffset(months=lead)).strftime("%Y-%m")
        for zc, p in zip(test_meta["zone_code"], preds):
            records.append({"zone_code": zc, "lead": lead, "target_month": target_month, "tabicl_pred": int(p)})
        print(f"  lead={lead:2d}  n={len(test_meta):3d}  n_features={n_feat}  done")

    out = pd.DataFrame(records)
    out_path = NOWCAST_DIR / "tabicl_nowcast.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
