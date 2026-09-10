"""Standalone TabICLv2 worker, run as a SEPARATE PROCESS from common.py's
main orchestration code -- same reason as lstm_worker.py: TabICLv2 is a
pretrained-weight torch model internally, and loading/running pretrained
torch weights segfaults whenever xgboost/shap have already been imported
in the same process (see lstm_worker.py's docstring for the full
investigation). This script only ever imports numpy/pandas/tabicl(->torch),
never xgboost.

Only a predict mode is needed here (no SHAP mode) -- TabICLv2 is excluded
from the ensemble's SHAP-ranking layer per the project-owner-confirmed RQ4
scope (see common.py's module docstring); it only needs to keep
contributing its own predictions to the ensemble average under masking.

Usage:
    python tabicl_worker.py --lead 0 [--mask-cols-file f.json] --out out.csv
"""

import argparse
import json
import sys
from pathlib import Path

RQ4_DIR = Path(__file__).resolve().parent
REPO_ROOT = RQ4_DIR.parent.parent
MODEL_TABLES_DIR = REPO_ROOT / "data" / "processed" / "model_tables"
TABICL_IMPORTANCE_PATH = REPO_ROOT / "experiments" / "RQ1" / "experiment_2" / "unhcr_level" / "feature_importance.csv"
TABICL_TOP_N = 80
SEED = 42

sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ2" / "experiment_2"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=int, required=True)
    p.add_argument("--window", choices=["default", "extended"], default="default")
    p.add_argument("--mask-cols-file", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    import pandas as pd
    import run_tabicl_ethiopia as t
    from tabicl import TabICLClassifier

    lead = args.lead
    df = t.load_lead_table(str(MODEL_TABLES_DIR), lead)
    sub = df[(df["observed"] == True) & df[t.TARGET_COL].notna()].copy()
    sub["time"] = pd.to_datetime(sub["month"])
    sub = sub.sort_values(["time", "zone_code"]).reset_index(drop=True)

    meta = sub[["time", "zone_code", "zone_name", "split", "dominant_livelihood_zone"]].copy()
    labels = t.to_class(sub[t.TARGET_COL])

    features_frame = t.build_feature_frame(sub)
    ranking = t.load_importance_ranking(str(TABICL_IMPORTANCE_PATH), lead)
    selected = t.select_features(features_frame, ranking, TABICL_TOP_N)
    X = features_frame[selected]

    train_mask = (meta["split"] == "train").to_numpy()
    if args.window == "extended":
        extended_start, extended_end = pd.Timestamp("2020-01-01"), pd.Timestamp("2024-12-31")
        test_mask = ((meta["time"] >= extended_start) & (meta["time"] <= extended_end)).to_numpy()
    else:
        test_mask = (meta["split"] == "test").to_numpy()
    train_x = X[train_mask].reset_index(drop=True)
    train_y = labels[train_mask]
    test_x = X[test_mask].reset_index(drop=True)
    test_meta = meta[test_mask].reset_index(drop=True)
    test_labels = labels[test_mask].reset_index(drop=True)

    if args.mask_cols_file:
        import numpy as np
        mask_cols = json.loads(Path(args.mask_cols_file).read_text())
        present = [col for col in mask_cols if col in test_x.columns]
        if present:
            test_x = test_x.copy()
            test_x[present] = np.nan

    clf = TabICLClassifier(random_state=SEED, device="cpu")
    clf.fit(train_x, train_y)
    pred_class = clf.predict(test_x)

    out = test_meta[["time", "zone_code"]].copy()
    out["lead"] = lead
    out["tabicl_pred"] = pred_class
    out["observed_class"] = test_labels.to_numpy()
    out.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
