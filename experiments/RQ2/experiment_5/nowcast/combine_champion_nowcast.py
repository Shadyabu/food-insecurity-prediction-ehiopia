"""Champion-ensemble nowcast, final step -- combine the 3 member CSVs
(xgb_rf_nowcast.csv, lstm_nowcast.csv, tabicl_nowcast.csv) via simple
average, same convention as build_2026_ensemble.py / build_full_ensemble.py
elsewhere in this project: mean of the 4 members' own "prediction" values
(XGBoost/RandomForest/LSTM continuous IPC scale, TabICLv2's discrete 1-5
class treated as a plain numeric value alongside them -- matching existing
precedent, not a new convention invented here), then round/clip to [1,5]
for the discrete class.
"""

from pathlib import Path

import numpy as np
import pandas as pd

NOWCAST_DIR = Path(__file__).resolve().parent
LEADS = [0, 1, 2, 3, 4, 8, 12]


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def main():
    xgb_rf = pd.read_csv(NOWCAST_DIR / "xgb_rf_nowcast.csv")
    lstm = pd.read_csv(NOWCAST_DIR / "lstm_nowcast.csv")
    tabicl = pd.read_csv(NOWCAST_DIR / "tabicl_nowcast.csv")

    key = ["zone_code", "lead"]
    merged = xgb_rf[["zone_code", "lead", "target_month", "cluster", "xgb_pred", "rf_pred"]].merge(
        lstm[key + ["lstm_pred"]], on=key, how="left"
    ).merge(
        tabicl[key + ["tabicl_pred"]], on=key, how="left"
    )

    assert merged["lstm_pred"].notna().all(), "unmatched LSTM predictions after merge"
    assert merged["tabicl_pred"].notna().all(), "unmatched TabICLv2 predictions after merge"
    assert len(merged) == 92 * len(LEADS), f"expected {92 * len(LEADS)} rows, got {len(merged)}"

    member_cols = ["xgb_pred", "rf_pred", "lstm_pred", "tabicl_pred"]
    merged["ipc_continuous"] = merged[member_cols].mean(axis=1)
    merged["ipc_class"] = to_ipc_class(merged["ipc_continuous"])

    out_cols = ["zone_code", "target_month", "lead", "cluster"] + member_cols + ["ipc_continuous", "ipc_class"]
    out = merged[out_cols].sort_values(["lead", "zone_code"]).reset_index(drop=True)

    out_path = NOWCAST_DIR / "champion_nowcast_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {len(out)} rows to {out_path}")
    print(f"\nipc_class distribution:\n{out['ipc_class'].value_counts().sort_index()}")
    print(f"\nby lead, mean ipc_continuous:\n{out.groupby('lead')['ipc_continuous'].mean()}")


if __name__ == "__main__":
    main()
