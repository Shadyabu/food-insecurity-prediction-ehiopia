"""Champion-ensemble nowcast, part 2/3 -- LSTM member, run as a SEPARATE
PROCESS (never imports xgboost/tabicl) for the same documented reason as
experiments/RQ4/lstm_worker.py: loading pretrained torch weights segfaults
if xgboost/shap have already been imported in the same process.

Key simplification versus the XGBoost/RandomForest members: the LSTM's
input is a 12-month sequence of RAW (pre-lead-shift) features ending at
the ORIGIN month, and since this nowcast's origin is fixed at 2026-06 for
every lead, that sequence is IDENTICAL across all 7 leads -- only the
target differs, and that's purely a function of which lead's trained
model processes the (same) sequence. That exact window already exists in
the historical cache (experiments/RQ2/experiment_1/sequences/lead00.npz,
the only lead whose target month equals its own origin month, 2026-06) --
no new sequence construction needed, unlike XGBoost/RandomForest's
synthetic tabular rows.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ4"))
import lstm_worker as w  # noqa: E402

NOWCAST_DIR = Path(__file__).resolve().parent
LEADS = [0, 1, 2, 3, 4, 8, 12]
ORIGIN_MONTH = pd.Timestamp("2026-06-01")


def main():
    # The origin window: lead=0's own cell has a real row at time==origin
    # (target==origin when lead=0) -- extract its 12-month raw sequence and
    # base1 per zone, reused unchanged for every other lead's model.
    origin_cell = w.get_lstm_cell(0, window="default")
    origin_idx = np.where(origin_cell["time"] == ORIGIN_MONTH)[0]
    assert len(origin_idx) == 92, f"expected 92 zones at origin month in lead00 cache, got {len(origin_idx)}"

    zone_codes = origin_cell["zone_code"][origin_idx]
    X_origin_raw = origin_cell["X_raw"][origin_idx]  # (92, 12, n_features)
    base1_origin = origin_cell["base1"][origin_idx]  # (92,) -- IPC value 1 month before origin, same for every lead

    order = np.argsort(zone_codes)
    zone_codes = zone_codes[order]
    X_origin_raw = X_origin_raw[order]
    base1_origin = base1_origin[order]

    records = []
    for lead in LEADS:
        cell = w.get_lstm_cell(lead, window="default")  # only used for this lead's own train-derived medians/mean/std
        X_scaled = w.lstm_scale(X_origin_raw, cell["medians"], cell["mean"], cell["std"])
        model = w.load_model(lead)
        pred = w.predict(model, X_scaled, base1_origin)  # already adds base1 back (delta -> continuous)

        target_month = (ORIGIN_MONTH + pd.DateOffset(months=lead)).strftime("%Y-%m")
        for zc, p in zip(zone_codes, pred):
            records.append({"zone_code": zc, "lead": lead, "target_month": target_month, "lstm_pred": float(p)})
        print(f"  lead={lead:2d}  n=92  done")

    out = pd.DataFrame(records)
    out_path = NOWCAST_DIR / "lstm_nowcast.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
