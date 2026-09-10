"""Champion-ensemble explanation, part 2/2 -- per-instance (not aggregated)
SHAP for the LSTM member, run as an isolated subprocess (same segfault
rationale as lstm_nowcast_worker.py / experiments/RQ4/lstm_worker.py).

experiments/RQ4/lstm_worker.py::run_shap() already computes per-instance
SHAP internally (`per_row = np.abs(sv[..., 0]).sum(axis=1)`, shape
(n_rows, n_features)) but then collapses it to one GLOBAL vector via
`.mean(axis=0)` before returning -- RQ4 only ever needed a global ranking.
This script keeps the per-row structure instead (and keeps the SIGN,
dropping RQ4's abs() -- the rule engine's polarity gate needs signed SHAP,
same convention as every other subject in this project), reusing
lstm_worker.py's cell-loading/scaling/model-loading machinery unchanged.

Raw feature values for the rule engine's polarity gate are taken from the
LAST timestep of each 12-month window (i.e. the origin month itself, the
most recent value in the sequence) -- the taxonomy's features are
single-value-per-instance, and "the current SPI_3 value" naturally means
the most recent one in the window, not an average over the trailing year.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RQ5_DIR = REPO_ROOT / "experiments" / "RQ5"
NOWCAST_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ4"))
sys.path.insert(0, str(RQ5_DIR))
import lstm_worker as w  # noqa: E402
from rule_engine import load_taxonomy, required_columns  # noqa: E402

LEADS = [0, 1, 2, 3, 4, 8, 12]
ORIGIN_MONTH = pd.Timestamp("2026-06-01")
BACKGROUND_N = 30
SEED = 42


def per_row_signed_shap(lead, X_origin_raw):
    import shap
    import torch

    cell = w.get_lstm_cell(lead, window="default")
    model = w.load_model(lead)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).unsqueeze(-1)

    wrapped = Wrap(model)

    X_scaled = w.lstm_scale(X_origin_raw, cell["medians"], cell["mean"], cell["std"])
    X_train_scaled = w.lstm_scale(cell["X_raw"][cell["train_mask"]], cell["medians"], cell["mean"], cell["std"])

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_train_scaled), size=min(BACKGROUND_N, len(X_train_scaled)), replace=False)
    background = torch.from_numpy(X_train_scaled[bg_idx].astype(np.float32))

    explainer = shap.GradientExplainer(wrapped, background)
    test_t = torch.from_numpy(X_scaled.astype(np.float32))
    sv = np.asarray(explainer.shap_values(test_t))  # (n, seq_len, n_features, 1)
    per_row_signed = sv[..., 0].sum(axis=1)  # sum over 12 timesteps, SIGN preserved -> (n, n_features)
    return per_row_signed, cell["feature_names"]


def main():
    taxonomy = load_taxonomy()
    need_cols = sorted(required_columns(taxonomy))

    origin_cell = w.get_lstm_cell(0, window="default")
    origin_idx = np.where(origin_cell["time"] == ORIGIN_MONTH)[0]
    assert len(origin_idx) == 92, f"expected 92 zones at origin, got {len(origin_idx)}"
    zone_codes = origin_cell["zone_code"][origin_idx]
    X_origin_raw = origin_cell["X_raw"][origin_idx]  # (92, 12, n_features)
    order = np.argsort(zone_codes)
    zone_codes = zone_codes[order]
    X_origin_raw = X_origin_raw[order]

    # Raw values for the rule engine: last timestep of the window == origin month.
    last_timestep_raw = X_origin_raw[:, -1, :]  # (92, n_features)

    raw_records = []
    shap_records = []
    for lead in LEADS:
        shap_vals, feature_names = per_row_signed_shap(lead, X_origin_raw)
        target_month = (ORIGIN_MONTH + pd.DateOffset(months=lead)).strftime("%Y-%m")

        shap_df_full = pd.DataFrame(shap_vals, columns=feature_names)
        raw_df_full = pd.DataFrame(last_timestep_raw, columns=feature_names)
        avail = [c for c in need_cols if c in feature_names]
        missing = [c for c in need_cols if c not in feature_names]

        shap_sub = shap_df_full[avail].copy()
        raw_sub = raw_df_full[avail].copy()
        for col in missing:
            shap_sub[col] = np.nan
            raw_sub[col] = np.nan
        shap_sub = shap_sub[need_cols]
        raw_sub = raw_sub[need_cols]

        meta = pd.DataFrame({"zone_code": zone_codes, "lead": lead, "target_month": target_month, "subject": "champion_lstm"})
        raw_records.append(pd.concat([meta.reset_index(drop=True), raw_sub.reset_index(drop=True)], axis=1))
        shap_records.append(pd.concat([meta.reset_index(drop=True), shap_sub.reset_index(drop=True)], axis=1))
        print(f"  lead={lead:2d}  n=92  done (champion lstm shap)")

    raw_all = pd.concat(raw_records, ignore_index=True)
    shap_all = pd.concat(shap_records, ignore_index=True)
    raw_all.to_parquet(NOWCAST_DIR / "champion_lstm_raw.parquet", index=False)
    shap_all.to_parquet(NOWCAST_DIR / "champion_lstm_shap.parquet", index=False)
    print(f"\nwrote {len(raw_all)} rows to champion_lstm_raw.parquet / champion_lstm_shap.parquet")


if __name__ == "__main__":
    main()
