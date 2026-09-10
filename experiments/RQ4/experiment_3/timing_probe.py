"""Timing probe for RQ4 Experiment 3 (local, leave-top-1-out SHAP stability).

Purpose: estimate the wall-time cost of LSTM's share of the local-stability
design BEFORE committing to the full 828-row x 7-lead run, the same way
this project ran an empirical timing probe before excluding TabICLv2 from
the SHAP-ranking layer (see ../common.py module docstring) rather than
assuming its cost.

Why this can't reuse lstm_worker.py's existing run_shap()/run_shap_batch()
directly: those two functions return only the AGGREGATE (mean-over-rows)
importance, and neither accepts a mask at all -- Experiment 1/2 never
needed per-row masking for LSTM (Experiment 1 masks the same population-
level column(s) for every row; Experiment 2 resamples which rows are
averaged together, never masks). The local design needs each row's own,
individually different, top-1 feature masked for that row alone, which
means this script computes RAW per-row SHAP first (not the pre-aggregated
output), works out each sampled row's own top-1 feature, then re-explains
that same row with only its own top-1 feature masked.

Run directly (not through common.py's subprocess wrapper -- this script
itself never imports xgboost, so it's safe to run standalone):
    /opt/miniconda3/bin/python3 experiments/RQ4/experiment_3/timing_probe.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RQ4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RQ4_DIR))
import lstm_worker as w  # noqa: E402

N_PROBE_ROWS = 10
PROBE_LEAD = 0


def per_row_shap(explainer, X_scaled_row_batch):
    """Raw (not mean-aggregated) per-row |SHAP| summed over the 12 sequence
    timesteps -- shape (n_rows, n_features). Mirrors lstm_worker.run_shap's
    own per_row computation (line ~181) but returns it instead of
    collapsing it with .mean(axis=0)."""
    import torch
    test_t = torch.from_numpy(X_scaled_row_batch.astype(np.float32))
    sv = np.asarray(explainer.shap_values(test_t))  # (n, seq_len, n_features, 1)
    return np.abs(sv[..., 0]).sum(axis=1)  # (n, n_features)


def main():
    import torch
    import shap

    t0 = time.time()
    lc = w.get_lstm_cell(PROBE_LEAD, window="default")
    model = w.load_model(PROBE_LEAD)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).unsqueeze(-1)

    wrapped = Wrap(model)

    X_test_raw_full = lc["X_raw"][lc["test_mask"]]
    n_test = X_test_raw_full.shape[0]
    rng = np.random.default_rng(w.SEED)
    probe_idx = rng.choice(n_test, size=min(N_PROBE_ROWS, n_test), replace=False)
    X_probe_raw = X_test_raw_full[probe_idx]

    X_train_scaled = w.lstm_scale(lc["X_raw"][lc["train_mask"]], lc["medians"], lc["mean"], lc["std"])
    bg_idx = rng.choice(len(X_train_scaled), size=min(30, len(X_train_scaled)), replace=False)
    background = torch.from_numpy(X_train_scaled[bg_idx].astype(np.float32))
    explainer = shap.GradientExplainer(wrapped, background)
    t_setup = time.time() - t0
    print(f"[setup] model+background load: {t_setup:.2f}s")

    # Step 1: unmasked SHAP for the probe batch, all at once (cheapest way
    # to get every probe row's own top-1 feature).
    t1 = time.time()
    X_probe_scaled = w.lstm_scale(X_probe_raw, lc["medians"], lc["mean"], lc["std"])
    unmasked = per_row_shap(explainer, X_probe_scaled)  # (N_PROBE_ROWS, n_features)
    t_unmasked_batch = time.time() - t1
    print(f"[unmasked batch, n={len(probe_idx)}] {t_unmasked_batch:.2f}s "
          f"({t_unmasked_batch / len(probe_idx):.2f}s/row)")

    feature_names = lc["feature_names"]
    top1_per_row = [feature_names[i] for i in unmasked.argmax(axis=1)]
    print(f"[top-1 features sampled] {top1_per_row}")
    print(f"[distinct top-1 features in sample] {len(set(top1_per_row))} / {len(top1_per_row)}")

    # Step 2: one masked recompute PER ROW (its own top-1 feature only),
    # done individually (no shared-mask batching yet) -- this is the
    # pessimistic, unoptimized cost the full design would pay if every row
    # needed its own separate explainer call.
    t2 = time.time()
    per_row_times = []
    for i, ridx in enumerate(probe_idx):
        tr = time.time()
        row_raw = X_test_raw_full[ridx:ridx + 1]
        masked_scaled = w.mask_nan_then_scale(
            row_raw, feature_names, [top1_per_row[i]], lc["medians"], lc["mean"], lc["std"]
        )
        _ = per_row_shap(explainer, masked_scaled)
        per_row_times.append(time.time() - tr)
    t_masked_individual = time.time() - t2
    print(f"[masked, one-call-per-row, n={len(probe_idx)}] {t_masked_individual:.2f}s total, "
          f"{np.mean(per_row_times):.2f}s/row (min {np.min(per_row_times):.2f}, "
          f"max {np.max(per_row_times):.2f})")

    # Extrapolation to full scope: 828 rows x 7 leads, one-call-per-row
    # (no shared-mask batching optimization applied).
    per_row_cost = np.mean(per_row_times)
    full_rows_per_lead = 828  # default-window test size, all clusters pooled
    n_leads = 7
    est_unmasked_full = (t_unmasked_batch / len(probe_idx)) * full_rows_per_lead * n_leads
    est_masked_full = per_row_cost * full_rows_per_lead * n_leads
    est_setup_full = t_setup * n_leads  # one model+background load per lead
    est_total = est_setup_full + est_unmasked_full + est_masked_full
    print()
    print("=== Extrapolation to full scope (828 rows x 7 leads, unoptimized) ===")
    print(f"  setup (x{n_leads} leads):        {est_setup_full / 60:.1f} min")
    print(f"  unmasked batch (x{n_leads}):      {est_unmasked_full / 60:.1f} min")
    print(f"  masked, one-call-per-row (x{n_leads}): {est_masked_full / 60:.1f} min")
    print(f"  TOTAL estimated:              {est_total / 60:.1f} min ({est_total / 3600:.2f} hr)")
    print()
    print("Note: if the number of DISTINCT top-1 features across all 828 rows "
          "is small relative to 828 (see 'distinct top-1 features in sample' "
          "above), a shared-mask-group batching optimization (analogous to "
          "run_shap_batch's 'load once, loop draws' pattern) could cut the "
          "masked-recompute cost substantially below this pessimistic estimate.")


if __name__ == "__main__":
    main()
