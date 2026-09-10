"""Timing probe for RQ4 Experiment 3 (local, leave-top-1-out SHAP stability)
-- XGBoost/RandomForest half, run in-process (no xgboost/torch conflict
concern for tree models). Companion to timing_probe.py's LSTM-side probe.
"""

import sys
import time
from pathlib import Path

import numpy as np

RQ4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RQ4_DIR))
import common as c  # noqa: E402

N_PROBE_ROWS = 10
PROBE_CLUSTER = "pastoral"
PROBE_LEAD = 0


def probe(subject, shap_fn, mask_fn, load_model_fn, impute):
    t0 = time.time()
    cell = c.get_tabular_cell(PROBE_CLUSTER, PROBE_LEAD, impute=impute)
    model = load_model_fn(PROBE_CLUSTER, PROBE_LEAD)
    t_setup = time.time() - t0

    test_x = cell["test_x"]
    rng = np.random.default_rng(c.SEED)
    probe_idx = rng.choice(len(test_x), size=min(N_PROBE_ROWS, len(test_x)), replace=False)
    X_probe = test_x.iloc[probe_idx].reset_index(drop=True)

    t1 = time.time()
    sv_unmasked = np.asarray(shap_fn(model, X_probe))
    t_unmasked = time.time() - t1

    top1_idx = np.abs(sv_unmasked).argmax(axis=1)
    top1_cols = [X_probe.columns[i] for i in top1_idx]
    print(f"[{subject}] distinct top-1 features in sample: {len(set(top1_cols))}/{len(top1_cols)}")

    t2 = time.time()
    per_row_times = []
    for i in range(len(X_probe)):
        tr = time.time()
        row = X_probe.iloc[[i]]
        if mask_fn is c.mask_median:
            masked = mask_fn(row, [top1_cols[i]], cell["train_medians"])
        else:
            masked = mask_fn(row, [top1_cols[i]])
        _ = shap_fn(model, masked)
        per_row_times.append(time.time() - tr)
    t_masked = time.time() - t2

    print(f"[{subject}] setup {t_setup:.2f}s | unmasked batch n={len(X_probe)} {t_unmasked:.2f}s "
          f"({t_unmasked/len(X_probe):.3f}s/row) | masked one-call-per-row {t_masked:.2f}s "
          f"({np.mean(per_row_times):.3f}s/row)")

    full_rows_per_lead = 828
    n_cells = 21  # 3 clusters x 7 leads
    est = (t_setup * n_cells) + (t_unmasked / len(X_probe)) * full_rows_per_lead * 7 \
        + np.mean(per_row_times) * full_rows_per_lead * 7
    print(f"[{subject}] extrapolated full scope (21 cells, 828 rows/lead pooled): "
          f"{est/60:.2f} min\n")


def main():
    probe("XGBoost", c.xgb_shap_values, c.mask_nan, c.load_xgb_model, impute=False)
    probe("RandomForest", c.rf_shap_values, c.mask_median, c.load_rf_model, impute=True)


if __name__ == "__main__":
    main()
