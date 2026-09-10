"""RQ4 Experiment 3 -- local (per-instance) leave-top-1-out SHAP stability,
XGBoost/RandomForest half.

H0: the mean per-row Spearman rank correlation between a row's SHAP
ranking of its own remaining features before and after masking that row's
own top-1 SHAP feature does not exceed 0.7.

Design (confirmed with the project owner after a timing probe showed the
full scope costs ~20 min total across all three subjects, not hours --
see timing_probe.py/timing_probe_trees.py):
  - For every test row in every (cluster, lead) cell: compute its own
    unmasked SHAP vector, find its own top-1 feature, mask ONLY that
    feature for THAT row, recompute SHAP, and correlate the ranking of
    the remaining features before vs after.
  - This is a genuinely different perturbation source from Experiment 2's
    aggregate bootstrap (which resamples which rows are averaged
    together and therefore cannot perturb any single row's own
    explanation at all -- see this experiment's own report.md section 1
    for why that reuse was rejected).
  - Same masking convention as Experiment 1/2: XGBoost -> NaN,
    RandomForest -> train-derived median (common.py::mask_nan/mask_median).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as c  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent


def row_local_stability(pre_sv_row, post_sv_row, top1_idx):
    """pre_sv_row, post_sv_row: 1D arrays over the SAME feature index space.
    top1_idx: index of the feature that was masked (excluded from both
    rankings before correlating -- it is undefined/near-zero post-mask by
    construction, so including it would trivially inflate agreement)."""
    keep = np.ones(len(pre_sv_row), dtype=bool)
    keep[top1_idx] = False
    pre = np.abs(pre_sv_row[keep])
    post = np.abs(post_sv_row[keep])
    if len(pre) < 2 or np.all(pre == pre[0]) or np.all(post == post[0]):
        return np.nan
    rho, _ = spearmanr(pre, post)
    return rho


def run_subject(subject, shap_fn, mask_fn, load_model_fn, impute):
    rows = []
    t_start = time.time()
    for cluster in c.CLUSTERS:
        for lead in c.LEADS:
            cell = c.get_tabular_cell(cluster, lead, impute=impute)
            model = load_model_fn(cluster, lead)
            test_x = cell["test_x"]
            test_meta = cell["test_meta"]
            n = len(test_x)
            if n == 0:
                continue

            pre_sv = np.asarray(shap_fn(model, test_x))  # (n, n_features)
            top1_idx = np.abs(pre_sv).argmax(axis=1)
            columns = list(test_x.columns)

            t_cell = time.time()
            for i in range(n):
                feat = columns[top1_idx[i]]
                row = test_x.iloc[[i]]
                if mask_fn is c.mask_median:
                    masked = mask_fn(row, [feat], cell["train_medians"])
                else:
                    masked = mask_fn(row, [feat])
                post_sv = np.asarray(shap_fn(model, masked))[0]
                rho = row_local_stability(pre_sv[i], post_sv, top1_idx[i])
                rows.append({
                    "subject": subject, "cluster": cluster, "lead": lead,
                    "zone_code": test_meta["zone_code"].iloc[i],
                    "time": test_meta["time"].iloc[i],
                    "top1_feature": feat, "rho": rho,
                })
            print(f"[{subject}] {cluster} lead={lead}: n={n}, "
                  f"{time.time() - t_cell:.1f}s", flush=True)
    print(f"[{subject}] TOTAL: {time.time() - t_start:.1f}s "
          f"({len(rows)} rows)", flush=True)
    return pd.DataFrame(rows)


def main():
    xgb_df = run_subject("xgboost", c.xgb_shap_values, c.mask_nan, c.load_xgb_model, impute=False)
    xgb_df.to_csv(OUT_DIR / "local_stability_xgboost.csv", index=False)

    rf_df = run_subject("randomforest", c.rf_shap_values, c.mask_median, c.load_rf_model, impute=True)
    rf_df.to_csv(OUT_DIR / "local_stability_randomforest.csv", index=False)

    print("\nDone. Wrote local_stability_xgboost.csv, local_stability_randomforest.csv")


if __name__ == "__main__":
    main()
