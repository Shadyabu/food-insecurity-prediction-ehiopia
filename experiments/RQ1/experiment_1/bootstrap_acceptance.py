"""Bootstrap standard error of the *pooled* R2 statistic, per zone x lead cell.

RQ1_Experiment1_Busker_Reproduction.md section 3 step 5's own named fallback:
"resample administrative units with replacement (500 iterations) within each
zone x lead-time group, recompute R2/MAE each time, and use the resulting
distribution's SD." Not used originally because per-unit metrics were
available (step 1-4's literal method) -- but that method's SD describes
unit-to-unit heterogeneity, not the sampling uncertainty of the pooled R2
number actually being compared to Busker's reference, and turned out far
too wide to be an informative test (mean 0.37 R2, vs a metric that only
spans 0-1). This is the more defensible alternative: resample which units
feed the pooled calculation, so a single small-sample/pathological unit
(see build_acceptance_tables.py's _mad_sd docstring) can only ever
contribute its own handful of points to the pool, not dominate a summary
statistic the way it dominates a per-unit R2 value.

Resampling is done at the unit level (whole time series moved together),
not per-row, so within-unit temporal autocorrelation isn't broken into
pseudo-independent rows.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from build_acceptance_tables import CLUSTER_LABELS, REF_R2_BY_ZONE_LEAD

PRED_PATH = "predictions/predictions.parquet"
N_ITER = 500
SEED = 42


def bootstrap_cell(df_cell, n_iter=N_ITER, rng=None):
    """df_cell: rows for one (lhz, lead), columns county/observed/prediction.

    Returns array of length n_iter: pooled R2 on each unit-resampled draw.
    """
    counties = df_cell["county"].unique()
    n_units = len(counties)
    by_county = {c: g[["observed", "prediction"]].to_numpy() for c, g in df_cell.groupby("county")}

    r2_draws = np.empty(n_iter)
    for i in range(n_iter):
        draw = rng.choice(counties, size=n_units, replace=True)
        pooled = np.concatenate([by_county[c] for c in draw], axis=0)
        r2_draws[i] = r2_score(pooled[:, 0], pooled[:, 1])
    return r2_draws


def main():
    predictions = pd.read_parquet(PRED_PATH)
    rng = np.random.default_rng(SEED)

    rows = []
    for (lhz, lead), ref in REF_R2_BY_ZONE_LEAD.items():
        cell = predictions[(predictions["lhz"] == lhz) & (predictions["lead"] == lead)]
        cell = cell.dropna(subset=["observed", "prediction"])
        n_units = cell["county"].nunique()

        draws = bootstrap_cell(cell, rng=rng)
        reproduced_r2 = r2_score(cell["observed"], cell["prediction"])
        boot_sd = draws.std(ddof=1)
        abs_diff = abs(reproduced_r2 - ref)

        rows.append({
            "livelihood_zone": CLUSTER_LABELS[lhz], "lhz_code": lhz, "lead": lead,
            "reference_r2": ref, "reproduced_r2": round(reproduced_r2, 4),
            "n_units": n_units, "n_bootstrap_iters": N_ITER,
            "bootstrap_r2_sd": round(boot_sd, 4),
            "bootstrap_r2_ci_lo": round(np.percentile(draws, 2.5), 4),
            "bootstrap_r2_ci_hi": round(np.percentile(draws, 97.5), 4),
            "abs_diff": round(abs_diff, 4),
            "within_1sd_bootstrap": bool(abs_diff <= boot_sd),
            "t_like_statistic": round(abs_diff / boot_sd, 3) if boot_sd > 0 else np.nan,
        })
        print(f"  {CLUSTER_LABELS[lhz]:14} lead {lead:2}  n_units={n_units:3}  "
              f"reproduced={reproduced_r2:.4f}  reference={ref:.4f}  "
              f"bootstrap_SD={boot_sd:.4f}  |diff|/SD={abs_diff / boot_sd if boot_sd else float('nan'):.2f}")

    out = pd.DataFrame(rows).sort_values(["lhz_code", "lead"])
    out.to_csv("metrics/statistical_acceptance_r2_bootstrap.csv", index=False)

    n_pass = out["within_1sd_bootstrap"].sum()
    print(f"\nBootstrap SD acceptance: {n_pass}/21 within 1 bootstrap SD")
    print(f"Mean bootstrap SD: {out['bootstrap_r2_sd'].mean():.4f} "
          f"(vs. mean robust per-unit SD 0.3737 previously)")
    print(f"Mean |diff|/SD (like a t-statistic; >~2 would typically be a 5% two-sided reject): "
          f"{out['t_like_statistic'].mean():.3f}, max {out['t_like_statistic'].max():.3f}")
    print("\nwrote metrics/statistical_acceptance_r2_bootstrap.csv")


if __name__ == "__main__":
    main()
