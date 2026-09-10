"""RQ3 Spearman correlation + bootstrap CI between data completeness and F1.

scipy.stats.spearmanr gives rho and a p-value but no confidence interval, so
a nonparametric bootstrap (resample the (completeness, F1) pairs with
replacement, recompute rho each time, take the 2.5th/97.5th percentile of the
bootstrap distribution) is used instead -- the same percentile-bootstrap
pattern already established in this project for statistical inference
(experiments/RQ1/experiment_1/bootstrap_acceptance.py,
experiments/RQ2/experiment_5/significance_test.py's zone-block bootstrap).
Fixed seed (42, this project's standing convention) for reproducibility.
"""
import numpy as np
from scipy.stats import spearmanr

DEFAULT_N_BOOT = 5000
DEFAULT_SEED = 42


def spearman_with_bootstrap_ci(x, y, n_boot=DEFAULT_N_BOOT, seed=DEFAULT_SEED):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)

    rho, p = spearmanr(x, y)

    rng = np.random.default_rng(seed)
    boot_rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # A resample can (rarely, at small n) draw a constant x or y, which
        # makes rho undefined (scipy returns nan) -- kept as nan and excluded
        # via nanpercentile below, not silently coerced to 0.
        boot_rhos[i], _ = spearmanr(x[idx], y[idx])

    ci_low, ci_high = np.nanpercentile(boot_rhos, [2.5, 97.5])
    n_nan_boot = int(np.isnan(boot_rhos).sum())

    return {
        "n": n,
        "rho": rho,
        "p_value": p,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_boot": n_boot,
        "n_boot_degenerate": n_nan_boot,
        "reject_h0_at_0.05": bool(p < 0.05),
    }
