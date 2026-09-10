"""Completes report.md section 5b's leakage-ablation test: the full 2^3
grid over the 3 leakage toggles (NDVI/SPI/SPEI/SSMI climatology cap,
GDP/CPI publication lag, teleconnection lag), not just baseline + the 3
single fixes + all-3-combined that report.md already covers. The 3
missing pairwise combinations (NDVI+GDPCPI, NDVI+teleconn, GDPCPI+teleconn)
were built by reconstructing the publication-lag CLI flags report.md
§5b says were removed from build_data_master.py after the original test
-- each reconstructed flag value was validated by regenerating the
already-cached single-fix input masters and diffing byte-for-byte
against them before being trusted (see 2026-08-26 experiment_log.md
entry): NDVI baseline cap = 2018 (exact match), CPI lag = 1 month (exact
except 139/58788 boundary rows), teleconnection lag = 1 month (exact
except 213/58788 boundary rows), GDP lag = 15 months / "visible from
April of the following year" (exact match, 0 mismatches).

Scores each of the 8 grid cells against Busker's 21 published R2 values
with the same three criteria already used elsewhere in this experiment:
mean |R2 diff|, robust MAD-SD pass count (1 & 2 SD), and bootstrap-SE
pass count (1 & 2 SD, 500 unit resamples, seed 42).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from build_acceptance_tables import REF_R2_BY_ZONE_LEAD, _mad_sd
from bootstrap_acceptance import N_ITER, SEED, bootstrap_cell

CELLS = [
    ("baseline", "metrics", "predictions/predictions.parquet"),
    ("ndvi", "leak_ndvi_only_rerun", "leak_ndvi_only_rerun/predictions.parquet"),
    ("gdpcpi", "leak_gdpcpi_only_rerun", "leak_gdpcpi_only_rerun/predictions.parquet"),
    ("teleconn", "leak_teleconn_only_rerun", "leak_teleconn_only_rerun/predictions.parquet"),
    ("ndvi+gdpcpi", "leak_ndvi_gdpcpi_rerun", "leak_ndvi_gdpcpi_rerun/predictions.parquet"),
    ("ndvi+teleconn", "leak_ndvi_teleconn_rerun", "leak_ndvi_teleconn_rerun/predictions.parquet"),
    ("gdpcpi+teleconn", "leak_gdpcpi_teleconn_rerun", "leak_gdpcpi_teleconn_rerun/predictions.parquet"),
    ("all_3", "noleak_all_rerun", "noleak_all_rerun/predictions.parquet"),
]


def robust_summary(metrics_dir):
    unit = pd.read_csv(f"{metrics_dir}/metrics_per_unit.csv")
    cluster = pd.read_csv(f"{metrics_dir}/metrics_per_cluster.csv")
    grouped = unit.groupby(["lhz", "lead"])["r2"]
    robust_sd = grouped.apply(_mad_sd)
    cluster_r2 = cluster.set_index(["lhz", "lead"])["r2"]

    diffs, within1, within2 = [], 0, 0
    for (lhz, lead), ref_r2 in REF_R2_BY_ZONE_LEAD.items():
        repro = cluster_r2.get((lhz, lead), np.nan)
        rs = robust_sd.get((lhz, lead), np.nan)
        abs_diff = abs(repro - ref_r2)
        diffs.append(abs_diff)
        within1 += abs_diff <= rs
        within2 += abs_diff <= 2 * rs
    return np.mean(diffs), within1, within2


def bootstrap_summary(pred_path):
    predictions = pd.read_parquet(pred_path)
    rng = np.random.default_rng(SEED)
    within1, within2 = 0, 0
    for (lhz, lead), ref in REF_R2_BY_ZONE_LEAD.items():
        cell = predictions[(predictions["lhz"] == lhz) & (predictions["lead"] == lead)]
        cell = cell.dropna(subset=["observed", "prediction"])
        draws = bootstrap_cell(cell, rng=rng)
        reproduced_r2 = r2_score(cell["observed"], cell["prediction"])
        boot_sd = draws.std(ddof=1)
        abs_diff = abs(reproduced_r2 - ref)
        within1 += abs_diff <= boot_sd
        within2 += abs_diff <= 2 * boot_sd
    return within1, within2


def main():
    rows = []
    for name, metrics_dir, pred_path in CELLS:
        mean_diff, r1, r2 = robust_summary(metrics_dir)
        b1, b2 = bootstrap_summary(pred_path)
        rows.append({
            "combo": name, "mean_abs_r2_diff": round(mean_diff, 4),
            "robust_within_1sd": r1, "robust_within_2sd": r2,
            "bootstrap_within_1sd": b1, "bootstrap_within_2sd": b2,
        })
        print(f"{name:18} mean|diff|={mean_diff:.4f}  robust 1SD={r1}/21 2SD={r2}/21  "
              f"bootstrap 1SD={b1}/21 2SD={b2}/21")

    out = pd.DataFrame(rows).sort_values("mean_abs_r2_diff")
    out.to_csv("metrics/leakage_grid_full_comparison.csv", index=False)
    print(f"\nwrote metrics/leakage_grid_full_comparison.csv")
    print("\nRanked by mean |R2 diff| (lower = closer to Busker's reference):")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
