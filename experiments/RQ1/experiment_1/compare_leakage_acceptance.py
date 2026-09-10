"""Follow-up to report.md section 5b: does the combined leakage-fix
variant ("noleak_all", closer to Busker's reference on mean |R2 diff|
and MAE/R2 @ matched leads) also pass MORE cells within 1 SD / 2 SD than
the baseline (official) reproduction? Reuses the exact SD definitions
and reference values already established for the baseline
(build_acceptance_tables.py's robust MAD-SD, bootstrap_acceptance.py's
bootstrap SE of the pooled R2), applied to a fresh run of the same
noleak_all input master (`data/interim/busker_baseline/input_master_noleak_all.parquet`,
still cached; the noleak_all/ prediction directory itself was not kept
per report.md section 5b's own note) -- confirmed to reproduce the
exact §5b table numbers (R2@lead1=0.7079, MAE@lead3=0.3121) before this
comparison was trusted.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from build_acceptance_tables import CLUSTER_LABELS, REF_R2_BY_ZONE_LEAD, _mad_sd
from bootstrap_acceptance import N_ITER, SEED, bootstrap_cell

BASELINE_DIR = "metrics"
NOLEAK_DIR = "noleak_all_rerun"


def robust_acceptance(metrics_dir):
    unit = pd.read_csv(f"{metrics_dir}/metrics_per_unit.csv")
    cluster = pd.read_csv(f"{metrics_dir}/metrics_per_cluster.csv")
    grouped = unit.groupby(["lhz", "lead"])["r2"]
    robust_sd = grouped.apply(_mad_sd)
    cluster_r2 = cluster.set_index(["lhz", "lead"])["r2"]

    rows = []
    for (lhz, lead), ref_r2 in REF_R2_BY_ZONE_LEAD.items():
        repro = cluster_r2.get((lhz, lead), np.nan)
        rs = robust_sd.get((lhz, lead), np.nan)
        abs_diff = abs(repro - ref_r2)
        rows.append({
            "lhz_code": lhz, "lead": lead, "reference_r2": ref_r2,
            "reproduced_r2": round(repro, 4), "abs_diff": round(abs_diff, 4),
            "robust_sd": round(rs, 4),
            "within_1sd": bool(abs_diff <= rs), "within_2sd": bool(abs_diff <= 2 * rs),
        })
    return pd.DataFrame(rows)


def bootstrap_acceptance_for(pred_path):
    predictions = pd.read_parquet(pred_path)
    rng = np.random.default_rng(SEED)
    rows = []
    for (lhz, lead), ref in REF_R2_BY_ZONE_LEAD.items():
        cell = predictions[(predictions["lhz"] == lhz) & (predictions["lead"] == lead)]
        cell = cell.dropna(subset=["observed", "prediction"])
        draws = bootstrap_cell(cell, rng=rng)
        reproduced_r2 = r2_score(cell["observed"], cell["prediction"])
        boot_sd = draws.std(ddof=1)
        abs_diff = abs(reproduced_r2 - ref)
        rows.append({
            "lhz_code": lhz, "lead": lead, "reference_r2": ref,
            "reproduced_r2": round(reproduced_r2, 4), "abs_diff": round(abs_diff, 4),
            "bootstrap_sd": round(boot_sd, 4),
            "within_1sd": bool(abs_diff <= boot_sd), "within_2sd": bool(abs_diff <= 2 * boot_sd),
        })
    return pd.DataFrame(rows)


def main():
    print("=== Robust MAD-SD acceptance ===")
    base_robust = robust_acceptance(BASELINE_DIR)
    noleak_robust = robust_acceptance(NOLEAK_DIR)
    print(f"Baseline (official):  {base_robust['within_1sd'].sum()}/21 within 1 SD, "
          f"{base_robust['within_2sd'].sum()}/21 within 2 SD")
    print(f"noleak_all (combined-fix): {noleak_robust['within_1sd'].sum()}/21 within 1 SD, "
          f"{noleak_robust['within_2sd'].sum()}/21 within 2 SD")

    print("\n=== Bootstrap-SE acceptance (500 unit resamples, seed=42) ===")
    base_boot = bootstrap_acceptance_for("predictions/predictions.parquet")
    noleak_boot = bootstrap_acceptance_for(f"{NOLEAK_DIR}/predictions.parquet")
    print(f"Baseline (official):  {base_boot['within_1sd'].sum()}/21 within 1 SD, "
          f"{base_boot['within_2sd'].sum()}/21 within 2 SD")
    print(f"noleak_all (combined-fix): {noleak_boot['within_1sd'].sum()}/21 within 1 SD, "
          f"{noleak_boot['within_2sd'].sum()}/21 within 2 SD")

    # which specific cells flip under bootstrap SE (the only criterion with headroom)
    merged = base_boot.merge(noleak_boot, on=["lhz_code", "lead"], suffixes=("_base", "_noleak"))
    flipped = merged[merged["within_1sd_base"] != merged["within_1sd_noleak"]]
    print("\nCells whose within-1-SD (bootstrap) status flips, baseline -> noleak_all:")
    print(flipped[["lhz_code", "lead", "within_1sd_base", "within_1sd_noleak",
                    "abs_diff_base", "abs_diff_noleak"]].to_string(index=False))

    base_robust.to_csv(f"{BASELINE_DIR}/leakage_acceptance_baseline_robust.csv", index=False)
    noleak_robust.to_csv(f"{NOLEAK_DIR}/leakage_acceptance_noleak_robust.csv", index=False)
    base_boot.to_csv(f"{BASELINE_DIR}/leakage_acceptance_baseline_bootstrap.csv", index=False)
    noleak_boot.to_csv(f"{NOLEAK_DIR}/leakage_acceptance_noleak_bootstrap.csv", index=False)


if __name__ == "__main__":
    main()
