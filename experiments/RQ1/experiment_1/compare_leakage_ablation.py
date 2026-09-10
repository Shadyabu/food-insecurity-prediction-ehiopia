"""Ablation: does removing the three leakage sources found in the "is there
data leakage" discussion reproduce Busker et al. (2024) better or worse?

  1. NDVI climatology capped pre-test-period (baseline_end_year=2018)
  2. GDP (15mo) + CPI (1mo) publication lag
  3. MEI/NINA34/IOD teleconnection lag (1mo, partial mitigation only)

Compares the baseline (his actual leaky code) against all-three-combined and
each fix individually, all against the same reference values used
throughout this experiment. Same structure as compare_variant.py.
"""

import numpy as np
import pandas as pd

from build_acceptance_tables import CLUSTER_LABELS, REF_R2_BY_ZONE_LEAD

RUNS = {
    "baseline (his actual leaky code)": "baseline_rerun_same_script",
    "all 3 fixes combined": "noleak_all",
    "NDVI climatology fix only": "noleak_ndvi",
    "GDP/CPI publication-lag fix only": "noleak_gdpcpi",
    "teleconnection-lag fix only": "noleak_teleconn",
}


def r2_table(run_dir):
    cluster = pd.read_csv(f"{run_dir}/metrics_per_cluster.csv").set_index(["lhz", "lead"])
    rows = []
    for (lhz, lead), ref in REF_R2_BY_ZONE_LEAD.items():
        repro = cluster.loc[(lhz, lead), "r2"]
        rows.append({"lhz": lhz, "lead": lead, "reference_r2": ref,
                     "reproduced_r2": repro, "abs_diff": abs(repro - ref)})
    return pd.DataFrame(rows)


def main():
    tables = {name: r2_table(path) for name, path in RUNS.items()}
    baseline_name = list(RUNS)[0]

    print("=== Per-cell |reproduced R2 - reference R2| across all 5 runs ===\n")
    merged = tables[baseline_name][["lhz", "lead", "reference_r2"]].copy()
    for name in RUNS:
        merged[name] = tables[name]["abs_diff"].values
    print(merged.round(4).to_string(index=False))

    print("\n=== Summary across 21 cells ===")
    summary_rows = []
    for name, table in tables.items():
        n_closer = (table["abs_diff"].values < tables[baseline_name]["abs_diff"].values).sum()
        summary_rows.append({
            "run": name,
            "mean_abs_diff": table["abs_diff"].mean(),
            "sum_abs_diff": table["abs_diff"].sum(),
            "n_cells_closer_than_baseline": n_closer if name != baseline_name else "-",
        })
    summary = pd.DataFrame(summary_rows)
    print(summary.round(4).to_string(index=False))

    print("\n=== Headline numbers ===")
    for name, path in RUNS.items():
        lead_tbl = pd.read_csv(f"{path}/metrics_per_lead.csv").set_index("lead")
        r2_1 = lead_tbl.loc[1, "r2"]
        mae_3 = lead_tbl.loc[3, "mae"]
        print(f"{name:38} R2@lead1={r2_1:.4f} (ref 0.72, |d|={abs(r2_1 - 0.72):.4f})  "
              f"MAE@lead3(pooled)={mae_3:.4f} (ref 0.35, |d|={abs(mae_3 - 0.35):.4f})")

    merged.to_csv("metrics/leakage_ablation_comparison.csv", index=False)
    summary.to_csv("metrics/leakage_ablation_summary.csv", index=False)
    print("\nwrote metrics/leakage_ablation_comparison.csv, metrics/leakage_ablation_summary.csv")


if __name__ == "__main__":
    main()
