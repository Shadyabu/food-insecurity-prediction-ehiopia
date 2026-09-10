"""Does matching the paper's literal text (NDVI 2000-2021 baseline, strict
>5-day dry spell) reproduce the published reference better than matching
his actual code (no NDVI upper bound, >=5 days) does?

Compares `baseline_rerun_same_script/` against `paper_literal_variant/`,
both produced by the same `run_model.py`, against the same reference values
used in `metrics/statistical_acceptance_r2.csv` (from
RQ1_Experiment1_Busker_Reproduction.md section 2.3).
"""

import numpy as np
import pandas as pd

from build_acceptance_tables import (
    CLUSTER_LABELS,
    REF_HEADLINE,
    REF_R2_BY_ZONE_LEAD,
)

RUNS = {
    "baseline (code-accurate: no NDVI cap, >=5 days)": "baseline_rerun_same_script",
    "paper_literal (NDVI capped 2021, >5 days strict)": "paper_literal_variant",
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

    print("=== Per-cell |reproduced R2 - reference R2|: baseline vs. paper-literal variant ===\n")
    merged = tables[list(RUNS)[0]][["lhz", "lead", "reference_r2"]].copy()
    for name in RUNS:
        merged[name] = tables[name]["abs_diff"].values
    merged["variant_closer"] = merged[list(RUNS)[1]] < merged[list(RUNS)[0]]
    print(merged.round(4).to_string(index=False))

    print(f"\nCells where the paper-literal variant is closer to reference: "
          f"{merged['variant_closer'].sum()}/21")
    print(f"Cells where the baseline (code-accurate) is closer: "
          f"{(~merged['variant_closer']).sum()}/21")

    for name, table in tables.items():
        print(f"\n{name}:")
        print(f"  mean |diff| across 21 cells: {table['abs_diff'].mean():.4f}")
        print(f"  sum  |diff| across 21 cells: {table['abs_diff'].sum():.4f}")

    print("\n=== Headline numbers ===")
    for name, path in RUNS.items():
        lead_tbl = pd.read_csv(f"{path}/metrics_per_lead.csv").set_index("lead")
        r2_1 = lead_tbl.loc[1, "r2"]
        print(f"\n{name}:")
        print(f"  R2 lead1 whole-region: {r2_1:.4f} "
              f"(reference 0.72, |diff| {abs(r2_1 - 0.72):.4f})")
        for lead3_read, val in (
            ("unit-weighted mean MAE @ lead3",
             pd.read_csv(f"{path}/metrics_per_unit.csv")
               .query("lead == 3")["mae"].mean()),
            ("cluster-mean MAE @ lead3",
             pd.read_csv(f"{path}/metrics_per_cluster.csv")
               .query("lead == 3")["mae"].mean()),
            ("pooled MAE @ lead3", lead_tbl.loc[3, "mae"]),
        ):
            print(f"  {lead3_read}: {val:.4f} "
                  f"(reference 0.35, |diff| {abs(val - 0.35):.4f})")

    merged.to_csv("metrics/paper_literal_variant_comparison.csv", index=False)
    print("\nwrote metrics/paper_literal_variant_comparison.csv")


if __name__ == "__main__":
    main()
