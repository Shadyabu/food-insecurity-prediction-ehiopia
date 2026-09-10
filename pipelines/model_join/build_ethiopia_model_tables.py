"""
build_ethiopia_model_tables.py

Stage 2 of the Ethiopia feature-target join. Consumes
data/interim/model_join/ethiopia_raw_wide_panel.csv (stage 1, natural/
unshifted month) and produces the 7 lead-time model tables at
data/processed/model_tables/ethiopia_lead{00,01,02,03,04,08,12}.csv --
same 7 leads as Busker's own LEADS (pipelines/busker_baseline/busker_paths.py)
and the existing ipc_target lead tables, confirmed 2026-08-20 to reuse the
Busker-parity train/test split already embedded in the IPC base panel
rather than dissertation_plan.md's 2011-2021/2022-2026 wording.

Every column in the raw wide panel is classified into exactly one of four
groups (CLASSIFICATION assertion below fails loudly if a column is missed
-- with 320+ columns from 18 sources, silent misclassification is a real
risk, not a hypothetical one):

1. IDENTITY -- never shifted, same value in every lead table regardless of
   lead. Two sub-cases:
   - Target/target-metadata (ipc_continuous, ha_share, split, ...) --
     matches the existing ipc_lead*.csv convention exactly (see
     pipelines/ipc_target/build_ipc_features.py's make_lead_table(), which
     already treats these as never-shifted).
   - Deterministic reference data with NO leakage risk regardless of lead:
     livelihood-zone shares (static per zone, never change) and crop
     calendar (is_harvest_season etc. -- which season a future calendar
     month falls in is always knowable in advance, unlike every other
     feature in this join, so evaluating it at the TARGET month rather
     than shifting it is the informative, leakage-safe choice, not a
     shortcut). glofas_reach_count and agss_surveyed_flag are static zone
     attributes for the same reason.

2. EXCLUDED -- RQ1-fidelity-only snapshot columns (headline_cpi_undated,
   gdp_per_capita_undated, wvg_undated, and their yoy_change counterparts).
   Each source's own docs explicitly say "do not use in an RQ2+ model" --
   this is that dataset, so they are dropped here rather than propagated
   into 7 files where they'd invite accidental misuse.

3. FORECAST_PRESELECT -- GloFAS/IRI-CPC/USGS-GEFS's pre-aligned lead1/
   lead3 columns. Per CLAUDE.md's note on each of these three pipelines,
   these must be SELECTED at the matching lead, not shifted again -- the
   column already represents the correct forecast vintage for predicting
   the target month at that lead, baked in at build time by each source's
   own pipeline. Renamed to drop the _lead1/_lead3 suffix in the output
   (a lead table doesn't need "lead1" in a column name once the whole
   file already is the lead-1 table). Absent (NaN) at lead times no
   forecast product reaches (0, 2, 4, 8, 12) -- an honest reflection of
   "no forecast exists at this horizon," not a gap to fill.

4. SHIFT -- everything else (the large majority: every source's own
   already-engineered lag/rolling/index columns). Shifted back by `lead`
   months per zone_code, ALL TOGETHER in one groupby().shift() call
   (CLAUDE.md Sec 3.2: "shift the whole feature block together... never
   shift feature families independently of each other").
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INPUT_PATH = os.path.join(REPO_ROOT, "data", "interim", "model_join", "ethiopia_raw_wide_panel.csv")
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "processed", "model_tables")

LEADS = [0, 1, 2, 3, 4, 8, 12]  # matches pipelines/busker_baseline/busker_paths.py LEADS exactly

IDENTITY_COLS = [
    "zone_code", "month", "zone_id", "zone_name",
    "ipc_continuous", "ipc_phase_20pct", "pct_phase3plus", "pop_coverage",
    "ha_share", "ipc_continuous_area", "assessment_month",
    "months_since_assessment", "observed", "split",
    "pct_pastoral", "pct_agropastoral", "pct_crop_farming", "dominant_livelihood_zone",
    "season_system", "is_harvest_season", "is_lean_season", "is_dry_season",
    "months_since_harvest_start", "months_since_lean_start",
    "glofas_reach_count", "agss_surveyed_flag",
]

EXCLUDED_COLS = [
    "headline_cpi_undated", "headline_cpi_yoy_change_undated",
    "food_cpi_undated", "food_cpi_yoy_change_undated",
    "gdp_per_capita_undated", "gdp_per_capita_yoy_change_undated",
    "wvg_undated",
]

# source_col -> output_col (suffix stripped). Applies identically whether
# the source column came from the _lead1 or _lead3 family; which family is
# read is determined by FORECAST_PRESELECT_BY_LEAD below, so there is no
# collision within a single lead table's output.
FORECAST_RENAME = {
    "glofas_exceed_2yr": "glofas_exceed_2yr",
    "glofas_exceed_20yr": "glofas_exceed_20yr",
    "glofas_source": "glofas_source",
    "seasonal_precip_prob_below": "seasonal_precip_prob_below",
    "seasonal_precip_prob_near": "seasonal_precip_prob_near",
    "seasonal_precip_prob_above": "seasonal_precip_prob_above",
    "seasonal_target_season": "seasonal_target_season",
    "seasonal_overlaps_kiremt": "seasonal_overlaps_kiremt",
    "seasonal_overlaps_belg": "seasonal_overlaps_belg",
    "seasonal_overlaps_gu": "seasonal_overlaps_gu",
    "seasonal_overlaps_deyr": "seasonal_overlaps_deyr",
    "seasonal_source": "seasonal_source",
    "usgs_gefs_precip_mm": "usgs_gefs_precip_mm",
    "usgs_gefs_precip_anom": "usgs_gefs_precip_anom",
    "usgs_gefs_issue_date": "usgs_gefs_issue_date",
    "usgs_gefs_source": "usgs_gefs_source",
    "usgs_gefs_anom_units": "usgs_gefs_anom_units",
}

# lead -> {raw_column_in_panel: output_column}. lead1 has GloFAS + IRI +
# USGS (USGS's only lead); lead3 has GloFAS + IRI only (USGS never reaches
# lead3). usgs_gefs_coverage_days has no _lead1 suffix in the source file
# (it's a constant, always 15) but is only meaningful alongside the lead1
# USGS forecast, so it's grouped here, not in IDENTITY.
FORECAST_PRESELECT_BY_LEAD = {
    1: {f"{k}_lead1": v for k, v in FORECAST_RENAME.items()},
    3: {f"{k}_lead3": v for k, v in FORECAST_RENAME.items() if not k.startswith("usgs_gefs")},
}
FORECAST_PRESELECT_BY_LEAD[1]["usgs_gefs_coverage_days"] = "usgs_gefs_coverage_days"

ALL_FORECAST_OUTPUT_COLS = sorted(set(FORECAST_PRESELECT_BY_LEAD[1].values()) | set(FORECAST_PRESELECT_BY_LEAD[3].values()))
ALL_FORECAST_SOURCE_COLS = sorted(set(FORECAST_PRESELECT_BY_LEAD[1].keys()) | set(FORECAST_PRESELECT_BY_LEAD[3].keys()))


def classify_columns(all_cols):
    identity = [c for c in IDENTITY_COLS if c in all_cols]
    excluded = [c for c in EXCLUDED_COLS if c in all_cols]
    forecast = [c for c in ALL_FORECAST_SOURCE_COLS if c in all_cols]
    shift = [c for c in all_cols if c not in identity and c not in excluded and c not in forecast]

    accounted = set(identity) | set(excluded) | set(forecast) | set(shift)
    missing = set(all_cols) - accounted
    if missing:
        raise ValueError(f"Unclassified columns (fix IDENTITY_COLS/EXCLUDED_COLS/FORECAST_RENAME): {sorted(missing)}")
    dupe_check = len(identity) + len(excluded) + len(forecast) + len(shift)
    if dupe_check != len(all_cols):
        raise ValueError("A column was classified into more than one group -- check for overlap")

    unmatched_identity = set(IDENTITY_COLS) - set(identity)
    unmatched_excluded = set(EXCLUDED_COLS) - set(excluded)
    unmatched_forecast = set(ALL_FORECAST_SOURCE_COLS) - set(forecast)
    if unmatched_identity or unmatched_excluded or unmatched_forecast:
        raise ValueError(
            f"Expected columns not found in raw panel -- source schema may have changed: "
            f"identity={sorted(unmatched_identity)}, excluded={sorted(unmatched_excluded)}, "
            f"forecast={sorted(unmatched_forecast)}"
        )
    return identity, shift, forecast


def make_lead_table(panel, identity_cols, shift_cols, lead):
    """Shift every SHIFT column back by `lead` months per zone, keep
    IDENTITY columns fixed at the target month, and select (not shift)
    the matching forecast-preselect columns if this lead has any --
    mirrors pipelines/ipc_target/build_ipc_features.py's make_lead_table()
    exactly for the shift mechanism, extended with the forecast-select
    exception documented in the module docstring."""
    out = panel.sort_values(["zone_code", "month"]).copy()
    grp = out.groupby("zone_code", sort=False)

    for col in shift_cols:
        out[col] = grp[col].shift(lead)

    forecast_map = FORECAST_PRESELECT_BY_LEAD.get(lead, {})
    forecast_out_cols = []
    for src_col, out_col in forecast_map.items():
        out[out_col] = out[src_col]  # same row, no shift -- already the correct forecast vintage
        forecast_out_cols.append(out_col)
    # Columns this lead's forecast products don't reach: present (schema
    # consistency across all 7 files) but NaN, not silently dropped.
    for out_col in ALL_FORECAST_OUTPUT_COLS:
        if out_col not in forecast_out_cols:
            out[out_col] = pd.NA

    out["lead_months"] = lead
    out["origin_month"] = (pd.PeriodIndex(out["month"], freq="M") - lead).astype(str)

    final_cols = identity_cols + shift_cols + ALL_FORECAST_OUTPUT_COLS + ["lead_months", "origin_month"]
    return out[final_cols]


def main():
    panel = pd.read_csv(INPUT_PATH)
    print(f"Loaded raw wide panel: {panel.shape[0]} rows x {panel.shape[1]} cols")

    identity_cols, shift_cols, forecast_source_cols = classify_columns(list(panel.columns))
    print(f"Classified: {len(identity_cols)} identity, {len(shift_cols)} shift, "
          f"{len(forecast_source_cols)} forecast-preselect source cols "
          f"({len(ALL_FORECAST_OUTPUT_COLS)} renamed output cols), "
          f"{len(EXCLUDED_COLS)} excluded (RQ1-only)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for lead in LEADS:
        lead_table = make_lead_table(panel, identity_cols, shift_cols, lead)
        out_path = os.path.join(OUTPUT_DIR, f"ethiopia_lead{lead:02d}.csv")
        lead_table.to_csv(out_path, index=False)

        n_forecast_populated = sum(lead_table[c].notna().any() for c in ALL_FORECAST_OUTPUT_COLS)
        print(f"  lead{lead:02d}: {lead_table.shape[0]} rows x {lead_table.shape[1]} cols -> {out_path} "
              f"({n_forecast_populated}/{len(ALL_FORECAST_OUTPUT_COLS)} forecast cols populated)")


if __name__ == "__main__":
    main()
