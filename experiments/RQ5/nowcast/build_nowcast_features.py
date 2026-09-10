"""RQ5 nowcast, Phase 0a -- construct genuine forward-looking feature rows
for target months beyond the model_tables' own scaffold ceiling.

WHY THIS EXISTS: every model_table row is (target_month, lead) with
target_month capped at the panel's own ceiling (currently 2026-06-01, the
most recent real observed==True IPC assessment). There is no row for a
target month beyond that ceiling in any lead file -- the "lead" columns for
a fixed target are all different-staleness *retrospective* views of that
SAME target, not a forward horizon. To get a genuine "predictions for the
next 12 months from today" product, this script builds NEW rows: for origin
month O = 2026-06-01 and lead L, target = O + L, features = O's own real,
already-known values (exactly what a lead-L model was trained to consume --
data known L months before its target).

Reuses, rather than reimplements, two pieces of already-validated logic:
  - pipelines/model_join/build_ethiopia_model_tables.py's IDENTITY_COLS /
    FORECAST_PRESELECT_BY_LEAD / ALL_FORECAST_OUTPUT_COLS (Stage 2's own
    column classification -- imported directly, not copy-pasted).
  - boundaries/crop_calendar_admin2_month.csv for the IDENTITY-bucket
    season columns, which must be evaluated at the TARGET month's calendar
    month (per CLAUDE.md: these are deterministic and knowable regardless
    of lead, so Stage 2 never shifts them) -- origin O's own calendar month
    is the wrong lookup for L>0.

One documented assumption: `months_since_assessment` (IDENTITY-bucket, kept
as a real feature by run_model_ethiopia.py, unlike most IDENTITY columns)
is extrapolated as `origin_value + lead` ("if no new FEWS release arrives
between now and the target, staleness grows by exactly `lead` months").
Confirmed empirically that origin O's own value is 0 for all 92 zones (O is
itself a fresh assessment month), so this reduces to
`months_since_assessment(target) = lead`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "model_join"))
import build_ethiopia_model_tables as mj  # noqa: E402

RAW_PANEL_PATH = REPO_ROOT / "data" / "interim" / "model_join" / "ethiopia_raw_wide_panel.csv"
CROP_CALENDAR_PATH = REPO_ROOT / "boundaries" / "crop_calendar_admin2_month.csv"
OUT_DIR = Path(__file__).resolve().parent

ORIGIN_MONTH = pd.Timestamp("2026-06-01")
LEADS = [0, 1, 2, 3, 4, 8, 12]

# IDENTITY-bucket columns whose value must be looked up fresh from the crop
# calendar at the TARGET month's calendar month, not carried over from O.
CROP_CALENDAR_COLS = [
    "season_system", "is_harvest_season", "is_lean_season", "is_dry_season",
    "months_since_harvest_start", "months_since_lean_start",
]

# IDENTITY-bucket columns that are static zone attributes -- correct to
# carry over from O unchanged (per build_ethiopia_model_tables.py's own
# docstring: "livelihood-zone shares... and crop calendar... glofas_reach_count
# and agss_surveyed_flag are static zone attributes for the same reason").
STATIC_IDENTITY_COLS = [
    "zone_code", "zone_id", "zone_name",
    "pct_pastoral", "pct_agropastoral", "pct_crop_farming", "dominant_livelihood_zone",
    "glofas_reach_count", "agss_surveyed_flag",
]


def load_origin_row(panel: pd.DataFrame) -> pd.DataFrame:
    # raw_wide_panel's own "month" column is "YYYY-MM" (confirmed by inspection),
    # matching model_tables' convention -- keep it as that same string
    # everywhere in this module rather than mixing string/datetime forms.
    origin = panel[panel["month"] == ORIGIN_MONTH.strftime("%Y-%m")].copy()
    assert len(origin) == 92, f"expected 92 zones at origin month, got {len(origin)}"
    assert (origin["observed"] == True).all(), "origin month must be a real observed assessment"  # noqa: E712
    assert (origin["months_since_assessment"] == 0).all(), "origin month must have months_since_assessment==0"
    return origin.reset_index(drop=True)


def load_crop_calendar() -> pd.DataFrame:
    cc = pd.read_csv(CROP_CALENDAR_PATH)
    return cc


def build_row_for_lead(origin: pd.DataFrame, crop_cal: pd.DataFrame, lead: int) -> pd.DataFrame:
    """One row per zone: target_month = ORIGIN_MONTH + lead months.

    Output columns are deliberately kept schema-compatible with
    data/processed/model_tables/ethiopia_lead{L}.csv (same `month` =
    target month, `origin_month` = origin, `lead_months` = lead) so this
    row can simply be concatenated onto a real lead table and run through
    the exact same `build_full_feature_matrix()` transform every other
    RQ5/RQ1/RQ2 script already uses, instead of re-deriving that transform
    here and risking a subtle mismatch.
    """
    target_month = ORIGIN_MONTH + pd.DateOffset(months=lead)
    target_calendar_month = target_month.month

    row = origin.copy()
    row["month"] = target_month.strftime("%Y-%m")  # lead-table convention is "YYYY-MM", not "YYYY-MM-DD"; was origin's month -- must be the TARGET month
    row["origin_month"] = ORIGIN_MONTH.strftime("%Y-%m")
    row["lead_months"] = lead
    row["lead"] = lead  # convenience alias, dropped again at model-feature reindex time
    row["target_month"] = target_month
    row["months_since_assessment"] = lead  # documented assumption, see module docstring
    row = row.drop(columns=[c for c in mj.EXCLUDED_COLS if c in row.columns])

    # --- crop calendar: re-lookup at the TARGET month's calendar month.
    # At lead=0, target==origin, so this is a no-op re-confirmation, not a
    # special case -- keeps the code path identical for every lead. ---
    cc_month = crop_cal[crop_cal["month"] == target_calendar_month].set_index("zone_code")
    for col in CROP_CALENDAR_COLS:
        row[col] = row["zone_code"].map(cc_month[col])

    # --- forecast-preselect: only meaningful at lead in {1, 3}, matching
    # Stage 2's own output schema exactly (bare renamed column, NaN at
    # every other lead including 0 -- GloFAS/IRI/USGS never reach lead0). ---
    forecast_map = mj.FORECAST_PRESELECT_BY_LEAD.get(lead, {})
    for src_col, out_col in forecast_map.items():
        if src_col in row.columns:
            row[out_col] = row[src_col]
    for out_col in mj.ALL_FORECAST_OUTPUT_COLS:
        if out_col not in forecast_map.values():
            row[out_col] = float("nan")
    # Drop the raw suffixed source columns -- Stage 2's own lead tables
    # never expose these, only the renamed bare-name output columns.
    # `usgs_gefs_coverage_days` is a source-column name AND an output-column
    # name simultaneously (it has no _lead1/_lead3 suffix in the raw panel,
    # per FORECAST_PRESELECT_BY_LEAD's own comment) -- must not drop that one.
    drop_cols = [c for c in mj.ALL_FORECAST_SOURCE_COLS if c in row.columns and c not in mj.ALL_FORECAST_OUTPUT_COLS]
    row = row.drop(columns=drop_cols)

    return row


def main():
    print(f"Loading raw wide panel from {RAW_PANEL_PATH} ...")
    panel = pd.read_csv(RAW_PANEL_PATH, low_memory=False)
    origin = load_origin_row(panel)
    crop_cal = load_crop_calendar()
    print(f"Origin month confirmed: {ORIGIN_MONTH.date()}, 92/92 zones, months_since_assessment==0 for all.")

    all_rows = []
    for lead in LEADS:
        r = build_row_for_lead(origin, crop_cal, lead)
        all_rows.append(r)
        print(f"  lead={lead:2d} -> target_month={ (ORIGIN_MONTH + pd.DateOffset(months=lead)).date() }  ({len(r)} zones)")

    out = pd.concat(all_rows, ignore_index=True)
    out_path = OUT_DIR / "nowcast_feature_rows.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {len(out)} rows ({len(LEADS)} leads x 92 zones) to {out_path}")


if __name__ == "__main__":
    main()
