"""
build_ethiopia_raw_wide_panel.py

Stage 1 of the Ethiopia feature-target join (CLAUDE.md Sec 3.6: Acquire/
Aggregate/Engineer/Validate stay separate, independently re-runnable
stages). Joins every non-locust feature source onto the IPC target panel
at NATURAL (unshifted) month -- one row per (zone_code, month), same as
every source's own native grain. The lead-time shift (CLAUDE.md Sec 3.2)
happens in a separate later stage, build_ethiopia_model_tables.py, so this
file can be re-run on its own whenever an upstream source changes without
re-deriving the shift logic.

ANCHOR: data/interim/ipc_target/model_tables/ipc_monthly_base.csv -- the
IPC target's own pre-lead-shift table (target + ipc_lag1/4/8/mean12 +
ha_share_lag1/4/8/mean12 + the Busker-parity train/test split column,
confirmed 2026-08-20 to be reused as-is for this project, not rebuilt to
docs/dissertation_plan.md's 2011-2021/2022-2026 wording). zone_code x
month, 2011-04 to 2026-06, 92 zones.

EXCLUDED SOURCE: pipelines/locust/ -- excluded from the join per the
2026-08-18 decision (documented CLAUDE.md Sec 2 / master table): despite
passing validation, the 2022 no-source gap was judged too unreliable.

JOIN PATTERNS (CLAUDE.md Sec 3.1/3.4):
  - zone-monthly: merge on (zone_code, year, month-of-year). 11 sources --
    chirps, gleam_ssmi_spei, ndvi x3, glofas, iri_cpc_seasonal,
    usgs_rainfall_forecast, acled, wfp_prices, agss_yields.
  - national broadcast: merge on (year, month-of-year) only, so every
    zone_code in a given month gets the identical value. 6 sources -- cpi,
    exchange_rate, imf_gdp, teleconnections, wvg_monthly,
    unhcr_displacement (added 2026-08-24 -- see CLAUDE.md Sec 2; national/
    annual, forward-filled monthly per its own June-of-Y+1 publication-lag
    rule, same broadcast mechanics as the other 5 despite the different
    native frequency).
  - static by zone: livelihood_zones_admin2.csv, one row per zone_code,
    no time dimension -- same value every month.
  - static by (zone, calendar-month-of-year): crop_calendar_admin2_month.csv,
    one row per (zone_code, 1-12), repeats identically every year.

Three different raw month representations across sources (year+month int
columns; a "YYYY-MM-DD" date string; the IPC base's own "YYYY-MM" period
string) are all normalised to internal _year/_mo integer columns before
merging, then dropped from the final output -- the base panel's own
`month` column (kept untouched throughout) is the single surviving time
column, matching every other pipeline's output convention.

No feature engineering happens in this file (every source already did its
own lag/rolling-window engineering upstream) and no NaN is filled --
per-source structural gaps (WFP teff pre-2020, AgSS pastoral zones never
surveyed, etc.) pass through unchanged, exactly as CLAUDE.md Sec 3
requires.
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

FEATURES_DIR = os.path.join(REPO_ROOT, "data", "processed", "features")
BOUNDARIES_DIR = os.path.join(REPO_ROOT, "boundaries")
IPC_BASE_PATH = os.path.join(REPO_ROOT, "data", "interim", "ipc_target", "model_tables", "ipc_monthly_base.csv")

OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "interim", "model_join")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "ethiopia_raw_wide_panel.csv")

# (filename, zone_column_name, month_representation)
#   month_representation: "year_month_int" (separate year/month int cols)
#                          "date_str"       (a "YYYY-MM-DD" month column)
ZONE_MONTHLY_SOURCES = [
    ("chirps_admin2_monthly.csv", "pcode", "year_month_int"),
    ("gleam_ssmi_spei_admin2_monthly.csv", "pcode", "year_month_int"),
    ("ndvi_admin2_monthly.csv", "pcode", "year_month_int"),
    ("ndvi_cropland_admin2_monthly.csv", "pcode", "year_month_int"),
    ("ndvi_rangeland_admin2_monthly.csv", "pcode", "year_month_int"),
    ("glofas_admin2_monthly.csv", "zone_code", "date_str"),
    ("iri_cpc_seasonal_admin2_monthly.csv", "pcode", "year_month_int"),
    ("usgs_rainfall_forecast_admin2_monthly.csv", "zone_code", "year_month_int"),
    ("acled_admin2_monthly.csv", "zone_code", "date_str"),
    ("wfp_prices_admin2_monthly.csv", "zone_code", "date_str"),
    ("agss_yields_admin2_monthly.csv", "zone_code", "year_month_int"),
]

NATIONAL_BROADCAST_SOURCES = [
    "cpi_monthly.csv",
    "exchange_rate_monthly.csv",
    "imf_gdp_monthly.csv",
    "teleconnections_monthly.csv",
    "wvg_monthly.csv",
    "unhcr_displacement_monthly.csv",
]

STATIC_ZONE_FILE = os.path.join(BOUNDARIES_DIR, "livelihood_zones_admin2.csv")
STATIC_ZONE_MONTH_OF_YEAR_FILE = os.path.join(BOUNDARIES_DIR, "crop_calendar_admin2_month.csv")


def load_ipc_base():
    """Loads ipc_monthly_base.csv and reindexes it to a fully gapless
    zone x month grid before anything else touches it.

    ipc_monthly_base.csv has a real, deliberate row gap: build_ipc_features.py's
    MAX_FILL_MONTHS=8 staleness rule DROPS (not NaN's) any row more than 8
    months past the zone's last real FEWS NET assessment -- confirmed
    2026-08-20 to remove exactly 3 calendar months (2025-07/08/09) for
    every one of the 92 zones uniformly (a national FEWS NET assessment
    cycle gap, not a per-zone data issue). That is a correct design choice
    for the TARGET's own monthly expansion.

    It is NOT safe to build on downstream, though: this join's lead-time
    shift (build_ethiopia_model_tables.py) uses groupby(zone_code).shift(N),
    which is POSITIONAL -- correct only when there are no row gaps. Across
    this hole, shift(3) would silently grab the row 3 POSITIONS back
    instead of 3 CALENDAR months back (confirmed empirically: without this
    reindex, lead3's total_rainfall_mm at 2025-10 read 2025-04's value
    instead of 2025-07's). Reindexing to the full grid here restores
    exactly-N-calendar-months-back semantics for every shift downstream --
    the reintroduced rows get NaN target/identity columns (correctly
    reflecting "target dropped for staleness"), but still correctly
    receive real external feature values in the merges below, since e.g.
    CHIRPS/WFP/ACLED genuinely have data for Jul-Sep 2025 even though the
    IPC assessment does not.

    NOTE: pipelines/ipc_target/build_ipc_features.py's OWN make_lead_table()
    has this same positional-shift-across-a-row-gap exposure for its
    ipc_lag1/4/8/mean12 features near this date -- out of scope to patch
    here (that file isn't part of this join), but flagged for awareness."""
    df = pd.read_csv(IPC_BASE_PATH)
    period = pd.PeriodIndex(df["month"], freq="M")
    df["_year"] = period.year
    df["_mo"] = period.month

    full_months = pd.period_range(period.min(), period.max(), freq="M")
    zones = df[["zone_code", "zone_id", "zone_name"]].drop_duplicates()
    full_grid = pd.MultiIndex.from_product(
        [zones["zone_code"], full_months], names=["zone_code", "_period"]
    ).to_frame(index=False)
    full_grid["_year"] = full_grid["_period"].dt.year
    full_grid["_mo"] = full_grid["_period"].dt.month
    full_grid["month"] = full_grid["_period"].astype(str)
    full_grid = full_grid.drop(columns=["_period"]).merge(zones, on="zone_code", how="left")

    n_gap_months = len(full_months) - period.nunique()
    reindexed = full_grid.merge(
        df.drop(columns=["zone_id", "zone_name"]), on=["zone_code", "_year", "_mo", "month"], how="left"
    )
    print(f"  ipc_monthly_base.csv: {len(df)} real rows, reindexed to {len(reindexed)} "
          f"({n_gap_months} calendar month(s) reintroduced as NaN-target placeholder rows per zone)")
    return reindexed


def _normalise_zone_month(df, zone_col, month_kind):
    df = df.copy()
    df["zone_code"] = df[zone_col]
    if month_kind == "year_month_int":
        df["_year"] = df["year"].astype(int)
        df["_mo"] = df["month"].astype(int)
    elif month_kind == "date_str":
        dt = pd.to_datetime(df["month"])
        df["_year"] = dt.dt.year
        df["_mo"] = dt.dt.month
    else:
        raise ValueError(f"unknown month_kind: {month_kind}")
    return df


def _merge_source(out, src, key_cols, feature_cols, label):
    """Left-merges src's feature_cols onto out via key_cols, asserting no
    silent column-name collision (pandas would otherwise append _x/_y
    suffixes, which is exactly the kind of thing that goes unnoticed until
    a downstream model quietly trains on the wrong column)."""
    collisions = (set(out.columns) & set(feature_cols)) - set(key_cols)
    if collisions:
        raise ValueError(f"{label}: column name collision with existing panel: {sorted(collisions)}")

    src = src[key_cols + feature_cols].drop_duplicates(subset=key_cols)
    n_before = len(out)
    out = out.merge(src, on=key_cols, how="left")
    if len(out) != n_before:
        raise ValueError(f"{label}: merge changed row count ({n_before} -> {len(out)}) -- "
                          f"src has duplicate keys after drop_duplicates, investigate")

    if feature_cols:
        n_matched = out[feature_cols[0]].notna().sum()
        print(f"  + {label}: {len(feature_cols)} cols, {n_matched}/{len(out)} rows matched ({n_matched/len(out):.1%})")
    return out


def main():
    out = load_ipc_base()
    print(f"IPC base panel (anchor): {out.shape[0]} rows, {out['zone_code'].nunique()} zones, "
          f"{out['month'].min()} to {out['month'].max()}")

    for filename, zone_col, month_kind in ZONE_MONTHLY_SOURCES:
        src = pd.read_csv(os.path.join(FEATURES_DIR, filename))
        src = _normalise_zone_month(src, zone_col, month_kind)
        key_source_cols = {zone_col, "zone_code", "year", "month"}
        feature_cols = [c for c in src.columns if c not in key_source_cols and c not in ("_year", "_mo")]
        out = _merge_source(out, src, ["zone_code", "_year", "_mo"], feature_cols, filename)

    for filename in NATIONAL_BROADCAST_SOURCES:
        src = pd.read_csv(os.path.join(FEATURES_DIR, filename))
        src["_year"] = src["year"].astype(int)
        src["_mo"] = src["month"].astype(int)
        feature_cols = [c for c in src.columns if c not in ("year", "month", "_year", "_mo")]
        out = _merge_source(out, src, ["_year", "_mo"], feature_cols, f"{filename} (national broadcast)")

    liv = pd.read_csv(STATIC_ZONE_FILE)
    liv_feature_cols = [c for c in liv.columns if c not in ("zone_code", "zone_name", "region")]
    out = _merge_source(out, liv, ["zone_code"], liv_feature_cols, "livelihood_zones_admin2.csv (static, by zone)")

    cc = pd.read_csv(STATIC_ZONE_MONTH_OF_YEAR_FILE).rename(columns={"month": "_mo"})
    cc_feature_cols = [c for c in cc.columns if c not in ("zone_code", "zone_name", "region", "_mo")]
    out = _merge_source(out, cc, ["zone_code", "_mo"], cc_feature_cols,
                         "crop_calendar_admin2_month.csv (static, by zone x calendar month)")

    out = out.drop(columns=["_year", "_mo"])

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWrote {out.shape[0]} rows x {out.shape[1]} cols -> {OUTPUT_PATH}")
    print(f"Zones: {out['zone_code'].nunique()} (expected 92)")
    print(f"Months: {out['month'].min()} to {out['month'].max()}")


if __name__ == "__main__":
    main()
