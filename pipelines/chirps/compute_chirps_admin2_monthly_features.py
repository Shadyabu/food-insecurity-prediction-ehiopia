"""
compute_chirps_admin2_monthly_features.py

Takes the admin2 daily CSV (chirps_admin2_daily_combined.csv) -- including
any fixes already applied by fix_small_zones_fractional.py -- and computes
the final monthly features: total rainfall, wet days, max dry-spell length,
and SPI at 5 accumulation windows (1/3/6/12/24 months).

This is deliberately SEPARATE from fetch_chirps_admin2.py's own built-in
monthly-feature step, specifically so it can be re-run on corrected daily
data without re-downloading or re-extracting anything -- rerunning
fetch_chirps_admin2.py itself would redo the raw extraction from scratch
and silently overwrite any small-zone fixes already applied.

Run this any time the daily CSV changes (e.g. after fix_small_zones_fractional.py)
to regenerate the final monthly feature table.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/chirps/ -> pipelines/ -> repo root

DAILY_CSV_PATH = os.path.join(REPO_ROOT, "data", "interim", "chirps", "chirps_admin2_daily_combined.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "chirps_admin2_monthly.csv")

ZONE_FIELD = "pcode"
FEATURE_START_YEAR = 2009
FEATURE_END_YEAR = 2026

WET_DAY_THRESHOLD_MM = 1.0
DRY_SPELL_THRESHOLD_MM = 1.0
SPI_MIN_YEARS = 5
SPI_WINDOWS = [1, 3, 6, 12, 24]

# --------------------------------------------------------------------------
# 1. Load daily data
# --------------------------------------------------------------------------

raw_df = pd.read_csv(DAILY_CSV_PATH)
raw_df["date"] = pd.to_datetime(raw_df["date"])
raw_df = raw_df.sort_values([ZONE_FIELD, "date"]).reset_index(drop=True)
raw_df["year"] = raw_df["date"].dt.year
raw_df["month"] = raw_df["date"].dt.month

print(f"Loaded {raw_df.shape[0]} daily rows, "
      f"{raw_df['date'].min().date()} to {raw_df['date'].max().date()}, "
      f"{raw_df[ZONE_FIELD].nunique()} zones.")

# --------------------------------------------------------------------------
# 2. Total monthly rainfall
# --------------------------------------------------------------------------

monthly_total = raw_df.groupby([ZONE_FIELD, "year", "month"])["precipitation_mm"].sum().rename("total_rainfall_mm")

# --------------------------------------------------------------------------
# 3. Wet days
# --------------------------------------------------------------------------

raw_df["is_wet"] = raw_df["precipitation_mm"] > WET_DAY_THRESHOLD_MM
monthly_wetdays = raw_df.groupby([ZONE_FIELD, "year", "month"])["is_wet"].sum().rename("wet_days")

# --------------------------------------------------------------------------
# 4. Max dry-spell length (backward-looking, leakage-safe)
# --------------------------------------------------------------------------


def compute_running_dry_spell(sub, zone_value):
    sub = sub.sort_values("date").reset_index(drop=True)
    is_dry = (sub["precipitation_mm"] <= DRY_SPELL_THRESHOLD_MM).to_numpy()
    wet_break = (~is_dry).cumsum()
    running_dry_days = pd.Series(is_dry).groupby(wet_break).cumcount().to_numpy() + 1
    running_dry_days = np.where(is_dry, running_dry_days, 0)
    sub["running_dry_days"] = running_dry_days
    sub[ZONE_FIELD] = zone_value
    return sub


print("Computing dry-spell runs ...")
dry_spell_pieces = []
for zone_value, sub in raw_df.groupby(ZONE_FIELD):
    dry_spell_pieces.append(compute_running_dry_spell(sub, zone_value))
raw_df_with_spells = pd.concat(dry_spell_pieces, ignore_index=True)

monthly_max_dryspell = (
    raw_df_with_spells.groupby([ZONE_FIELD, "year", "month"])["running_dry_days"].max().rename("max_dry_spell_length")
)

# --------------------------------------------------------------------------
# 5. Assemble monthly table (full range, needed for SPI baseline)
# --------------------------------------------------------------------------

monthly_features_full = pd.concat([monthly_total, monthly_wetdays, monthly_max_dryspell], axis=1).reset_index()
monthly_features_full = monthly_features_full.sort_values([ZONE_FIELD, "year", "month"]).reset_index(drop=True)

# --------------------------------------------------------------------------
# 6. SPI at 5 accumulation windows
# --------------------------------------------------------------------------


def compute_spi_for_zone(zone_df, value_col, month_col="month"):
    zone_df = zone_df.copy()
    spi_col = f"spi_from_{value_col}"
    zone_df[spi_col] = np.nan
    for m in range(1, 13):
        mask = zone_df[month_col] == m
        vals = zone_df.loc[mask, value_col].to_numpy()
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < SPI_MIN_YEARS:
            continue
        valid_vals = vals[valid_mask]
        zero_frac = np.mean(valid_vals == 0)
        nonzero_vals = valid_vals[valid_vals > 0]
        if len(nonzero_vals) < 3:
            continue
        shape, loc, scale = stats.gamma.fit(nonzero_vals, floc=0)
        cdf = np.full_like(vals, np.nan, dtype=float)
        cdf[valid_mask] = np.where(
            valid_vals == 0, zero_frac,
            zero_frac + (1 - zero_frac) * stats.gamma.cdf(valid_vals, shape, loc=loc, scale=scale),
        )
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spi_vals = np.full_like(vals, np.nan, dtype=float)
        spi_vals[valid_mask] = stats.norm.ppf(cdf[valid_mask])
        zone_df.loc[mask, spi_col] = spi_vals
    return zone_df


print(f"Computing rolling accumulation windows: {SPI_WINDOWS} ...")
for window in SPI_WINDOWS:
    col_name = f"rainfall_sum_{window}m"
    monthly_features_full[col_name] = (
        monthly_features_full.groupby(ZONE_FIELD)["total_rainfall_mm"].transform(
            lambda s: s.rolling(window, min_periods=window).sum()
        )
    )

print("Fitting SPI per zone, per accumulation window ...")
for window in SPI_WINDOWS:
    value_col = f"rainfall_sum_{window}m"
    spi_pieces = []
    for zone_value, zone_df in monthly_features_full.groupby(ZONE_FIELD):
        spi_pieces.append(compute_spi_for_zone(zone_df, value_col=value_col))
    monthly_features_full = pd.concat(spi_pieces, ignore_index=True)
    monthly_features_full = monthly_features_full.rename(columns={f"spi_from_{value_col}": f"spi_{window}"})

# --------------------------------------------------------------------------
# 7. Restrict to feature years and save
# --------------------------------------------------------------------------

final_df = monthly_features_full[
    (monthly_features_full["year"] >= FEATURE_START_YEAR) & (monthly_features_full["year"] <= FEATURE_END_YEAR)
].copy()
final_df = final_df.sort_values([ZONE_FIELD, "year", "month"]).reset_index(drop=True)

final_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinal monthly feature table: {final_df.shape[0]} rows ({FEATURE_START_YEAR}-{FEATURE_END_YEAR})")
print(f"Saved to {OUTPUT_PATH}")
print(f"Columns: {list(final_df.columns)}")
for window in SPI_WINDOWS:
    col = f"spi_{window}"
    print(f"  {col} NaN rate: {final_df[col].isna().mean():.2%}")
