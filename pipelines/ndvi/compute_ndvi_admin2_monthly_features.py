"""
compute_ndvi_admin2_monthly_features.py

ENGINEER stage. Takes the raw (pre-anomaly) admin2 monthly NDVI table
produced by fetch_ndvi_admin2.py and computes the actual feature: a
monthly anomaly relative to a fixed 2000-2021 climatology, fit separately
per zone and per calendar month -- matching Busker et al.'s stated method
("NDVI values were expressed as anomalies per month using 2000-2021 as
reference period, and averaged over the administrative units").

Kept separate from the fetch script (same rationale as CHIRPS's own
compute_..._monthly_features.py) so the anomaly logic can be re-run
without re-doing the network-heavy acquisition step.

Outputs three files at data/processed/features/, same shape as every
other source (pcode, year, month, <feature columns>):
  ndvi_admin2_monthly.csv
  ndvi_cropland_admin2_monthly.csv
  ndvi_rangeland_admin2_monthly.csv
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

INTERIM_PATH = os.path.join(REPO_ROOT, "data", "interim", "ndvi", "ndvi_admin2_monthly_raw.csv")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed", "features")

ZONE_FIELD = "pcode"
FEATURE_START_YEAR = 2009
FEATURE_END_YEAR = 2026
BASELINE_START_YEAR = 2000
BASELINE_END_YEAR = 2021

LAGS = [1, 3, 6, 12]  # same lag set as every other price/index column (WFP prices, CPI, exchange rate)

VARIANTS = [
    ("ndvi_raw", "ndvi", "ndvi_admin2_monthly.csv"),
    ("ndvi_cropland_raw", "ndvi_cropland", "ndvi_cropland_admin2_monthly.csv"),
    ("ndvi_rangeland_raw", "ndvi_rangeland", "ndvi_rangeland_admin2_monthly.csv"),
]

df = pd.read_csv(INTERIM_PATH)
print(f"Loaded {df.shape[0]} raw zone-month rows, "
      f"{df[ZONE_FIELD].nunique()} zones, {df['year'].min()}-{df['year'].max()}.")


def compute_anomaly(df, raw_col):
    """Anomaly = value - climatological mean for that zone + calendar
    month, fit over BASELINE_START_YEAR-BASELINE_END_YEAR only (matches
    Busker et al.'s stated 2000-2021 reference period; this baseline
    window overlaps feature years 2009-2021, same convention already used
    for this repo's CHIRPS SPI baseline, which also fits against the full
    available record rather than excluding train/test years)."""
    baseline_mask = (df["year"] >= BASELINE_START_YEAR) & (df["year"] <= BASELINE_END_YEAR)
    climatology = (
        df[baseline_mask]
        .groupby([ZONE_FIELD, "month"])[raw_col]
        .mean()
        .rename("climatology")
    )
    merged = df.merge(climatology, on=[ZONE_FIELD, "month"], how="left")
    return merged[raw_col] - merged["climatology"]


for raw_col, out_prefix, out_filename in VARIANTS:
    df[f"{out_prefix}_anomaly"] = compute_anomaly(df, raw_col)

# ----------------------------------------------------------------------
# Lag features (1/3/6/12 months), on both the raw value and the anomaly --
# same lag set and (raw + derived-column) pattern as WFP prices. Computed
# on the full df (back to BASELINE_START_YEAR's raw history, 1981) before
# slicing to FEATURE_START_YEAR below, and sorted first, so feature years
# from 2009 onward get real lagged values instead of spurious NaN at the
# start of the published window. The interim raw table is a complete
# zone x month rectangle with no gaps (verified: every zone has the same
# row count), so shift(n) per zone already equals "n calendar months
# ago" -- no additional scaffold needed. Backward-looking only (positive
# shift pulls from strictly earlier rows within the same zone group) --
# no data leakage.
# ----------------------------------------------------------------------
df = df.sort_values([ZONE_FIELD, "year", "month"]).reset_index(drop=True)
for raw_col, out_prefix, out_filename in VARIANTS:
    anomaly_col = f"{out_prefix}_anomaly"
    for lag in LAGS:
        df[f"{raw_col}_lag{lag}"] = df.groupby(ZONE_FIELD)[raw_col].shift(lag)
        df[f"{anomaly_col}_lag{lag}"] = df.groupby(ZONE_FIELD)[anomaly_col].shift(lag)

final_cols = [ZONE_FIELD, "year", "month"]
for raw_col, out_prefix, out_filename in VARIANTS:
    anomaly_col = f"{out_prefix}_anomaly"
    lag_cols = [f"{raw_col}_lag{lag}" for lag in LAGS] + [f"{anomaly_col}_lag{lag}" for lag in LAGS]
    variant_df = df[final_cols + [raw_col, anomaly_col] + lag_cols].copy()
    variant_df = variant_df.rename(columns={raw_col: out_prefix})
    variant_df = variant_df.rename(columns={f"{raw_col}_lag{lag}": f"{out_prefix}_lag{lag}" for lag in LAGS})
    variant_df = variant_df[
        (variant_df["year"] >= FEATURE_START_YEAR) & (variant_df["year"] <= FEATURE_END_YEAR)
    ].sort_values([ZONE_FIELD, "year", "month"]).reset_index(drop=True)

    out_path = os.path.join(PROCESSED_DIR, out_filename)
    variant_df.to_csv(out_path, index=False)

    print(f"\n{out_filename}: {variant_df.shape[0]} rows, "
          f"{variant_df[ZONE_FIELD].nunique()} zones, "
          f"{FEATURE_START_YEAR}-{FEATURE_END_YEAR}")
    print(f"  {out_prefix} NaN rate: {variant_df[out_prefix].isna().mean():.2%}")
    print(f"  {anomaly_col} NaN rate: {variant_df[anomaly_col].isna().mean():.2%}")
    print(f"  {anomaly_col} range: {variant_df[anomaly_col].min():.3f} to {variant_df[anomaly_col].max():.3f}")
    print(f"  {out_prefix}_lag1 NaN rate: {variant_df[f'{out_prefix}_lag1'].isna().mean():.2%}")
    print(f"  {anomaly_col}_lag12 NaN rate: {variant_df[f'{anomaly_col}_lag12'].isna().mean():.2%}")
