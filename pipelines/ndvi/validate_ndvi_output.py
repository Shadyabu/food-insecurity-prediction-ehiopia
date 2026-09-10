"""
validate_ndvi_output.py

Validates the three NDVI feature outputs against Busker et al.'s own
released grids (zonally re-aggregated by build_busker_ndvi_reference.py
into pipelines/ndvi/busker_comparison/*.csv -- run that script first).

Since NDVI has no ready-made admin2-level reference table (unlike CHIRPS's
data_prec_Admin.xlsx), this compares the RAW (pre-anomaly) zone-month
value against Busker's raw grid, zonally aggregated with the same
NaN-safe weighting logic -- for every zone and every overlapping month,
not a handful of spot checks, per CLAUDE.md 3.5.

Checks:
1. Overall shape, missing rate, value range for each of the 3 outputs.
2. Zone-month match against the Busker reference for the general,
   cropland, and rangeland variants -- median/mean % difference, %
   of zones within agreement bands.
3. Structural sanity: cropland/rangeland NDVI should be systematically
   different from general NDVI (different land cover -> different
   vegetation signal) -- a near-1.0 correlation between all three would
   indicate the masking step did nothing.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed", "features")
REF_DIR = os.path.join(SCRIPT_DIR, "busker_comparison")

VARIANTS = [
    ("ndvi_admin2_monthly.csv", "ndvi", "busker_ndvi_admin2_monthly.csv", "general"),
    ("ndvi_cropland_admin2_monthly.csv", "ndvi_cropland", "busker_ndvi_cropland_admin2_monthly.csv", "cropland"),
    ("ndvi_rangeland_admin2_monthly.csv", "ndvi_rangeland", "busker_ndvi_rangeland_admin2_monthly.csv", "rangeland"),
]

pd.set_option("display.width", 120)

all_results = {}

for out_filename, value_col, ref_filename, label in VARIANTS:
    print("=" * 70)
    print(f"VALIDATING: {label} ({out_filename})")
    print("=" * 70)

    ours = pd.read_csv(os.path.join(PROCESSED_DIR, out_filename))
    ref = pd.read_csv(os.path.join(REF_DIR, ref_filename))

    print(f"Shape: {ours.shape}")
    print(f"Unique zones: {ours['pcode'].nunique()}")
    print(f"Year range: {ours['year'].min()}-{ours['year'].max()}")
    print(f"Missing rate ({value_col}): {ours[value_col].isna().mean():.2%}")
    print(f"Missing rate ({value_col}_anomaly): {ours[f'{value_col}_anomaly'].isna().mean():.2%}")
    valid_vals = ours[value_col].dropna()
    print(f"Value range: {valid_vals.min():.3f} to {valid_vals.max():.3f}")

    per_zone_missing = ours.groupby("pcode")[value_col].apply(lambda x: x.isna().mean())
    print(f"Zones 100% missing: {(per_zone_missing == 1.0).sum()}")
    print(f"Zones 0% missing: {(per_zone_missing == 0.0).sum()}")
    print(f"Zones partially missing: {((per_zone_missing > 0) & (per_zone_missing < 1)).sum()}")
    print()

    merged = ours.merge(ref, on=["pcode", "year", "month"], how="inner", suffixes=("", "_ref"))
    merged = merged.dropna(subset=[value_col, "ndvi_ref"])
    # Skip near-zero denominators (division blows up % difference meaninglessly)
    merged = merged[merged["ndvi_ref"].abs() > 0.01]
    merged["pct_diff"] = (merged[value_col] - merged["ndvi_ref"]).abs() / merged["ndvi_ref"].abs() * 100

    print(f"Zone-months matched against Busker reference: {merged.shape[0]}")
    if merged.shape[0] == 0:
        print("WARNING: no overlapping zone-months found -- check date ranges and pcode alignment.")
        print()
        continue

    per_zone = merged.groupby("pcode")["pct_diff"].median().sort_values()
    print(f"N zones validated: {per_zone.shape[0]}")
    print(f"Median absolute %% difference (per zone-month, all matched rows): {merged['pct_diff'].median():.2f}%")
    print(f"Mean absolute %% difference (per zone-month, all matched rows): {merged['pct_diff'].mean():.2f}%")
    print(f"Zones within 5%% agreement (per-zone median): {(per_zone <= 5).sum()} / {per_zone.shape[0]}")
    print(f"Zones within 10%% agreement (per-zone median): {(per_zone <= 10).sum()} / {per_zone.shape[0]}")
    print(f"Zones within 20%% agreement (per-zone median): {(per_zone <= 20).sum()} / {per_zone.shape[0]}")
    print(f"Worst-agreeing zones (top 5): {per_zone.tail(5).round(2).to_dict()}")
    print()

    all_results[label] = merged

print("=" * 70)
print("CROSS-VARIANT SANITY CHECK")
print("=" * 70)
general = pd.read_csv(os.path.join(PROCESSED_DIR, "ndvi_admin2_monthly.csv"))
crop = pd.read_csv(os.path.join(PROCESSED_DIR, "ndvi_cropland_admin2_monthly.csv"))
rangeland = pd.read_csv(os.path.join(PROCESSED_DIR, "ndvi_rangeland_admin2_monthly.csv"))

merged_all = (
    general[["pcode", "year", "month", "ndvi"]]
    .merge(crop[["pcode", "year", "month", "ndvi_cropland"]], on=["pcode", "year", "month"])
    .merge(rangeland[["pcode", "year", "month", "ndvi_rangeland"]], on=["pcode", "year", "month"])
    .dropna()
)
corr_crop = merged_all["ndvi"].corr(merged_all["ndvi_cropland"])
corr_range = merged_all["ndvi"].corr(merged_all["ndvi_rangeland"])
corr_crop_range = merged_all["ndvi_cropland"].corr(merged_all["ndvi_rangeland"])
print(f"Correlation general vs cropland: {corr_crop:.3f}")
print(f"Correlation general vs rangeland: {corr_range:.3f}")
print(f"Correlation cropland vs rangeland: {corr_crop_range:.3f}")
if corr_crop > 0.999 or corr_range > 0.999:
    print("WARNING: near-perfect correlation with general NDVI -- masking may not be taking effect.")
else:
    print("Masking is producing a distinct signal from general NDVI, as expected.")
