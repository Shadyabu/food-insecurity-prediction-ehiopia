"""
validate_gleam_ssmi_spei_output.py

Validates our SSMI/SPEI output against Busker et al.'s own published
admin-level indices (pulled from their Zenodo replication package into
busker_comparison/ -- same role data_prec_Admin.xlsx / spi_*.xlsx play for
CHIRPS's validation).

SSMI/SPEI are standardized (z-score-like, mean ~0, SD ~1) rather than a
ratio quantity like rainfall mm, so "% difference" isn't a meaningful
summary here (undefined/unstable near zero). Instead this reports absolute
difference in standardized units, Pearson correlation, and % of
zone-months within absolute-difference bands -- the same "distribution,
not a handful of examples" principle from CLAUDE.md Sec 3.5, adapted to
the variable's actual scale.

Overlap window: our output covers 2009-2025 (FEATURE_START_YEAR-
FEATURE_END_YEAR); Busker's own indices cover 1982-2022. Comparison uses
the intersection, 2009-2022.

Also runs four checks that do NOT depend on matching Busker's numbers --
needed because GLEAM's v3.5a-vs-v4.3a version change (see
docs/GLEAM_SSMI_SPEI_Pipeline_Documentation.md Sec 4.1) means the Busker
comparison above can't by itself distinguish "different methodology" from
"wrong/broken data": z-score mechanics, cross-correlation against the
already-validated SPI, known Horn of Africa drought years, and the
pastoral-vs-highland soil moisture gradient. See Sec 4.3 of that doc for
the narrative writeup this reproduces.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/gleam_ssmi_spei/ -> pipelines/ -> repo root

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "gleam_ssmi_spei_admin2_monthly.csv")
ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
BUSKER_DIR = os.path.join(SCRIPT_DIR, "busker_comparison")

WINDOWS = [1, 3, 6, 12, 24]
ZONE_FIELD = "pcode"
ADMIN2_NAME_FIELD = "ADM2_EN"
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

# --------------------------------------------------------------------------
# 1. Load our output + admin2 boundary (for pcode <-> name mapping)
# --------------------------------------------------------------------------

df = pd.read_csv(OUTPUT_PATH)
gdf = gpd.read_file(ADMIN2_SHP_PATH)[[ADMIN2_PCODE_FIELD, ADMIN2_NAME_FIELD]]
pcode_to_name = dict(zip(gdf[ADMIN2_PCODE_FIELD], gdf[ADMIN2_NAME_FIELD]))

print("=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Unique zones: {df[ZONE_FIELD].nunique()}")
print(f"Year range: {df['year'].min()}-{df['year'].max()}")
for window in WINDOWS:
    print(f"  ssmi_{window} missing rate: {df[f'ssmi_{window}'].isna().mean():.2%}, "
          f"spei_{window} missing rate: {df[f'spei_{window}'].isna().mean():.2%}")
print()

# --------------------------------------------------------------------------
# 2. Compare against Busker et al.'s own ssmi_*/spei_*.xlsx
# --------------------------------------------------------------------------


def load_busker_index(path):
    wide = pd.read_excel(path)
    time_col = wide.columns[0]
    wide = wide.rename(columns={time_col: "time"})
    wide["time"] = pd.to_datetime(wide["time"])
    long = wide.melt(id_vars="time", var_name="zone_name", value_name="busker_value")
    long["year"] = long["time"].dt.year
    long["month"] = long["time"].dt.month
    return long[["zone_name", "year", "month", "busker_value"]]


def compare_index(our_col, busker_path, index_label):
    busker_long = load_busker_index(busker_path)

    ours = df[[ZONE_FIELD, "year", "month", our_col]].copy()
    ours["zone_name"] = ours[ZONE_FIELD].map(pcode_to_name)
    ours = ours.dropna(subset=["zone_name"])

    merged = ours.merge(busker_long, on=["zone_name", "year", "month"], how="inner")
    merged = merged.dropna(subset=[our_col, "busker_value"])

    if merged.empty:
        print(f"{index_label}: no overlapping rows found.")
        return None

    merged["abs_diff"] = (merged[our_col] - merged["busker_value"]).abs()
    n_zones = merged["zone_name"].nunique()

    per_zone = merged.groupby("zone_name")["abs_diff"].mean().sort_values()
    corr = merged[[our_col, "busker_value"]].corr().iloc[0, 1]

    print(f"--- {index_label} ---")
    print(f"  N zone-months compared: {len(merged)} ({n_zones} zones, "
          f"{merged['year'].min()}-{merged['year'].max()})")
    print(f"  Pearson correlation: {corr:.3f}")
    print(f"  Median |diff| (zone-month): {merged['abs_diff'].median():.3f} SD-units")
    print(f"  Mean |diff| (zone-month): {merged['abs_diff'].mean():.3f} SD-units")
    print(f"  Zones with mean |diff| <= 0.25 SD: {(per_zone <= 0.25).sum()} / {n_zones}")
    print(f"  Zones with mean |diff| <= 0.5 SD: {(per_zone <= 0.5).sum()} / {n_zones}")
    print(f"  Zones with mean |diff| <= 1.0 SD: {(per_zone <= 1.0).sum()} / {n_zones}")
    print(f"  Worst-agreeing zones (top 5 by mean |diff|): {per_zone.tail(5).index.tolist()}")
    print()

    return {
        "index": index_label,
        "n_compared": len(merged),
        "n_zones": n_zones,
        "correlation": corr,
        "median_abs_diff": merged["abs_diff"].median(),
        "mean_abs_diff": merged["abs_diff"].mean(),
        "pct_within_0.5sd": (per_zone <= 0.5).mean() * 100,
    }


print("=" * 60)
print("SSMI / SPEI vs. Busker et al. (2024) -- per accumulation window")
print("=" * 60)

summary_rows = []
for window in WINDOWS:
    result = compare_index(f"ssmi_{window}", os.path.join(BUSKER_DIR, f"ssmi_{window}.xlsx"), f"SSMI-{window}")
    if result:
        summary_rows.append(result)

for window in WINDOWS:
    result = compare_index(f"spei_{window}", os.path.join(BUSKER_DIR, f"spei_{window}.xlsx"), f"SPEI-{window}")
    if result:
        summary_rows.append(result)

print("=" * 60)
print("SUMMARY ACROSS ALL WINDOWS")
print("=" * 60)
summary_df = pd.DataFrame(summary_rows)
pd.set_option("display.width", 120)
print(summary_df.to_string(index=False))
print()

# --------------------------------------------------------------------------
# 3. Independent sanity checks -- don't depend on matching Busker's numbers
#    (see docs/GLEAM_SSMI_SPEI_Pipeline_Documentation.md Sec 4.3)
# --------------------------------------------------------------------------

CHIRPS_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "chirps_admin2_monthly.csv")
GLEAM_RAW_PATH = os.path.join(REPO_ROOT, "data", "interim", "gleam_ssmi_spei", "gleam_admin2_monthly_raw.csv")

SOMALI_PASTORAL_ZONES = ["Jarar", "Nogob", "Shabelle", "Fafan", "Erer", "Doolo", "Afder", "Liben", "Korahe"]
HIGHLAND_CROP_ZONES = ["Bale", "Arsi", "West Shewa", "East Shewa", "Jimma", "Sidama", "Gedeo", "Gurage"]

print("=" * 60)
print("INDEPENDENT SANITY CHECKS (do not rely on matching Busker's numbers)")
print("=" * 60)

# 3.1 Z-score mechanics
print("--- 3.1 Z-score sanity (std should be ~1; mean may drift from 0 for a ---")
print("---     non-random feature-year subset of the full fit baseline)    ---")
for col in ["ssmi_1", "ssmi_12", "spei_1", "spei_12"]:
    print(f"  {col}: mean={df[col].mean():.3f}, std={df[col].std():.3f}, "
          f"min={df[col].min():.2f}, max={df[col].max():.2f}")
print()

# 3.2 Cross-correlation against the already-validated SPI
chirps = pd.read_csv(CHIRPS_PATH)
merged_spi = df.merge(chirps[["pcode", "year", "month", "spi_1", "spi_3", "spi_12"]], on=["pcode", "year", "month"])
print("--- 3.2 Correlation vs. already-validated SPI (same drought family; ---")
print("---     moderate positive correlation expected, not near 1.0)      ---")
for w in ["1", "3", "12"]:
    c_ssmi = merged_spi[f"ssmi_{w}"].corr(merged_spi[f"spi_{w}"])
    c_spei = merged_spi[f"spei_{w}"].corr(merged_spi[f"spi_{w}"])
    print(f"  window {w}m: corr(SSMI,SPI)={c_ssmi:.3f}  corr(SPEI,SPI)={c_spei:.3f}")
print()

# 3.3 Known Horn of Africa drought years (2009; 2015-17 post-El Nino; 2020-2023,
# 5 consecutive failed rainy seasons, worst in ~40 years, peaking 2022)
print("--- 3.3 Known drought years: mean SSMI-12 / SPEI-12 across all zones, by year ---")
yearly = df.groupby("year")[["ssmi_12", "spei_12"]].mean().round(3)
print(yearly.to_string())
driest_ssmi = yearly["ssmi_12"].sort_values().head(4)
driest_spei = yearly["spei_12"].sort_values().head(2)
print(f"  4 driest years by SSMI-12: {driest_ssmi.index.tolist()} (expect 2009/2017/2020-2023 to appear)")
print(f"  2 driest years by SPEI-12: {driest_spei.index.tolist()} (expect 2015 or 2020-2023 to appear)")
print()

# 3.4 Spatial gradient: pastoral Somali-region zones should be much drier
# than the highland crop-farming belt -- a well-known Ethiopia livelihood fact
gleam_raw = pd.read_csv(GLEAM_RAW_PATH)
gleam_raw["zone_name"] = gleam_raw["pcode"].map(pcode_to_name)
pastoral_sm = gleam_raw[gleam_raw["zone_name"].isin(SOMALI_PASTORAL_ZONES)]["sm_root"].mean()
highland_sm = gleam_raw[gleam_raw["zone_name"].isin(HIGHLAND_CROP_ZONES)]["sm_root"].mean()
print("--- 3.4 Spatial gradient: pastoral (arid) vs. highland (crop-farming) soil moisture ---")
print(f"  Somali-region pastoral zones mean sm_root: {pastoral_sm:.3f} m3/m3")
print(f"  Highland crop-farming zones mean sm_root:  {highland_sm:.3f} m3/m3")
print(f"  Ratio (highland / pastoral): {highland_sm / pastoral_sm:.2f}x (expect >> 1.0)")
