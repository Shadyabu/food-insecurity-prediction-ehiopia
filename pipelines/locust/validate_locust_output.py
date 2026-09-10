"""
validate_locust_output.py

Checks, per CLAUDE.md Sec 3.5:
1. Overall shape, the three-way locust_data_source split, and the explicit
   NaN-for-gap / 0-for-confirmed-no-report distinction actually held.
2. Zone-month spot-check against Busker et al.'s OWN aggregated locust
   feature (desert_locust_dataset.xlsx, raw AREAHA sums, name-matched to
   admin2 -- same name-overlap approach validate_chirps_admin2_output.py
   uses, since Busker's units are a different, older boundary vintage).
   Busker's file only covers their archive period (1987-2021), so this
   only validates the busker_archive_1986_2021 rows -- RAMSES-era rows
   have no independent numeric reference (see docs note on this).
"""

import os

import geopandas as gpd
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "locust_admin2_monthly.csv")
BUSKER_REF_PATH = os.path.join(SCRIPT_DIR, "busker_comparison", "desert_locust_dataset.xlsx")
ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")

EQUAL_AREA_CRS = "ESRI:102022"

# --------------------------------------------------------------------------
# 1. Overall summary
# --------------------------------------------------------------------------

df = pd.read_csv(OUTPUT_PATH, parse_dates=["month"])

print("=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Zones: {df['zone_code'].nunique()} (expected 92)")
print(f"Date range: {df['month'].min().date()} to {df['month'].max().date()}")
print()
print("Row counts by locust_data_source:")
print(df["locust_data_source"].value_counts())
print()

gap = df[df["locust_data_source"] == "no_source_2022_gap"]
covered = df[df["locust_data_source"] != "no_source_2022_gap"]

assert gap["locust_pct_area_affected"].isna().all(), \
    "BUG: gap-period rows must be NaN, not an implied zero"
assert gap["locust_sightings_surveyed"].isna().all(), \
    "BUG: gap-period sightings count must be NaN, not an implied zero"
assert covered["locust_pct_area_affected"].notna().all(), \
    "BUG: covered-period rows must never be NaN -- explicit zero required (Sec 3.3/Sec 5)"
assert covered["locust_sightings_surveyed"].notna().all(), \
    "BUG: covered-period sightings count must never be NaN"
print("PASS: gap months are NaN, covered months are never NaN (explicit zero enforced).")
print()

print(f"Value range (covered rows): "
      f"{covered['locust_pct_area_affected'].min():.4f} to {covered['locust_pct_area_affected'].max():.4f}")
print(f"Zone-months with any swarm activity: "
      f"{(covered['locust_pct_area_affected'] > 0).sum()} / {len(covered)} "
      f"({(covered['locust_pct_area_affected'] > 0).mean():.2%})")
print()

# --------------------------------------------------------------------------
# 2. Spot-check against Busker et al.'s own aggregated feature
# --------------------------------------------------------------------------

gdf_admin2 = gpd.read_file(ADMIN2_SHP_PATH)
gdf_proj = gdf_admin2.to_crs(EQUAL_AREA_CRS)
gdf_admin2["area_km2"] = gdf_proj.geometry.area / 1e6

busker = pd.read_excel(BUSKER_REF_PATH)
busker["month"] = pd.to_datetime(busker["STARTDATE"]).dt.to_period("M").dt.to_timestamp()
busker_area_km2 = busker["AREAHA"] / 100.0
busker = busker.assign(area_km2=busker_area_km2)

busker_units = set(busker["unit"].unique())
admin2_names = set(gdf_admin2["ADM2_EN"])
matched_names = sorted(busker_units & admin2_names)

print("=" * 60)
print(f"SPOT-CHECK: {len(matched_names)} name matches "
      f"(admin2 vs. Busker's {len(busker_units)} Horn-of-Africa units)")
print("=" * 60)

name_to_pcode = gdf_admin2.set_index("ADM2_EN")["ADM2_PCODE"].to_dict()
name_to_area = gdf_admin2.set_index("ADM2_EN")["area_km2"].to_dict()

results = []
for name in matched_names:
    pcode = name_to_pcode[name]
    zone_busker = busker[busker["unit"] == name].groupby("month")["area_km2"].sum()

    our_zone = df[(df["zone_code"] == pcode) & (df["locust_data_source"] == "busker_archive_1986_2021")]
    our_area_km2 = (our_zone.set_index("month")["locust_pct_area_affected"] / 100.0) * name_to_area[name]

    for month, busker_val in zone_busker.items():
        if month not in our_area_km2.index or busker_val <= 0:
            continue
        our_val = our_area_km2.loc[month]
        pct_diff = abs(our_val - busker_val) / busker_val * 100
        results.append({"name": name, "pcode": pcode, "month": month,
                         "ours_km2": our_val, "busker_km2": busker_val, "pct_diff": pct_diff})

results_df = pd.DataFrame(results).sort_values("pct_diff")
pd.set_option("display.width", 120)

print(f"\nN zone-months compared (busker_val > 0): {len(results_df)}")
if len(results_df) > 0:
    print(f"Median absolute % difference: {results_df['pct_diff'].median():.1f}%")
    print(f"Mean absolute % difference: {results_df['pct_diff'].mean():.1f}%")
    print(f"Zone-months within 5% agreement: {(results_df['pct_diff'] <= 5).sum()} / {len(results_df)}")
    print(f"Zone-months within 20% agreement: {(results_df['pct_diff'] <= 20).sum()} / {len(results_df)}")
    print(f"Zone-months within 50% agreement: {(results_df['pct_diff'] <= 50).sum()} / {len(results_df)}")
    print(f"\nWorst-agreeing (top 10):")
    print(results_df.tail(10).to_string(index=False))
