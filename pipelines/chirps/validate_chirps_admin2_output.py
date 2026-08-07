"""
validate_chirps_admin2_output.py

Same validation approach used for the fsc_admin pipeline, adapted for the
admin2 output structure (joined on ADM2_EN name directly -- no vintage
matching needed, since there's only one boundary set).

Checks:
1. Overall shape, missing rate, value range
2. Precipitation spot-check against Busker et al.'s published July 2009
   values, across all name-matched zones (not just a handful)
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/chirps/ -> pipelines/ -> repo root

DAILY_CSV_PATH = os.path.join(REPO_ROOT, "data", "interim", "chirps", "chirps_admin2_daily_combined.csv")
BUSKER_PREC_PATH = os.path.join(SCRIPT_DIR, "busker_comparison", "data_prec_Admin.xlsx")
ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")

# --------------------------------------------------------------------------
# 1. Overall summary
# --------------------------------------------------------------------------

df = pd.read_csv(DAILY_CSV_PATH)
print("=" * 60)
print("OVERALL SUMMARY")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"Unique zones (pcode): {df['pcode'].nunique()}")
print(f"Date range: {df['date'].min()} to {df['date'].max()}")
print(f"Missing rate: {df['precipitation_mm'].isna().mean():.2%}")
print(f"Value range: {df['precipitation_mm'].min():.2f} to {df['precipitation_mm'].max():.2f}")
print()

per_zone_missing = df.groupby("pcode")["precipitation_mm"].apply(lambda x: x.isna().mean())
print(f"Zones 100% missing: {(per_zone_missing == 1.0).sum()}")
print(f"Zones 0% missing: {(per_zone_missing == 0.0).sum()}")
print(f"Zones partially missing: {((per_zone_missing > 0) & (per_zone_missing < 1)).sum()}")
print()

# --------------------------------------------------------------------------
# 2. Spot-check against Busker et al. -- ALL name matches, not just 5
# --------------------------------------------------------------------------

import geopandas as gpd

gdf_admin2 = gpd.read_file(ADMIN2_SHP_PATH)

busker = pd.read_excel(BUSKER_PREC_PATH)
busker["time"] = pd.to_datetime(busker["time"])
busker_july = busker[(busker["time"].dt.year == 2009) & (busker["time"].dt.month == 7)]

busker_zone_names = set(busker.columns) - {"time"}
admin2_names = set(gdf_admin2["ADM2_EN"])
all_matches = sorted(busker_zone_names & admin2_names)

print("=" * 60)
print(f"SPOT-CHECK: {len(all_matches)} name matches found (admin2 vs. Busker's 214 zones)")
print("=" * 60)
print(f"Matched names: {all_matches}\n")

df["date"] = pd.to_datetime(df["date"])

results = []
for name in all_matches:
    match = gdf_admin2[gdf_admin2["ADM2_EN"] == name]
    pcode = match["ADM2_PCODE"].values[0]

    our_rows = df[df["pcode"] == pcode]
    our_july = our_rows[(our_rows["date"].dt.year == 2009) & (our_rows["date"].dt.month == 7)]
    our_total = our_july["precipitation_mm"].sum()

    busker_val = busker_july[name].values
    busker_total = busker_val[0] if len(busker_val) > 0 else None

    if busker_total is not None and busker_total > 1:  # skip near-zero denominators
        pct_diff = abs(our_total - busker_total) / busker_total * 100
        results.append({"name": name, "pcode": pcode, "ours": our_total,
                         "busker": busker_total, "pct_diff": pct_diff})

results_df = pd.DataFrame(results).sort_values("pct_diff")
pd.set_option("display.width", 120)
print(results_df.to_string(index=False))

print()
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(f"N validated: {len(results_df)}")
print(f"Median absolute % difference: {results_df['pct_diff'].median():.1f}%")
print(f"Mean absolute % difference: {results_df['pct_diff'].mean():.1f}%")
print(f"Zones within 5% agreement: {(results_df['pct_diff'] <= 5).sum()} / {len(results_df)}")
print(f"Zones within 10% agreement: {(results_df['pct_diff'] <= 10).sum()} / {len(results_df)}")
print(f"Zones within 20% agreement: {(results_df['pct_diff'] <= 20).sum()} / {len(results_df)}")
print(f"Worst-agreeing zones (top 5): {results_df.tail(5)['name'].tolist()}")
