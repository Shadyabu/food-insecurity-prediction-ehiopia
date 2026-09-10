"""
validate_acled_output.py

Checks, per CLAUDE.md Sec 3.5:
1. Overall shape and the zero-vs-scaffold logic actually held (no NaNs,
   92 zones, full observed month range).
2. Zone-month comparison against Busker et al.'s own staged ACLED extract
   (busker_comparison/acled_1900-01-01-2023-04-17-Eastern_Africa-Ethiopia-Kenya-Somalia.csv).

Unlike locust/WFP (which validate against Busker's own *already-aggregated*
feature), Busker's ACLED reference is raw point data, not a pre-built
zone-month table. So this script independently runs the exact same
point-in-polygon join used in compute_acled_admin2_monthly_features.py on
Busker's raw points, and compares the resulting zone-months to our own
over the window both sources actually cover (2000-01 to 2023-04, Busker's
extract's cutoff). This isolates genuine data-vintage drift (ACLED's own
retroactive revisions between Busker's 2023 download and this project's
2026 download -- see docs/data_sources_master_table.md) from any join-logic
bug, since both sides go through identical join code.
"""

import os

import geopandas as gpd
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "acled_admin2_monthly.csv")
BUSKER_RAW_PATH = os.path.join(
    SCRIPT_DIR, "busker_comparison",
    "acled_1900-01-01-2023-04-17-Eastern_Africa-Ethiopia-Kenya-Somalia.csv",
)
ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

OVERLAP_START = "2000-01"
OVERLAP_END = "2023-04"  # Busker's extract's actual cutoff (2023-04-17)

import sys
sys.path.insert(0, SCRIPT_DIR)
from compute_acled_admin2_monthly_features import point_in_polygon_join  # noqa: E402

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

assert df["acled_event_count"].notna().all(), "BUG: event_count must never be NaN (confirmed zero, not gap)"
assert df["acled_fatalities"].notna().all(), "BUG: fatalities must never be NaN (confirmed zero, not gap)"
assert (df["acled_event_count"] >= 0).all(), "BUG: negative event count"
assert (df["acled_fatalities"] >= 0).all(), "BUG: negative fatalities"
n_expected_rows = 92 * len(pd.period_range(df["month"].min(), df["month"].max(), freq="M"))
assert len(df) == n_expected_rows, f"BUG: expected {n_expected_rows} zone-month rows, got {len(df)}"
print("PASS: no NaNs, no negative counts, full zone x month scaffold present.")
print()

# --------------------------------------------------------------------------
# 2. Independent join of Busker's own raw points, same window
# --------------------------------------------------------------------------

gdf_admin2 = gpd.read_file(ADMIN2_SHP_PATH)

busker = pd.read_csv(BUSKER_RAW_PATH, encoding="utf-8-sig")
busker = busker[busker["country"] == "Ethiopia"].copy()
busker["event_date"] = pd.to_datetime(busker["event_date"])
busker["month"] = busker["event_date"].dt.to_period("M")

print("Point-in-polygon join: Busker's raw Ethiopia ACLED points")
busker_joined, busker_dropped = point_in_polygon_join(busker, gdf_admin2)

busker_agg = busker_joined.groupby(["zone_code", "month"]).agg(
    acled_event_count=("event_id_cnty", "size"),
    acled_fatalities=("fatalities", "sum"),
).reset_index()
busker_agg["month"] = busker_agg["month"].dt.to_timestamp()

ours = df[(df["month"] >= OVERLAP_START) & (df["month"] <= OVERLAP_END)]
# Busker's raw file goes back to 1997, before our file's 2000-01-01 start --
# without this filter those pre-2000 rows leak into the comparison as
# spurious "ours=0 vs busker=high" mismatches (they were never in-scope,
# not a real discrepancy).
busker_agg = busker_agg[(busker_agg["month"] >= OVERLAP_START) & (busker_agg["month"] <= OVERLAP_END)]

merged = ours.merge(
    busker_agg, on=["zone_code", "month"], how="outer", suffixes=("_ours", "_busker"), indicator=True
).fillna({"acled_event_count_ours": 0, "acled_fatalities_ours": 0,
          "acled_event_count_busker": 0, "acled_fatalities_busker": 0})

print()
print("=" * 60)
print(f"COMPARISON: overlap window {OVERLAP_START} to {OVERLAP_END}")
print("=" * 60)
print(f"Zone-months only in our (new) file: {(merged['_merge'] == 'left_only').sum()}")
print(f"Zone-months only in Busker's file: {(merged['_merge'] == 'right_only').sum()}")
print(f"Zone-months in both: {(merged['_merge'] == 'both').sum()}")
print()

for col in ["acled_event_count", "acled_fatalities"]:
    ours_col = f"{col}_ours"
    busker_col = f"{col}_busker"
    both = merged[merged[busker_col] > 0].copy()
    both["abs_diff"] = (both[ours_col] - both[busker_col]).abs()
    both["pct_diff"] = both["abs_diff"] / both[busker_col] * 100

    print(f"--- {col} (zone-months where Busker's value > 0, n={len(both)}) ---")
    print(f"Median absolute % difference: {both['pct_diff'].median():.1f}%")
    print(f"Mean absolute % difference: {both['pct_diff'].mean():.1f}%")
    print(f"Zone-months exact match: {(both['abs_diff'] == 0).sum()} / {len(both)} "
          f"({(both['abs_diff'] == 0).mean():.1%})")
    print(f"Zone-months within 20% agreement: {(both['pct_diff'] <= 20).sum()} / {len(both)} "
          f"({(both['pct_diff'] <= 20).mean():.1%})")
    print()

print(f"Totals over overlap window -- ours: {ours['acled_event_count'].sum()} events / "
      f"{ours['acled_fatalities'].sum()} fatalities; "
      f"Busker: {busker_agg[(busker_agg['month'] >= OVERLAP_START) & (busker_agg['month'] <= OVERLAP_END)]['acled_event_count'].sum()} events / "
      f"{busker_agg[(busker_agg['month'] >= OVERLAP_START) & (busker_agg['month'] <= OVERLAP_END)]['acled_fatalities'].sum()} fatalities")
