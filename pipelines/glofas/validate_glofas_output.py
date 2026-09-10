"""
validate_glofas_output.py

VALIDATE stage. No Busker et al. reference exists for this source (this
pipeline is a pure Ethiopia-specific addition -- see
docs/GloFAS_Pipeline_Documentation.md Sec 1), so validation here is
structural correctness plus a magnitude sanity check against GloFAS's own
published skill claims for Ethiopian basins, per
docs/GloFAS_Pipeline_Documentation.md Sec 8:

  1. Structural checks: 92/92 zones present; glofas_reach_count == 0 rows
     have exactly 0.0 exceedance (not just near-zero); glofas_source_lead
     {1,3} only take the three documented values; reach_count is constant
     per zone (it's a static property, not time-varying).
  2. Magnitude sanity check: zones known to contain the Awash basin (the
     one Ethiopian basin GloFAS's own documentation explicitly claims
     validation for) should show materially higher glofas_reach_count and
     non-trivial glofas_exceed_2yr_lead1 activity than known pastoral
     lowland zones with no significant perennial river.

Has NOT been run yet (docs/GloFAS_Pipeline_Documentation.md Sec 0) --
depends on compute_glofas_admin2_monthly_features.py's output existing.
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "glofas_admin2_monthly.csv")
ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")

VALID_SOURCE_VALUES = {"operational", "reforecast", "no_river_network", "no_reforecast_archive"}

# Zone name substrings known from the flood-vulnerability literature to sit
# within/along the Awash basin (Upper/Middle Awash zones) -- used only as a
# directional sanity check, not a numeric reference (none exists for this
# source). Confirm/adjust against boundaries/eth_admbnda_adm2_csa_bofedb_2021.shp's
# ADM1_EN/ADM2_EN fields before trusting this list.
AWASH_BASIN_ZONE_NAME_HINTS = ["Adama", "Nazret", "Shewa", "Afar"]

# Zones known to be arid pastoral lowland with no significant perennial
# river (used as the "should show ~zero" side of the sanity check). Reuses
# the 7 pastoral zones pipelines/agss_yields/ already confirmed have ZERO
# real yield-survey coverage across the full 1995-2024 window -- the same
# arid/pastoral signal, independently verified there. NOTE (2026-08-16):
# the original list here had "Dollo" (a misspelling of the real ADM2_EN
# name "Doolo") and "Liben", which isn't a real admin2 zone name in
# boundaries/eth_admbnda_adm2_csa_bofedb_2021.shp at all -- between the
# two, only "Afder" ever matched, so this check ran on a 1-zone sample.
# Afder also happens to sit on a major river corridor (glofas_reach_count
# 166, higher than any matched Awash-basin zone), which is presumably why
# that 1-zone sample showed HIGHER exceedance activity than Awash -- the
# opposite of what this check expects -- before this fix.
DRY_LOWLAND_ZONE_NAME_HINTS = ["Doolo", "Korahe", "Fanti-Zone 4", "Daawa", "Nogob", "Erer", "Afder"]


def main():
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(
            f"{OUTPUT_PATH} does not exist -- run fetch_glofas_static.py, "
            "fetch_glofas_medium_range.py, fetch_glofas_seasonal.py, then "
            "compute_glofas_admin2_monthly_features.py first."
        )

    df = pd.read_csv(OUTPUT_PATH, parse_dates=["month"])

    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Zones: {df['zone_code'].nunique()} (expected 92)")
    print(f"Date range: {df['month'].min().date()} to {df['month'].max().date()}")
    print()

    # ------------------------------------------------------------------
    # 1. Structural checks
    # ------------------------------------------------------------------
    assert df["zone_code"].nunique() == 92, \
        f"BUG: expected 92 zones, got {df['zone_code'].nunique()}"

    bad_sources_1 = set(df["glofas_source_lead1"].unique()) - VALID_SOURCE_VALUES
    bad_sources_3 = set(df["glofas_source_lead3"].unique()) - VALID_SOURCE_VALUES
    assert not bad_sources_1, f"BUG: unexpected glofas_source_lead1 values: {bad_sources_1}"
    assert not bad_sources_3, f"BUG: unexpected glofas_source_lead3 values: {bad_sources_3}"
    print("PASS: glofas_source_lead1/lead3 only take documented values.")

    reach_per_zone = df.groupby("zone_code")["glofas_reach_count"].nunique()
    non_constant = reach_per_zone[reach_per_zone > 1]
    assert non_constant.empty, \
        f"BUG: glofas_reach_count varies over time within a zone (should be static): {non_constant.index.tolist()}"
    print("PASS: glofas_reach_count is constant per zone (static property).")

    zero_reach = df[df["glofas_reach_count"] == 0]
    exceed_cols = ["glofas_exceed_2yr_lead1", "glofas_exceed_20yr_lead1",
                   "glofas_exceed_2yr_lead3", "glofas_exceed_20yr_lead3"]
    if not zero_reach.empty:
        non_zero_exceed = zero_reach[(zero_reach[exceed_cols] != 0.0).any(axis=1)]
        assert non_zero_exceed.empty, \
            f"BUG: {len(non_zero_exceed)} zero-reach rows have a non-zero exceedance value:\n{non_zero_exceed}"
        print(f"PASS: all {len(zero_reach)} zero-reach-count rows have exactly 0.0 exceedance "
              f"across all four columns.")
    else:
        print("NOTE: no zero-reach-count zones found -- confirm this is expected "
              "(every admin2 zone has at least one river-network pixel) rather than "
              "a join bug before trusting this as a pass.")

    n_zero_reach_zones = df.loc[df["glofas_reach_count"] == 0, "zone_code"].nunique()
    print(f"\nZones with glofas_reach_count == 0: {n_zero_reach_zones} / 92")
    print()

    # ------------------------------------------------------------------
    # 2. Magnitude sanity check: Awash-basin zones vs. dry lowland zones
    # ------------------------------------------------------------------
    print("=" * 60)
    print("SANITY CHECK: Awash-basin vs. dry-lowland zones")
    print("=" * 60)

    import geopandas as gpd
    gdf_admin2 = gpd.read_file(ADMIN2_SHP_PATH)
    name_to_pcode = gdf_admin2.set_index("ADM2_EN")["ADM2_PCODE"].to_dict()

    def matched_pcodes(hints):
        return [pcode for name, pcode in name_to_pcode.items()
                if any(h.lower() in name.lower() for h in hints)]

    awash_pcodes = matched_pcodes(AWASH_BASIN_ZONE_NAME_HINTS)
    dry_pcodes = matched_pcodes(DRY_LOWLAND_ZONE_NAME_HINTS)
    print(f"Awash-basin candidate zones matched by name: {len(awash_pcodes)}")
    print(f"Dry-lowland candidate zones matched by name: {len(dry_pcodes)}")

    static = df.drop_duplicates("zone_code").set_index("zone_code")
    if awash_pcodes:
        awash_reach = static.loc[static.index.intersection(awash_pcodes), "glofas_reach_count"]
        print(f"\nAwash-candidate glofas_reach_count: mean {awash_reach.mean():.1f}, "
              f"zones with reach_count > 0: {(awash_reach > 0).sum()}/{len(awash_reach)}")
    if dry_pcodes:
        dry_reach = static.loc[static.index.intersection(dry_pcodes), "glofas_reach_count"]
        print(f"Dry-lowland-candidate glofas_reach_count: mean {dry_reach.mean():.1f}, "
              f"zones with reach_count > 0: {(dry_reach > 0).sum()}/{len(dry_reach)}")

    if awash_pcodes:
        awash_rows = df[df["zone_code"].isin(awash_pcodes)]
        print(f"\nAwash-candidate glofas_exceed_2yr_lead1: "
              f"mean {awash_rows['glofas_exceed_2yr_lead1'].mean():.4f}, "
              f"max {awash_rows['glofas_exceed_2yr_lead1'].max():.4f}")
    if dry_pcodes:
        dry_rows = df[df["zone_code"].isin(dry_pcodes)]
        print(f"Dry-lowland-candidate glofas_exceed_2yr_lead1: "
              f"mean {dry_rows['glofas_exceed_2yr_lead1'].mean():.4f}, "
              f"max {dry_rows['glofas_exceed_2yr_lead1'].max():.4f}")
    print(
        "\nThis is a directional sanity check only -- no independent numeric "
        "reference exists for this source (Sec 1 of the pipeline doc). If "
        "Awash-candidate zones do NOT show materially higher reach_count/"
        "exceedance activity than dry-lowland zones, investigate the "
        "zone-name hint lists above and the network-threshold value "
        "(NETWORK_THRESHOLD_KM2 in fetch_glofas_static.py) before trusting "
        "the pipeline's output."
    )


if __name__ == "__main__":
    main()
