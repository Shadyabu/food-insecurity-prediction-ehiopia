"""
validate_wfp_prices_output.py

Checks, per CLAUDE.md Sec 3.5:
1. Overall shape, zone coverage, and per-commodity fill-rate sanity.
2. Zone-month spot-check against Busker et al.'s OWN retail extract
   (busker_comparison/WFP_Ethiopia_FoodPrices_new_tool_retail.csv) --
   a separate historical snapshot pulled from the same underlying WFP
   database, so this is a genuine independent-extraction check even though
   the ultimate source is shared (same pattern as validating locust's
   RAMSES-era rows against Busker's own archive-period feature).

   Busker's file has no lat/lon, only a Market Name string. Its 92 markets
   match the live pull's market names exactly (verified: 92/92 overlap),
   so Busker's rows are assigned to zone_code via the SAME market ->
   zone_code lookup this pipeline already built from live market
   coordinates (point-in-polygon), rather than via Busker's own `Admin 2`
   string field (which uses a stale/ambiguous zone-naming convention --
   see compute_wfp_prices_admin2_monthly_features.py docstring point 4).

   Busker's paper itself only ever built a maize+diesel ALPS feature, not
   a raw zone-month price for every commodity here -- so this validates
   raw price LEVELS against Busker's raw extracted rows, not against a
   Busker-published feature value.
3. Livestock (goat/sheep) and terms-of-trade sanity checks. NOT diffed
   against Busker's extract -- his file has zero livestock commodities
   (he did not use this data), so there is no independent reference to
   validate against. These checks instead confirm internal consistency:
   plausible price range, and that the ratio columns are always exactly
   reconstructible from the two price legs that produced them (catches a
   join/alignment bug the way a numeric diff against an external source
   would catch a units/scale bug).
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "wfp_prices_admin2_monthly.csv")
BUSKER_REF_PATH = os.path.join(SCRIPT_DIR, "busker_comparison", "WFP_Ethiopia_FoodPrices_new_tool_retail.csv")
MARKETS_CSV = os.path.join(REPO_ROOT, "data", "raw", "wfp_prices", "wfp_markets_eth_live.csv")
ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

SNAP_TOLERANCE_KM = 5.0

# Busker's raw `Commodity` labels -> this pipeline's commodity_out name +
# unit-normalisation factor (mirrors COMMODITY_MAP in the compute script).
BUSKER_COMMODITY_MAP = {
    "Maize (white)":  ("maize", {"KG": 1.0, "100 KG": 1.0 / 100.0}),
    "Wheat":          ("wheat", {"KG": 1.0, "100 KG": 1.0 / 100.0}),
    "Wheat (white)":  ("wheat", {"KG": 1.0, "100 KG": 1.0 / 100.0}),
    "Sorghum":        ("sorghum", {"KG": 1.0, "100 KG": 1.0 / 100.0}),
    "Sorghum (white)": ("sorghum", {"KG": 1.0, "100 KG": 1.0 / 100.0}),
    "Teff (mixed)":   ("teff", {"KG": 1.0, "100 KG": 1.0 / 100.0}),
    "Teff (white)":   ("teff", {"KG": 1.0, "100 KG": 1.0 / 100.0}),
    "Fuel (diesel)":  ("fuel_diesel", {"L": 1.0}),
    "Fuel (petrol-gasoline)": ("fuel_petrol", {"L": 1.0}),
}

AGREEMENT_THRESHOLD_PCT = 20.0  # "within X% agreement" band, matching CHIRPS/locust convention


def point_in_polygon_join(points_df, gdf_admin2, lon_col="longitude", lat_col="latitude"):
    geom = gpd.GeoSeries([Point(xy) for xy in zip(points_df[lon_col], points_df[lat_col])], crs="EPSG:4326")
    pts_gdf = gpd.GeoDataFrame(points_df.copy(), geometry=geom, crs="EPSG:4326")
    joined = gpd.sjoin(pts_gdf, gdf_admin2[[ADMIN2_PCODE_FIELD, "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=["index_right"])

    missing_mask = joined[ADMIN2_PCODE_FIELD].isna()
    if missing_mask.sum() > 0:
        equal_area_crs = "ESRI:102022"
        gdf_proj = gdf_admin2.to_crs(equal_area_crs)
        missing_pts = pts_gdf.loc[missing_mask].to_crs(equal_area_crs)
        nearest = gpd.sjoin_nearest(missing_pts, gdf_proj[[ADMIN2_PCODE_FIELD, "geometry"]],
                                     how="left", distance_col="dist_m")
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        within_tolerance = nearest["dist_m"] <= (SNAP_TOLERANCE_KM * 1000)
        snapped = nearest.loc[within_tolerance, ADMIN2_PCODE_FIELD]
        joined.loc[snapped.index, ADMIN2_PCODE_FIELD] = snapped.values

    joined = joined[joined[ADMIN2_PCODE_FIELD].notna()]
    return joined.rename(columns={ADMIN2_PCODE_FIELD: "zone_code"})


def main():
    df = pd.read_csv(OUTPUT_PATH, parse_dates=["month"])

    print("=" * 60)
    print("1. OVERALL SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Zones: {df['zone_code'].nunique()} (expected 92)")
    print(f"Date range: {df['month'].min().date()} to {df['month'].max().date()}")
    assert df["zone_code"].nunique() == 92, "BUG: expected all 92 admin2 zones represented"

    commodities = ["maize", "wheat", "sorghum", "teff", "fuel_diesel", "fuel_petrol",
                   "livestock_goat", "livestock_sheep"]
    for c in commodities:
        price_col = f"{c}_price_etb"
        idx_col = f"{c}_price_index"
        n = df[price_col].notna().sum()
        n_idx = df[idx_col].notna().sum()
        print(f"  {c}: {n}/{len(df)} zone-months with a price ({n/len(df):.1%}), "
              f"{n_idx} with a price_index ({n_idx/len(df):.1%})")
        # price_index should never be non-null where price itself is null
        bad = df[df[price_col].isna() & df[idx_col].notna()]
        assert len(bad) == 0, f"BUG: {c}_price_index non-null with no {c}_price_etb"
    print("PASS: price_index is never populated without a corresponding raw price.")
    print()

    # ------------------------------------------------------------------
    # 2. Spot-check against Busker's own retail extract
    # ------------------------------------------------------------------
    print("=" * 60)
    print("2. VALIDATION AGAINST BUSKER ET AL.'S RETAIL EXTRACT")
    print("=" * 60)

    gdf_admin2 = gpd.read_file(ADMIN2_SHP_PATH)
    markets = pd.read_csv(MARKETS_CSV)
    unique_markets = markets.drop_duplicates(subset=["market_id"])
    market_zone_by_name = point_in_polygon_join(unique_markets, gdf_admin2).drop_duplicates(
        subset=["market"]).set_index("market")["zone_code"]

    busker = pd.read_csv(BUSKER_REF_PATH, sep=";", decimal=",")
    busker["Price Date"] = pd.to_datetime(busker["Price Date"], format="%d-%m-%Y", errors="coerce")
    busker = busker[busker["Price Type"] == "Retail"]
    busker["zone_code"] = busker["Market Name"].map(market_zone_by_name)
    n_unmatched = busker["zone_code"].isna().sum()
    print(f"Busker rows with no market->zone match: {n_unmatched}/{len(busker)}")
    busker = busker[busker["zone_code"].notna()]

    rows = []
    for raw_label, (commodity_out, unit_factors) in BUSKER_COMMODITY_MAP.items():
        sub = busker[busker["Commodity"] == raw_label].copy()
        if sub.empty:
            continue
        unrecognised = set(sub["Unit"].unique()) - set(unit_factors.keys())
        if unrecognised:
            raise ValueError(f"{raw_label}: unrecognised unit(s) {unrecognised}")
        sub["price_norm"] = sub["Price"] * sub["Unit"].map(unit_factors)
        sub["commodity_out"] = commodity_out
        sub["month"] = sub["Price Date"].dt.to_period("M").dt.to_timestamp()
        rows.append(sub[["zone_code", "month", "commodity_out", "price_norm"]])
    busker_long = pd.concat(rows, ignore_index=True)

    busker_zone_month = busker_long.groupby(["zone_code", "month", "commodity_out"]).agg(
        busker_price=("price_norm", "mean")
    ).reset_index()

    overall_diffs = []
    for commodity in commodities:
        ours = df[["zone_code", "month", f"{commodity}_price_etb"]].rename(
            columns={f"{commodity}_price_etb": "our_price"})
        ref = busker_zone_month[busker_zone_month["commodity_out"] == commodity]
        merged = ref.merge(ours, on=["zone_code", "month"], how="inner")
        merged = merged.dropna(subset=["our_price", "busker_price"])
        merged = merged[merged["busker_price"] != 0]
        if merged.empty:
            print(f"  {commodity}: no overlapping zone-months with Busker reference -- skipped")
            continue
        pct_diff = ((merged["our_price"] - merged["busker_price"]).abs() / merged["busker_price"]) * 100
        within_band = (pct_diff <= AGREEMENT_THRESHOLD_PCT).mean() * 100
        print(f"  {commodity}: n={len(merged)} zone-months matched, "
              f"median % diff={pct_diff.median():.1f}%, mean % diff={pct_diff.mean():.1f}%, "
              f"{within_band:.1f}% within {AGREEMENT_THRESHOLD_PCT:.0f}% agreement")
        overall_diffs.append(pct_diff)

    if overall_diffs:
        all_diffs = pd.concat(overall_diffs, ignore_index=True)
        print()
        print(f"OVERALL across all commodities: n={len(all_diffs)}, "
              f"median % diff={all_diffs.median():.1f}%, mean % diff={all_diffs.mean():.1f}%, "
              f"{(all_diffs <= AGREEMENT_THRESHOLD_PCT).mean()*100:.1f}% within "
              f"{AGREEMENT_THRESHOLD_PCT:.0f}% agreement")
    print()

    # ------------------------------------------------------------------
    # 3. Livestock / terms-of-trade internal consistency checks
    # ------------------------------------------------------------------
    print("=" * 60)
    print("3. LIVESTOCK & TERMS-OF-TRADE SANITY CHECKS")
    print("=" * 60)

    for c in ["livestock_goat", "livestock_sheep"]:
        col = f"{c}_price_etb"
        vals = df[col].dropna()
        print(f"  {c}: n={len(vals)}, min={vals.min():.0f}, median={vals.median():.0f}, "
              f"max={vals.max():.0f} ETB/head")
        assert (vals > 0).all(), f"BUG: non-positive {c} price"
        # sanity band -- a goat/sheep should always be worth noticeably more
        # than a kg of maize; catches a units/decimal-place bug, not a
        # precise economic bound
        assert vals.min() > 50, f"BUG: implausibly low {c} price (< 50 ETB/head)"

    tot_pairs = [("goat_maize_tot", "livestock_goat"), ("sheep_maize_tot", "livestock_sheep")]
    for ratio_col, livestock_commodity in tot_pairs:
        n = df[ratio_col].notna().sum()
        print(f"  {ratio_col}: {n}/{len(df)} zone-months with a ratio ({n/len(df):.1%})")

        # Ratio must be non-null EXACTLY where both legs are non-null -- not
        # a looser subset check, since a silent partial-fill here would be
        # the same class of bug as price_index appearing without a price.
        both_legs = df[f"{livestock_commodity}_price_etb"].notna() & df["maize_price_etb"].notna()
        assert (df[ratio_col].notna() == both_legs).all(), \
            f"BUG: {ratio_col} non-null pattern doesn't match exactly where both price legs are present"

        # Reconstruct the ratio from the two price legs and confirm exact
        # agreement -- catches any join/alignment bug directly, the same
        # role the Busker % diff plays for the plain price columns above.
        recon = df[f"{livestock_commodity}_price_etb"] / df["maize_price_etb"]
        mismatch = (df[ratio_col] - recon).abs() > 1e-9
        mismatch = mismatch & df[ratio_col].notna()
        assert mismatch.sum() == 0, f"BUG: {ratio_col} does not reconstruct from its two price legs"
    print("PASS: terms-of-trade ratios reconstruct exactly from their price legs, "
          "no silent partial-fill.")


if __name__ == "__main__":
    main()
