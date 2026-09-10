"""
compute_wfp_prices_admin2_monthly_features.py

AGGREGATE + ENGINEER stage. Turns the cached live WFP retail price panel
into zone-month price and price-index features at multiple lags.

DESIGN DECISIONS (confirmed 2026-08-08, see chat log / this docstring for
the record -- no separate decision doc, this pipeline is straightforward
enough not to warrant one per CLAUDE.md Sec 6 step 5):

1. Retail only (not wholesale). Matches Busker et al.'s own scope (their
   ALPS-based feature is retail-sourced) and has far better coverage
   (52,976 rows vs. 9,153 wholesale rows in the live pull); fuel commodities
   only exist in the retail series at all.

2. Commodities: maize, wheat, sorghum, teff (staple grains) + diesel and
   petrol-gasoline (fuel). Kerosene excluded -- live reporting stops
   2022-08. Exchange rate excluded -- only 82 rows nationally over
   2017-2025, too sparse for a zone-month panel feature; left for a future
   dedicated series rather than forced into this pipeline.

   WFP relabelled several commodities partway through the series (e.g.
   generic "Sorghum" stops 2020-03 right as "Sorghum (white)" starts
   2020-01) -- same underlying product, new commodity code. Variant labels
   are merged into one series per commodity (see COMMODITY_MAP) to avoid an
   artificial discontinuity at the relabel date. Teff has NO series at all
   before 2020 in the retail data -- this is a real coverage gap, not a
   pipeline bug; teff features are NaN for zone-months before 2020-01.

2a. Livestock (added 2026-08-13): goat and sheep retail prices, unit
   "Head" throughout (no KG/100KG-style conversion needed, unlike grain).
   Cattle/ox/bull/camel were considered and excluded -- they only start
   reporting 2020-01 (vs. 2015-08 for goat/sheep) and cover far fewer
   markets (12-38 vs. 74/72); donkey has 4 rows total and is unusable.
   Goat/sheep cover 42 admin2 zones by raw name, concentrated in exactly
   the pastoral/agropastoral areas the terms-of-trade mechanism is about
   (Afder, Gode, Korahe, Liben, Jijiga in Somali; Zone 1-5 in Afar;
   Borena/Guji in lowland Oromia) -- the gap is temporal (NaN before
   2015-08), not spatial-where-it-matters. No independent reference
   exists to validate against (Busker et al.'s own WFP extract,
   busker_comparison/WFP_Ethiopia_FoodPrices_new_tool_retail.csv, has no
   livestock commodities at all -- he did not use this data), so
   validate_wfp_prices_output.py covers these two with range/coverage
   sanity checks only, not a cross-source diff.

   Terms-of-trade features: `{goat,sheep}_maize_tot` = livestock price
   (ETB/head) / maize price (ETB/kg), the standard FEWS NET
   pastoral-livelihoods ToT pairing (goat:cereal, cereal = maize as the
   most widely reported staple). Computed only where BOTH legs have a
   price for that zone-month -- no fill, no proxying a missing leg,
   consistent with rule 6 below applied to every other price feature
   here. Same lag set (1/3/6/12) as every other price/index column.

2b. Isolated price-spike filter (livestock only, added 2026-08-13): 7
   raw rows are a single-month collapse to ~1/8-1/15 of that market's
   surrounding trend with a full recovery the very next report (e.g.
   Assosa goat sits at 2800-16000 ETB every month for 5+ years except one
   2022-11 report of 25 ETB) -- a WFP data-entry artifact, not an actual
   market crash-and-instant-recovery (three of the seven cluster in the
   same October-2025 bulletin across different zones/species, pointing to
   a release-level entry error, not independent organic price shocks).
   `filter_isolated_price_spikes()` drops any report that is < 20% of
   BOTH its immediately preceding and following report at the same
   market -- a symmetric V-shape detector, not a one-sided floor, so a
   genuine sustained crash (which would NOT recover the next month) is
   left alone. Applied only to livestock: grain/fuel commodities already
   validate at median 0.0% / 90-100% within 20% against Busker's
   independent extract (see validate_wfp_prices_output.py Sec 2), which
   would not be possible if a comparable systemic error existed there, so
   there was no reason to re-check them here (one concern at a time).

3. Unit normalisation: WFP prices are not consistently denominated -- e.g.
   "Sorghum" (pre-2020) is priced per KG while "Sorghum (white)" (2020+) is
   priced per 100 KG. Blindly merging or averaging these would silently
   corrupt the series by a factor of 100. All grain prices are normalised
   to ETB per KG before merging/averaging; fuel is natively litre-priced
   throughout (no conversion needed).

4. Spatial join: point-in-polygon on each market's lat/lon (from
   wfp_markets_eth.csv) against the admin2 boundary, with a 5 km
   nearest-zone fallback -- reusing the exact pattern from
   pipelines/locust/compute_locust_admin2_monthly_features.py. NOT a
   string match on the panel's own `admin2` field, which uses a different,
   partly pre-2021 zone-naming convention (e.g. "GAMO GOFA" predates the
   boundary's later split into separate Gamo/Gofa zones; Gambela's
   "ZONE 1/2/3" don't correspond to the boundary's named zones). Geocoding
   resolves these correctly instead of requiring a fragile manual crosswalk.
   Per CLAUDE.md Sec 3.4, market/point data always joins spatially, never
   through a raster zonal-stats pattern.

5. Zone-month aggregation: mean across all markets in a zone that reported
   that commodity that month (median 2 markets/zone, max 6). A support
   column `{commodity}_n_markets` records how many markets contributed, so
   a mean-of-one is distinguishable from a mean-of-several downstream.

6. Fill rule: NO fill. A zone-commodity-month with no reporting market is
   left NaN, not forward-filled or carried over -- confirmed choice, since
   with a median of 2 markets/zone a single market's reporting gap can
   otherwise be silently papered over.

7. Price index: price_index_t = price_t / median(price_1..t-1), i.e. an
   EXPANDING, BACKWARD-LOOKING median per zone-commodity that excludes the
   current month from its own baseline (never uses future data, per
   CLAUDE.md Sec 3.2). Requires >= MIN_INDEX_HISTORY_MONTHS non-null prior
   zone-months; NaN before that. This is deliberately not a fixed
   pre-training-period baseline (unlike SPI/SPEI) -- a price index is
   meant to answer "how does this month compare to this zone's own history
   so far," which is well-defined at any point in time, not just after a
   fixed climatology window.

8. Lags: 1, 3, 6, 12 months for both price_etb and price_index, computed on
   a complete zone x month calendar scaffold so a "lag" is a true calendar-
   month shift, not "N rows back" (which would be wrong across a gap).

9. Vintage/leakage caveat: this pipeline reads whatever HDX's live snapshot
   currently reports. WFP price bulletins ARE occasionally revised after
   first publication (re-surveyed markets, corrections), but there is no
   practical way to reconstruct the value as it was actually first
   published at each historical month -- HDX does not publish a bulk
   historical-vintage archive for this dataset (unlike GloFAS forecasts,
   which do carry an explicit issue date). This is a real, documented
   limitation, same class as the GLEAM v4.3/v3.5a version deviation noted
   in docs/GLEAM_SSMI_SPEI_Pipeline_Documentation.md Sec 4 -- flag before
   using in a strict real-time-simulation experiment.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "wfp_prices")
PRICES_CSV = os.path.join(RAW_DIR, "wfp_food_prices_eth_live.csv")
MARKETS_CSV = os.path.join(RAW_DIR, "wfp_markets_eth_live.csv")

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "wfp_prices_admin2_monthly.csv")

SNAP_TOLERANCE_KM = 5.0  # boundary fallback for markets outside every polygon -- matches locust pipeline

# commodity_out_name -> (list of raw `commodity` labels to merge, unit-normalisation table)
# unit -> multiplier to convert to the canonical per-unit price (ETB/KG for grain, ETB/L for fuel)
KG_UNIT_FACTORS = {"KG": 1.0, "100 KG": 1.0 / 100.0}
L_UNIT_FACTORS = {"L": 1.0}
HEAD_UNIT_FACTORS = {"Head": 1.0}

COMMODITY_MAP = {
    "maize":           (["Maize (white)"], KG_UNIT_FACTORS),
    "wheat":           (["Wheat", "Wheat (white)"], KG_UNIT_FACTORS),
    "sorghum":         (["Sorghum", "Sorghum (white)"], KG_UNIT_FACTORS),
    "teff":            (["Teff (mixed)", "Teff (white)"], KG_UNIT_FACTORS),
    "fuel_diesel":     (["Fuel (diesel)"], L_UNIT_FACTORS),
    "fuel_petrol":     (["Fuel (petrol-gasoline)"], L_UNIT_FACTORS),
    "livestock_goat":  (["Livestock (Goat)"], HEAD_UNIT_FACTORS),
    "livestock_sheep": (["Livestock (Sheep)"], HEAD_UNIT_FACTORS),
}

# Livestock:cereal terms-of-trade pairs -- (livestock commodity_out name, cereal commodity_out name)
TOT_PAIRS = [
    ("livestock_goat", "maize", "goat_maize_tot"),
    ("livestock_sheep", "maize", "sheep_maize_tot"),
]

MIN_INDEX_HISTORY_MONTHS = 12
LAGS = [1, 3, 6, 12]


def load_admin2():
    return gpd.read_file(ADMIN2_SHP_PATH)


def point_in_polygon_join(points_df, gdf_admin2, lon_col="longitude", lat_col="latitude"):
    """Point-in-polygon join with a nearest-zone fallback -- identical
    pattern to pipelines/locust/compute_locust_admin2_monthly_features.py."""
    geom = gpd.GeoSeries(
        [Point(xy) for xy in zip(points_df[lon_col], points_df[lat_col])],
        crs="EPSG:4326",
    )
    pts_gdf = gpd.GeoDataFrame(points_df.copy(), geometry=geom, crs="EPSG:4326")

    joined = gpd.sjoin(pts_gdf, gdf_admin2[[ADMIN2_PCODE_FIELD, "geometry"]],
                        how="left", predicate="within")
    joined = joined.drop(columns=["index_right"])

    missing_mask = joined[ADMIN2_PCODE_FIELD].isna()
    n_missing_initial = missing_mask.sum()

    if n_missing_initial > 0:
        equal_area_crs = "ESRI:102022"
        gdf_proj = gdf_admin2.to_crs(equal_area_crs)
        missing_pts = pts_gdf.loc[missing_mask].to_crs(equal_area_crs)

        nearest = gpd.sjoin_nearest(
            missing_pts, gdf_proj[[ADMIN2_PCODE_FIELD, "geometry"]],
            how="left", distance_col="dist_m",
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        within_tolerance = nearest["dist_m"] <= (SNAP_TOLERANCE_KM * 1000)

        snapped = nearest.loc[within_tolerance, ADMIN2_PCODE_FIELD]
        joined.loc[snapped.index, ADMIN2_PCODE_FIELD] = snapped.values

        n_snapped = within_tolerance.sum()
        n_dropped = n_missing_initial - n_snapped
        print(f"  {n_missing_initial} points outside every polygon: "
              f"{n_snapped} snapped to nearest zone (<= {SNAP_TOLERANCE_KM} km), "
              f"{n_dropped} dropped (beyond tolerance).")
    else:
        n_dropped = 0

    joined = joined[joined[ADMIN2_PCODE_FIELD].notna()]
    return joined.rename(columns={ADMIN2_PCODE_FIELD: "zone_code"}), n_dropped


def build_market_zone_lookup(markets_df, gdf_admin2):
    unique_markets = markets_df.drop_duplicates(subset=["market_id"])
    joined, n_dropped = point_in_polygon_join(unique_markets, gdf_admin2)
    print(f"Market -> zone geocoding: {len(joined)}/{len(unique_markets)} markets matched "
          f"({n_dropped} dropped beyond {SNAP_TOLERANCE_KM}km tolerance).")
    return joined.set_index("market_id")["zone_code"]


SPIKE_DROP_RATIO = 0.2


def filter_isolated_price_spikes(sub, drop_ratio=SPIKE_DROP_RATIO):
    """Drop a raw report that is < drop_ratio of BOTH its immediately
    preceding and following report at the SAME market -- an isolated
    V-shaped collapse-and-recovery, which is a WFP data-entry error
    signature (see docstring point 2b), not a real price shock (a real
    shock would not fully round-trip back to trend the very next
    report)."""
    sub = sub.sort_values(["market_id", "date"]).copy()
    prev_price = sub.groupby("market_id")["price_norm"].shift(1)
    next_price = sub.groupby("market_id")["price_norm"].shift(-1)
    is_spike = (
        prev_price.notna() & next_price.notna()
        & (sub["price_norm"] < drop_ratio * prev_price)
        & (sub["price_norm"] < drop_ratio * next_price)
    )
    n_dropped = int(is_spike.sum())
    if n_dropped:
        for _, row in sub.loc[is_spike].iterrows():
            print(f"    dropping isolated price spike: {row['date'].date()} "
                  f"market_id={row['market_id']} price_norm={row['price_norm']:.0f}")
    return sub[~is_spike]


def normalise_commodity_block(retail_df, out_name, raw_labels, unit_factors):
    sub = retail_df[retail_df["commodity"].isin(raw_labels)].copy()
    if sub.empty:
        return sub
    unrecognised = set(sub["unit"].unique()) - set(unit_factors.keys())
    if unrecognised:
        raise ValueError(f"{out_name}: unrecognised unit(s) {unrecognised} -- add to unit_factors "
                          "rather than silently guessing a conversion.")
    sub["price_norm"] = sub["price"] * sub["unit"].map(unit_factors)
    sub["commodity_out"] = out_name
    return sub[["zone_code", "market_id", "date", "commodity_out", "price_norm"]]


def build_zone_month_scaffold(all_zone_codes, all_months, commodities):
    idx = pd.MultiIndex.from_product(
        [all_zone_codes, all_months, commodities], names=["zone_code", "month", "commodity_out"]
    )
    return idx.to_frame(index=False)


def compute_price_index(prices):
    """Backward-looking expanding median, excluding the current month from
    its own baseline. `prices` is one zone-commodity's chronologically
    sorted price series (with NaN gaps already present in the row scaffold)."""
    # shift(1) so the expanding window covers strictly PRIOR months only
    prior = prices.shift(1)
    count = prior.expanding().count()
    baseline = prior.expanding().median()
    baseline = baseline.where(count >= MIN_INDEX_HISTORY_MONTHS)
    return prices / baseline


def main():
    gdf_admin2 = load_admin2()
    all_zone_codes = gdf_admin2[ADMIN2_PCODE_FIELD].tolist()

    prices = pd.read_csv(PRICES_CSV)
    markets = pd.read_csv(MARKETS_CSV)
    prices["date"] = pd.to_datetime(prices["date"])

    retail = prices[prices["pricetype"] == "Retail"].copy()

    market_zone = build_market_zone_lookup(markets, gdf_admin2)
    retail["zone_code"] = retail["market_id"].map(market_zone)
    n_unmatched = retail["zone_code"].isna().sum()
    if n_unmatched:
        print(f"Dropping {n_unmatched} retail rows whose market_id has no zone match.")
    retail = retail[retail["zone_code"].notna()]

    blocks = []
    for out_name, (raw_labels, unit_factors) in COMMODITY_MAP.items():
        block = normalise_commodity_block(retail, out_name, raw_labels, unit_factors)
        if out_name.startswith("livestock_") and len(block):
            block = filter_isolated_price_spikes(block)
        print(f"{out_name}: {len(block)} rows from {block['market_id'].nunique() if len(block) else 0} markets, "
              f"{block['zone_code'].nunique() if len(block) else 0} zones "
              f"(raw labels: {raw_labels})")
        blocks.append(block)
    long_df = pd.concat(blocks, ignore_index=True)
    long_df["month"] = long_df["date"].dt.to_period("M")

    # ------------------------------------------------------------------
    # Zone-month aggregation: mean across markets + a market-count support column
    # ------------------------------------------------------------------
    agg = long_df.groupby(["zone_code", "month", "commodity_out"]).agg(
        price_etb=("price_norm", "mean"),
        n_markets=("price_norm", "size"),
    ).reset_index()

    # ------------------------------------------------------------------
    # Full zone x month x commodity scaffold -- NO fill (confirmed choice),
    # but needed so lags/index reference true calendar months.
    # ------------------------------------------------------------------
    all_months = pd.period_range(long_df["month"].min(), long_df["month"].max(), freq="M")
    scaffold = build_zone_month_scaffold(all_zone_codes, all_months, list(COMMODITY_MAP.keys()))
    out = scaffold.merge(agg, on=["zone_code", "month", "commodity_out"], how="left")
    out["n_markets"] = out["n_markets"].fillna(0).astype(int)

    out = out.sort_values(["zone_code", "commodity_out", "month"]).reset_index(drop=True)
    out["price_index"] = out.groupby(["zone_code", "commodity_out"])["price_etb"].transform(compute_price_index)

    for lag in LAGS:
        out[f"price_etb_lag{lag}"] = out.groupby(["zone_code", "commodity_out"])["price_etb"].shift(lag)
        out[f"price_index_lag{lag}"] = out.groupby(["zone_code", "commodity_out"])["price_index"].shift(lag)

    # ------------------------------------------------------------------
    # Reshape long (one row per zone/month/commodity) -> wide (one row per
    # zone/month, commodity-prefixed columns) to match the project's
    # standard output shape.
    # ------------------------------------------------------------------
    value_cols = ["price_etb", "price_index", "n_markets"] + \
                 [f"price_etb_lag{lag}" for lag in LAGS] + \
                 [f"price_index_lag{lag}" for lag in LAGS]

    wide = out.pivot(index=["zone_code", "month"], columns="commodity_out", values=value_cols)
    wide.columns = [f"{commodity}_{col}" for col, commodity in wide.columns]
    wide = wide.reset_index()

    n_market_cols = [c for c in wide.columns if c.endswith("_n_markets")]
    wide[n_market_cols] = wide[n_market_cols].fillna(0).astype(int)

    wide["month"] = wide["month"].dt.to_timestamp()
    wide = wide.sort_values(["zone_code", "month"]).reset_index(drop=True)

    # ------------------------------------------------------------------
    # Livestock:cereal terms-of-trade ratios -- NaN whenever either leg is
    # missing that zone-month (no fill, no proxying), same lag set as every
    # other price feature.
    # ------------------------------------------------------------------
    for livestock_commodity, cereal_commodity, out_name in TOT_PAIRS:
        wide[out_name] = wide[f"{livestock_commodity}_price_etb"] / wide[f"{cereal_commodity}_price_etb"]
        for lag in LAGS:
            wide[f"{out_name}_lag{lag}"] = wide.groupby("zone_code")[out_name].shift(lag)

    # Stable, documented column order: zone_code, month, then each
    # commodity's block (price, lags, index, index-lags, n_markets), then
    # the terms-of-trade ratio blocks.
    ordered_cols = ["zone_code", "month"]
    for commodity in COMMODITY_MAP:
        ordered_cols.append(f"{commodity}_price_etb")
        ordered_cols += [f"{commodity}_price_etb_lag{lag}" for lag in LAGS]
        ordered_cols.append(f"{commodity}_price_index")
        ordered_cols += [f"{commodity}_price_index_lag{lag}" for lag in LAGS]
        ordered_cols.append(f"{commodity}_n_markets")
    for _, _, out_name in TOT_PAIRS:
        ordered_cols.append(out_name)
        ordered_cols += [f"{out_name}_lag{lag}" for lag in LAGS]
    wide = wide[ordered_cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    wide.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(wide)} zone-month rows, {len(wide.columns)} columns -> {OUTPUT_PATH}")
    print(f"Zones: {wide['zone_code'].nunique()} (expected {len(all_zone_codes)})")
    print(f"Months: {wide['month'].min().date()} to {wide['month'].max().date()}")
    for commodity in COMMODITY_MAP:
        col = f"{commodity}_price_etb"
        n_nonnull = wide[col].notna().sum()
        print(f"  {commodity}: {n_nonnull}/{len(wide)} zone-months with a price "
              f"({n_nonnull / len(wide):.1%})")
    for _, _, out_name in TOT_PAIRS:
        n_nonnull = wide[out_name].notna().sum()
        print(f"  {out_name}: {n_nonnull}/{len(wide)} zone-months with a ratio "
              f"({n_nonnull / len(wide):.1%})")


if __name__ == "__main__":
    main()
