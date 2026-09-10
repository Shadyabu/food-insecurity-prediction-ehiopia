"""
compute_iri_cpc_seasonal_admin2_monthly_features.py

AGGREGATE + ENGINEER stages for the IRI seasonal precipitation
tercile-probability forecast.

--------------------------------------------------------------------------
CONFIRMED DECISIONS (2026-08-09, all verified against real downloaded
IRIDL data -- see docs/IRI_CPC_Seasonal_Pipeline_Documentation.md)
--------------------------------------------------------------------------

1. LEAD-TIME SCHEME (GloFAS-style, per pipelines/glofas's own precedent):
   only L=1 and L=3 are built as this project's lead1/lead3 features.
   L6/L12 are NOT built -- IRI's own product only ever publishes L=1..4
   months ahead (confirmed via the /dataselection.html grid description
   on both products), so lead6/lead12 would have to be faked from a
   forecast that structurally doesn't reach that far. Exactly the same
   reasoning GloFAS used to drop lead6/lead12.

   IRI's own L index conveniently already IS "months ahead of the issue
   month" (confirmed empirically: for issue month F, L=k's target_season
   is the 3-month season starting at month F+k -- checked against F=Jul
   2026, Jan 2020, Jun 2021, Feb 2017, all four gave the exact same
   L->season-start-offset rule). So this project's lead1 = IRI's L=1 and
   lead3 = IRI's L=3 directly, with no day-window slicing needed the way
   GloFAS's medium-range/seasonal day-selection required.

2. TWO PRODUCTS, SPLICED (mirrors GloFAS's operational/reforecast split
   and GLEAM's v3.5a/v4.3a version-caveat, same *shape* of caveat, not
   copied blindly):
   - "legacy" (SOURCES/.IRI/.FD/.Seasonal_Forecast/.Precipitation/.prob):
     2.5 deg grid, F = Sep 1997 to Mar 2017. Confirmed to have COMPLETE
     monthly coverage from 2009-01 through 2017-01 (zero missing months)
     -- covers this project's full FEATURE_START_YEAR=2009 requirement.
   - "nmme_elr" (SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob):
     1.0 deg grid, F = Feb 2017 onward (rolling, updated monthly).
   These OVERLAP in Feb-Mar 2017 -- nmme_elr is preferred for any month
   present in both (the current, finer-resolution, actively-maintained
   product), legacy used only strictly before nmme_elr's own start. Every
   row carries `seasonal_source_lead{1,3}` ("legacy_two_tier" /
   "nmme_elr") so this resolution discontinuity is visible and can be
   controlled for, not silently smoothed over.

3. TERCILE FORMAT KEPT, NOT COLLAPSED: three columns per lead time
   (seasonal_precip_prob_{below,near,above}_lead{1,3}), values in PERCENT
   (0-100, the product's native units -- confirmed via a real data pull:
   the three categories sum to ~100.0 at essentially every non-edge grid
   cell). No expected-value collapse -- would discard exactly the
   uncertainty information this product exists to convey.

4. ETHIOPIA CALENDAR ALIGNMENT: generic rolling 3-month seasons (every
   month still gets a feature) PLUS four boolean overlap flags per lead
   time -- `seasonal_overlaps_{kiremt,belg,gu,deyr}_lead{1,3}` -- against
   FEWS NET's standard documented rain windows (Kiremt Jun-Sep, Belg
   Feb-May, Gu Apr-Jun, Deyr Oct-Dec). These are FIXED, nationally-known
   calendar windows, not re-derived per zone -- crop_calendar's own
   per-zone season_system assignment (national_belg_meher /
   somali_gu_deyr / benishangul_meher_only, in
   boundaries/crop_calendar_admin2_month.csv) is the existing source of
   truth for WHICH of these actually applies to a given zone, and should
   be joined from there at model-table build time rather than duplicated
   in this file.

5. PRE-ALIGNED, LIKE GLOFAS: this file's lead1/lead3 columns must be
   SELECTED at model-table build time, never shifted through
   make_lead_table() -- the lead-time alignment already happened here
   (different underlying forecast slice per lead), exactly per GloFAS's
   Sec 7a. Do not run this file through the standard shift logic.

--------------------------------------------------------------------------
Spatial aggregation follows the CHIRPS/GLEAM raster zonal-mean pattern
(rasterize the fixed admin2 boundary once per product's native
resolution, area-weighted fractional overlap for small zones, NaN-safe
weight renormalization) -- NOT GloFAS's river-network point-join pattern,
since this is a genuine continuous gridded field, not a discrete network.
compute_pixel_weights()/weighted_value_nan_safe() are copy-pasted from
pipelines/chirps/fetch_chirps_admin2.py, matching this project's existing
convention of copy-pasting these per pipeline rather than importing from
src/common/ (which is currently empty -- see e.g. the ACLED pipeline
doc's note that the same function already exists independently in
pipelines/locust/ and pipelines/wfp_prices/).
"""

import math
import os

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box


def compute_pixel_weights(zone_geom, transform, grid_shape, buffer_pixels=3):
    """Exact fractional area overlap between a polygon and nearby raster
    pixels -- copy-pasted from pipelines/chirps/fetch_chirps_admin2.py."""
    minx, miny, maxx, maxy = zone_geom.bounds
    col_min, row_min = ~transform * (minx, maxy)
    col_max, row_max = ~transform * (maxx, miny)
    row_start = max(0, int(row_min) - buffer_pixels)
    row_end = min(grid_shape[0], int(row_max) + buffer_pixels + 1)
    col_start = max(0, int(col_min) - buffer_pixels)
    col_end = min(grid_shape[1], int(col_max) + buffer_pixels + 1)

    weights = {}
    zone_area = zone_geom.area
    if zone_area == 0:
        return weights

    for r in range(row_start, row_end):
        for c in range(col_start, col_end):
            px_minx, px_maxy = transform * (c, r)
            px_maxx, px_miny = transform * (c + 1, r + 1)
            pixel_box = box(px_minx, px_miny, px_maxx, px_maxy)
            overlap = zone_geom.intersection(pixel_box).area
            if overlap > 0:
                weights[(r, c)] = overlap / zone_area
    return weights


def weighted_value_nan_safe(grid2d, weights):
    """Area-weighted mean excluding NaN pixels with weight renormalization
    -- copy-pasted from pipelines/chirps/fetch_chirps_admin2.py."""
    pixel_vals = np.array([grid2d[r, c] for (r, c) in weights.keys()])
    pixel_weights = np.array(list(weights.values()))
    valid = ~np.isnan(pixel_vals)
    if valid.sum() == 0:
        return np.nan
    valid_weights = pixel_weights[valid]
    valid_weights = valid_weights / valid_weights.sum()
    return float(np.sum(pixel_vals[valid] * valid_weights))


# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "iri_cpc_seasonal")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed", "features")
os.makedirs(PROCESSED_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "iri_cpc_seasonal_admin2_monthly.csv")

FEATURE_START_YEAR = 2009
FEATURE_END_YEAR = 2026

LEAD_TIMES = [1, 3]
CATEGORY_ORDER = ["Below_Normal", "Normal", "Above_Normal"]  # confirmed C-axis order via real data pull
CATEGORY_COL_SUFFIX = {"Below_Normal": "below", "Normal": "near", "Above_Normal": "above"}

MIN_PIXELS_THRESHOLD = 10  # same threshold CHIRPS uses; at 1deg/2.5deg most admin2
                            # zones will land under this and need fractional treatment

# Standard 12-season cyclic table, indexed by the season's OWN starting
# calendar month (1=Jan .. 12=Dec) -- matches IRI's own published codes
# exactly (confirmed empirically: F=Jul2026 L=1 -> "aso" i.e. season
# starting Aug; F=Jan2020 L=1 -> "fma" i.e. season starting Feb; etc.)
SEASON_CODE_BY_START_MONTH = {
    1: "jfm", 2: "fma", 3: "mam", 4: "amj", 5: "mjj", 6: "jja",
    7: "jas", 8: "aso", 9: "son", 10: "ond", 11: "ndj", 12: "djf",
}

# FEWS NET's standard, nationally-documented Ethiopia rain windows (fixed
# calendar months, not re-derived per zone -- see module docstring point 4).
KIREMT_MONTHS = {6, 7, 8, 9}
BELG_MONTHS = {2, 3, 4, 5}
GU_MONTHS = {4, 5, 6}
DEYR_MONTHS = {10, 11, 12}


def decode_iridl_month(f_value):
    """See fetch_iri_cpc_seasonal.py's decode_iridl_month() docstring --
    IRIDL's F axis is months-since-1960-01 under a 360-day pseudo
    calendar that xarray's CF decoder can't parse; decode manually."""
    month_index = math.floor(f_value)
    year = 1960 + month_index // 12
    month = month_index % 12 + 1
    return year, month


def season_months(issue_year, issue_month, lead_time):
    """3 consecutive calendar (year, month) tuples starting lead_time
    months after the issue month -- the confirmed L->season-start rule."""
    start_index = (issue_year * 12 + (issue_month - 1)) + lead_time
    months = []
    for offset in range(3):
        idx = start_index + offset
        y = idx // 12
        m = idx % 12 + 1
        months.append((y, m))
    return months


def season_code_and_overlaps(issue_year, issue_month, lead_time):
    months = season_months(issue_year, issue_month, lead_time)
    start_month = months[0][1]
    code = SEASON_CODE_BY_START_MONTH[start_month]
    month_set = {m for (_y, m) in months}
    return {
        "target_season": code,
        "overlaps_kiremt": bool(month_set & KIREMT_MONTHS),
        "overlaps_belg": bool(month_set & BELG_MONTHS),
        "overlaps_gu": bool(month_set & GU_MONTHS),
        "overlaps_deyr": bool(month_set & DEYR_MONTHS),
    }


def rasterize_zones(gdf, lats, lons):
    """Builds a zone-id raster + fractional-weight fallback for small
    zones, at whatever native resolution `lats`/`lons` describe. Returns
    (zone_raster, transform, grid_shape, standard_zone_ids,
    fractional_weights)."""
    # Force latitude into a known, explicit descending order before
    # building the transform -- per CLAUDE.md 3.3, this is exactly the
    # check that caught CHIRPS's ~0.55deg offset bug. IRIDL's Y axis for
    # this product comes back ascending (3.0 -> 15.0), so this is a real,
    # necessary flip, not a no-op.
    lat_order = np.argsort(-lats)  # descending
    lats_sorted = lats[lat_order]

    lat_res = abs(lats_sorted[1] - lats_sorted[0]) if len(lats_sorted) > 1 else 1.0
    lon_res = abs(lons[1] - lons[0]) if len(lons) > 1 else 1.0
    transform = from_origin(lons.min() - lon_res / 2, lats_sorted.max() + lat_res / 2, lon_res, lat_res)
    grid_shape = (len(lats_sorted), len(lons))

    shapes = list(zip(gdf.geometry, gdf["zone_id"]))
    zone_raster = rasterize(shapes, out_shape=grid_shape, transform=transform, fill=0, all_touched=True, dtype="int32")

    pixel_counts = {zid: int(np.sum(zone_raster == zid)) for zid in gdf["zone_id"]}
    zones_needing_fractional = [zid for zid, count in pixel_counts.items() if count < MIN_PIXELS_THRESHOLD]

    fractional_weights = {}
    for zone_id in zones_needing_fractional:
        zone_geom = gdf.loc[gdf["zone_id"] == zone_id, "geometry"].values[0]
        weights = compute_pixel_weights(zone_geom, transform, grid_shape)
        fractional_weights[zone_id] = weights

    present_zone_ids = np.unique(zone_raster)
    present_zone_ids = present_zone_ids[present_zone_ids != 0]
    standard_zone_ids = np.array([zid for zid in present_zone_ids if zid not in fractional_weights])

    return zone_raster, lat_order, standard_zone_ids, fractional_weights


def extract_zone_means(ds, gdf, zone_id_to_pcode):
    """For one product's already-lead-restricted NetCDF (dims F, L, C, Y,
    X with L of length 1), returns a long DataFrame of
    pcode/year/month/category/prob rows."""
    lats = ds["Y"].values
    lons = ds["X"].values
    zone_raster, lat_order, standard_zone_ids, fractional_weights = rasterize_zones(gdf, lats, lons)

    lead_time = int(round(float(ds["L"].values[0])))
    f_values = ds["F"].values
    # Dim ORDER of 'prob' differs between the two products (confirmed:
    # legacy is F,L,Y,X,C; nmme_elr is F,L,C,Y,X) -- select by NAME via
    # xarray + explicit .transpose(), never assume a fixed positional
    # order (the exact lesson GloFAS's _latlon_dim_names() already
    # encodes for its own dimension-naming inconsistency).
    prob_da = ds["prob"]

    rows = []
    for f_idx, f_val in enumerate(f_values):
        issue_year, issue_month = decode_iridl_month(float(f_val))
        for c_idx, category in enumerate(CATEGORY_ORDER):
            grid2d = prob_da.isel(F=f_idx, L=0, C=c_idx).transpose("Y", "X").values
            grid2d = grid2d[lat_order, :]  # match the descending-lat order used to build the transform

            if len(standard_zone_ids) > 0:
                valid = ~np.isnan(grid2d)
                filled = np.where(valid, grid2d, 0.0)
                sums = ndimage.sum(filled, labels=zone_raster, index=standard_zone_ids)
                counts = ndimage.sum(valid.astype(np.float64), labels=zone_raster, index=standard_zone_ids)
                means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
                for zid, mean_val in zip(standard_zone_ids, means):
                    rows.append({
                        "pcode": zone_id_to_pcode[zid], "year": issue_year, "month": issue_month,
                        "lead_time": lead_time, "category": category, "prob": mean_val,
                    })

            for zid, weights in fractional_weights.items():
                if not weights:
                    continue
                weighted_val = weighted_value_nan_safe(grid2d, weights)
                rows.append({
                    "pcode": zone_id_to_pcode[zid], "year": issue_year, "month": issue_month,
                    "lead_time": lead_time, "category": category, "prob": weighted_val,
                })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("IRI seasonal precipitation forecast: admin2 x month features")

    gdf = gpd.read_file(ADMIN2_SHP_PATH)
    gdf = gdf[[ADMIN2_PCODE_FIELD, "geometry"]].reset_index(drop=True)
    gdf["zone_id"] = gdf.index + 1
    zone_id_to_pcode = dict(zip(gdf["zone_id"], gdf[ADMIN2_PCODE_FIELD]))
    print(f"Loaded {gdf.shape[0]} admin2 zones.")

    product_frames = {}  # (product, lead_time) -> long df
    for product in ["legacy", "nmme_elr"]:
        for lead_time in LEAD_TIMES:
            nc_path = os.path.join(RAW_DIR, f"{product}_lead{lead_time}.nc")
            print(f"\n[{product} L={lead_time}] loading {nc_path}")
            ds = xr.open_dataset(nc_path, decode_times=False)
            df = extract_zone_means(ds, gdf, zone_id_to_pcode)
            df["product"] = product
            product_frames[(product, lead_time)] = df
            print(f"  {df.shape[0]} rows, {df[['year', 'month']].drop_duplicates().shape[0]} issue-months")
            ds.close()

    # --------------------------------------------------------------------
    # Splice legacy + nmme_elr per lead time: nmme_elr preferred for any
    # (year, month) present in both (see module docstring point 2).
    # --------------------------------------------------------------------
    spliced_by_lead = {}
    for lead_time in LEAD_TIMES:
        nmme_df = product_frames[("nmme_elr", lead_time)]
        legacy_df = product_frames[("legacy", lead_time)]
        nmme_months = set(zip(nmme_df["year"], nmme_df["month"]))
        legacy_df = legacy_df[~legacy_df.apply(lambda r: (r["year"], r["month"]) in nmme_months, axis=1)]
        combined = pd.concat([legacy_df, nmme_df], ignore_index=True)
        combined["source"] = combined["product"].map({"legacy": "legacy_two_tier", "nmme_elr": "nmme_elr"})
        spliced_by_lead[lead_time] = combined
        print(
            f"\nLead {lead_time}: {legacy_df.shape[0]} legacy rows + {nmme_df.shape[0]} nmme_elr rows "
            f"after overlap resolution -> {combined.shape[0]} total"
        )

    # --------------------------------------------------------------------
    # Pivot categories to columns, add target_season/overlap flags, merge
    # lead1 + lead3 into one wide table keyed by pcode/year/month.
    # --------------------------------------------------------------------
    wide_frames = []
    for lead_time in LEAD_TIMES:
        df = spliced_by_lead[lead_time]
        # dropna=False is REQUIRED here -- pandas' pivot_table default
        # (dropna=True) silently drops any (pcode, year, month) group
        # whose aggregated value is NaN across every category column, not
        # just all-NaN COLUMNS as the name suggests. Confirmed via a real
        # case (ET0101, Nov 2010): the legacy archive genuinely has no
        # valid pixel data for that zone/month (see module docstring),
        # and without dropna=False the whole row -- including its
        # `source` flag -- silently vanished instead of surfacing as an
        # honest NaN, which would have made a real archive gap
        # indistinguishable from "never queried."
        #
        # Side effect (confirmed, and welcome): with a MultiIndex and
        # dropna=False, pivot_table reindexes to the full cartesian
        # product of each index level's unique values, not just the
        # (pcode, year, month) combos actually observed -- e.g. once any
        # zone has a Dec row, EVERY zone gets a Dec row, even for a
        # calendar month neither source product ever actually reached
        # (e.g. 2026-08 onward, past nmme_elr's Jul-2026 archive tip).
        # This is exactly what's wanted: it produces a complete
        # rectangular pcode x year x month panel matching every other
        # pipeline's output shape, and these genuinely-out-of-archive
        # rows get `seasonal_source_lead{1,3}` == NaN (no product covers
        # them at all) -- distinct from the Nov-2010-style case above,
        # where source IS set but the pixel value itself was NaN. Verify
        # this distinction holds before trusting either NaN pattern.
        pivoted = df.pivot_table(
            index=["pcode", "year", "month"], columns="category", values="prob", aggfunc="first", dropna=False
        ).reset_index()
        pivoted.columns.name = None
        for category in CATEGORY_ORDER:
            if category not in pivoted.columns:
                pivoted[category] = np.nan

        source_lookup = df.drop_duplicates(["pcode", "year", "month"]).set_index(["pcode", "year", "month"])["source"]
        pivoted["source"] = pivoted.set_index(["pcode", "year", "month"]).index.map(source_lookup)

        season_info = pivoted.apply(
            lambda r: season_code_and_overlaps(int(r["year"]), int(r["month"]), lead_time), axis=1
        )
        season_info_df = pd.DataFrame(list(season_info))

        out = pd.DataFrame({
            "pcode": pivoted["pcode"], "year": pivoted["year"], "month": pivoted["month"],
            f"seasonal_precip_prob_below_lead{lead_time}": pivoted["Below_Normal"],
            f"seasonal_precip_prob_near_lead{lead_time}": pivoted["Normal"],
            f"seasonal_precip_prob_above_lead{lead_time}": pivoted["Above_Normal"],
            f"seasonal_target_season_lead{lead_time}": season_info_df["target_season"],
            f"seasonal_overlaps_kiremt_lead{lead_time}": season_info_df["overlaps_kiremt"],
            f"seasonal_overlaps_belg_lead{lead_time}": season_info_df["overlaps_belg"],
            f"seasonal_overlaps_gu_lead{lead_time}": season_info_df["overlaps_gu"],
            f"seasonal_overlaps_deyr_lead{lead_time}": season_info_df["overlaps_deyr"],
            f"seasonal_source_lead{lead_time}": pivoted["source"],
        })
        wide_frames.append(out)

    final_df = wide_frames[0]
    for other in wide_frames[1:]:
        final_df = final_df.merge(other, on=["pcode", "year", "month"], how="outer")

    final_df = final_df[
        (final_df["year"] >= FEATURE_START_YEAR) & (final_df["year"] <= FEATURE_END_YEAR)
    ].copy()
    final_df = final_df.sort_values(["pcode", "year", "month"]).reset_index(drop=True)

    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nFinal monthly feature table: {final_df.shape[0]} rows")
    print(f"Saved to {OUTPUT_PATH}")
    print(f"Columns: {list(final_df.columns)}")
    for lead_time in LEAD_TIMES:
        col = f"seasonal_precip_prob_below_lead{lead_time}"
        print(f"  {col} NaN rate: {final_df[col].isna().mean():.2%}")
        src_col = f"seasonal_source_lead{lead_time}"
        print(f"  {src_col} value counts:\n{final_df[src_col].value_counts(dropna=False).to_string()}")
