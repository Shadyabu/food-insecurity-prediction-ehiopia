"""
compute_usgs_gefs_admin2_monthly_features.py

AGGREGATE + ENGINEER stages for the USGS Rainfall Forecast feature
(CHIRPS-GEFS 15-day forecast -- see fetch_chirps_gefs_15day.py's docstring
for product identity/scope, and docs/USGS_GEFS_Pipeline_Documentation.md
for the full design rationale).

--------------------------------------------------------------------------
LEAD-TIME SCHEME (confirmed with project owner 2026-08-09)
--------------------------------------------------------------------------
Single lead1 column-set, explicitly flagged as PARTIAL-MONTH coverage --
NOT the same guarantee as GloFAS/IRI-CPC's lead1 (which cover the full
target month). A forecast issued at (or nearest before) month-end t has
only a 15-day horizon, so it covers ~days 1-15 of month t+1, not the full
month. `usgs_gefs_coverage_days` (=15) is carried in the output so this
is never silently conflated with a full-month lead1 feature at model-table
build time. No lead3/6/12 -- the product's horizon never reaches there
(same reasoning as GloFAS's own lead6/12 omission, see
docs/GloFAS_Pipeline_Documentation.md §5a).

Like GloFAS and IRI-CPC, this file's lead1 column is PRE-ALIGNED -- it
must be SELECTED at model-table build time, never shifted through
make_lead_table(), since the lead-time alignment (which issue date maps
to which row) already happened here.

--------------------------------------------------------------------------
SPATIAL AGGREGATION
--------------------------------------------------------------------------
Same raster zonal-mean pattern as CHIRPS/GLEAM/NDVI/IRI-CPC: rasterize the
fixed admin2 boundary once against this product's own native grid,
fractional area-weighted overlap for zones with too few pixels under
standard rasterization, NaN-safe weight renormalization.
compute_pixel_weights()/weighted_value_nan_safe() are copy-pasted from
pipelines/chirps/fetch_chirps_admin2.py, matching this project's existing
per-pipeline-copy convention (src/common/ is currently empty).

Unlike CHIRPS's NetCDF ingestion, this product is served as GeoTIFF, which
carries an explicit affine transform in the file itself -- there is no
risk of the "sort the coordinate array" bug that caused CHIRPS's ~0.55°
Harari offset, since rasterio reads the transform directly from the file
header rather than an unordered coordinate array. Still, per CLAUDE.md
§3.3 ("don't assume a new gridded source's latitude ordering"), the
transform's vertical direction is asserted explicitly at runtime rather
than taken on faith.

--------------------------------------------------------------------------
UNITS (confirmed empirically at runtime, not assumed)
--------------------------------------------------------------------------
`data/` band = raw mm, 15-day cumulative forecast total (always >= 0).
`anom/` band = departure from climatology -- the published methodology
(Harrison et al. 2022, Scientific Data) describes CHIRPS-GEFS as offering
BOTH an mm-departure product and a separate standardized (z-score)
product; this pipeline's own directory only exposes one "anom" band, so
its scale is checked empirically below (mm-departure values span a much
wider, location/season-dependent range than a z-score, which is
essentially always within about +/-4). Logged explicitly so a future
session doesn't have to re-derive this.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_PCODE_FIELD = "ADM2_PCODE"
ADMIN2_NAME_FIELD = "ADM2_EN"

INTERIM_DIR = os.path.join(REPO_ROOT, "data", "interim", "usgs_gefs")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed", "features")
MANIFEST_PATH = os.path.join(INTERIM_DIR, "usgs_gefs_issue_manifest.csv")

FEATURE_START_YEAR = 2009
FEATURE_END_YEAR = 2026
COVERAGE_DAYS = 15

MIN_PIXELS_THRESHOLD = 10  # matches CHIRPS's own fractional-overlap trigger


def compute_pixel_weights(zone_geom, transform, grid_shape, buffer_pixels=3):
    """Exact fractional area overlap between a polygon and nearby raster
    pixels. Copy-pasted from pipelines/chirps/fetch_chirps_admin2.py --
    see that file's docstring for why this project copies rather than
    imports this helper."""
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


def weighted_value_nan_safe(band_data, weights):
    """Area-weighted mean for one zone/issue-date, excluding NaN (nodata)
    pixels and renormalizing remaining weights. Copy-pasted from
    pipelines/chirps/fetch_chirps_admin2.py."""
    pixel_vals = np.array([band_data[r, c] for (r, c) in weights.keys()])
    pixel_weights = np.array(list(weights.values()))
    valid = ~np.isnan(pixel_vals)
    if valid.sum() == 0:
        return np.nan
    valid_weights = pixel_weights[valid]
    valid_weights = valid_weights / valid_weights.sum()
    return float(np.sum(pixel_vals[valid] * valid_weights))


def main():
    manifest_df = pd.read_csv(MANIFEST_PATH)

    resolved = manifest_df[manifest_df["source"] == "v3"].reset_index(drop=True)
    if resolved.empty:
        raise RuntimeError("No resolved issue dates in manifest -- run fetch_chirps_gefs_15day.py first.")

    gdf = gpd.read_file(ADMIN2_SHP_PATH)
    gdf = gdf[[ADMIN2_PCODE_FIELD, ADMIN2_NAME_FIELD, "geometry"]].reset_index(drop=True)
    gdf["zone_id"] = gdf.index + 1
    zone_id_to_pcode = dict(zip(gdf["zone_id"], gdf[ADMIN2_PCODE_FIELD]))
    print(f"Loaded {gdf.shape[0]} admin2 zones.")

    # --- Rasterize zones once, against this product's own native grid ---
    sample_path = resolved.iloc[0]["mm_path"]
    with rasterio.open(sample_path) as src:
        transform = src.transform
        grid_shape = (src.height, src.width)
        crs = src.crs

    # §3.3: verify, don't assume, vertical pixel-size sign (north-up
    # convention -> transform.e negative, i.e. row index increases as
    # latitude decreases). Every subsequent file is opened with the SAME
    # assumption; if a future file violates it, ndimage.mean-based zonal
    # stats below would be silently wrong the same way CHIRPS's raw
    # NetCDF ingestion was before its own fix.
    if transform.e >= 0:
        raise RuntimeError(
            f"CHIRPS-GEFS grid is not north-up as expected (transform.e={transform.e} >= 0) "
            "-- do not proceed without re-deriving the rasterization logic for this ordering."
        )
    print(f"Grid: {grid_shape}, CRS={crs}, transform.e={transform.e:.6f} (north-up confirmed).")

    shapes = list(zip(gdf.geometry, gdf["zone_id"]))
    zone_raster = rasterize(shapes, out_shape=grid_shape, transform=transform, fill=0, all_touched=True, dtype="int32")
    present_zone_ids = np.unique(zone_raster)
    present_zone_ids = present_zone_ids[present_zone_ids != 0]

    pixel_counts = {zid: int(np.sum(zone_raster == zid)) for zid in gdf["zone_id"]}
    zones_needing_fractional = [zid for zid, cnt in pixel_counts.items() if cnt < MIN_PIXELS_THRESHOLD]

    fractional_weights = {}
    for zid in zones_needing_fractional:
        zone_geom = gdf.loc[gdf["zone_id"] == zid, "geometry"].values[0]
        weights = compute_pixel_weights(zone_geom, transform, grid_shape)
        fractional_weights[zid] = weights
    if zones_needing_fractional:
        flagged_names = gdf[gdf["zone_id"].isin(zones_needing_fractional)][ADMIN2_NAME_FIELD].tolist()
        print(f"{len(zones_needing_fractional)} zones flagged for fractional-overlap treatment: {flagged_names}")

    standard_zone_ids = np.array([zid for zid in present_zone_ids if zid not in fractional_weights])

    # --- Empirically confirm anomaly-band units (mm-departure vs z-score) ---
    sample_anom_path = resolved.iloc[0]["anom_path"]
    with rasterio.open(sample_anom_path) as src:
        anom_sample = src.read(1)
    finite_anom = anom_sample[np.isfinite(anom_sample)]
    anom_p1, anom_p99 = np.nanpercentile(finite_anom, [1, 99])
    if abs(anom_p1) <= 6 and abs(anom_p99) <= 6:
        anom_units = "zscore"
    else:
        anom_units = "mm_departure"
    print(f"Anomaly band 1st/99th percentile: {anom_p1:.2f} / {anom_p99:.2f} mm-or-zscore -> inferred units: {anom_units}")

    # --- Extract per zone, per resolved issue date ---
    rows = []
    for _, manifest_row in resolved.iterrows():
        year, month = int(manifest_row["year"]), int(manifest_row["month"])
        with rasterio.open(manifest_row["mm_path"]) as src:
            mm_data = src.read(1).astype(np.float64)
            mm_nodata = src.nodata
        with rasterio.open(manifest_row["anom_path"]) as src:
            anom_data = src.read(1).astype(np.float64)
            anom_nodata = src.nodata

        if mm_nodata is not None:
            mm_data = np.where(mm_data == mm_nodata, np.nan, mm_data)
        if anom_nodata is not None:
            anom_data = np.where(anom_data == anom_nodata, np.nan, anom_data)

        # standard zones: vectorized NaN-safe mean via ndimage
        if len(standard_zone_ids) > 0:
            for band_data, col_name in [(mm_data, "usgs_gefs_precip_mm_lead1"), (anom_data, "usgs_gefs_precip_anom_lead1")]:
                valid = ~np.isnan(band_data)
                filled = np.where(valid, band_data, 0.0)
                sums = ndimage.sum(filled, labels=zone_raster, index=standard_zone_ids)
                counts = ndimage.sum(valid.astype(np.float64), labels=zone_raster, index=standard_zone_ids)
                means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
                for zid, val in zip(standard_zone_ids, means):
                    rows.append({
                        "zone_code": zone_id_to_pcode[zid], "year": year, "month": month,
                        "_col": col_name, "_val": val,
                    })

        # small zones: fractional-overlap weighting
        for zid, weights in fractional_weights.items():
            if not weights:
                continue
            for band_data, col_name in [(mm_data, "usgs_gefs_precip_mm_lead1"), (anom_data, "usgs_gefs_precip_anom_lead1")]:
                val = weighted_value_nan_safe(band_data, weights)
                rows.append({
                    "zone_code": zone_id_to_pcode[zid], "year": year, "month": month,
                    "_col": col_name, "_val": val,
                })

    long_df = pd.DataFrame(rows)
    wide_df = long_df.pivot_table(index=["zone_code", "year", "month"], columns="_col", values="_val").reset_index()
    wide_df.columns.name = None

    # attach issue-date / provenance / coverage columns
    manifest_lookup = resolved.set_index(["year", "month"])["issue_date"].to_dict()
    wide_df["usgs_gefs_issue_date_lead1"] = wide_df.apply(lambda r: manifest_lookup.get((int(r["year"]), int(r["month"]))), axis=1)
    wide_df["usgs_gefs_source_lead1"] = "v3"
    wide_df["usgs_gefs_coverage_days"] = COVERAGE_DAYS
    wide_df["usgs_gefs_anom_units_lead1"] = anom_units

    # add explicit gap rows for months that never resolved (no_archive/no_data)
    gap_df = manifest_df[manifest_df["source"] != "v3"].copy()
    if not gap_df.empty:
        gap_rows = []
        for _, r in gap_df.iterrows():
            for zid, pcode in zone_id_to_pcode.items():
                gap_rows.append({
                    "zone_code": pcode, "year": int(r["year"]), "month": int(r["month"]),
                    "usgs_gefs_precip_mm_lead1": np.nan, "usgs_gefs_precip_anom_lead1": np.nan,
                    "usgs_gefs_issue_date_lead1": None, "usgs_gefs_source_lead1": r["source"],
                    "usgs_gefs_coverage_days": COVERAGE_DAYS, "usgs_gefs_anom_units_lead1": anom_units,
                })
        gap_wide = pd.DataFrame(gap_rows)
        wide_df = pd.concat([wide_df, gap_wide], ignore_index=True)

    wide_df = wide_df[
        (wide_df["year"] >= FEATURE_START_YEAR) & (wide_df["year"] <= FEATURE_END_YEAR)
    ].copy()
    wide_df = wide_df.sort_values(["zone_code", "year", "month"]).reset_index(drop=True)

    col_order = [
        "zone_code", "year", "month",
        "usgs_gefs_precip_mm_lead1", "usgs_gefs_precip_anom_lead1",
        "usgs_gefs_issue_date_lead1", "usgs_gefs_source_lead1",
        "usgs_gefs_coverage_days", "usgs_gefs_anom_units_lead1",
    ]
    wide_df = wide_df[col_order]

    out_path = os.path.join(PROCESSED_DIR, "usgs_rainfall_forecast_admin2_monthly.csv")
    wide_df.to_csv(out_path, index=False)
    print(f"\nSaved {wide_df.shape[0]} rows -> {out_path}")
    print(wide_df["usgs_gefs_source_lead1"].value_counts())


if __name__ == "__main__":
    main()
