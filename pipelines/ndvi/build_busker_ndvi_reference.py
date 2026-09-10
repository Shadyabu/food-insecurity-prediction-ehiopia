"""
build_busker_ndvi_reference.py

One-off reference builder. NDVI has no ready-made admin2-level reference
table the way CHIRPS did (data_prec_Admin.xlsx) -- Busker et al. only
released gridded NDVI. This script fetches their three released grids
(raw, cropland-masked, rangeland-masked -- same Zenodo package already
used in fetch_ndvi_admin2.py) and zonally aggregates each one using the
SAME admin2 boundary and NaN-safe weighting logic, producing three small
admin2-monthly comparison CSVs that validate_ndvi_output.py checks our
own pipeline's output against.

No additional land-cover-fraction weighting is applied here: Busker's
cropmask/rangemask grids are already masked (non-cropland/rangeland
pixels are NaN), so a plain NaN-safe zonal mean is the correct reference
aggregation for them.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from remotezip import RemoteZip

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_NAME_FIELD = "ADM2_EN"
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "ndvi", "busker_source")
os.makedirs(RAW_DIR, exist_ok=True)
OUT_DIR = os.path.join(SCRIPT_DIR, "busker_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

BUSKER_ZENODO_ZIP_URL = "https://zenodo.org/records/10853668/files/input_data.zip"

TARGETS = [
    ("input_data/NDVI/NDVI_NOA_STAR_1981_2022.nc", "NDVI_NOA_STAR_1981_2022.nc",
     "NDVI", "busker_ndvi_admin2_monthly.csv"),
    ("input_data/NDVI/NDVI_cropmask.nc", "NDVI_cropmask.nc",
     "NDVI", "busker_ndvi_cropland_admin2_monthly.csv"),
    ("input_data/NDVI/NDVI_rangemask.nc", "NDVI_rangemask.nc",
     "NDVI", "busker_ndvi_rangeland_admin2_monthly.csv"),
]

ETHIOPIA_LON_RANGE = (32.5, 48.5)
ETHIOPIA_LAT_RANGE = (3.0, 15.0)
MIN_PIXELS_THRESHOLD = 10


def verify_file_opens_nc(path):
    with xr.open_dataset(path):
        pass


def fetch_zip_entry(entry_name, out_path):
    if os.path.exists(out_path):
        try:
            verify_file_opens_nc(out_path)
            print(f"Skipping {os.path.basename(out_path)}, already cached and verified.")
            return out_path
        except Exception:
            print(f"Cached {os.path.basename(out_path)} failed to open -- re-extracting.")
            os.remove(out_path)

    print(f"Extracting {entry_name} ...")
    tmp_path = out_path + ".partial"
    with RemoteZip(BUSKER_ZENODO_ZIP_URL) as zf:
        with zf.open(entry_name) as src, open(tmp_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    os.replace(tmp_path, out_path)
    verify_file_opens_nc(out_path)
    print(f"  saved and verified -> {out_path}")
    return out_path


def make_slice(coord_values, requested_range):
    lo, hi = min(requested_range), max(requested_range)
    if coord_values[0] > coord_values[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


def compute_pixel_weights(zone_geom, transform, grid_shape, buffer_pixels=3):
    from shapely.geometry import box

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


def weighted_value_nan_safe(data_2d, weights):
    pixel_vals = np.array([data_2d[r, c] for (r, c) in weights.keys()])
    pixel_weights = np.array(list(weights.values()))
    valid = ~np.isnan(pixel_vals)
    if valid.sum() == 0:
        return np.nan
    valid_weights = pixel_weights[valid]
    valid_weights = valid_weights / valid_weights.sum()
    return float(np.sum(pixel_vals[valid] * valid_weights))


gdf = gpd.read_file(ADMIN2_SHP_PATH)
gdf = gdf[[ADMIN2_PCODE_FIELD, ADMIN2_NAME_FIELD, "geometry"]].reset_index(drop=True)
gdf["zone_id"] = gdf.index + 1
zone_id_to_pcode = dict(zip(gdf["zone_id"], gdf[ADMIN2_PCODE_FIELD]))

for entry_name, cache_filename, var_name, out_filename in TARGETS:
    local_path = fetch_zip_entry(entry_name, os.path.join(RAW_DIR, cache_filename))

    ds = xr.open_dataset(local_path)
    ds = ds.sortby("latitude", ascending=False)
    lat_slice = make_slice(ds["latitude"].values, ETHIOPIA_LAT_RANGE)
    lon_slice = make_slice(ds["longitude"].values, ETHIOPIA_LON_RANGE)
    ds = ds.sel(latitude=lat_slice, longitude=lon_slice)

    lats = ds["latitude"].values
    lons = ds["longitude"].values
    lat_res = float(np.median(np.abs(np.diff(lats))))
    lon_res = float(np.median(np.abs(np.diff(lons))))
    transform = from_origin(lons.min() - lon_res / 2, lats.max() + lat_res / 2, lon_res, lat_res)
    grid_shape = (len(lats), len(lons))

    shapes = list(zip(gdf.geometry, gdf["zone_id"]))
    zone_raster = rasterize(shapes, out_shape=grid_shape, transform=transform, fill=0, all_touched=True, dtype="int32")
    pixel_counts = {zid: int(np.sum(zone_raster == zid)) for zid in gdf["zone_id"]}
    zones_needing_fractional = [zid for zid, count in pixel_counts.items() if count < MIN_PIXELS_THRESHOLD]

    zone_weights = {}
    for zone_id in gdf["zone_id"]:
        if zone_id in zones_needing_fractional:
            zone_geom = gdf.loc[gdf["zone_id"] == zone_id, "geometry"].values[0]
            zone_weights[zone_id] = compute_pixel_weights(zone_geom, transform, grid_shape)
        else:
            rows, cols = np.where(zone_raster == zone_id)
            n = len(rows)
            zone_weights[zone_id] = {(r, c): 1.0 / n for r, c in zip(rows, cols)} if n > 0 else {}

    rows_out = []
    var = ds[var_name]
    has_band_dim = "band" in var.dims
    for t in ds["time"].values:
        ts = pd.Timestamp(t)
        arr = var.sel(time=t).isel(band=0).values if has_band_dim else var.sel(time=t).values
        for zone_id, weights in zone_weights.items():
            if not weights:
                continue
            val = weighted_value_nan_safe(arr, weights)
            rows_out.append({"pcode": zone_id_to_pcode[zone_id], "year": ts.year, "month": ts.month, "ndvi_ref": val})
    ds.close()

    ref_df = pd.DataFrame(rows_out).sort_values(["pcode", "year", "month"]).reset_index(drop=True)
    out_path = os.path.join(OUT_DIR, out_filename)
    ref_df.to_csv(out_path, index=False)
    print(f"{out_filename}: {ref_df.shape[0]} rows -> {out_path}")
