"""
fetch_gleam_admin2.py

GLEAM v4.3a extraction against Ethiopia's fixed 92-zone admin2 boundary --
same spatial framework as chirps/fetch_chirps_admin2.py and
ndvi/fetch_ndvi_admin2.py.

Pulls two GLEAM variables via SFTP (registration-gated, no anonymous bulk
HTTPS like CHIRPS -- see docs/GLEAM_SSMI_SPEI_Pipeline_Documentation.md):
  - Ep   (potential evaporation, mm/month)  -- SPEI input
  - SMrz (root-zone soil moisture, m3/m3)   -- SSMI input

GLEAM's own "monthly" product is used directly (one NetCDF per variable per
year, 12 time-steps each), not the daily product -- GLEAM has already made
the flux-accumulation/state-averaging call for us at that stage, and it's
~30x smaller to download. Busker et al. (2024) used GLEAM v3.5a (0.25 deg);
v3.5a is no longer served, so this uses the current v4.3a (0.1 deg,
satellite+reanalysis, 1980-2025) -- a documented version deviation, flagged
for RQ1 comparability. GLEAM data is only updated once a year (Busker et
al., citing GLEAM 2024), so feature coverage ends at the latest full year
available (2025), one year short of CHIRPS's 2026.

Reuses the validated fixes from the CHIRPS pipeline:
  - Auto-forced lat descending order before building the transform
  - NaN-safe weighted zonal aggregation (never a plain sum/mean)
  - Fractional area-weighted overlap for zones with too few pixels
  - Verify-before-trusting-cache pattern for downloads

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
pip install rasterio geopandas pandas numpy scipy paramiko xarray netCDF4 --break-system-packages

Credentials: pipelines/gleam_ssmi_spei/.gleam_credentials.json (gitignored),
shape: {"host": ..., "port": ..., "username": ..., "password": ...}
Register at https://www.gleam.eu/ ("Downloads" section) to obtain one.
"""

import functools
import json
import os
import socket

import geopandas as gpd
import numpy as np
import pandas as pd
import paramiko
import xarray as xr
from rasterio.features import rasterize

print = functools.partial(print, flush=True)  # unbuffered -- progress must be visible in a redirected log
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box


def compute_pixel_weights(zone_geom, transform, grid_shape, buffer_pixels=3):
    """Exact fractional area overlap between a polygon and nearby raster
    pixels -- fallback for zones too small/oddly-shaped for binary
    all_touched rasterization. Returns {(row, col): weight}, summing to ~1.0."""
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


def weighted_value_nan_safe(grid_data, weights):
    """Area-weighted mean for one zone/time-step, excluding NaN (nodata)
    pixels and renormalizing remaining weights."""
    pixel_vals = np.array([grid_data[r, c] for (r, c) in weights.keys()])
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
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/gleam_ssmi_spei/ -> pipelines/ -> repo root

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_NAME_FIELD = "ADM2_EN"
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

CREDENTIALS_PATH = os.path.join(SCRIPT_DIR, ".gleam_credentials.json")

# Matches CHIRPS's own baseline start year (1981) and Busker et al.'s study
# period start -- see docs/CHIRPS_Pipeline_Documentation.md Sec 4.5.
START_YEAR = 1981
# GLEAM v4.3a's latest full year on the SFTP server as of this build (GLEAM
# is only updated once a year -- Busker et al., citing GLEAM 2024). One year
# short of CHIRPS's 2026 feature coverage; document this gap, don't force it.
FEATURE_END_YEAR = 2025

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "gleam")
INTERIM_DIR = os.path.join(REPO_ROOT, "data", "interim", "gleam_ssmi_spei")

ETHIOPIA_LON_RANGE = (32.5, 48.5)
ETHIOPIA_LAT_RANGE = (3.0, 15.0)

GLEAM_VARIABLES = {
    "Ep": {"remote_subdir": "Ep", "out_col": "pet_mm"},
    "SMrz": {"remote_subdir": "SMrz", "out_col": "sm_root"},
}
GLEAM_VERSION = "v4.3a"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Load admin2 boundaries
# --------------------------------------------------------------------------

gdf = gpd.read_file(ADMIN2_SHP_PATH)
gdf = gdf[[ADMIN2_PCODE_FIELD, ADMIN2_NAME_FIELD, "geometry"]].reset_index(drop=True)
gdf["zone_id"] = gdf.index + 1
print(f"Loaded {gdf.shape[0]} admin2 zones.")

zone_id_to_pcode = dict(zip(gdf["zone_id"], gdf[ADMIN2_PCODE_FIELD]))

# --------------------------------------------------------------------------
# 2. SFTP download, one NetCDF per variable per year (verify-before-cache)
# --------------------------------------------------------------------------

with open(CREDENTIALS_PATH) as f:
    creds = json.load(f)


SOCKET_TIMEOUT_SECONDS = 60  # a silently-stalled transfer (server stops sending, connection
                              # stays open) hangs paramiko forever without this -- hit in practice


def get_sftp_client():
    sock = socket.create_connection((creds["host"], creds["port"]), timeout=SOCKET_TIMEOUT_SECONDS)
    sock.settimeout(SOCKET_TIMEOUT_SECONDS)
    transport = paramiko.Transport(sock)
    transport.connect(username=creds["username"], password=creds["password"])
    sftp_client = paramiko.SFTPClient.from_transport(transport)
    sftp_client.get_channel().settimeout(SOCKET_TIMEOUT_SECONDS)
    return sftp_client, transport


def download_year(var, year, max_retries=5):
    """Reconnects fresh on every attempt -- a stalled transfer leaves the
    whole paramiko connection unusable, not just that one file, so retrying
    on the same connection would just hang again."""
    filename = f"{var}_{year}_GLEAM_{GLEAM_VERSION}_MO.nc"
    remote_path = f"data/{GLEAM_VERSION}/monthly/{GLEAM_VARIABLES[var]['remote_subdir']}/{filename}"
    out_path = os.path.join(RAW_DIR, filename)

    if os.path.exists(out_path):
        try:
            with xr.open_dataset(out_path) as test_ds:
                pass
            print(f"Skipping {var} {year}, already downloaded and verified.")
            return out_path
        except Exception:
            print(f"Cached file for {var} {year} failed to open -- re-downloading.")
            os.remove(out_path)

    for attempt in range(1, max_retries + 1):
        sftp_conn = transport_conn = None
        try:
            print(f"Downloading {var} {year} (attempt {attempt}/{max_retries}) ...")
            sftp_conn, transport_conn = get_sftp_client()
            sftp_conn.get(remote_path, out_path)
            with xr.open_dataset(out_path) as test_ds:
                pass
            print(f"  saved and verified -> {out_path}")
            return out_path
        except Exception as exc:
            print(f"  attempt {attempt} failed: {exc}")
            if os.path.exists(out_path):
                os.remove(out_path)
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {var} {year} after {max_retries} attempts.") from exc
        finally:
            if sftp_conn is not None:
                sftp_conn.close()
            if transport_conn is not None:
                transport_conn.close()


year_files = {}
for var in GLEAM_VARIABLES:
    for year in range(START_YEAR, FEATURE_END_YEAR + 1):
        year_files[(var, year)] = download_year(var, year)

# --------------------------------------------------------------------------
# 3. Rasterize admin2 zones ONCE
# --------------------------------------------------------------------------

first_ds = xr.open_dataset(year_files[("Ep", START_YEAR)])

lat_slice = slice(max(ETHIOPIA_LAT_RANGE), min(ETHIOPIA_LAT_RANGE))  # GLEAM lat is descending already
lon_slice = slice(min(ETHIOPIA_LON_RANGE), max(ETHIOPIA_LON_RANGE))
first_ds = first_ds.sel(lat=lat_slice, lon=lon_slice)
first_ds = first_ds.sortby("lat", ascending=False)  # CRITICAL -- never assume storage order, see CLAUDE.md Sec 3.3

lats = first_ds["lat"].values
lons = first_ds["lon"].values
lat_res = abs(lats[1] - lats[0])
lon_res = abs(lons[1] - lons[0])
transform = from_origin(lons.min() - lon_res / 2, lats.max() + lat_res / 2, lon_res, lat_res)
grid_shape = (len(lats), len(lons))
first_ds.close()

shapes = list(zip(gdf.geometry, gdf["zone_id"]))
zone_raster = rasterize(
    shapes, out_shape=grid_shape, transform=transform, fill=0, all_touched=True, dtype="int32"
)
present_zone_ids = np.unique(zone_raster)
present_zone_ids = present_zone_ids[present_zone_ids != 0]

MIN_PIXELS_THRESHOLD = 10
pixel_counts = {zid: int(np.sum(zone_raster == zid)) for zid in gdf["zone_id"]}
zones_needing_fractional = [zid for zid, count in pixel_counts.items() if count < MIN_PIXELS_THRESHOLD]

if zones_needing_fractional:
    flagged_names = gdf[gdf["zone_id"].isin(zones_needing_fractional)][ADMIN2_NAME_FIELD].tolist()
    print(f"{len(zones_needing_fractional)} zones flagged for fractional-overlap treatment "
          f"(< {MIN_PIXELS_THRESHOLD} pixels under standard rasterization): {flagged_names}")

fractional_weights = {}
for zone_id in zones_needing_fractional:
    zone_geom = gdf.loc[gdf["zone_id"] == zone_id, "geometry"].values[0]
    weights = compute_pixel_weights(zone_geom, transform, grid_shape)
    fractional_weights[zone_id] = weights
    zone_name = gdf.loc[gdf["zone_id"] == zone_id, ADMIN2_NAME_FIELD].values[0]
    if weights:
        print(f"  {zone_name}: {len(weights)} pixels with fractional weights (sum={sum(weights.values()):.3f})")
    else:
        print(f"  WARNING: {zone_name} still has zero overlapping pixels -- check geometry validity.")

standard_zone_ids = np.array([zid for zid in present_zone_ids if zid not in fractional_weights])

# --------------------------------------------------------------------------
# 4. Extract zone-month means for every year, both variables
# --------------------------------------------------------------------------

all_rows = {}  # (pcode, year, month) -> {"pet_mm": ..., "sm_root": ...}

for var, var_cfg in GLEAM_VARIABLES.items():
    out_col = var_cfg["out_col"]
    print(f"Extracting {var} -> {out_col} ...")
    for year in range(START_YEAR, FEATURE_END_YEAR + 1):
        ds = xr.open_dataset(year_files[(var, year)])
        ds = ds.sel(lat=lat_slice, lon=lon_slice)
        ds = ds.sortby("lat", ascending=False)

        for t in ds.time.values:
            month = pd.Timestamp(t).month
            month_data = ds[var].sel(time=t).values

            if len(standard_zone_ids) > 0:
                valid = ~np.isnan(month_data)
                data_filled = np.where(valid, month_data, 0.0)
                sums = ndimage.sum(data_filled, labels=zone_raster, index=standard_zone_ids)
                counts = ndimage.sum(valid.astype(np.float64), labels=zone_raster, index=standard_zone_ids)
                means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
                for zone_id_val, mean_val in zip(standard_zone_ids, means):
                    key = (zone_id_to_pcode[zone_id_val], year, month)
                    all_rows.setdefault(key, {})[out_col] = mean_val

            for zone_id_val, weights in fractional_weights.items():
                if not weights:
                    continue
                weighted_val = weighted_value_nan_safe(month_data, weights)
                key = (zone_id_to_pcode[zone_id_val], year, month)
                all_rows.setdefault(key, {})[out_col] = weighted_val

        ds.close()

# --------------------------------------------------------------------------
# 5. Assemble and save
# --------------------------------------------------------------------------

records = []
for (pcode, year, month), vals in all_rows.items():
    row = {"pcode": pcode, "year": year, "month": month}
    row.update(vals)
    records.append(row)

raw_df = pd.DataFrame(records).sort_values(["pcode", "year", "month"]).reset_index(drop=True)
out_path = os.path.join(INTERIM_DIR, "gleam_admin2_monthly_raw.csv")
raw_df.to_csv(out_path, index=False)
print(f"\nSaved {raw_df.shape[0]} rows -> {out_path}")
print(f"Missing sm_root: {raw_df['sm_root'].isna().mean():.2%}, missing pet_mm: {raw_df['pet_mm'].isna().mean():.2%}")
