"""
fetch_ndvi_admin2.py

NDVI extraction against Ethiopia's fixed 92-zone admin2 boundary, following
Busker et al.'s methodology: NOAA STAR "Blended-VHP" smoothed NDVI (SM.SMN),
4km, weekly composites -> monthly mean, masked by JRC ASAP crop/rangeland
masks before zonal aggregation.

ACQUIRE stage. See docs/ndvi_pipeline_documentation.md for the full decision
record (product choice, historic-data-source tradeoff, masking approach).

--------------------------------------------------------------------------
WHY THIS SCRIPT IS MORE INVOLVED THAN CHIRPS'S fetch_chirps_admin2.py
--------------------------------------------------------------------------
NOAA STAR's own live bulk mirror only has loose per-week files back to
~2024; everything from 1981-2023 exists only as ANNUAL tar.gz archives
containing the full GLOBAL 4km grid (~7.4 GB/year -- confirmed via HTTP
HEAD on the 2010 archive). Pulling the full 2000-2021 baseline period that
way would mean ~160 GB of transfer, which is impractical here.

Resolution (confirmed with the user):
  - 2000-07/2022: reuse Busker et al.'s own released monthly NDVI grid
    (NDVI_NOA_STAR_1981_2022.nc, Zenodo 10.5281/zenodo.10853668, CC-BY 4.0)
    -- the exact same NOAA STAR product, already decompressed, cropped to
    the Horn of Africa, and aggregated to monthly by the original authors.
  - 08/2022-12/2023: NOAA's own live mirror has no loose per-week files for
    this window either, so this gap is closed by streaming just these two
    years' annual tar.gz archives directly from NOAA (~15 GB total, not
    ~160 GB) and cropping to Ethiopia's bounding box in memory, one weekly
    member at a time, without ever writing the global raster to disk.
  - 2024-2026: loose per-week GeoTIFFs, live-downloaded directly (small).

Crop/rangeland masks: Busker et al.'s exact bundled ASAP v02 "HAD" binary
masks (same Zenodo package) are reused rather than re-sourced from JRC
directly (JRC's own ASAP data-portal page 404'd when checked) -- this
guarantees the same mask vintage Busker used, for RQ1 comparability.
"""

import io
import os
import tarfile

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from remotezip import RemoteZip

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/ndvi/ -> pipelines/ -> repo root

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_NAME_FIELD = "ADM2_EN"
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "ndvi")
BUSKER_SOURCE_DIR = os.path.join(RAW_DIR, "busker_source")
ASAP_MASK_DIR = os.path.join(RAW_DIR, "asap_masks")
LIVE_WEEKLY_CACHE_DIR = os.path.join(RAW_DIR, "live_weekly")
INTERIM_DIR = os.path.join(REPO_ROOT, "data", "interim", "ndvi")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed", "features")

for d in [BUSKER_SOURCE_DIR, ASAP_MASK_DIR, LIVE_WEEKLY_CACHE_DIR, INTERIM_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

BUSKER_ZENODO_ZIP_URL = "https://zenodo.org/records/10853668/files/input_data.zip"
BUSKER_NDVI_ENTRY = "input_data/NDVI/NDVI_NOA_STAR_1981_2022.nc"
BUSKER_CROP_MASK_ENTRY = "input_data/NDVI/asap_mask_crop_v02_HAD_BINARY.tif"
BUSKER_RANGE_MASK_ENTRY = "input_data/NDVI/asap_mask_rangeland_v02_HAD_BINARY.tif"

NOAA_LIVE_BASE_URL = "https://www.star.nesdis.noaa.gov/pub/corp/scsb/wguo/data/Blended_VH_4km/geo_TIFF"
NOAA_SATELLITE_ID = "j01"  # JPSS-1, current operational VIIRS feed
GAP_ARCHIVE_YEARS = [2022, 2023]  # only years available solely as annual tar.gz, not loose weekly
LIVE_LOOSE_YEARS = [2024, 2025, 2026]  # available as loose per-week files

BUSKER_GRID_LAST_MONTH = pd.Timestamp("2022-07-01")  # last month in NDVI_NOA_STAR_1981_2022.nc

ETHIOPIA_LON_RANGE = (32.5, 48.5)
ETHIOPIA_LAT_RANGE = (3.0, 15.0)

FEATURE_START_YEAR = 2009
FEATURE_END_YEAR = 2026
BASELINE_START_YEAR = 2000  # matches Busker et al.'s stated 2000-2021 anomaly reference period
BASELINE_END_YEAR = 2021

MIN_PIXELS_THRESHOLD = 10  # same convention as CHIRPS: below this, use fractional overlap


# --------------------------------------------------------------------------
# Reusable helpers (same pattern as pipelines/chirps -- src/common/ is not
# yet populated in this repo, so these are kept local, matching how CHIRPS
# itself keeps them local rather than importing from an empty module).
# --------------------------------------------------------------------------


def verify_file_opens_tif(path):
    with rasterio.open(path):
        pass


def verify_file_opens_nc(path):
    with xr.open_dataset(path):
        pass


def compute_pixel_weights(zone_geom, transform, grid_shape, buffer_pixels=3):
    """Exact fractional area overlap between a polygon and nearby raster
    pixels. Returns {(row, col): weight}, weights summing to ~1.0."""
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
    """Area-weighted mean for one zone/month, excluding NaN pixels and
    renormalizing remaining weights."""
    pixel_vals = np.array([data_2d[r, c] for (r, c) in weights.keys()])
    pixel_weights = np.array(list(weights.values()))
    valid = ~np.isnan(pixel_vals)
    if valid.sum() == 0 or pixel_weights[valid].sum() == 0:
        return np.nan
    valid_weights = pixel_weights[valid]
    valid_weights = valid_weights / valid_weights.sum()
    return float(np.sum(pixel_vals[valid] * valid_weights))


def make_slice(coord_values, requested_range):
    lo, hi = min(requested_range), max(requested_range)
    if coord_values[0] > coord_values[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


# --------------------------------------------------------------------------
# 1. Acquire: Busker's raw NDVI grid + ASAP masks (via range-request
#    extraction from the Zenodo zip -- avoids downloading the full 415MB
#    archive when only ~210MB of it (the NDVI folder) is needed).
# --------------------------------------------------------------------------


def fetch_busker_zip_entry(entry_name, out_path, verify_fn):
    if os.path.exists(out_path):
        try:
            verify_fn(out_path)
            print(f"Skipping {os.path.basename(out_path)}, already downloaded and verified.")
            return out_path
        except Exception:
            print(f"Cached {os.path.basename(out_path)} failed to open -- re-extracting.")
            os.remove(out_path)

    print(f"Extracting {entry_name} from Busker et al. Zenodo package (range requests, no full-zip download)...")
    tmp_path = out_path + ".partial"
    with RemoteZip(BUSKER_ZENODO_ZIP_URL) as zf:
        with zf.open(entry_name) as src, open(tmp_path, "wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                dst.write(chunk)
    os.replace(tmp_path, out_path)
    verify_fn(out_path)
    print(f"  saved and verified -> {out_path}")
    return out_path


busker_ndvi_path = fetch_busker_zip_entry(
    BUSKER_NDVI_ENTRY, os.path.join(BUSKER_SOURCE_DIR, "NDVI_NOA_STAR_1981_2022.nc"), verify_file_opens_nc
)
crop_mask_path = fetch_busker_zip_entry(
    BUSKER_CROP_MASK_ENTRY, os.path.join(ASAP_MASK_DIR, "asap_mask_crop_v02_HAD_BINARY.tif"), verify_file_opens_tif
)
range_mask_path = fetch_busker_zip_entry(
    BUSKER_RANGE_MASK_ENTRY, os.path.join(ASAP_MASK_DIR, "asap_mask_rangeland_v02_HAD_BINARY.tif"), verify_file_opens_tif
)

# --------------------------------------------------------------------------
# 2. Acquire: live continuation, Aug 2022 - FEATURE_END_YEAR
# --------------------------------------------------------------------------


def download_bytes_with_retry(url, max_retries=3, timeout=(30, 300)):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as exc:
            print(f"  attempt {attempt} failed for {url}: {exc}")
            if attempt == max_retries:
                raise


def read_smn_cropped(tif_bytes, lon_range, lat_range):
    """Open a global SM.SMN GeoTIFF from bytes, windowed-read only the
    Ethiopia bounding box, and return (array, transform, crs). Never
    materializes the full global raster."""
    with MemoryFile(tif_bytes) as memfile:
        with memfile.open() as src:
            window = rasterio.windows.from_bounds(
                lon_range[0], lat_range[0], lon_range[1], lat_range[1], transform=src.transform
            )
            window = window.round_offsets().round_lengths()
            data = src.read(1, window=window)
            win_transform = src.window_transform(window)
            data = data.astype("float32")
            nodata = src.nodata
            if nodata is not None:
                data = np.where(data == nodata, np.nan, data)
            # These GeoTIFFs carry no nodata tag (confirmed via src.nodata is
            # None + src.tags()), but fill pixels are physically-impossible
            # -9999 sentinels (real NDVI is bounded in [-1, 1]) -- verified
            # against a live sample file before trusting this threshold.
            data = np.where(data < -1.0, np.nan, data)
            return data, win_transform, src.crs


def period_to_date(year, period):
    """NOAA VHP's 'www' filename field is a sequential period INDEX
    (1, 2, 3, ... up to 52/53), not a day-of-year -- confirmed by
    inspecting the live directory listing (P2024001, P2024002, ...
    sequential, not jumping by 7). Period p covers days
    [(p-1)*7+1, p*7] of the year; use the period's start day."""
    start_doy = (int(period) - 1) * 7 + 1
    return pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=start_doy - 1)


live_weekly_frames = []  # list of (date, array, transform, crs)

# --- 2a. Loose per-week files, 2024-2026 ---
for year in LIVE_LOOSE_YEARS:
    for period in range(1, 54):  # up to 53 weekly periods/year; missing ones 404 and are skipped
        date_val = period_to_date(year, period)
        if date_val > pd.Timestamp.today():
            continue
        fname = f"VHP.G04.C07.{NOAA_SATELLITE_ID}.P{year}{period:03d}.SM.SMN.tif"
        cache_path = os.path.join(LIVE_WEEKLY_CACHE_DIR, fname)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    tif_bytes = f.read()
                verify_file_opens_tif(cache_path)
            except Exception:
                os.remove(cache_path)
                tif_bytes = None
        else:
            tif_bytes = None

        if tif_bytes is None:
            url = f"{NOAA_LIVE_BASE_URL}/{fname}"
            try:
                tif_bytes = download_bytes_with_retry(url)
            except Exception:
                print(f"  skipping missing week {fname} (not published or download failed)")
                continue
            with open(cache_path, "wb") as f:
                f.write(tif_bytes)

        arr, win_transform, crs = read_smn_cropped(tif_bytes, ETHIOPIA_LON_RANGE, ETHIOPIA_LAT_RANGE)
        live_weekly_frames.append((date_val, arr, win_transform, crs))
    print(f"Loose weekly fetch: {year} done ({len(live_weekly_frames)} weeks so far)")

# --- 2b. Gap years only available as annual tar.gz, streamed and cropped ---


def stream_crop_annual_archive(year):
    url = f"{NOAA_LIVE_BASE_URL}/VHP.G04.C07_{year}.tar.gz"
    print(f"Streaming annual archive for {year} (large -- ~7.4GB transfer, cropped in memory, not written to disk)...")
    with requests.get(url, stream=True, timeout=(30, 600)) as r:
        r.raise_for_status()
        with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
            for member in tf:
                if not member.name.endswith(".SM.SMN.tif"):
                    continue
                fname = os.path.basename(member.name)
                parts = fname.split(".")
                # VHP.G04.C07.<sat>.P<yyyy><period>.SM.SMN.tif
                pfield = [p for p in parts if p.startswith("P") and len(p) == 8][0]
                yyyy, period = int(pfield[1:5]), int(pfield[5:8])
                date_val = period_to_date(yyyy, period)

                f = tf.extractfile(member)
                if f is None:
                    continue
                tif_bytes = f.read()
                arr, win_transform, crs = read_smn_cropped(tif_bytes, ETHIOPIA_LON_RANGE, ETHIOPIA_LAT_RANGE)
                live_weekly_frames.append((date_val, arr, win_transform, crs))
    print(f"  {year} archive processed.")


for year in GAP_ARCHIVE_YEARS:
    cache_marker = os.path.join(LIVE_WEEKLY_CACHE_DIR, f"_gap_{year}_cropped.npz")
    if os.path.exists(cache_marker):
        cached = np.load(cache_marker, allow_pickle=True)
        for date_str, arr in zip(cached["dates"], cached["arrays"]):
            live_weekly_frames.append((pd.Timestamp(str(date_str)), arr, None, None))
        print(f"Skipping {year}, already streamed and cached ({len(cached['dates'])} weeks).")
        continue

    before = len(live_weekly_frames)
    stream_crop_annual_archive(year)
    new_frames = live_weekly_frames[before:]
    np.savez_compressed(
        cache_marker,
        dates=np.array([str(d.date()) for d, *_ in new_frames]),
        arrays=np.array([a for _, a, _, _ in new_frames], dtype=object),
    )

print(f"\nTotal live weekly frames (2022 gap + 2024-2026 loose): {len(live_weekly_frames)}")

# All windowed reads share the same bounding box/resolution, so grid shape
# should be consistent; use the first frame's transform as the common grid.
common_transform = next(t for _, _, t, _ in live_weekly_frames if t is not None)
common_shape = next(a.shape for _, a, _, _ in live_weekly_frames)

live_df_rows = []
for date_val, arr, transform, crs in live_weekly_frames:
    if arr.shape != common_shape:
        continue  # skip any malformed/partial read rather than silently misaligning grids
    live_df_rows.append({"date": date_val, "array": arr})

live_weekly_df = pd.DataFrame(live_df_rows).sort_values("date").reset_index(drop=True)
live_weekly_df["year"] = live_weekly_df["date"].dt.year
live_weekly_df["month"] = live_weekly_df["date"].dt.month

live_monthly = {}
for (year, month), sub in live_weekly_df.groupby(["year", "month"]):
    stacked = np.stack(sub["array"].to_list(), axis=0).astype("float64")
    valid = ~np.isnan(stacked)
    counts = valid.sum(axis=0)
    sums = np.where(valid, stacked, 0.0).sum(axis=0)
    # manual NaN-safe mean (not np.nanmean): a pixel with zero valid weeks
    # across the whole month -- possible at record edges where only one
    # week was available and its Ethiopia window happened to be fully
    # nodata -- must resolve to NaN, not a division by zero.
    monthly_mean = np.divide(sums, counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    live_monthly[(year, month)] = monthly_mean.astype("float32")

print(f"Live portion aggregated to {len(live_monthly)} monthly grids.")

# --------------------------------------------------------------------------
# 3. Combine Busker's historic monthly grid with the live continuation onto
#    one common Ethiopia-cropped grid.
# --------------------------------------------------------------------------

busker_ds = xr.open_dataset(busker_ndvi_path)
busker_ds = busker_ds.sortby("latitude", ascending=False)  # CRITICAL -- verify lat order explicitly, per CLAUDE.md 3.3
lat_slice = make_slice(busker_ds["latitude"].values, ETHIOPIA_LAT_RANGE)
lon_slice = make_slice(busker_ds["longitude"].values, ETHIOPIA_LON_RANGE)
busker_ds = busker_ds.sel(latitude=lat_slice, longitude=lon_slice)

busker_lats = busker_ds["latitude"].values
busker_lons = busker_ds["longitude"].values
lat_res = float(np.median(np.abs(np.diff(busker_lats))))
lon_res = float(np.median(np.abs(np.diff(busker_lons))))
busker_transform = from_origin(busker_lons.min() - lon_res / 2, busker_lats.max() + lat_res / 2, lon_res, lat_res)
busker_grid_shape = (len(busker_lats), len(busker_lons))

# Resample every live monthly grid onto the exact Busker grid (nearest
# native resolution match: both are ~4km, Busker's grid is the fixed
# target since it defines the historic majority of the record).
resampled_live_monthly = {}
for (year, month), arr in live_monthly.items():
    dst = np.full(busker_grid_shape, np.nan, dtype="float32")
    reproject(
        source=arr,
        destination=dst,
        src_transform=common_transform,
        src_crs="EPSG:4326",
        dst_transform=busker_transform,
        dst_crs="EPSG:4326",
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )
    resampled_live_monthly[(year, month)] = dst

# Build one combined {(year, month): array} dict spanning the full record.
combined_monthly = {}
for t in busker_ds["time"].values:
    ts = pd.Timestamp(t)
    if ts > BUSKER_GRID_LAST_MONTH:
        continue
    arr = busker_ds["NDVI"].sel(time=t).values
    combined_monthly[(ts.year, ts.month)] = arr.astype("float32")
combined_monthly.update(resampled_live_monthly)

busker_ds.close()

print(f"Combined monthly grid record: {len(combined_monthly)} months "
      f"({min(combined_monthly)} to {max(combined_monthly)}).")

# --------------------------------------------------------------------------
# 4. Rasterize admin2 zones ONCE against the combined grid.
# --------------------------------------------------------------------------

gdf = gpd.read_file(ADMIN2_SHP_PATH)
gdf = gdf[[ADMIN2_PCODE_FIELD, ADMIN2_NAME_FIELD, "geometry"]].reset_index(drop=True)
gdf["zone_id"] = gdf.index + 1
zone_id_to_pcode = dict(zip(gdf["zone_id"], gdf[ADMIN2_PCODE_FIELD]))
print(f"Loaded {gdf.shape[0]} admin2 zones.")

from rasterio.features import rasterize

shapes = list(zip(gdf.geometry, gdf["zone_id"]))
zone_raster = rasterize(
    shapes, out_shape=busker_grid_shape, transform=busker_transform, fill=0, all_touched=True, dtype="int32"
)
present_zone_ids = np.unique(zone_raster)
present_zone_ids = present_zone_ids[present_zone_ids != 0]

pixel_counts = {zid: int(np.sum(zone_raster == zid)) for zid in gdf["zone_id"]}
zones_needing_fractional = [zid for zid, count in pixel_counts.items() if count < MIN_PIXELS_THRESHOLD]
if zones_needing_fractional:
    flagged_names = gdf[gdf["zone_id"].isin(zones_needing_fractional)][ADMIN2_NAME_FIELD].tolist()
    print(f"{len(zones_needing_fractional)} zones flagged for fractional-overlap treatment "
          f"(< {MIN_PIXELS_THRESHOLD} pixels): {flagged_names}")

zone_weights = {}
for zone_id in gdf["zone_id"]:
    if zone_id in zones_needing_fractional:
        zone_geom = gdf.loc[gdf["zone_id"] == zone_id, "geometry"].values[0]
        zone_weights[zone_id] = compute_pixel_weights(zone_geom, busker_transform, busker_grid_shape)
    else:
        rows, cols = np.where(zone_raster == zone_id)
        n = len(rows)
        zone_weights[zone_id] = {(r, c): 1.0 / n for r, c in zip(rows, cols)} if n > 0 else {}

# --------------------------------------------------------------------------
# 5. Resample ASAP crop/rangeland masks fractionally onto the NDVI grid.
#    Confirmed with the user: fractional area-weighted reconciliation, not
#    a hard threshold -- avoids an unjustifiable cutoff choice and matches
#    the fractional-overlap convention already used for small zones.
# --------------------------------------------------------------------------


def resample_mask_fraction(mask_path):
    with rasterio.open(mask_path) as src:
        src_data = src.read(1)
        dst = np.zeros(busker_grid_shape, dtype="float32")
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=busker_transform,
            dst_crs="EPSG:4326",
            resampling=Resampling.average,  # fractional coverage, not binary
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
    # A destination pixel gets NaN only where NO native ASAP pixel
    # contributed (fully outside the mask's classified domain) --
    # ~69% of pixels in practice. Treating that as "unknown -> propagate
    # NaN" is wrong: sum(weights.values()) over a dict containing even one
    # NaN returns NaN, which silently zeroes out an ENTIRE zone's
    # cropland/rangeland NDVI even when most of the zone has real
    # coverage (confirmed: this produced 67/92 zones 100%-missing here,
    # vs 17/92 in Busker et al.'s own released cropmask). "Not classified"
    # is treated as "not this land-cover class" (weight 0), matching the
    # NaN-safe weighting convention already used for pixel values.
    return np.where(np.isnan(dst), 0.0, dst)


crop_fraction_grid = resample_mask_fraction(crop_mask_path)
range_fraction_grid = resample_mask_fraction(range_mask_path)
np.save(os.path.join(INTERIM_DIR, "ndvi_grid_crop_fraction.npy"), crop_fraction_grid)
np.save(os.path.join(INTERIM_DIR, "ndvi_grid_range_fraction.npy"), range_fraction_grid)
print("Cropland/rangeland fractional coverage resampled onto NDVI grid and cached.")

# --------------------------------------------------------------------------
# 6. Zonal aggregation, per month, for general / cropland / rangeland NDVI.
# --------------------------------------------------------------------------

rows_out = []
for (year, month), arr in sorted(combined_monthly.items()):
    for zone_id, weights in zone_weights.items():
        if not weights:
            continue
        pcode = zone_id_to_pcode[zone_id]

        # General: plain NaN-safe area-weighted mean.
        ndvi_general = weighted_value_nan_safe(arr, weights)

        # Cropland / rangeland: same NaN-safe weighting, with an extra
        # per-pixel factor for land-cover fraction, renormalized jointly.
        crop_weighted = {k: w * crop_fraction_grid[k] for k, w in weights.items()}
        range_weighted = {k: w * range_fraction_grid[k] for k, w in weights.items()}
        ndvi_cropland = weighted_value_nan_safe(arr, crop_weighted) if sum(crop_weighted.values()) > 0 else np.nan
        ndvi_rangeland = weighted_value_nan_safe(arr, range_weighted) if sum(range_weighted.values()) > 0 else np.nan

        rows_out.append(
            {
                "pcode": pcode,
                "year": year,
                "month": month,
                "ndvi_raw": ndvi_general,
                "ndvi_cropland_raw": ndvi_cropland,
                "ndvi_rangeland_raw": ndvi_rangeland,
            }
        )

raw_monthly_df = pd.DataFrame(rows_out).sort_values(["pcode", "year", "month"]).reset_index(drop=True)

interim_path = os.path.join(INTERIM_DIR, "ndvi_admin2_monthly_raw.csv")
raw_monthly_df.to_csv(interim_path, index=False)
print(f"\nSaved raw (pre-anomaly) admin2 monthly NDVI: {raw_monthly_df.shape[0]} rows -> {interim_path}")
print(f"Zones: {raw_monthly_df['pcode'].nunique()}, "
      f"months: {raw_monthly_df[['year','month']].drop_duplicates().shape[0]}")
