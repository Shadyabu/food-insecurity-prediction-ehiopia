"""
fetch_chirps_admin2.py

CHIRPS extraction against Ethiopia's fixed 92-zone admin2 boundary,
following Busker et al.'s methodology (a single, stable spatial framework
for the whole study period, rather than FEWS NET's multi-vintage fsc_admin
system).

This reuses every validated fix from the fsc_admin pipeline:
  - Direct-to-PC via yearly NetCDF downloads (not thousands of daily files)
  - Auto-detected lat/lon ordering (the bug that caused the ~0.55 degree
    Harari pixel-offset -- see diagnose_harari_pixels.py history)
  - Leakage-safe, backward-looking dry-spell logic
  - SPI fit against a long historical baseline (1981+), only output for
    feature years

WHAT'S SIMPLER HERE THAN THE fsc_admin VERSION:
Only ONE set of boundaries, not six overlapping vintages -- so there's no
per-vintage rasterization, no cross-vintage pixel-overwrite risk, and no
naming-collision ambiguity. Rasterize once, use for the whole period.

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
pip install rasterio rasterstats geopandas pandas numpy scipy requests xarray netCDF4 --break-system-packages
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage, stats
from shapely.geometry import box


def compute_pixel_weights(zone_geom, transform, grid_shape, buffer_pixels=3):
    """Exact fractional area overlap between a polygon and nearby raster
    pixels -- used as a fallback for zones too small/oddly-shaped for
    binary all_touched rasterization to represent well. Returns
    {(row, col): weight}, weights summing to ~1.0."""
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


def weighted_value_nan_safe(day_data, weights):
    """Area-weighted mean for one zone/day, excluding any NaN (nodata)
    pixels and renormalizing remaining weights -- so a single bad pixel
    can't silently propagate NaN across an entire zone's time series."""
    pixel_vals = np.array([day_data[r, c] for (r, c) in weights.keys()])
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
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/chirps/ -> pipelines/ -> repo root

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_NAME_FIELD = "ADM2_EN"
ADMIN2_PCODE_FIELD = "ADM2_PCODE"  # use pcode as the stable join key, not name

START_YEAR = 1981  # matches SPI baseline needs from the start -- no separate
                    # baseline-only pass needed, since there's only one
                    # boundary set to rasterize regardless of year range
FEATURE_START_YEAR = 2009
FEATURE_END_YEAR = 2026

NETCDF_BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p05"
RAW_NC_DIR = os.path.join(REPO_ROOT, "data", "raw", "chirps")
INTERIM_DIR = os.path.join(REPO_ROOT, "data", "interim", "chirps")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed", "features")

ETHIOPIA_LON_RANGE = (32.5, 48.5)
ETHIOPIA_LAT_RANGE = (3.0, 15.0)

WET_DAY_THRESHOLD_MM = 1.0
DRY_SPELL_THRESHOLD_MM = 1.0
SPI_MIN_YEARS = 5
SPI_WINDOWS = [1, 3, 6, 12, 24]

os.makedirs(RAW_NC_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --------------------------------------------------------------------------
# 1. Load admin2 boundaries
# --------------------------------------------------------------------------

gdf = gpd.read_file(ADMIN2_SHP_PATH)
gdf = gdf[[ADMIN2_PCODE_FIELD, ADMIN2_NAME_FIELD, "geometry"]].reset_index(drop=True)
gdf["zone_id"] = gdf.index + 1
print(f"Loaded {gdf.shape[0]} admin2 zones.")

zone_id_to_pcode = dict(zip(gdf["zone_id"], gdf[ADMIN2_PCODE_FIELD]))

# --------------------------------------------------------------------------
# 2. Download each year's NetCDF (skip if already downloaded and valid)
# --------------------------------------------------------------------------


def download_year(year, max_retries=3):
    filename = f"chirps-v2.0.{year}.days_p05.nc"
    out_path = os.path.join(RAW_NC_DIR, filename)

    if os.path.exists(out_path):
        try:
            with xr.open_dataset(out_path) as test_ds:
                pass
            print(f"Skipping {year}, already downloaded and verified.")
            return out_path
        except Exception:
            print(f"Cached file for {year} failed to open -- re-downloading.")
            os.remove(out_path)

    url = f"{NETCDF_BASE_URL}/{filename}"
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {year} (attempt {attempt}/{max_retries}) ...")
            with requests.get(url, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            with xr.open_dataset(out_path) as test_ds:
                pass
            print(f"  saved and verified -> {out_path}")
            return out_path
        except Exception as exc:
            print(f"  attempt {attempt} failed: {exc}")
            if os.path.exists(out_path):
                os.remove(out_path)
            if attempt == max_retries:
                raise RuntimeError(f"Failed to download {year} after {max_retries} attempts.") from exc


year_files = {}
for year in range(START_YEAR, FEATURE_END_YEAR + 1):
    year_files[year] = download_year(year)

# --------------------------------------------------------------------------
# 3. Rasterize admin2 zones ONCE (no vintage complexity here)
# --------------------------------------------------------------------------

first_ds = xr.open_dataset(year_files[START_YEAR])

precip_var = None
for candidate in ["precip", "precipitation", "pcp"]:
    if candidate in first_ds.variables:
        precip_var = candidate
        break
if precip_var is None:
    raise ValueError(f"Couldn't find precip variable. Available: {list(first_ds.variables)}")

lat_name = "latitude" if "latitude" in first_ds.coords else "lat"
lon_name = "longitude" if "longitude" in first_ds.coords else "lon"


def make_slice(coord_values, requested_range):
    lo, hi = min(requested_range), max(requested_range)
    if coord_values[0] > coord_values[-1]:
        return slice(hi, lo)
    return slice(lo, hi)


lat_slice = make_slice(first_ds[lat_name].values, ETHIOPIA_LAT_RANGE)
lon_slice = make_slice(first_ds[lon_name].values, ETHIOPIA_LON_RANGE)
first_ds = first_ds.sel({lat_name: lat_slice, lon_name: lon_slice})
first_ds = first_ds.sortby(lat_name, ascending=False)  # CRITICAL -- see module docstring

lats = first_ds[lat_name].values
lons = first_ds[lon_name].values
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

# Identify zones needing fractional (area-weighted) treatment instead of
# the standard binary all_touched approach: zones with ZERO pixels (missed
# entirely), and zones with SUSPICIOUSLY FEW pixels (below MIN_PIXELS_THRESHOLD),
# where binary equal-weighting is likely to be unreliable. Confirmed via
# validation against Busker et al.: Dire Dawa urban (8 pixels) needed this
# treatment; the other ~91 admin2 zones validated well under the standard
# approach and don't need it. (Kilbati-Zone 2 has 1,086 pixels -- well above
# this threshold -- and is instead handled by the NaN-safe standard-path fix
# below, since its issue is a persistent-nodata pocket, not a small zone.)
MIN_PIXELS_THRESHOLD = 10

pixel_counts = {}
for zone_id in gdf["zone_id"]:
    pixel_counts[zone_id] = int(np.sum(zone_raster == zone_id))

zones_needing_fractional = [
    zid for zid, count in pixel_counts.items() if count < MIN_PIXELS_THRESHOLD
]

if zones_needing_fractional:
    flagged_names = gdf[gdf["zone_id"].isin(zones_needing_fractional)][ADMIN2_NAME_FIELD].tolist()
    print(f"{len(zones_needing_fractional)} zones flagged for fractional-overlap treatment "
          f"(< {MIN_PIXELS_THRESHOLD} pixels under standard rasterization): {flagged_names}")

# Precompute fractional pixel weights for flagged zones only (small list,
# cheap to compute once and reuse across every day of every year)
fractional_weights = {}
for zone_id in zones_needing_fractional:
    zone_geom = gdf.loc[gdf["zone_id"] == zone_id, "geometry"].values[0]
    weights = compute_pixel_weights(zone_geom, transform, grid_shape)
    fractional_weights[zone_id] = weights
    zone_name = gdf.loc[gdf["zone_id"] == zone_id, ADMIN2_NAME_FIELD].values[0]
    if weights:
        print(f"  {zone_name}: {len(weights)} pixels with fractional weights "
              f"(sum={sum(weights.values()):.3f})")
    else:
        print(f"  WARNING: {zone_name} still has zero overlapping pixels even with "
              f"fractional method -- check geometry validity manually.")

# Zones handled by the standard fast path (excludes flagged zones, which
# get the fractional treatment instead during extraction below)
standard_zone_ids = np.array([zid for zid in present_zone_ids if zid not in fractional_weights])

# --------------------------------------------------------------------------
# 4. Extract daily zone means for every year
# --------------------------------------------------------------------------

all_rows = []
for year in range(START_YEAR, FEATURE_END_YEAR + 1):
    print(f"Extracting {year} ...")
    ds = xr.open_dataset(year_files[year])
    ds = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    ds = ds.sortby(lat_name, ascending=False)

    for t in ds.time.values:
        date_val = pd.Timestamp(t)
        day_data = ds[precip_var].sel(time=t).values

        # Standard zones: fast vectorized mean via precomputed zone_raster.
        # NaN-safe: scipy.ndimage.mean() does NOT skip NaN, so a zone with even
        # one persistent-nodata pixel (e.g. Kilbati-Zone 2's Danakil Depression
        # gap) would otherwise return NaN for every day of its entire record.
        # Excludes NaN pixels with weight renormalization instead, matching the
        # same rule already applied in weighted_value_nan_safe() for small zones.
        if len(standard_zone_ids) > 0:
            valid = ~np.isnan(day_data)
            day_filled = np.where(valid, day_data, 0.0)
            sums = ndimage.sum(day_filled, labels=zone_raster, index=standard_zone_ids)
            counts = ndimage.sum(valid.astype(np.float64), labels=zone_raster, index=standard_zone_ids)
            means = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
            for zone_id_val, mean_val in zip(standard_zone_ids, means):
                all_rows.append(
                    {"pcode": zone_id_to_pcode[zone_id_val], "date": date_val, "precipitation_mm": mean_val}
                )

        # Flagged small zones: NaN-safe fractional-overlap weighting
        for zone_id_val, weights in fractional_weights.items():
            if not weights:
                continue
            weighted_val = weighted_value_nan_safe(day_data, weights)
            all_rows.append(
                {"pcode": zone_id_to_pcode[zone_id_val], "date": date_val, "precipitation_mm": weighted_val}
            )

    ds.close()

raw_df = pd.DataFrame(all_rows)
raw_df["date"] = pd.to_datetime(raw_df["date"])
raw_df = raw_df.sort_values(["pcode", "date"]).reset_index(drop=True)
raw_df["year"] = raw_df["date"].dt.year
raw_df["month"] = raw_df["date"].dt.month

daily_out_path = os.path.join(INTERIM_DIR, "chirps_admin2_daily_combined.csv")
raw_df.to_csv(daily_out_path, index=False)
print(f"\nSaved daily data: {raw_df.shape[0]} rows -> {daily_out_path}")

# --------------------------------------------------------------------------
# 5. Monthly features: total rainfall, wet days, dry spells
# --------------------------------------------------------------------------

monthly_total = raw_df.groupby(["pcode", "year", "month"])["precipitation_mm"].sum().rename("total_rainfall_mm")

raw_df["is_wet"] = raw_df["precipitation_mm"] > WET_DAY_THRESHOLD_MM
monthly_wetdays = raw_df.groupby(["pcode", "year", "month"])["is_wet"].sum().rename("wet_days")


def compute_running_dry_spell(sub, pcode_value):
    sub = sub.sort_values("date").reset_index(drop=True)
    is_dry = (sub["precipitation_mm"] <= DRY_SPELL_THRESHOLD_MM).to_numpy()
    wet_break = (~is_dry).cumsum()
    running_dry_days = pd.Series(is_dry).groupby(wet_break).cumcount().to_numpy() + 1
    running_dry_days = np.where(is_dry, running_dry_days, 0)
    sub["running_dry_days"] = running_dry_days
    sub["pcode"] = pcode_value
    return sub


print("Computing dry-spell runs ...")
dry_spell_pieces = []
for pcode_value, sub in raw_df.groupby("pcode"):
    dry_spell_pieces.append(compute_running_dry_spell(sub, pcode_value))
raw_df_with_spells = pd.concat(dry_spell_pieces, ignore_index=True)

monthly_max_dryspell = (
    raw_df_with_spells.groupby(["pcode", "year", "month"])["running_dry_days"].max().rename("max_dry_spell_length")
)

monthly_features_full = pd.concat([monthly_total, monthly_wetdays, monthly_max_dryspell], axis=1).reset_index()
monthly_features_full = monthly_features_full.sort_values(["pcode", "year", "month"]).reset_index(drop=True)

# --------------------------------------------------------------------------
# 6. SPI across all five accumulation windows
# --------------------------------------------------------------------------


def compute_spi_for_zone(zone_df, value_col, month_col="month"):
    zone_df = zone_df.copy()
    spi_col = f"spi_from_{value_col}"
    zone_df[spi_col] = np.nan
    for m in range(1, 13):
        mask = zone_df[month_col] == m
        vals = zone_df.loc[mask, value_col].to_numpy()
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < SPI_MIN_YEARS:
            continue
        valid_vals = vals[valid_mask]
        zero_frac = np.mean(valid_vals == 0)
        nonzero_vals = valid_vals[valid_vals > 0]
        if len(nonzero_vals) < 3:
            continue
        shape, loc, scale = stats.gamma.fit(nonzero_vals, floc=0)
        cdf = np.full_like(vals, np.nan, dtype=float)
        cdf[valid_mask] = np.where(
            valid_vals == 0, zero_frac,
            zero_frac + (1 - zero_frac) * stats.gamma.cdf(valid_vals, shape, loc=loc, scale=scale),
        )
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spi_vals = np.full_like(vals, np.nan, dtype=float)
        spi_vals[valid_mask] = stats.norm.ppf(cdf[valid_mask])
        zone_df.loc[mask, spi_col] = spi_vals
    return zone_df


print(f"Computing rolling accumulation windows: {SPI_WINDOWS} ...")
for window in SPI_WINDOWS:
    col_name = f"rainfall_sum_{window}m"
    monthly_features_full[col_name] = (
        monthly_features_full.groupby("pcode")["total_rainfall_mm"].transform(
            lambda s: s.rolling(window, min_periods=window).sum()
        )
    )

print("Fitting SPI per zone, per accumulation window ...")
for window in SPI_WINDOWS:
    value_col = f"rainfall_sum_{window}m"
    spi_pieces = []
    for pcode_value, zone_df in monthly_features_full.groupby("pcode"):
        spi_pieces.append(compute_spi_for_zone(zone_df, value_col=value_col))
    monthly_features_full = pd.concat(spi_pieces, ignore_index=True)
    monthly_features_full = monthly_features_full.rename(columns={f"spi_from_{value_col}": f"spi_{window}"})

# --------------------------------------------------------------------------
# 7. Restrict to feature years and save
# --------------------------------------------------------------------------

final_df = monthly_features_full[
    (monthly_features_full["year"] >= FEATURE_START_YEAR) & (monthly_features_full["year"] <= FEATURE_END_YEAR)
].copy()
final_df = final_df.sort_values(["pcode", "year", "month"]).reset_index(drop=True)

final_path = os.path.join(PROCESSED_DIR, "chirps_admin2_monthly.csv")
final_df.to_csv(final_path, index=False)

print(f"\nFinal monthly feature table: {final_df.shape[0]} rows ({FEATURE_START_YEAR}-{FEATURE_END_YEAR})")
print(f"Saved to {final_path}")
print(f"Columns: {list(final_df.columns)}")
