"""
compute_glofas_admin2_monthly_features.py

AGGREGATE + ENGINEER stage. Turns the cached raw NetCDF/GRIB forecasts from
fetch_glofas_medium_range.py / fetch_glofas_seasonal.py, plus the static
river-network pixel list and flood-threshold grid from
fetch_glofas_static.py, into the final zone-month feature table -- see
docs/GloFAS_Pipeline_Documentation.md Sec 3-7 for the full rationale behind
every decision reproduced here as code:

  1. River-network pixel -> admin2 zone join (Sec 3): point-in-polygon,
     same pattern/fallback as pipelines/locust's point_in_polygon_join.
     Zones with zero network pixels get glofas_reach_count = 0 and an
     explicit 0.0 exceedance value, not NaN -- a real physical fact about
     the zone, not a data gap.
  2. Per issue month, per network pixel, per ensemble member: does the
     member's max discharge across the LEADTIME sample set (not a dense
     daily window -- see below) exceed the 2-yr / 20-yr threshold at that
     pixel? (Sec 4, 6)
  3. Zone-month value = max across the zone's own network pixels and
     across the leadtime samples (Sec 3, 4) -- never mean.
  4. lead1 <- medium-range's target window = issue month + 1 month.
     lead3 <- seasonal's target window = issue month + 3 months. (Sec 5a)
  5. glofas_source_lead{1,3} flags operational vs. reforecast vs.
     no_reforecast_archive (Sec 5b) -- carried through so this boundary is
     visible downstream, not silently smoothed over.

REDESIGNED 2026-08-09 around fetch_glofas_medium_range.py / fetch_glofas_
seasonal.py's own redesign to bulk, date-bundled requests (see those
scripts' module docstrings for the full rationale -- EWDS's real
chokepoint is the number of separate request/download cycles, not
per-request size). The acquisition axes are now INVERTED from the
original per-issue-date design:

  OLD: one file per issue date, containing many leadtime/step values
       spanning that date's target-month window (a dense ~30-sample
       window).
  NEW: one file per LEADTIME value, containing many issue dates (as a
       "time" dimension) -- confirmed 2026-08-09 against a real manually-
       downloaded cems-glofas-seasonal-reforecast file: 'time' spans many
       calendar months, 'step' is a SCALAR (one fixed leadtime), not a
       dimension to select a window from.

This means the target-month window is no longer a dense daily sample --
it's now exactly LEN(LEADTIME_VALUES) = 6 discrete samples per product
(the specific hours chosen in each fetch script to land within the target
month across every possible issue date, per the precise min/max
computation documented in docs/GloFAS_Pipeline_Documentation.md). The
"max across the window" reduction (Sec 4) now reduces across these 6
samples instead of ~30-34 daily steps -- same operation, sparser input,
an explicit and accepted trade-off (a flood event confined between two
sample points could be missed for that month).

dataType handling (cf/pf/fc) and the lat/lon-vs-latitude/longitude
inconsistency across GloFAS's own files are unchanged from the previous
per-date design -- see _open_leadtime_chunk_discharge()'s docstring.
"""

import os
import re
from collections import OrderedDict

import geopandas as gpd
import pandas as pd
import xarray as xr
from datetime import date
from shapely.geometry import Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "glofas")
MEDIUM_RANGE_DIR = os.path.join(RAW_DIR, "medium_range")
SEASONAL_DIR = os.path.join(RAW_DIR, "seasonal")
NETWORK_PIXELS_CSV = os.path.join(RAW_DIR, "glofas_network_pixels_ethiopia.csv")
RL2_NC = os.path.join(RAW_DIR, "flood_threshold_glofas_v4_rl_2.0.nc")   # matches fetch_glofas_static.py's cache paths
RL20_NC = os.path.join(RAW_DIR, "flood_threshold_glofas_v4_rl_20.0.nc")

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "glofas_admin2_monthly.csv")

EQUAL_AREA_CRS = "ESRI:102022"  # matches pipelines/locust -- Africa Albers Equal Area Conic
SNAP_TOLERANCE_KM = 5.0

FEATURE_START_YEAR = 2009
DIS_VAR = "dis24"
MEMBER_DIM = "number"

# Must match fetch_glofas_medium_range.py / fetch_glofas_seasonal.py's own
# LEADTIME_HOURS exactly (as ints, matching what the filename regex below
# parses out) -- these two scripts are the source of truth; duplicated
# here rather than imported to keep each fetch script runnable standalone
# without needing the others on the path.
MEDIUM_RANGE_LEADTIMES = [24, 192, 360, 528, 696, 720]  # 720 not 816 -- corrected
    # 2026-08-12, operational's true max leadtime is 720h (744-816h all rejected live)
SEASONAL_LEADTIMES = [1920, 2088, 2256, 2424, 2592, 2712]
MEDIUM_RANGE_OPERATIONAL_START = date(2020, 11, 1)  # corrected 2026-08-12 (was
    # 2019-11-05 -- that date is when GloFAS v2.1 launched, not this query
    # pattern's real archive start; confirmed live, see fetch_glofas_medium_range.py)
SEASONAL_OPERATIONAL_START = date(2021, 1, 1)  # corrected 2026-08-12 (was 2020-12-01) --
    # pure reclassification, Dec 2020 already downloaded via reforecast with no gaps
SEASONAL_ISSUE_DAY = 10


def load_admin2():
    return gpd.read_file(ADMIN2_SHP_PATH)


def point_in_polygon_join(points_df, gdf_admin2, lon_col="lon", lat_col="lat"):
    """Same pattern as pipelines/locust/compute_locust_admin2_monthly_features.py
    -- point-in-polygon with a nearest-zone snap fallback for pixels that
    land just outside every polygon at a boundary."""
    geom = gpd.GeoSeries(
        [Point(xy) for xy in zip(points_df[lon_col], points_df[lat_col])], crs="EPSG:4326"
    )
    pts_gdf = gpd.GeoDataFrame(points_df.copy(), geometry=geom, crs="EPSG:4326")

    joined = gpd.sjoin(pts_gdf, gdf_admin2[[ADMIN2_PCODE_FIELD, "geometry"]], how="left", predicate="within")
    joined = joined.drop(columns=["index_right"])

    missing_mask = joined[ADMIN2_PCODE_FIELD].isna()
    n_missing = missing_mask.sum()
    if n_missing > 0:
        gdf_proj = gdf_admin2.to_crs(EQUAL_AREA_CRS)
        missing_pts = pts_gdf.loc[missing_mask].to_crs(EQUAL_AREA_CRS)
        nearest = gpd.sjoin_nearest(
            missing_pts, gdf_proj[[ADMIN2_PCODE_FIELD, "geometry"]], how="left", distance_col="dist_m"
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        within_tol = nearest["dist_m"] <= (SNAP_TOLERANCE_KM * 1000)
        snapped = nearest.loc[within_tol, ADMIN2_PCODE_FIELD]
        joined.loc[snapped.index, ADMIN2_PCODE_FIELD] = snapped.values
        print(f"  {n_missing} network pixels outside every polygon: "
              f"{within_tol.sum()} snapped, {n_missing - within_tol.sum()} dropped.")

    joined = joined[joined[ADMIN2_PCODE_FIELD].notna()]
    return joined.rename(columns={ADMIN2_PCODE_FIELD: "zone_code"})


def load_network_pixels_with_zone(gdf_admin2):
    pixels = pd.read_csv(NETWORK_PIXELS_CSV)
    joined = point_in_polygon_join(pixels, gdf_admin2)
    return joined[["zone_code", "lat", "lon"]].reset_index(drop=True)


def _latlon_dim_names(da):
    """GloFAS's own products don't agree on this: the upstream-area file
    uses "latitude"/"longitude" but the threshold files use "lat"/"lon"
    (confirmed 2026-08-09 against real downloaded files). Detect per-file
    rather than hardcode either."""
    lat_name = "latitude" if "latitude" in da.dims else "lat"
    lon_name = "longitude" if "longitude" in da.dims else "lon"
    return lat_name, lon_name


def _sample_threshold(nc_path, pixel_df):
    """Each threshold file is single-variable -- auto-detect the name
    rather than hardcode one not yet confirmed against a real file."""
    with xr.open_dataset(nc_path) as ds:
        var_name = list(ds.data_vars)[0]
        da = ds[var_name]
        lat_name, lon_name = _latlon_dim_names(da)
        return [float(da.sel(**{lat_name: row.lat, lon_name: row.lon}, method="nearest").values)
                for row in pixel_df.itertuples()]


def attach_thresholds(pixel_df):
    pixel_df = pixel_df.copy()
    pixel_df["rl2"] = _sample_threshold(RL2_NC, pixel_df)
    pixel_df["rl20"] = _sample_threshold(RL20_NC, pixel_df)
    return pixel_df


# --------------------------------------------------------------------------
# Leadtime-chunk file discovery and discharge extraction
# --------------------------------------------------------------------------

CHUNK_FNAME_RE = re.compile(
    r"^glofas_(?:mediumrange|seasonal)_(operational|reforecast)_leadtime(\d+)_chunk(\d+)\.grib2$"
)


def discover_leadtime_chunks(directory):
    """Returns {(era, leadtime_hours): [chunk file paths]} -- each chunk
    file covers ONE leadtime value across a sub-range of years (<=6 for
    reforecast, all years in one chunk for operational -- see
    fetch_glofas_medium_range.py's build_*_chunks())."""
    groups = {}
    if not os.path.isdir(directory):
        return groups
    for fname in sorted(os.listdir(directory)):
        m = CHUNK_FNAME_RE.match(fname)
        if not m:
            continue
        era, leadtime_str, _chunk_str = m.groups()
        key = (era, int(leadtime_str))
        groups.setdefault(key, []).append(os.path.join(directory, fname))
    return groups


_NPTS_RE = re.compile(r"'numberOfPoints': (\d+)")


def _open_discharge_for_dataType(nc_path, data_type):
    """Returns {numberOfPoints: DataArray} for one dataType in one file --
    normally a single entry. A handful of operational chunks (one per
    leadtime, both medium_range and seasonal, always the chunk spanning
    2022-2023) straddle GloFAS's real 2023-07 operational grid-resolution
    upgrade: cfgrib raises its own "multiple values for unique key:
    numberOfPoints" ambiguity error, and the two numberOfPoints values
    resolve to disjoint, non-overlapping time ranges (0.1deg/20000pts
    through 2023-06, 0.05deg/80000pts from 2023-07) -- confirmed 2026-08-16
    directly against the affected files, a genuine external product-vintage
    boundary, not corrupted data. See docs/GloFAS_Pipeline_Documentation.md
    Sec 5e."""
    base_keys = {"dataType": data_type}
    try:
        with xr.open_dataset(
            nc_path, engine="cfgrib",
            backend_kwargs={"filter_by_keys": base_keys, "indexpath": ""},
        ) as ds:
            if DIS_VAR not in ds.variables:
                return {}
            da = ds[DIS_VAR].load()
            lat_name, lon_name = _latlon_dim_names(da)
            npts = da.sizes[lat_name] * da.sizes[lon_name]
            return {npts: da}
    except Exception as e:
        npts_values = {int(n) for n in _NPTS_RE.findall(str(e))}
        if not npts_values:
            return {}  # this dataType genuinely isn't present -- fine
        out = {}
        for npts in npts_values:
            try:
                with xr.open_dataset(
                    nc_path, engine="cfgrib",
                    backend_kwargs={**{"filter_by_keys": {**base_keys, "numberOfPoints": npts}}, "indexpath": ""},
                ) as ds:
                    if DIS_VAR in ds.variables:
                        out[npts] = ds[DIS_VAR].load()
            except Exception:
                pass
        return out


def _open_leadtime_chunk_discharge(nc_path):
    """Opens one leadtime-chunk file (many 'time' entries = issue dates,
    ONE fixed leadtime/step value, many ensemble members). Same dataType
    handling as the pipeline's earlier per-date design -- GloFAS forecast
    GRIB2 files can mix multiple dataType labels that cfgrib can't merge
    in one open() call ("multiple values for unique key"). Confirmed
    2026-08-09 against real downloaded files that THREE different labels
    can appear:
      - dataType='cf' (control forecast, medium-range only) -- no
        ensemble-member dimension of its own; assigned member id 0 here.
      - dataType='pf' (perturbed forecast) -- has its own 'number'
        dimension that concatenates cleanly after cf's [0] (confirmed, no
        renumbering needed).
      - dataType='fc' (plain "forecast", seen in a manually downloaded
        cems-glofas-seasonal-reforecast file) -- ALSO carries its own
        'number' dimension (25 members observed), behaving like 'pf'
        despite the different label. Detected by checking for an existing
        MEMBER_DIM after opening, not by assuming the label implies shape.
    Unlike the old per-date design, there is no STEP_DIM to select a
    window from here -- 'step' is a scalar (confirmed 2026-08-09), since
    each file only ever contains one leadtime value.

    Returns a LIST of DataArrays, one per distinct grid resolution found
    in the file (normally length 1 -- see _open_discharge_for_dataType for
    the one real exception). Groups are never concatenated across
    resolutions (different lat/lon grids entirely); cf/pf/fc are merged
    on MEMBER_DIM only within the same resolution group, so an ensemble's
    control + perturbed members always stay together. Downstream,
    _extract_issue_month_slice searches every group in the list by time,
    the same way it already searches across multiple chunk files, and
    nearest-neighbor pixel sampling in target_window_exceedance() handles
    either resolution transparently."""
    by_npts = {}
    for data_type in ("cf", "pf", "fc"):
        for npts, da in _open_discharge_for_dataType(nc_path, data_type).items():
            if MEMBER_DIM not in da.dims:
                da = da.expand_dims(MEMBER_DIM).assign_coords({MEMBER_DIM: [0]})
            by_npts.setdefault(npts, []).append(da)

    if not by_npts:
        raise RuntimeError(f"{nc_path}: no readable dataType='cf'/'pf'/'fc' discharge messages found")
    return [das[0] if len(das) == 1 else xr.concat(das, dim=MEMBER_DIM) for das in by_npts.values()]


# Memory safety (2026-08-16): ~24GB of raw GRIB2 now on disk across
# medium_range + seasonal chunks, on a 24GB-RAM machine, decompressing
# larger than on-disk size once loaded as float32 arrays. Two things
# together keep this bounded:
#   1. A monotonic search cursor per (era, leadtime) chunk-file list below
#      -- issue months are processed chronologically in collect_*_rows(),
#      and each key's chunk_paths are chronological sub-ranges, so once a
#      match is found at index i, no earlier issue month (and therefore no
#      chunk before i) will ever be looked up again for that key. Without
#      this, _extract_issue_month_slice rescanned chunk_paths from index 0
#      on every call, re-touching (and re-caching) every earlier chunk file
#      forever -- by the end of a run that amounts to loading the entire
#      dataset for that key into memory regardless of any cache eviction
#      policy, since the "stale" prefix keeps getting marked recently-used.
#   2. A total-bytes-capped LRU on top, as a backstop for the remaining
#      (now much smaller) working set -- e.g. a genuine multi-month gap
#      that forces a jump across several chunks in one call.
_CHUNK_CACHE = OrderedDict()  # path -> list of loaded DataArrays (one per grid
                                # resolution present, normally length 1), LRU order
_CHUNK_CACHE_MAX_BYTES = 6 * 1024 ** 3  # well under 24GB total RAM
_CHUNK_CACHE_BYTES = 0

_CHUNK_SEARCH_CURSOR = {}  # tuple(chunk_paths) -> last matched index


def _group_bytes(groups):
    return sum(int(da.nbytes) for da in groups)


def _get_cached_chunk(path):
    global _CHUNK_CACHE_BYTES
    if path in _CHUNK_CACHE:
        _CHUNK_CACHE.move_to_end(path)
        return _CHUNK_CACHE[path]
    groups = _open_leadtime_chunk_discharge(path)
    nbytes = _group_bytes(groups)
    while _CHUNK_CACHE and _CHUNK_CACHE_BYTES + nbytes > _CHUNK_CACHE_MAX_BYTES:
        _, evicted = _CHUNK_CACHE.popitem(last=False)
        _CHUNK_CACHE_BYTES -= _group_bytes(evicted)
    _CHUNK_CACHE[path] = groups
    _CHUNK_CACHE_BYTES += nbytes
    return groups


def _extract_issue_month_slice(chunk_paths, issue_period):
    """Searches the given chunk files (all covering the SAME leadtime,
    different year sub-ranges, chronologically ordered) for the 'time'
    entry matching issue_period. Returns a DataArray (member, lat, lon), or
    None if not found -- either because it's genuinely outside the
    reforecast archive (Sec 5b's no_reforecast_archive case) or a
    mixed-validity gap (see fetch_glofas_medium_range.py's
    report_time_completeness()).

    Resumes from the last matched index for this chunk_paths list rather
    than rescanning from 0 -- see the memory-safety note above
    _CHUNK_CACHE. Safe because collect_medium_range_rows/collect_seasonal_
    rows call this with issue_period strictly increasing per key. Within a
    single chunk file, checks every grid-resolution group returned by
    _get_cached_chunk (normally just one -- see _open_leadtime_chunk_
    discharge for the 2023-07 resolution-upgrade exception)."""
    cursor_key = tuple(chunk_paths)
    start = _CHUNK_SEARCH_CURSOR.get(cursor_key, 0)
    for offset, path in enumerate(chunk_paths[start:]):
        idx = start + offset
        for da in _get_cached_chunk(path):
            time_key = "time" if "time" in da.coords else ("hdate" if "hdate" in da.coords else None)
            if time_key is None:
                continue
            times = pd.to_datetime(da[time_key].values)
            mask = (times.year == issue_period.year) & (times.month == issue_period.month)
            if mask.any():
                _CHUNK_SEARCH_CURSOR[cursor_key] = idx
                return da.isel({time_key: int(mask.nonzero()[0][0])})
    return None


def target_window_exceedance(era, glofas_dir, leadtime_values, pixel_df, issue_period):
    """For a given issue month, gathers ONE discharge sample per leadtime
    value (from that leadtime's chunk file(s)) -- the sparse equivalent of
    the old dense daily-step window (Sec 4) -- takes the max across those
    samples, then computes per-pixel fraction of ensemble members exceeding
    rl2/rl20, then the max across the zone's own pixels (Sec 3). Returns a
    dict {zone_code: (frac_exceed_2yr, frac_exceed_20yr)}."""
    chunks = discover_leadtime_chunks(glofas_dir)
    samples = []
    for lt in leadtime_values:
        key = (era, lt)
        if key not in chunks:
            continue
        da_slice = _extract_issue_month_slice(chunks[key], issue_period)
        if da_slice is not None:
            samples.append(da_slice)
    if not samples:
        return {}

    lat_name, lon_name = _latlon_dim_names(samples[0])
    da_stack = xr.concat(samples, dim="sample")  # (sample, member, lat, lon)
    member_max = da_stack.max(dim="sample")  # (member, lat, lon)

    zone_results = {}
    for zone_code, group in pixel_df.groupby("zone_code"):
        pixel_frac_2yr, pixel_frac_20yr = [], []
        for row in group.itertuples():
            pixel_vals = member_max.sel(**{lat_name: row.lat, lon_name: row.lon}, method="nearest").values
            n_members = pixel_vals.size
            if n_members == 0:
                continue
            pixel_frac_2yr.append(float((pixel_vals > row.rl2).sum()) / n_members)
            pixel_frac_20yr.append(float((pixel_vals > row.rl20).sum()) / n_members)
        if pixel_frac_2yr:
            zone_results[zone_code] = (max(pixel_frac_2yr), max(pixel_frac_20yr))
    return zone_results


def _month_end(year, month):
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1)


def _all_months():
    today = pd.Timestamp.today()
    return pd.period_range(f"{FEATURE_START_YEAR}-01", today.strftime("%Y-%m"), freq="M")


def collect_medium_range_rows(pixel_df, all_zone_codes):
    rows = []
    if not os.path.isdir(MEDIUM_RANGE_DIR):
        print(f"WARNING: {MEDIUM_RANGE_DIR} does not exist -- run fetch_glofas_medium_range.py first.")
        return pd.DataFrame(columns=["zone_code", "month", "glofas_exceed_2yr_lead1",
                                      "glofas_exceed_20yr_lead1", "glofas_source_lead1"])

    for period in _all_months():
        era = "reforecast" if _month_end(period.year, period.month) < MEDIUM_RANGE_OPERATIONAL_START else "operational"
        zone_results = target_window_exceedance(
            era, MEDIUM_RANGE_DIR, MEDIUM_RANGE_LEADTIMES, pixel_df, period
        )
        for zone_code in all_zone_codes:
            frac2, frac20 = zone_results.get(zone_code, (0.0, 0.0))
            rows.append({"zone_code": zone_code, "month": period.to_timestamp(),
                         "glofas_exceed_2yr_lead1": frac2, "glofas_exceed_20yr_lead1": frac20,
                         "glofas_source_lead1": era if zone_results else None})
    return pd.DataFrame(rows)


def collect_seasonal_rows(pixel_df, all_zone_codes):
    rows = []
    if not os.path.isdir(SEASONAL_DIR):
        print(f"WARNING: {SEASONAL_DIR} does not exist -- run fetch_glofas_seasonal.py first.")
        return pd.DataFrame(columns=["zone_code", "month", "glofas_exceed_2yr_lead3",
                                      "glofas_exceed_20yr_lead3", "glofas_source_lead3"])

    for period in _all_months():
        era = "reforecast" if date(period.year, period.month, SEASONAL_ISSUE_DAY) < SEASONAL_OPERATIONAL_START else "operational"
        zone_results = target_window_exceedance(
            era, SEASONAL_DIR, SEASONAL_LEADTIMES, pixel_df, period
        )
        for zone_code in all_zone_codes:
            frac2, frac20 = zone_results.get(zone_code, (0.0, 0.0))
            rows.append({"zone_code": zone_code, "month": period.to_timestamp(),
                         "glofas_exceed_2yr_lead3": frac2, "glofas_exceed_20yr_lead3": frac20,
                         "glofas_source_lead3": era if zone_results else None})
    return pd.DataFrame(rows)


def main():
    gdf_admin2 = load_admin2()
    all_zone_codes = gdf_admin2[ADMIN2_PCODE_FIELD].tolist()

    print("Joining river-network pixels to admin2 zones...")
    pixel_df = load_network_pixels_with_zone(gdf_admin2)
    pixel_df = attach_thresholds(pixel_df)

    reach_count = pixel_df.groupby("zone_code").size().reindex(all_zone_codes, fill_value=0)
    reach_count.name = "glofas_reach_count"
    zero_reach_zones = reach_count[reach_count == 0].index.tolist()
    print(f"Zones with zero river-network pixels (expected for dry lowland zones): "
          f"{len(zero_reach_zones)} / {len(all_zone_codes)}")

    print("\nProcessing medium-range (lead1) forecast files...")
    lead1_df = collect_medium_range_rows(pixel_df, all_zone_codes)
    print("\nProcessing seasonal (lead3) forecast files...")
    lead3_df = collect_seasonal_rows(pixel_df, all_zone_codes)

    all_months = _all_months().to_timestamp()
    scaffold = pd.MultiIndex.from_product([all_zone_codes, all_months], names=["zone_code", "month"]).to_frame(index=False)

    out = scaffold.merge(lead1_df, on=["zone_code", "month"], how="left")
    out = out.merge(lead3_df, on=["zone_code", "month"], how="left")
    out = out.merge(reach_count.reset_index().rename(columns={"index": "zone_code"}), on="zone_code", how="left")

    # Zones with no river-network pixel: real physical zero, not a gap (Sec 3).
    zero_mask = out["glofas_reach_count"] == 0
    out.loc[zero_mask, ["glofas_exceed_2yr_lead1", "glofas_exceed_20yr_lead1"]] = 0.0
    out.loc[zero_mask, ["glofas_exceed_2yr_lead3", "glofas_exceed_20yr_lead3"]] = 0.0
    out.loc[zero_mask & out["glofas_source_lead1"].isna(), "glofas_source_lead1"] = "no_river_network"
    out.loc[zero_mask & out["glofas_source_lead3"].isna(), "glofas_source_lead3"] = "no_river_network"

    # Any remaining missing rows (zone has a river, but no cached chunk file
    # covers that issue month at all) are a genuine pre-archive/fetch gap --
    # explicit NaN + flagged source, not an implied zero (Sec 5b, matching
    # locust's no-source-gap convention).
    out["glofas_source_lead1"] = out["glofas_source_lead1"].fillna("no_reforecast_archive")
    out["glofas_source_lead3"] = out["glofas_source_lead3"].fillna("no_reforecast_archive")

    out = out[["zone_code", "month",
               "glofas_exceed_2yr_lead1", "glofas_exceed_20yr_lead1", "glofas_source_lead1",
               "glofas_exceed_2yr_lead3", "glofas_exceed_20yr_lead3", "glofas_source_lead3",
               "glofas_reach_count"]]
    out = out.sort_values(["zone_code", "month"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(out)} zone-month rows -> {OUTPUT_PATH}")
    print(f"Zones: {out['zone_code'].nunique()} (expected {len(all_zone_codes)})")
    print(out["glofas_source_lead1"].value_counts())
    print(out["glofas_source_lead3"].value_counts())


if __name__ == "__main__":
    main()
