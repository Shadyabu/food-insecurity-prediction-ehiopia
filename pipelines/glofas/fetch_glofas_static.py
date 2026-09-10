"""
fetch_glofas_static.py

ACQUIRE stage (static reference layers). Three one-time, non-time-varying
GloFAS v4.0 grids, needed before any forecast can be turned into a
zone-level feature -- see docs/GloFAS_Pipeline_Documentation.md Sec 2-3 for
why these are required and why neither is "just another CHIRPS-style
raster download":

  1. Upstream area -- used to build the river-network pixel mask: only
     pixels with upstream area > NETWORK_THRESHOLD_KM2 (GloFAS's own
     documented "main network" cutoff, 1,000 km^2) represent a real river
     and are eligible for the zone join in
     compute_glofas_admin2_monthly_features.py.
  2. Flood-threshold return-period grids, 2-year and 20-year only (the
     5-year grid exists too but was deliberately dropped -- see the
     confirmed ensemble-collapse decision in
     docs/GloFAS_Pipeline_Documentation.md Sec 6). These are the static
     discharge thresholds a forecast ensemble member has to exceed to
     count as a "2-year"/"20-year" flood at a given pixel.

URLs confirmed live 2026-08-09 (curl HEAD returned HTTP 200,
content-type application/x-netcdf, no authentication) from the ECMWF
Confluence "Auxiliary Data" page
(https://confluence.ecmwf.int/display/CEMS/Auxiliary+Data) -- these are
GloFAS v4.0-specific links (there are separate EFAS versions on the same
page; do not substitute those). Files are large (upstream area ~87MB,
each threshold grid ~173MB) -- expect the first run to take a few minutes
depending on connection speed, and ~430MB of disk under data/raw/glofas/.
"""

import os

import requests
import xarray as xr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "glofas")
os.makedirs(RAW_DIR, exist_ok=True)

UPAREA_CACHE_PATH = os.path.join(RAW_DIR, "uparea_glofas_v4_0.nc")
RL2_CACHE_PATH = os.path.join(RAW_DIR, "flood_threshold_glofas_v4_rl_2.0.nc")
RL20_CACHE_PATH = os.path.join(RAW_DIR, "flood_threshold_glofas_v4_rl_20.0.nc")

_CONFLUENCE_BASE = "https://confluence.ecmwf.int/download/attachments/242067380"
UPAREA_URL = f"{_CONFLUENCE_BASE}/uparea_glofas_v4_0.nc"
RL2_URL = f"{_CONFLUENCE_BASE}/flood_threshold_glofas_v4_rl_2.0.nc"
RL20_URL = f"{_CONFLUENCE_BASE}/flood_threshold_glofas_v4_rl_20.0.nc"

# Ethiopia bounding box with a small buffer, degrees (N, W, S, E) -- matches
# the convention used for the CDS "area" request parameter in
# fetch_glofas_medium_range.py / fetch_glofas_seasonal.py. Derived from
# boundaries/eth_admbnda_adm2_csa_bofedb_2021.shp total bounds + 0.5 deg pad.
ETHIOPIA_BBOX_NWSE = (15.5, 32.5, 3.0, 48.5)

# GloFAS's own documented "main river network" cutoff.
NETWORK_THRESHOLD_KM2 = 1000.0


def verify_netcdf_opens(path):
    """Actually open the file, not just check it exists -- a truncated
    download still 'exists' on disk (CLAUDE.md Sec 3.6)."""
    try:
        with xr.open_dataset(path):
            return True
    except Exception as e:
        print(f"  WARNING: {path} failed to open: {e}")
        return False


def download_file(url, dest_path):
    tmp_path = dest_path + ".tmp"
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    os.replace(tmp_path, dest_path)  # atomic -- never leave a partial file in place


def fetch_static_grid(url, cache_path, label):
    if os.path.exists(cache_path) and verify_netcdf_opens(cache_path):
        print(f"{label} cache OK -> {cache_path}")
        return cache_path

    print(f"Downloading {label} from {url} ...")
    download_file(url, cache_path)
    if not verify_netcdf_opens(cache_path):
        raise RuntimeError(f"Downloaded {cache_path} but it failed to open -- truncated?")
    print(f"Cached -> {cache_path}")
    return cache_path


def build_network_mask(uparea_path, bbox_nwse=ETHIOPIA_BBOX_NWSE):
    """Returns a DataFrame of (lat, lon, upstream_area_km2) for pixels
    within the Ethiopia bbox that exceed NETWORK_THRESHOLD_KM2 -- the
    candidate set that compute_glofas_admin2_monthly_features.py then
    point-in-polygon joins to admin2 zones."""
    n, w, s, e = bbox_nwse
    with xr.open_dataset(uparea_path) as ds:
        var_name = list(ds.data_vars)[0]  # single-variable file; auto-detect rather
                                            # than hardcode a name not yet confirmed
        print(f"  upstream-area variable name: {var_name!r}, units: {ds[var_name].attrs.get('units')!r}")
        # Confirmed 2026-08-09 against a real downloaded file: dimensions are
        # "latitude"/"longitude", not the generic CF short names "lat"/"lon".
        da = ds[var_name].sel(latitude=slice(n, s), longitude=slice(w, e))
        units = str(ds[var_name].attrs.get("units", "")).lower()
        # Confirmed 2026-08-09: units attribute is "m2" (not "km2") -- the
        # divisor-by-1e6 branch is the one that actually applies.
        divisor = 1.0 if units.startswith("km") else 1e6
        network = da.where(da > (NETWORK_THRESHOLD_KM2 * (1.0 if units.startswith("km") else 1e6)))
        df = network.to_dataframe(name="upstream_area_raw").dropna().reset_index()
        df["upstream_area_km2"] = df["upstream_area_raw"] / divisor
        df = df.rename(columns={"latitude": "lat", "longitude": "lon"})  # keep this
            # script's own CSV output schema ("lat"/"lon") stable regardless of what
            # name the underlying source file happens to use
    print(f"Network pixels within Ethiopia bbox (>{NETWORK_THRESHOLD_KM2:.0f} km^2): {len(df)}")
    return df[["lat", "lon", "upstream_area_km2"]]


if __name__ == "__main__":
    uparea_path = fetch_static_grid(UPAREA_URL, UPAREA_CACHE_PATH, "upstream-area grid")
    fetch_static_grid(RL2_URL, RL2_CACHE_PATH, "2-yr flood-threshold grid")
    fetch_static_grid(RL20_URL, RL20_CACHE_PATH, "20-yr flood-threshold grid")

    network_df = build_network_mask(uparea_path)
    out_path = os.path.join(RAW_DIR, "glofas_network_pixels_ethiopia.csv")
    network_df.to_csv(out_path, index=False)
    print(f"Wrote {len(network_df)} candidate river-network pixels -> {out_path}")
