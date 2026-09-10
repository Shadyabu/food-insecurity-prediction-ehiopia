"""
fetch_iri_cpc_seasonal.py

ACQUIRE stage: downloads IRI's seasonal precipitation tercile-probability
forecast for Ethiopia's bounding box, restricted to lead times L=1 and
L=3 months (the two this project uses -- see module docstring in
compute_iri_cpc_seasonal_admin2_monthly_features.py for why L6/L12 are
not built), from BOTH of IRI's two seasonal-forecast products:

  - LEGACY (two-tier system, 2.5 deg grid): F = Sep 1997 to Mar 2017.
    SOURCES/.IRI/.FD/.Seasonal_Forecast/.Precipitation/.prob
  - CURRENT (NMME-based, 1.0 deg grid): F = Feb 2017 onward (rolling).
    SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob

Both product identities, resolutions, and archive boundaries were
confirmed empirically 2026-08-09 by querying IRIDL directly with a real
access key -- not assumed from secondary documentation (which the login
wall blocks access to anyway). See
docs/IRI_CPC_Seasonal_Pipeline_Documentation.md Sec 2.

Requires IRI_API_KEY in .env -- see iridl_credentials.py.

Output: 4 small (~500-600KB) NetCDF files cached under
data/raw/iri_cpc_seasonal/. The LEGACY files are a closed historical
archive (Sep1997-Mar2017 will never change) and are cached normally, with
the verify-before-trust pattern used everywhere else in this project. The
CURRENT (NMME) files are a growing rolling archive (new F months appear
every month) -- these are always re-downloaded, not cache-skipped, since
the file is small (~600KB) and staleness risk outweighs the trivial
download cost.
"""

import math
import os

import requests
import xarray as xr

from iridl_credentials import get_iridl_cookie

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/iri_cpc_seasonal/ -> pipelines/ -> repo root

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "iri_cpc_seasonal")
os.makedirs(RAW_DIR, exist_ok=True)

# Same Ethiopia bounding box CHIRPS/NDVI/GLEAM already use, for consistency
# across every gridded pipeline in this project.
ETHIOPIA_LON_RANGE = (32.5, 48.5)
ETHIOPIA_LAT_RANGE = (3.0, 15.0)

LEAD_TIMES = [1, 3]  # model lead1/lead3 only -- confirmed decision, see compute script

LEGACY_BASE = "https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.Seasonal_Forecast/.Precipitation/.prob"
NMME_BASE = "https://iridl.ldeo.columbia.edu/SOURCES/.IRI/.FD/.NMME_Seasonal_Forecast/.Precipitation_ELR/.prob"

PRODUCTS = {
    "legacy": {"base_url": LEGACY_BASE, "cache": True},
    "nmme_elr": {"base_url": NMME_BASE, "cache": False},
}


def build_data_url(base_url, lead_time):
    lon_lo, lon_hi = ETHIOPIA_LON_RANGE
    lat_lo, lat_hi = ETHIOPIA_LAT_RANGE
    return (
        f"{base_url}/X/({lon_lo})/({lon_hi})/RANGEEDGES"
        f"/Y/({lat_lo})/({lat_hi})/RANGEEDGES"
        f"/L/({lead_time}.0)/({lead_time}.0)/RANGEEDGES"
        "/data.nc"
    )


def verify_nc_opens(path):
    """Actually open and check for the expected 'prob' variable -- not
    just that the file exists, matching every other pipeline's
    verify-before-trust rule (a truncated download still 'exists')."""
    with xr.open_dataset(path, decode_times=False) as ds:
        if "prob" not in ds.data_vars:
            raise ValueError(f"'prob' variable missing from {path}; found {list(ds.data_vars)}")
        if ds.sizes.get("F", 0) == 0:
            raise ValueError(f"Zero F (issue-month) records in {path} -- empty/failed download")


def download_product_lead(product_name, base_url, lead_time, use_cache, max_retries=3):
    out_path = os.path.join(RAW_DIR, f"{product_name}_lead{lead_time}.nc")

    if use_cache and os.path.exists(out_path):
        try:
            verify_nc_opens(out_path)
            print(f"Skipping {product_name} L={lead_time}, already downloaded and verified.")
            return out_path
        except Exception:
            print(f"Cached file for {product_name} L={lead_time} failed to open -- re-downloading.")
            os.remove(out_path)

    url = build_data_url(base_url, lead_time)
    cookies = get_iridl_cookie()

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Downloading {product_name} L={lead_time} (attempt {attempt}/{max_retries}) ...")
            with requests.get(url, cookies=cookies, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                # A login-wall redirect response would come back as an HTML
                # page, not NetCDF -- catch that explicitly rather than
                # caching an unusable file that "exists" but isn't data.
                content_type = r.headers.get("Content-Type", "")
                if "html" in content_type.lower():
                    raise RuntimeError(
                        "Received an HTML response instead of NetCDF -- IRI_API_KEY is "
                        "likely missing/invalid/expired. Regenerate at "
                        "https://iridl.ldeo.columbia.edu/auth/genkey"
                    )
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            verify_nc_opens(out_path)
            print(f"  saved and verified -> {out_path}")
            return out_path
        except Exception as exc:
            print(f"  attempt {attempt} failed: {exc}")
            if os.path.exists(out_path):
                os.remove(out_path)
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download {product_name} L={lead_time} after {max_retries} attempts."
                ) from exc


def decode_iridl_month(f_value):
    """IRIDL's F axis is 'months since 1960-01-01' under a 360-day pseudo
    calendar (12 months/year, no actual day count) -- xarray's CF decoder
    can't handle this (confirmed 2026-08-09: raises OutOfBoundsDatetime /
    'months since units only allowed for 360_day calendar'). Decode
    manually instead: floor(F) is a whole-month index from Jan 1960.
    Confirmed empirically against known values (F=685.5 -> Feb 2017,
    matching the NMME-ELR product's documented archive start)."""
    month_index = math.floor(f_value)
    year = 1960 + month_index // 12
    month = month_index % 12 + 1
    return year, month


if __name__ == "__main__":
    downloaded = {}
    for product_name, cfg in PRODUCTS.items():
        for lead_time in LEAD_TIMES:
            path = download_product_lead(
                product_name, cfg["base_url"], lead_time, use_cache=cfg["cache"]
            )
            downloaded[(product_name, lead_time)] = path

    print("\nDownload summary:")
    for (product_name, lead_time), path in downloaded.items():
        with xr.open_dataset(path, decode_times=False) as ds:
            f_vals = ds["F"].values
            y0, m0 = decode_iridl_month(float(f_vals[0]))
            y1, m1 = decode_iridl_month(float(f_vals[-1]))
            print(
                f"  {product_name} L={lead_time}: {len(f_vals)} issue-months, "
                f"{y0}-{m0:02d} to {y1}-{m1:02d}, grid {ds.sizes['Y']}x{ds.sizes['X']} -> {path}"
            )
