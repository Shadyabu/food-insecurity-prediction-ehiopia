"""
fetch_wvg_ersst.py

Acquisition stage for the Western V Gradient (WVG) pipeline.

SOURCING FINDING (confirmed 2026-08-09 -- see
docs/WVG_Pipeline_Documentation.md Sec 1 for the full investigation): no
maintained, ongoing public WVG time series exists. Funk et al. (2023)'s own
data release (Dryad doi:10.25349/D9MC8Z, "Figure 1E" tab of
TailoredForecastsDataRepository.xlsx) is a static one-time snapshot
covering 1981-2022 with no update mechanism, and CHC does not publish a
live WVG bulletin/index file. WVG must therefore be DERIVED from the raw
NOAA ERSSTv5 gridded SST product, following the exact box-difference
formula documented in Busker et al.'s own staged reference
(pipelines/teleconnections/busker_comparison/SST.xlsx, sheet "WVG" header
block) -- independently corroborated by the Dryad dataset's own methods
text (same formula, same season split). This is unlike every other
teleconnection in this project (IOD/MEI/NINO3.4), which are direct
fetch-and-validate pulls of an already-computed NOAA index.

This single netCDF file (~150MB) IS the full global monthly SST record
(1854-present, 2 degree grid) -- there is no per-year download to loop
over, unlike CHIRPS/NDVI/GLEAM. Land pixels are NaN (already decoded by
xarray from the file's fill-value convention -- confirmed by inspection,
not assumed).

SETUP
    pip install requests netCDF4 xarray --break-system-packages
    (this project's venv has these under /opt/miniconda3 -- run this
    pipeline's scripts with that interpreter, e.g.
    /opt/miniconda3/bin/python3 fetch_wvg_ersst.py)
"""

import os

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/wvg/ -> pipelines/ -> repo root
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "wvg")

URL = "https://downloads.psl.noaa.gov/Datasets/noaa.ersst.v5/sst.mnmean.nc"
OUT_PATH = os.path.join(RAW_DIR, "sst.mnmean.nc")
MIN_SIZE_BYTES = 100_000_000  # full file is ~150MB; a truncated download would be far smaller
MIN_TIME_STEPS = 2000  # ~2071 months (1854-01 to 2026-07) as of this pipeline's build date


def verify_file_opens(path):
    """Actually open the netCDF file and check it has a plausible number
    of time steps and the expected 'sst' variable -- a truncated download
    can still "exist" on disk and even pass a basic netCDF header check."""
    import xarray as xr

    with xr.open_dataset(path) as ds:
        if "sst" not in ds.variables:
            raise ValueError(f"{path}: no 'sst' variable found")
        n_time = ds.sizes.get("time", 0)
        if n_time < MIN_TIME_STEPS:
            raise ValueError(f"{path}: only {n_time} time steps, expected >= {MIN_TIME_STEPS}")


def download_source(max_retries=3):
    if os.path.exists(OUT_PATH):
        try:
            if os.path.getsize(OUT_PATH) < MIN_SIZE_BYTES:
                raise ValueError(f"cached file only {os.path.getsize(OUT_PATH)} bytes")
            verify_file_opens(OUT_PATH)
            print(f"cached file OK: {OUT_PATH}")
            return OUT_PATH
        except Exception as e:
            print(f"cached file failed verification ({e}), re-downloading")
            os.remove(OUT_PATH)

    os.makedirs(RAW_DIR, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(URL, timeout=180, stream=True) as resp:
                resp.raise_for_status()
                with open(OUT_PATH, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            if os.path.getsize(OUT_PATH) < MIN_SIZE_BYTES:
                raise ValueError(f"downloaded file only {os.path.getsize(OUT_PATH)} bytes")
            verify_file_opens(OUT_PATH)
            print(f"downloaded OK: {OUT_PATH}")
            return OUT_PATH
        except Exception as e:
            if os.path.exists(OUT_PATH):
                os.remove(OUT_PATH)  # never leave a partial file behind
            print(f"attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise

    return None


def main():
    download_source()


if __name__ == "__main__":
    main()
