"""
fetch_teleconnections.py

Acquisition stage for the three climate teleconnection indices: IOD, MEI,
NINO3.4. Unlike every other pipeline in this project, there is no spatial
dimension here -- each source is a single global monthly value, not a grid
or a point layer, so there is no rasterization/zonal-aggregation stage at
all (Stage 2 in the usual Acquire->Aggregate->Engineer->Validate shape
collapses to a no-op). See docs/Teleconnections_Pipeline_Documentation.md
for the full source-selection reasoning; summary:

  - IOD  = NOAA PSL's Dipole Mode Index (DMI), HadISST-based long record.
           https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data
           Matches Busker et al.'s staged reference exactly (1870-01 in
           both = -0.438).
  - MEI  = NOAA PSL's MEI.v2 (NOT the discontinued original MEI).
           https://psl.noaa.gov/enso/mei/data/meiv2.data
           Close to but not exactly Busker's staged reference (e.g.
           1987-06: 2.07 here vs 1.78 in Busker's file) -- MEI.v2 is
           revised over time as ERSST reanalysis updates, which is itself
           the empirical evidence for the real-time-vs-revised leakage
           risk this pipeline documents rather than ignores.
  - NINO3.4 = NOAA CPC's raw monthly SST anomaly (ERSSTv5, 1991-2020
           baseline) -- NOT the 3-month-running-mean ONI.
           https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii
           Matches Busker et al.'s staged reference exactly (1950-01 in
           both = -1.99). This is also the file NOAA PSL's own "Climate
           Indices" reference page (Busker's cited NOAA(2023a) source)
           points to as the current Nino 3.4 series.

Each source's own missing-value sentinel is preserved as NaN, never
silently coerced to 0 (would corrupt any downstream lag/anomaly math).

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
pip install requests --break-system-packages
"""

import os

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/teleconnections/ -> pipelines/ -> repo root
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "teleconnections")

SOURCES = {
    "mei": {
        "url": "https://psl.noaa.gov/enso/mei/data/meiv2.data",
        "filename": "meiv2.data",
        "min_lines": 40,  # ~47 years of data rows + a handful of header/footer lines
    },
    "dmi": {
        "url": "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data",
        "filename": "dmi_had_long.data",
        "min_lines": 150,  # ~156 years (1870-2026) + header/footer
    },
    "nino34": {
        "url": "https://www.cpc.ncep.noaa.gov/data/indices/ersst5.nino.mth.91-20.ascii",
        "filename": "ersst5_nino_mth_91-20.ascii",
        "min_lines": 900,  # ~918 monthly rows (1950-2026) + header
    },
}


def verify_file_opens(path, min_lines):
    """Actually read the file and check it has a plausible number of
    lines -- a truncated/failed download can still "exist" on disk."""
    with open(path, "r") as f:
        lines = f.readlines()
    if len(lines) < min_lines:
        raise ValueError(f"{path} has only {len(lines)} lines, expected >= {min_lines}")
    return lines


def download_source(key, max_retries=3):
    spec = SOURCES[key]
    out_path = os.path.join(RAW_DIR, spec["filename"])

    if os.path.exists(out_path):
        try:
            verify_file_opens(out_path, spec["min_lines"])
            print(f"[{key}] cached file OK: {out_path}")
            return out_path
        except Exception as e:
            print(f"[{key}] cached file failed verification ({e}), re-downloading")
            os.remove(out_path)

    os.makedirs(RAW_DIR, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(spec["url"], timeout=60)
            resp.raise_for_status()
            with open(out_path, "w") as f:
                f.write(resp.text)
            verify_file_opens(out_path, spec["min_lines"])
            print(f"[{key}] downloaded OK: {out_path}")
            return out_path
        except Exception as e:
            if os.path.exists(out_path):
                os.remove(out_path)  # never leave a partial file behind
            print(f"[{key}] attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise

    return None


def main():
    for key in SOURCES:
        download_source(key)


if __name__ == "__main__":
    main()
