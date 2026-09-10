"""
fetch_chirps_gefs_15day.py

ACQUIRE stage for the USGS Rainfall Forecast feature.

--------------------------------------------------------------------------
IDENTITY (confirmed 2026-08-09, see docs/USGS_GEFS_Pipeline_Documentation.md)
--------------------------------------------------------------------------
docs/data_sources_master_table.md's "Rainfall Forecast (USGS)" row is
CHIRPS-GEFS: a bias-corrected/downscaled NCEP GEFS v12 precipitation
forecast, produced by UCSB's Climate Hazards Center and distributed
through USGS EROS's Early Warning eXplorer -- the only USGS-adjacent
rainfall *forecast* product confirmed to exist as real, bulk-downloadable,
gridded data (the other candidate reading -- a USGS-produced seasonal/
NMME-style outlook -- turned up no independent gridded dataset, only
report/PDF synthesis of the same NMME ensemble IRI's own seasonal forecast
already draws from). Not a Busker et al. feature (zero mentions of USGS/
GEFS in docs/busker_2024.pdf) -- an Ethiopia-specific RQ2+ addition, same
framing as GloFAS.

This pipeline uses the **v3** archive (bias-corrected to CHIRPS3 --
CHIRPS2-GEFS/v2 was discontinued 2026-07-01, before this build) and the
**africa** regional subset (~7MB/file vs. ~65MB for the global grid --
same values, much less to download for an Ethiopia-only project), at the
**15-day** horizon (the longest of the three native horizons -- 5/10/15-day
-- to give the most forecast information; the shorter horizons are strict
subsets of the same underlying bias-corrected GEFS run and would be highly
collinear with the 15-day product, so they are deliberately not also
built, per CLAUDE.md's "don't default to finest granularity" §0.2
guidance applied to *horizon count* here rather than to temporal
resolution).

--------------------------------------------------------------------------
WHAT'S DOWNLOADED
--------------------------------------------------------------------------
One issue-date pair of GeoTIFFs per calendar month (not the full daily
archive -- confirmed via CLAUDE.md guidance and pipeline_replication_guide
§0.2 not to default to the finest resolution when it isn't needed): the
forecast issued nearest to, but not after, the last calendar day of the
month, matching GloFAS's own "forecast issued near month-end t" choice for
its lead1 (see docs/GloFAS_Pipeline_Documentation.md §5a). Both bands are
fetched per issue date:
  - data/{year}/c3g_{Y}.{M}.{D}.tif       -- 15-day cumulative precip, mm
  - anom/{year}/c3g_anom_{Y}.{M}.{D}.tif  -- 15-day anomaly (units
    confirmed empirically at Aggregate time -- see compute script)

Archive confirmed (2026-08-09, live directory checks against
data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/15_day/africa/):
  - 2001-2019 and 2021-present: full daily coverage (365/366 files/year)
  - 2020: confirmed EMPTY (0 files) -- GEFS v12 became operational Sep
    2020 at NCEP; this is a real, documented archive gap, not a bug.
    Every month in 2020 gets glofas-locust-style explicit
    usgs_gefs_source_lead1 = "no_archive" and no downloaded file.
  - 2026 data present through today (2026-08-09) -- live daily updates.

Verify-before-cache: every download is opened with rasterio before being
trusted, matching every other pipeline's convention (a truncated download
still "exists" on disk).
"""

import calendar
import datetime
import os

import rasterio
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "usgs_gefs")
MM_DIR = os.path.join(RAW_DIR, "data")
ANOM_DIR = os.path.join(RAW_DIR, "anom")
INTERIM_DIR = os.path.join(REPO_ROOT, "data", "interim", "usgs_gefs")

os.makedirs(MM_DIR, exist_ok=True)
os.makedirs(ANOM_DIR, exist_ok=True)
os.makedirs(INTERIM_DIR, exist_ok=True)

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS-GEFS/v3/15_day/africa"
MM_URL_TMPL = BASE_URL + "/data/{year}/c3g_{year}.{month:02d}.{day:02d}.tif"
ANOM_URL_TMPL = BASE_URL + "/anom/{year}/c3g_anom_{year}.{month:02d}.{day:02d}.tif"

FEATURE_START_YEAR = 2009  # matches every other pipeline's convention
FEATURE_END_YEAR = 2026

NO_ARCHIVE_YEARS = {2020}  # confirmed empty directory, GEFS v12 pre-operational

# how many days to step backward from calendar month-end looking for the
# nearest available issue date, if the exact last day is ever missing
# (e.g. a transient gap not already known) -- never step forward, per
# CLAUDE.md §3.2's "only the vintage issued at or before t" rule.
MAX_BACKWARD_STEP_DAYS = 5

TODAY = datetime.date.today()


def _download(url, out_path, max_retries=3):
    if os.path.exists(out_path):
        try:
            with rasterio.open(out_path) as ds:
                ds.read(1, window=((0, 1), (0, 1)))
            return True
        except Exception:
            print(f"    cached file failed to open, re-downloading: {out_path}")
            os.remove(out_path)

    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(url, stream=True, timeout=(30, 120)) as r:
                if r.status_code == 404:
                    return False
                r.raise_for_status()
                with open(out_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
            with rasterio.open(out_path) as ds:
                ds.read(1, window=((0, 1), (0, 1)))
            return True
        except Exception as exc:
            print(f"    attempt {attempt} failed for {url}: {exc}")
            if os.path.exists(out_path):
                os.remove(out_path)
            if attempt == max_retries:
                return False
    return False


def fetch_month_end_issue(year, month):
    """Find the forecast issued nearest to, but not after, the last
    calendar day of (year, month). Returns a dict with the resolved issue
    date and both file paths, or a no_archive/no_data marker."""
    if year in NO_ARCHIVE_YEARS:
        return {
            "year": year, "month": month, "issue_date": None,
            "mm_path": None, "anom_path": None, "source": "no_archive",
        }

    last_day = calendar.monthrange(year, month)[1]
    target_date = datetime.date(year, month, last_day)
    if target_date > TODAY:
        return {
            "year": year, "month": month, "issue_date": None,
            "mm_path": None, "anom_path": None, "source": "future",
        }

    for step in range(MAX_BACKWARD_STEP_DAYS + 1):
        candidate = target_date - datetime.timedelta(days=step)
        if candidate.month != month:
            break  # don't cross back into the previous month
        if candidate > TODAY:
            continue

        mm_url = MM_URL_TMPL.format(year=candidate.year, month=candidate.month, day=candidate.day)
        anom_url = ANOM_URL_TMPL.format(year=candidate.year, month=candidate.month, day=candidate.day)
        mm_path = os.path.join(MM_DIR, f"c3g_{candidate.isoformat()}.tif")
        anom_path = os.path.join(ANOM_DIR, f"c3g_anom_{candidate.isoformat()}.tif")

        print(f"  {year}-{month:02d}: trying issue date {candidate.isoformat()} ...")
        mm_ok = _download(mm_url, mm_path)
        if not mm_ok:
            continue
        anom_ok = _download(anom_url, anom_path)
        if not anom_ok:
            os.remove(mm_path)
            continue

        return {
            "year": year, "month": month, "issue_date": candidate.isoformat(),
            "mm_path": mm_path, "anom_path": anom_path, "source": "v3",
        }

    print(f"  {year}-{month:02d}: no issue date found within {MAX_BACKWARD_STEP_DAYS} days of month-end.")
    return {
        "year": year, "month": month, "issue_date": None,
        "mm_path": None, "anom_path": None, "source": "no_data",
    }


def main():
    manifest_rows = []
    for year in range(FEATURE_START_YEAR, FEATURE_END_YEAR + 1):
        for month in range(1, 13):
            row = fetch_month_end_issue(year, month)
            manifest_rows.append(row)
            if row["source"] == "future":
                break  # nothing more to fetch once we pass today

    import pandas as pd
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df = manifest_df[manifest_df["source"] != "future"].reset_index(drop=True)
    manifest_path = os.path.join(INTERIM_DIR, "usgs_gefs_issue_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)

    n_v3 = (manifest_df["source"] == "v3").sum()
    n_gap = (manifest_df["source"] != "v3").sum()
    print(f"\nManifest saved -> {manifest_path}")
    print(f"{n_v3} months resolved to a real issue date, {n_gap} months flagged as gaps.")
    print(manifest_df["source"].value_counts())


if __name__ == "__main__":
    main()
