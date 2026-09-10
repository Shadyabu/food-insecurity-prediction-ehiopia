"""
fetch_glofas_seasonal.py

ACQUIRE stage for the lead3 (3-month) GloFAS feature.

REDESIGNED 2026-08-09 around bulk date-bundled requests -- see
fetch_glofas_medium_range.py's module docstring for the full rationale
(EWDS's chokepoint is the number of separate queue-wait-then-download
cycles, not per-request size; a small fixed set of LEADTIME_HOURS values
is looped, each requesting the entire needed date range in as few chunks
as possible).

CONFIRMED 2026-08-09 via a live "Show API request" example from the
project owner's own account for `cems-glofas-seasonal-reforecast`
specifically (not carried over by analogy this time):
  - Same fixed-reference-date structure as medium-range reforecast:
    year/month/day = a fixed system-reference date (2019-11-05 -- same
    value, confirmed identical across both reforecast datasets), hyear/
    hmonth = the actual target date. Unlike medium-range reforecast,
    there is NO hday/day field for the target date at all (matches this
    dataset's own operational request, which also has no day field --
    GloFAS-Seasonal issues monthly, no day-level granularity needed
    anywhere in this dataset).
  - hyear accepts a list up to 6 years per request (same cap as
    medium-range reforecast) -- the confirmed example requested
    hyear=[2009..2014] (6 years), hmonth=[01..12] (all months), in ONE
    request.
  - leadtime_hour is a single value per request (not a list) in the
    confirmed example -- matches this redesign's one-leadtime-per-request
    loop structure directly, no format change needed.

The operational dataset (`cems-glofas-seasonal`) was already confirmed
working via a live download 2026-08-09 (317MB, full leadtime range in one
call) -- this redesign only changes how leadtime/date axes are split
across requests, not the confirmed field names.
"""

import os
import time
import zipfile
from datetime import date

import pandas as pd
import xarray as xr

from cds_credentials import get_cds_client
from progress_log import log, log_header

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "glofas", "seasonal")
os.makedirs(RAW_DIR, exist_ok=True)

ETHIOPIA_BBOX_NWSE = (15.5, 32.5, 3.0, 48.5)  # matches fetch_glofas_static.py

FEATURE_START_YEAR = 2009
OPERATIONAL_START = date(2021, 1, 1)  # corrected 2026-08-12 (was 2020-12-01) -- live
    # testing during the medium-range investigation showed the documented archive-start
    # date pattern doesn't hold for the "operational" query tag (mirrors medium-range's
    # own correction); Dec 2020 already downloaded successfully via reforecast with zero
    # gaps reported, so this is a pure reclassification, no new download needed
SEASONAL_ISSUE_DAY = 10  # nominal only -- this dataset has no day-level request field;
                           # see compute_glofas_admin2_monthly_features.py for how issue
                           # months are matched on (year,month) alone, not an exact day

# Confirmed 2026-08-09 by direct computation: target month t+3's first/last
# day sit between 1920h and 2712h (80-113 days) after issue, across every
# possible issue month. 6 evenly-spaced samples -- final leadtime count,
# see fetch_glofas_medium_range.py's module docstring for the "how many"
# decision history.
LEADTIME_HOURS = ["1920", "2088", "2256", "2424", "2592", "2712"]

REFORECAST_REFERENCE_DATE = (2019, 11, 5)  # confirmed identical to medium-range
                                             # reforecast's own reference date, 2026-08-09
REFORECAST_HYEAR_CHUNK_SIZE = 6  # confirmed via a live "Show API request" example

CDS_DATASET_OPERATIONAL = "cems-glofas-seasonal"
CDS_DATASET_REFORECAST = "cems-glofas-seasonal-reforecast"

client = None  # created lazily in main() via get_cds_client()


def compute_needed_months():
    """Same pattern as fetch_glofas_medium_range.py's equivalent function
    -- single source of truth reused for chunk-building and the compute
    stage's scaffold."""
    today = date.today()
    months = pd.period_range(f"{FEATURE_START_YEAR}-01", today.strftime("%Y-%m"), freq="M")
    return [(p.year, p.month, date(p.year, p.month, SEASONAL_ISSUE_DAY) < OPERATIONAL_START) for p in months]


OPERATIONAL_YEAR_CHUNK_SIZE = 2  # matches fetch_glofas_medium_range.py's confirmed-live value
    # (2 full years accepted, 4 rejected with "cost limits exceeded" for that dataset) --
    # NOT independently confirmed for this dataset specifically; reused conservatively
    # rather than assuming a looser untested limit. report_time_completeness() is the
    # safety net if this turns out to be unnecessarily small.


def build_operational_chunks(needed):
    """See fetch_glofas_medium_range.py's equivalent function for the full
    rationale (confirmed live 2026-08-09): a boundary year with fewer than
    12 actually-needed months gets its own chunk with exactly those
    months, never bundled with a full year; full years are grouped into
    chunks of OPERATIONAL_YEAR_CHUNK_SIZE rather than all bundled into one
    request."""
    year_months = {}
    for y, m, r in needed:
        if not r:
            year_months.setdefault(y, set()).add(m)

    full_years = sorted(y for y, months in year_months.items() if len(months) == 12)
    partial_years = sorted(y for y in year_months if y not in full_years)

    chunks = []
    for y in partial_years:
        months = sorted(year_months[y])
        chunks.append({"year": [str(y)], "month": [f"{m:02d}" for m in months]})
    for i in range(0, len(full_years), OPERATIONAL_YEAR_CHUNK_SIZE):
        chunk_years = full_years[i:i + OPERATIONAL_YEAR_CHUNK_SIZE]
        chunks.append({"year": [str(y) for y in chunk_years], "month": [f"{m:02d}" for m in range(1, 13)]})
    return chunks


def build_reforecast_chunks(needed):
    years = sorted({y for y, m, r in needed if r})
    chunks = []
    for i in range(0, len(years), REFORECAST_HYEAR_CHUNK_SIZE):
        chunk_years = years[i:i + REFORECAST_HYEAR_CHUNK_SIZE]
        chunks.append({"hyear": [str(y) for y in chunk_years], "hmonth": [f"{m:02d}" for m in range(1, 13)]})
    return chunks


def request_dict_operational(leadtime_hour, chunk):
    return {
        "system_version": ["operational"],
        "hydrological_model": ["lisflood"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "year": chunk["year"],
        "month": chunk["month"],
        "leadtime_hour": [leadtime_hour],
        "area": list(ETHIOPIA_BBOX_NWSE),
        "data_format": "grib2",
        "download_format": "zip",
    }


def request_dict_reforecast(leadtime_hour, chunk):
    ref_year, ref_month, ref_day = REFORECAST_REFERENCE_DATE
    return {
        "hydrological_model": ["lisflood"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "year": [str(ref_year)], "month": [f"{ref_month:02d}"], "day": [f"{ref_day:02d}"],
        "hyear": chunk["hyear"],
        "hmonth": chunk["hmonth"],
        "leadtime_hour": [leadtime_hour],
        "area": list(ETHIOPIA_BBOX_NWSE),
        "data_format": "grib2",
        "download_format": "zip",
    }


def verify_grib_opens(path):
    opened_any = False
    for data_type in ("cf", "pf", "fc"):
        try:
            with xr.open_dataset(path, engine="cfgrib",
                                  backend_kwargs={"filter_by_keys": {"dataType": data_type}, "indexpath": ""}):
                opened_any = True
        except Exception as e:
            print(f"  note: {path} has no readable dataType={data_type!r} messages ({e})")
    if not opened_any:
        print(f"  WARNING: {path} failed to open via cfgrib for any known dataType")
    return opened_any


def extract_grib_from_zip(zip_path, dest_path):
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith((".grib", ".grib2"))]
        if len(members) != 1:
            raise RuntimeError(f"{zip_path}: expected exactly one .grib/.grib2 member, found {members}")
        with zf.open(members[0]) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())


QUEUE_LIMIT_RETRY_WAIT_SECONDS = 180
QUEUE_LIMIT_MAX_RETRIES = 8


def is_queue_limit_error(exc):
    return "temporarily limited" in str(exc).lower()


def retrieve_with_retry(dataset, request, zip_tmp_path):
    for attempt in range(1, QUEUE_LIMIT_MAX_RETRIES + 1):
        try:
            client.retrieve(dataset, request).download(zip_tmp_path)
            return
        except Exception as e:
            if not is_queue_limit_error(e) or attempt == QUEUE_LIMIT_MAX_RETRIES:
                raise
            print(f"  queue-limited (attempt {attempt}/{QUEUE_LIMIT_MAX_RETRIES}) -- "
                  f"waiting {QUEUE_LIMIT_RETRY_WAIT_SECONDS}s before retrying: {e}")
            time.sleep(QUEUE_LIMIT_RETRY_WAIT_SECONDS)


def fetch_one_request(dataset, request, cache_path):
    label = os.path.basename(cache_path)
    if os.path.exists(cache_path) and verify_grib_opens(cache_path):
        log(f"[seasonal] {label} -- already cached, skipping")
        return cache_path

    zip_tmp_path = cache_path + ".zip.tmp"
    grib_tmp_path = cache_path + ".tmp"
    log(f"[seasonal] {label} -- requesting {dataset}...")
    try:
        retrieve_with_retry(dataset, request, zip_tmp_path)
        extract_grib_from_zip(zip_tmp_path, grib_tmp_path)
    except Exception as e:
        log(f"[seasonal] {label} -- FAILED: {e}")
        for p in (zip_tmp_path, grib_tmp_path):
            if os.path.exists(p):
                os.remove(p)
        return None
    finally:
        if os.path.exists(zip_tmp_path):
            os.remove(zip_tmp_path)

    if not verify_grib_opens(grib_tmp_path):
        log(f"[seasonal] {label} -- downloaded but failed cfgrib verification")
        os.remove(grib_tmp_path)
        return None
    os.replace(grib_tmp_path, cache_path)
    size_mb = os.path.getsize(cache_path) / 1e6
    log(f"[seasonal] {label} -- SUCCESS ({size_mb:.1f}MB)")
    return cache_path


def report_time_completeness(path, expected_year_month_pairs):
    """See fetch_glofas_medium_range.py's equivalent function -- same
    diagnostic-only completeness check."""
    try:
        with xr.open_dataset(path, engine="cfgrib",
                              backend_kwargs={"filter_by_keys": {"dataType": "pf"}, "indexpath": ""}) as ds:
            time_key = "time" if "time" in ds.coords else ("hdate" if "hdate" in ds.coords else None)
            if time_key is None:
                return
            times = pd.to_datetime(ds[time_key].values)
            got = {(t.year, t.month) for t in times}
    except Exception:
        return
    missing = set(expected_year_month_pairs) - got
    if missing:
        log(f"[seasonal] note: {os.path.basename(path)} -- "
            f"{len(missing)}/{len(expected_year_month_pairs)} requested (year,month) "
            f"combinations not present in output: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")


def main():
    global client
    client = get_cds_client()

    needed = compute_needed_months()
    operational_chunks = build_operational_chunks(needed)
    reforecast_chunks = build_reforecast_chunks(needed)
    total_planned = len(LEADTIME_HOURS) * (len(operational_chunks) + len(reforecast_chunks))

    log_header("Seasonal backfill started")
    log(f"[seasonal] Plan: {len(LEADTIME_HOURS)} leadtimes x "
        f"({len(operational_chunks)} operational + {len(reforecast_chunks)} reforecast chunks) "
        f"= {total_planned} requests")

    fetched, failed = 0, 0
    for leadtime in LEADTIME_HOURS:
        for i, chunk in enumerate(operational_chunks, start=1):
            cache_path = os.path.join(RAW_DIR, f"glofas_seasonal_operational_leadtime{leadtime}_chunk{i}.grib2")
            request = request_dict_operational(leadtime, chunk)
            path = fetch_one_request(CDS_DATASET_OPERATIONAL, request, cache_path)
            if path is None:
                failed += 1
                continue
            fetched += 1
            expected = [(int(y), m) for y in chunk["year"] for m in map(int, chunk["month"])]
            report_time_completeness(path, expected)
            log(f"[seasonal] progress: {fetched + failed}/{total_planned} requests attempted "
                f"({fetched} ok, {failed} failed)")

        for i, chunk in enumerate(reforecast_chunks, start=1):
            cache_path = os.path.join(RAW_DIR, f"glofas_seasonal_reforecast_leadtime{leadtime}_chunk{i}.grib2")
            request = request_dict_reforecast(leadtime, chunk)
            path = fetch_one_request(CDS_DATASET_REFORECAST, request, cache_path)
            if path is None:
                failed += 1
                continue
            fetched += 1
            expected = [(int(y), m) for y in chunk["hyear"] for m in map(int, chunk["hmonth"])]
            report_time_completeness(path, expected)
            log(f"[seasonal] progress: {fetched + failed}/{total_planned} requests attempted "
                f"({fetched} ok, {failed} failed)")

    log(f"[seasonal] DONE -- fetched {fetched}, failed {failed} of {total_planned} planned requests")


if __name__ == "__main__":
    main()
