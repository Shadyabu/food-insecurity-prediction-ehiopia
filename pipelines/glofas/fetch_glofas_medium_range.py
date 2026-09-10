"""
fetch_glofas_medium_range.py

ACQUIRE stage for the lead1 (1-month) GloFAS feature.

REDESIGNED 2026-08-09 around bulk date-bundled requests, not one request
per issue date. The original per-date design (one request per issue date,
covering that date's full leadtime window) needed ~200+ requests for the
medium-range dataset alone. Since EWDS's real chokepoint is the number of
separate queue-wait-then-download cycles (each taking anywhere from ~20s
to ~10min regardless of file size, per the project owner's own experience
hitting a "Number queued requests for this dataset is temporarily
limited" rejection), the redesign instead loops over a small, fixed set of
LEADTIME values (6 -- see LEADTIME_HOURS) and, for each, issues as few
bulk requests as possible covering the ENTIRE historical range in one go:

  - operational (`cems-glofas-forecast`): year=[all needed years],
    month=[01..12] in ONE request per leadtime -- confirmed empirically
    (see below) that CDS/EWDS tolerates a request whose year x month cross
    product includes some genuinely nonexistent combinations (e.g.
    year=2019 month=01..10, which predates this dataset's 2019-11-05
    archive start) without rejecting the whole request; it just returns
    whatever's actually valid.
  - reforecast (`cems-glofas-reforecast`): hyear=[chunk of <=6 years],
    hmonth=[01..12] per chunk (2 chunks needed for 2009-2019) -- the
    hyear<=6 cap is a real, confirmed EWDS limit for this dataset, not
    negotiable the way the operational year list is.

CORRECTED 2026-08-12, after a real backfill run and a lot of live diagnosis
(see docs/GloFAS_Pipeline_Documentation.md Sec 5d for the full story) --
THREE things the 2026-08-09 design got wrong, found by actually running it:

1. **Reforecast's single FIXED_HDAY silently dropped ~58% of months.** The
   "mixed-validity tolerance" finding was real (CDS accepts a request with
   some invalid combinations rather than rejecting it outright), but it
   turned out to mean "silently omit the invalid ones from the output,"
   not "return everything anyway." A single hday=28 only coincided with a
   real reforecast issue day for ~5 of 12 months. Fixed by requesting
   `hday` as a list of every day confirmed valid across multiple real
   month/year checks (REFORECAST_HDAY_CANDIDATES) rather than one guess.

2. **The reforecast reference-date fields (year/month/day) actively broke
   things once `system_version` is specified.** Confirmed live 2026-08-12:
   including a fixed year/month/day reference date alongside an explicit
   `system_version` produces a genuine MARS-server-side error ("Ambiguous
   parameter: day could be DATE or DATABASE") -- a real request-shape bug,
   not a data-availability question. The fix (confirmed working via the
   project owner's own live test) is to specify `system_version` and drop
   the year/month/day reference fields entirely -- they were only ever a
   way to *implicitly* select a system version; once you specify it
   directly they're not just unnecessary, they're actively harmful.

3. **Operational's real archive start is ~November 2020, not 2019-11-05.**
   The originally-documented date (which happens to be exactly when
   GloFAS v2.1 launched -- a version-launch date, not necessarily this
   query pattern's archive-start date) does NOT work: every single test
   for Jan-Oct 2020 was rejected outright (400) under `system_version=
   operational`, tried against several other explicit version tags too,
   all rejected. Nov 2020 onward works. This 12-month window
   (2019-11 through 2020-10) is NOT covered by the reforecast dataset
   either (`hyear=2020` alone was rejected under the ORIGINAL reforecast
   request shape) -- but IS reachable via the FIXED shape above
   (`system_version=version_4_0`, no reference date), confirmed live. So
   OPERATIONAL_START is now the true boundary, and reforecast's coverage
   extends through most of what used to be an assumed-unrecoverable gap.

4. Operational's true max leadtime is 720h (30 days), not 816h -- 744h/
   768h/792h/816h were all rejected outright when tested in isolation.
   Reforecast genuinely does support 816h (confirmed working in the
   original backfill). LEADTIME_HOURS now uses 720h as the top sample for
   BOTH eras (one shared list, simpler than era-specific lists) --
   reforecast loses a small amount of theoretical reach at the very top of
   its window as a result, an accepted, minor trade-off for code
   simplicity given it already lost the true worst-case edge to the
   hday-candidate-window discretization anyway.

RATE LIMITS / SEQUENTIAL-ONLY: unchanged -- see retrieve_with_retry() and
the strictly sequential loop in main(). The reforecast hday-candidate-list
fix increases per-request field count substantially (15 hday values now,
not 1), which is why REFORECAST_HYEAR_CHUNK_SIZE dropped from 6 to 2 --
still comfortably under the 950-field cap (15 hday x 12 hmonth x 2
product_type x 2 hyear = 720 < 950).
"""

import os
import time
import zipfile
from datetime import date, timedelta

import pandas as pd
import xarray as xr

from cds_credentials import get_cds_client
from progress_log import log, log_header

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "glofas", "medium_range")
os.makedirs(RAW_DIR, exist_ok=True)

ETHIOPIA_BBOX_NWSE = (15.5, 32.5, 3.0, 48.5)  # matches fetch_glofas_static.py

FEATURE_START_YEAR = 2009
OPERATIONAL_START = date(2020, 11, 1)  # CORRECTED 2026-08-12 (was 2019-11-05) -- see
    # module docstring point 3. Exact day-of-month within November 2020 unconfirmed
    # (only tested at month granularity: Oct 2020 rejected, Nov 2020 accepted); using
    # the 1st is a safe choice since era classification compares against month-END
    # dates, so any day-of-month value here classifies all of November correctly.

# Corrected 2026-08-12: operational's real max leadtime is 720h (30 days),
# not 816h -- 744-816h all rejected outright when tested in isolation (see
# module docstring point 4). Shared by both eras now for simplicity.
LEADTIME_HOURS = ["24", "192", "360", "528", "696", "720"]

# Evidence-based, not derived from a clean rule -- an initial "Wed/Sat only"
# theory (module docstring point 1) turned out to be wrong (two of the
# project owner's confirmed-working March 2020 days, the 27th and 30th,
# are Friday and Monday). This is the UNION of every hday value confirmed
# valid across three independent real checks (Jan 2009, Mar 2009, Mar
# 2020 -- the last one from a live successful request, not a form
# calendar): mostly stable across at least a decade of years, with a
# couple of extra days appearing in the more recent check. Not guaranteed
# complete for months never checked -- report_time_completeness() is the
# safety net if a month turns out to need a day not in this list.
REFORECAST_HDAY_CANDIDATES = [
    "03", "04", "07", "10", "11", "14", "17", "18", "21", "24", "25", "27", "28", "30", "31",
]
REFORECAST_HYEAR_CHUNK_SIZE = 2  # dropped from 6 -- the larger hday candidate list above
    # multiplies the field count per request (15 hday x 12 hmonth x 2 product_type = 360
    # per hyear), so 950/360 = 2.6 -> 2 years/chunk stays under cems-glofas-reforecast's
    # 950-field cap (confirmed 2026-08-09; see docs/GloFAS_Pipeline_Documentation.md)
OPERATIONAL_FIXED_DAY = "28"  # operational issues daily -- any fixed day works, no
                                # weekday constraint (unlike reforecast's Wed/Sat-ish cadence)
REFORECAST_SYSTEM_VERSION = "version_4_0"  # confirmed 2026-08-12 -- must be explicit;
    # omitting it is what forced the old year/month/day reference-date workaround, which
    # is what caused the MARS "Ambiguous parameter" error once combined with this field

CDS_DATASET_OPERATIONAL = "cems-glofas-forecast"
CDS_DATASET_REFORECAST = "cems-glofas-reforecast"

client = None  # created lazily in main() via get_cds_client()


def month_end(year, month):
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def compute_needed_months():
    """Single source of truth for which (year, month) this project needs,
    and which era each belongs to -- reused for chunk-building and for the
    compute stage's own scaffold, so the two can never silently drift
    apart."""
    today = date.today()
    months = pd.period_range(f"{FEATURE_START_YEAR}-01", today.strftime("%Y-%m"), freq="M")
    return [(p.year, p.month, month_end(p.year, p.month) < OPERATIONAL_START) for p in months]


OPERATIONAL_YEAR_CHUNK_SIZE = 2  # confirmed empirically 2026-08-09 (live requests, not
    # analytical): 2 full years (2020-2021) accepted; 4 full years (2020-2023) rejected
    # with a 403 "cost limits exceeded / Your request is too large" -- a genuine data-
    # volume cap, distinct from the fields-count cap this pipeline already accounts for.
    # NOT independently confirmed for cems-glofas-seasonal (fetch_glofas_seasonal.py
    # reuses this same conservative value rather than assuming a looser one untested).


def build_operational_chunks(needed):
    """CORRECTED 2026-08-12: an earlier version of this function isolated
    boundary (partial) years into their own chunk with exactly their valid
    months, reasoning that operational didn't tolerate a boundary year's
    nonexistent months in a bundled request. Confirmed live that this was
    backwards -- an ISOLATED small boundary-year request (year=[2020],
    month=[11,12] alone) gets rejected with a 400, even though the SAME
    months succeed fine when bundled normally with an adjacent full year
    (year=[2020,2021], month=[01..12] -- confirmed working in the original
    backfill, silently dropping the invalid Jan-Oct 2020 combinations via
    the same mixed-validity tolerance reforecast relies on). So every year
    (partial or full, including the current in-progress year) is now
    chunked uniformly in groups of OPERATIONAL_YEAR_CHUNK_SIZE with
    month=[01..12] always -- simpler than the old per-year-exactness
    design, and the one that actually works."""
    years = sorted({y for y, m, r in needed if not r})
    chunks = []
    for i in range(0, len(years), OPERATIONAL_YEAR_CHUNK_SIZE):
        chunk_years = years[i:i + OPERATIONAL_YEAR_CHUNK_SIZE]
        chunks.append({"year": [str(y) for y in chunk_years], "month": [f"{m:02d}" for m in range(1, 13)]})
    return chunks


def build_reforecast_chunks(needed):
    """Splits the full reforecast-era year range into <=6-year chunks
    (confirmed EWDS cap), each requesting all 12 months uniformly (relies
    on the same mixed-validity tolerance -- confirmed for exactly this
    dataset, see module docstring)."""
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
        "product_type": ["control_forecast", "ensemble_perturbed_forecasts"],
        "variable": "river_discharge_in_the_last_24_hours",
        "year": chunk["year"],
        "month": chunk["month"],
        "day": [OPERATIONAL_FIXED_DAY],
        "leadtime_hour": [leadtime_hour],
        "area": list(ETHIOPIA_BBOX_NWSE),
        "data_format": "grib2",
        "download_format": "zip",
    }


def request_dict_reforecast(leadtime_hour, chunk):
    """Confirmed 2026-08-12 (see module docstring points 1-2): NO year/
    month/day reference-date fields -- their presence alongside an
    explicit system_version is what caused a genuine MARS server error,
    not just redundant. hday uses the evidence-based candidate list, not
    a single guessed value."""
    return {
        "system_version": [REFORECAST_SYSTEM_VERSION],
        "hydrological_model": ["lisflood"],
        "product_type": ["control_reforecast", "ensemble_perturbed_reforecast"],
        "variable": ["river_discharge_in_the_last_24_hours"],
        "hyear": chunk["hyear"],
        "hmonth": chunk["hmonth"],
        "hday": REFORECAST_HDAY_CANDIDATES,
        "leadtime_hour": [leadtime_hour],
        "area": list(ETHIOPIA_BBOX_NWSE),
        "data_format": "grib2",
        "download_format": "zip",
    }


def verify_grib_opens(path):
    # See docs/GloFAS_Pipeline_Documentation.md for the full dataType
    # history (cf/pf for medium-range, fc seen in a seasonal-reforecast
    # file) -- try all three, any one present is sufficient.
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
    """Retried ONLY for the transient shared-queue-slot rejection -- any
    other error (e.g. a genuine 400) is not retried, since retrying a
    malformed request just burns another shared queue slot for nothing."""
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
    """One fully sequential fetch: submit, block until downloaded, extract,
    verify, cache. The caller never overlaps this with another in-flight
    request."""
    label = os.path.basename(cache_path)
    if os.path.exists(cache_path) and verify_grib_opens(cache_path):
        log(f"[medium_range] {label} -- already cached, skipping")
        return cache_path

    zip_tmp_path = cache_path + ".zip.tmp"
    grib_tmp_path = cache_path + ".tmp"
    log(f"[medium_range] {label} -- requesting {dataset}...")
    try:
        retrieve_with_retry(dataset, request, zip_tmp_path)
        extract_grib_from_zip(zip_tmp_path, grib_tmp_path)
    except Exception as e:
        log(f"[medium_range] {label} -- FAILED: {e}")
        for p in (zip_tmp_path, grib_tmp_path):
            if os.path.exists(p):
                os.remove(p)
        return None
    finally:
        if os.path.exists(zip_tmp_path):
            os.remove(zip_tmp_path)

    if not verify_grib_opens(grib_tmp_path):
        log(f"[medium_range] {label} -- downloaded but failed cfgrib verification")
        os.remove(grib_tmp_path)
        return None
    os.replace(grib_tmp_path, cache_path)
    size_mb = os.path.getsize(cache_path) / 1e6
    log(f"[medium_range] {label} -- SUCCESS ({size_mb:.1f}MB)")
    return cache_path


def report_time_completeness(path, expected_year_month_pairs):
    """Diagnostic only (not blocking): confirms the mixed-validity
    tolerance confirmed 2026-08-09 (see module docstring) actually returns
    every requested (year,month) combination that SHOULD be valid, not
    just the ones it happens to have. Prints a warning on a shortfall
    rather than failing -- the compute stage's own scaffold-based gap
    handling is the real safety net, this is just an early signal."""
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
        log(f"[medium_range] note: {os.path.basename(path)} -- "
            f"{len(missing)}/{len(expected_year_month_pairs)} requested (year,month) "
            f"combinations not present in output (expected for genuinely pre-archive dates; "
            f"investigate if unexpectedly large): "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}")


def main():
    global client
    client = get_cds_client()

    needed = compute_needed_months()
    operational_chunks = build_operational_chunks(needed)
    reforecast_chunks = build_reforecast_chunks(needed)
    total_planned = len(LEADTIME_HOURS) * (len(operational_chunks) + len(reforecast_chunks))

    log_header(f"Medium-range backfill started")
    log(f"[medium_range] Plan: {len(LEADTIME_HOURS)} leadtimes x "
        f"({len(operational_chunks)} operational + {len(reforecast_chunks)} reforecast chunks) "
        f"= {total_planned} requests")

    fetched, failed = 0, 0
    for leadtime in LEADTIME_HOURS:
        for i, chunk in enumerate(operational_chunks, start=1):
            cache_path = os.path.join(RAW_DIR, f"glofas_mediumrange_operational_leadtime{leadtime}_chunk{i}.grib2")
            request = request_dict_operational(leadtime, chunk)
            path = fetch_one_request(CDS_DATASET_OPERATIONAL, request, cache_path)
            if path is None:
                failed += 1
                continue
            fetched += 1
            expected = [(int(y), m) for y in chunk["year"] for m in map(int, chunk["month"])]
            report_time_completeness(path, expected)
            log(f"[medium_range] progress: {fetched + failed}/{total_planned} requests attempted "
                f"({fetched} ok, {failed} failed)")

        for i, chunk in enumerate(reforecast_chunks, start=1):
            cache_path = os.path.join(RAW_DIR, f"glofas_mediumrange_reforecast_leadtime{leadtime}_chunk{i}.grib2")
            request = request_dict_reforecast(leadtime, chunk)
            path = fetch_one_request(CDS_DATASET_REFORECAST, request, cache_path)
            if path is None:
                failed += 1
                continue
            fetched += 1
            expected = [(int(y), m) for y in chunk["hyear"] for m in map(int, chunk["hmonth"])]
            report_time_completeness(path, expected)
            log(f"[medium_range] progress: {fetched + failed}/{total_planned} requests attempted "
                f"({fetched} ok, {failed} failed)")

    log(f"[medium_range] DONE -- fetched {fetched}, failed {failed} of {total_planned} planned requests")


if __name__ == "__main__":
    main()
