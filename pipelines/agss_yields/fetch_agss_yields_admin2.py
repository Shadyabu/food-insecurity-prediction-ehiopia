"""
fetch_agss_yields_admin2.py

ACQUIRE stage for the ESS Agricultural Sample Survey (AgSS) yield pipeline.

SOURCE
------
FEWS NET's own Data Warehouse (FDW), `cropproductionfacts` endpoint
(https://fdw.fews.net/api/cropproductionfacts.json) -- NOT HarvestStat Africa
(Lee et al. 2025, Dryad), which was investigated first and rejected. Two
reasons: (1) HarvestStat's own paper credits Ethiopia's data only to
"Ministry of Agriculture, Ethiopia", never CSA/ESS/AgSS by name, whereas FDW
records for Ethiopia carry `source_organization: "CSA, Ethiopia"` (also seen
as "CSO, Ethiopia" -- both prior names of today's ESS) and
`publication_name: "Agricultural Sample Survey, <year>, <season>, Ethiopia"`
directly -- this is genuinely, verifiably the AgSS. (2) FDW is already this
project's trusted access pattern (ipc_target, crop_calendar, ha_share all use
it), so no new infrastructure/access risk is introduced.

Busker et al. do NOT use crop-yield data: "We did not have access to
crop-yield data on a monthly time scale for all the administrative units.
Its exclusion may be a reason why the food-security crises could not be
predicted effectively in crop-farming regions" (Busker et al. 2024, S4.3.1).
So this is a pure Ethiopia-specific RQ2+ addition -- there is no RQ1 parity
question and no Busker reference to validate against for this source.

SCOPE
-----
Four staple grains, matching the commodities already scoped in
pipelines/wfp_prices/ for consistency: Maize (Corn), Wheat Grain, Sorghum,
Mixed Teff (FDW's own product names -- confirmed by sampling the live data,
not guessed). Indicator `crop:yield` only (Area Harvested / Area Planted /
Quantity Produced are NOT fetched -- redundant with yield once both area and
production are known, per the project owner's confirmed scope decision).

RECENCY CEILING -- CONFIRMED, NOT A BUG
----------------------------------------
The single most recent `period_date` across EVERY crop/indicator for Ethiopia
in FDW's cropproductionfacts, unfiltered, is 2022-07-31. This was checked
directly (not inferred from the 4 staple crops alone) before committing to
this source: HarvestStat Africa has the identical 2022 ceiling, confirming
this is a real, current limit on publicly available Ethiopia AgSS data (both
HarvestStat's own paper and this project's other pipelines already document
2020-2022 Tigray-conflict-era reporting disruption), not a source-specific
staleness artefact. 2023-2026 therefore has NO AgSS data from any known
source -- the engineering stage carries this as an explicit "no source"
gap (status = "no_data_post_2022"), never an implied zero or a forward-fill,
same convention as pipelines/locust/'s 2022 gap.

API MECHANICS -- WINDOWED PAGINATION REQUIRED, BY EXACT period_date
------------------------------------------------------------------------
FDW hard-caps at 1,000 records per query: `page_size`/`offset` pagination
past offset=1000 returns a persistent 403 (confirmed: this is a hard
ceiling, not a transient rate limit -- retries with exponential backoff
never recover). Each of the 4 crops' `crop:yield` series has ~1,000-1,600
total records for Ethiopia (NOT the ~5,000-6,000 an earlier, unfiltered-by-
indicator probe suggested), so pagination alone still cannot retrieve the
larger ones.

Two candidate range filters were tried and BOTH silently no-op on this
endpoint -- confirmed empirically, not assumed: `period_date__gte`/`__lt`
and `start_date__gte`/`__lte`/`__gt`/`__lt` all return the exact same
`count` regardless of the bound supplied (including a single-day window),
meaning Django REST Framework is silently ignoring an unrecognised filter
key rather than erroring on it (it errors clearly on a genuinely invalid
*value*, e.g. `indicator=Yield` -- but not on an unsupported *field name*
with a `__suffix`). Only **exact-match** filtering works: `period_date=`
and `start_date=` both genuinely narrow the result set (confirmed: distinct
counts for distinct exact values).

This pipeline only needs `season_name == "Meher"` records (see
compute_agss_yields_admin2_monthly_features.py), and every observed
`period_date` in this data follows exactly one of two patterns --
`YYYY-01-31` (Meher) or `YYYY-07-31` (Belg), confirmed by inspecting the
month-day suffix across ~2,800 previously-sampled records with zero
exceptions. So the fetch partitions by exact `period_date=YYYY-01-31` per
calendar year across the plausible AgSS coverage range: each such query
returns at most a few hundred records (one per admin_2 x fnid-vintage,
Ethiopia has 92-105 admin_2 units), safely under the 1,000-record cap with
no bisection needed. `07-31` (Belg) is fetched too and cached, even though
the ENGINEER stage only uses Meher, so the raw cache stays complete for any
future use.

Run this whenever you want to refresh the cache -- cached files are
verified to actually parse as JSON with the expected keys before being
trusted, per the project's io.py convention; a truncated/failed response is
never silently reused.
"""

import json
import os
import time

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/agss_yields/ -> pipelines/ -> repo root

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "agss_yields")
os.makedirs(RAW_DIR, exist_ok=True)

FDW_BASE = "https://fdw.fews.net/api/cropproductionfacts.json"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# FDW's own product names, confirmed empirically (NOT guessed) by sampling
# unfiltered ET crop-production records and reading the `product` field.
CROPS = {
    "maize": "Maize (Corn)",
    "wheat": "Wheat Grain",
    "sorghum": "Sorghum",
    "teff": "Mixed Teff",
}

INDICATOR = "crop:yield"  # confirmed via indicator_abbreviation in sampled records

# AgSS/CSA data for Ethiopia in FDW starts in the mid-1990s (fnid vintages
# like ET1994R2xxxx / ET1996R2xxxx); 2027 gives one year of headroom past the
# confirmed 2022-07-31 recency ceiling in case FDW backfills later data.
YEARS = range(1993, 2028)
PERIOD_DATE_SUFFIXES = ["01-31", "07-31"]  # Meher, Belg -- see docstring

PAGE_SIZE = 500  # 1000 triggers a 403; 500 is confirmed safe
MAX_RETRIES = 5


def _verify(records):
    """A cached page is trusted only if it actually parses as the expected
    list-of-dicts shape with the fields this pipeline depends on -- not just
    because the file exists on disk (a truncated/error response still
    'exists')."""
    if not isinstance(records, list):
        return False
    if not records:
        return True
    required = {"admin_2", "period_date", "value", "status", "product",
                "source_organization", "fnid", "season_name"}
    return required.issubset(records[0].keys())


def _get(params):
    resp = None
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(FDW_BASE, params=params, headers=HEADERS, timeout=60)
            if resp.status_code == 200:
                return resp.json()
            last_error = f"status {resp.status_code}"
        except requests.exceptions.RequestException as exc:
            # Transient network issue (read timeout, connection reset) --
            # retry rather than crash the whole multi-hour fetch.
            last_error = f"{type(exc).__name__}: {exc}"
        time.sleep(3 * attempt)
    raise RuntimeError(
        f"gave up after {MAX_RETRIES} retries, last error: {last_error}, params={params}"
    )


def fetch_exact_period(product_name, period_date):
    """Page through one (product, exact period_date) query. Each such slice
    is at most ~100-200 records (one per admin_2 x fnid-vintage for a single
    reporting date), safely under the 1,000-record cap without needing to
    bisect further."""
    records = []
    offset = 0
    while True:
        params = {
            "country_code": "ET",
            "product": product_name,
            "indicator": INDICATOR,
            "period_date": period_date,
            "format": "json",
            "page_size": PAGE_SIZE,
            "offset": offset,
        }
        data = _get(params)
        count = data.get("count", 0)
        results = data.get("results", [])
        records.extend(results)
        offset += PAGE_SIZE
        if offset >= count or not results:
            break
        if offset > 1000:
            raise RuntimeError(
                f"{product_name} period_date={period_date}: {count} records, "
                f"exceeds the 1,000-record safe zone -- needs finer partitioning."
            )
        time.sleep(1.0)
    return records


def fetch_all_periods(product_name):
    records = []
    for year in YEARS:
        for suffix in PERIOD_DATE_SUFFIXES:
            period_date = f"{year}-{suffix}"
            recs = fetch_exact_period(product_name, period_date)
            if recs:
                print(f"    period_date={period_date} -> {len(recs)} records")
                records.extend(recs)
            time.sleep(0.3)
    return records


def fetch_crop(crop_key, product_name):
    cache_path = os.path.join(RAW_DIR, f"cropproductionfacts_{crop_key}.json")
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if _verify(cached):
                print(f"  {crop_key}: cached, {len(cached)} records -- skipping fetch")
                return cached
            print(f"  {crop_key}: cache failed verification, re-fetching")
        except (json.JSONDecodeError, OSError):
            print(f"  {crop_key}: cache unreadable, re-fetching")
        os.remove(cache_path)

    all_records = fetch_all_periods(product_name)

    if not _verify(all_records):
        raise RuntimeError(f"{crop_key}: fetched data failed shape verification, not caching")

    tmp_path = cache_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(all_records, f)
    os.replace(tmp_path, cache_path)  # atomic -- never leaves a partial file at the real path
    print(f"  {crop_key}: fetched {len(all_records)} total, cached -> {cache_path}")
    return all_records


def main():
    print("=" * 74)
    print("AgSS yield ACQUIRE -- FDW cropproductionfacts, crop:yield, Ethiopia")
    print("=" * 74)
    for crop_key, product_name in CROPS.items():
        print(f"\n{crop_key} ({product_name}):")
        fetch_crop(crop_key, product_name)
    print("\nDone. Run compute_agss_yields_admin2_monthly_features.py next.")


if __name__ == "__main__":
    main()
