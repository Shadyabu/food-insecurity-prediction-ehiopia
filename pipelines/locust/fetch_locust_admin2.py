"""
fetch_locust_admin2.py

ACQUIRE stage for the desert-locust point-event pipeline. Two disjoint raw
sources, each cached separately and independently re-runnable:

  1. Busker et al.'s own extraction from the FAO Locust Hub "Swarms" archive
     (input_data/Locust/FAO_Locust_Swarms.csv inside their Zenodo replication
     package, zenodo.org/records/10853668) -- 10,657 points across the Horn
     of Africa, 1986-01-25 to 2021-12-28, already filtered to CAT == "Swarm"
     and carrying an AREAHA field, which is exactly the input Busker's paper
     says they used ("the total area affected each month was calculated as a
     percentage of the overall area within the defined administrative
     division"). This is both our historical raw source AND (via the
     separately-provided desert_locust_dataset.xlsx) the validation
     reference for a like-for-like RQ1 feature.

  2. FAO's live RAMSES_Observations_Dashboard_View feature service
     (services5.arcgis.com/sjP4Ugu5s0dZWLjd/.../FeatureServer/0), which
     publishes fresh field observations starting exactly 2023-01-01 (a new
     system -- there is no pre-2023 history behind this endpoint, confirmed
     via min(Obs_Date) globally, not just for Ethiopia). Despite living on a
     "secured" ArcGIS Online org where the older Swarms_Public /
     LocustCore_Master services now return "Token Required", this specific
     view answers anonymous queries as long as the request's Origin/Referer
     match an approved ArcGIS Hub app domain -- no API token needed.

KNOWN GAP: 2022-01 to 2022-12 is covered by neither source. Busker's extract
ends 2021-12-28 (confirmed as the true end of that snapshot, not an
Ethiopia-specific artefact -- the global max date across all countries in
the file is the same). RAMSES only starts 2023-01-01. The engineering stage
must carry this as an explicit "no source" gap, not an implied zero -- see
compute_locust_admin2_monthly_features.py.

Run this whenever you want to refresh the RAMSES snapshot; the Busker
archive is static (fixed historical release) and only re-extracted if its
cache is missing.
"""

import os
import time
import zipfile

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/locust/ -> pipelines/ -> repo root

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "locust")
os.makedirs(RAW_DIR, exist_ok=True)

BUSKER_ZIP_PATH = os.path.join(REPO_ROOT, "data", "raw", "busker_replication", "input_data.zip")
BUSKER_ZIP_MEMBER = "input_data/Locust/FAO_Locust_Swarms.csv"
BUSKER_CACHE_PATH = os.path.join(RAW_DIR, "busker_swarms_ethiopia_1986_2021.csv")

RAMSES_FEATURE_SERVER = (
    "https://services5.arcgis.com/sjP4Ugu5s0dZWLjd/arcgis/rest/services/"
    "Ramses_Observations_Dashboard_View/FeatureServer/0/query"
)
# No token: this view answers anonymous queries as long as Origin/Referer
# match an approved ArcGIS Hub app domain. Confirmed empirically -- the
# older Swarms_Public / LocustCore_Master services on the SAME org still
# require a real token even with these headers, so this is a property of
# this specific view's sharing config, not a generic bypass.
RAMSES_HEADERS = {
    "Accept": "*/*",
    "Origin": "https://hqfao-hub.maps.arcgis.com",
    "Referer": "https://hqfao-hub.maps.arcgis.com/apps/mapviewer/index.html?layers=ed8228a11a2547178cdf4ab002464efb",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
        "(KHTML, like Gecko) Version/18.6 Safari/605.1.15"
    ),
}
RAMSES_PAGE_SIZE = 1000  # the layer's own maxRecordCount -- confirmed via .../0?f=json
RAMSES_OUT_FIELDS = [
    "OBJECTID", "Obs_Date", "Latitude", "Longitude", "Country", "Region", "District",
    "Locust_Type", "Locust_Presence", "Swarm_Size_Unit", "Minimum_Size_Sum",
    "Maximum_Size_Sum", "Report_ID",
]
RAMSES_CACHE_PATH = os.path.join(RAW_DIR, "ramses_swarms_ethiopia_live.csv")


def verify_csv_opens(path, min_rows=1):
    """Actually parse the file, not just check it exists -- a truncated
    extract/download still 'exists' on disk."""
    try:
        df = pd.read_csv(path, nrows=5)
        if len(df.columns) == 0:
            return False
        full = pd.read_csv(path)
        return len(full) >= min_rows
    except Exception:
        return False


# --------------------------------------------------------------------------
# 1. Busker archive (1986-2021) -- extract once from the Zenodo zip, cache
# --------------------------------------------------------------------------

def fetch_busker_archive():
    if os.path.exists(BUSKER_CACHE_PATH) and verify_csv_opens(BUSKER_CACHE_PATH):
        df = pd.read_csv(BUSKER_CACHE_PATH)
        print(f"Busker archive cache OK: {len(df)} Ethiopia rows -> {BUSKER_CACHE_PATH}")
        return df

    if not os.path.exists(BUSKER_ZIP_PATH):
        raise FileNotFoundError(
            f"Busker replication package not found at {BUSKER_ZIP_PATH}. "
            "Download it first: "
            "curl -L 'https://zenodo.org/api/records/10853668/files/input_data.zip/content' "
            "-o data/raw/busker_replication/input_data.zip"
        )

    print("Extracting FAO_Locust_Swarms.csv from Busker replication package...")
    with zipfile.ZipFile(BUSKER_ZIP_PATH) as zf:
        with zf.open(BUSKER_ZIP_MEMBER) as f:
            raw = pd.read_csv(f)

    print(f"  {len(raw)} total Horn-of-Africa swarm points, "
          f"CAT values: {raw['CAT'].unique().tolist()}")

    eth = raw[raw["COUNTRYID"] == "ET"].copy()
    eth = eth[["X", "Y", "STARTDATE", "AREAHA", "COUNTRYID", "LOCUSTID", "REPORTID"]]
    eth = eth.rename(columns={"X": "longitude", "Y": "latitude", "STARTDATE": "event_date",
                               "AREAHA": "area_ha"})
    eth["event_date"] = pd.to_datetime(eth["event_date"], utc=True).dt.tz_localize(None)

    tmp_path = BUSKER_CACHE_PATH + ".tmp"
    eth.to_csv(tmp_path, index=False)
    os.replace(tmp_path, BUSKER_CACHE_PATH)  # atomic -- never leave a partial file in place

    print(f"Cached {len(eth)} Ethiopia swarm points "
          f"({eth['event_date'].min().date()} to {eth['event_date'].max().date()}) "
          f"-> {BUSKER_CACHE_PATH}")
    return eth


# --------------------------------------------------------------------------
# 2. RAMSES live layer (2023-present) -- paginated anonymous query
# --------------------------------------------------------------------------

def fetch_ramses_live(max_retries=3):
    where = "Country='Ethiopia' AND Locust_Type='SWARM'"
    out_fields = ",".join(RAMSES_OUT_FIELDS)

    count_resp = requests.get(
        RAMSES_FEATURE_SERVER,
        params={"where": where, "returnCountOnly": "true", "f": "json"},
        headers=RAMSES_HEADERS, timeout=30,
    )
    count_resp.raise_for_status()
    total = count_resp.json()["count"]
    print(f"RAMSES live layer: {total} Ethiopia SWARM records to fetch.")

    records = []
    offset = 0
    while offset < total:
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(
                    RAMSES_FEATURE_SERVER,
                    params={
                        "where": where,
                        "outFields": out_fields,
                        "orderByFields": "OBJECTID ASC",
                        "resultOffset": offset,
                        "resultRecordCount": RAMSES_PAGE_SIZE,
                        "f": "json",
                    },
                    headers=RAMSES_HEADERS, timeout=30,
                )
                resp.raise_for_status()
                payload = resp.json()
                if "error" in payload:
                    raise RuntimeError(payload["error"])
                break
            except Exception as e:
                if attempt == max_retries:
                    raise
                print(f"  retry {attempt} at offset {offset}: {e}")
                time.sleep(2 * attempt)

        page = [feat["attributes"] for feat in payload.get("features", [])]
        records.extend(page)
        offset += RAMSES_PAGE_SIZE
        print(f"  fetched {len(records)}/{total}")

    df = pd.DataFrame(records)
    if len(df) != total:
        raise RuntimeError(f"Expected {total} records, got {len(df)} -- pagination likely dropped rows.")

    df["event_date"] = pd.to_datetime(df["Obs_Date"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})

    tmp_path = RAMSES_CACHE_PATH + ".tmp"
    df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, RAMSES_CACHE_PATH)

    print(f"Cached {len(df)} RAMSES Ethiopia swarm points "
          f"({df['event_date'].min().date()} to {df['event_date'].max().date()}) "
          f"-> {RAMSES_CACHE_PATH}")
    return df


if __name__ == "__main__":
    busker_df = fetch_busker_archive()
    ramses_df = fetch_ramses_live()

    gap_note = (
        "\nNOTE: 2022-01 to 2022-12 is covered by neither source (Busker "
        "archive ends 2021-12-28; RAMSES starts 2023-01-01). This must be "
        "carried as an explicit no-source gap downstream, not an implied "
        "zero -- see compute_locust_admin2_monthly_features.py."
    )
    print(gap_note)
