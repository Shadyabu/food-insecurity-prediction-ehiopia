"""
fetch_imf_gdp.py

Acquisition stage for the IMF GDP per capita feature. Single national
value per year for Ethiopia -- no spatial dimension, no sub-annual
resolution, same "no Aggregate stage" shape as teleconnections.

SOURCE (confirmed against Busker et al.'s own staged reference,
pipelines/macro_indicators/busker_comparison/GDP_PER_CAPITA_IMF.xlsx,
sheet NGDPDPC): IMF World Economic Outlook (WEO) database, indicator
NGDPDPC = "GDP per capita, current prices (U.S. dollars per capita)" --
NOMINAL, current USD, NOT PPP-adjusted, NOT local currency. This is
exactly Busker et al.'s own source and unit.

ACCESS PATH (checked 2026-08-08, see
docs/IMF_GDP_Pipeline_Documentation.md Sec 2 for the full investigation):
imf.org's WEO web pages and bulk vintage-archive files
(weo-database/data-archives, WEOxxxxall.ashx) all return HTTP 403 from
this environment (Akamai bot-block) -- NOT fetchable here, at any vintage.
The DataMapper JSON API (imf.org/external/datamapper/api/v1/NGDPDPC) IS
reachable, but only ever serves the CURRENT, fully-revised series -- no
historical-vintage capability. This constrains the vintage/leakage
handling to Stage 3 (compute step), not this stage: there is nothing to
choose between vintages at acquisition time, only one series is fetchable.

SETUP
    pip install requests --break-system-packages
"""

import json
import os

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/imf_gdp/ -> pipelines/ -> repo root
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "imf_gdp")

DATAMAPPER_URL = "https://www.imf.org/external/datamapper/api/v1/NGDPDPC"
RAW_PATH = os.path.join(RAW_DIR, "weo_ngdpdpc_datamapper.json")
MIN_YEARS_ETH = 30  # ETH series starts 1980; expect several decades of data + forecast years


def verify_file_opens(path):
    """Actually parse the cached JSON and check the ETH series looks
    plausible -- a truncated/failed download can still "exist" on disk."""
    with open(path, "r") as f:
        data = json.load(f)
    eth = data["values"]["NGDPDPC"]["ETH"]
    if len(eth) < MIN_YEARS_ETH:
        raise ValueError(f"{path}: ETH series has only {len(eth)} years, expected >= {MIN_YEARS_ETH}")
    return data


def download_source(max_retries=3):
    if os.path.exists(RAW_PATH):
        try:
            verify_file_opens(RAW_PATH)
            print(f"cached file OK: {RAW_PATH}")
            return RAW_PATH
        except Exception as e:
            print(f"cached file failed verification ({e}), re-downloading")
            os.remove(RAW_PATH)

    os.makedirs(RAW_DIR, exist_ok=True)
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(DATAMAPPER_URL, timeout=30)
            resp.raise_for_status()
            with open(RAW_PATH, "w") as f:
                f.write(resp.text)
            verify_file_opens(RAW_PATH)
            print(f"downloaded OK: {RAW_PATH}")
            return RAW_PATH
        except Exception as e:
            if os.path.exists(RAW_PATH):
                os.remove(RAW_PATH)  # never leave a partial file behind
            print(f"attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise

    return None


def main():
    download_source()


if __name__ == "__main__":
    main()
