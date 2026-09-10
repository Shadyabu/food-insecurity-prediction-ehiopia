"""
fetch_wfp_prices_admin2.py

ACQUIRE stage for WFP food/fuel prices. Single live source, no gap-stitching
needed (unlike locust/ACLED): WFP's own bulk CSV export on HDX covers
2000-01-15 to the present in one file and is refreshed roughly monthly.

SOURCE: HDX dataset "wfp-food-prices-for-ethiopia" (CKAN id
2e4f1922-e446-4b57-a98a-d0e2d5e34afa), two resources:
  1. wfp_food_prices_eth.csv -- the price panel itself (date, admin1, admin2,
     market, market_id, latitude, longitude, category, commodity, unit,
     priceflag, pricetype, currency, price, usdprice).
  2. wfp_markets_eth.csv -- one row per market_id with lat/lon. Needed
     because the price panel's own `admin2` field is WFP's own zone-naming
     convention (e.g. "ZONE2", "GAMO GOFA") which does not cleanly
     string-match the current admin2 boundary vintage (GAMO GOFA predates
     the boundary's 2021 split into separate Gamo/Gofa zones; Gambela's
     "ZONE 1/2/3" don't correspond to the boundary's named zones; Dire Dawa
     is one WFP market but two admin2 zones (urban/rural) in the boundary).
     The compute stage does a point-in-polygon join on market lat/lon
     instead (CLAUDE.md Sec 3.4 -- point/market data joins spatially, not
     by name), which resolves all of these correctly and geometrically,
     reusing the same pattern as pipelines/locust/.

NOT used as the primary source: pipelines/wfp_prices/busker_comparison/
WFP_Ethiopia_FoodPrices_new_tool_{retail,wholesale}.csv. Those are Busker et
al.'s OWN historical extract from the same underlying WFP database (retail
2006-2024, wholesale 2000-2022) -- kept as the independent validation
reference in validate_wfp_prices_output.py, not re-used for extraction,
since the live HDX pull already covers the full 2000-2026 dissertation
window in one continuous file.

Busker et al. (2024) themselves used a much narrower slice of this same WFP
database: maize-only (+ diesel fuel only), standardized via WFP's own ALPS
indicator, with a nearest-single-market-per-admin-unit spatial rule (see
docs/busker_2024.pdf Sec 2.1.2.2). This pipeline deliberately covers more
commodities (maize, wheat, sorghum, teff) with raw price + a custom
backward-looking price index at multiple lags, per the Ethiopia-specific
extension design (CLAUDE.md Sec 1, RQ2-RQ5) -- it is not an RQ1-parity
pipeline the way CHIRPS/NDVI/locust are.
"""

import os

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/wfp_prices/ -> pipelines/ -> repo root

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "wfp_prices")
os.makedirs(RAW_DIR, exist_ok=True)

PRICES_URL = (
    "https://data.humdata.org/dataset/2e4f1922-e446-4b57-a98a-d0e2d5e34afa/"
    "resource/87bac18e-f3aa-4b29-8cf8-76763e823dc5/download/wfp_food_prices_eth.csv"
)
MARKETS_URL = (
    "https://data.humdata.org/dataset/2e4f1922-e446-4b57-a98a-d0e2d5e34afa/"
    "resource/ffa5efea-691a-46ea-b8d0-7e3f25eb56c8/download/wfp_markets_eth.csv"
)

PRICES_CACHE_PATH = os.path.join(RAW_DIR, "wfp_food_prices_eth_live.csv")
MARKETS_CACHE_PATH = os.path.join(RAW_DIR, "wfp_markets_eth_live.csv")

HEADERS = {"User-Agent": "Mozilla/5.0 (dissertation pipeline; contact office@i-methods.com)"}

PRICES_EXPECTED_COLUMNS = {
    "date", "admin1", "admin2", "market", "market_id", "latitude", "longitude",
    "category", "commodity", "unit", "priceflag", "pricetype", "currency",
    "price", "usdprice",
}
MARKETS_EXPECTED_COLUMNS = {"market_id", "market", "countryiso3", "admin1", "admin2", "latitude", "longitude"}


def verify_csv_opens(path, expected_columns, min_rows=1):
    """Actually parse the file and check its columns, not just that it
    exists -- a truncated download still 'exists' on disk."""
    try:
        df = pd.read_csv(path, nrows=5)
        if not expected_columns.issubset(set(df.columns)):
            return False
        full = pd.read_csv(path)
        return len(full) >= min_rows
    except Exception:
        return False


def download_csv(url, out_path, expected_columns, max_retries=3):
    if os.path.exists(out_path) and verify_csv_opens(out_path, expected_columns):
        df = pd.read_csv(out_path)
        print(f"Cache OK: {len(df)} rows -> {out_path}")
        return df

    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            tmp_path = out_path + ".tmp"
            with open(tmp_path, "wb") as f:
                f.write(resp.content)
            if not verify_csv_opens(tmp_path, expected_columns):
                raise ValueError("Downloaded file failed verification (missing expected columns or empty).")
            os.replace(tmp_path, out_path)  # atomic -- never leave a partial file in place
            break
        except Exception as e:
            if os.path.exists(out_path + ".tmp"):
                os.remove(out_path + ".tmp")
            if attempt == max_retries:
                raise
            print(f"  retry {attempt} for {url}: {e}")

    df = pd.read_csv(out_path)
    print(f"Downloaded {len(df)} rows -> {out_path}")
    return df


def fetch_prices():
    return download_csv(PRICES_URL, PRICES_CACHE_PATH, PRICES_EXPECTED_COLUMNS)


def fetch_markets():
    return download_csv(MARKETS_URL, MARKETS_CACHE_PATH, MARKETS_EXPECTED_COLUMNS)


if __name__ == "__main__":
    prices_df = fetch_prices()
    markets_df = fetch_markets()

    retail = prices_df[prices_df["pricetype"] == "Retail"]
    print(f"\nRetail rows: {len(retail)} ({retail['date'].min()} to {retail['date'].max()})")
    print(f"Distinct retail market_id values: {retail['market_id'].nunique()}")
    print(f"Markets file covers {markets_df['market_id'].nunique()} market_id values total")
    missing = set(retail["market_id"].dropna().unique()) - set(markets_df["market_id"].dropna().unique())
    if missing:
        print(f"WARNING: {len(missing)} retail market_id values have no coordinate row: {missing}")
    else:
        print("All retail market_id values have a coordinate row in the markets file.")
