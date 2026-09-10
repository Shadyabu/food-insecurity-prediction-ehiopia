"""
fetch_exchange_rate.py

ACQUIRE stage for the exchange-rate pipeline.

No anonymous bulk-download endpoint was found for a monthly ETB/USD series
back to 2009 (checked and ruled out: IMF DataMapper/IFS -- 403 from this
environment, same wall already documented for imf_gdp; National Bank of
Ethiopia's own site -- 404 on the exchange-rates page; WFP's VAM dataviz
tool at dataviz.vam.wfp.org -- Cloudflare-blocked, 403 to every scripted
client tried, including WebFetch; no dedicated HDX dataset exists either).
This stage is therefore a manual export, same pattern as ACLED
(fetch_acled_admin2.py):

  1. Get a monthly USD/ETB historical series from investing.com (or a
     re-export of the same page) covering 2009-01 to present.
  2. Drop the exported CSV in data/raw/exchange_rate/ (any filename --
     matched below by glob, newest file wins).

This script's only job is to cache that file under a stable, code-facing
name and verify it actually opens and has the expected shape before
anything downstream trusts it (CLAUDE.md Sec 3.6).

SOURCE IDENTITY (confirmed 2026-08-11, see
compute_exchange_rate_monthly_features.py docstring for the full
cross-validation): investing.com's "Price" column tracks Ethiopia's
OFFICIAL/interbank USD rate, not the black-market/parallel rate --
confirmed by comparing it against an independently-derived implied-official
rate (price / usdprice, from WFP's own "Exchange rate (unofficial)" retail
panel rows) at multiple points, including the July 2024 devaluation
discontinuity, where both series jump from ~57 to ~111+ in the same month.
investing.com's own page does not label which rate this is -- do not take
that at face value if the file is ever re-exported from a different site.

FORMAT QUIRK: date field is DD/MM/YYYY with DD always "01" -- confirmed by
row-to-row continuity, NOT by the column header (which gives no timezone/
convention info). "Price" is the value at MONTH-END, not month-start: each
row's Open equals the *previous* row's Price exactly (checked across the
full file), so a row labelled "01/08/2026" is functionally August's closing
value, dated to the 1st of that month by the export's own convention. This
is the same "value observed during month t, available from t+1" shape as
CHIRPS monthly totals or WFP prices -- handled by the project's uniform
lead-time shift at model-build time, not by anything special in this
pipeline.
"""

import glob
import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/exchange_rate/ -> pipelines/ -> repo root

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "exchange_rate")
CACHE_PATH = os.path.join(RAW_DIR, "usd_etb_investing_monthly.csv")

REQUIRED_COLUMNS = ["Date", "Price", "Open", "High", "Low", "Vol.", "Change %"]


def find_latest_export():
    matches = sorted(
        p for p in glob.glob(os.path.join(RAW_DIR, "*.csv"))
        if os.path.abspath(p) != os.path.abspath(CACHE_PATH)
    )
    if not matches:
        raise FileNotFoundError(
            f"No exchange-rate export found in {RAW_DIR}. "
            "Export a monthly USD/ETB historical series from investing.com "
            "and drop the CSV there first."
        )
    return max(matches, key=os.path.getmtime)


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    source_path = find_latest_export()
    print(f"Found export: {source_path}")

    df = pd.read_csv(source_path)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{source_path} is missing expected columns: {missing_cols}")

    dates = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    if not (dates.dt.day == 1).all():
        raise ValueError(
            f"{source_path}: expected every row's day field to be 1 (monthly export) -- "
            "format assumption (DD/MM/YYYY) may no longer hold, check by hand before proceeding"
        )

    df.to_csv(CACHE_PATH, index=False)

    ym = dates.dt.to_period("M")
    full_range = pd.period_range(ym.min(), ym.max(), freq="M")
    missing_months = sorted(set(full_range) - set(ym))
    dupes = ym.duplicated().sum()

    print(f"Cached -> {CACHE_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {ym.min()} to {ym.max()} ({len(full_range)} expected months)")
    print(f"Missing months: {len(missing_months)} {missing_months if missing_months else ''}")
    print(f"Duplicate months: {dupes}")


if __name__ == "__main__":
    main()
