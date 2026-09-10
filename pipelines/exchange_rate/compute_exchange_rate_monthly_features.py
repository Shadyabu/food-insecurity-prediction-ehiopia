"""
compute_exchange_rate_monthly_features.py

Engineering stage for the ETB/USD exchange-rate feature. Turns the cached
investing.com monthly series (fetch_exchange_rate.py) into
data/processed/features/exchange_rate_monthly.csv.

NATIONAL-ONLY, no spatial join anywhere in this file -- same broadcast
pattern as cpi/imf_gdp/teleconnections/wvg. The eventual multi-source
feature join must broadcast this table's rows to every zone_code for the
matching month (plain many-to-one merge on year+month at join time, NOT
baked into this file).

ETHIOPIA-SPECIFIC ADDITION -- Busker et al. do not use exchange-rate data
at all (checked: zero matches for "exchange" anywhere in busker_scripts/).
There is therefore no RQ1-parity split here (no _undated companion column,
unlike cpi/imf_gdp/teleconnections, which all carry a Busker-vintage
snapshot) -- this is a pure RQ2+ feature, same posture as wvg/agss_yields.

SERIES IDENTITY: this is the OFFICIAL/interbank ETB/USD rate, not the
black-market/parallel rate -- see fetch_exchange_rate.py's docstring and
validate_exchange_rate_output.py for the cross-check that established
this (investing.com's own page does not state which rate it is).

VINTAGE: investing.com serves only the current historical series -- there
is no revision-vintage concern of the kind imf_gdp/cpi document (a
market-quoted FX rate is not later "revised" the way a national accounts
statistic is), so no lag0-vs-lag1 caveat is needed here. lag1/3/6/12 are
provided purely as feature-engineering conveniences (matches CPI's
monthly-market-data granularity), not as a leakage workaround.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_PATH = os.path.join(REPO_ROOT, "data", "raw", "exchange_rate", "usd_etb_investing_monthly.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "exchange_rate_monthly.csv")

FEATURE_START_YEAR = 2009  # matches the other feature pipelines' (teleconnections, NDVI, GLEAM) feature-year window
LAGS = [1, 3, 6, 12]


def load_raw_series():
    df = pd.read_csv(RAW_PATH)
    df["date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df = df.rename(columns={"Price": "exchange_rate"})[["date", "year", "month", "exchange_rate"]]
    return df.sort_values("date").reset_index(drop=True)


def build_monthly_table():
    raw = load_raw_series()

    # Reindex onto a continuous monthly grid so lag shifts are never
    # silently misaligned by a gap (mirrors teleconnections/cpi) -- the raw
    # file is already gap-free (verified in fetch_exchange_rate.py) but
    # this makes that an enforced invariant, not an assumption.
    full_range = pd.date_range(raw["date"].min(), raw["date"].max(), freq="MS")
    grid = raw.set_index("date").reindex(full_range)
    grid.index.name = "date"
    if grid["exchange_rate"].isna().any():
        raise ValueError("Reindexing onto a continuous monthly grid introduced gaps -- raw file is not gap-free")

    grid["year"] = grid.index.year
    grid["month"] = grid.index.month

    # Explicit division, not pandas .pct_change(12) -- consistent with
    # cpi's documented reasoning (older pandas defaults silently
    # forward-fill NaNs before dividing unless fill_method=None is passed).
    # No NaNs exist in this series, but the explicit form costs nothing and
    # keeps every national-series pipeline in this repo doing the same
    # thing the same way.
    grid["exchange_rate_yoy_change"] = grid["exchange_rate"] / grid["exchange_rate"].shift(12) - 1

    for col in ["exchange_rate", "exchange_rate_yoy_change"]:
        for lag in LAGS:
            grid[f"{col}_lag{lag}"] = grid[col].shift(lag)

    grid = grid.reset_index(drop=True)
    grid = grid[grid["year"] >= FEATURE_START_YEAR].reset_index(drop=True)
    return grid


def main():
    df = build_monthly_table()

    cols = ["year", "month"]
    for col in ["exchange_rate", "exchange_rate_yoy_change"]:
        cols += [col] + [f"{col}_lag{lag}" for lag in LAGS]
    df = df[cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    first, last = df.iloc[0], df.iloc[-1]
    print(f"Wrote {len(df)} rows ({int(first['year'])}-{int(first['month']):02d} to "
          f"{int(last['year'])}-{int(last['month']):02d}) to {OUTPUT_PATH}")
    for col in ["exchange_rate", "exchange_rate_yoy_change"]:
        n_nan = df[col].isna().sum()
        print(f"  {col}: {n_nan} NaN rows")


if __name__ == "__main__":
    main()
