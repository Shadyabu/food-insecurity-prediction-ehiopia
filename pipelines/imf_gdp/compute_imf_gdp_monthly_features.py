"""
compute_imf_gdp_monthly_features.py

Engineering stage for the IMF GDP per capita feature. Turns the single
annual national series fetched by fetch_imf_gdp.py into a month-indexed
table and writes data/processed/features/imf_gdp_monthly.csv.

NATIONAL-ONLY LIMITATION (flag explicitly, do not let this get mistaken
for a zone-informative feature later): there is no spatial join anywhere
in this pipeline. Every zone_code sees the identical value in a given
month -- the eventual multi-source feature join must broadcast this
table's rows to every zone_code for the matching month (plain many-to-one
merge on year+month, executed at join time, NOT baked into this file),
exactly like teleconnections. Because it carries zero within-Ethiopia
variation, this feature can only ever act as a national-macro-trend
control in a model or in SHAP -- it cannot explain, and must not be
interpreted as explaining, any livelihood-zone or subnational difference
in predicted risk (CLAUDE.md Sec 7 RQ3 stratification).

FILL RULE: forward-fill each annual value unchanged across its 12 months
(no interpolation). Confirmed against Busker et al.'s own staged monthly
file (busker_comparison/GDP_df.xlsx) -- they used the identical rule (a
single annual figure repeated across all 12 months, e.g. 227.616 for
every month of 1980), so this is both the simplest choice and RQ1-exact.

VINTAGE/LEAKAGE (see docs/IMF_GDP_Pipeline_Documentation.md Sec 3 for the
full reasoning; confirmed with the project owner 2026-08-08): true
real-time-vintage reconstruction (the value as it would actually have
been known at each historical point in time) is not currently buildable
-- imf.org's WEO archive pages return HTTP 403 from this environment, and
the one IMF endpoint that IS reachable (the DataMapper API used by
fetch_imf_gdp.py) only ever serves the CURRENT, fully-revised series, with
no historical-vintage capability. Busker et al. do not solve this either:
their own staged file is a single current-release snapshot (already
extending to 2028 forecast years at their 2023 download) applied
uniformly across their whole 2011-2021 study window -- i.e. no vintage
control at all, same undated-series pattern already documented for their
teleconnections use.

Given that, this file outputs BOTH:
  - gdp_per_capita_undated / gdp_per_capita_yoy_change_undated: the
    current release's value assigned to its own calendar year with no
    lag protection at all. Matches Busker et al.'s method exactly -- for
    RQ1 fidelity / reference-table completeness only. Do NOT use in any
    RQ2+ model.
  - gdp_per_capita / gdp_per_capita_yoy_change: the RQ2+ default. Applies
    the annual/low-frequency publication-lag rule from CLAUDE.md Sec 3.2 --
    year Y's value only becomes visible starting April of year Y+1 (WEO's
    April release is a conservative cutoff; by then at least one full WEO
    cycle covering year Y has been published). Before that cutoff, a
    given month still sees year Y-1's (or Y-2's, in Jan-Mar) value.

IMPORTANT residual caveat even for the lag-safe column: because only the
CURRENT release is fetchable (see above), "year Y-1's value" here still
means today's most-revised figure for Y-1, not the figure as it actually
read at the time. This reduces revision-leakage risk (by the time a value
is used, it typically refers to a year old enough to have mostly
stabilized) but does not eliminate it. This is empirically not a
theoretical concern for Ethiopia: the July 2024 birr devaluation alone
moved the DataMapper API's own value for 2024 from Busker's staged
~$1,715 (2023-vintage forecast) to ~$1,311 (current release, 2026-08-08
pull) -- a real, large, still-ongoing revision, not noise.

Same compromise CLAUDE.md already documents for teleconnections' MEI
lag0 vs lag1/3/6 -- kept for completeness, gated for modeling use.
"""

import json
import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_PATH = os.path.join(REPO_ROOT, "data", "raw", "imf_gdp", "weo_ngdpdpc_datamapper.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "imf_gdp_monthly.csv")

FEATURE_START_YEAR = 2009  # matches the other feature pipelines' (teleconnections, NDVI, GLEAM) feature-year window
PUBLICATION_LAG_CUTOFF_MONTH = 4  # WEO April release -- year Y first becomes visible in April of Y+1


def load_annual_series():
    with open(RAW_PATH, "r") as f:
        data = json.load(f)
    eth = data["values"]["NGDPDPC"]["ETH"]
    series = pd.Series({int(y): v for y, v in eth.items()}).sort_index()
    series.name = "level"
    return series


def build_monthly_grid(start_year, end_date):
    full_range = pd.date_range(f"{start_year}-01-01", end_date, freq="MS")
    return pd.DataFrame({"date": full_range, "year": full_range.year, "month": full_range.month})


def known_year_asof(row_year, row_month):
    """Publication-lag rule: year Y is not visible until April of Y+1."""
    if row_month >= PUBLICATION_LAG_CUTOFF_MONTH:
        return row_year - 1
    return row_year - 2


def main():
    annual = load_annual_series()
    yoy = annual.pct_change()

    today = pd.Timestamp.now().normalize().replace(day=1)
    grid = build_monthly_grid(FEATURE_START_YEAR, today)

    # Undated (RQ1-parity, Busker-exact): each row's own calendar year, no lag.
    grid["gdp_per_capita_undated"] = grid["year"].map(annual)
    grid["gdp_per_capita_yoy_change_undated"] = grid["year"].map(yoy)

    # Publication-lag-safe (RQ2+ default): shifted per known_year_asof().
    known_year = grid.apply(lambda r: known_year_asof(r["year"], r["month"]), axis=1)
    grid["gdp_per_capita"] = known_year.map(annual)
    grid["gdp_per_capita_yoy_change"] = known_year.map(yoy)

    cols = [
        "year", "month",
        "gdp_per_capita", "gdp_per_capita_yoy_change",
        "gdp_per_capita_undated", "gdp_per_capita_yoy_change_undated",
    ]
    out = grid[cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    first, last = out.iloc[0], out.iloc[-1]
    n_lagsafe_nan = out["gdp_per_capita"].isna().sum()
    print(f"Wrote {len(out)} rows ({int(first['year'])}-{int(first['month']):02d} to "
          f"{int(last['year'])}-{int(last['month']):02d}) to {OUTPUT_PATH}")
    print(f"gdp_per_capita (lag-safe) NaN rows: {n_lagsafe_nan} "
          f"(expected only at the very start of the series, before any prior-year data exists)")


if __name__ == "__main__":
    main()
