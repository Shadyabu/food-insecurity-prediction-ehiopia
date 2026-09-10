"""
compute_unhcr_displacement_monthly.py

Aggregation + engineering stage for the UNHCR persons-of-concern source.
Collapses the per-origin-country annual rows produced by
prepare_persons_of_concern.py into a small set of NATIONAL annual scalars,
then broadcasts them to a month-indexed table and writes
data/processed/features/unhcr_displacement_monthly.csv -- same national
broadcast pattern as cpi/imf_gdp/teleconnections/wvg (CLAUDE.md Sec 2/3.1):
every zone_code sees the identical value in a given month, joined at
model-join time on (year, month) only, not baked in here.

WHY THE PER-ORIGIN-COUNTRY GRANULARITY DOESN'T SURVIVE INTO THIS FILE: the
`country_of_origin_iso3` category dtype built in prepare_persons_of_concern.py
is only meaningful if origin-country stays a row dimension -- but this
project's model tables have one row per (zone_code, month), and this source
has no zone dimension at all (see that script's docstring). To broadcast
onto the zone-month panel the way cpi/imf_gdp do, the ~29 origin-country
rows per year are summed into national totals here. Two conceptually
different signals are kept separate, not combined into one blob:
  - total_refugees_hosted / total_asylum_seekers_hosted: people Ethiopia
    HOSTS from OTHER countries (sum of `refugees`/`asylum_seekers` across
    every origin-country row that year) -- a regional-instability-spillover
    proxy.
  - idps_ethiopia: Ethiopia's OWN internally displaced population --
    read directly off the country_of_origin == "Ethiopia" row's `idps`
    value (confirmed empirically: `idps` is nonzero ONLY on that row, every
    other origin country's `idps` is always 0, and `refugees`/
    `asylum_seekers` are always 0 on the Ethiopia row -- the three columns
    partition cleanly by construction, not by assumption).
  A DIFFERENT signal entirely, conceptually closer to conflict/crisis
  dynamics already covered (partly) by acled_conflict.

STRUCTURAL GAP, NOT IMPUTED: `idps_ethiopia` has no value for 2008-2011 or
2015 (no country_of_origin == "Ethiopia" row those years in the raw file --
UNHCR simply did not report an Ethiopia IDP figure, not a true zero). Left
NaN, per this project's standing no-imputation rule -- same treatment as
WFP teff pre-2020 or AgSS's never-surveyed pastoral zones.

PUBLICATION LAG (CLAUDE.md Sec 3.2, confirmed with the project owner
2026-08-24): UNHCR's annual "persons of concern" figures -- covering the
FULL prior calendar year -- are finalized and released in mid-June of the
following year (the Global Trends report cycle); a provisional mid-year
update for the CURRENT year also appears in October/December, but that
update describes only Jan-Jun of that same year, not the full-year total
this raw file's one-row-per-year figure represents, so it cannot be used
to move this file's visibility date earlier without leaking H2 dynamics
into an H1-only release. Year Y's full-year total is therefore treated as
visible starting **June of year Y+1** (PUBLICATION_LAG_CUTOFF_MONTH=6),
mirroring imf_gdp's April-of-Y+1 rule exactly, just with UNHCR's own later
cutoff month. Before that cutoff, a given month still sees year Y-1's (or
Y-2's, Jan-May) value -- see known_year_asof().

NOT BUILT YET, FLAGGED NOT FORGOTTEN: the project owner also described a
separate UNHCR "nowcasting" product released on the 15th of every month,
which would support a genuinely monthly (not annual-forward-filled) and
lower-latency displacement feature. No raw file for that product exists
in data/raw/UNHCR/ yet -- this script only covers the annual
persons_of_concern.csv actually present. Wiring in the nowcast product is
a follow-up once that raw data is acquired, not a gap in this stage.

No RQ1-parity ("_undated") columns -- Busker et al. do not use any UNHCR
displacement data at all (same posture as exchange_rate/acled_conflict/
wfp_prices' Ethiopia-specific additions), so there is no Busker-exact
snapshot to preserve alongside the lag-safe default.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)

from prepare_persons_of_concern import prepare_persons_of_concern  # noqa: E402

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "unhcr_displacement_monthly.csv")

FEATURE_START_YEAR = 2009  # matches teleconnections/NDVI/GLEAM/imf_gdp's feature-year window
PUBLICATION_LAG_CUTOFF_MONTH = 6  # UNHCR Global Trends report -- year Y first becomes visible in June of Y+1

LEVEL_COLUMNS = ["total_refugees_hosted", "total_asylum_seekers_hosted", "idps_ethiopia"]


def build_annual_series(clean_df: pd.DataFrame) -> pd.DataFrame:
    totals = clean_df.groupby("year", observed=True).agg(
        total_refugees_hosted=("refugees", "sum"),
        total_asylum_seekers_hosted=("asylum_seekers", "sum"),
    )

    eth_idps = (
        clean_df.loc[clean_df["country_of_origin"] == "Ethiopia"]
        .set_index("year")["idps"]
        .rename("idps_ethiopia")
    )

    full_years = range(int(clean_df["year"].min()), int(clean_df["year"].max()) + 1)
    annual = totals.reindex(full_years)
    annual["idps_ethiopia"] = eth_idps.reindex(full_years)
    return annual


def build_monthly_grid(start_year: int, end_date: pd.Timestamp) -> pd.DataFrame:
    full_range = pd.date_range(f"{start_year}-01-01", end_date, freq="MS")
    return pd.DataFrame({"date": full_range, "year": full_range.year, "month": full_range.month})


def known_year_asof(row_year: int, row_month: int) -> int:
    """Publication-lag rule: year Y is not visible until June of Y+1."""
    if row_month >= PUBLICATION_LAG_CUTOFF_MONTH:
        return row_year - 1
    return row_year - 2


def compute_unhcr_displacement_monthly(write_output: bool = True) -> pd.DataFrame:
    clean = prepare_persons_of_concern(write_output=False)
    annual = build_annual_series(clean)
    # fill_method=None: a NaN year (structural gap, see module docstring) must
    # propagate as NaN into the following year's yoy_change, never be padded
    # forward first -- pandas' pct_change() default (fill_method='pad') would
    # otherwise silently compute a change against a forward-filled prior
    # value instead of leaving it unknown.
    yoy = annual.pct_change(fill_method=None)
    # A real zero baseline (idps_ethiopia was a genuine reported 0 in 2016,
    # jumping to 1,078,429 in 2017) makes pct_change() return +/-inf, not
    # NaN -- caught empirically when XGBoost refused to train on it
    # ("Input data contains `inf`"). Percent change FROM zero is undefined,
    # not infinite; treated as NaN like every other "can't compute this"
    # case in this file, not silently dropped or clipped to a large finite
    # number.
    yoy = yoy.replace([np.inf, -np.inf], np.nan)

    today = pd.Timestamp.now().normalize().replace(day=1)
    grid = build_monthly_grid(FEATURE_START_YEAR, today)

    known_year = grid.apply(lambda r: known_year_asof(r["year"], r["month"]), axis=1)
    for col in LEVEL_COLUMNS:
        grid[col] = known_year.map(annual[col])
        grid[f"{col}_yoy_change"] = known_year.map(yoy[col])

    cols = ["year", "month"] + [c for col in LEVEL_COLUMNS for c in (col, f"{col}_yoy_change")]
    out = grid[cols].reset_index(drop=True)

    if write_output:
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        out.to_csv(OUTPUT_PATH, index=False)

    return out


def main():
    out = compute_unhcr_displacement_monthly()
    first, last = out.iloc[0], out.iloc[-1]
    print(f"Wrote {len(out)} rows ({int(first['year'])}-{int(first['month']):02d} to "
          f"{int(last['year'])}-{int(last['month']):02d}) to {OUTPUT_PATH}")
    for col in LEVEL_COLUMNS:
        n_nan = out[col].isna().sum()
        print(f"  {col}: {n_nan}/{len(out)} NaN rows")


if __name__ == "__main__":
    main()
