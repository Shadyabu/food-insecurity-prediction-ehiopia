"""
validate_unhcr_displacement_monthly_output.py

Validates data/processed/features/unhcr_displacement_monthly.csv, per
CLAUDE.md Sec 3.5. No independent third-party reference exists for this
derived national broadcast series (UNHCR is already the authoritative
source; there is no second dataset to diff against, same posture as
exchange_rate/acled_conflict) -- structural checks only:

1. Complete monthly grid, no duplicate (year, month), FEATURE_START_YEAR
   through the current month.
2. Publication-lag rule actually holds: every row's value for each level
   column matches the annual series at known_year_asof(year, month) exactly
   (not just plausible -- exact, since this is a deterministic lookup).
3. NaN boundaries land exactly where the documented structural gaps say
   they should (idps_ethiopia NaN only while its June-of-Y+1 visibility
   window still points at a genuinely-missing raw year: 2007 [pre-series],
   2008-2011, or 2015).
4. yoy_change columns are NaN exactly where the level column two years
   back was NaN (fill_method=None propagation, not silently padded).
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from compute_unhcr_displacement_monthly import (  # noqa: E402
    FEATURE_START_YEAR,
    LEVEL_COLUMNS,
    OUTPUT_PATH,
    build_annual_series,
    known_year_asof,
)
from prepare_persons_of_concern import prepare_persons_of_concern  # noqa: E402

# Raw years with no country_of_origin == "Ethiopia" row at all (checked
# 2026-08-24) -- idps_ethiopia is structurally missing for these.
MISSING_IDPS_YEARS = {2008, 2009, 2010, 2011, 2015}


def main() -> None:
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(
            f"{OUTPUT_PATH} not found -- run "
            "`python -m pipelines.unhcr_displacement.compute_unhcr_displacement_monthly` first."
        )

    df = pd.read_csv(OUTPUT_PATH)
    print(f"Loaded {len(df)} rows from {OUTPUT_PATH}")

    today = pd.Timestamp.now().normalize().replace(day=1)
    expected_grid = pd.date_range(f"{FEATURE_START_YEAR}-01-01", today, freq="MS")
    assert len(df) == len(expected_grid), (
        f"Row count {len(df)} != expected complete monthly grid {len(expected_grid)}"
    )
    assert not df.duplicated(subset=["year", "month"]).any(), "Duplicate (year, month) rows found"
    print(f"Complete gapless monthly grid, {FEATURE_START_YEAR}-01 to {today.strftime('%Y-%m')}: PASS")

    clean = prepare_persons_of_concern(write_output=False)
    annual = build_annual_series(clean)
    min_raw_year = int(clean["year"].min())

    all_pass = True
    for _, row in df.iterrows():
        ky = known_year_asof(int(row["year"]), int(row["month"]))
        for col in LEVEL_COLUMNS:
            expected = annual[col].get(ky) if ky in annual.index else float("nan")
            actual = row[col]
            both_nan = pd.isna(expected) and pd.isna(actual)
            if not both_nan and expected != actual:
                all_pass = False
                print(f"  MISMATCH {col} at {int(row['year'])}-{int(row['month']):02d} "
                      f"(known_year={ky}): expected {expected}, got {actual}")
    print(f"Publication-lag lookup matches annual series exactly for every row: "
          f"{'PASS' if all_pass else 'FAIL'}")

    idps_nan_rows = df[df["idps_ethiopia"].isna()]
    bad = []
    for _, row in idps_nan_rows.iterrows():
        ky = known_year_asof(int(row["year"]), int(row["month"]))
        if ky >= min_raw_year and ky not in MISSING_IDPS_YEARS:
            bad.append((int(row["year"]), int(row["month"]), ky))
    print(f"idps_ethiopia NaN rows land only on documented gap years "
          f"(pre-{min_raw_year} or {sorted(MISSING_IDPS_YEARS)}): "
          f"{'PASS' if not bad else f'FAIL -- unexpected NaN at {bad}'}")

    # fill_method=None: a gap year's NaN must propagate into the following
    # year's yoy_change, never be padded forward first (mirrors the
    # compute script's own reasoning) -- 0/0 legitimately yields NaN too
    # (e.g. idps_ethiopia 2012->2013, both real reported zeros), which is
    # why this re-derives the expected value directly from annual.pct_change()
    # rather than a hand-rolled "both years present => not NaN" rule.
    # inf/-inf (a real zero baseline, e.g. idps_ethiopia 2016->2017) is
    # replaced with NaN here too, mirroring the compute script exactly --
    # XGBoost cannot train on raw inf, caught empirically 2026-08-24.
    yoy = annual.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)
    yoy_all_pass = True
    for col in LEVEL_COLUMNS:
        yoy_col = f"{col}_yoy_change"
        for _, row in df.iterrows():
            ky = known_year_asof(int(row["year"]), int(row["month"]))
            expected = yoy[col].get(ky) if ky in yoy.index else float("nan")
            actual = row[yoy_col]
            both_nan = pd.isna(expected) and pd.isna(actual)
            # rtol tolerance: values round-trip through a CSV float write/read,
            # which loses a few ULPs of precision -- not a real mismatch.
            close_enough = (
                pd.notna(expected) and pd.notna(actual)
                and abs(expected - actual) <= 1e-9 * max(abs(expected), 1e-12)
            )
            if not both_nan and not close_enough:
                yoy_all_pass = False
                print(f"  MISMATCH {yoy_col} at {int(row['year'])}-{int(row['month']):02d} "
                      f"(known_year={ky}): expected {expected}, got {actual}")
    print(f"yoy_change matches annual.pct_change(fill_method=None) exactly: "
          f"{'PASS' if yoy_all_pass else 'FAIL'}")

    print("\nAll validation checks passed." if all_pass and not bad and yoy_all_pass
          else "\nSOME CHECKS FAILED -- see above.")


if __name__ == "__main__":
    main()
