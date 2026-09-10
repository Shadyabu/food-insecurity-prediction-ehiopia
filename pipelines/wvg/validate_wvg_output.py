"""
validate_wvg_output.py

Validates the derived WVG series against Busker et al.'s own staged
reference (busker_comparison/SST.xlsx, sheet "WVG"), which in turn traces
back to Funk et al. (2023)'s own published series (see
docs/WVG_Pipeline_Documentation.md Sec 1) -- the only available reference,
since no independent third-party agency computes this index the way NOAA
independently cross-checks NINO3.4 against CPC.

*** VALIDATION LIMITATION -- READ BEFORE TRUSTING THIS PIPELINE ***
This is fundamentally weaker evidence than every fetch-based pipeline in
this project (IOD/MEI/NINO3.4/CHIRPS/NDVI/etc). Those compare an
independently-produced value against a second source computed by a
different method or agency. Here, Busker's reference almost certainly
derives from the SAME formula and the SAME underlying ERSSTv5 product this
pipeline re-derives from -- so a close match confirms "this pipeline
correctly re-implements the published formula," NOT "two independent
methods agree." Treat this validation as internal-consistency evidence,
not agreement-between-independent-sources evidence, when writing up the
dissertation's methodology/limitations section.

Expected result (documented investigation in
docs/WVG_Pipeline_Documentation.md Sec 3, several formula variants tested
before landing on this one): r~0.93, R^2~0.87 against Busker's 103-year
overlap (1920-2022), median|diff| ~0.30, NOT the near-exact match achieved
by IOD/NINO3.4 in the sibling teleconnections pipeline. This is treated as
a PASS for this pipeline (correct sign, correct order of magnitude, correct
climate-narrative direction, best of several tested formula variants) --
not as evidence-free, but explicitly weaker than this project's other
validations.
"""

import os

import numpy as np
import openpyxl
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
ANNUAL_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "wvg_annual_mam.csv")
MONTHLY_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "wvg_monthly.csv")
BUSKER_XLSX = os.path.join(REPO_ROOT, "pipelines", "teleconnections", "busker_comparison", "SST.xlsx")


def load_busker_wvg():
    wb = openpyxl.load_workbook(BUSKER_XLSX, data_only=True)
    ws = wb["WVG"]
    rows = []
    for row in ws.iter_rows(min_row=19, values_only=True):  # data starts row 19; rows 1-18 are the header block
        year, wvg_val = row[0], row[1]
        if year is None or wvg_val is None:
            continue
        rows.append({"year": int(year), "busker_wvg": float(wvg_val)})
    return pd.DataFrame(rows)


def validate_against_busker(annual):
    print("=== 1. Validation against Busker et al.'s staged reference (SST.xlsx, sheet WVG) ===")
    print("    (weaker evidence than other pipelines -- see module docstring)\n")

    busker = load_busker_wvg()
    m = annual[["year", "wvg"]].merge(busker, on="year", how="inner")
    diff = m["wvg"] - m["busker_wvg"]
    abs_diff = diff.abs()
    n = len(m)
    corr = m["wvg"].corr(m["busker_wvg"])
    r2 = corr ** 2

    print(f"  n (overlap years) = {n} ({m['year'].min()}-{m['year'].max()})")
    print(f"  correlation (r) = {corr:.4f}, r^2 = {r2:.4f}")
    print(f"  median|diff| = {abs_diff.median():.4f}")
    print(f"  mean|diff|   = {abs_diff.mean():.4f}")
    print(f"  max|diff|    = {abs_diff.max():.4f}")
    print(f"  within 0.5   = {(abs_diff < 0.5).sum()}/{n} ({100 * (abs_diff < 0.5).mean():.1f}%)")
    print(f"  within 1.0   = {(abs_diff < 1.0).sum()}/{n} ({100 * (abs_diff < 1.0).mean():.1f}%)")
    print(f"  sign agreement = {(np.sign(m['wvg']) == np.sign(m['busker_wvg'])).sum()}/{n} "
          f"({100 * (np.sign(m['wvg']) == np.sign(m['busker_wvg'])).mean():.1f}%)")

    return m


def check_undated_matches_annual(monthly, annual):
    print("\n=== 2. wvg_undated internal-consistency check ===")
    # Every January row's wvg_undated should equal that year's annual wvg exactly
    jan = monthly[monthly["month"] == 1][["year", "wvg_undated"]]
    m = jan.merge(annual[["year", "wvg"]], on="year", how="inner")
    mismatch = (m["wvg_undated"] - m["wvg"]).abs()
    n_bad = (mismatch > 1e-9).sum()
    print(f"  wvg_undated(Jan) vs annual wvg: {n_bad} mismatches out of {len(m)} rows "
          f"({'PASS' if n_bad == 0 else 'FAIL'})")


def check_publication_lag(monthly):
    print("\n=== 3. Publication-lag cutoff check ===")
    # Before June of year Y, wvg must still show year Y-1's value (not Y's) -- and
    # from June onward it must show year Y's value. Verified directly against
    # the annual series loaded from disk, not re-derived.
    annual = pd.read_csv(ANNUAL_PATH).set_index("year")["wvg"]
    bad = 0
    checked = 0
    for _, row in monthly.iterrows():
        year, month = int(row["year"]), int(row["month"])
        expected_year = year if month >= 6 else year - 1
        expected = annual.get(expected_year, np.nan)
        checked += 1
        if pd.isna(expected) and pd.isna(row["wvg"]):
            continue
        if pd.isna(expected) or pd.isna(row["wvg"]):
            bad += 1
            continue
        if abs(expected - row["wvg"]) > 1e-9:
            bad += 1
    print(f"  {bad} mismatches out of {checked} rows ({'PASS' if bad == 0 else 'FAIL'})")


def check_lag_alignment(annual):
    print("\n=== 4. Year-lag alignment check ===")
    monthly = pd.read_csv(MONTHLY_PATH)
    annual_series = annual.set_index("year")["wvg"]
    # Recompute known_year_asof the same way and check wvg_lag1yr matches wvg shifted 1 year back
    n_bad = 0
    n_checked = 0
    for _, row in monthly.dropna(subset=["wvg", "wvg_lag1yr"]).iterrows():
        year, month = int(row["year"]), int(row["month"])
        known_year = year if month >= 6 else year - 1
        expected_lag1 = annual_series.get(known_year - 1, np.nan)
        n_checked += 1
        if pd.isna(expected_lag1):
            continue
        if abs(expected_lag1 - row["wvg_lag1yr"]) > 1e-9:
            n_bad += 1
    print(f"  wvg_lag1yr vs recomputed prior-year value: {n_bad} mismatches out of {n_checked} rows "
          f"({'PASS' if n_bad == 0 else 'FAIL'})")


def main():
    annual = pd.read_csv(ANNUAL_PATH)
    monthly = pd.read_csv(MONTHLY_PATH)
    print(f"Loaded {len(annual)} annual rows, {len(monthly)} monthly rows\n")

    validate_against_busker(annual)
    check_undated_matches_annual(monthly, annual)
    check_publication_lag(monthly)
    check_lag_alignment(annual)


if __name__ == "__main__":
    main()
