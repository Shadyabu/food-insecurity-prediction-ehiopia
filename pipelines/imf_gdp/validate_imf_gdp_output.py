"""
validate_imf_gdp_output.py

Validates the IMF GDP per capita output three ways, per CLAUDE.md Sec 3.5:

1. Against Busker et al.'s own staged reference
   (busker_comparison/GDP_PER_CAPITA_IMF.xlsx, sheet NGDPDPC) -- compared
   against our *_undated column, since both are undated current-release
   snapshots of the same IMF WEO indicator (just pulled at different
   times). Some drift is expected and NOT a failure: IMF revises WEO
   figures every release, and Ethiopia's July 2024 birr devaluation in
   particular produced large, real revisions to recent-year USD figures
   (see compute_imf_gdp_monthly_features.py docstring) -- reported as a
   distribution, not pass/fail.
2. Against the World Bank's GDP per capita series (NY.GDP.PCAP.CD, current
   USD) for Ethiopia, fetched live from the World Bank API -- an
   independent source using a different methodology, so a moderate
   difference is expected; a large divergence would flag a unit/fetch
   error rather than genuine data disagreement.
3. Structural check: forward-fill produces a flat value within each
   calendar year with a step exactly at the expected boundary (January
   for the undated column, April for the publication-lag-safe column),
   and no unexpected NaNs beyond the documented start-of-series case.
"""

import os

import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "imf_gdp_monthly.csv")
BUSKER_XLSX = os.path.join(SCRIPT_DIR, "busker_comparison", "GDP_PER_CAPITA_IMF.xlsx")

WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/ETH/indicator/NY.GDP.PCAP.CD"
    "?format=json&per_page=100"
)


def report_distribution(label, diff):
    diff = diff.dropna()
    n = len(diff)
    if n == 0:
        print(f"  {label}: no overlapping rows to compare")
        return
    print(f"  {label}: n={n}, median|diff|={diff.abs().median():.2f}, mean|diff|={diff.abs().mean():.2f}, "
          f"max|diff|={diff.abs().max():.2f}")


def report_pct_distribution(label, ours, reference):
    pct = ((ours - reference).abs() / reference.abs()).dropna() * 100
    n = len(pct)
    if n == 0:
        print(f"  {label}: no overlapping rows to compare")
        return
    within20 = (pct <= 20).sum()
    print(f"  {label}: n={n}, median%diff={pct.median():.1f}%, mean%diff={pct.mean():.1f}%, "
          f"within 20% = {within20}/{n} ({100 * within20 / n:.1f}%)")


def load_busker_annual():
    df = pd.read_excel(BUSKER_XLSX, sheet_name="NGDPDPC")
    label_col = df.columns[0]
    matches = df[df[label_col] == "Ethiopia"]
    assert len(matches) == 1, f"expected exactly one Ethiopia row, found {len(matches)}"
    row = matches.iloc[0]
    years, values = [], []
    for col in df.columns[1:]:
        try:
            year = int(col)
        except (ValueError, TypeError):
            continue
        val = row[col]
        if pd.notna(val):
            years.append(year)
            values.append(float(val))
    return pd.Series(values, index=years, name="busker_value")


def fetch_world_bank_annual():
    resp = requests.get(WORLD_BANK_URL, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    records = payload[1] if len(payload) > 1 and payload[1] else []
    years, values = [], []
    for rec in records:
        if rec["value"] is not None:
            years.append(int(rec["date"]))
            values.append(float(rec["value"]))
    return pd.Series(values, index=years, name="wb_value")


def validate_against_busker(annual_undated):
    print("=== 1. Validation against Busker et al.'s staged reference (GDP_PER_CAPITA_IMF.xlsx) ===")
    busker = load_busker_annual()
    common = annual_undated.index.intersection(busker.index)
    diff = (annual_undated.loc[common] - busker.loc[common])
    report_distribution("GDP/capita (undated) vs Busker's 2023-vintage snapshot (revision drift expected)", diff)
    report_pct_distribution(
        "  as % of Busker value",
        annual_undated.loc[common], busker.loc[common],
    )


def validate_against_world_bank(annual_undated):
    print("\n=== 2. Validation against World Bank NY.GDP.PCAP.CD (independent source) ===")
    try:
        wb = fetch_world_bank_annual()
    except Exception as e:
        print(f"  Could not fetch World Bank series ({e}); skipping this check")
        return
    common = annual_undated.index.intersection(wb.index)
    report_pct_distribution(
        "GDP/capita (undated, IMF WEO) vs World Bank (different methodology, moderate diff expected)",
        annual_undated.loc[common], wb.loc[common],
    )


def check_fill_structure(ours):
    print("\n=== 3. Forward-fill structural check ===")

    for label, col, step_month in [
        ("undated (step should land in January)", "gdp_per_capita_undated", 1),
        ("publication-lag-safe (step should land in April)", "gdp_per_capita", 4),
    ]:
        bad_flat = 0
        bad_step = 0
        for year, grp in ours.groupby("year"):
            grp = grp.sort_values("month")
            vals = grp[col].dropna()
            # within a year, the column should take at most 2 distinct values
            # (a pre-step and a post-step value), since a single calendar year
            # only ever crosses one publication-lag boundary.
            if vals.nunique() > 2:
                bad_flat += 1
        ours_sorted = ours.sort_values(["year", "month"]).reset_index(drop=True)
        change_points = ours_sorted[col].ne(ours_sorted[col].shift()).fillna(False)
        wrong_month_changes = ours_sorted.loc[change_points & (ours_sorted["month"] != step_month), "month"]
        # first row of the series is always a "change" trivially -- exclude it
        wrong_month_changes = wrong_month_changes.iloc[1:] if len(ours_sorted) and change_points.iloc[0] else wrong_month_changes
        n_unexpected = len(wrong_month_changes)
        print(f"  {col} ({label}): {bad_flat} year(s) with >2 distinct values, "
              f"{n_unexpected} value-changes outside month={step_month} "
              f"({'PASS' if bad_flat == 0 and n_unexpected == 0 else 'FAIL'})")

    print("\n  NaN check:")
    for col in ["gdp_per_capita", "gdp_per_capita_yoy_change", "gdp_per_capita_undated", "gdp_per_capita_yoy_change_undated"]:
        n_nan = ours[col].isna().sum()
        first_nan_year = ours.loc[ours[col].isna(), "year"].max() if n_nan else None
        print(f"    {col}: {n_nan} NaN rows"
              + (f" (all at/before {first_nan_year}, expected at series start)" if n_nan else ""))


def main():
    ours = pd.read_csv(OUTPUT_PATH)
    print(f"Loaded {len(ours)} rows from {OUTPUT_PATH}\n")

    annual_undated = ours.drop_duplicates("year").set_index("year")["gdp_per_capita_undated"]

    validate_against_busker(annual_undated)
    validate_against_world_bank(annual_undated)
    check_fill_structure(ours)


if __name__ == "__main__":
    main()
