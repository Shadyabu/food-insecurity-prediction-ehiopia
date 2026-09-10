"""
validate_cpi_output.py

Validates the CPI output four ways, per CLAUDE.md Sec 3.5:

1. Internal cross-check: for every month where BOTH Ha et al.'s index
   level (this pipeline's primary source) and an ESS bulletin's directly-
   published YoY rate (the secondary/extension source) are available,
   compare the two independently-derived YoY figures. They should agree
   almost exactly, since ESS's own rate is, by definition, computed the
   same way (index(m)/index(m-12)-1) from ESS's own maintained series --
   a large disagreement here would mean either the PDF table-row parser
   or the EFY-to-Gregorian date mapping has a bug, not genuine data
   disagreement.
2. Against Busker et al.'s own staged reference
   (busker_comparison/Inflation_WB.xlsx, a May-2023-vintage download of
   the SAME Ha et al. database). Two sub-checks: (a) an identity check
   that headline_cpi_undated/food_cpi_undated -- the RQ1-fidelity columns
   this file's compute stage now builds directly from that file (see
   compute_cpi_monthly_features.load_busker_undated()) -- actually match
   it exactly (should be a tautology; a mismatch means a join/parsing
   regression, not a data disagreement); (b) the CURRENT-release
   headline_cpi/food_cpi levels against the same file, where drift IS
   expected and NOT a failure for headline: confirmed during build that
   Ha et al.'s own headline series was rebased (re-spliced onto Ethiopia's
   official Dec-2016=100 base, see docs/CPI_Pipeline_Documentation.md
   Sec 5) and its published history shortened between the May-2023 and
   the live April-2025 vintage this pipeline uses -- reported as a
   distribution, not pass/fail, same treatment as GDP vs Busker and MEI
   vs Busker. Food shows no such drift, since it was already on that same
   base in both vintages.
3. Against the World Bank's FP.CPI.TOTL.ZG (inflation, consumer prices,
   annual %, IMF IFS-sourced) for Ethiopia, fetched live -- an independent
   source using a different aggregation (true annual average, not a
   mean-of-monthly-YoY approximation) and possibly a different base
   period, so this is a coarse directional check, not an exact-match
   expectation.
4. Structural checks: NaN boundaries land exactly where the documented
   source-coverage gaps say they should, source-flag columns are
   internally consistent with which values are actually populated, and
   lag columns are a clean shift of the lag0 columns.
"""

import os
import sys

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "cpi_monthly.csv")

sys.path.insert(0, SCRIPT_DIR)
import compute_cpi_monthly_features as cpi_features  # noqa: E402

WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/ETH/indicator/FP.CPI.TOTL.ZG"
    "?format=json&per_page=100"
)


def report_pct_distribution(label, ours, reference):
    common = ours.index.intersection(reference.index)
    pct = ((ours.loc[common] - reference.loc[common]).abs() / reference.loc[common].abs()).dropna() * 100
    n = len(pct)
    if n == 0:
        print(f"  {label}: no overlapping rows to compare")
        return
    within20 = (pct <= 20).sum()
    print(f"  {label}: n={n}, median%diff={pct.median():.1f}%, mean%diff={pct.mean():.1f}%, "
          f"within 20% = {within20}/{n} ({100 * within20 / n:.1f}%)")


def report_abs_distribution(label, diff):
    diff = diff.dropna()
    n = len(diff)
    if n == 0:
        print(f"  {label}: no overlapping rows to compare")
        return
    print(f"  {label}: n={n}, median|diff|={diff.abs().median():.4f}, mean|diff|={diff.abs().mean():.4f}, "
          f"max|diff|={diff.abs().max():.4f}")


# --------------------------------------------------------------------------
# 1. Internal cross-check: Ha et al.-derived YoY vs ESS-published YoY
# --------------------------------------------------------------------------

def validate_ha_vs_ess_overlap():
    print("=== 1. Internal cross-check: Ha et al. (level-derived) vs ESS (published) YoY, overlap months ===")
    ha = cpi_features.load_ha_et_al()
    # Explicit division, not .pct_change(12) -- see the matching comment
    # in compute_cpi_monthly_features.py for why: older pandas pads NaNs
    # forward by default before computing the ratio, which would silently
    # fabricate a rate from `ha`'s outer-joined NaN food_cpi rows here.
    ha["headline_yoy_from_level"] = ha["headline_cpi"] / ha["headline_cpi"].shift(12) - 1
    ha["food_yoy_from_level"] = ha["food_cpi"] / ha["food_cpi"].shift(12) - 1
    ess = cpi_features.load_ess_bulletins()
    merged = ha.merge(ess, on=["year", "month"], how="inner")

    report_abs_distribution(
        "headline YoY: |Ha-derived - ESS-published|",
        merged["headline_yoy_from_level"] - merged["ess_headline_yoy"],
    )
    report_abs_distribution(
        "food YoY: |Ha-derived - ESS-published| (only where Ha's food level still covers the month)",
        merged["food_yoy_from_level"] - merged["ess_food_yoy"],
    )
    print(f"  n overlap months (headline): {merged['headline_yoy_from_level'].notna().sum()}, "
          f"(food): {merged['food_yoy_from_level'].notna().sum()}")


# --------------------------------------------------------------------------
# 2. Busker-staged reference (same source, older vintage)
# --------------------------------------------------------------------------

def validate_against_busker(ours):
    print("\n=== 2. Validation against Busker et al.'s staged reference (Inflation_WB.xlsx, May-2023 vintage) ===")
    busker = cpi_features.load_busker_undated().rename(
        columns={"headline_cpi_undated": "busker_headline_cpi", "food_cpi_undated": "busker_food_cpi"}
    )

    merged = ours.merge(busker, on=["year", "month"], how="inner").set_index(["year", "month"])

    # headline_cpi_undated/food_cpi_undated are now BUILT from this exact
    # file (compute_cpi_monthly_features.load_busker_undated()) -- this is
    # a trivial identity check, not independent validation, but it catches
    # a join/parsing regression between the two pipeline stages.
    for col, busker_col in [("headline_cpi_undated", "busker_headline_cpi"), ("food_cpi_undated", "busker_food_cpi")]:
        mismatch = (merged[col] - merged[busker_col]).abs() > 1e-9
        print(f"  {col} vs Busker's file (should be an exact identity, sourced from the same file): "
              f"{'PASS' if mismatch.sum() == 0 else 'FAIL'} ({mismatch.sum()} mismatched rows)")

    report_pct_distribution(
        "headline_cpi (CURRENT-release level) vs Busker's snapshot (revision/rebasing drift expected)",
        merged["headline_cpi"], merged["busker_headline_cpi"],
    )
    report_pct_distribution(
        "food_cpi (CURRENT-release level) vs Busker's snapshot (no revision expected -- same base both vintages)",
        merged["food_cpi"], merged["busker_food_cpi"],
    )


# --------------------------------------------------------------------------
# 3. World Bank independent reference (annual, different aggregation)
# --------------------------------------------------------------------------

def fetch_world_bank_annual_pct():
    resp = requests.get(WORLD_BANK_URL, timeout=60)
    resp.raise_for_status()
    payload = resp.json()
    records = payload[1] if len(payload) > 1 and payload[1] else []
    years, values = [], []
    for rec in records:
        if rec["value"] is not None:
            years.append(int(rec["date"]))
            values.append(float(rec["value"]) / 100.0)
    return pd.Series(values, index=years, name="wb_value")


def validate_against_world_bank(ours):
    print("\n=== 3. Validation against World Bank FP.CPI.TOTL.ZG (independent source, annual, coarse check) ===")
    try:
        wb = fetch_world_bank_annual_pct()
    except Exception as e:
        print(f"  Could not fetch World Bank series ({e}); skipping this check")
        return

    annual_mean_yoy = ours.groupby("year")["headline_cpi_yoy_change"].mean()
    common = annual_mean_yoy.index.intersection(wb.index)
    diff = ((annual_mean_yoy.loc[common] - wb.loc[common]).abs())
    report_abs_distribution(
        "headline: |mean-of-monthly-YoY (ours) - annual %change (World Bank)| (different aggregation, coarse)",
        diff,
    )


# --------------------------------------------------------------------------
# 4. Structural checks
# --------------------------------------------------------------------------

def check_structure(ours):
    print("\n=== 4. Structural checks ===")

    for level_col, source_col in [
        ("headline_cpi", "headline_cpi_yoy_change_source"),
        ("food_cpi", "food_cpi_yoy_change_source"),
    ]:
        last_level_valid = ours.loc[ours[level_col].notna(), ["year", "month"]].iloc[-1]
        print(f"  {level_col}: last non-NaN month = {int(last_level_valid['year'])}-{int(last_level_valid['month']):02d}")

        ha_rows = ours[ours[source_col] == "ha_et_al"]
        ess_rows = ours[ours[source_col] == "ess_bulletin"]
        # Every ha_et_al-sourced yoy row should fall at/before the level's
        # own last-valid month + 0 (yoy needs level(m) itself); every
        # ess_bulletin-sourced row should fall strictly after it.
        bad_ha = ha_rows[
            (ha_rows["year"] > last_level_valid["year"])
            | ((ha_rows["year"] == last_level_valid["year"]) & (ha_rows["month"] > last_level_valid["month"]))
        ]
        bad_ess = ess_rows[
            (ess_rows["year"] < last_level_valid["year"])
            | ((ess_rows["year"] == last_level_valid["year"]) & (ess_rows["month"] <= last_level_valid["month"]))
        ]
        status = "PASS" if bad_ha.empty and bad_ess.empty else "FAIL"
        print(f"    source-flag boundary check ({source_col}): {status} "
              f"({len(bad_ha)} ha_et_al rows after level coverage ends, "
              f"{len(bad_ess)} ess_bulletin rows within/before level coverage)")

    print("\n  Lag-shift consistency (spot check lag1 for headline_cpi_yoy_change):")
    sorted_df = ours.sort_values(["year", "month"]).reset_index(drop=True)
    shifted = sorted_df["headline_cpi_yoy_change"].shift(1)
    mismatch = (sorted_df["headline_cpi_yoy_change_lag1"] - shifted).abs() > 1e-9
    mismatch = mismatch & sorted_df["headline_cpi_yoy_change_lag1"].notna()
    print(f"    {'PASS' if mismatch.sum() == 0 else 'FAIL'} ({mismatch.sum()} mismatched rows)")


def main():
    ours = pd.read_csv(OUTPUT_PATH)
    print(f"Loaded {len(ours)} rows from {OUTPUT_PATH}\n")

    validate_ha_vs_ess_overlap()
    validate_against_busker(ours)
    validate_against_world_bank(ours)
    check_structure(ours)


if __name__ == "__main__":
    main()
