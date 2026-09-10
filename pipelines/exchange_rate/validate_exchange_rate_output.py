"""
validate_exchange_rate_output.py

Validates the exchange-rate output per CLAUDE.md Sec 3.5. There is no
Busker et al. reference (they do not use this feature at all -- see
compute_exchange_rate_monthly_features.py's docstring), so this validates
two ways instead, both against genuinely independent sources:

1. WFP-IMPLIED OFFICIAL RATE: WFP's own retail price panel
   (data/raw/wfp_prices/wfp_food_prices_eth_live.csv) carries a commodity
   row "Exchange rate (unofficial)" with both `price` (the unofficial/
   parallel rate) and `usdprice` (that same price expressed in USD). Their
   ratio, price / usdprice, is WFP's own implicit OFFICIAL rate -- a
   completely independent construction (different provider, different
   method: market-survey-derived vs. investing.com's own quote) from this
   pipeline's investing.com series. This is also the check that
   established investing.com's series tracks the OFFICIAL rate, not the
   unofficial one it's confusingly commodity-labelled as in WFP's own file
   -- see the large early-2024 gap (investing.com ~56.6, WFP's raw
   unofficial price ~113) versus the near-exact match against the implied
   ratio (~56.2) reported below.

2. WORLD BANK ANNUAL OFFICIAL RATE (PA.NUS.FCRF, period average, live
   API fetch): independent source and independent methodology (annual
   period-average vs. this pipeline's monthly point-in-time). Reported as
   a distribution/correlation, not a pass/fail threshold, same posture as
   agss_yields' World Bank cross-check.
"""

import os

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "exchange_rate_monthly.csv")
WFP_RAW_PATH = os.path.join(REPO_ROOT, "data", "raw", "wfp_prices", "wfp_food_prices_eth_live.csv")

WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/ETH/indicator/PA.NUS.FCRF"
    "?format=json&per_page=100"
)


def load_output():
    df = pd.read_csv(OUTPUT_PATH)
    return df[["year", "month", "exchange_rate"]].copy()


def load_wfp_implied_official():
    df = pd.read_csv(WFP_RAW_PATH, low_memory=False, parse_dates=["date"])
    ex = df[df["commodity"] == "Exchange rate (unofficial)"].copy()
    if ex.empty:
        raise ValueError(f"No 'Exchange rate (unofficial)' rows found in {WFP_RAW_PATH}")
    ex["year"] = ex["date"].dt.year
    ex["month"] = ex["date"].dt.month
    ex["wfp_implied_official"] = ex["price"] / ex["usdprice"]
    return ex[["year", "month", "wfp_implied_official"]].sort_values(["year", "month"]).reset_index(drop=True)


def fetch_world_bank_official_rate():
    resp = requests.get(WORLD_BANK_URL, timeout=30)
    resp.raise_for_status()
    _, records = resp.json()
    rows = [
        {"year": int(r["date"]), "wb_official_rate": r["value"]}
        for r in records if r["value"] is not None
    ]
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def check_wfp_implied_official(ours):
    print("\n[1] Cross-check vs. WFP-implied official rate (price / usdprice)")
    print("    (independent construction -- market-survey-derived, not investing.com-sourced)")

    wfp = load_wfp_implied_official()
    merged = wfp.merge(ours, on=["year", "month"], how="inner")
    print(f"    Overlapping months: {len(merged)} of {len(wfp)} WFP-reported months")
    if merged.empty:
        print("    No overlap -- skipping")
        return

    diff_pct = (merged["exchange_rate"] - merged["wfp_implied_official"]) / merged["wfp_implied_official"] * 100
    abs_diff_pct = diff_pct.abs()
    corr = merged["exchange_rate"].corr(merged["wfp_implied_official"])

    print(f"    Correlation: {corr:.4f}")
    print(f"    Median diff: {diff_pct.median():+.2f}%  |  Mean abs diff: {abs_diff_pct.mean():.2f}%")
    for thresh in [1, 5, 10, 20]:
        pct_within = (abs_diff_pct <= thresh).mean() * 100
        print(f"    Within {thresh}%: {pct_within:.1f}% of {len(merged)} months")

    worst = merged.assign(abs_diff_pct=abs_diff_pct).nlargest(5, "abs_diff_pct")
    print("    Largest 5 discrepancies:")
    print(worst[["year", "month", "exchange_rate", "wfp_implied_official", "abs_diff_pct"]].to_string(index=False))


def check_world_bank_annual(ours):
    print("\n[2] Cross-check vs. World Bank PA.NUS.FCRF (annual official rate, period average)")
    print("    (independent source & methodology -- annual period-average vs. our monthly series;")
    print("     reported as a distribution, not a pass/fail threshold)")

    wb = fetch_world_bank_official_rate()
    if wb.empty:
        print("    World Bank fetch returned no data -- skipping")
        return

    annual_mean = ours.groupby("year")["exchange_rate"].mean().rename("our_annual_mean").reset_index()
    merged = wb.merge(annual_mean, on="year", how="inner")
    if merged.empty:
        print("    No overlapping years -- skipping")
        return

    diff_pct = (merged["our_annual_mean"] - merged["wb_official_rate"]) / merged["wb_official_rate"] * 100
    corr = merged["our_annual_mean"].corr(merged["wb_official_rate"])

    print(f"    Overlapping years: {len(merged)} ({int(merged['year'].min())}-{int(merged['year'].max())})")
    print(f"    Correlation: {corr:.4f}")
    print(f"    Median diff: {diff_pct.median():+.2f}%  |  Mean abs diff: {diff_pct.abs().mean():.2f}%")
    for thresh in [5, 10, 20]:
        pct_within = (diff_pct.abs() <= thresh).mean() * 100
        print(f"    Within {thresh}%: {pct_within:.1f}% of {len(merged)} years")


def main():
    ours = load_output()
    check_wfp_implied_official(ours)
    check_world_bank_annual(ours)


if __name__ == "__main__":
    main()
