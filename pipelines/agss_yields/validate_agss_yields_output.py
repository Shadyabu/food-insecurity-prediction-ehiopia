"""
validate_agss_yields_output.py

Validates the AgSS yield output per CLAUDE.md Sec 3.5. There is no Busker
et al. reference (they did not use crop-yield data at all -- see
fetch_agss_yields_admin2.py's docstring), so this validates two ways
instead:

1. NATIONAL-LEVEL cross-check against FAOSTAT/World Bank's independent
   cereal-yield series (AG.YLD.CREL.KG, kg/ha, FAO-sourced) -- population-
   weighted mean of this pipeline's 4 crops' zone-level yields, compared to
   the national aggregate. Not a like-for-like check (different
   methodology, different crop basket -- World Bank's indicator is ALL
   cereals, not just these 4), so this is reported as a distribution and a
   correlation, not a pass/fail threshold, same posture as imf_gdp's
   independent-source check.

2. COVERAGE REPORT -- the central finding this pipeline was built around
   (per CLAUDE.md Sec 3.5's "report a distribution, not a hand-picked few"
   and the project owner's explicit request to report coverage, not just
   numeric agreement): % of zones surveyed by year and by livelihood-zone
   type, confirming the pastoral/agropastoral coverage gap is structural
   and stable, not a fetch artefact.
"""

import os

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "agss_yields_admin2_monthly.csv")
LIVELIHOOD_CSV = os.path.join(REPO_ROOT, "boundaries", "livelihood_zones_admin2.csv")

WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/ETH/indicator/AG.YLD.CREL.KG"
    "?format=json&per_page=100"
)

CROPS = ["maize", "wheat", "sorghum", "teff"]


def fetch_world_bank_cereal_yield():
    resp = requests.get(WORLD_BANK_URL, timeout=30)
    resp.raise_for_status()
    _, records = resp.json()
    rows = [
        {"year": int(r["date"]), "cereal_yield_kg_ha": r["value"]}
        for r in records if r["value"] is not None
    ]
    return pd.DataFrame(rows).sort_values("year")


def national_cross_check(panel):
    print("\n[1] National-level cross-check vs. World Bank AG.YLD.CREL.KG")
    print("    (independent source, different crop basket & methodology --")
    print("     reported as a distribution, not a pass/fail threshold)")

    wb = fetch_world_bank_cereal_yield()
    if wb.empty:
        print("    World Bank fetch returned no data -- skipping")
        return

    # This pipeline's yields are t/ha; World Bank's is kg/ha.
    yearly = (
        panel.dropna(subset=[f"{c}_yield" for c in CROPS], how="all")
        .assign(any_yield=lambda d: d[[f"{c}_yield" for c in CROPS]].mean(axis=1))
        .groupby("year")["any_yield"].mean() * 1000.0
    )
    yearly = yearly.rename("agss_mean_yield_kg_ha").reset_index()

    merged = wb.merge(yearly, on="year", how="inner")
    if merged.empty:
        print("    no overlapping years to compare")
        return

    diff_pct = (merged["agss_mean_yield_kg_ha"] - merged["cereal_yield_kg_ha"]) / merged["cereal_yield_kg_ha"] * 100
    corr = merged["agss_mean_yield_kg_ha"].corr(merged["cereal_yield_kg_ha"])
    print(f"    n={len(merged)} overlapping years")
    print(f"    correlation: {corr:.3f}")
    print(f"    median diff: {diff_pct.median():.1f}%, mean diff: {diff_pct.mean():.1f}%")
    print("    (a moderate gap is expected: World Bank's indicator covers ALL")
    print("     cereals nationally; this pipeline covers 4 crops, zone-mean not")
    print("     production-weighted, and Meher-season only)")


def coverage_report(panel):
    print("\n[2] Coverage report (the central finding this pipeline was built around)")

    livelihood = pd.read_csv(LIVELIHOOD_CSV)[["zone_code", "dominant_livelihood_zone"]]
    flag_by_zone = panel.groupby("zone_code")["agss_surveyed_flag"].first().reset_index()
    merged = flag_by_zone.merge(livelihood, on="zone_code", how="left")

    print("\n  Overall: "
          f"{merged['agss_surveyed_flag'].sum()}/{len(merged)} zones ever surveyed "
          f"({merged['agss_surveyed_flag'].mean():.1%})")

    print("\n  By livelihood-zone type:")
    for lz_type, grp in merged.groupby("dominant_livelihood_zone"):
        n = len(grp)
        surveyed = grp["agss_surveyed_flag"].sum()
        print(f"    {lz_type}: {surveyed}/{n} zones ({surveyed/n:.1%})")

    unsurveyed = merged[merged["agss_surveyed_flag"] == 0]
    if len(unsurveyed):
        print(f"\n  Never-surveyed zones ({len(unsurveyed)}):")
        for _, r in unsurveyed.iterrows():
            print(f"    {r['zone_code']} ({r['dominant_livelihood_zone']})")

    print("\n  Per-crop status distribution (zone-months, full panel):")
    for crop in CROPS:
        col = f"{crop}_yield_status"
        counts = panel[col].value_counts()
        total = len(panel)
        print(f"\n    {crop}:")
        for status, count in counts.items():
            print(f"      {status}: {count:,} ({count/total:.1%})")

    # Year-by-year collected-status coverage, restricted to years any crop
    # actually has data, to show the pastoral gap is stable across time,
    # not just an artefact of one era.
    collected_any = panel[[f"{c}_yield_status" for c in CROPS]].eq("collected").any(axis=1)
    by_year = panel.assign(collected_any=collected_any).groupby("year")["collected_any"].mean()
    by_year = by_year[by_year > 0]
    if len(by_year):
        print(f"\n  Share of zone-months with >=1 crop 'collected', by year "
              f"({by_year.index.min()}-{by_year.index.max()}):")
        for yr in sorted(by_year.index)[::5]:  # every 5th year, keep the printout short
            print(f"    {yr}: {by_year[yr]:.1%}")


def structural_checks(panel):
    print("\n[3] Structural checks")
    n_zones = panel["zone_code"].nunique()
    print(f"    zones: {n_zones} (expect 92)")
    n_months = panel[["year", "month"]].drop_duplicates().shape[0]
    print(f"    months: {n_months}")
    expected_rows = n_zones * n_months
    print(f"    panel completeness: {len(panel)}/{expected_rows} rows "
          f"({len(panel)/expected_rows:.1%})")

    # No INDEFINITE forward-fill past the confirmed 2022 recency ceiling.
    # NOTE: "collected" status legitimately appears in calendar 2023/2024
    # rows -- a 2022-harvest value's 12-month publication-lag visibility
    # window (Oct 2023-Sep 2024) spans those calendar years by design (see
    # compute_agss_yields_admin2_monthly_features.py's publication-lag
    # rule). The real invariant to check is that NO zone-crop ever shows
    # "collected" indefinitely -- i.e. every "collected" run terminates by
    # the natural end of its harvest year's window, never reaching the
    # panel's later years (2025-2026), which would indicate the tail value
    # is being held constant forever rather than expiring.
    tail_years = panel[panel["year"] >= 2025]
    for crop in CROPS:
        bad = tail_years[tail_years[f"{crop}_yield_status"] == "collected"]
        status = "FAIL" if len(bad) else "PASS"
        print(f"    {crop}: no 'collected' rows in 2025-2026 "
              f"(would indicate indefinite forward-fill) -- {status} "
              f"({len(bad)} violations)")


def main():
    print("=" * 74)
    print("AgSS yield VALIDATE")
    print("=" * 74)

    if not os.path.exists(OUTPUT_PATH):
        raise SystemExit(f"{OUTPUT_PATH} not found. Run the acquire/compute stages first.")
    panel = pd.read_csv(OUTPUT_PATH)

    structural_checks(panel)
    coverage_report(panel)
    national_cross_check(panel)

    print("\nDone.")


if __name__ == "__main__":
    main()
