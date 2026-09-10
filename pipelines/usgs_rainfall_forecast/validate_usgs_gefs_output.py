"""
validate_usgs_gefs_output.py

VALIDATE stage for the USGS Rainfall Forecast (CHIRPS-GEFS 15-day) feature.

Not a Busker et al. feature (see fetch_chirps_gefs_15day.py's docstring),
so there is no published reference table to diff against the way CHIRPS/
NDVI/locust/etc. validate. Per CLAUDE.md §3.5's "every available
independent reference" and the task's own step 2/7 guidance, the natural
reference here is CHIRPS's own SUBSEQUENTLY-OBSERVED rainfall -- once the
15-day forecast window has passed, did the forecast agree, directionally,
with what CHIRPS's satellite/gauge product later recorded for the same
zone and days? This is a stronger check than an external third-party
comparison since both products come from the same institution's
methodology (UCSB CHC), but it is a forecast-skill check, not an exact-
match validation -- a forecast and an observation are never expected to
match exactly, so this reports correlation / directional agreement /
error distribution, not a % exact-match figure the way e.g. IOD/NINO3.4
report against Busker.

Reference series: data/interim/chirps/chirps_admin2_daily_combined.csv
(the CHIRPS pipeline's own daily zone-level output, already built and
validated -- see docs/CHIRPS_Pipeline_Documentation.md). For each
usgs_gefs row (zone, year, month, issue_date), the observed comparator is
the SUM of that same zone's CHIRPS daily rainfall over the 15 calendar
days following the issue date (i.e. the exact window the forecast claims
to cover) -- not the full target month, since the forecast only ever
claims to cover ~15 of that month's ~30 days (see
usgs_gefs_coverage_days in the pipeline's own output).

Cross-check against IRI/CPC seasonal (if available): IRI/CPC's tercile
probability forecast and this 15-day mm forecast are different variable
types (probabilistic tercile category vs. continuous total) and different
lead-time bands (1-3 month vs. <=15 day), so no numeric agreement is
expected -- reported here only as a soft directional check (does this
month's below/near/above-normal tercile lean match the sign of the
15-day forecast's own anomaly?), not a primary validation criterion.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

USGS_GEFS_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "usgs_rainfall_forecast_admin2_monthly.csv")
CHIRPS_DAILY_PATH = os.path.join(REPO_ROOT, "data", "interim", "chirps", "chirps_admin2_daily_combined.csv")
IRI_CPC_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "iri_cpc_seasonal_admin2_monthly.csv")

FORECAST_WINDOW_DAYS = 15


def main():
    if not os.path.exists(USGS_GEFS_PATH):
        raise FileNotFoundError(f"{USGS_GEFS_PATH} does not exist -- run the fetch/compute scripts first.")
    if not os.path.exists(CHIRPS_DAILY_PATH):
        raise FileNotFoundError(f"{CHIRPS_DAILY_PATH} does not exist -- run pipelines/chirps/fetch_chirps_admin2.py first.")

    gefs_df = pd.read_csv(USGS_GEFS_PATH)
    gefs_resolved = gefs_df[gefs_df["usgs_gefs_source_lead1"] == "v3"].dropna(subset=["usgs_gefs_issue_date_lead1"]).copy()
    gefs_resolved["usgs_gefs_issue_date_lead1"] = pd.to_datetime(gefs_resolved["usgs_gefs_issue_date_lead1"])

    print("=" * 60)
    print("OVERALL SUMMARY")
    print("=" * 60)
    print(f"Shape: {gefs_df.shape}")
    print(f"Zones: {gefs_df['zone_code'].nunique()} (expected 92)")
    print(f"Rows with a resolved v3 forecast: {len(gefs_resolved)} / {len(gefs_df)}")
    print(f"Source-value breakdown:\n{gefs_df['usgs_gefs_source_lead1'].value_counts()}")
    print()

    assert gefs_df["zone_code"].nunique() == 92, f"BUG: expected 92 zones, got {gefs_df['zone_code'].nunique()}"
    assert gefs_df["usgs_gefs_coverage_days"].eq(FORECAST_WINDOW_DAYS).all(), "BUG: coverage_days should be constant 15."
    print("PASS: 92/92 zones present; usgs_gefs_coverage_days constant at 15.")

    valid_sources = {"v3", "no_archive", "no_data"}
    bad_sources = set(gefs_df["usgs_gefs_source_lead1"].unique()) - valid_sources
    assert not bad_sources, f"BUG: unexpected usgs_gefs_source_lead1 values: {bad_sources}"
    no_archive_years = gefs_df.loc[gefs_df["usgs_gefs_source_lead1"] == "no_archive", "year"].unique()
    print(f"PASS: usgs_gefs_source_lead1 only takes documented values. no_archive years: {sorted(no_archive_years)}")
    print()

    # ------------------------------------------------------------------
    # Build observed-rainfall comparator from CHIRPS daily data
    # ------------------------------------------------------------------
    print("Loading CHIRPS daily reference ...")
    chirps_daily = pd.read_csv(CHIRPS_DAILY_PATH, parse_dates=["date"])
    chirps_daily = chirps_daily.rename(columns={"pcode": "zone_code"})
    chirps_daily = chirps_daily.set_index(["zone_code", "date"])["precipitation_mm"]

    def observed_15day_sum(row):
        zone = row["zone_code"]
        start = row["usgs_gefs_issue_date_lead1"] + pd.Timedelta(days=1)
        dates = pd.date_range(start, periods=FORECAST_WINDOW_DAYS, freq="D")
        try:
            vals = chirps_daily.loc[(zone, dates)]
        except KeyError:
            return np.nan
        vals = vals.reindex(pd.MultiIndex.from_product([[zone], dates]))
        if vals.isna().any():
            return np.nan
        return float(vals.sum())

    print(f"Computing observed 15-day-ahead CHIRPS rainfall for {len(gefs_resolved)} zone-months "
          "(this walks a daily lookup per row, may take a minute) ...")
    gefs_resolved["observed_mm"] = gefs_resolved.apply(observed_15day_sum, axis=1)

    comparable = gefs_resolved.dropna(subset=["observed_mm", "usgs_gefs_precip_mm_lead1"])
    print(f"\n{len(comparable)} / {len(gefs_resolved)} rows have a complete observed comparator "
          "(some recent issue dates fall within the last 15 days and have no observation yet).")

    # ------------------------------------------------------------------
    # Forecast-skill comparison: forecast mm vs. observed mm
    # ------------------------------------------------------------------
    # NOTE: percent error is only meaningful away from a near-zero
    # denominator -- rainfall is zero-bounded and Ethiopia has a real dry
    # season, so a large share of zone-months have very low observed
    # 15-day rainfall. A few mm of absolute miss on a ~1mm observed total
    # is a >>100% error despite being a tiny, unremarkable absolute miss.
    # Reporting a blanket mean %-error over ALL rows (including these)
    # produces a wildly misleading headline number dominated by a handful
    # of near-zero-denominator rows -- confirmed empirically 2026-08-09
    # (mean %-error over all rows: ~498%; over the >=10mm-observed subset
    # alone: ~51%). Both the all-rows and >=10mm-only views are reported
    # below so this is visible rather than hidden behind one number.
    print()
    print("=" * 60)
    print("FORECAST vs. SUBSEQUENTLY-OBSERVED CHIRPS RAINFALL (mm, 15-day sum)")
    print("=" * 60)
    corr = comparable["usgs_gefs_precip_mm_lead1"].corr(comparable["observed_mm"])
    abs_diff = (comparable["usgs_gefs_precip_mm_lead1"] - comparable["observed_mm"]).abs()
    denom = comparable["observed_mm"].replace(0, np.nan)
    pct_diff = (abs_diff / denom).abs() * 100

    print(f"Pearson correlation, ALL resolved rows (forecast mm vs. observed mm): {corr:.3f}")
    print(f"Mean absolute error: {abs_diff.mean():.1f} mm  |  Median absolute error: {abs_diff.median():.1f} mm")
    n_low = (comparable["observed_mm"] <= 10).sum()
    print(f"\n{n_low} / {len(comparable)} zone-months ({n_low / len(comparable) * 100:.1f}%) have "
          "<=10mm observed 15-day rainfall -- percent-error is not a meaningful metric on these "
          "(near-zero denominator), so it is reported both ways below.")

    print(f"\n  Mean/median absolute %% error, ALL rows (misleading -- see note above): "
          f"{pct_diff.mean():.1f}% / {pct_diff.median():.1f}%")

    meaningful = comparable[comparable["observed_mm"] >= 10]
    m_abs_diff = (meaningful["usgs_gefs_precip_mm_lead1"] - meaningful["observed_mm"]).abs()
    m_pct_diff = (m_abs_diff / meaningful["observed_mm"]) * 100
    m_ratio = meaningful["usgs_gefs_precip_mm_lead1"] / meaningful["observed_mm"]
    m_corr = meaningful["usgs_gefs_precip_mm_lead1"].corr(meaningful["observed_mm"])
    print(f"\n  Restricted to observed >= 10mm (n={len(meaningful)}, {len(meaningful) / len(comparable) * 100:.1f}% of rows):")
    print(f"    Pearson correlation: {m_corr:.3f} (vs. {corr:.3f} over all rows -- the all-rows figure "
          f"is partly inflated by trivially-agreeing near-zero dry-season months)")
    print(f"    Mean/median absolute %% error: {m_pct_diff.mean():.1f}% / {m_pct_diff.median():.1f}%")
    print(f"    Forecast/observed ratio -- median {m_ratio.median():.2f}, mean {m_ratio.mean():.2f} "
          "(near 1.0 = no major systematic over/under-forecast bias)")
    for pct in [10, 20, 30, 50]:
        within = (m_pct_diff <= pct).mean() * 100
        print(f"    Zone-months within {pct}% of observed: {within:.1f}%")

    # anomaly sign vs. each zone's OWN CALENDAR-MONTH climatology (not the
    # flat all-time mean -- Ethiopia's rainfall is strongly seasonal, so
    # comparing against an all-time mean would trivially "succeed" just by
    # knowing wet-season-vs-dry-season, not by any real forecast skill)
    month_climatology = comparable.groupby(["zone_code", "month"])["observed_mm"].transform("mean")
    same_sign = np.sign(comparable["usgs_gefs_precip_anom_lead1"]) == np.sign(comparable["observed_mm"] - month_climatology)
    print(f"\nAnomaly-sign directional agreement (forecast anomaly sign vs. observed deviation from "
          f"that zone's OWN calendar-month climatology): {same_sign.mean() * 100:.1f}% "
          f"({same_sign.sum()}/{len(same_sign)}) -- vs. 50% no-skill baseline")
    same_sign_meaningful = same_sign.loc[meaningful.index]
    print(f"  Same check, restricted to observed >= 10mm: {same_sign_meaningful.mean() * 100:.1f}%")
    print(
        "\nInterpretation: this is a forecast-SKILL check, not an exact-match validation "
        "(a 15-day rainfall forecast and the eventual observation are never expected to "
        "match closely at the daily/zone level -- meaningful skill shows up as positive "
        "correlation and >50% directional agreement on real-rain months, not near-zero error)."
    )

    # ------------------------------------------------------------------
    # Soft cross-check against IRI/CPC seasonal, if available
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print("SOFT CROSS-CHECK vs. IRI/CPC seasonal forecast (directional only)")
    print("=" * 60)
    if not os.path.exists(IRI_CPC_PATH):
        print("pipelines/iri_cpc_seasonal/ output not found yet -- skipping this cross-check "
              "(it is a separate, independently-built pipeline; see CLAUDE.md §2 once it lands).")
    else:
        iri_df = pd.read_csv(IRI_CPC_PATH)
        iri_df = iri_df.rename(columns={"pcode": "zone_code"})
        merged = comparable.merge(
            iri_df[["zone_code", "year", "month", "seasonal_precip_prob_below_lead1",
                    "seasonal_precip_prob_above_lead1"]],
            on=["zone_code", "year", "month"], how="inner",
        )
        if merged.empty:
            print("No overlapping zone-months between the two outputs yet -- skipping.")
        else:
            merged["iri_lean"] = np.sign(
                merged["seasonal_precip_prob_above_lead1"] - merged["seasonal_precip_prob_below_lead1"]
            )
            merged["gefs_lean"] = np.sign(merged["usgs_gefs_precip_anom_lead1"])
            agree = (merged["iri_lean"] == merged["gefs_lean"]).mean() * 100
            print(f"{len(merged)} overlapping zone-months. Directional agreement (IRI tercile lean vs. "
                  f"CHIRPS-GEFS anomaly sign): {agree:.1f}%")
            print(
                "NOTE: these are different lead bands (IRI lead1 = target month t+1's full-month "
                "3-month-season-average lean; CHIRPS-GEFS lead1 = ~days 1-15 of month t+1 only) and "
                "different variable types -- disagreement is expected and does not imply either "
                "pipeline is wrong."
            )


if __name__ == "__main__":
    main()
