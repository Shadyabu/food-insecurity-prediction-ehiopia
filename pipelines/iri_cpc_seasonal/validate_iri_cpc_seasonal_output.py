"""
validate_iri_cpc_seasonal_output.py

VALIDATE stage. Per CLAUDE.md 3.5/7 and this pipeline's own task spec:
no Busker reference exists for this source (master table confirms no
`busker_comparison/` file was staged for it, and Busker et al. don't use
this feature), so validation instead:

1. Structural checks: 92/92 zones present; complete rectangular
   pcode x year x month panel; `seasonal_source_lead{1,3}` only the
   documented values; target_season strings are all valid 3-letter codes.
2. Tercile-sum sanity check: below + near + above ~= 100 per zone-month-
   lead. Reported SEPARATELY by source, because the two source products
   have genuinely different numeric precision (confirmed below) -- this
   is not a bug to average away.
3. Leakage check: the target season for lead L must start STRICTLY after
   the issue month (i.e. never include the issue month itself, and lead1
   must never reach further back than lead3) -- the one most likely to
   silently break if the season-offset formula is ever touched, mirroring
   GloFAS's own most-likely-to-break leakage check (Sec 8).
4. Historical cross-check: 2015 Kiremt (El Nino-driven drought) as a
   qualitative below-average-leaning sanity check, reported honestly
   either way -- per CLAUDE.md 3.5 and the GloFAS precedent ("report
   whichever of these checks the archive's real start date allows, rather
   than assuming").
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "iri_cpc_seasonal_admin2_monthly.csv")
ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")

LEAD_TIMES = [1, 3]
VALID_SOURCES = {"legacy_two_tier", "nmme_elr"}
VALID_SEASON_CODES = {
    "jfm", "fma", "mam", "amj", "mjj", "jja", "jas", "aso", "son", "ond", "ndj", "djf",
}


def month_index(year, month):
    return year * 12 + (month - 1)


if __name__ == "__main__":
    df = pd.read_csv(OUTPUT_PATH)
    print(f"Loaded {df.shape[0]} rows, {df['pcode'].nunique()} zones, "
          f"{df['year'].min()}-{df['month'].min():02d} to {df['year'].max()}-{df['month'].max():02d}")

    # ------------------------------------------------------------------
    # 1. Structural checks
    # ------------------------------------------------------------------
    print("\n[1] Structural checks")

    import geopandas as gpd
    gdf = gpd.read_file(ADMIN2_SHP_PATH)
    expected_zones = set(gdf["ADM2_PCODE"])
    actual_zones = set(df["pcode"])
    print(f"  Zones: {len(actual_zones)}/92 present. "
          f"Missing: {expected_zones - actual_zones if expected_zones - actual_zones else 'none'}")

    n_months = df[["year", "month"]].drop_duplicates().shape[0]
    expected_rows = n_months * df["pcode"].nunique()
    print(f"  Rectangular panel check: {df.shape[0]} rows vs {n_months} months x "
          f"{df['pcode'].nunique()} zones = {expected_rows} expected "
          f"({'OK' if df.shape[0] == expected_rows else 'MISMATCH'})")

    for lead in LEAD_TIMES:
        src_col = f"seasonal_source_lead{lead}"
        bad_sources = set(df[src_col].dropna().unique()) - VALID_SOURCES
        print(f"  lead{lead} source values: {sorted(df[src_col].dropna().unique())} "
              f"({'OK' if not bad_sources else f'UNEXPECTED VALUES: {bad_sources}'})")

        season_col = f"seasonal_target_season_lead{lead}"
        bad_seasons = set(df[season_col].dropna().unique()) - VALID_SEASON_CODES
        print(f"  lead{lead} target_season values: {'OK, all valid 3-letter codes' if not bad_seasons else f'INVALID: {bad_seasons}'}")

    # ------------------------------------------------------------------
    # 2. Tercile-sum sanity check, split by source
    # ------------------------------------------------------------------
    print("\n[2] Tercile-sum sanity check (below + near + above, expect ~100)")
    for lead in LEAD_TIMES:
        below = df[f"seasonal_precip_prob_below_lead{lead}"]
        near = df[f"seasonal_precip_prob_near_lead{lead}"]
        above = df[f"seasonal_precip_prob_above_lead{lead}"]
        total = below + near + above
        src = df[f"seasonal_source_lead{lead}"]
        print(f"  lead{lead}:")
        for source_name in ["nmme_elr", "legacy_two_tier"]:
            mask = (src == source_name) & total.notna()
            vals = total[mask]
            if len(vals) == 0:
                continue
            within_1pct = ((vals - 100).abs() <= 1.001).mean()  # inclusive: legacy's integer
                                                                  # rounding can land exactly at 99
            print(f"    {source_name}: n={len(vals)}, mean={vals.mean():.3f}, "
                  f"min={vals.min():.3f}, max={vals.max():.3f}, within 1pt={within_1pct:.1%}")
        n_no_source = total[src.isna()].notna().sum()
        print(f"    rows with no source but non-NaN prob sum (should be 0): {n_no_source}")

    print("\n  NOTE: legacy_two_tier sums cluster at 99-100, not exactly 100 like nmme_elr "
          "(confirmed via raw NetCDF inspection 2026-08-09: the legacy 2.5deg product's "
          "published probabilities are pre-rounded to the nearest whole percent by IRI "
          "itself, e.g. 33/33/33 sums to 99 -- a genuine source-precision difference, not "
          "a pipeline bug. nmme_elr sums to 100 to within float precision on effectively "
          "100% of rows.")

    # ------------------------------------------------------------------
    # 3. Leakage check
    # ------------------------------------------------------------------
    print("\n[3] Leakage check: target season must start strictly after the issue month")
    issue_idx = df["year"] * 12 + (df["month"] - 1)

    # Re-derive season start month index from the target_season code isn't
    # directly invertible (codes repeat every 12 months), so instead
    # re-verify the SOURCE FORMULA directly: season_months() must never
    # place the issue month itself inside the lead-L window, for every L.
    import sys
    sys.path.insert(0, SCRIPT_DIR)
    from compute_iri_cpc_seasonal_admin2_monthly_features import season_months

    violations = 0
    checked = 0
    for _, row in df[["year", "month"]].drop_duplicates().iterrows():
        y, m = int(row["year"]), int(row["month"])
        issue_i = month_index(y, m)
        for lead in LEAD_TIMES:
            months = season_months(y, m, lead)
            season_indices = [month_index(sy, sm) for sy, sm in months]
            checked += 1
            if min(season_indices) <= issue_i:
                violations += 1
        # lead1's season must never start later than lead3's season
        s1 = min(month_index(sy, sm) for sy, sm in season_months(y, m, 1))
        s3 = min(month_index(sy, sm) for sy, sm in season_months(y, m, 3))
        if s1 >= s3:
            violations += 1

    print(f"  Checked {checked} (issue-month, lead) combinations across all distinct issue months.")
    print(f"  Leakage violations: {violations} ({'OK' if violations == 0 else 'FAIL'})")

    # ------------------------------------------------------------------
    # 4. Historical cross-check: 2015 Kiremt (El Nino drought)
    # ------------------------------------------------------------------
    print("\n[4] Historical cross-check: 2015 Kiremt (JJAS) -- known El Nino-driven "
          "below-average rains in parts of Ethiopia")
    kiremt_2015 = df[
        (df["year"] == 2015) & (df["month"].isin([4, 5, 6]))
        & (df["seasonal_overlaps_kiremt_lead1"] == True)  # noqa: E712
    ]
    summary = kiremt_2015.groupby("month")[
        ["seasonal_precip_prob_below_lead1", "seasonal_precip_prob_above_lead1"]
    ].mean(numeric_only=True)
    print(summary.to_string())
    print("  Interpretation: reported as-is, not oversold -- a below>above lean here would be "
          "qualitatively consistent with the documented 2015 event, a near-climatological "
          "(33/33) or above-leaning result is also plausible given forecast skill limits at "
          "this lead time and is not treated as a pipeline failure either way (per CLAUDE.md "
          "3.5's cheapest-to-most-expensive investigation order -- this is a single qualitative "
          "check, not a systematic reference).")
