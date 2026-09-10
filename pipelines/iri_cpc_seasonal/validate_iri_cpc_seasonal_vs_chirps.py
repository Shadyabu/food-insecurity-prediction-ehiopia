"""
validate_iri_cpc_seasonal_vs_chirps.py

SUPPLEMENTARY skill check, separate from validate_iri_cpc_seasonal_output.py
(kept as its own script per CLAUDE.md 3.6 -- a different concern, a
different data dependency, independently re-runnable).

The structural checks in validate_iri_cpc_seasonal_output.py confirm the
IRI forecast data was extracted correctly (well-formed probabilities,
no leakage), but say nothing about whether the forecast has any real
relationship to what actually happened. This script checks that, using
CHIRPS's own already-downloaded/rasterized daily rainfall
(data/interim/chirps/chirps_admin2_daily_combined.csv, 1981-01 to
2026-06, 92 zones) as ground truth.

METHOD
------
1. For each zone, compute the observed 3-month-forward rainfall total
   starting at every (year, month) -- i.e. the exact same "3 consecutive
   months starting at a given month" window the IRI forecast itself
   targets (season_months() from compute_iri_cpc_seasonal_admin2_monthly_
   features.py).
2. For each zone and each of the 12 possible season-start months (Jan..
   Dec), bin that zone-season's full 1981-2026 historical record of
   3-month totals into empirical terciles (33rd/67th percentile cutoffs)
   -- this is each zone's own climatology, the same conceptual baseline
   IRI's own tercile categories are defined against (though IRI's own
   published baseline is a fixed 1991-2020 window -- using the full
   1981-2026 CHIRPS record instead is a deliberate, documented choice:
   more years gives more stable tercile cutoffs, and the exact baseline
   window mismatch is a minor, expected source of disagreement, not the
   thing this check is trying to detect).
3. For every real (non-NaN) forecast row, look up what CHIRPS actually
   observed for that row's target season, bin it into below/near/above
   using that zone-season's own climatology, and compare against the
   forecast's own probabilities.

METRICS (both reported against the 33.3% no-skill baseline)
-------------------------------------------------------------------------
- Hit rate: does the forecast's HIGHEST-probability category match what
  was actually observed?
- Mean probability assigned to the category that actually occurred: a
  simple calibration-style check -- if the forecast has real skill, it
  should on average put MORE than 33.3% probability on whichever category
  actually happens, not just get the argmax right sometimes.

Both are reported split by lead time (1 vs 3) and by source product
(legacy_two_tier vs nmme_elr), since skill could plausibly differ across
either split and averaging them together would hide that.
"""

import os
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, SCRIPT_DIR)
from compute_iri_cpc_seasonal_admin2_monthly_features import season_months  # noqa: E402

CHIRPS_DAILY_PATH = os.path.join(REPO_ROOT, "data", "interim", "chirps", "chirps_admin2_daily_combined.csv")
IRI_OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "iri_cpc_seasonal_admin2_monthly.csv")

LEAD_TIMES = [1, 3]
CATEGORIES = ["below", "near", "above"]


def month_index(year, month):
    return year * 12 + (month - 1)


def index_to_year_month(idx):
    return idx // 12, idx % 12 + 1


if __name__ == "__main__":
    print("Loading CHIRPS daily data ...")
    chirps = pd.read_csv(CHIRPS_DAILY_PATH)
    chirps["date"] = pd.to_datetime(chirps["date"])
    chirps["year"] = chirps["date"].dt.year
    chirps["month"] = chirps["date"].dt.month
    monthly = chirps.groupby(["pcode", "year", "month"])["precipitation_mm"].sum().rename("total_mm").reset_index()
    print(f"  {monthly.shape[0]} zone-months, {monthly['pcode'].nunique()} zones, "
          f"{monthly['year'].min()}-{monthly['year'].max()}")

    # ------------------------------------------------------------------
    # Observed 3-month-forward totals, indexed by (pcode, month_index of
    # the season's OWN start month) -- same convention as season_months().
    # ------------------------------------------------------------------
    monthly["month_idx"] = monthly.apply(lambda r: month_index(int(r["year"]), int(r["month"])), axis=1)
    monthly_lookup = monthly.set_index(["pcode", "month_idx"])["total_mm"]

    print("Computing observed 3-month-forward totals per zone/season-start ...")
    all_pcodes = monthly["pcode"].unique()
    min_idx = monthly["month_idx"].min()
    max_idx = monthly["month_idx"].max()

    records = []
    for pcode in all_pcodes:
        sub = monthly_lookup.loc[pcode] if pcode in monthly_lookup.index.get_level_values(0) else None
        if sub is None:
            continue
        idx_to_val = sub.to_dict()
        for start_idx in range(min_idx, max_idx - 1):  # needs start, start+1, start+2 all present
            if start_idx in idx_to_val and (start_idx + 1) in idx_to_val and (start_idx + 2) in idx_to_val:
                total = idx_to_val[start_idx] + idx_to_val[start_idx + 1] + idx_to_val[start_idx + 2]
                records.append({"pcode": pcode, "start_month_idx": start_idx, "observed_3m_total": total})

    season_df = pd.DataFrame(records)
    start_year_month = season_df["start_month_idx"].apply(index_to_year_month)
    season_df["start_year"] = start_year_month.apply(lambda t: t[0])
    season_df["start_month"] = start_year_month.apply(lambda t: t[1])
    print(f"  {season_df.shape[0]} zone-season observed totals computed.")

    # ------------------------------------------------------------------
    # Per zone / season-start-month climatological terciles (full record)
    # ------------------------------------------------------------------
    print("Computing per-zone, per-season-start-month climatological terciles ...")

    def tercile_bin(group):
        vals = group["observed_3m_total"].to_numpy()
        lo, hi = np.percentile(vals, [100 / 3, 200 / 3])
        bins = np.where(vals <= lo, "below", np.where(vals >= hi, "above", "near"))
        return pd.Series(bins, index=group.index)

    season_df["observed_category"] = (
        season_df.groupby(["pcode", "start_month"], group_keys=False).apply(tercile_bin)
    )

    obs_lookup = season_df.set_index(["pcode", "start_year", "start_month"])["observed_category"]

    # ------------------------------------------------------------------
    # Merge against the IRI forecast output
    # ------------------------------------------------------------------
    print("Loading IRI forecast output and matching against observed categories ...")
    iri = pd.read_csv(IRI_OUTPUT_PATH)

    results = []
    for lead in LEAD_TIMES:
        below_col = f"seasonal_precip_prob_below_lead{lead}"
        near_col = f"seasonal_precip_prob_near_lead{lead}"
        above_col = f"seasonal_precip_prob_above_lead{lead}"
        src_col = f"seasonal_source_lead{lead}"

        for _, row in iri.iterrows():
            if pd.isna(row[below_col]):
                continue
            issue_year, issue_month = int(row["year"]), int(row["month"])
            months = season_months(issue_year, issue_month, lead)
            start_year, start_month = months[0]
            key = (row["pcode"], start_year, start_month)
            if key not in obs_lookup.index:
                continue
            observed_category = obs_lookup.loc[key]
            probs = {"below": row[below_col], "near": row[near_col], "above": row[above_col]}
            max_val = max(probs.values())
            # A genuine lean requires a STRICT, unique max -- ~40% of rows
            # (mostly the integer-rounded legacy product) are an exact
            # 33/33/33 climatological-neutral forecast with no real
            # "dominant" pick. Naive argmax (Python max()/pandas idxmax())
            # silently breaks such ties in favor of whichever key comes
            # first ("below"), which was confirmed to artificially inflate
            # below's hit rate and crush near's -- caught by checking the
            # dominant-category distribution directly (11,402 "below" vs
            # an expected ~1/3 of ~19,100). Ties are excluded from the
            # hit-rate metric rather than arbitrarily resolved.
            tied_for_max = sum(1 for v in probs.values() if v == max_val) > 1
            dominant_category = None if tied_for_max else max(probs, key=probs.get)
            # Official IRI Heidke hit proportion (per "Descriptions of the
            # IRI Climate Forecast Verification Scores", iri.columbia.edu):
            # partial credit = 1 / (# categories tied for the max) if the
            # observed category is among the tied-for-max set, else 0. A
            # full 33/33/33 climatological forecast always scores exactly
            # 1/3 by this convention, regardless of what was observed --
            # this is IRI's own official scoring rule, distinct from this
            # script's separate "leaning forecasts only" hit rate above
            # (which excludes ties instead of partial-crediting them), and
            # is the metric directly comparable to IRI's own published
            # skill numbers.
            tied_categories = [c for c, v in probs.items() if v == max_val]
            heidke_credit = (1.0 / len(tied_categories)) if observed_category in tied_categories else 0.0
            results.append({
                "pcode": row["pcode"], "lead": lead, "source": row[src_col],
                "observed_category": observed_category, "dominant_category": dominant_category,
                "is_tie": tied_for_max,
                "hit": (None if tied_for_max else observed_category == dominant_category),
                "heidke_credit": heidke_credit,
                "prob_assigned_to_observed": probs[observed_category],
            })

    results_df = pd.DataFrame(results)
    max_possible = iri.shape[0] * len(LEAD_TIMES)
    print(f"\nMatched {results_df.shape[0]} (zone-month, lead) forecast rows against observed CHIRPS "
          f"outcomes (out of {max_possible} possible = {iri.shape[0]} rows x {len(LEAD_TIMES)} leads -- "
          f"unmatched cases are either NaN forecasts, or target seasons extending past CHIRPS's own "
          f"June-2026 coverage).")

    print("\n" + "=" * 70)
    print("SKILL CHECK RESULTS (33.3% = no-skill baseline for both metrics)")
    print("=" * 70)
    for lead in LEAD_TIMES:
        for source in ["legacy_two_tier", "nmme_elr"]:
            sub = results_df[(results_df["lead"] == lead) & (results_df["source"] == source)]
            if len(sub) == 0:
                continue
            leaning = sub[~sub["is_tie"]]
            n_ties = int(sub["is_tie"].sum())
            hit_rate = leaning["hit"].mean() if len(leaning) else float("nan")
            mean_prob_assigned = sub["prob_assigned_to_observed"].mean()
            heidke_hit_proportion = sub["heidke_credit"].mean()
            print(f"\nlead{lead} / {source} (n={len(sub)}, {n_ties} exact-tie/no-lean rows):")
            print(f"  Hit rate on leaning forecasts only (n={len(leaning)}):  {hit_rate:.1%}  (baseline 33.3%)")
            print(f"  Mean probability assigned to observed category:        {mean_prob_assigned:.1f}%  (baseline 33.3%, all rows incl. ties)")
            print(f"  Official IRI Heidke hit proportion (all rows, ties get partial credit): "
                  f"{heidke_hit_proportion:.1%}  (baseline 33.3% -- directly comparable to IRI's own published skill numbers)")

    print("\nBreakdown by observed category (checks for systematic bias, not just overall skill; "
          "hit rate computed on leaning forecasts only, ties excluded):")
    for lead in LEAD_TIMES:
        sub = results_df[results_df["lead"] == lead]
        print(f"\n  lead{lead}:")
        for obs_cat in CATEGORIES:
            cat_sub = sub[sub["observed_category"] == obs_cat]
            cat_leaning = cat_sub[~cat_sub["is_tie"]]
            if len(cat_sub) == 0:
                continue
            print(f"    when observed was '{obs_cat}' (n={len(cat_sub)}): "
                  f"mean prob assigned to '{obs_cat}' = {cat_sub['prob_assigned_to_observed'].mean():.1f}%, "
                  f"hit rate = {cat_leaning['hit'].mean():.1%} (n_leaning={len(cat_leaning)})")
