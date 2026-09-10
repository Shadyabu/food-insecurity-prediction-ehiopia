"""RQ5 Phase 4 — validation of the rule engine's case-level predicted cause
against the project owner's ground-truth labels.

Reads:
  - labeling_worksheet.csv (must have ground_truth_cause filled in for every
    case -- the reconciled label after Phase 3's two labeling passes, per
    the project owner's own process; this script does not adjudicate
    disagreements itself)
  - case_zone_period_aggregates.csv (Phase 2's summed_shap_label /
    majority_vote_label / zone_month_distribution per case x subject)

Produces, per subject (xgboost, randomforest, ensemble):
  - a case-by-case table (the primary evidence at this sample size, per the
    project owner's own instruction -- not just a summary statistic)
  - raw agreement rate
  - Cohen's kappa (predicted vs. ground truth)
  - a chance-agreement baseline via permutation test using the CURATED
    SET'S OWN observed ground-truth category base rates (not a uniform
    1/5 prior) -- scipy.stats.fisher_exact only supports 2x2 contingency
    tables, so an r x c Fisher-Freeman-Halton exact test is approximated
    here by Monte Carlo permutation (shuffle the ground-truth label
    sequence 100,000 times, preserving its exact observed frequency,
    recompute agreement each draw; p = P(perm_agreement >= observed)).
    Documented explicitly as an approximation to the exact test, not the
    exact test itself.
  - the within-case zone-month consistency (Phase 2 step 3a) reported next
    to every case, not just the pooled stats
  - an explicit flag for the "ground truth = CONFLICT but the model
    predicts something else" pattern named in the original spec (a
    predictable failure mode given UNHCR displacement data is excluded
    from the rule engine by design -- see taxonomy_config.yaml)

n=14 -- every statistic below is reported as indicative/descriptive, not
inferential. This script says so in its own output; do not strip that
framing out when copying results into the dissertation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

RQ5_DIR = Path(__file__).resolve().parent
SEED = 42
N_PERMUTATIONS = 100_000
LABELS = ["DROUGHT", "FLOODING", "CONFLICT", "MARKET_SHOCK", "COMPOUND"]


def normalize_label(x: str) -> str:
    """Any label starting with COMPOUND (e.g. 'COMPOUND (DROUGHT + CONFLICT)',
    an annotated variant naming which categories compound) is treated as
    COMPOUND for agreement/kappa/base-rate purposes -- the original string
    stays in the case table for display."""
    x = str(x).strip()
    return "COMPOUND" if x.upper().startswith("COMPOUND") else x


def permutation_chance_baseline(predicted: pd.Series, ground_truth: pd.Series, seed=SEED, n_perm=N_PERMUTATIONS):
    rng = np.random.default_rng(seed)
    observed_agreement = (predicted.to_numpy() == ground_truth.to_numpy()).mean()
    gt_arr = ground_truth.to_numpy().copy()
    n_ge = 0
    for _ in range(n_perm):
        rng.shuffle(gt_arr)
        perm_agreement = (predicted.to_numpy() == gt_arr).mean()
        if perm_agreement >= observed_agreement - 1e-12:
            n_ge += 1
    p_value = (n_ge + 1) / (n_perm + 1)
    return observed_agreement, p_value


def main():
    worksheet = pd.read_csv(RQ5_DIR / "labeling_worksheet.csv")
    agg = pd.read_csv(RQ5_DIR / "case_zone_period_aggregates.csv")

    blank = worksheet["ground_truth_cause"].isna() | (worksheet["ground_truth_cause"].astype(str).str.strip() == "")
    if blank.any():
        print(f"ground_truth_cause is not yet filled in for {blank.sum()}/{len(worksheet)} cases "
              f"({worksheet.loc[blank, 'case_id'].tolist()}).")
        print("Phase 4 statistics require the full curated set labeled (Phase 2 step 2, "
              "reconciled after Phase 3). Showing what Phase 1-2 already produced instead:\n")
        preview_cols = ["case_id", "subject", "n_zone_months", "majority_vote_label",
                         "summed_shap_label", "within_case_consistency"]
        print(agg[preview_cols].to_string(index=False))
        return

    worksheet["ground_truth_cause_norm"] = worksheet["ground_truth_cause"].apply(normalize_label)
    bad = set(worksheet["ground_truth_cause_norm"].unique()) - set(LABELS)
    if bad:
        raise ValueError(f"ground_truth_cause has values outside the taxonomy: {bad}")

    gt_base_rates = worksheet["ground_truth_cause_norm"].value_counts(normalize=True)
    print("=== Observed ground-truth category base rates in the curated set (n={}, COMPOUND-normalized) ===".format(len(worksheet)))
    print(gt_base_rates.to_string())
    print()

    merged = agg.merge(worksheet[["case_id", "zone_codes", "zone_names", "region", "time_period",
                                   "report_url", "ground_truth_cause", "ground_truth_cause_norm"]], on="case_id")

    all_subject_tables = []
    summary_rows = []

    for subject in sorted(merged["subject"].unique()):
        sub = merged[merged["subject"] == subject].copy()
        sub["predicted_cause"] = sub["summed_shap_label"]
        sub["match"] = sub["predicted_cause"] == sub["ground_truth_cause_norm"]

        table_cols = ["case_id", "zone_names", "time_period", "predicted_cause", "ground_truth_cause",
                      "match", "score_DROUGHT", "score_FLOODING", "score_CONFLICT", "score_MARKET_SHOCK",
                      "within_case_consistency", "n_zone_months"]
        case_table = sub[table_cols].copy()
        case_table.insert(0, "subject", subject)
        all_subject_tables.append(case_table)

        agree_rate = sub["match"].mean()
        try:
            kappa = cohen_kappa_score(sub["predicted_cause"], sub["ground_truth_cause_norm"], labels=LABELS)
        except Exception:
            kappa = float("nan")
        obs_agree, p_value = permutation_chance_baseline(sub["predicted_cause"], sub["ground_truth_cause_norm"])

        conflict_gt = sub[sub["ground_truth_cause_norm"] == "CONFLICT"]
        conflict_missed = conflict_gt[conflict_gt["predicted_cause"] != "CONFLICT"]

        print(f"\n=== subject: {subject} ===")
        print(case_table.drop(columns=["subject"]).to_string(index=False))
        print(f"\nraw agreement: {agree_rate:.3f} ({sub['match'].sum()}/{len(sub)})")
        print(f"Cohen's kappa (vs ground truth): {kappa:.3f}")
        print(f"chance-agreement baseline (permutation test, n={N_PERMUTATIONS}, seed={SEED}): "
              f"observed={obs_agree:.3f}, p={p_value:.4f}"
              f" {'(SIGNIFICANT at alpha=0.05 -- but see n=14 caveat)' if p_value < 0.05 else '(not significant)'}")

        if len(conflict_missed):
            print(f"\n*** FLAG: {len(conflict_missed)} case(s) with ground truth CONFLICT where "
                  f"{subject} predicted something else -- UNHCR displacement is excluded from the rule "
                  f"engine by design (taxonomy_config.yaml), so this is exactly the predictable failure "
                  f"mode named in the RQ5 spec. Manually check whether each source_quote_or_paraphrase in "
                  f"labeling_worksheet.csv emphasizes displacement over active violence: "
                  f"{conflict_missed['case_id'].tolist()}")

        summary_rows.append({
            "subject": subject, "n_cases": len(sub), "agreement_rate": agree_rate,
            "cohens_kappa": kappa, "chance_baseline_p_value": p_value,
            "n_conflict_gt_missed": len(conflict_missed),
        })

    full_table = pd.concat(all_subject_tables, ignore_index=True)
    full_table.to_csv(RQ5_DIR / "phase4_case_table.csv", index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RQ5_DIR / "phase4_summary_stats.csv", index=False)

    print(f"\n\nwrote {RQ5_DIR}/phase4_case_table.csv and phase4_summary_stats.csv")
    print("\n" + "=" * 70)
    print("REMINDER: n={} -- every statistic above is indicative/descriptive.".format(len(worksheet)))
    print("Do not present agreement rate, kappa, or the permutation p-value as")
    print("inferential/generalizable evidence beyond this specific curated set.")
    print("=" * 70)


if __name__ == "__main__":
    main()
