"""RQ5 Phase 2 step 3 — aggregate the rule engine's zone-month predictions
UP to each curated case's zone(s)/time-period.

RESOLUTION NOTE (confirmed with project owner 2026-08-31): this project has
no wareda/admin3 panel anywhere (CLAUDE.md's RQ3 Experiment 1 entry already
establishes this) -- the finest resolution anywhere in this repo is admin2
(zone_code, 92 zones). The original RQ5 spec's "aggregate wareda-month
instances up to livelihood zone" is therefore reframed here as "aggregate
zone-month instances up to the case's zone(s)/period" -- the admin2 zone IS
the base unit, not an intermediate one being aggregated from something
finer. This still gives everything the original design wanted:
  (a) the distribution of zone-month predicted causes within a case --
      diagnostic for whether the case's zone(s) are internally consistent
      or mixed, same purpose as the original wareda-level diagnostic, just
      at the resolution this project actually has.
  (b) a single case-level predicted cause label, by two methods:
      - majority vote of the per-zone-month predicted_cause labels
      - summed-SHAP: sum each category's raw SHAP score across every
        zone-month in the case, then apply the identical argmax/compound
        logic rule_engine.score_instance uses on a single instance, to
        this pooled category-score vector. This is the primary label used
        in Phase 4 -- majority vote is reported alongside as the
        diagnostic cross-check the original spec asked for.

Only lead=0 (nowcast) rows are used for case-level scoring -- FEWS NET's
"current situation" narrative describes conditions AT the report month, not
a multi-month-ahead forecast, so lead0 is the directly comparable local
explanation. Other leads remain in predicted_causes.csv for any later
lead-sensitivity extension, not used here.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rule_engine import CATEGORY_NAMES, COMPOUND, UNCLASSIFIED, load_taxonomy

RQ5_DIR = Path(__file__).resolve().parent
LEAD_USED = 0


def parse_period(time_period: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_s, end_s = [s.strip() for s in time_period.split(" to ")]
    start = pd.Timestamp(start_s + "-01")
    end = pd.Timestamp(end_s + "-01") + pd.offsets.MonthEnd(1)
    return start, end


def aggregate_case(case_row: pd.Series, predictions: pd.DataFrame, taxonomy: dict, margin_threshold: float | None = None):
    if margin_threshold is None:
        margin_threshold = taxonomy["compound"]["margin_threshold"]

    zone_codes = [z.strip() for z in case_row["zone_codes"].split(",")]
    start, end = parse_period(case_row["time_period"])

    sub = predictions[
        (predictions["zone_code"].isin(zone_codes))
        & (predictions["lead"] == LEAD_USED)
        & (predictions["time"] >= start)
        & (predictions["time"] <= end)
    ]

    out = {}
    for subject, g in sub.groupby("subject"):
        n = len(g)
        vote_counts = g["predicted_cause"].value_counts().to_dict()
        majority_label = max(vote_counts, key=vote_counts.get) if vote_counts else "NO_ROWS"

        score_sums = {cat: g[f"score_{cat}"].sum() for cat in CATEGORY_NAMES}
        ranked = sorted(CATEGORY_NAMES, key=lambda c: score_sums[c], reverse=True)
        top1, top2 = ranked[0], ranked[1]
        top1_score, top2_score = score_sums[top1], score_sums[top2]
        if n == 0:
            summed_label = "NO_ROWS"
            margin = float("nan")
        elif top1_score <= 0:
            summed_label = UNCLASSIFIED
            margin = float("nan")
        else:
            margin = (top1_score - max(top2_score, 0.0)) / top1_score
            summed_label = COMPOUND if (top2_score > 0 and margin <= margin_threshold) else top1

        out[subject] = {
            "n_zone_months": n,
            "wareda_level_distribution": vote_counts,  # naming kept for spec traceability -- see module docstring
            "majority_vote_label": majority_label,
            "summed_shap_label": summed_label,
            "summed_shap_scores": score_sums,
            "summed_shap_margin": margin,
            "within_case_consistency": (max(vote_counts.values()) / n) if n else float("nan"),
        }
    return out


def main():
    taxonomy = load_taxonomy()
    worksheet = pd.read_csv(RQ5_DIR / "labeling_worksheet.csv")
    predictions = pd.read_csv(RQ5_DIR / "predicted_causes_2020_2024.csv", parse_dates=["time"])

    records = []
    for _, case_row in worksheet.iterrows():
        agg = aggregate_case(case_row, predictions, taxonomy)
        for subject, result in agg.items():
            records.append({
                "case_id": case_row["case_id"],
                "subject": subject,
                "candidate_taxonomy_cause": case_row["candidate_taxonomy_cause"],
                **{k: v for k, v in result.items() if k != "summed_shap_scores" and k != "wareda_level_distribution"},
                **{f"score_{c}": result["summed_shap_scores"][c] for c in CATEGORY_NAMES},
                "zone_month_distribution": result["wareda_level_distribution"],
            })

    out = pd.DataFrame(records)
    out.to_csv(RQ5_DIR / "case_zone_period_aggregates_2020_2024.csv", index=False)
    print(f"wrote {len(out)} (case x subject) aggregate rows to case_zone_period_aggregates.csv")
    print(out[["case_id", "subject", "n_zone_months", "majority_vote_label", "summed_shap_label", "within_case_consistency"]].to_string(index=False))


if __name__ == "__main__":
    main()
