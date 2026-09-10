"""RQ3 Experiment 1 -- does predictive performance correlate with data
completeness, across zones and across livelihood zones?

Best model used: the RQ2 Experiment 5 4-way averaging ensemble
(XGBoost + RandomForest + LSTM + TabICLv2, default test window), the highest
mean weighted F1 (0.7218, all zones) of every model/ensemble built in this
project so far (CLAUDE.md Sec 2). Predictions:
experiments/RQ2/experiment_5/all_combinations/predictions/XGB_RF_LSTM_TabICL.parquet

Spatial unit note (read before citing this in the dissertation): this
project's only built spatial key is the 92-unit admin2 boundary
(`zone_code`, CLAUDE.md Sec 3.1) -- there is no wareda/woreda (admin3, a
finer unit than admin2) panel anywhere in this repo. `docs/dissertation_plan.md`'s
RQ3 wording ("per woreda") is loose relative to what was actually built
(RQ2 Experiment 3's own report makes the same correction for its admin2
graph: "not a wareda/livelihood-zone hierarchy"). Everything below is at
admin2 (zone) granularity; treat "zone" and "wareda" as synonyms for this
analysis's purposes, and flag the terminology gap explicitly if citing this
against the plan's literal text.

Reusable pieces (per the project owner's request, for RQ4/RQ5):
- completeness.py: feature-column selection + per-zone/per-cluster completeness scoring
- region_metrics.py: per-zone/per-lhz weighted F1 + balanced accuracy from a predictions parquet
- correlation.py: Spearman rho + percentile-bootstrap CI
"""
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import completeness as cmp
import region_metrics as rm
from correlation import spearman_with_bootstrap_ci

OUT_DIR = Path(__file__).resolve().parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

LHZ_COLORS = {"pastoral": "#c0392b", "agropastoral": "#d68910", "crop_farming": "#1f6f43"}
LHZ_ORDER = ["pastoral", "agropastoral", "crop_farming"]


def build_completeness():
    panel = cmp.load_raw_panel()
    display_names, clusters = cmp.select_feature_columns(panel.columns)
    feature_cols = list(display_names.keys())

    zone_completeness, per_feature = cmp.compute_zone_completeness(panel, feature_cols, clusters)
    zone_completeness = cmp.attach_livelihood_zone(zone_completeness)

    alt = cmp.sensitivity_alt_completeness(per_feature, clusters, feature_cols)
    zone_completeness = zone_completeness.merge(
        alt.rename("completeness_min_cluster").reset_index(), on="zone_code", how="left"
    )

    lhz_completeness = cmp.aggregate_to_livelihood_zone(zone_completeness)

    n_by_cluster = pd.Series(clusters).value_counts()
    print(f"Feature columns scored: {len(feature_cols)} "
          f"({dict(n_by_cluster)})")
    return panel, zone_completeness, lhz_completeness, feature_cols, clusters


def build_f1(predictions_path=None, discretize=None):
    kwargs = {}
    if predictions_path is not None:
        kwargs["predictions_path"] = predictions_path
    if discretize is not None:
        kwargs["discretize"] = discretize
    pred_df = rm.load_predictions(**kwargs)
    zone_f1 = rm.score_by_zone(pred_df)
    lhz_f1 = rm.score_by_lhz(pred_df)
    return pred_df, zone_f1, lhz_f1


def merge_results(zone_completeness, zone_f1):
    merged = zone_completeness.merge(zone_f1, on="zone_code", how="inner", validate="one_to_one")
    n_missing = len(zone_completeness) - len(merged)
    if n_missing:
        print(f"WARNING: {n_missing} zones had completeness but no predictions (or vice versa)")
    return merged


def merge_lhz_results(lhz_completeness, lhz_f1):
    return lhz_completeness.merge(lhz_f1, on="lhz", how="inner", validate="one_to_one")


def completeness_distribution_report(zone_completeness):
    desc = zone_completeness["completeness_mean"].describe()
    print("\n=== Completeness score distribution (zone-level, mean-across-features) ===")
    print(desc.to_string())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(zone_completeness["completeness_mean"], bins=20, color="#4a6fa5", edgecolor="white")
    ax.set_xlabel("Completeness score (mean across features)")
    ax.set_ylabel("Number of zones")
    ax.set_title("Distribution of zone-level data-completeness scores (n=92 admin2 zones)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "completeness_histogram.png", dpi=150)
    plt.close(fig)
    return desc


def scatter_plot(merged):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for lhz in LHZ_ORDER:
        sub = merged[merged["dominant_livelihood_zone"] == lhz]
        ax.scatter(
            sub["completeness_mean"], sub["f1_weighted"],
            label=f"{lhz} (n={len(sub)})", color=LHZ_COLORS[lhz],
            s=55, alpha=0.85, edgecolor="white", linewidth=0.5,
        )

    valid = merged.dropna(subset=["completeness_mean", "f1_weighted"])
    if len(valid) >= 2:
        # OLS line shown purely as a visual trend guide -- the reported
        # statistic is Spearman's rho (rank-based, computed separately),
        # not this line's slope/R^2.
        coeffs = np.polyfit(valid["completeness_mean"], valid["f1_weighted"], 1)
        xs = np.linspace(valid["completeness_mean"].min(), valid["completeness_mean"].max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color="black", linestyle="--", linewidth=1.2,
                label="linear trend (visual guide only)")

    ax.set_xlabel("Data completeness score (zone-level, mean across features)")
    ax.set_ylabel("Weighted F1 (pooled test set, best ensemble)")
    ax.set_title("Data completeness vs. predictive performance, by zone")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_completeness_vs_f1.png", dpi=150)
    plt.close(fig)


def constant_target_diagnostic(merged):
    """Diagnostic surfaced during the run, not part of the original spec:
    zones whose true IPC class never varies across the pooled test set score
    a trivial ~1.0 weighted F1 regardless of model skill. Checked empirically
    (mean F1 0.993 for the 30/92 constant-target zones vs. 0.574 for the
    other 62) -- large enough to matter for the correlation, since these
    zones are not evenly spread across completeness or livelihood zone.
    Reported here and re-run as a robustness check (trivial zones excluded)
    alongside the two correlations the spec actually asked for.
    """
    n_trivial = int(merged["trivial_constant_target"].sum())
    print(f"\n{n_trivial} of {len(merged)} zones have a constant true IPC class across the pooled "
          f"test set (trivial F1 ~1.0 regardless of skill) -- see outputs/rq3_summary.md caveat.")
    by_lhz = merged.groupby("dominant_livelihood_zone")["trivial_constant_target"].mean()
    print("Share of zones that are constant-target, by livelihood zone:")
    print(by_lhz.to_string())
    return n_trivial, by_lhz


def confound_checks(panel, merged):
    """Flag, not model: is completeness itself correlated with something else
    that could independently drive low F1? Checked against what's actually in
    this repo -- conflict intensity (ACLED) and typical food-insecurity
    severity (ipc_continuous). No zone-level population feature exists in
    this repo (WorldPop's raw raster was never aggregated to admin2 as a
    feature -- CLAUDE.md Sec 5's `worldpop/` entry is a raw raster, not a
    processed feature file), so that candidate confound is reported as
    "not checkable here", not fabricated.
    """
    conflict = panel.groupby("zone_code")["acled_event_count"].mean().rename("mean_acled_event_count")
    severity = panel.groupby("zone_code")["ipc_continuous"].mean().rename("mean_ipc_continuous")
    confound_df = merged.merge(conflict, on="zone_code").merge(severity, on="zone_code")

    results = {}
    for col, label in [("mean_acled_event_count", "conflict intensity (mean ACLED events/month)"),
                        ("mean_ipc_continuous", "typical food-insecurity severity (mean ipc_continuous)")]:
        rho, p = spearmanr(confound_df["completeness_mean"], confound_df[col])
        results[label] = {"rho": rho, "p_value": p}
        print(f"Confound check -- completeness vs {label}: rho={rho:.3f}, p={p:.4f}")

    results["population"] = "not checkable: no admin2-level population feature exists in this repo " \
        "(WorldPop is a raw raster under worldpop/, never aggregated to a processed feature)"
    return results


def write_markdown_summary(zone_result, lhz_result, sensitivity_result, confound_results,
                            zone_excluded, completeness_desc, n_zones, n_features,
                            n_trivial, trivial_by_lhz, robustness_result):
    lines = []
    lines.append("# RQ3 Results -- Data Completeness vs. Predictive Performance\n")
    lines.append(f"**Bottom line:** the primary (mean-of-features, n=92) and livelihood-zone "
                 f"(n=3) correlations do **not** reject H0. The required sensitivity check "
                 f"(min-of-cluster-means) does reject H0 -- but in the **opposite direction** "
                 f"to the fairness-bias hypothesis (rho={sensitivity_result['rho']:.3f}: "
                 f"*less*-complete zones score *higher* F1). A post-hoc diagnostic found this is "
                 f"driven by 30/92 zones with a constant true IPC class in the test set (trivial "
                 f"F1~1.0 regardless of skill, concentrated in crop_farming, CLAUDE.md-style "
                 f"verify-before-trust §5); excluding them flips the primary correlation "
                 f"positive (rho={robustness_result['rho']:.3f}) but still short of significance "
                 f"(p={robustness_result['p_value']:.3f}). Net finding: **no robust evidence of a "
                 f"completeness-driven performance bias in either direction** at this sample "
                 f"size -- not the clean fairness-bias confirmation the RQ3 hypothesis "
                 f"anticipated, and not a clean rebuttal either.\n")
    lines.append(f"**Model:** RQ2 4-way averaging ensemble (XGBoost + RandomForest + LSTM + "
                 f"TabICLv2, default test window) -- this project's highest mean weighted F1 "
                 f"to date (0.7218, all zones, CLAUDE.md Sec 2).\n")
    lines.append(f"**Spatial unit:** 92 admin2 zones (`zone_code`) -- this repo has no "
                 f"wareda/woreda (admin3) panel; `dissertation_plan.md`'s \"woreda\" wording is "
                 f"loose relative to what was actually built (see RQ2 Experiment 3's own "
                 f"correction of the same point). Read \"wareda\" as \"admin2 zone\" throughout.\n")
    lines.append(f"**Feature set scored for completeness:** {n_features} feature columns across "
                 f"climate / agriculture / economic / conflict clusters (market data folds into "
                 f"\"economic\" -- this repo's own `FEATURE_CLUSTER_PREFIXES` taxonomy, "
                 f"`experiments/RQ1/experiment_2/run_model_ethiopia.py`, has no separate "
                 f"\"market\" bucket; WFP prices/ToT live under \"economic\").\n")

    lines.append("## 1. Completeness score distribution (zone-level)\n")
    lines.append("```")
    lines.append(completeness_desc.to_string())
    lines.append("```")
    lines.append("See `outputs/completeness_histogram.png` and `outputs/completeness_by_zone.csv`.\n")

    lines.append("## 2. Per-region F1\n")
    n_excl = zone_excluded
    lines.append(f"{n_excl} of {n_zones} zones excluded from the correlation for having fewer than "
                 f"{rm.MIN_TEST_OBS} pooled test observations "
                 f"({'none -- every zone has 63 pooled (month x lead) test rows' if n_excl == 0 else 'see outputs/f1_by_zone.csv for which'}).\n")

    lines.append("## 3. Correlation: completeness vs. weighted F1\n")
    for label, res in [("Zone level (a)", zone_result), ("Livelihood-zone level (b)", lhz_result)]:
        lines.append(f"**{label}:** n={res['n']}, rho={res['rho']:.4f}, p={res['p_value']:.4f}, "
                     f"95% bootstrap CI=[{res['ci_low']:.4f}, {res['ci_high']:.4f}] "
                     f"({res['n_boot']} resamples, seed=42). "
                     f"H0 {'REJECTED' if res['reject_h0_at_0.05'] else 'NOT REJECTED'} at p<0.05.\n")
    lines.append("**Livelihood-zone-level caveat:** n=3 (pastoral/agropastoral/crop_farming) is far "
                 "too small for a Spearman correlation (and its bootstrap CI) to be statistically "
                 "meaningful -- report as descriptive only, not as a second independent test of H0.\n")

    lines.append("## 4. Sensitivity check: min-of-cluster-means completeness\n")
    lines.append(f"Zone level, alternative completeness definition (min across "
                 f"climate/agriculture/economic/conflict cluster means, instead of the primary "
                 f"mean-of-all-features): n={sensitivity_result['n']}, rho={sensitivity_result['rho']:.4f}, "
                 f"p={sensitivity_result['p_value']:.4f}, "
                 f"95% CI=[{sensitivity_result['ci_low']:.4f}, {sensitivity_result['ci_high']:.4f}]. "
                 f"H0 {'REJECTED' if sensitivity_result['reject_h0_at_0.05'] else 'NOT REJECTED'} at p<0.05 "
                 f"-- {'same conclusion as the primary definition' if sensitivity_result['reject_h0_at_0.05'] == zone_result['reject_h0_at_0.05'] else 'DIFFERENT conclusion from the primary definition -- flag this'}.\n")

    lines.append("## 5. Diagnostic: constant-target zones (surfaced during this run, not in the "
                 "original spec)\n")
    lines.append(f"{n_trivial} of {n_zones} zones have only one true IPC class across the pooled "
                 f"test set, scoring a trivial ~1.0 weighted F1 regardless of model skill "
                 f"(checked: mean F1 0.993 for these zones vs. 0.574 for the rest). Share of "
                 f"constant-target zones by livelihood zone:\n")
    lines.append("```")
    lines.append(trivial_by_lhz.to_string())
    lines.append("```")
    lines.append(f"**Robustness re-run excluding these {n_trivial} zones** (primary, "
                 f"mean-of-features completeness definition): n={robustness_result['n']}, "
                 f"rho={robustness_result['rho']:.4f}, p={robustness_result['p_value']:.4f}, "
                 f"95% CI=[{robustness_result['ci_low']:.4f}, {robustness_result['ci_high']:.4f}]. "
                 f"H0 {'REJECTED' if robustness_result['reject_h0_at_0.05'] else 'NOT REJECTED'} at p<0.05 "
                 f"-- {'same conclusion as the full 92-zone primary result' if robustness_result['reject_h0_at_0.05'] == zone_result['reject_h0_at_0.05'] else 'DIFFERENT conclusion from the full 92-zone primary result -- flag this'}.\n")

    lines.append("## 6. Confound checks (flagged, not modeled)\n")
    for label, res in confound_results.items():
        if isinstance(res, dict):
            lines.append(f"- Completeness vs. {label}: rho={res['rho']:.3f}, p={res['p_value']:.4f}")
        else:
            lines.append(f"- Population: {res}")
    lines.append("")

    lines.append("## 7. Artefacts\n")
    lines.append("- `outputs/completeness_by_zone.csv`, `outputs/completeness_by_lhz.csv`")
    lines.append("- `outputs/f1_by_zone.csv`, `outputs/f1_by_lhz.csv`")
    lines.append("- `outputs/rq3_results_zone.csv`, `outputs/rq3_results_lhz.csv` (merged, used for correlation)")
    lines.append("- `outputs/completeness_histogram.png`, `outputs/scatter_completeness_vs_f1.png`")
    lines.append("- `outputs/correlation_results.json`\n")

    text = "\n".join(lines)
    (OUT_DIR / "rq3_summary.md").write_text(text)
    return text


def main():
    panel, zone_completeness, lhz_completeness, feature_cols, clusters = build_completeness()
    pred_df, zone_f1, lhz_f1 = build_f1()

    zone_completeness.to_csv(OUT_DIR / "completeness_by_zone.csv", index=False)
    lhz_completeness.to_csv(OUT_DIR / "completeness_by_lhz.csv", index=False)
    zone_f1.to_csv(OUT_DIR / "f1_by_zone.csv", index=False)
    lhz_f1.to_csv(OUT_DIR / "f1_by_lhz.csv", index=False)

    merged_zone = merge_results(zone_completeness, zone_f1)
    merged_lhz = merge_lhz_results(lhz_completeness, lhz_f1)
    merged_zone.to_csv(OUT_DIR / "rq3_results_zone.csv", index=False)
    merged_lhz.to_csv(OUT_DIR / "rq3_results_lhz.csv", index=False)

    completeness_desc = completeness_distribution_report(zone_completeness)
    scatter_plot(merged_zone)

    valid_zone = merged_zone[~merged_zone["excluded"]]
    zone_excluded = int(merged_zone["excluded"].sum())

    zone_result = spearman_with_bootstrap_ci(
        valid_zone["completeness_mean"], valid_zone["f1_weighted"]
    )
    lhz_result = spearman_with_bootstrap_ci(
        merged_lhz["completeness_mean"], merged_lhz["f1_weighted"]
    )
    sensitivity_result = spearman_with_bootstrap_ci(
        valid_zone["completeness_min_cluster"], valid_zone["f1_weighted"]
    )

    print("\n=== Zone-level correlation ===")
    print(zone_result)
    print("\n=== Livelihood-zone-level correlation (n=3, underpowered) ===")
    print(lhz_result)
    print("\n=== Sensitivity check (min-of-cluster-means) ===")
    print(sensitivity_result)

    n_trivial, trivial_by_lhz = constant_target_diagnostic(valid_zone)
    non_trivial = valid_zone[~valid_zone["trivial_constant_target"]]
    robustness_result = spearman_with_bootstrap_ci(
        non_trivial["completeness_mean"], non_trivial["f1_weighted"]
    )
    print("\n=== Robustness re-run (constant-target zones excluded) ===")
    print(robustness_result)

    confound_results = confound_checks(panel, merged_zone)

    with open(OUT_DIR / "correlation_results.json", "w") as f:
        json.dump({
            "zone_level": zone_result,
            "lhz_level": lhz_result,
            "sensitivity_min_cluster": sensitivity_result,
            "robustness_excl_constant_target": robustness_result,
            "n_constant_target_zones": n_trivial,
            "confounds": {k: v for k, v in confound_results.items() if isinstance(v, dict)},
            "n_zones_excluded": zone_excluded,
            "n_features_scored": len(feature_cols),
        }, f, indent=2, default=float)

    write_markdown_summary(
        zone_result, lhz_result, sensitivity_result, confound_results,
        zone_excluded, completeness_desc, n_zones=len(zone_completeness), n_features=len(feature_cols),
        n_trivial=n_trivial, trivial_by_lhz=trivial_by_lhz, robustness_result=robustness_result,
    )

    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
