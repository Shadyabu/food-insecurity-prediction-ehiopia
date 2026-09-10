"""RQ3 rebuilt on the 2020-2024 window (the project's primary window as of
the 2026-09-02 test-window revision, docs/dissertation_plan.md section 6),
at both the 1-5 and 1/2/3+ scales -- the original run_rq3_analysis.py
predates that revision and stays on 2020-2022, 1-5 only, unchanged.

Reuses completeness.py's build_completeness() as-is (completeness is a
feature-availability property of the whole historical panel, not
test-window-dependent -- unchanged by this rebuild) and region_metrics.py's
now-parameterized load_predictions()/score_by_zone()/score_by_lhz() (see
that file's 2026-09-02 addition) pointed at the extended-window champion
ensemble (experiments/RQ2/experiment_5/all_combinations_extended/predictions/
XGB_RF_LSTM_TabICL.parquet) instead of the default-window one.

Produces one output directory per discretization (outputs_2020_2024_1to5/,
outputs_2020_2024_3plus/) with the same core statistics as the original --
zone-level correlation, livelihood-zone-level correlation, sensitivity
check (min-of-cluster-means), robustness check (constant-target zones
excluded), confound checks, and a scatter figure. Does not reproduce every
prose paragraph of the original's rq3_summary.md verbatim -- a leaner
summary sufficient to cite, not a duplicate of the original's full report.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import completeness as cmp
import region_metrics as rm
from correlation import spearman_with_bootstrap_ci

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTENDED_PREDICTIONS_PATH = (
    REPO_ROOT / "experiments" / "RQ2" / "experiment_5" / "all_combinations_extended"
    / "predictions" / "XGB_RF_LSTM_TabICL.parquet"
)

LHZ_COLORS = {"pastoral": "#c0392b", "agropastoral": "#d68910", "crop_farming": "#1f6f43"}
LHZ_ORDER = ["pastoral", "agropastoral", "crop_farming"]


def scatter_plot(merged, out_dir, scale_label):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for lhz in LHZ_ORDER:
        sub = merged[merged["dominant_livelihood_zone"] == lhz]
        ax.scatter(sub["completeness_mean"], sub["f1_weighted"], label=f"{lhz} (n={len(sub)})",
                   color=LHZ_COLORS[lhz], s=55, alpha=0.85, edgecolor="white", linewidth=0.5)
    valid = merged.dropna(subset=["completeness_mean", "f1_weighted"])
    if len(valid) >= 2:
        coeffs = np.polyfit(valid["completeness_mean"], valid["f1_weighted"], 1)
        xs = np.linspace(valid["completeness_mean"].min(), valid["completeness_mean"].max(), 50)
        ax.plot(xs, np.polyval(coeffs, xs), color="black", linestyle="--", linewidth=1.2,
                label="linear trend (visual guide only)")
    ax.set_xlabel("Data completeness score (zone-level, mean across features)")
    ax.set_ylabel(f"Weighted F1 ({scale_label}, pooled test set, best ensemble)")
    ax.set_title(f"Data completeness vs. predictive performance, by zone\n"
                 f"2020-2024 window, {scale_label} scale")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "scatter_completeness_vs_f1.png", dpi=150)
    plt.close(fig)


def run_for_scale(zone_completeness, lhz_completeness, panel, discretize, scale_label, out_dir):
    out_dir.mkdir(exist_ok=True, parents=True)

    pred_df = rm.load_predictions(predictions_path=EXTENDED_PREDICTIONS_PATH, discretize=discretize)
    zone_f1 = rm.score_by_zone(pred_df)
    lhz_f1 = rm.score_by_lhz(pred_df)
    zone_f1.to_csv(out_dir / "f1_by_zone.csv", index=False)
    lhz_f1.to_csv(out_dir / "f1_by_lhz.csv", index=False)

    merged_zone = zone_completeness.merge(zone_f1, on="zone_code", how="inner", validate="one_to_one")
    merged_lhz = lhz_completeness.merge(lhz_f1, on="lhz", how="inner", validate="one_to_one")
    merged_zone.to_csv(out_dir / "rq3_results_zone.csv", index=False)
    merged_lhz.to_csv(out_dir / "rq3_results_lhz.csv", index=False)

    scatter_plot(merged_zone, out_dir, scale_label)

    valid_zone = merged_zone[~merged_zone["excluded"]]
    zone_excluded = int(merged_zone["excluded"].sum())

    zone_result = spearman_with_bootstrap_ci(valid_zone["completeness_mean"], valid_zone["f1_weighted"])
    lhz_result = spearman_with_bootstrap_ci(merged_lhz["completeness_mean"], merged_lhz["f1_weighted"])
    sensitivity_result = spearman_with_bootstrap_ci(valid_zone["completeness_min_cluster"], valid_zone["f1_weighted"])

    n_trivial = int(valid_zone["trivial_constant_target"].sum())
    non_trivial = valid_zone[~valid_zone["trivial_constant_target"]]
    robustness_result = spearman_with_bootstrap_ci(non_trivial["completeness_mean"], non_trivial["f1_weighted"])

    conflict = panel.groupby("zone_code")["acled_event_count"].mean().rename("mean_acled_event_count")
    severity = panel.groupby("zone_code")["ipc_continuous"].mean().rename("mean_ipc_continuous")
    confound_df = merged_zone.merge(conflict, on="zone_code").merge(severity, on="zone_code")
    confound_results = {}
    for col, label in [("mean_acled_event_count", "conflict intensity (mean ACLED events/month)"),
                        ("mean_ipc_continuous", "typical food-insecurity severity (mean ipc_continuous)")]:
        rho, p = spearmanr(confound_df["completeness_mean"], confound_df[col])
        confound_results[label] = {"rho": rho, "p_value": p}

    print(f"\n=== {scale_label}: zone-level ===  {zone_result}")
    print(f"=== {scale_label}: lhz-level (n=3, underpowered) ===  {lhz_result}")
    print(f"=== {scale_label}: sensitivity (min-of-cluster-means) ===  {sensitivity_result}")
    print(f"=== {scale_label}: robustness (constant-target excluded, n_excl={n_trivial}) ===  {robustness_result}")

    with open(out_dir / "correlation_results.json", "w") as f:
        json.dump({
            "scale": scale_label, "zone_level": zone_result, "lhz_level": lhz_result,
            "sensitivity_min_cluster": sensitivity_result,
            "robustness_excl_constant_target": robustness_result,
            "n_constant_target_zones": n_trivial, "n_zones_excluded": zone_excluded,
            "confounds": confound_results,
        }, f, indent=2, default=float)

    lines = [
        f"# RQ3 (2020-2024 window, {scale_label} scale) -- Data Completeness vs. Predictive Performance\n",
        "Rebuilt 2026-09-02 on the project's revised primary test window "
        "(`docs/dissertation_plan.md` section 6). Same ensemble "
        "(XGBoost+RandomForest+LSTM+TabICLv2), scored on 2020-01..2024-12 "
        "instead of the original's 2020-01..2022-12.\n",
        f"**Zone level (n={zone_result['n']}):** rho={zone_result['rho']:.4f}, "
        f"p={zone_result['p_value']:.4f}, 95% CI=[{zone_result['ci_low']:.4f}, "
        f"{zone_result['ci_high']:.4f}]. H0 "
        f"{'REJECTED' if zone_result['reject_h0_at_0.05'] else 'NOT REJECTED'} at p<0.05.\n",
        f"**Livelihood-zone level (n=3, descriptive only):** rho={lhz_result['rho']:.4f}, "
        f"p={lhz_result['p_value']:.4f}.\n",
        f"**Sensitivity check (min-of-cluster-means):** rho={sensitivity_result['rho']:.4f}, "
        f"p={sensitivity_result['p_value']:.4f}. H0 "
        f"{'REJECTED' if sensitivity_result['reject_h0_at_0.05'] else 'NOT REJECTED'} at p<0.05 -- "
        f"{'same direction as primary' if (sensitivity_result['rho'] > 0) == (zone_result['rho'] > 0) else 'OPPOSITE direction from primary -- flag this'}.\n",
        f"**Robustness check ({n_trivial} constant-target zones excluded):** "
        f"rho={robustness_result['rho']:.4f}, p={robustness_result['p_value']:.4f}. H0 "
        f"{'REJECTED' if robustness_result['reject_h0_at_0.05'] else 'NOT REJECTED'} at p<0.05.\n",
        "**Confound checks:**",
    ]
    for label, res in confound_results.items():
        lines.append(f"- Completeness vs. {label}: rho={res['rho']:.3f}, p={res['p_value']:.4f}")
    lines.append("")
    (out_dir / "rq3_summary.md").write_text("\n".join(lines))

    return {"zone": zone_result, "lhz": lhz_result, "sensitivity": sensitivity_result,
            "robustness": robustness_result, "n_trivial": n_trivial}


def main():
    print("[1] Building completeness scores (window-independent, unchanged from original)")
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

    out_base = Path(__file__).resolve().parent / "outputs"
    results = {}
    for discretize, scale_label, out_dir in [
        (rm.to_ipc_class, "1-5", out_base.parent / "outputs_2020_2024_1to5"),
        (rm.to_ipc_class_3plus, "1/2/3+", out_base.parent / "outputs_2020_2024_3plus"),
    ]:
        print(f"\n[2] Scoring + correlating -- {scale_label}")
        results[scale_label] = run_for_scale(zone_completeness, lhz_completeness, panel,
                                              discretize, scale_label, out_dir)

    print("\n=== Summary ===")
    for scale_label, r in results.items():
        print(f"{scale_label}: zone rho={r['zone']['rho']:.4f} (p={r['zone']['p_value']:.4f}), "
              f"sensitivity rho={r['sensitivity']['rho']:.4f} (p={r['sensitivity']['p_value']:.4f})")


if __name__ == "__main__":
    main()
