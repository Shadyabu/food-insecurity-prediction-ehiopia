"""Fills gaps found in the 2026-08-26 figure audit against the
dissertation plan's "RQ1 Reproduction" figure list:
  - overall (single pooled value) MAE & R2 -- did not exist
  - MAE & R2 by lead time, all zones COMBINED (pooled) -- fig_4_1_r2_by_leadtime()
    in build_figures.py only had the split-by-zone version, R2 only
  - MAE by lead time AND livelihood zone -- had no equivalent to the
    existing R2-by-lead-and-zone panel at all

Busker et al. only publish ONE MAE reference point (0.35, pooled over all
213 units at 3-month lead -- see RQ1_Experiment1_Busker_Reproduction.md
section 2.1/2.2 and report.md section 2.2) -- no per-zone or per-lead MAE
grid the way R2 has (their own Figure 5). So the MAE figures below plot
that single number as a flat dashed reference line, not a per-cell
reference series -- there is nothing else to plot. R2's pooled headline
reference (0.72 at lead 1, whole region) is likewise a single point, not
a full by-lead series.
"""

import matplotlib.pyplot as plt
import pandas as pd

METRICS_DIR = "metrics"
FIG_DIR = "figures"
LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTER_LABELS = {"p": "Pastoral", "ap": "Agro-pastoral", "other": "Crop farming"}
CLUSTERS = ["p", "ap", "other"]

COLOR_XGB = "#1b7a3d"
REF_MAE_POOLED = 0.35   # 3-month lead, all 213 units (paper's only MAE reference)
REF_R2_LEAD1 = 0.72     # 1-month lead, whole region (paper's headline R2)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def fig_overall():
    per_lead = pd.read_csv(f"{METRICS_DIR}/metrics_per_lead.csv")
    overall_mae = per_lead["mae"].mean()
    overall_r2 = per_lead["r2"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    axes[0].bar(["Reproduced\n(mean over leads)"], [overall_mae], color=COLOR_XGB)
    axes[0].axhline(REF_MAE_POOLED, color="grey", ls="--", lw=1.5,
                     label=f"Busker et al. 2024 (MAE={REF_MAE_POOLED}, lead 3 only)")
    axes[0].text(0, overall_mae + 0.005, f"{overall_mae:.3f}", ha="center")
    axes[0].set_ylabel("MAE (lower is better)")
    axes[0].set_title("Overall MAE")
    axes[0].legend(fontsize=7.5, loc="upper right")
    axes[0].set_ylim(0, max(0.5, overall_mae * 1.3))

    axes[1].bar(["Reproduced\n(mean over leads)"], [overall_r2], color=COLOR_XGB)
    axes[1].axhline(REF_R2_LEAD1, color="grey", ls="--", lw=1.5,
                     label=f"Busker et al. 2024 (R2={REF_R2_LEAD1}, lead 1 only)")
    axes[1].text(0, overall_r2 + 0.02, f"{overall_r2:.3f}", ha="center")
    axes[1].set_ylabel("R$^2$")
    axes[1].set_title("Overall R$^2$")
    axes[1].legend(fontsize=7.5, loc="upper right")
    axes[1].set_ylim(0, 1.0)

    fig.suptitle("RQ1 Experiment 1 -- overall pooled MAE / R$^2$ (all 213 units, all leads)\n"
                  "reference lines are single published points, not a matching pooled figure", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq1_overall_mae_r2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_overall_mae_r2.png")


def fig_combined_by_lead():
    per_lead = pd.read_csv(f"{METRICS_DIR}/metrics_per_lead.csv").set_index("lead")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].plot(LEADS, [per_lead.loc[l, "r2"] for l in LEADS], "-o", color=COLOR_XGB, lw=2, ms=6,
                 label="Reproduced (all zones pooled)")
    axes[0].scatter([1], [REF_R2_LEAD1], color="grey", marker="x", s=80, zorder=3,
                     label=f"Busker et al. 2024 (R2={REF_R2_LEAD1} @ lead 1)")
    axes[0].set_xlabel("lead time (months)")
    axes[0].set_ylabel("R$^2$")
    axes[0].set_xticks(LEADS)
    axes[0].axhline(0, color="grey", lw=0.5)
    axes[0].legend(fontsize=8)
    axes[0].set_title("R$^2$ by lead time, all zones combined")

    axes[1].plot(LEADS, [per_lead.loc[l, "mae"] for l in LEADS], "-o", color="#a33", lw=2, ms=6,
                 label="Reproduced (all zones pooled)")
    axes[1].axhline(REF_MAE_POOLED, color="grey", ls="--", lw=1.5,
                     label=f"Busker et al. 2024 (MAE={REF_MAE_POOLED}, lead 3 only)")
    axes[1].set_xlabel("lead time (months)")
    axes[1].set_ylabel("MAE (lower is better)")
    axes[1].set_xticks(LEADS)
    axes[1].legend(fontsize=8)
    axes[1].set_title("MAE by lead time, all zones combined")

    fig.suptitle("RQ1 Experiment 1 -- MAE & R$^2$ by lead time, all 213 units pooled", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq1_mae_r2_by_leadtime_combined.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_mae_r2_by_leadtime_combined.png")


def fig_mae_by_lead_and_zone():
    cluster = pd.read_csv(f"{METRICS_DIR}/metrics_per_cluster.csv").set_index(["lhz", "lead"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, lhz in zip(axes, CLUSTERS):
        mae = [cluster.loc[(lhz, l), "mae"] for l in LEADS]
        ax.plot(LEADS, mae, "-o", color="#a33", lw=2.2, ms=5, label="Reproduced")
        ax.axhline(REF_MAE_POOLED, color="grey", ls="--", lw=1.4,
                    label=f"Busker et al. 2024\n(single pooled ref, MAE={REF_MAE_POOLED} @ lead 3)")
        ax.set_title(CLUSTER_LABELS[lhz])
        ax.set_xlabel("lead time (months)")
        ax.set_xticks(LEADS)
    axes[0].set_ylabel("MAE (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1), frameon=False, fontsize=9)
    fig.suptitle("RQ1 Experiment 1 -- MAE by lead time and livelihood zone\n"
                  "(Busker et al. publish only a single pooled MAE reference, not a per-zone/per-lead grid)", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq1_mae_by_leadtime_zone.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_mae_by_leadtime_zone.png")


if __name__ == "__main__":
    fig_overall()
    fig_combined_by_lead()
    fig_mae_by_lead_and_zone()
