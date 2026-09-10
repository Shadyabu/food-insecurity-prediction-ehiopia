"""Figures for the FEWS NET outlook (ML1/ML2) reproduction
(build_fews_outlook_reproduction.py). Matches build_figures.py's established
RQ1 Experiment 1 style (3-subplot-per-cluster layout, solid = reproduced,
dashed = paper reference where one exists).
"""

import matplotlib.pyplot as plt
import pandas as pd

RESULT_PATH = "fews_outlook_f1_2020_2024.csv"
FIG_DIR = "figures"

CLUSTER_ORDER = ["pastoral", "agropastoral", "crop_farming"]
CLUSTER_LABELS = {"pastoral": "Pastoral", "agropastoral": "Agro-pastoral", "crop_farming": "Crop farming"}
CLUSTER_TO_LHZ = {"pastoral": "p", "agropastoral": "ap", "crop_farming": "other"}  # for the paper-ref lookup below

COLOR_FEWS = "#e69f1b"
COLOR_FEWS_ALT = "#8a5a00"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

# Transcribed from Busker et al. (2024) Figure 5 -- see build_figures.py.
# NOTE: this uses the PAPER's own lead numbering (0,1,3,4,8 -- pooled across
# Kenya/Somalia/Ethiopia, 2019-mid test window), which is NOT the same lead
# definition as this reproduction's lead_end (0-3/4-7, Ethiopia only,
# 2020-2024 window). Plotted on the same axis for visual reference only --
# not a like-for-like overlay, and the figure caption says so explicitly.
REF_FEWS_R2 = {
    "p":     {0: 0.57, 1: 0.57, 3: 0.08, 4: 0.08, 8: 0.18},
    "ap":    {0: 0.50, 1: 0.50, 3: 0.35, 4: 0.0},
    "other": {0: 0.75, 1: 0.75, 3: 0.70, 4: 0.60, 8: 0.54},
}


def _series(d, leads):
    return [d.get(lead, float("nan")) for lead in leads]


def fig_f1_by_leadtime(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, cluster in zip(axes, CLUSTER_ORDER):
        sub = df[df["cluster"] == cluster].sort_values("lead_end")
        ax.plot(sub["lead_end"], sub["weighted_f1"], "-o", color=COLOR_FEWS, lw=2.2, ms=5,
                label="Weighted F1 (1-5 phase)")
        ax.plot(sub["lead_end"], sub["weighted_f1_3plus"], "--o", color=COLOR_FEWS_ALT, lw=1.8,
                ms=4, alpha=0.8, label="Weighted F1 (1/2/3+ collapsed)")
        ax.axvline(3.5, color="grey", lw=0.6, ls=":")
        ax.text(1.5, 0.03, "ML1", ha="center", fontsize=8, color="grey", transform=ax.get_xaxis_transform())
        ax.text(5.5, 0.03, "ML2", ha="center", fontsize=8, color="grey", transform=ax.get_xaxis_transform())
        ax.set_title(f"{CLUSTER_LABELS[cluster]} ({int(sub['n_zones'].iloc[0])} zones)")
        ax.set_xlabel("lead time (months, lead_end)")
        ax.set_xticks(range(8))
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Weighted F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.08),
               frameon=False, fontsize=9)
    fig.suptitle("FEWS NET outlook (ML1/ML2) -- weighted F1 by lead time and livelihood zone\n"
                 "Ethiopia, 92 zones, 2020-2024 (independently reproduced, not from the paper)", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fews_outlook_f1_by_leadtime.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/fews_outlook_f1_by_leadtime.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote fews_outlook_f1_by_leadtime.{png,svg}")


def fig_r2_by_leadtime(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, cluster in zip(axes, CLUSTER_ORDER):
        sub = df[df["cluster"] == cluster].sort_values("lead_end")
        lhz = CLUSTER_TO_LHZ[cluster]
        ax.plot(sub["lead_end"], sub["r2"], "-o", color=COLOR_FEWS, lw=2.2, ms=5,
                label="R$^2$ (this reproduction, Ethiopia, 2020-2024, lead_end)")
        ref_leads = sorted(REF_FEWS_R2[lhz])
        ax.plot(ref_leads, _series(REF_FEWS_R2[lhz], ref_leads), "--x", color="grey", lw=1.4,
                ms=6, alpha=0.7, label="R$^2$ (Busker et al. 2024 Fig. 5, transcribed --\n"
                                        "3-country pool, own lead scheme, NOT directly comparable)")
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_title(CLUSTER_LABELS[cluster])
        ax.set_xlabel("lead time (months)")
        ax.set_ylim(-0.1, 0.9)
    axes[0].set_ylabel("R$^2$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.16),
               frameon=False, fontsize=8)
    fig.suptitle("FEWS NET outlook -- R$^2$ by lead time: reproduction vs. paper reference\n"
                 "(sanity check only -- different geography, window, and lead definition)", y=1.05)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/fews_outlook_r2_by_leadtime.png", dpi=200, bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/fews_outlook_r2_by_leadtime.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote fews_outlook_r2_by_leadtime.{png,svg}")


if __name__ == "__main__":
    df = pd.read_csv(RESULT_PATH)
    df = df[df["cluster"] != "all_zones"]
    fig_f1_by_leadtime(df)
    fig_r2_by_leadtime(df)
