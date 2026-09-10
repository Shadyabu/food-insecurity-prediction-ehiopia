"""
build_ablation_comparison_figures.py

Two presentation-ready figures for RQ1 Experiment 2's domain
feature-cluster ablation: which data source clusters matter for
predictive power. Reads the CSVs written by
build_ablation_comparison_tables.py -- run that first.

Colors: dataviz skill's validated default categorical palette, slots 1-5
(blue/orange/aqua/yellow/magenta) -- passes all adjacent-pair checks for
a 5-series bar/line comparison (this is a bar/line context, not an
all-pairs one, so the 3-series cap in palette.md doesn't apply here).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "ablation_comparison")

LEADS = [0, 1, 2, 3, 4, 8, 12]
LHZ_ORDER = ["pastoral", "agropastoral", "crop_farming"]
LHZ_LABELS = {"pastoral": "Pastoral", "agropastoral": "Agro-pastoral", "crop_farming": "Crop farming"}

VARIANT_ORDER = ["baseline", "no_climate", "no_agriculture", "no_economic", "no_conflict"]
VARIANT_LABELS = {
    "baseline": "All clusters (baseline)",
    "no_climate": "No climate",
    "no_agriculture": "No agriculture",
    "no_economic": "No economic",
    "no_conflict": "No conflict (ACLED+UNHCR)",
}
VARIANT_COLORS = {
    "baseline": "#2a78d6",       # blue
    "no_climate": "#eb6834",     # orange
    "no_agriculture": "#1baf7a", # aqua
    "no_economic": "#eda100",    # yellow
    "no_conflict": "#e87ba4",    # magenta
}
VARIANT_MARKERS = {"baseline": "o", "no_climate": "s", "no_agriculture": "^", "no_economic": "D", "no_conflict": "v"}

BUSKER_KEY = "busker_arch"
BUSKER_LABEL = "Busker architecture\n(reference)"
BUSKER_COLOR = "#52514e"
BUSKER_MARKER = "x"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def fig_overall_absolute_bar():
    """Absolute overall weighted F1 for all 5 models (baseline + 4
    ablations) -- did not exist before the 2026-08-26 figure audit
    (only the delta-vs-baseline version, fig_a_delta_bar, existed)."""
    overall = pd.read_csv(os.path.join(DATA_DIR, "overall.csv"))
    plot_rows = overall[overall["variant"] != BUSKER_KEY].set_index("variant").reindex(VARIANT_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = [VARIANT_COLORS[v] for v in plot_rows["variant"]]
    bars = ax.bar(range(len(plot_rows)), plot_rows["mean_f1_weighted"], color=colors)
    for bar, val in zip(bars, plot_rows["mean_f1_weighted"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(plot_rows)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in plot_rows["variant"]], fontsize=8.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("weighted F1 (pooled, all zones, all leads)")
    ax.set_title("RQ1 Experiment 2 -- overall weighted F1, full model + each cluster ablated")
    fig.tight_layout()
    out_path = os.path.join(DATA_DIR, "ablation_overall_f1.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_a_delta_bar():
    overall = pd.read_csv(os.path.join(DATA_DIR, "overall.csv"))
    plot_rows = overall[~overall["variant"].isin(["baseline", BUSKER_KEY])].sort_values("delta_f1_vs_baseline")

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    colors = [VARIANT_COLORS[v] for v in plot_rows["variant"]]
    bars = ax.barh(plot_rows["label"], plot_rows["delta_f1_vs_baseline"], color=colors)
    ax.axvline(0, color="#52514e", linewidth=1)
    ax.set_xlabel("change in pooled weighted F1 vs. baseline (all clusters in)")
    ax.set_title("RQ1 Experiment 2 -- which feature cluster matters?\nΔF1 when that cluster is removed (Ethiopia, all zones, all leads pooled)")
    vmin, vmax = plot_rows["delta_f1_vs_baseline"].min(), plot_rows["delta_f1_vs_baseline"].max()
    span = max(vmax - vmin, 0.01)
    ax.set_xlim(vmin - 0.28 * span, vmax + 0.28 * span)
    for bar, val in zip(bars, plot_rows["delta_f1_vs_baseline"]):
        offset = 0.02 * span if val >= 0 else -0.02 * span
        ha = "left" if val >= 0 else "right"
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2, f"{val:+.4f}", va="center", ha=ha, fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(DATA_DIR, "ablation_delta_f1.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def fig_b_by_lead():
    by_lead = pd.read_csv(os.path.join(DATA_DIR, "by_lead.csv"))
    pastoral_by_lead = pd.read_csv(os.path.join(DATA_DIR, "pastoral_by_lead.csv"))

    fig, (ax_all, ax_past) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for variant in VARIANT_ORDER:
        sub = by_lead[by_lead["variant"] == variant].sort_values("lead")
        ax_all.plot(sub["lead"], sub["f1_weighted"], marker=VARIANT_MARKERS[variant],
                    color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant], linewidth=2, markersize=7)
        sub_p = pastoral_by_lead[pastoral_by_lead["variant"] == variant].sort_values("lead")
        ax_past.plot(sub_p["lead"], sub_p["f1_weighted"], marker=VARIANT_MARKERS[variant],
                     color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant], linewidth=2, markersize=7)

    ax_all.set_title("All zones")
    ax_past.set_title("Pastoral only")
    for ax in (ax_all, ax_past):
        ax.set_xlabel("lead time (months)")
        ax.set_xticks(LEADS)
    ax_all.set_ylabel("weighted F1")

    handles, labels = ax_all.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.suptitle("RQ1 Experiment 2 -- weighted F1 by lead time, all 5 feature-cluster variants", fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.94])

    out_path = os.path.join(DATA_DIR, "ablation_by_lead.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def fig_c_by_lead_and_lhz():
    """Weighted F1 by lead time, split by all 3 livelihood zones (not just
    pastoral), WITH the Busker-reproduced-baseline reference overlaid --
    the dissertation plan's third required figure for this experiment;
    did not exist before the 2026-08-26 figure audit (neither the 3-way
    zone split nor the Busker comparison existed anywhere in this
    experiment's figures)."""
    by_lhz = pd.read_csv(os.path.join(DATA_DIR, "by_lead_and_lhz.csv"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, lhz in zip(axes, LHZ_ORDER):
        sub_lhz = by_lhz[by_lhz["lhz"] == lhz]
        for variant in VARIANT_ORDER:
            sub = sub_lhz[sub_lhz["variant"] == variant].sort_values("lead")
            ax.plot(sub["lead"], sub["f1_weighted"], marker=VARIANT_MARKERS[variant],
                     color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant], linewidth=1.8, markersize=6)
        busker = sub_lhz[sub_lhz["variant"] == BUSKER_KEY].sort_values("lead")
        ax.plot(busker["lead"], busker["f1_weighted"], marker=BUSKER_MARKER, color=BUSKER_COLOR,
                 linestyle="--", linewidth=1.8, markersize=7, label=BUSKER_LABEL.replace("\n", " "))
        ax.set_title(LHZ_LABELS[lhz])
        ax.set_xlabel("lead time (months)")
        ax.set_xticks(LEADS)
    axes[0].set_ylabel("weighted F1")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14), frameon=False, fontsize=8.5)
    fig.suptitle("RQ1 Experiment 2 -- weighted F1 by lead time and livelihood zone\n"
                 "all 5 feature-cluster variants vs. Busker-reproduced baseline "
                 "(reference test data ends 2022-06)", fontsize=11)
    fig.tight_layout(rect=[0, 0.14, 1, 0.92])

    out_path = os.path.join(DATA_DIR, "ablation_by_lead_lhz_vs_busker.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    fig_overall_absolute_bar()
    fig_a_delta_bar()
    fig_b_by_lead()
    fig_c_by_lead_and_lhz()
