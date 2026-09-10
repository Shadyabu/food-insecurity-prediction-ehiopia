"""
build_experiment1_comparison_figures.py

Two presentation-ready figures for RQ1 Experiment 1's Ethiopia-only
comparison (report.md/SESSION_SUMMARY.md): same XGBoost architecture,
same (overlap-restricted) test period, weighted F1 and MAE, broken out
by lead time and by livelihood zone. Reads the CSVs written by
build_experiment1_comparison_tables.py -- run that first.

Colors: dataviz skill's validated default categorical palette, slots 1-3
(blue/orange/aqua) -- passes all checks including the --pairs all gate,
so safe for a 3-series line+bar comparison. Reference-line dark gray
(matching experiment_1/build_figures.py's own COLOR convention) is NOT
used here since all three series are equal-status experimental results,
not one fixed external reference.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "experiment_1_vs_2_comparison")

LEADS = [0, 1, 2, 3, 4, 8, 12]
LHZ_ORDER = ["pastoral", "agropastoral", "crop_farming"]
LHZ_LABELS = {"pastoral": "Pastoral", "agropastoral": "Agro-pastoral", "crop_farming": "Crop farming"}

CONFIG_ORDER = ["busker_arch", "enriched_level", "enriched_best"]
CONFIG_LABELS = {
    "busker_arch": "Busker architecture (baseline)",
    "enriched_level": "Ethiopia-enriched, same target",
    "enriched_best": "Ethiopia-enriched, best variant",
}
CONFIG_COLORS = {
    "busker_arch": "#eb6834",       # orange -- the thing being compared against
    "enriched_level": "#2a78d6",    # blue
    "enriched_best": "#1baf7a",     # aqua -- headline result
}
CONFIG_MARKERS = {"busker_arch": "o", "enriched_level": "s", "enriched_best": "^"}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def fig_overall(overall):
    """Overall (single pooled value) MAE/R2/F1 bars, all 3 configs -- did
    not exist before the 2026-08-26 figure audit (by_lead/by_lhz panels
    existed, but no single-value summary)."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    for ax, metric, ylabel, title in [
        (axes[0], "mean_f1_weighted", "weighted F1", "Overall weighted F1"),
        (axes[1], "mean_mae", "MAE (lower is better)", "Overall MAE"),
        (axes[2], "mean_r2", "R$^2$", "Overall R$^2$"),
    ]:
        vals = [overall[overall["config"] == cfg][metric].iloc[0] for cfg in CONFIG_ORDER]
        colors = [CONFIG_COLORS[cfg] for cfg in CONFIG_ORDER]
        bars = ax.bar(range(len(CONFIG_ORDER)), vals, color=colors)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(len(CONFIG_ORDER)))
        ax.set_xticklabels([CONFIG_LABELS[cfg].replace(", ", "\n") for cfg in CONFIG_ORDER], fontsize=7.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
    fig.suptitle("RQ1 Experiment 1 -- overall pooled MAE / R$^2$ / weighted F1\n"
                  "Ethiopia only, train<=2019-12, test 2020-01..2022-06/10 (see caption)", fontsize=10)
    fig.tight_layout()
    out_path = os.path.join(DATA_DIR, "experiment1_ethiopia_overall.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    overall = pd.read_csv(os.path.join(DATA_DIR, "overall.csv"))
    by_lead = pd.read_csv(os.path.join(DATA_DIR, "by_lead.csv"))
    by_lhz = pd.read_csv(os.path.join(DATA_DIR, "by_lhz.csv"))

    fig_overall(overall)

    fig, axes = plt.subplots(3, 2, figsize=(13, 13))
    ax_f1_lead, ax_f1_lhz = axes[0, 0], axes[0, 1]
    ax_mae_lead, ax_mae_lhz = axes[1, 0], axes[1, 1]
    ax_r2_lead, ax_r2_lhz = axes[2, 0], axes[2, 1]

    # --- top-left: F1 by lead ---
    for cfg in CONFIG_ORDER:
        sub = by_lead[by_lead["config"] == cfg].sort_values("lead")
        ax_f1_lead.plot(sub["lead"], sub["f1_weighted"], marker=CONFIG_MARKERS[cfg],
                         color=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg], linewidth=2, markersize=8)
    ax_f1_lead.set_title("Weighted F1 by lead time")
    ax_f1_lead.set_xlabel("lead time (months)")
    ax_f1_lead.set_ylabel("weighted F1")
    ax_f1_lead.set_xticks(LEADS)

    # --- top-right: F1 by livelihood zone ---
    x = range(len(LHZ_ORDER))
    width = 0.25
    for i, cfg in enumerate(CONFIG_ORDER):
        sub = by_lhz[by_lhz["config"] == cfg].set_index("lhz").reindex(LHZ_ORDER)
        offsets = [xi + (i - 1) * width for xi in x]
        ax_f1_lhz.bar(offsets, sub["f1_weighted"], width=width, color=CONFIG_COLORS[cfg],
                       label=CONFIG_LABELS[cfg])
    ax_f1_lhz.set_title("Weighted F1 by livelihood zone")
    ax_f1_lhz.set_xticks(list(x))
    ax_f1_lhz.set_xticklabels([LHZ_LABELS[z] for z in LHZ_ORDER])
    ax_f1_lhz.set_ylabel("weighted F1")

    # --- bottom-left: MAE by lead ---
    for cfg in CONFIG_ORDER:
        sub = by_lead[by_lead["config"] == cfg].sort_values("lead")
        ax_mae_lead.plot(sub["lead"], sub["mae"], marker=CONFIG_MARKERS[cfg],
                          color=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg], linewidth=2, markersize=8)
    ax_mae_lead.set_title("MAE by lead time")
    ax_mae_lead.set_xlabel("lead time (months)")
    ax_mae_lead.set_ylabel("MAE (lower is better)")
    ax_mae_lead.set_xticks(LEADS)

    # --- bottom-right: MAE by livelihood zone ---
    for i, cfg in enumerate(CONFIG_ORDER):
        sub = by_lhz[by_lhz["config"] == cfg].set_index("lhz").reindex(LHZ_ORDER)
        offsets = [xi + (i - 1) * width for xi in x]
        ax_mae_lhz.bar(offsets, sub["mae"], width=width, color=CONFIG_COLORS[cfg],
                        label=CONFIG_LABELS[cfg])
    ax_mae_lhz.set_title("MAE by livelihood zone")
    ax_mae_lhz.set_xticks(list(x))
    ax_mae_lhz.set_xticklabels([LHZ_LABELS[z] for z in LHZ_ORDER])
    ax_mae_lhz.set_ylabel("MAE (lower is better)")

    # --- R2 by lead ---
    for cfg in CONFIG_ORDER:
        sub = by_lead[by_lead["config"] == cfg].sort_values("lead")
        ax_r2_lead.plot(sub["lead"], sub["r2"], marker=CONFIG_MARKERS[cfg],
                         color=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg], linewidth=2, markersize=8)
    ax_r2_lead.axhline(0, color="grey", lw=0.5)
    ax_r2_lead.set_title("R$^2$ by lead time")
    ax_r2_lead.set_xlabel("lead time (months)")
    ax_r2_lead.set_ylabel("R$^2$")
    ax_r2_lead.set_xticks(LEADS)

    # --- R2 by livelihood zone ---
    for i, cfg in enumerate(CONFIG_ORDER):
        sub = by_lhz[by_lhz["config"] == cfg].set_index("lhz").reindex(LHZ_ORDER)
        offsets = [xi + (i - 1) * width for xi in x]
        ax_r2_lhz.bar(offsets, sub["r2"], width=width, color=CONFIG_COLORS[cfg], label=CONFIG_LABELS[cfg])
    ax_r2_lhz.axhline(0, color="grey", lw=0.5)
    ax_r2_lhz.set_title("R$^2$ by livelihood zone")
    ax_r2_lhz.set_xticks(list(x))
    ax_r2_lhz.set_xticklabels([LHZ_LABELS[z] for z in LHZ_ORDER])
    ax_r2_lhz.set_ylabel("R$^2$")

    handles, labels = ax_f1_lead.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(
        "RQ1 Experiment 1 -- Ethiopia only, same architecture, same train<=2019-12/test 2020+ split boundary\n"
        f"test window: {overall['n'].iloc[0]} rows/config; Busker-arch scoring window 2020-02 to 2022-06 "
        "(its own target data ends there), enriched-dataset scoring window 2020-02 to 2022-10, "
        "restricted here to the overlap for apples-to-apples comparison",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])

    out_path = os.path.join(DATA_DIR, "experiment1_ethiopia_comparison.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
