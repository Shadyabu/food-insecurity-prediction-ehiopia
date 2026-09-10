"""RQ4 Experiment 1 figure: F1 degradation vs k, top-k vs bottom-k removal,
one small-multiple panel per subject (XGBoost / RandomForest / Ensemble).

Palette/marks follow this project's dataviz convention (categorical slots
1/2 = blue/orange for the two series, fixed order; chart chrome per the
reference palette) -- reads from pooled_degradation_table.csv, written by
run_experiment1.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

SUBJECTS = ["XGBoost", "RandomForest", "Ensemble"]


def main():
    df = pd.read_csv(OUT_DIR / "pooled_degradation_table.csv")

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": SECONDARY_INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
    })

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    fig.suptitle("RQ4 Experiment 1 — F1 degradation from top-k vs bottom-k SHAP feature removal",
                 fontsize=12, color=INK, y=0.99)

    for ax, subject in zip(axes, SUBJECTS):
        sub = df[df["subject"] == subject].sort_values("k_pct")
        ax.plot(sub["k_pct"], sub["top_degradation"], marker="o", markersize=6,
                linewidth=2, color=BLUE, label="Top-k removed")
        ax.plot(sub["k_pct"], sub["bottom_degradation"], marker="o", markersize=6,
                linewidth=2, color=ORANGE, label="Bottom-k removed")
        ax.axhline(0, color=BASELINE, linewidth=1)
        ax.set_title(subject, fontsize=11, color=INK, pad=8)
        ax.set_xlabel("k (% of features removed)", fontsize=9.5)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.set_xticks(sub["k_pct"].unique())

    axes[0].set_ylabel("F1 degradation\n(baseline F1 − masked F1)", fontsize=9.5)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93),
               ncol=2, frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.86])
    out_path = OUT_DIR / "fig_degradation_by_k.png"
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
