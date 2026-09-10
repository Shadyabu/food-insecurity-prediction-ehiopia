"""RQ4 Experiment 2 figure: distribution of pairwise Spearman rho (SHAP
ranking stability across 100 bootstrap resamples), one violin per subject
(XGBoost / RandomForest / Ensemble), plus the H0=0.7 reference line.

Reads pairwise_rho_{subject}.npy, written by run_experiment2.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = Path(__file__).resolve().parent

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"
RED = "#e34948"

SUBJECTS = ["XGBoost", "RandomForest", "Ensemble"]
COLORS = [BLUE, ORANGE, AQUA]


def main():
    pairwise = {s: np.load(OUT_DIR / f"pairwise_rho_{s.lower()}.npy") for s in SUBJECTS}

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

    fig, (ax, ax_ctx) = plt.subplots(
        1, 2, figsize=(10.5, 5.2), gridspec_kw={"width_ratios": [3.2, 1]}
    )

    data = [pairwise[s] for s in SUBJECTS]
    y_min = min(d.min() for d in data)
    parts = ax.violinplot(data, showmeans=True, showextrema=True, widths=0.7)

    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(COLORS[i])
        body.set_edgecolor(COLORS[i])
        body.set_alpha(0.35)
    for key in ["cbars", "cmeans", "cmaxes", "cmins"]:
        parts[key].set_edgecolor(SECONDARY_INK)
        parts[key].set_linewidth(1.2)

    ax.set_xticks(range(1, len(SUBJECTS) + 1))
    ax.set_xticklabels(SUBJECTS, fontsize=10.5)
    ax.set_ylabel("Pairwise Spearman ρ across 100 bootstrap SHAP rankings", fontsize=10)
    ax.set_title("Zoomed to observed range", fontsize=10.5, color=SECONDARY_INK, pad=10)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0.4, 3.6)
    ax.set_ylim(y_min - 0.0004, 1.0003)

    # Context panel: full 0.7-1.0 axis, showing how far above the H0
    # threshold every value sits (the zoomed panel alone would make this
    # look like a coin-flip-scale plot when it is in fact a near-ceiling
    # effect -- both panels are needed to read this result honestly).
    for i, s in enumerate(SUBJECTS):
        ax_ctx.scatter([i + 1] * len(pairwise[s]), pairwise[s], color=COLORS[i],
                        alpha=0.15, s=8, zorder=2)
    ax_ctx.axhline(0.7, color=RED, linewidth=1.5, linestyle="--", zorder=1)
    ax_ctx.text(0.55, 0.685, "H0 threshold (ρ = 0.7)", color=RED, fontsize=8.5, va="top", ha="left")
    ax_ctx.set_xticks(range(1, len(SUBJECTS) + 1))
    ax_ctx.set_xticklabels(SUBJECTS, fontsize=8.5, rotation=20, ha="right")
    ax_ctx.set_ylim(0.68, 1.02)
    ax_ctx.set_xlim(0.4, 3.6)
    ax_ctx.set_title("Full 0.7–1.0 context", fontsize=10.5, color=SECONDARY_INK, pad=10)
    ax_ctx.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax_ctx.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax_ctx.spines[spine].set_visible(False)

    fig.suptitle("RQ4 Experiment 2 — SHAP ranking stability under test-set bootstrap resampling",
                 fontsize=12, color=INK, y=1.01)

    fig.tight_layout()
    out_path = OUT_DIR / "fig_pairwise_rho_distribution.png"
    fig.savefig(out_path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
