"""Third variant of the reference-vs-reproduction scatter (alongside
fig_4_4's robust-MAD-SD and fig_4_4b's bootstrap-SE versions in
build_figures.py): the +-5% relative-difference criterion itself, plotted
as a shaded band around the 1:1 line rather than left as a pass/fail table
(report.md section 3). Reads the same within_5pct column already computed
by build_acceptance_tables.py -- no new acceptance logic here, just a
visualization of the existing +-5% rule alongside the two SD-based ones.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_DIR = "metrics"
FIG_DIR = "figures"


def main():
    acc = pd.read_csv(f"{METRICS_DIR}/statistical_acceptance_r2.csv")

    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = -0.1, 0.85
    xs = np.linspace(lo, hi, 200)
    upper = xs + 0.05 * np.abs(xs)
    lower = xs - 0.05 * np.abs(xs)
    ax.fill_between(xs, lower, upper, color="#2a78d6", alpha=0.15, zorder=0, label="±5% band")
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="-", zorder=1, label="1:1 line")

    within = acc["within_5pct"].astype(bool)
    ax.scatter(acc.loc[within, "reference_r2"], acc.loc[within, "reproduced_r2"],
               color="#1b7a3d", s=55, zorder=3, label=f"within ±5% (n={within.sum()})")
    ax.scatter(acc.loc[~within, "reference_r2"], acc.loc[~within, "reproduced_r2"],
               color="#c81414", s=55, zorder=3, label=f"outside ±5% (n={(~within).sum()})")

    for _, row in acc.iterrows():
        ax.annotate(f"{row['lhz_code']}{row['lead']}",
                    (row["reference_r2"], row["reproduced_r2"]),
                    fontsize=6.5, alpha=0.7, xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("Reference R$^2$ (Busker et al. 2024, Figure 5)")
    ax.set_ylabel("Reproduced R$^2$ (this study)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)
    ax.set_title("RQ1 Experiment 1 -- reference vs. reproduction R$^2$ (±5% relative-difference rule)\n"
                  "all 21 livelihood-zone x lead-time cells")
    fig.tight_layout()
    out_path = f"{FIG_DIR}/rq1_reference_vs_reproduction_scatter_5pct.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
