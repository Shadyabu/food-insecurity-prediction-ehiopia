"""±2 SD variants of the reference-vs-reproduction R2 scatter (alongside
the ±1 SD versions in build_figures.py: fig_4_4 robust MAD-SD, fig_4_4b
bootstrap-SE). Requested by the project owner after reviewing the
existing 1-SD figures -- same two SD definitions, same plot style, just
a wider (2x) acceptance band. Reuses the SD/abs_diff columns already
computed by build_acceptance_tables.py / bootstrap_acceptance.py --
no change to either acceptance-table script or the official 1-SD
criterion they produce.
"""

import matplotlib.pyplot as plt
import pandas as pd

METRICS_DIR = "metrics"
FIG_DIR = "figures"


def _scatter(acc, yerr_col, within_mask, yerr_label, out_name, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = -0.1, 0.85
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="-", zorder=1, label="1:1 line")

    ax.errorbar(acc["reference_r2"], acc["reproduced_r2"],
                yerr=2 * acc[yerr_col], fmt="none",
                ecolor="grey", elinewidth=1, capsize=3, alpha=0.5, zorder=2,
                label=yerr_label)

    ax.scatter(acc.loc[within_mask, "reference_r2"], acc.loc[within_mask, "reproduced_r2"],
               color="#1b7a3d", s=55, zorder=3, label=f"within 2 SD (n={within_mask.sum()})")
    ax.scatter(acc.loc[~within_mask, "reference_r2"], acc.loc[~within_mask, "reproduced_r2"],
               color="#c81414", s=55, zorder=3, label=f"outside 2 SD (n={(~within_mask).sum()})")

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
    ax.set_title(title)
    fig.tight_layout()
    out_path = f"{FIG_DIR}/{out_name}"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def fig_robust_2sd():
    acc = pd.read_csv(f"{METRICS_DIR}/statistical_acceptance_r2.csv")
    within = acc["abs_diff"] <= 2 * acc["per_unit_r2_robust_sd_mad"]
    _scatter(
        acc, "per_unit_r2_robust_sd_mad", within,
        "±2 SD (robust, that cell's own per-unit spread)",
        "rq1_reference_vs_reproduction_scatter_2sd.png",
        "RQ1 Experiment 1 -- reference vs. reproduction R$^2$ (±2 SD, robust)\n"
        "all 21 livelihood-zone x lead-time cells",
    )


def fig_bootstrap_2sd():
    acc = pd.read_csv(f"{METRICS_DIR}/statistical_acceptance_r2_bootstrap.csv")
    within = acc["abs_diff"] <= 2 * acc["bootstrap_r2_sd"]
    _scatter(
        acc, "bootstrap_r2_sd", within,
        "±2 bootstrap SE of the pooled R$^2$ (500 unit resamples)",
        "rq1_reference_vs_reproduction_scatter_bootstrap_2sd.png",
        "RQ1 Experiment 1 -- reference vs. reproduction R$^2$ (±2 SD, bootstrap SE)\n"
        "all 21 livelihood-zone x lead-time cells",
    )


if __name__ == "__main__":
    fig_robust_2sd()
    fig_bootstrap_2sd()
