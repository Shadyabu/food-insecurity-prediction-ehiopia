"""Build the 4 required comparison figures for RQ1 Experiment 1
(RQ1_Experiment1_Busker_Reproduction.md section 4).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_DIR = "metrics"
FIG_DIR = "figures"
LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTER_LABELS = {"p": "Pastoral", "ap": "Agro-pastoral", "other": "Crop farming"}
CLUSTERS = ["p", "ap", "other"]

COLOR_XGB = "#1b7a3d"
COLOR_PERSIST = "#222222"
COLOR_SEASON = "#c2199b"
COLOR_FEWS = "#e69f1b"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

# --------------------------------------------------------------------------
# Reference data, transcribed from RQ1_Experiment1_Busker_Reproduction.md §2
# --------------------------------------------------------------------------
REF_XGB_R2 = {
    "p":     {0: 0.49, 1: 0.55, 2: 0.44, 3: 0.53, 4: 0.35, 8: 0.28, 12: 0.30},
    "ap":    {0: 0.48, 1: 0.48, 2: 0.48, 3: 0.37, 4: 0.40, 8: 0.23, 12: 0.05},
    "other": {0: 0.54, 1: 0.65, 2: 0.61, 3: 0.57, 4: 0.46, 8: 0.48, 12: 0.40},
}
REF_PERSIST_R2 = {
    "p":     {0: 0.20, 1: 0.20, 3: 0.20, 4: 0.0},
    "ap":    {0: 0.08, 1: 0.08, 3: 0.08, 4: 0.0},
    "other": {0: 0.62, 1: 0.62, 3: 0.62, 4: 0.46, 8: 0.40, 12: 0.25},
}
REF_SEASON_R2 = {"other": 0.46}  # pastoral/agropastoral: negative, off-scale (paper's own Fig 5 note)
REF_FEWS_R2 = {
    "p":     {0: 0.57, 1: 0.57, 3: 0.08, 4: 0.08, 8: 0.18},
    "ap":    {0: 0.50, 1: 0.50, 3: 0.35, 4: 0.0},
    "other": {0: 0.75, 1: 0.75, 3: 0.70, 4: 0.60, 8: 0.54},
}
REF_HR_FAR = {
    ("p", 0): (0.12, 0.10), ("ap", 0): (0.40, 0.04), ("other", 0): (0.0, 0.001),
    ("p", 1): (0.20, 0.08), ("ap", 1): (0.50, 0.04), ("other", 1): (0.0, 0.0),
    ("p", 2): (0.18, 0.08), ("ap", 2): (0.30, 0.06), ("other", 2): (0.0, 0.0),
    ("p", 3): (0.18, 0.07), ("ap", 3): (0.20, 0.08), ("other", 3): (0.0, 0.0),
    ("p", 4): (0.05, 0.06), ("ap", 4): (0.05, 0.05), ("other", 4): (0.0, 0.0),
    ("p", 8): (0.04, 0.07), ("ap", 8): (0.09, 0.08), ("other", 8): (0.0, 0.0),
    ("p", 12): (0.02, 0.02), ("ap", 12): (0.31, 0.14), ("other", 12): (0.0, 0.0),
}


def _series(d, leads=LEADS):
    return [d.get(lead, np.nan) for lead in leads]


def fig_4_1_r2_by_leadtime():
    """XGBoost only -- reproduced vs. Busker et al. (2024) reference.

    Per user request (2026-08-19): excludes the persistence, seasonality,
    and FEWS NET outlook series that were plotted here previously (all
    still independently reproduced/tabulated in metrics/metrics_per_cluster.csv
    and referenced in report.md sections 2.1/2.4 -- this is a display
    simplification, not a change to what was reproduced).
    """
    cluster = pd.read_csv(f"{METRICS_DIR}/metrics_per_cluster.csv")
    cluster = cluster.set_index(["lhz", "lead"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, lhz in zip(axes, CLUSTERS):
        repro_xgb = [cluster.loc[(lhz, l), "r2"] for l in LEADS]

        ax.plot(LEADS, repro_xgb, "-o", color=COLOR_XGB, lw=2.2, ms=4,
                label="XGBoost (reproduced, this study)")
        ax.plot(LEADS, _series(REF_XGB_R2[lhz]), "--", color=COLOR_XGB, lw=1.6,
                alpha=0.6, marker="x", ms=5, label="XGBoost (Busker et al. 2024)")

        ax.set_title(CLUSTER_LABELS[lhz])
        ax.set_xlabel("lead time (months)")
        ax.set_xticks(LEADS)
        ax.axhline(0, color="grey", lw=0.6)
        ax.set_ylim(-0.1, 0.85)
    axes[0].set_ylabel("Coefficient of determination (R$^2$)")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.1),
               frameon=False, fontsize=9)
    fig.suptitle("RQ1 Experiment 1 -- R$^2$ by lead time and livelihood zone: "
                  "reproduction vs. Busker et al. (2024) (XGBoost only)", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq1_r2_by_leadtime_zone_comparison.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(f"{FIG_DIR}/rq1_r2_by_leadtime_zone_comparison.svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_r2_by_leadtime_zone_comparison.{png,svg}")


def fig_4_2_mae_spatial_map():
    import sys
    sys.path.insert(0, "../../../pipelines/busker_baseline")
    from busker_paths import load_all_units

    units = load_all_units()  # OBJECTID, county, country, geometry
    unit_metrics = pd.read_csv(f"{METRICS_DIR}/metrics_per_unit.csv")
    mae3 = unit_metrics[unit_metrics["lead"] == 3][["county", "mae"]]

    gdf = units.merge(mae3, on="county", how="left")
    matched = gdf["mae"].notna().sum()
    print(f"MAE map: matched {matched}/{len(gdf)} units to lead-3 MAE")

    mae_max = max(1.5, float(gdf["mae"].max()) + 0.01)
    bounds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, mae_max]
    colors = ["#1a7837", "#3d8f3f", "#5fa646", "#82bd4e", "#a8d15a", "#c9dd6c",
              "#d9b84a", "#e0932f", "#e2701f", "#e2481a", "#c81414"]
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N)

    # Each panel's data has its own natural aspect ratio (the reference
    # screenshot's pixel width/height; the reproduction's lon/lat bounding
    # box, which is what geopandas' own default equal-aspect plotting uses
    # internally). `imshow` and `GeoDataFrame.plot()` both lock their axes
    # to their content's aspect ratio, so handing them equal-size boxes
    # (as plain `plt.subplots` would) makes one auto-shrink inside its box
    # to preserve that ratio while the other doesn't -- leaving the two
    # maps at visibly different scales, and their titles, each positioned
    # relative to its own (differently shrunk) axes top, off the same
    # line. The fix is to size each column's box to its content's own
    # aspect ratio *before* plotting, at a shared target content height,
    # so neither axes needs to shrink and both boxes -- hence both titles
    # -- start at the same y position. The colorbar gets its own dedicated
    # third column (`cax=`) rather than `fig.colorbar(..., ax=ax1)`, which
    # would shrink ax1 again after the fact to make room for it.
    from matplotlib.gridspec import GridSpec

    ref_img_path = f"{FIG_DIR}/_reference_pages/fig4_mae_map_crop.png"
    ref_img = plt.imread(ref_img_path)
    ref_aspect = ref_img.shape[1] / ref_img.shape[0]  # width / height

    minx, miny, maxx, maxy = gdf.total_bounds
    gdf_aspect = (maxx - minx) / (maxy - miny)

    content_height = 6.5  # inches, shared map-content height for both panels
    cbar_width = 0.35
    width0 = content_height * ref_aspect
    width1 = content_height * gdf_aspect
    top_margin = 0.16  # figure-fraction reserved above the maps for titles/suptitle

    fig_width = width0 + width1 + cbar_width + 0.6  # + inter-panel gaps
    fig_height = content_height / (1 - top_margin)
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = GridSpec(1, 3, width_ratios=[width0, width1, cbar_width], wspace=0.12,
                  figure=fig, left=0.02, right=0.98, top=1 - top_margin, bottom=0.03)

    ax0 = fig.add_subplot(gs[0])
    ax0.imshow(ref_img)
    ax0.axis("off")
    ax0.set_title("Busker et al. (2024) Figure 4\n(published reference, 3-month lead)")

    ax1 = fig.add_subplot(gs[1])
    gdf.plot(column="mae", cmap=cmap, norm=norm, linewidth=0.15, edgecolor="black",
              ax=ax1, missing_kwds={"color": "white", "hatch": "///", "label": "no data"})
    ax1.set_aspect("equal")
    ax1.set_title("Reproduction (this study)\n3-month-lead MAE, 213 units")
    ax1.axis("off")

    cax = fig.add_subplot(gs[2])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = fig.colorbar(sm, cax=cax, ticks=bounds[:-1])
    cbar.set_label("Mean Absolute Error (MAE)")
    cax.set_box_aspect(20)  # keep the colorbar narrow regardless of its column width

    fig.suptitle("RQ1 Experiment 1 -- MAE spatial map comparison (3-month lead)", y=0.995)
    fig.savefig(f"{FIG_DIR}/rq1_mae_spatial_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_mae_spatial_comparison.png")


def fig_4_3_hitrate_farate():
    onsets = pd.read_csv(f"{METRICS_DIR}/crisis_onset_rates.csv").set_index(["lhz", "lead"])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for col, lhz in enumerate(CLUSTERS):
        repro_hr = [onsets.loc[(lhz, l), "hr"] for l in LEADS]
        repro_far = [onsets.loc[(lhz, l), "far"] for l in LEADS]
        ref_hr = [REF_HR_FAR[(lhz, l)][0] for l in LEADS]
        ref_far = [REF_HR_FAR[(lhz, l)][1] for l in LEADS]

        ax_hr, ax_far = axes[0, col], axes[1, col]
        ax_hr.plot(LEADS, repro_hr, "-o", color=COLOR_XGB, lw=2, ms=4, label="HR (reproduced)")
        ax_hr.plot(LEADS, ref_hr, "--", color="grey", lw=1.6, label="HR (Busker et al. 2024)")
        ax_hr.set_title(CLUSTER_LABELS[lhz])
        ax_hr.set_ylabel("Hit Rate (HR)" if col == 0 else "")
        ax_hr.set_ylim(-0.02, 0.6)

        ax_far.plot(LEADS, repro_far, "-o", color="#a33", lw=2, ms=4, label="FAR (reproduced)")
        ax_far.plot(LEADS, ref_far, "--", color="grey", lw=1.6, label="FAR (Busker et al. 2024)")
        ax_far.set_ylabel("False Alarm Rate (FAR)" if col == 0 else "")
        ax_far.set_xlabel("lead time (months)")
        ax_far.set_xticks(LEADS)
        ax_far.set_ylim(-0.01, 0.20)

    handles = (axes[0, 0].get_legend_handles_labels()[0]
               + axes[1, 0].get_legend_handles_labels()[0])
    labels = (axes[0, 0].get_legend_handles_labels()[1]
              + axes[1, 0].get_legend_handles_labels()[1])
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05),
               frameon=False)
    fig.suptitle("RQ1 Experiment 1 -- Hit rate / false-alarm rate for crisis-onset "
                  "detection: reproduction vs. Busker et al. (2024)", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq1_hitrate_farate_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_hitrate_farate_comparison.png")


def fig_4_4_reference_vs_reproduction_scatter():
    acc = pd.read_csv(f"{METRICS_DIR}/statistical_acceptance_r2.csv")

    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = -0.1, 0.85
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="-", zorder=1, label="1:1 line")

    # Per-point vertical error bar = that cell's own robust (MAD-based)
    # per-unit R2 SD, since the raw per-unit SD is inflated by outlier units
    # with a near-zero-variance test-set target (see report.md section 3 /
    # build_acceptance_tables.py). A continuous band isn't meaningful here --
    # the SD is a property of each (zone, lead) cell, not a smooth function of
    # the reference R2 -- so each cell gets its own error bar instead.
    within = acc["within_1sd_robust"].astype(bool)
    ax.errorbar(acc["reference_r2"], acc["reproduced_r2"],
                yerr=acc["per_unit_r2_robust_sd_mad"], fmt="none",
                ecolor="grey", elinewidth=1, capsize=3, alpha=0.5, zorder=2,
                label="±1 SD (robust, that cell's own per-unit spread)")

    ax.scatter(acc.loc[within, "reference_r2"], acc.loc[within, "reproduced_r2"],
               color="#1b7a3d", s=55, zorder=3, label=f"within 1 SD (n={within.sum()})")
    ax.scatter(acc.loc[~within, "reference_r2"], acc.loc[~within, "reproduced_r2"],
               color="#c81414", s=55, zorder=3, label=f"outside 1 SD (n={(~within).sum()})")

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
    ax.set_title("RQ1 Experiment 1 -- reference vs. reproduction R$^2$\n"
                  "all 21 livelihood-zone x lead-time cells")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq1_reference_vs_reproduction_scatter.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_reference_vs_reproduction_scatter.png")


def fig_4_4b_bootstrap_scatter():
    """Same scatter as fig_4_4, but error bars are the bootstrap SE of the
    pooled R2 statistic (bootstrap_acceptance.py) instead of the per-unit
    heterogeneity SD -- see report.md for why this is the more defensible
    of the two. Separate file so both are available to compare.
    """
    acc = pd.read_csv(f"{METRICS_DIR}/statistical_acceptance_r2_bootstrap.csv")

    fig, ax = plt.subplots(figsize=(7, 7))
    lo, hi = -0.1, 0.85
    ax.plot([lo, hi], [lo, hi], color="grey", lw=1, ls="-", zorder=1, label="1:1 line")

    within = acc["within_1sd_bootstrap"].astype(bool)
    ax.errorbar(acc["reference_r2"], acc["reproduced_r2"],
                yerr=acc["bootstrap_r2_sd"], fmt="none",
                ecolor="grey", elinewidth=1, capsize=3, alpha=0.6, zorder=2,
                label="±1 bootstrap SE of the pooled R$^2$ (500 unit resamples)")

    ax.scatter(acc.loc[within, "reference_r2"], acc.loc[within, "reproduced_r2"],
               color="#1b7a3d", s=55, zorder=3, label=f"within 1 SD (n={within.sum()})")
    ax.scatter(acc.loc[~within, "reference_r2"], acc.loc[~within, "reproduced_r2"],
               color="#c81414", s=55, zorder=3, label=f"outside 1 SD (n={(~within).sum()})")

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
    ax.set_title("RQ1 Experiment 1 -- reference vs. reproduction R$^2$ (bootstrap SE)\n"
                  "all 21 livelihood-zone x lead-time cells")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/rq1_reference_vs_reproduction_scatter_bootstrap.png", dpi=200,
                bbox_inches="tight")
    plt.close(fig)
    print("wrote rq1_reference_vs_reproduction_scatter_bootstrap.png")


if __name__ == "__main__":
    fig_4_1_r2_by_leadtime()
    fig_4_2_mae_spatial_map()
    fig_4_3_hitrate_farate()
    fig_4_4_reference_vs_reproduction_scatter()
    fig_4_4b_bootstrap_scatter()
