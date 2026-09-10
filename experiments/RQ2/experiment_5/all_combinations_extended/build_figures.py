"""Three figures for the exhaustive ensemble-combination search on the
extended (2020-02..2024-10) test window
(build_all_combinations_extended.py). Reads combos_summary.csv,
top5_detailed_metrics.csv, persistence_detailed_metrics.csv.

Colors: dataviz skill's validated default categorical palette, slots 1-5
(blue/orange/aqua/yellow/magenta), assigned in fixed rank order (best
combo = slot 1) -- same convention as ../all_combinations/build_figures.py.
Persistence is a reference, not a competing series: muted gray, dashed,
no marker.

**Updated 2026-08-26 (figure audit)**: added the Busker-reproduced
baseline as a second reference series (`busker_detailed_metrics.csv`,
built by `../build_busker_reference.py`) -- the dissertation plan
explicitly asks for this comparison and it was missing entirely.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]
CLUSTER_TITLES = {"pastoral": "Pastoral", "agropastoral": "Agropastoral", "crop_farming": "Crop-farming"}

COMBO_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # slots 1-5
COMBO_MARKERS = ["o", "s", "^", "D", "v"]
PERSISTENCE_COLOR = "#898781"
BUSKER_COLOR = "#52514e"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def load_data():
    summary = pd.read_csv(os.path.join(SCRIPT_DIR, "combos_summary.csv"))
    detail = pd.read_csv(os.path.join(SCRIPT_DIR, "top5_detailed_metrics.csv"))
    persistence = pd.read_csv(os.path.join(SCRIPT_DIR, "persistence_detailed_metrics.csv"))
    busker = pd.read_csv(os.path.join(SCRIPT_DIR, "busker_detailed_metrics.csv"))
    top5_labels = summary.head(5)["combo"].tolist()
    return summary, detail, persistence, busker, top5_labels


def figure1_overall_bar(summary, top5_labels, busker):
    top5 = summary.set_index("combo").loc[top5_labels].reset_index()
    persistence_f1 = pd.read_csv(os.path.join(SCRIPT_DIR, "persistence_detailed_metrics.csv"))
    persistence_all = persistence_f1[persistence_f1["subset"] == "all"]["f1_weighted"].mean()
    busker_all = busker[busker["subset"] == "all"]["f1_weighted"].mean()

    labels = top5["combo"].tolist() + ["Persistence", "Busker (reference)"]
    values = top5["mean_f1_all"].tolist() + [persistence_all, busker_all]
    colors = COMBO_COLORS + [PERSISTENCE_COLOR, BUSKER_COLOR]

    order = list(range(len(labels)))[::-1]
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlabel("mean weighted F1 (all zones, averaged across 7 lead times)")
    ax.set_title("Figure 1 -- Overall performance of the top 5 ensemble combinations\n"
                  "(extended test window 2020-02..2024-10; exhaustive search over all 26 size-2..5 subsets)")
    ax.set_xlim(0, max(values) * 1.15)
    for bar, val in zip(bars, values):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.4f}", va="center", ha="left", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, "figure1_overall_bar.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def figure2_by_lead_allzones(detail, persistence, busker, top5_labels):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, label in enumerate(top5_labels):
        sub = detail[(detail["combo"] == label) & (detail["subset"] == "all")].sort_values("lead")
        ax.plot(sub["lead"], sub["f1_weighted"], marker=COMBO_MARKERS[i], color=COMBO_COLORS[i],
                label=label, linewidth=2, markersize=7)

    persist_all = persistence[persistence["subset"] == "all"].sort_values("lead")
    ax.plot(persist_all["lead"], persist_all["f1_weighted"], color=PERSISTENCE_COLOR, linestyle="--",
            linewidth=2, label="Persistence (reference)")
    busker_all = busker[busker["subset"] == "all"].sort_values("lead")
    ax.plot(busker_all["lead"], busker_all["f1_weighted"], color=BUSKER_COLOR, linestyle=":",
            marker="x", linewidth=2, label="Busker reproduction (reference, ends 2022-06)")

    ax.set_xlabel("lead time (months)")
    ax.set_ylabel("weighted F1")
    ax.set_xticks(LEADS)
    ax.set_title("Figure 2 -- Top 5 ensemble combinations by lead time (extended window 2020-02..2024-10)\n"
                  "(all zones, all livelihood regions pooled)")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    out_path = os.path.join(FIG_DIR, "figure2_by_lead_allzones.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def figure3_by_lead_per_region(detail, persistence, busker, top5_labels):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
    for ax, cluster in zip(axes, CLUSTERS):
        for i, label in enumerate(top5_labels):
            sub = detail[(detail["combo"] == label) & (detail["subset"] == cluster)].sort_values("lead")
            ax.plot(sub["lead"], sub["f1_weighted"], marker=COMBO_MARKERS[i], color=COMBO_COLORS[i],
                    label=label, linewidth=2, markersize=7)
        persist_sub = persistence[persistence["subset"] == cluster].sort_values("lead")
        ax.plot(persist_sub["lead"], persist_sub["f1_weighted"], color=PERSISTENCE_COLOR, linestyle="--",
                linewidth=2, label="Persistence (reference)")
        busker_sub = busker[busker["subset"] == cluster].sort_values("lead")
        ax.plot(busker_sub["lead"], busker_sub["f1_weighted"], color=BUSKER_COLOR, linestyle=":",
                marker="x", linewidth=2, label="Busker reproduction (reference, ends 2022-06)")
        ax.set_title(CLUSTER_TITLES[cluster])
        ax.set_xlabel("lead time (months)")
        ax.set_xticks(LEADS)
    axes[0].set_ylabel("weighted F1")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.suptitle("Figure 3 -- Top 5 ensemble combinations by lead time, per livelihood-zone region\n"
                 "(extended test window 2020-02..2024-10)", fontsize=12)
    fig.tight_layout(rect=[0, 0.13, 1, 0.94])
    out_path = os.path.join(FIG_DIR, "figure3_by_lead_per_region.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    summary, detail, persistence, busker, top5_labels = load_data()
    figure1_overall_bar(summary, top5_labels, busker)
    figure2_by_lead_allzones(detail, persistence, busker, top5_labels)
    figure3_by_lead_per_region(detail, persistence, busker, top5_labels)


if __name__ == "__main__":
    main()
