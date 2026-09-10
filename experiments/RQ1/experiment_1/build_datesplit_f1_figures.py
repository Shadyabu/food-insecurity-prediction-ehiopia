"""Figures for the Busker-architecture reproduction re-scored on a
date-based split (train<=2019-12-31, test 2020-01-01..2022-12-31 -- the
input_master's real data ceiling; Busker's own released dataset stops at
2022-12, so a genuine 2020-2024 test window is not available for this
arm). Requested per the RQ1 Reproduction plan: overall weighted F1,
weighted F1 by lead time (Ethiopia, all zones combined), and weighted F1
by lead time split by livelihood zone.

**Fixed 2026-08-26 (figure audit)**: the original version of this script
split "pastoral" as a binary Ethiopia-all-zones vs. Ethiopia-pastoral-only
comparison. Every other multi-region figure in this project (ablation
comparison, experiment_1_vs_2_comparison, RQ2 Experiment 5's exhaustive-
search figures) splits into the full 3 livelihood-zone clusters
(pastoral/agropastoral/crop farming) -- this script now matches that
convention instead of inventing its own.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "busker_arch_datesplit_2020_2022")
FIG_DIR = os.path.join(DATA_DIR, "figures")

LEADS = [0, 1, 2, 3, 4, 8, 12]
LHZ_ORDER = ["p", "ap", "other"]
LHZ_LABELS = {"p": "Pastoral", "ap": "Agro-pastoral", "other": "Crop farming"}
LHZ_COLORS = {"p": "#eb6834", "ap": "#1baf7a", "other": "#eda100"}
COLOR_ALL = "#2a78d6"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def f1_by_lead(df):
    rows = []
    for lead in LEADS:
        sub = df[df["lead"] == lead]
        if len(sub) == 0:
            continue
        y_true = to_ipc_class(sub["observed"])
        y_pred = to_ipc_class(sub["prediction"])
        rows.append({
            "lead": lead, "n": len(sub),
            "accuracy": accuracy_score(y_true, y_pred),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    preds = pd.read_parquet(os.path.join(DATA_DIR, "predictions.parquet"))
    eth = preds[preds["country"] == "Ethiopia"]

    ethiopia_all = f1_by_lead(eth)
    per_lhz = {lhz: f1_by_lead(eth[eth["lhz"] == lhz]) for lhz in LHZ_ORDER}

    # --- Figure 1: overall weighted F1 (single value, Ethiopia all zones) ---
    fig, ax = plt.subplots(figsize=(4, 5))
    overall_f1 = ethiopia_all["f1_weighted"].mean()
    ax.bar(["Ethiopia\n(all zones)"], [overall_f1], color=COLOR_ALL)
    ax.text(0, overall_f1 + 0.01, f"{overall_f1:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mean weighted F1 (across leads 0,1,2,3,4,8,12)")
    ax.set_title("Busker-architecture reproduction, date-split\n"
                  "train<=2019-12, test 2020-01..2022-12")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "overall_f1.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote overall_f1.png")

    # --- Figure 2: weighted F1 by lead time, Ethiopia all zones combined ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ethiopia_all["lead"], ethiopia_all["f1_weighted"], "-o", color=COLOR_ALL, lw=2, ms=6)
    ax.set_xlabel("lead time (months)")
    ax.set_ylabel("weighted F1")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(LEADS)
    ax.set_title("Busker-architecture reproduction -- weighted F1 by lead time\n"
                  "Ethiopia, all zones combined, test 2020-01..2022-12")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "f1_by_leadtime_ethiopia.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote f1_by_leadtime_ethiopia.png")

    # --- Figure 3: weighted F1 by lead time, split by livelihood zone (3-way) ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(ethiopia_all["lead"], ethiopia_all["f1_weighted"], "-o", color=COLOR_ALL, lw=2, ms=6,
            label="Ethiopia (all zones combined)")
    for lhz in LHZ_ORDER:
        sub = per_lhz[lhz]
        ax.plot(sub["lead"], sub["f1_weighted"], "-s", color=LHZ_COLORS[lhz], lw=2, ms=6,
                label=LHZ_LABELS[lhz])
    ax.set_xlabel("lead time (months)")
    ax.set_ylabel("weighted F1")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(LEADS)
    ax.set_title("Busker-architecture reproduction -- weighted F1 by lead time and livelihood zone\n"
                  "Ethiopia, test 2020-01..2022-12")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "f1_by_leadtime_pastoral.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote f1_by_leadtime_pastoral.png (now a 3-way livelihood-zone split)")


if __name__ == "__main__":
    main()
