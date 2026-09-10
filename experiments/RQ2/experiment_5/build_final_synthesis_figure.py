"""RQ2 Experiment 5's closing figure set (dissertation plan): the best
ensemble combination against the 3 best individual models, the
Busker-reproduced baseline, the persistence model, and the FEWS NET
outlook's own real-world performance. Did not exist before the
2026-08-26 figure audit.

Best ensemble (XGB+RF+LSTM+TabICL) and the 3 best solo architectures
(RandomForest, XGBoost, LSTM-delta) are all scored on the SAME default
test window (2020-02..2022-10) that `all_combinations/` already uses --
matches this project's established persistence/Busker reference files in
that same folder.

**"3 best individual" selection note**: `full_ensemble_summary.csv`
ranks TabICLv2 (0.6672) above LSTM-delta (0.6641) -- but that ranking is
computed on the row set the 5-way ensemble intersects across all
architectures, not each model's own full native test set. Scored here
the same way every other solo model in this figure is scored (each
architecture's own full default-window test set, per
`experiment_2/comparison_summary.csv`), TabICLv2's real solo number is
0.6336 -- *below* LSTM-delta's 0.6641. Using LSTM-delta (not TabICLv2)
as the 3rd-best individual model keeps this figure internally consistent
(every solo number scored the same way) rather than mixing two different
row-subset conventions.

FEWS NET's outlook uses a DIFFERENT lead definition (`lead_end`, ML1
leads 0-3 / ML2 leads 4-7 -- see
`experiments/RQ1/experiment_1/build_fews_outlook_reproduction.py`) than
every model here (0,1,2,3,4,8,12). Leads 0-3 line up directly; FEWS's
4-7 do NOT correspond to this project's 4/8/12 -- plotted on the same
axis for visual reference only, with that mismatch stated in the
caption, not hidden.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(SCRIPT_DIR, "all_combinations")
OUT_DIR = os.path.join(SCRIPT_DIR, "final_synthesis")
LEADS = [0, 1, 2, 3, 4, 8, 12]
LHZ_ORDER = ["pastoral", "agropastoral", "crop_farming"]
LHZ_LABELS = {"pastoral": "Pastoral", "agropastoral": "Agro-pastoral", "crop_farming": "Crop farming"}

XGB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "RQ1", "experiment_2", "unhcr_delta_noclimate", "predictions.parquet")
RF_PATH = os.path.join(SCRIPT_DIR, "..", "experiment_4", "rf_delta_noclimate", "predictions.parquet")
LSTM_PATH = os.path.join(SCRIPT_DIR, "..", "experiment_1", "lstm_delta", "predictions.parquet")
FEWS_PATH = os.path.join(SCRIPT_DIR, "..", "..", "RQ1", "experiment_1", "fews_outlook_f1_2020_2024.csv")

SERIES_ORDER = ["best_ensemble", "random_forest", "xgboost", "lstm", "busker_arch", "persistence", "fews_outlook"]
SERIES_LABELS = {
    "best_ensemble": "Best ensemble (XGB+RF+LSTM+TabICL)",
    "random_forest": "RandomForest (best solo)",
    "xgboost": "XGBoost (best solo)",
    "lstm": "LSTM-delta (3rd best solo)",
    "busker_arch": "Busker reproduction (ends 2022-06)",
    "persistence": "Persistence",
    "fews_outlook": "FEWS NET outlook (own lead scheme, see caption)",
}
SERIES_COLORS = {
    "best_ensemble": "#2a78d6", "random_forest": "#1baf7a", "xgboost": "#eb6834",
    "lstm": "#eda100", "busker_arch": "#52514e", "persistence": "#898781", "fews_outlook": "#e87ba4",
}
SERIES_MARKERS = {"best_ensemble": "o", "random_forest": "^", "xgboost": "s", "lstm": "D",
                   "busker_arch": "x", "persistence": "*", "fews_outlook": "P"}

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def score_solo(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        yt, yp = to_ipc_class(g["observed"]), to_ipc_class(g["prediction"])
        rows.append({**dict(zip(group_cols, keys)), "n": len(g),
                     "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    xgb = pd.read_parquet(XGB_PATH)
    rf = pd.read_parquet(RF_PATH)
    lstm = pd.read_parquet(LSTM_PATH)
    lstm = lstm[lstm["window"] == "default"].copy()

    solo = {"xgboost": xgb, "random_forest": rf, "lstm": lstm}

    ensemble_detail = pd.read_csv(os.path.join(DEFAULT_DIR, "top5_detailed_metrics.csv"))
    ensemble_detail = ensemble_detail[ensemble_detail["combo"] == "XGB+RF+LSTM+TabICL"]
    persistence = pd.read_csv(os.path.join(DEFAULT_DIR, "persistence_detailed_metrics.csv"))
    busker = pd.read_csv(os.path.join(DEFAULT_DIR, "busker_detailed_metrics.csv"))
    fews = pd.read_csv(FEWS_PATH)

    # --- build a common by_lead / by_lead_lhz table for the 6 model series ---
    by_lead_rows, by_lhz_rows = [], []
    for key in ["random_forest", "xgboost", "lstm"]:
        by_lead_rows.append(score_solo(solo[key], ["lead"]).assign(series=key))
        by_lhz_rows.append(score_solo(solo[key], ["lhz", "lead"]).assign(series=key))
    by_lead_rows.append(ensemble_detail[ensemble_detail["subset"] == "all"][["lead", "n", "f1_weighted"]].assign(series="best_ensemble"))
    for lhz in LHZ_ORDER:
        by_lhz_rows.append(ensemble_detail[ensemble_detail["subset"] == lhz][["lead", "n", "f1_weighted"]].assign(lhz=lhz, series="best_ensemble"))
    by_lead_rows.append(persistence[persistence["subset"] == "all"][["lead", "n", "f1_weighted"]].assign(series="persistence"))
    for lhz in LHZ_ORDER:
        by_lhz_rows.append(persistence[persistence["subset"] == lhz][["lead", "n", "f1_weighted"]].assign(lhz=lhz, series="persistence"))
    by_lead_rows.append(busker[busker["subset"] == "all"][["lead", "n", "f1_weighted"]].assign(series="busker_arch"))
    for lhz in LHZ_ORDER:
        by_lhz_rows.append(busker[busker["subset"] == lhz][["lead", "n", "f1_weighted"]].assign(lhz=lhz, series="busker_arch"))

    by_lead = pd.concat(by_lead_rows, ignore_index=True)
    by_lhz = pd.concat(by_lhz_rows, ignore_index=True)
    by_lead.to_csv(os.path.join(OUT_DIR, "by_lead.csv"), index=False)
    by_lhz.to_csv(os.path.join(OUT_DIR, "by_lead_and_lhz.csv"), index=False)

    overall = pd.DataFrame([
        {"series": s, "f1_weighted": by_lead[by_lead["series"] == s]["f1_weighted"].mean()}
        for s in ["best_ensemble", "random_forest", "xgboost", "lstm", "busker_arch", "persistence"]
    ] + [{"series": "fews_outlook", "f1_weighted": fews[fews["cluster"] == "all_zones"]["weighted_f1"].mean()}])
    overall.to_csv(os.path.join(OUT_DIR, "overall.csv"), index=False)
    print(overall.to_string(index=False))

    # --- Figure 1: overall bar, all 7 series ---
    fig, ax = plt.subplots(figsize=(11, 5.5))
    vals = [overall[overall["series"] == s]["f1_weighted"].iloc[0] for s in SERIES_ORDER]
    colors = [SERIES_COLORS[s] for s in SERIES_ORDER]
    bars = ax.bar(range(len(SERIES_ORDER)), vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(SERIES_ORDER)))
    ax.set_xticklabels([SERIES_LABELS[s].replace(" (", "\n(") for s in SERIES_ORDER], fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("weighted F1")
    ax.set_title("RQ2 Experiment 5 -- final synthesis: best ensemble vs. best individual models,\n"
                  "Busker reproduction, persistence, and FEWS NET's own outlook performance")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "overall_f1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote overall_f1.png")

    # --- Figure 2: by lead, all zones combined ---
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for s in SERIES_ORDER:
        if s == "fews_outlook":
            continue
        sub = by_lead[by_lead["series"] == s].sort_values("lead")
        ls = "--" if s in ("busker_arch", "persistence") else "-"
        ax.plot(sub["lead"], sub["f1_weighted"], marker=SERIES_MARKERS[s], color=SERIES_COLORS[s],
                 label=SERIES_LABELS[s], linewidth=2, markersize=7, linestyle=ls)
    fews_all = fews[fews["cluster"] == "all_zones"].sort_values("lead_end")
    ax.plot(fews_all["lead_end"], fews_all["weighted_f1"], marker=SERIES_MARKERS["fews_outlook"],
             color=SERIES_COLORS["fews_outlook"], linewidth=2, markersize=7, linestyle=":",
             label=SERIES_LABELS["fews_outlook"])
    ax.set_xlabel("lead time (months) -- FEWS uses its own lead_end scale, see caption")
    ax.set_ylabel("weighted F1")
    ax.set_xticks(sorted(set(LEADS) | {5, 6, 7}))
    ax.set_ylim(0, 1.0)
    ax.set_title("RQ2 -- weighted F1 by lead time, all zones combined\n"
                  "FEWS NET's lead 0-3 matches this project's 0-3; its 4-7 do NOT correspond to 4/8/12")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f1_by_leadtime_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote f1_by_leadtime_combined.png")

    # --- Figure 3: by lead, split by livelihood zone ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, lhz in zip(axes, LHZ_ORDER):
        sub_lhz = by_lhz[by_lhz["lhz"] == lhz]
        for s in SERIES_ORDER:
            if s == "fews_outlook":
                continue
            sub = sub_lhz[sub_lhz["series"] == s].sort_values("lead")
            ls = "--" if s in ("busker_arch", "persistence") else "-"
            ax.plot(sub["lead"], sub["f1_weighted"], marker=SERIES_MARKERS[s], color=SERIES_COLORS[s],
                     label=SERIES_LABELS[s], linewidth=1.8, markersize=6, linestyle=ls)
        fews_lhz = fews[fews["cluster"] == lhz].sort_values("lead_end")
        ax.plot(fews_lhz["lead_end"], fews_lhz["weighted_f1"], marker=SERIES_MARKERS["fews_outlook"],
                 color=SERIES_COLORS["fews_outlook"], linewidth=1.8, markersize=6, linestyle=":",
                 label=SERIES_LABELS["fews_outlook"])
        ax.set_title(LHZ_LABELS[lhz])
        ax.set_xlabel("lead time (months)")
    axes[0].set_ylabel("weighted F1")
    axes[0].set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.16), frameon=False, fontsize=8)
    fig.suptitle("RQ2 -- weighted F1 by lead time and livelihood zone: final synthesis\n"
                 "FEWS lead 4-7 does not correspond to this project's 4/8/12 -- see caption", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f1_by_leadtime_lhz.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote f1_by_leadtime_lhz.png")


if __name__ == "__main__":
    main()
