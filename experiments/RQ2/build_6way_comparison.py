"""RQ2's core "Figures" requirement (dissertation plan), for Experiments
1-4: a unified comparison across persistence, XGBoost (best variant),
Busker-reproduced baseline, LSTM, T-GCN, and RandomForest -- overall
weighted F1, by lead time (combined), and by lead time x livelihood zone.

Did not exist before the 2026-08-26 figure audit: each architecture's own
`build_comparison.py` only ever wrote CSV summary tables (pairwise
against XGBoost/persistence), never a unified plot across all 6 series,
and none of them included the Busker-reproduced baseline at all.

Test window: "extended" (2020-02 to 2024-10) for every architecture that
has one -- matches the dissertation plan's "2020-2024 test" framing for
RQ2 Experiments 1-4. Busker architecture is the one exception: its own
target data has no observations after 2022-06 (Busker's released dataset
ceiling -- see experiments/RQ1/experiment_1/busker_arch_datesplit_2020_2022),
so its line necessarily stops there. T-GCN's own "extended" file runs to
2026-06 (never restricted to 2024 in its own build); trimmed here to
<=2024-10 to match every other source's ceiling, same convention already
used in RQ2 Experiment 5's ensemble work.

Discretization: round+clip to the 1-5 IPC scale for the four
continuous-output models (persistence, XGBoost, Busker, RandomForest,
LSTM) -- standard convention used everywhere else in this project.
T-GCN is classification-native (`true_class`/`pred_class`, already
discrete) -- scored directly, no conversion needed (F1 is invariant to
label encoding as long as true/pred share it).
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADS = [0, 1, 2, 3, 4, 8, 12]
LHZ_ORDER = ["pastoral", "agropastoral", "crop_farming"]
LHZ_LABELS = {"pastoral": "Pastoral", "agropastoral": "Agro-pastoral", "crop_farming": "Crop farming"}
LHZ_RENAME = {"p": "pastoral", "ap": "agropastoral", "other": "crop_farming"}
EXTENDED_CEILING = "2024-10-31"

MODEL_ORDER = ["persistence", "xgboost", "busker_arch", "random_forest", "lstm", "tgcn"]
MODEL_LABELS = {
    "persistence": "Persistence",
    "xgboost": "XGBoost (best variant)",
    "busker_arch": "Busker reproduction (ref., ends 2022-06)",
    "random_forest": "RandomForest (best variant)",
    "lstm": "LSTM (delta)",
    "tgcn": "T-GCN (CE head)",
}
MODEL_COLORS = {
    "persistence": "#52514e",
    "xgboost": "#2a78d6",
    "busker_arch": "#eb6834",
    "random_forest": "#1baf7a",
    "lstm": "#eda100",
    "tgcn": "#e87ba4",
}
MODEL_MARKERS = {"persistence": "x", "xgboost": "o", "busker_arch": "D",
                  "random_forest": "^", "lstm": "s", "tgcn": "v"}

XGB_PATH = os.path.join(SCRIPT_DIR, "..", "RQ1", "experiment_2", "unhcr_delta_noclimate_extended", "predictions.parquet")
RF_PATH = os.path.join(SCRIPT_DIR, "experiment_4", "rf_delta_noclimate_extended", "predictions.parquet")
LSTM_PATH = os.path.join(SCRIPT_DIR, "experiment_1", "lstm_delta", "predictions.parquet")
TGCN_PATH = os.path.join(SCRIPT_DIR, "experiment_3", "tgcn_ce", "predictions.parquet")
BUSKER_PATH = os.path.join(SCRIPT_DIR, "..", "RQ1", "experiment_1", "busker_arch_datesplit_2020_2022", "predictions.parquet")

OUT_DIR = os.path.join(SCRIPT_DIR, "rq2_6way_comparison")

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def score_continuous(df, group_cols, obs_col="observed", pred_col="prediction"):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        yt, yp = to_ipc_class(g[obs_col]), to_ipc_class(g[pred_col])
        rows.append({**dict(zip(group_cols, keys)), "n": len(g),
                     "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0)})
    return pd.DataFrame(rows)


def score_discrete(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rows.append({**dict(zip(group_cols, keys)), "n": len(g),
                     "f1_weighted": f1_score(g["true_class"], g["pred_class"], average="weighted", zero_division=0)})
    return pd.DataFrame(rows)


def load_all():
    xgb = pd.read_parquet(XGB_PATH)
    rf = pd.read_parquet(RF_PATH)
    lstm = pd.read_parquet(LSTM_PATH)
    lstm = lstm[lstm["window"] == "extended"].copy()
    tgcn = pd.read_parquet(TGCN_PATH)
    tgcn = tgcn[(tgcn["window"] == "extended") & (tgcn["time"] <= EXTENDED_CEILING)].copy()
    busker = pd.read_parquet(BUSKER_PATH)
    busker["time"] = pd.to_datetime(busker["time"])
    busker = busker[busker["country"] == "Ethiopia"].copy()
    busker["lhz"] = busker["lhz"].map(LHZ_RENAME)
    # persistence: derived from XGBoost's own base1_preds column (same
    # panel/rows as every other continuous-output model here)
    persistence = xgb.drop(columns=["prediction"]).rename(columns={"base1_preds": "prediction"}).copy()

    return {
        "persistence": ("continuous", persistence),
        "xgboost": ("continuous", xgb),
        "busker_arch": ("continuous", busker),
        "random_forest": ("continuous", rf),
        "lstm": ("continuous", lstm),
        "tgcn": ("discrete", tgcn),
    }


def score_model(kind, df, group_cols):
    if kind == "continuous":
        return score_continuous(df, group_cols)
    return score_discrete(df, group_cols)


def overall_f1(kind, df):
    if kind == "continuous":
        yt, yp = to_ipc_class(df["observed"]), to_ipc_class(df["prediction"])
        return f1_score(yt, yp, average="weighted", zero_division=0)
    return f1_score(df["true_class"], df["pred_class"], average="weighted", zero_division=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = load_all()

    for key in MODEL_ORDER:
        kind, df = data[key]
        print(f"{key}: n={len(df)}, {pd.to_datetime(df['time']).min().date()} to {pd.to_datetime(df['time']).max().date()}")

    overall = pd.DataFrame([
        {"model": key, "n": len(data[key][1]), "f1_weighted": overall_f1(*data[key])}
        for key in MODEL_ORDER
    ])
    by_lead = pd.concat([score_model(*data[k], ["lead"]).assign(model=k) for k in MODEL_ORDER], ignore_index=True)
    by_lead_lhz = pd.concat([score_model(*data[k], ["lhz", "lead"]).assign(model=k) for k in MODEL_ORDER], ignore_index=True)

    overall.to_csv(os.path.join(OUT_DIR, "overall.csv"), index=False)
    by_lead.to_csv(os.path.join(OUT_DIR, "by_lead.csv"), index=False)
    by_lead_lhz.to_csv(os.path.join(OUT_DIR, "by_lead_and_lhz.csv"), index=False)
    print("\n=== Overall ===")
    print(overall.to_string(index=False))

    # --- Figure 1: overall weighted F1, 6 models ---
    fig, ax = plt.subplots(figsize=(10, 5.5))
    vals = [overall[overall["model"] == m]["f1_weighted"].iloc[0] for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    bars = ax.bar(range(len(MODEL_ORDER)), vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels([MODEL_LABELS[m].replace(" (", "\n(") for m in MODEL_ORDER], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("weighted F1")
    ax.set_title("RQ2 -- overall weighted F1, all architectures\nEthiopia, 2020-2024 test period (Busker line ends 2022-06)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "overall_f1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote overall_f1.png")

    # --- Figure 2: weighted F1 by lead, all zones combined ---
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in MODEL_ORDER:
        sub = by_lead[by_lead["model"] == m].sort_values("lead")
        ls = "--" if m in ("persistence", "busker_arch") else "-"
        ax.plot(sub["lead"], sub["f1_weighted"], marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                 label=MODEL_LABELS[m], linewidth=2, markersize=7, linestyle=ls)
    ax.set_xlabel("lead time (months)")
    ax.set_ylabel("weighted F1")
    ax.set_xticks(LEADS)
    ax.set_ylim(0, 1.0)
    ax.set_title("RQ2 -- weighted F1 by lead time, all zones combined\nEthiopia, 2020-2024 test period")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f1_by_leadtime_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote f1_by_leadtime_combined.png")

    # --- Figure 3: weighted F1 by lead time, split by livelihood zone ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), sharey=True)
    for ax, lhz in zip(axes, LHZ_ORDER):
        sub_lhz = by_lead_lhz[by_lead_lhz["lhz"] == lhz]
        for m in MODEL_ORDER:
            sub = sub_lhz[sub_lhz["model"] == m].sort_values("lead")
            ls = "--" if m in ("persistence", "busker_arch") else "-"
            ax.plot(sub["lead"], sub["f1_weighted"], marker=MODEL_MARKERS[m], color=MODEL_COLORS[m],
                     label=MODEL_LABELS[m], linewidth=1.8, markersize=6, linestyle=ls)
        ax.set_title(LHZ_LABELS[lhz])
        ax.set_xlabel("lead time (months)")
        ax.set_xticks(LEADS)
    axes[0].set_ylabel("weighted F1")
    axes[0].set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.12), frameon=False, fontsize=8.5)
    fig.suptitle("RQ2 -- weighted F1 by lead time and livelihood zone\nEthiopia, 2020-2024 test period", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f1_by_leadtime_lhz.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote f1_by_leadtime_lhz.png")


if __name__ == "__main__":
    main()
