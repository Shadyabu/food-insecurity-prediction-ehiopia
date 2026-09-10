"""RQ1 Experiment 1's second required figure set (dissertation plan):
2020-2024 test period, weighted F1, overall / by lead (combined) / by lead
and livelihood zone, comparing the Ethiopia-enriched XGBoost model against
the Busker-reproduced baseline. Did not exist before the 2026-08-26
figure audit.

Busker-architecture line stops at 2022-06 (its own target data's real
ceiling -- see experiments/RQ1/experiment_1/busker_arch_datesplit_2020_2022,
built 2026-08-26) -- a genuine 2020-2024 test window does not exist for
that architecture on Busker's own released dataset. This is disclosed in
every figure caption/legend rather than silently truncated.
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

CONFIGS = [
    ("busker_arch", "Busker architecture (baseline, test ends 2022-06)",
     os.path.join(SCRIPT_DIR, "experiment_1", "busker_arch_datesplit_2020_2022", "predictions.parquet"), True),
    ("enriched_level", "Ethiopia-enriched dataset, same architecture",
     os.path.join(SCRIPT_DIR, "experiment_2", "rq2_level_test2020to2024", "predictions.parquet"), False),
    ("enriched_best", "Ethiopia-enriched dataset, best variant (delta target + no climate)",
     os.path.join(SCRIPT_DIR, "experiment_2", "unhcr_delta_noclimate_extended", "predictions.parquet"), False),
]
CONFIG_ORDER = [c[0] for c in CONFIGS]
CONFIG_LABELS = {c[0]: c[1] for c in CONFIGS}
CONFIG_COLORS = {"busker_arch": "#eb6834", "enriched_level": "#2a78d6", "enriched_best": "#1baf7a"}
CONFIG_MARKERS = {"busker_arch": "o", "enriched_level": "s", "enriched_best": "^"}

OUT_DIR = os.path.join(SCRIPT_DIR, "experiment_1_2020_2024_comparison")

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load(path, is_busker):
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    if is_busker:
        df = df[df["country"] == "Ethiopia"].copy()
        df["lhz"] = df["lhz"].map(LHZ_RENAME)
    return df


def score(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        yt, yp = to_ipc_class(g["observed"]), to_ipc_class(g["prediction"])
        rows.append({**dict(zip(group_cols, keys)), "n": len(g),
                     "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0)})
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    loaded = {key: load(path, is_busker) for key, _, path, is_busker in CONFIGS}
    for key, df in loaded.items():
        print(f"{key}: n={len(df)}, {df['time'].min().date()} to {df['time'].max().date()}")

    overall = pd.DataFrame([
        {"config": key, "n": len(df),
         "f1_weighted": f1_score(to_ipc_class(df["observed"]), to_ipc_class(df["prediction"]),
                                  average="weighted", zero_division=0)}
        for key, df in loaded.items()
    ])
    by_lead = pd.concat([score(loaded[k], ["lead"]).assign(config=k) for k in CONFIG_ORDER], ignore_index=True)
    by_lhz_lead = pd.concat([score(loaded[k], ["lhz", "lead"]).assign(config=k) for k in CONFIG_ORDER],
                             ignore_index=True)
    overall.to_csv(os.path.join(OUT_DIR, "overall.csv"), index=False)
    by_lead.to_csv(os.path.join(OUT_DIR, "by_lead.csv"), index=False)
    by_lhz_lead.to_csv(os.path.join(OUT_DIR, "by_lead_and_lhz.csv"), index=False)
    print(overall.to_string(index=False))

    # --- Figure 1: overall weighted F1 ---
    fig, ax = plt.subplots(figsize=(6, 5))
    vals = [overall[overall["config"] == c]["f1_weighted"].iloc[0] for c in CONFIG_ORDER]
    colors = [CONFIG_COLORS[c] for c in CONFIG_ORDER]
    bars = ax.bar(range(len(CONFIG_ORDER)), vals, color=colors)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(CONFIG_ORDER)))
    ax.set_xticklabels([CONFIG_LABELS[c].replace(", ", "\n") for c in CONFIG_ORDER], fontsize=7.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("weighted F1")
    ax.set_title("RQ1 Experiment 1 -- overall weighted F1, 2020-2024 test period\n"
                  "(Busker-architecture line's own test data ends 2022-06)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "overall_f1.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote overall_f1.png")

    # --- Figure 2: weighted F1 by lead, all zones combined ---
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for cfg in CONFIG_ORDER:
        sub = by_lead[by_lead["config"] == cfg].sort_values("lead")
        ax.plot(sub["lead"], sub["f1_weighted"], marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg],
                 label=CONFIG_LABELS[cfg], linewidth=2, markersize=8)
    ax.set_xlabel("lead time (months)")
    ax.set_ylabel("weighted F1")
    ax.set_xticks(LEADS)
    ax.set_ylim(0, 1.0)
    ax.set_title("RQ1 Experiment 1 -- weighted F1 by lead time, all zones combined\n2020-2024 test period")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f1_by_leadtime_combined.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote f1_by_leadtime_combined.png")

    # --- Figure 3: weighted F1 by lead time, split by livelihood zone ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, lhz in zip(axes, LHZ_ORDER):
        for cfg in CONFIG_ORDER:
            sub = by_lhz_lead[(by_lhz_lead["config"] == cfg) & (by_lhz_lead["lhz"] == lhz)].sort_values("lead")
            ax.plot(sub["lead"], sub["f1_weighted"], marker=CONFIG_MARKERS[cfg], color=CONFIG_COLORS[cfg],
                     label=CONFIG_LABELS[cfg], linewidth=2, markersize=6)
        ax.set_title(LHZ_LABELS[lhz])
        ax.set_xlabel("lead time (months)")
        ax.set_xticks(LEADS)
    axes[0].set_ylabel("weighted F1")
    axes[0].set_ylim(0, 1.0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=9)
    fig.suptitle("RQ1 Experiment 1 -- weighted F1 by lead time and livelihood zone\n2020-2024 test period", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "f1_by_leadtime_lhz.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote f1_by_leadtime_lhz.png")


if __name__ == "__main__":
    main()
