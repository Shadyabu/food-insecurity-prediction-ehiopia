"""
build_experiment1_comparison_tables.py

RQ1 Experiment 1 deliverable: a clean, presentation-ready set of tables
covering both halves the project owner asked for (2026-08-24):

1. REPRODUCTION (all of Horn of Africa, 213 units) -- already fully
   covered by experiment_1/figures/ (built earlier this project via
   experiment_1/build_figures.py, not rebuilt here). This script does not
   touch that half.
2. COMPARISON (Ethiopia only, same XGBoost architecture, same test
   period): Busker-architecture reproduction vs. the Ethiopia-enriched
   dataset -- both the plain (level-mode) addition and the best variant
   (delta target + no climate) -- weighted F1 and MAE, broken out by
   lead time and livelihood zone. This script builds that half.

**Updated 2026-08-26** (figure audit): now uses
`experiment_1/busker_arch_datesplit_2020_2022/predictions.parquet`
instead of the original `experiment_1/predictions/predictions.parquet`.
The original comparison used Busker's own POSITIONAL 80:20 slice
(train_test_split(shuffle=False), landing inside 2019-06 -- CLAUDE.md
Sec 7a), which was not the same split boundary as experiment_2's
date-based split (train<=2019-12, test 2020-01..2022-12) -- an
apples-to-oranges split methodology, not just a data-coverage
difference. `busker_arch_datesplit_2020_2022` (built 2026-08-26 via
`run_model.py --train-end 2019-12-31 --test-start 2020-01-01 --test-end
2022-12-31`) fixes that: both configs now share the identical
train<=2019/test-2020+ split boundary. A small residual date-range
difference remains, but it is now a genuine data-availability fact, not
a split-methodology mismatch: the Busker-architecture run's own target
data (`FEWS_CS`) has no observations after 2022-06 regardless of the
split boundary, while the enriched dataset's target extends to 2022-10 --
confirmed by inspecting each config's actual min/max test `time`. Still
restricted to the overlap (`COMMON_WINDOW`) below for a strictly
apples-to-apples scoring window.

Discretization is the standard 1-5 IPC-phase scale (np.round + clip),
matching every other classification-metric computation in this project.
Now also scores R2 (previously F1/MAE only), per the dissertation plan's
figure spec for the 2019-2022 comparison.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADS = [0, 1, 2, 3, 4, 8, 12]

COMMON_WINDOW = ("2020-02-01", "2022-06-01")  # overlap of Exp1's and Exp2's own test windows

LHZ_RENAME = {"p": "pastoral", "ap": "agropastoral", "other": "crop_farming"}
LHZ_ORDER = ["pastoral", "agropastoral", "crop_farming"]
LHZ_LABELS = {"pastoral": "Pastoral", "agropastoral": "Agro-pastoral", "crop_farming": "Crop farming"}

CONFIGS = [
    ("busker_arch", "Busker architecture (baseline)",
     os.path.join(SCRIPT_DIR, "experiment_1", "busker_arch_datesplit_2020_2022", "predictions.parquet"), True),
    ("enriched_level", "Ethiopia-enriched dataset, same architecture",
     os.path.join(SCRIPT_DIR, "experiment_2", "unhcr_level", "predictions.parquet"), False),
    ("enriched_best", "Ethiopia-enriched dataset, best variant (delta target + no climate)",
     os.path.join(SCRIPT_DIR, "experiment_2", "unhcr_delta_noclimate", "predictions.parquet"), False),
]

OUT_DIR = os.path.join(SCRIPT_DIR, "experiment_1_vs_2_comparison")


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_config(path, is_exp1):
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    if is_exp1:
        df = df[df["country"] == "Ethiopia"].copy()
        df["lhz"] = df["lhz"].map(LHZ_RENAME)
    df = df[(df["time"] >= COMMON_WINDOW[0]) & (df["time"] <= COMMON_WINDOW[1])]
    return df


def score(df, group_cols):
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        yt, yp = to_ipc_class(g["observed"]), to_ipc_class(g["prediction"])
        rows.append({
            **dict(zip(group_cols, keys)),
            "n": len(g),
            "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
            "mae": mean_absolute_error(g["observed"], g["prediction"]),
            "r2": r2_score(g["observed"], g["prediction"]) if g["observed"].nunique() > 1 else np.nan,
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    loaded = {key: load_config(path, is_exp1) for key, _, path, is_exp1 in CONFIGS}

    for key, df in loaded.items():
        print(f"{key}: n={len(df)}, time range {df['time'].min().date()} to {df['time'].max().date()}")

    # --- overall (one row per config) ---
    overall_rows = []
    for key, label, _, _ in CONFIGS:
        df = loaded[key]
        yt, yp = to_ipc_class(df["observed"]), to_ipc_class(df["prediction"])
        overall_rows.append({
            "config": key, "label": label, "n": len(df),
            "mean_f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
            "mean_mae": mean_absolute_error(df["observed"], df["prediction"]),
            "mean_r2": r2_score(df["observed"], df["prediction"]) if df["observed"].nunique() > 1 else np.nan,
        })
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(os.path.join(OUT_DIR, "overall.csv"), index=False)
    print("\n=== Overall (common test window, Ethiopia only) ===")
    print(overall.to_string(index=False))

    # --- by lead time ---
    by_lead_frames = []
    for key, label, _, _ in CONFIGS:
        t = score(loaded[key], ["lead"])
        t["config"] = key
        t["label"] = label
        by_lead_frames.append(t)
    by_lead = pd.concat(by_lead_frames, ignore_index=True)
    by_lead = by_lead[["config", "label", "lead", "n", "f1_weighted", "mae", "r2"]].sort_values(["lead", "config"])
    by_lead.to_csv(os.path.join(OUT_DIR, "by_lead.csv"), index=False)
    print("\n=== By lead time ===")
    print(by_lead.to_string(index=False))

    # --- by livelihood zone ---
    by_lhz_frames = []
    for key, label, _, _ in CONFIGS:
        t = score(loaded[key], ["lhz"])
        t["config"] = key
        t["label"] = label
        by_lhz_frames.append(t)
    by_lhz = pd.concat(by_lhz_frames, ignore_index=True)
    by_lhz["lhz"] = pd.Categorical(by_lhz["lhz"], categories=LHZ_ORDER, ordered=True)
    by_lhz = by_lhz[["config", "label", "lhz", "n", "f1_weighted", "mae", "r2"]].sort_values(["lhz", "config"])
    by_lhz.to_csv(os.path.join(OUT_DIR, "by_lhz.csv"), index=False)
    print("\n=== By livelihood zone ===")
    print(by_lhz.to_string(index=False))

    # --- by lead time x livelihood zone (finest grain, for reference) ---
    by_both_frames = []
    for key, label, _, _ in CONFIGS:
        t = score(loaded[key], ["lhz", "lead"])
        t["config"] = key
        t["label"] = label
        by_both_frames.append(t)
    by_both = pd.concat(by_both_frames, ignore_index=True)
    by_both.to_csv(os.path.join(OUT_DIR, "by_lead_and_lhz.csv"), index=False)

    print(f"\nWrote overall.csv, by_lead.csv, by_lhz.csv, by_lead_and_lhz.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
