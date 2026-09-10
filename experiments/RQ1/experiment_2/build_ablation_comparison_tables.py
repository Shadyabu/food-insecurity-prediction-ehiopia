"""
build_ablation_comparison_tables.py

RQ1 Experiment 2 deliverable: a clean, presentation-ready set of tables
for the domain feature-cluster ablation (report.md Sec 6/12) -- the
project owner's own framing: "train XGBoost with my dataset but remove a
certain cluster of features (climate, agriculture, economic, conflict),
compare the results using F1 score... to see which features are the most
relevant and add to interpretability."

Uses the CURRENT (UNHCR-included, conflict = ACLED+UNHCR) dataset --
`unhcr_level` (all clusters in) and the 4 `unhcr_ablation_no_*` runs, per
report.md Sec 12. Level-mode target throughout (matches Sec 6's own
"run against the level-mode baseline, not the delta variant" scoping).

Outputs, all Ethiopia (92 zones), no test-window restriction needed here
(single dataset throughout, unlike the Exp1-vs-2 comparison):
  - overall.csv: one row per variant, pooled weighted F1 + MAE
  - by_lead.csv: weighted F1 + MAE per variant per lead
  - pastoral_by_lead.csv: same, pastoral zones only
  - by_lead_and_lhz.csv: weighted F1 + MAE per variant, per lead, all 3
    livelihood zones (added 2026-08-26, figure audit -- matches the
    3-way split convention used everywhere else in this project)

**Added 2026-08-26 (figure audit)**: a `busker_arch` REFERENCE row/series
(`experiments/RQ1/experiment_1/busker_arch_datesplit_2020_2022`, the
date-matched Busker-architecture reproduction) -- the ablation comparison
previously had no Busker baseline anywhere, despite the dissertation
plan explicitly asking for it. Flagged as a REFERENCE (`is_reference=True`
in overall.csv), not a 6th ablation "variant" -- it does not participate
in the delta-from-baseline bar chart, which stays about the 4 feature-
cluster removals. Its own test data ends 2022-06 (ceiling of Busker's
released dataset), so its by-lead/by-lhz series only has as many points
as its own available rows per group -- not silently padded.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mean_absolute_error

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADS = [0, 1, 2, 3, 4, 8, 12]
LHZ_ORDER = ["pastoral", "agropastoral", "crop_farming"]
LHZ_RENAME = {"p": "pastoral", "ap": "agropastoral", "other": "crop_farming"}

VARIANTS = [
    ("baseline", "All clusters (baseline)", "unhcr_level"),
    ("no_climate", "No climate", "unhcr_ablation_no_climate"),
    ("no_agriculture", "No agriculture", "unhcr_ablation_no_agriculture"),
    ("no_economic", "No economic", "unhcr_ablation_no_economic"),
    ("no_conflict", "No conflict (ACLED+UNHCR)", "unhcr_ablation_no_conflict"),
]

BUSKER_KEY = "busker_arch"
BUSKER_LABEL = "Busker architecture (reference, test ends 2022-06)"
BUSKER_PATH = os.path.join(SCRIPT_DIR, "..", "experiment_1",
                            "busker_arch_datesplit_2020_2022", "predictions.parquet")

OUT_DIR = os.path.join(SCRIPT_DIR, "ablation_comparison")


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load(folder):
    df = pd.read_parquet(os.path.join(SCRIPT_DIR, folder, "predictions.parquet"))
    return df


def load_busker():
    df = pd.read_parquet(BUSKER_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df[df["country"] == "Ethiopia"].copy()
    df["lhz"] = df["lhz"].map(LHZ_RENAME)
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
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    loaded = {key: load(folder) for key, _, folder in VARIANTS}
    loaded[BUSKER_KEY] = load_busker()
    all_keys = [k for k, _, _ in VARIANTS] + [BUSKER_KEY]
    all_labels = {k: l for k, l, _ in VARIANTS}
    all_labels[BUSKER_KEY] = BUSKER_LABEL

    # --- overall ---
    overall_rows = []
    base_f1 = None
    for key in all_keys:
        df = loaded[key]
        yt, yp = to_ipc_class(df["observed"]), to_ipc_class(df["prediction"])
        f1 = f1_score(yt, yp, average="weighted", zero_division=0)
        if key == "baseline":
            base_f1 = f1
        overall_rows.append({
            "variant": key, "label": all_labels[key], "n": len(df),
            "mean_f1_weighted": f1,
            "mean_mae": mean_absolute_error(df["observed"], df["prediction"]),
            "delta_f1_vs_baseline": (f1 - base_f1) if key not in ("baseline", BUSKER_KEY) else (0.0 if key == "baseline" else np.nan),
            "is_reference": key == BUSKER_KEY,
        })
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(os.path.join(OUT_DIR, "overall.csv"), index=False)
    print("=== Overall (all Ethiopia zones, pooled) ===")
    print(overall.to_string(index=False))

    # --- by lead time, all zones ---
    by_lead_frames = []
    for key in all_keys:
        t = score(loaded[key], ["lead"])
        t["variant"] = key
        t["label"] = all_labels[key]
        by_lead_frames.append(t)
    by_lead = pd.concat(by_lead_frames, ignore_index=True)
    by_lead = by_lead[["variant", "label", "lead", "n", "f1_weighted", "mae"]].sort_values(["lead", "variant"])
    by_lead.to_csv(os.path.join(OUT_DIR, "by_lead.csv"), index=False)
    print("\n=== By lead time (all zones) ===")
    print(by_lead.to_string(index=False))

    # --- pastoral only, by lead (kept for backward compatibility) ---
    pastoral_frames = []
    for key in all_keys:
        sub = loaded[key][loaded[key]["lhz"] == "pastoral"]
        t = score(sub, ["lead"])
        t["variant"] = key
        t["label"] = all_labels[key]
        pastoral_frames.append(t)
    pastoral_by_lead = pd.concat(pastoral_frames, ignore_index=True)
    pastoral_by_lead = pastoral_by_lead[["variant", "label", "lead", "n", "f1_weighted", "mae"]].sort_values(["lead", "variant"])
    pastoral_by_lead.to_csv(os.path.join(OUT_DIR, "pastoral_by_lead.csv"), index=False)
    print("\n=== By lead time (pastoral only) ===")
    print(pastoral_by_lead.to_string(index=False))

    # --- pastoral overall ---
    pastoral_overall_rows = []
    for key in all_keys:
        sub = loaded[key][loaded[key]["lhz"] == "pastoral"]
        yt, yp = to_ipc_class(sub["observed"]), to_ipc_class(sub["prediction"])
        pastoral_overall_rows.append({
            "variant": key, "label": all_labels[key], "n": len(sub),
            "mean_f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
            "mean_mae": mean_absolute_error(sub["observed"], sub["prediction"]),
        })
    pastoral_overall = pd.DataFrame(pastoral_overall_rows)
    pastoral_overall.to_csv(os.path.join(OUT_DIR, "pastoral_overall.csv"), index=False)
    print("\n=== Pastoral, pooled across leads ===")
    print(pastoral_overall.to_string(index=False))

    # --- by lead time AND livelihood zone, all 3 zones (2026-08-26 fix) ---
    by_lhz_frames = []
    for key in all_keys:
        for lhz in LHZ_ORDER:
            sub = loaded[key][loaded[key]["lhz"] == lhz]
            t = score(sub, ["lead"])
            t["lhz"] = lhz
            t["variant"] = key
            t["label"] = all_labels[key]
            by_lhz_frames.append(t)
    by_lead_and_lhz = pd.concat(by_lhz_frames, ignore_index=True)
    by_lead_and_lhz = by_lead_and_lhz[["variant", "label", "lhz", "lead", "n", "f1_weighted", "mae"]]
    by_lead_and_lhz.to_csv(os.path.join(OUT_DIR, "by_lead_and_lhz.csv"), index=False)

    print(f"\nWrote overall.csv, by_lead.csv, pastoral_by_lead.csv, pastoral_overall.csv, "
          f"by_lead_and_lhz.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
