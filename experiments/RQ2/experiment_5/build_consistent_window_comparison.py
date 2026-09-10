"""One consistent-test-window model comparison, prompted by finding (in
score_1_2_3plus_coverage.py's rq2_1_2_3plus_comparison.md) that the FEWS
NET row used a different window (2020-2024) from every other row
(2020-2022 default). Rather than build a FEWS-NET-on-2020-2022 number,
the project owner reframed the plan itself (2026-09-02): 2022-2026 was
the originally intended test window; it was pulled back to 2020-2024
because (1) 2025 ground truth is unavailable -- FEWS NET went offline for
~6 months in 2025 -- and (2) this project's own ACLED access is capped at
~12 months lagged, which the 2026-09-02 root-cause session showed
actively damages models that still carry conflict features into that gap.
2020-2024 is therefore now the standard window for cross-model
comparison, not 2020-2022 -- and FEWS NET's already-existing
fews_outlook_f1_2020_2024.csv turns out to already be the correct match,
no new FEWS NET build needed.

Busker is excluded per the project owner's explicit instruction: his own
released dataset (input_master.parquet) has a hard ceiling at 2022-12-01,
so no 2020-2024 Busker number can exist -- see
experiments/results/experiment_log.md's 2026-08-26 entry.

Every row is rescored from its own predictions.parquet (or, for FEWS NET,
its own already-scored CSV) -- no hardcoded numbers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def to_ipc_class_3plus(x):
    return np.clip(np.round(x), 1, 3).astype(int)


def score(y_obs, y_pred, discretize):
    return f1_score(discretize(y_obs), discretize(y_pred), average="weighted", zero_division=0)


def build_table(df, label):
    rows = []
    for lead in sorted(df["lead"].unique()):
        sub = df[df["lead"] == lead]
        if len(sub) == 0:
            continue
        rows.append({
            "config": label, "lead": lead, "n": len(sub),
            "f1_1to5": score(sub["observed"], sub["prediction"], to_ipc_class),
            "f1_3plus": score(sub["observed"], sub["prediction"], to_ipc_class_3plus),
        })
    out = pd.DataFrame(rows)
    mean_row = {"config": label, "lead": "mean", "n": out["n"].sum(),
                "f1_1to5": out["f1_1to5"].mean(), "f1_3plus": out["f1_3plus"].mean()}
    return pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)


def build_cluster_table(df, label):
    rows = []
    leads = sorted(df["lead"].unique())
    for cluster in CLUSTERS:
        csub = df[df["lhz"] == cluster]
        f15 = [score(csub[csub["lead"] == ld]["observed"], csub[csub["lead"] == ld]["prediction"], to_ipc_class)
               for ld in leads if len(csub[csub["lead"] == ld]) > 0]
        f3p = [score(csub[csub["lead"] == ld]["observed"], csub[csub["lead"] == ld]["prediction"], to_ipc_class_3plus)
               for ld in leads if len(csub[csub["lead"] == ld]) > 0]
        rows.append({"config": label, "cluster": cluster,
                      "mean_f1_1to5": np.mean(f15), "mean_f1_3plus": np.mean(f3p)})
    return pd.DataFrame(rows)


def load_window(path, window=None, lhz_col="lhz", is_tgcn=False):
    df = pd.read_parquet(path)
    if window is not None and "window" in df.columns:
        df = df[df["window"] == window].copy()
    if is_tgcn:
        observed = df["true_class"].to_numpy() + 1
        prediction = df["pred_class"].to_numpy() + 1
    else:
        observed = df["observed"].to_numpy()
        prediction = df["prediction"].to_numpy()
    return pd.DataFrame({"lead": df["lead"].to_numpy(), "lhz": df[lhz_col].to_numpy(),
                          "observed": observed, "prediction": prediction})


def load_persistence(path):
    df = pd.read_parquet(path)
    return pd.DataFrame({"lead": df["lead"].to_numpy(), "lhz": df["lhz"].to_numpy(),
                          "observed": df["observed"].to_numpy(), "prediction": df["base1_preds"].to_numpy()})


def fews_table(csv_path, label):
    df = pd.read_csv(csv_path)
    az = df[df["cluster"] == "all_zones"].sort_values("lead_end")
    rows = [{"config": label, "lead": row.lead_end, "n": row.n,
             "f1_1to5": row.weighted_f1, "f1_3plus": row.weighted_f1_3plus} for row in az.itertuples()]
    out = pd.DataFrame(rows)
    mean_row = {"config": label, "lead": "mean", "n": out["n"].sum(),
                "f1_1to5": out["f1_1to5"].mean(), "f1_3plus": out["f1_3plus"].mean()}
    return pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)


def fews_cluster_table(csv_path, label):
    df = pd.read_csv(csv_path)
    rows = []
    for cluster in CLUSTERS:
        csub = df[df["cluster"] == cluster]
        if len(csub) == 0:
            continue
        rows.append({"config": label, "cluster": cluster,
                      "mean_f1_1to5": csub["weighted_f1"].mean(), "mean_f1_3plus": csub["weighted_f1_3plus"].mean()})
    return pd.DataFrame(rows)


# All on the 2020-2024 window. (label, loader-args)
SOURCES = [
    ("XGBoost, Ethiopia (plain baseline)",
     lambda: load_window("../../RQ1/experiment_2/predictions_extended/predictions.parquet")),
    ("XGBoost, Ethiopia (best variant)",
     lambda: load_window("../../RQ1/experiment_2/unhcr_delta_noclimate_extended/predictions.parquet")),
    ("RandomForest (delta+noclimate)",
     lambda: load_window("../experiment_4/rf_delta_noclimate_extended/predictions.parquet")),
    ("LSTM (delta)",
     lambda: load_window("../experiment_1/lstm_delta/predictions.parquet", window="extended")),
    ("TabICLv2",
     lambda: load_window("../experiment_2/tabicl_level/predictions.parquet", window="extended")),
    ("T-GCN (CE)",
     lambda: load_window("../experiment_3/tgcn_ce/predictions.parquet", window="extended", is_tgcn=True)),
    ("Ensemble, 2-way (XGB+RF)",
     lambda: load_window("all_combinations_extended/predictions/XGB_RF.parquet")),
    ("Ensemble, 5-way (all architectures)",
     lambda: load_window("full_ensemble_predictions_extended.parquet")),
    ("Ensemble, 4-way champion (drop T-GCN)",
     lambda: load_window("all_combinations_extended/predictions/XGB_RF_LSTM_TabICL.parquet")),
    ("Persistence",
     lambda: load_persistence("../../RQ1/experiment_2/unhcr_delta_noclimate_extended/predictions.parquet")),
]


def main():
    all_tables, all_cluster_tables = [], []
    for label, loader in SOURCES:
        try:
            df = loader()
        except FileNotFoundError as e:
            print(f"SKIP (not found): {label} -> {e}")
            continue
        table = build_table(df, label)
        all_tables.append(table)
        if "lhz" in df.columns:
            all_cluster_tables.append(build_cluster_table(df, label))
        mean_row = table[table["lead"] == "mean"].iloc[0]
        print(f"{label:42s} mean F1(1-5)={mean_row['f1_1to5']:.4f}  mean F1(3+)={mean_row['f1_3plus']:.4f}")

    fews_path = "../../RQ1/experiment_1/fews_outlook_f1_2020_2024.csv"
    fews_lbl = "FEWS NET outlook (2020-2024)"
    table = fews_table(fews_path, fews_lbl)
    all_tables.append(table)
    all_cluster_tables.append(fews_cluster_table(fews_path, fews_lbl))
    mean_row = table[table["lead"] == "mean"].iloc[0]
    print(f"{fews_lbl:42s} mean F1(1-5)={mean_row['f1_1to5']:.4f}  mean F1(3+)={mean_row['f1_3plus']:.4f}")

    pooled = pd.concat(all_tables, ignore_index=True)
    cluster = pd.concat(all_cluster_tables, ignore_index=True)
    pooled.to_csv("consistent_window_2020_2024_by_lead.csv", index=False)
    cluster.to_csv("consistent_window_2020_2024_by_cluster.csv", index=False)
    print("\nwrote consistent_window_2020_2024_by_lead.csv, consistent_window_2020_2024_by_cluster.csv")

    build_markdown(pooled)


def build_markdown(pooled):
    means = pooled[pooled["lead"] == "mean"].sort_values("f1_1to5", ascending=False).reset_index(drop=True)
    persist = means[means["config"] == "Persistence"].iloc[0]
    fews = means[means["config"] == "FEWS NET outlook (2020-2024)"].iloc[0]

    def mark(cond):
        return "✓" if cond else "✗"

    lines = [
        "# Model comparison, single consistent test window: 2020-01 to 2024-12",
        "",
        "Auto-generated by `build_consistent_window_comparison.py` -- do not",
        "hand-edit, regenerate instead. Every row (Busker excluded -- his own",
        "dataset cannot reach past 2022-12) is scored on the exact same",
        "2020-2024 window, including FEWS NET's own outlook, which already",
        "used this window natively -- no window-mismatch caveat needed here,",
        "unlike the earlier default-window table.",
        "",
        "| Model | F1 (1-5) | F1 (1/2/3+) | Beats persistence? | Beats FEWS NET? |",
        "|---|---|---|---|---|",
    ]
    for row in means.itertuples():
        is_benchmark = row.config in ("Persistence", "FEWS NET outlook (2020-2024)")
        beats_p = "—" if is_benchmark else f"{mark(row.f1_1to5 > persist['f1_1to5'])} / {mark(row.f1_3plus > persist['f1_3plus'])}"
        beats_f = "—" if is_benchmark else f"{mark(row.f1_1to5 > fews['f1_1to5'])} / {mark(row.f1_3plus > fews['f1_3plus'])}"
        lines.append(f"| {row.config} | {row.f1_1to5:.4f} | {row.f1_3plus:.4f} | {beats_p} | {beats_f} |")
    lines += ["", "Beats-columns read as \"1-5 / 3+\"."]
    with open("consistent_window_2020_2024_comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote consistent_window_2020_2024_comparison.md")


if __name__ == "__main__":
    main()
