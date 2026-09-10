"""Closes the 1/2/3+ coverage gap flagged 2026-09-02: the Crisis-or-worse
(1/2/3+) discretization was only ever built for RQ1 Experiment 1
(Busker-architecture) and RQ1 Experiment 2 (Ethiopia XGBoost), per the
project owner's original 2026-08-24 request scoped to "both experiments".
The project owner has since confirmed every RQ2 model/ensemble should
carry the same 1/2/3+ table alongside its existing 1-5 numbers -- 1-5
remains what every RQ's stated conclusion is drawn from; 3+ is a
supplementary layer, matching how RQ1 already treats it (CLAUDE.md
2026-08-24 entry: "1/2/3+ discretization alongside the existing 1-5
scale").

Pure rescoring of already-computed predictions.parquet files -- no
retraining, no new experiments. Handles two prediction-file shapes seen
across RQ2:
  - continuous/already-discrete "observed"/"prediction" columns
    (XGBoost, RandomForest, LSTM, TabICLv2, all ensembles)
  - T-GCN's 0-indexed "true_class"/"pred_class" columns (never continuous)

Same discretization as experiments/RQ1/experiment_1/build_classification_metrics.py:
to_ipc_class = clip(round(x), 1, 5); to_ipc_class_3plus = clip(round(x), 1, 3).

Also pulls in Busker's architecture and both Ethiopia-XGBoost RQ1 variants
(rescored from their own predictions.parquet, not hardcoded) plus
Persistence and FEWS NET's outlook, so the emitted comparison table/CSVs
cover every model in the project, not only the RQ2 arms -- matches
build_1_2_3plus_comparison_figure.py's own coverage.
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


def load_scored(path, window=None, lhz_col="lhz"):
    df = pd.read_parquet(path)
    if window is not None and "window" in df.columns:
        df = df[df["window"] == window].copy()
    if "true_class" in df.columns:  # T-GCN: already discrete, 0-indexed
        observed = df["true_class"].to_numpy() + 1
        prediction = df["pred_class"].to_numpy() + 1
    else:
        observed = df["observed"].to_numpy()
        prediction = df["prediction"].to_numpy()
    return pd.DataFrame({
        "lead": df["lead"].to_numpy(),
        "lhz": df[lhz_col].to_numpy(),
        "observed": observed,
        "prediction": prediction,
    })


def score(df, discretize):
    y_true = discretize(df["observed"])
    y_pred = discretize(df["prediction"])
    return f1_score(y_true, y_pred, average="weighted", zero_division=0)


def build_table(df, label, lead_scheme="RQ2 grid (0,1,2,3,4,8,12)"):
    rows = []
    for lead in sorted(df["lead"].unique()):
        sub = df[df["lead"] == lead]
        if len(sub) == 0:
            continue
        rows.append({
            "config": label, "lead_scheme": lead_scheme, "lead": lead, "n": len(sub),
            "f1_1to5": score(sub, to_ipc_class),
            "f1_3plus": score(sub, to_ipc_class_3plus),
        })
    out = pd.DataFrame(rows)
    mean_row = {
        "config": label, "lead_scheme": lead_scheme, "lead": "mean", "n": out["n"].sum(),
        "f1_1to5": out["f1_1to5"].mean(), "f1_3plus": out["f1_3plus"].mean(),
    }
    out = pd.concat([out, pd.DataFrame([mean_row])], ignore_index=True)
    return out


def build_cluster_table(df, label):
    rows = []
    leads = sorted(df["lead"].unique())
    for cluster in CLUSTERS:
        csub = df[df["lhz"] == cluster]
        cf1_15 = [score(csub[csub["lead"] == lead], to_ipc_class)
                  for lead in leads if len(csub[csub["lead"] == lead]) > 0]
        cf1_3p = [score(csub[csub["lead"] == lead], to_ipc_class_3plus)
                  for lead in leads if len(csub[csub["lead"] == lead]) > 0]
        rows.append({
            "config": label, "cluster": cluster,
            "mean_f1_1to5": np.mean(cf1_15), "mean_f1_3plus": np.mean(cf1_3p),
        })
    return pd.DataFrame(rows)


def load_persistence(path, window=None, lhz_col="lhz"):
    df = pd.read_parquet(path)
    if window is not None and "window" in df.columns:
        df = df[df["window"] == window].copy()
    return pd.DataFrame({
        "lead": df["lead"].to_numpy(),
        "lhz": df[lhz_col].to_numpy(),
        "observed": df["observed"].to_numpy(),
        "prediction": df["base1_preds"].to_numpy(),
    })


def fews_pooled_table(csv_path, label):
    df = pd.read_csv(csv_path)
    az = df[df["cluster"] == "all_zones"].sort_values("lead_end")
    rows = [{"config": label, "lead_scheme": "FEWS ML1/ML2 lead_end (0-3=ML1, 4-7=ML2)",
              "lead": row.lead_end, "n": row.n, "f1_1to5": row.weighted_f1, "f1_3plus": row.weighted_f1_3plus}
            for row in az.itertuples()]
    out = pd.DataFrame(rows)
    mean_row = {"config": label, "lead_scheme": rows[0]["lead_scheme"] if rows else "", "lead": "mean",
                "n": out["n"].sum(), "f1_1to5": out["f1_1to5"].mean(), "f1_3plus": out["f1_3plus"].mean()}
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


# (label, path, window-or-None, lhz_col)
TARGETS = [
    ("LSTM-level (default)", "../experiment_1/lstm_level/predictions.parquet", "default", "lhz"),
    ("LSTM-level (extended)", "../experiment_1/lstm_level/predictions.parquet", "extended", "lhz"),
    ("LSTM-delta (default)", "../experiment_1/lstm_delta/predictions.parquet", "default", "lhz"),
    ("LSTM-delta (extended)", "../experiment_1/lstm_delta/predictions.parquet", "extended", "lhz"),
    ("TabICLv2 (default)", "../experiment_2/tabicl_level/predictions.parquet", "default", "lhz"),
    ("TabICLv2 (extended)", "../experiment_2/tabicl_level/predictions.parquet", "extended", "lhz"),
    ("RandomForest-delta-noclimate (default)", "../experiment_4/rf_delta_noclimate/predictions.parquet", None, "lhz"),
    ("RandomForest-delta-noclimate (extended)", "../experiment_4/rf_delta_noclimate_extended/predictions.parquet", None, "lhz"),
    ("T-GCN-CE (default)", "../experiment_3/tgcn_ce/predictions.parquet", "default", "lhz"),
    ("T-GCN-CE (extended)", "../experiment_3/tgcn_ce/predictions.parquet", "extended", "lhz"),
    ("Ensemble 2-way blend (XGB+RF)", "blend_predictions.parquet", None, "lhz"),
    ("Ensemble 5-way (all)", "full_ensemble_predictions.parquet", None, "lhz"),
    ("Ensemble 4-way champion (drop T-GCN)", "loo_drop_T-GCN_predictions.parquet", None, "lhz"),
]


def main():
    all_tables, all_cluster_tables = [], []
    for label, path, window, lhz_col in TARGETS:
        try:
            df = load_scored(path, window=window, lhz_col=lhz_col)
        except FileNotFoundError:
            print(f"SKIP (not found): {label} -> {path}")
            continue
        table = build_table(df, label)
        cluster_table = build_cluster_table(df, label)
        all_tables.append(table)
        all_cluster_tables.append(cluster_table)
        mean_row = table[table["lead"] == "mean"].iloc[0]
        print(f"{label:42s} mean F1(1-5)={mean_row['f1_1to5']:.4f}  mean F1(3+)={mean_row['f1_3plus']:.4f}")

    # RQ1's two models -- included here too so the comparison table/figure
    # covers everything, not just the RQ2 arms. Busker needs an Ethiopia
    # filter (its file pools Kenya/Somalia/Ethiopia).
    try:
        busker = pd.read_parquet("../../RQ1/experiment_1/busker_arch_datesplit_2020_2022/predictions.parquet")
        busker = busker[busker["country"] == "Ethiopia"]
        table = build_table(busker[["lead", "lhz", "observed", "prediction"]], "Busker architecture")
        all_tables.append(table)
        mean_row = table[table["lead"] == "mean"].iloc[0]
        print(f"{'Busker architecture':42s} mean F1(1-5)={mean_row['f1_1to5']:.4f}  mean F1(3+)={mean_row['f1_3plus']:.4f}")
    except FileNotFoundError as e:
        print(f"SKIP (not found): Busker architecture -> {e}")

    for label, path in [
        ("XGBoost, Ethiopia (plain baseline)", "../../RQ1/experiment_2/predictions/predictions.parquet"),
        ("XGBoost, Ethiopia (best variant)", "../../RQ1/experiment_2/unhcr_delta_noclimate/predictions.parquet"),
    ]:
        try:
            df = pd.read_parquet(path)[["lead", "lhz", "observed", "prediction"]]
        except FileNotFoundError:
            print(f"SKIP (not found): {label} -> {path}")
            continue
        table = build_table(df, label)
        all_tables.append(table)
        mean_row = table[table["lead"] == "mean"].iloc[0]
        print(f"{label:42s} mean F1(1-5)={mean_row['f1_1to5']:.4f}  mean F1(3+)={mean_row['f1_3plus']:.4f}")

    # Persistence -- same base1_preds input across XGB/RF/LSTM/TabICL (see
    # build_full_ensemble.py's own comment), default + extended windows.
    for label, path in [
        ("Persistence (default)", "../../RQ1/experiment_2/unhcr_delta_noclimate/predictions.parquet"),
        ("Persistence (extended)", "../../RQ1/experiment_2/unhcr_delta_noclimate_extended/predictions.parquet"),
    ]:
        try:
            df = load_persistence(path)
        except FileNotFoundError:
            print(f"SKIP (not found): {label} -> {path}")
            continue
        table = build_table(df, label)
        cluster_table = build_cluster_table(df, label)
        all_tables.append(table)
        all_cluster_tables.append(cluster_table)
        mean_row = table[table["lead"] == "mean"].iloc[0]
        print(f"{label:42s} mean F1(1-5)={mean_row['f1_1to5']:.4f}  mean F1(3+)={mean_row['f1_3plus']:.4f}")

    # FEWS NET's own outlook -- already-aggregated, different lead scheme
    # (ML1/ML2 lead_end, not the RQ2 0/1/2/3/4/8/12 grid) -- kept as its
    # own labeled rows rather than forced onto the RQ2 lead axis.
    for label, path in [
        ("FEWS NET outlook (2020-2024)", "../../RQ1/experiment_1/fews_outlook_f1_2020_2024.csv"),
        ("FEWS NET outlook (2026-only)", "../../RQ1/experiment_1/fews_outlook_f1_2026.csv"),
    ]:
        try:
            table = fews_pooled_table(path, label)
            cluster_table = fews_cluster_table(path, label)
        except FileNotFoundError:
            print(f"SKIP (not found): {label} -> {path}")
            continue
        all_tables.append(table)
        all_cluster_tables.append(cluster_table)
        mean_row = table[table["lead"] == "mean"].iloc[0]
        print(f"{label:42s} mean F1(1-5)={mean_row['f1_1to5']:.4f}  mean F1(3+)={mean_row['f1_3plus']:.4f}")

    pooled = pd.concat(all_tables, ignore_index=True)
    cluster = pd.concat(all_cluster_tables, ignore_index=True)
    pooled.to_csv("rq2_1_2_3plus_by_lead.csv", index=False)
    cluster.to_csv("rq2_1_2_3plus_by_cluster.csv", index=False)
    print("\nwrote rq2_1_2_3plus_by_lead.csv, rq2_1_2_3plus_by_cluster.csv")

    build_comparison_markdown(pooled)


def build_comparison_markdown(pooled):
    """Sorted, checkmarked comparison table -- this project's own models
    plus Persistence and FEWS NET side by side, F1(1-5) descending.
    Benchmarks: Persistence (default window) and FEWS NET outlook
    (2020-2024) -- the two windows that actually match every RQ2 model's
    own default-window reporting; the extended/2026-only rows are still in
    rq2_1_2_3plus_by_lead.csv but excluded here to keep one clean,
    like-for-like ranking rather than mixing windows in one table.
    """
    means = pooled[pooled["lead"] == "mean"].copy()
    exclude_windows = ["(extended)", "2026-only"]
    means = means[~means["config"].apply(lambda c: any(tok in c for tok in exclude_windows))]

    persist_row = means[means["config"] == "Persistence (default)"].iloc[0]
    fews_row = means[means["config"] == "FEWS NET outlook (2020-2024)"].iloc[0]
    persist_15, persist_3p = persist_row["f1_1to5"], persist_row["f1_3plus"]
    fews_15, fews_3p = fews_row["f1_1to5"], fews_row["f1_3plus"]

    means = means.sort_values("f1_1to5", ascending=False).reset_index(drop=True)

    def mark(cond):
        return "✓" if cond else "✗"

    lines = [
        "# Full model comparison: 1-5 vs. 1/2/3+, vs. Persistence and FEWS NET",
        "",
        "Auto-generated by `score_1_2_3plus_coverage.py` -- do not hand-edit,",
        "regenerate instead. Default window only (see module docstring for",
        "why extended/2026-only rows are excluded from this specific table).",
        "Sorted by F1(1-5) descending. Persistence and FEWS NET rows are the",
        "two benchmark rows this table checks every model against, not just",
        "two more entries in the ranking.",
        "",
        "| Model | F1 (1-5) | F1 (1/2/3+) | Beats persistence? | Beats FEWS NET? |",
        "|---|---|---|---|---|",
    ]
    for row in means.itertuples():
        is_benchmark = row.config in ("Persistence (default)", "FEWS NET outlook (2020-2024)")
        beats_p = "—" if is_benchmark else f"{mark(row.f1_1to5 > persist_15)} / {mark(row.f1_3plus > persist_3p)}"
        beats_f = "—" if is_benchmark else f"{mark(row.f1_1to5 > fews_15)} / {mark(row.f1_3plus > fews_3p)}"
        label = row.config.replace(" (default)", "")
        lines.append(f"| {label} | {row.f1_1to5:.4f} | {row.f1_3plus:.4f} | {beats_p} | {beats_f} |")
    lines += [
        "",
        "Beats-columns read as \"1-5 / 3+\" -- a model can clear one scale",
        "and not the other (see e.g. the 5-way ensemble in the by-lead CSV).",
    ]
    with open("rq2_1_2_3plus_comparison.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote rq2_1_2_3plus_comparison.md")


if __name__ == "__main__":
    main()
