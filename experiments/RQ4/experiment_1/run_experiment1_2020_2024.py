"""RQ4 Experiment 1, rebuilt on the 2020-2024 window -- copy of
run_experiment1.py with common.set_window("extended") set before any
tabular/LSTM/TabICLv2 cell is built, and OUT_DIR pointed at a separate
outputs_2020_2024/ subdirectory so the original 2020-2022 outputs (and
report.md, which cites them) are untouched. See that file's own docstring
for the full method -- unchanged here except for the window.

Same three subjects, same models (no retraining -- the saved XGBoost/
RandomForest model objects only depend on the training set, identical
between windows; LSTM's saved weights are likewise train-window-only).
Scope: pooled across all 7 leads, all 3 livelihood-zone clusters, on
2020-01..2024-12 instead of 2020-01..2022-12.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as c

c.set_window("extended")

K_PERCENTS = [10, 25, 50]
OUT_DIR = Path(__file__).resolve().parent / "outputs_2020_2024"
OUT_DIR.mkdir(exist_ok=True)
KEY = ["time", "zone_code", "lead"]


def xgb_predictions_all(mask_cols=None):
    frames = []
    for cluster in c.CLUSTERS:
        for lead in c.LEADS:
            cell = c.get_tabular_cell(cluster, lead, impute=False)
            model = c.load_xgb_model(cluster, lead)
            X = cell["test_x"] if mask_cols is None else c.mask_nan(cell["test_x"], mask_cols)
            pred = c.predict_delta(model, X, cell["base1_test"])
            df = cell["test_meta"][["time", "zone_code"]].copy()
            df["lead"] = lead
            df["xgb_pred"] = pred
            df["observed"] = cell["test_y"].to_numpy()
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def rf_predictions_all(mask_cols=None):
    frames = []
    for cluster in c.CLUSTERS:
        for lead in c.LEADS:
            cell = c.get_tabular_cell(cluster, lead, impute=True)
            model = c.load_rf_model(cluster, lead)
            X = cell["test_x"] if mask_cols is None else c.mask_median(cell["test_x"], mask_cols, cell["train_medians"])
            pred = c.predict_delta(model, X, cell["base1_test"])
            df = cell["test_meta"][["time", "zone_code"]].copy()
            df["lead"] = lead
            df["rf_pred"] = pred
            df["observed"] = cell["test_y"].to_numpy()
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def lstm_predictions_all(mask_cols=None):
    frames = [c.lstm_predictions(lead, mask_cols=mask_cols) for lead in c.LEADS]
    return pd.concat(frames, ignore_index=True)


def tabicl_predictions_all(mask_cols=None):
    frames = [c.tabicl_predictions(lead, mask_cols=mask_cols) for lead in c.LEADS]
    return pd.concat(frames, ignore_index=True)


def per_lead_f1(pred_df, pred_col, observed_col="observed", class_valued=False):
    rows = []
    for lead, g in pred_df.groupby("lead"):
        obs_class = g[observed_col].to_numpy() if class_valued else c.to_ipc_class(g[observed_col])
        pred_class = g[pred_col].to_numpy() if class_valued else c.to_ipc_class(g[pred_col])
        rows.append({"lead": lead, "n": len(g), "f1": c.weighted_f1(obs_class, pred_class)})
    return pd.DataFrame(rows).sort_values("lead").reset_index(drop=True)


def ensemble_frame(xgb_df, rf_df, lstm_df, tabicl_df):
    m = xgb_df.set_index(KEY)[["xgb_pred", "observed"]]
    m = m.join(rf_df.set_index(KEY)[["rf_pred"]], how="inner")
    m = m.join(lstm_df.set_index(KEY)[["lstm_pred"]], how="inner")
    m = m.join(tabicl_df.set_index(KEY)[["tabicl_pred"]], how="inner")
    m["ensemble_pred"] = m[["xgb_pred", "rf_pred", "lstm_pred", "tabicl_pred"]].mean(axis=1)
    return m.reset_index()


def main():
    t_start = time.time()
    print("=" * 70)
    print("Building global SHAP rankings (once, on the full test set)")
    print("=" * 70)
    xgb_imp = c.xgb_global_ranking()
    print(f"  XGBoost ranking: {len(xgb_imp)} features")
    rf_imp = c.rf_global_ranking()
    print(f"  RandomForest ranking: {len(rf_imp)} features")
    lstm_imp = c.lstm_global_ranking()
    print(f"  LSTM ranking: {len(lstm_imp)} features")
    ens_imp = c.combined_ensemble_ranking(xgb_imp, rf_imp, lstm_imp)
    print(f"  Combined ensemble ranking: {len(ens_imp)} features")

    xgb_imp_sorted = xgb_imp.sort_values(ascending=False)
    rf_imp_sorted = rf_imp.sort_values(ascending=False)

    xgb_imp_sorted.to_csv(OUT_DIR / "shap_ranking_xgboost.csv", header=["mean_abs_shap"])
    rf_imp_sorted.to_csv(OUT_DIR / "shap_ranking_randomforest.csv", header=["mean_abs_shap"])
    ens_imp.to_csv(OUT_DIR / "shap_ranking_ensemble.csv", header=["combined_rank_percentile"])

    print("\nBuilding baseline (unmasked) predictions for all 4 members...")
    xgb_base = xgb_predictions_all()
    rf_base = rf_predictions_all()
    lstm_base = lstm_predictions_all()
    tabicl_base = tabicl_predictions_all()

    baseline_xgb_f1 = per_lead_f1(xgb_base, "xgb_pred")
    baseline_rf_f1 = per_lead_f1(rf_base, "rf_pred")
    ens_base = ensemble_frame(xgb_base, rf_base, lstm_base, tabicl_base)
    baseline_ens_f1 = per_lead_f1(ens_base, "ensemble_pred")

    print(f"  XGBoost baseline F1 by lead:\n{baseline_xgb_f1}")
    print(f"  RandomForest baseline F1 by lead:\n{baseline_rf_f1}")
    print(f"  Ensemble baseline F1 by lead:\n{baseline_ens_f1}")

    results = []
    for k_pct in K_PERCENTS:
        k_xgb = max(1, round(k_pct / 100 * len(xgb_imp_sorted)))
        k_rf = max(1, round(k_pct / 100 * len(rf_imp_sorted)))
        k_ens = max(1, round(k_pct / 100 * len(ens_imp)))

        top_xgb = list(xgb_imp_sorted.index[:k_xgb])
        bot_xgb = list(xgb_imp_sorted.index[-k_xgb:])
        top_rf = list(rf_imp_sorted.index[:k_rf])
        bot_rf = list(rf_imp_sorted.index[-k_rf:])
        top_ens = list(ens_imp.index[:k_ens])
        bot_ens = list(ens_imp.index[-k_ens:])

        print(f"\n--- k={k_pct}% (XGB k={k_xgb}, RF k={k_rf}, Ensemble k={k_ens}) ---")

        t0 = time.time()
        xgb_top_f1 = per_lead_f1(xgb_predictions_all(top_xgb), "xgb_pred")
        xgb_bot_f1 = per_lead_f1(xgb_predictions_all(bot_xgb), "xgb_pred")
        print(f"  XGBoost top/bottom done in {time.time()-t0:.1f}s")

        t0 = time.time()
        rf_top_f1 = per_lead_f1(rf_predictions_all(top_rf), "rf_pred")
        rf_bot_f1 = per_lead_f1(rf_predictions_all(bot_rf), "rf_pred")
        print(f"  RandomForest top/bottom done in {time.time()-t0:.1f}s")

        t0 = time.time()
        xgb_top_e = xgb_predictions_all(top_ens)
        rf_top_e = rf_predictions_all(top_ens)
        lstm_top_e = lstm_predictions_all(top_ens)
        tabicl_top_e = tabicl_predictions_all(top_ens)
        ens_top_f1 = per_lead_f1(ensemble_frame(xgb_top_e, rf_top_e, lstm_top_e, tabicl_top_e), "ensemble_pred")

        xgb_bot_e = xgb_predictions_all(bot_ens)
        rf_bot_e = rf_predictions_all(bot_ens)
        lstm_bot_e = lstm_predictions_all(bot_ens)
        tabicl_bot_e = tabicl_predictions_all(bot_ens)
        ens_bot_f1 = per_lead_f1(ensemble_frame(xgb_bot_e, rf_bot_e, lstm_bot_e, tabicl_bot_e), "ensemble_pred")
        print(f"  Ensemble top/bottom done in {time.time()-t0:.1f}s")

        for subject, base_f1, top_f1, bot_f1, k_used in [
            ("XGBoost", baseline_xgb_f1, xgb_top_f1, xgb_bot_f1, k_xgb),
            ("RandomForest", baseline_rf_f1, rf_top_f1, rf_bot_f1, k_rf),
            ("Ensemble", baseline_ens_f1, ens_top_f1, ens_bot_f1, k_ens),
        ]:
            merged = base_f1.merge(top_f1, on="lead", suffixes=("", "_top")).merge(bot_f1, on="lead", suffixes=("", "_bot"))
            for _, row in merged.iterrows():
                results.append({
                    "subject": subject, "k_pct": k_pct, "k_n": k_used, "lead": row["lead"],
                    "n": row["n"], "baseline_f1": row["f1"],
                    "top_f1": row["f1_top"], "top_degradation": row["f1"] - row["f1_top"],
                    "bottom_f1": row["f1_bot"], "bottom_degradation": row["f1"] - row["f1_bot"],
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "per_lead_degradation.csv", index=False)

    # Pooled (mean-over-leads) table, k x subject.
    pooled = results_df.groupby(["subject", "k_pct"]).agg(
        k_n=("k_n", "first"),
        baseline_f1=("baseline_f1", "mean"),
        top_f1=("top_f1", "mean"), top_degradation=("top_degradation", "mean"),
        bottom_f1=("bottom_f1", "mean"), bottom_degradation=("bottom_degradation", "mean"),
    ).reset_index()
    pooled.to_csv(OUT_DIR / "pooled_degradation_table.csv", index=False)
    print("\n" + "=" * 70)
    print("POOLED (mean over 7 leads) DEGRADATION TABLE")
    print("=" * 70)
    print(pooled.to_string(index=False))

    # --- Statistical tests -------------------------------------------------
    stat_rows = []
    for subject in ["XGBoost", "RandomForest", "Ensemble"]:
        sub = results_df[results_df["subject"] == subject]

        # Primary: paired across every (k, lead) combination, n=21.
        top_vals = sub["top_degradation"].to_numpy()
        bot_vals = sub["bottom_degradation"].to_numpy()
        try:
            stat, p = wilcoxon(top_vals, bot_vals, alternative="greater")
        except ValueError as e:
            stat, p = np.nan, np.nan
        stat_rows.append({
            "subject": subject, "test": "primary_all_k_lead_pairs", "n_pairs": len(top_vals),
            "mean_top_degradation": top_vals.mean(), "mean_bottom_degradation": bot_vals.mean(),
            "wilcoxon_stat": stat, "p_value": p, "reject_h0_at_0.05": (p < 0.05) if pd.notna(p) else np.nan,
        })

        # Secondary: pooled-across-leads per k, n=len(K_PERCENTS).
        pooled_sub = pooled[pooled["subject"] == subject]
        try:
            stat2, p2 = wilcoxon(pooled_sub["top_degradation"], pooled_sub["bottom_degradation"], alternative="greater")
        except ValueError:
            stat2, p2 = np.nan, np.nan
        stat_rows.append({
            "subject": subject, "test": "secondary_pooled_across_k", "n_pairs": len(pooled_sub),
            "mean_top_degradation": pooled_sub["top_degradation"].mean(),
            "mean_bottom_degradation": pooled_sub["bottom_degradation"].mean(),
            "wilcoxon_stat": stat2, "p_value": p2, "reject_h0_at_0.05": (p2 < 0.05) if pd.notna(p2) else np.nan,
        })

    stat_df = pd.DataFrame(stat_rows)
    stat_df.to_csv(OUT_DIR / "statistical_tests.csv", index=False)
    print("\n" + "=" * 70)
    print("STATISTICAL TESTS (one-sided Wilcoxon signed-rank, top > bottom degradation)")
    print("=" * 70)
    print(stat_df.to_string(index=False))

    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
