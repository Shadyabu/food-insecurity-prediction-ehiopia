"""RQ4 Experiment 2 -- bootstrap stability of SHAP feature-importance
rankings.

H0: the mean pairwise Spearman rank correlation of SHAP feature-importance
rankings across bootstrap resamples of the held-out test set does not
exceed 0.7.

Three subjects (see ../common.py module docstring for the full scope
decisions confirmed with the project owner 2026-08-27): XGBoost alone,
RandomForest alone, and the ensemble's SHAP-ranking layer (XGBoost +
RandomForest + LSTM -- TabICLv2 excluded from this layer only, its own
permutation SHAP costs ~500 days at this bootstrap count; see common.py).

Bootstrap design: 100 draws, each cell (XGBoost/RandomForest: one of the 21
livelihood-cluster x lead cells; LSTM: one of the 7 pooled-per-lead cells)
resampled INDEPENDENTLY to its own original size, with replacement, using a
reproducible RNG stream seeded from (SEED, bootstrap_iter, cell_key). Cells
are NOT resampled in lockstep by shared (time, zone_code) key across
architectures -- XGBoost/RandomForest's per-cluster row ordering and LSTM's
lead-pooled-across-clusters ordering are different groupings of the same
underlying test set, and reconciling them into one identical draw isn't
needed here: this experiment measures whether EACH architecture's own
global ranking is stable under resampling, and the ensemble ranking for
bootstrap b is the b-th independent draw's fused XGBoost+RandomForest+LSTM
ranking -- consistent bookkeeping (same b), not a shared row-level draw.

Cost note: XGBoost/RandomForest TreeExplainer is fast (~1s/cell), so
100 x 21 cells finishes in ~25-35 min each. LSTM uses GradientExplainer via
a subprocess (segfaults if run in-process alongside xgboost -- see
common.py/lstm_worker.py) batched internally over all 100 draws per lead
(one subprocess call per lead reuses the loaded model + explainer
background across draws, not once per draw) -- this is the dominant cost.
The original 2020-2022 run's *actual* wall time was 482.4 min (~8.0 hours,
not the ~2-3 hour estimate above -- see experiment_2/report.md) -- this
2020-2024 rebuild (this file) has ~1.67x the test rows (1380 vs. 828) and
should be expected to take correspondingly longer, likely several hours
more than the original run, not less.

--- 2020-2024 REBUILD (2026-09-02) ---
Copy of run_experiment2.py with common.set_window("extended") set before
any cell is built, and OUT_DIR pointed at outputs_2020_2024/ so the
original 2020-2022 outputs/report.md are untouched. No other logic
changed -- same subjects, same models (no retraining needed, see
run_experiment1_2020_2024.py's docstring), same bootstrap design.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as c

c.set_window("extended")

N_BOOTSTRAP = 100
CI_BOOTSTRAP_ITERS = 2000
TOP_N_STABILITY = 10
OUT_DIR = Path(__file__).resolve().parent / "outputs_2020_2024"
OUT_DIR.mkdir(exist_ok=True)


def bootstrap_row_idx(n, seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=n)


def build_xgb_rf_draws(seed_base):
    """Returns dict: {(cluster, lead): [row_idx array, one per bootstrap]}."""
    draws = {}
    for cluster in c.CLUSTERS:
        for lead in c.LEADS:
            n = len(c.get_tabular_cell(cluster, lead, impute=False)["test_x"])
            draws[(cluster, lead)] = [
                bootstrap_row_idx(n, hash((seed_base, "xgbrf", cluster, lead, b)) % (2**32))
                for b in range(N_BOOTSTRAP)
            ]
    return draws


def build_lstm_draws(seed_base):
    draws = {}
    for lead in c.LEADS:
        n = c.lstm_test_size(lead)
        draws[lead] = [
            bootstrap_row_idx(n, hash((seed_base, "lstm", lead, b)) % (2**32))
            for b in range(N_BOOTSTRAP)
        ]
    return draws


def xgb_bootstrap_importances(draws_by_cell):
    """Returns DataFrame: index=feature, columns=bootstrap_iter (0..99)."""
    cols = {}
    for b in range(N_BOOTSTRAP):
        row_idx_by_cell = {k: v[b] for k, v in draws_by_cell.items()}
        cols[b] = c.xgb_global_ranking(row_idx_by_cell)
    return pd.DataFrame(cols)


def rf_bootstrap_importances(draws_by_cell):
    cols = {}
    for b in range(N_BOOTSTRAP):
        row_idx_by_cell = {k: v[b] for k, v in draws_by_cell.items()}
        cols[b] = c.rf_global_ranking(row_idx_by_cell)
    return pd.DataFrame(cols)


def lstm_bootstrap_importances(draws_by_lead):
    """One subprocess call per lead (batched over all 100 draws), then
    combine across leads per bootstrap iteration (equal weights -- every
    lead's bootstrap draw is always exactly its own original size, 828)."""
    per_lead = {}
    for lead in c.LEADS:
        df = c.lstm_shap_importance_batch_by_lead(lead, draws_by_lead[lead])
        per_lead[lead] = df
    cols = {}
    for b in range(N_BOOTSTRAP):
        parts = []
        for lead in c.LEADS:
            sub = per_lead[lead][per_lead[lead]["bootstrap_iter"] == b]
            parts.append(pd.Series(sub["importance"].to_numpy(), index=sub["feature"]))
        cols[b] = pd.concat(parts, axis=1).mean(axis=1)
    return pd.DataFrame(cols)


def combined_tree_bootstrap(xgb_imps, rf_imps):
    """xgb_imps/rf_imps: DataFrame(feature x bootstrap). Returns
    DataFrame(feature x bootstrap) of rank-percentile-averaged importance."""
    common_feats = sorted(set(xgb_imps.index) & set(rf_imps.index))
    cols = {}
    for b in xgb_imps.columns:
        px = c.rank_percentile(xgb_imps.loc[common_feats, b])
        pr = c.rank_percentile(rf_imps.loc[common_feats, b])
        cols[b] = (px + pr) / 2
    return pd.DataFrame(cols)


def combined_ensemble_bootstrap(tree_imps, lstm_imps):
    cols = {}
    all_names = sorted(set(tree_imps.index) | set(lstm_imps.index))
    for b in tree_imps.columns:
        tree_pct = c.rank_percentile(tree_imps[b])
        lstm_pct = c.rank_percentile(lstm_imps[b])
        combined = pd.Series(index=all_names, dtype=float)
        for name in all_names:
            vals = []
            if name in tree_pct.index:
                vals.append(tree_pct[name])
            if name in lstm_pct.index:
                vals.append(lstm_pct[name])
            combined[name] = np.mean(vals)
        cols[b] = combined
    return pd.DataFrame(cols)


def pairwise_spearman(imp_df):
    """imp_df: DataFrame(feature x bootstrap_iter). Returns the (100x100)
    Spearman correlation matrix and the flat array of upper-triangle
    (excluding diagonal) pairwise values."""
    corr = imp_df.corr(method="spearman")
    iu = np.triu_indices_from(corr.to_numpy(), k=1)
    pairwise = corr.to_numpy()[iu]
    return corr, pairwise


def bootstrap_ci_of_mean(values, n_iters=CI_BOOTSTRAP_ITERS, seed=0):
    rng = np.random.default_rng(seed)
    means = np.empty(n_iters)
    n = len(values)
    for i in range(n_iters):
        idx = rng.integers(0, n, size=n)
        means[i] = values[idx].mean()
    return means.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)


def top_n_stability_table(imp_df, top_n=TOP_N_STABILITY):
    counts = pd.Series(0, index=imp_df.index)
    for b in imp_df.columns:
        top_feats = imp_df[b].sort_values(ascending=False).index[:top_n]
        counts.loc[top_feats] += 1
    counts = counts.sort_values(ascending=False)
    table = counts[counts > 0].to_frame("times_in_top10")
    table["pct_of_bootstraps"] = 100 * table["times_in_top10"] / imp_df.shape[1]
    return table


def summarize_subject(name, imp_df):
    corr, pairwise = pairwise_spearman(imp_df)
    mean_rho, ci_lo, ci_hi = bootstrap_ci_of_mean(pairwise)
    stability = top_n_stability_table(imp_df)
    reject = ci_lo > 0.7
    return {
        "subject": name, "n_features": imp_df.shape[0], "n_bootstrap": imp_df.shape[1],
        "n_pairs": len(pairwise), "mean_rho": mean_rho, "ci_lo_95": ci_lo, "ci_hi_95": ci_hi,
        "reject_h0_mean_rho_gt_0.7": reject,
    }, corr, pairwise, stability


def main():
    t_start = time.time()
    print("Generating bootstrap draws...")
    xgb_rf_draws = build_xgb_rf_draws(seed_base=c.SEED)
    lstm_draws = build_lstm_draws(seed_base=c.SEED)

    print(f"\n[{(time.time()-t_start)/60:.1f} min] Computing XGBoost bootstrap SHAP rankings ({N_BOOTSTRAP} x 21 cells)...")
    xgb_imps = xgb_bootstrap_importances(xgb_rf_draws)
    xgb_imps.to_csv(OUT_DIR / "bootstrap_importances_xgboost.csv")

    print(f"[{(time.time()-t_start)/60:.1f} min] Computing RandomForest bootstrap SHAP rankings...")
    rf_imps = rf_bootstrap_importances(xgb_rf_draws)
    rf_imps.to_csv(OUT_DIR / "bootstrap_importances_randomforest.csv")

    print(f"[{(time.time()-t_start)/60:.1f} min] Computing LSTM bootstrap SHAP rankings (7 subprocess calls, batched)...")
    lstm_imps = lstm_bootstrap_importances(lstm_draws)
    lstm_imps.to_csv(OUT_DIR / "bootstrap_importances_lstm.csv")

    print(f"[{(time.time()-t_start)/60:.1f} min] Combining ensemble ranking (XGB+RF+LSTM) per bootstrap...")
    tree_imps = combined_tree_bootstrap(xgb_imps, rf_imps)
    ensemble_imps = combined_ensemble_bootstrap(tree_imps, lstm_imps)
    ensemble_imps.to_csv(OUT_DIR / "bootstrap_importances_ensemble.csv")

    print(f"\n[{(time.time()-t_start)/60:.1f} min] Summarizing...")
    summary_rows = []
    corrs = {}
    pairwises = {}
    stability_tables = {}
    for name, imp_df in [("XGBoost", xgb_imps), ("RandomForest", rf_imps), ("Ensemble", ensemble_imps)]:
        row, corr, pairwise, stability = summarize_subject(name, imp_df)
        summary_rows.append(row)
        corrs[name] = corr
        pairwises[name] = pairwise
        stability_tables[name] = stability
        stability.to_csv(OUT_DIR / f"top10_stability_{name.lower()}.csv")
        np.save(OUT_DIR / f"pairwise_rho_{name.lower()}.npy", pairwise)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "stability_summary.csv", index=False)

    print("\n" + "=" * 70)
    print("STABILITY SUMMARY (one-sided test: mean rho > 0.7)")
    print("=" * 70)
    print(summary_df.to_string(index=False))

    print(f"\nTotal wall time: {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    main()
