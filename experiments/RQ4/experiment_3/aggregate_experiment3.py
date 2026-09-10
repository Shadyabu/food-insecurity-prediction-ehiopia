"""RQ4 Experiment 3 -- aggregation and H0 test for local, leave-top-1-out
SHAP stability.

Reads local_stability_{xgboost,randomforest,lstm}.csv (one row per
test instance, each with a Spearman rho between its pre- and post-mask
ranking of its own remaining features -- see run_experiment3_trees.py /
run_experiment3_lstm.py for the design).

Ensemble reading: this project's own row groupings differ structurally
across subjects (XGBoost/RandomForest: 21 livelihood-cluster x lead
cells; LSTM: 7 lead-pooled-across-clusters cells) -- the same mismatch
Experiment 2's own docstring already documents and explicitly declines to
reconcile row-for-row ("this experiment measures whether EACH
architecture's own global ranking is stable... not a shared row-level
draw"). Consistent with that precedent, the "ensemble" reading here is
the n-weighted average of the three subjects' own pooled mean
correlations, not a per-row joint fusion requiring a shared masked
feature across all three models simultaneously -- a disclosed
simplification, not a silent one.

H0 (same threshold as Experiment 2, for direct comparability): the mean
per-row rank correlation does not exceed 0.7.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).resolve().parent
CI_BOOTSTRAP_ITERS = 2000
SEED = 42
THRESHOLD = 0.7


def bootstrap_ci_of_mean(values, n_iters=CI_BOOTSTRAP_ITERS, seed=SEED):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_iters)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def summarize(df, subject):
    valid = df["rho"].dropna()
    n_total, n_valid = len(df), len(valid)
    mean_rho = float(valid.mean())
    lo, hi = bootstrap_ci_of_mean(valid.to_numpy())
    reject_h0 = lo > THRESHOLD
    print(f"\n=== {subject} ===")
    print(f"n rows: {n_total} (valid rho: {n_valid}, "
          f"{n_total - n_valid} undefined -- constant ranking pre or post mask)")
    print(f"mean rho: {mean_rho:.4f}  95% CI: [{lo:.4f}, {hi:.4f}]  "
          f"reject H0 (rho>{THRESHOLD}): {reject_h0}")

    by_cell_cols = [c for c in ["cluster", "lead"] if c in df.columns]
    if by_cell_cols:
        by_cell = df.groupby(by_cell_cols)["rho"].agg(["mean", "count"])
        print(by_cell.to_string())
    else:
        by_lead = df.groupby("lead")["rho"].agg(["mean", "count"])
        print(by_lead.to_string())

    return {"subject": subject, "n_total": n_total, "n_valid": n_valid,
            "mean_rho": mean_rho, "ci_lo": lo, "ci_hi": hi, "reject_h0": reject_h0}


def main():
    xgb = pd.read_csv(OUT_DIR / "local_stability_xgboost.csv")
    rf = pd.read_csv(OUT_DIR / "local_stability_randomforest.csv")
    lstm = pd.read_csv(OUT_DIR / "local_stability_lstm.csv")

    summaries = [
        summarize(xgb, "xgboost"),
        summarize(rf, "randomforest"),
        summarize(lstm, "lstm"),
    ]

    # Ensemble: n-weighted average of the three subjects' own pooled means
    # (disclosed simplification -- see module docstring).
    total_n = sum(s["n_valid"] for s in summaries)
    ens_mean = sum(s["mean_rho"] * s["n_valid"] for s in summaries) / total_n
    all_rho = pd.concat([xgb["rho"].dropna(), rf["rho"].dropna(), lstm["rho"].dropna()])
    lo, hi = bootstrap_ci_of_mean(all_rho.to_numpy())
    print(f"\n=== ensemble (n-weighted average of the three subjects' own "
          f"pooled means; NOT a row-level joint fusion -- see docstring) ===")
    print(f"n-weighted mean rho: {ens_mean:.4f}")
    print(f"pooled-rows CI: [{lo:.4f}, {hi:.4f}]  reject H0 (rho>{THRESHOLD}): {lo > THRESHOLD}")

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "local_stability_summary.csv", index=False)
    print("\nWrote local_stability_summary.csv")


if __name__ == "__main__":
    main()
