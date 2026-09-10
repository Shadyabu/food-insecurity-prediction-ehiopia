"""RQ2 Experiment 5 (supplementary, not one of the four architectures named
in docs/dissertation_plan.md's RQ2 question -- same "add a note, don't edit
the RQ" pattern as experiment_4's RandomForest arm): pairwise significance
tests between the strongest RQ2 architectures, motivated directly by a
proposed ensemble. Before building any ensemble, this checks whether the
per-cluster performance gaps that would justify one (RandomForest's nominal
overall edge over XGBoost; LSTM-delta's pastoral edge over XGBoost) are
distinguishable from zone-resampling noise, or are themselves noise-level
-- the same open question report.md already flagged for RF vs XGBoost
("statistically indistinguishable... not resolved") and never tested for
LSTM vs XGBoost.

Method: unit(zone)-level block bootstrap, same convention as
experiments/RQ1/experiment_1/bootstrap_acceptance.py -- resample zone_codes
with replacement (whole per-zone row set moved together, so within-zone
correlation across leads/months isn't broken into pseudo-independent rows),
recompute each model's mean-weighted-F1-across-leads on the resampled
multiset (identical statistic and discretization to
build_classification_metrics.py: nearest-integer IPC class, clipped to
[1,5]), take the paired difference. 500 iterations, seed 42.

All three models being compared score the *identical* 5,796 test rows (828
zone-months x 7 leads, confirmed by exact key/observed-value match before
writing this script) -- default-window Busker-parity split (train
<=2019-12, test 2020-01..2022-12), delta target mode throughout (the
already-established best target framing for both trees and the LSTM).

--- 2020-2024 REBUILD (2026-09-02) ---
Same comparisons/method, on the project's revised primary window. XGBoost/
RandomForest paths point at freshly-built _extended runs (same recipe,
--test-start 2020-01 --test-end 2024-12); LSTM/TabICLv2 select
window=="extended" instead of "default" from their existing prediction
files (both already cover this window, no rebuild needed for those two).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
N_ITER = 500
SEED = 42

MODELS = {
    "XGBoost-delta": "../../RQ1/experiment_2/rq2_delta_default_extended/predictions.parquet",
    "RandomForest-delta": "../experiment_4/rf_delta_default_extended/predictions.parquet",
    "LSTM-delta": "../experiment_1/lstm_delta/predictions.parquet",
    "TabICLv2": "../experiment_2/tabicl_level/predictions.parquet",
}

COMPARISONS = [
    ("RandomForest-delta", "XGBoost-delta", "all"),
    ("RandomForest-delta", "XGBoost-delta", "pastoral"),
    ("RandomForest-delta", "XGBoost-delta", "agropastoral"),
    ("RandomForest-delta", "XGBoost-delta", "crop_farming"),
    ("LSTM-delta", "XGBoost-delta", "pastoral"),
    ("LSTM-delta", "XGBoost-delta", "agropastoral"),
    ("LSTM-delta", "XGBoost-delta", "all"),
    ("TabICLv2", "LSTM-delta", "agropastoral"),
    ("TabICLv2", "XGBoost-delta", "pastoral"),
]


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_model(path):
    df = pd.read_parquet(path)
    if "window" in df.columns:
        df = df[df["window"] == "extended"].copy()
    df = df[["time", "zone_code", "lhz", "lead", "observed", "prediction"]].copy()
    df["time"] = pd.to_datetime(df["time"])
    return df


def mean_f1_across_leads(df):
    """df: rows for one model, one bootstrap draw. Returns mean weighted F1
    across LEADS, matching build_classification_metrics.py's summary_row."""
    vals = []
    for lead in LEADS:
        sub = df[df["lead"] == lead]
        if len(sub) == 0:
            continue
        y_true = to_ipc_class(sub["observed"])
        y_pred = to_ipc_class(sub["prediction"])
        vals.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    return float(np.mean(vals))


def bootstrap_diff(df_a, df_b, cluster, rng, n_iter=N_ITER):
    """df_a, df_b: full (all-lead) frames for model A / B, same key set.
    cluster: 'all' or an lhz value. Returns (observed_diff, draws)."""
    if cluster != "all":
        df_a = df_a[df_a["lhz"] == cluster]
        df_b = df_b[df_b["lhz"] == cluster]

    zones = df_a["zone_code"].unique()
    n_zones = len(zones)

    by_zone_a = {z: g for z, g in df_a.groupby("zone_code")}
    by_zone_b = {z: g for z, g in df_b.groupby("zone_code")}

    observed_diff = mean_f1_across_leads(df_a) - mean_f1_across_leads(df_b)

    draws = np.empty(n_iter)
    for i in range(n_iter):
        sample_zones = rng.choice(zones, size=n_zones, replace=True)
        pooled_a = pd.concat([by_zone_a[z] for z in sample_zones], axis=0)
        pooled_b = pd.concat([by_zone_b[z] for z in sample_zones], axis=0)
        draws[i] = mean_f1_across_leads(pooled_a) - mean_f1_across_leads(pooled_b)

    return observed_diff, draws, n_zones


def main():
    cache = {name: load_model(path) for name, path in MODELS.items()}

    # sanity: every model scores the same test keys (checked once already
    # in an ad-hoc shell probe before writing this script; re-asserted here
    # so a future re-run fails loudly if that stops being true).
    key_cols = ["time", "zone_code", "lead"]
    ref_keys = set(map(tuple, cache["XGBoost-delta"][key_cols].to_numpy()))
    for name, df in cache.items():
        keys = set(map(tuple, df[key_cols].to_numpy()))
        assert keys == ref_keys, f"{name} scores a different test set than XGBoost-delta"

    rng = np.random.default_rng(SEED)
    rows = []
    for model_a, model_b, cluster in COMPARISONS:
        obs_diff, draws, n_zones = bootstrap_diff(cache[model_a], cache[model_b], cluster, rng)
        boot_sd = draws.std(ddof=1)
        ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5])
        p_two_sided = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
        p_two_sided = min(p_two_sided, 1.0)
        rows.append({
            "model_a": model_a, "model_b": model_b, "cluster": cluster, "n_zones": n_zones,
            "observed_diff_f1": round(obs_diff, 4),
            "bootstrap_mean_diff": round(draws.mean(), 4),
            "bootstrap_sd": round(boot_sd, 4),
            "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4),
            "p_two_sided_approx": round(p_two_sided, 4),
            "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        })
        sig = "SIGNIFICANT" if (ci_lo > 0 or ci_hi < 0) else "not significant"
        print(f"{model_a:18} vs {model_b:14} [{cluster:13}] (n_zones={n_zones:2}): "
              f"diff={obs_diff:+.4f}  95% CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  "
              f"p~{p_two_sided:.3f}  -> {sig}")

    out = pd.DataFrame(rows)
    out.to_csv("significance_test_results_2020_2024.csv", index=False)
    print("\nwrote significance_test_results_2020_2024.csv")


if __name__ == "__main__":
    main()
