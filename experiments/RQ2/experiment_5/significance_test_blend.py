"""Follow-up to significance_test.py: is the 50/50 blend
(build_ensemble.py's blend_predictions.parquet) actually better than its
two single-model components, and than persistence -- or is its small
observed edge (all-zones mean F1 0.7098 vs. XGBoost 0.7074 / RandomForest
0.7093 / persistence 0.7009) within zone-resampling noise, same question
already asked of the RF/LSTM architecture gaps in significance_test.py.

Same method: unit(zone)-level block bootstrap (500 iters, seed 42),
resample zone_codes with replacement, recompute mean-weighted-F1-across-
leads on the resampled multiset for each side, paired difference.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
N_ITER = 500
SEED = 42

XGB_PATH = "../../RQ1/experiment_2/unhcr_delta_noclimate/predictions.parquet"
RF_PATH = "../experiment_4/rf_delta_noclimate/predictions.parquet"
BLEND_PATH = "blend_predictions.parquet"


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load(path, pred_col="prediction"):
    df = pd.read_parquet(path)
    df = df[["time", "zone_code", "lhz", "lead", "observed", pred_col]].copy()
    df = df.rename(columns={pred_col: "prediction"})
    df["time"] = pd.to_datetime(df["time"])
    return df


def mean_f1_across_leads(df):
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
    xgb = load(XGB_PATH)
    rf = load(RF_PATH)
    blend = load(BLEND_PATH)
    persistence = load(XGB_PATH, pred_col="base1_preds")  # same rows, persistence-as-prediction

    models = {"Blend": blend, "XGBoost-noclimate": xgb, "RandomForest-noclimate": rf,
              "Persistence": persistence}

    comparisons = [
        ("Blend", "XGBoost-noclimate", "all"),
        ("Blend", "RandomForest-noclimate", "all"),
        ("Blend", "Persistence", "all"),
        ("Blend", "XGBoost-noclimate", "pastoral"),
        ("Blend", "RandomForest-noclimate", "pastoral"),
        ("Blend", "Persistence", "pastoral"),
        ("Blend", "Persistence", "agropastoral"),
        ("Blend", "Persistence", "crop_farming"),
    ]

    rng = np.random.default_rng(SEED)
    rows = []
    for model_a, model_b, cluster in comparisons:
        obs_diff, draws, n_zones = bootstrap_diff(models[model_a], models[model_b], cluster, rng)
        boot_sd = draws.std(ddof=1)
        ci_lo, ci_hi = np.percentile(draws, [2.5, 97.5])
        p = min(2 * min((draws <= 0).mean(), (draws >= 0).mean()), 1.0)
        rows.append({
            "model_a": model_a, "model_b": model_b, "cluster": cluster, "n_zones": n_zones,
            "observed_diff_f1": round(obs_diff, 4), "bootstrap_sd": round(boot_sd, 4),
            "ci_95_lo": round(ci_lo, 4), "ci_95_hi": round(ci_hi, 4),
            "p_two_sided_approx": round(p, 4), "ci_excludes_zero": bool(ci_lo > 0 or ci_hi < 0),
        })
        sig = "SIGNIFICANT" if (ci_lo > 0 or ci_hi < 0) else "not significant"
        print(f"{model_a:10} vs {model_b:23} [{cluster:13}] (n={n_zones:2}): "
              f"diff={obs_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p~{p:.3f}  -> {sig}")

    pd.DataFrame(rows).to_csv("significance_test_blend_results.csv", index=False)
    print("\nwrote significance_test_blend_results.csv")


if __name__ == "__main__":
    main()
