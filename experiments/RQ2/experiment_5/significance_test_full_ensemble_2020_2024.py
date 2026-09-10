"""Follow-up to build_full_ensemble.py: the leave-one-out ablation showed
T-GCN's inclusion drags the 5-way average down badly (all-zones mean F1
0.6798 with it, 0.7218 without it -- a 4-way average of XGBoost+
RandomForest+LSTM+TabICLv2 that beats every single component AND
persistence). Same question as significance_test_blend.py: is that
apparent edge real, or zone-resampling noise?

Same method: unit(zone)-level block bootstrap (500 iters, seed 42),
resample zone_codes with replacement, recompute mean-weighted-F1-across-
leads for each side, paired difference.

--- 2020-2024 REBUILD (2026-09-02) ---
Same comparisons, same method, on the project's revised primary window
(docs/dissertation_plan.md section 6). XGB/RF paths point at their
already-built _extended runs; FULL5/4-way paths point at the
already-built extended-window ensemble files
(full_ensemble_predictions_extended.parquet,
all_combinations_extended/predictions/XGB_RF_LSTM_TabICL.parquet) --
no new ensemble build needed, only new significance draws. Output
filename suffixed _2020_2024 so the original 2020-2022 result file is
untouched.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
N_ITER = 500
SEED = 42

XGB_PATH = "../../RQ1/experiment_2/unhcr_delta_noclimate_extended/predictions.parquet"
RF_PATH = "../experiment_4/rf_delta_noclimate_extended/predictions.parquet"
FULL5_PATH = "full_ensemble_predictions_extended.parquet"
LOO_DROP_TGCN_PATH = "all_combinations_extended/predictions/XGB_RF_LSTM_TabICL.parquet"


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
    full5 = load(FULL5_PATH)
    loo4 = load(LOO_DROP_TGCN_PATH)
    persistence = load(XGB_PATH, pred_col="base1_preds")

    models = {
        "4-way-ensemble(no T-GCN)": loo4,
        "5-way-ensemble": full5,
        "XGBoost-noclimate": xgb,
        "RandomForest-noclimate": rf,
        "Persistence": persistence,
    }

    comparisons = [
        ("4-way-ensemble(no T-GCN)", "5-way-ensemble", "all"),
        ("4-way-ensemble(no T-GCN)", "RandomForest-noclimate", "all"),
        ("4-way-ensemble(no T-GCN)", "XGBoost-noclimate", "all"),
        ("4-way-ensemble(no T-GCN)", "Persistence", "all"),
        ("4-way-ensemble(no T-GCN)", "RandomForest-noclimate", "pastoral"),
        ("4-way-ensemble(no T-GCN)", "Persistence", "pastoral"),
        ("4-way-ensemble(no T-GCN)", "RandomForest-noclimate", "agropastoral"),
        ("4-way-ensemble(no T-GCN)", "Persistence", "agropastoral"),
        ("4-way-ensemble(no T-GCN)", "RandomForest-noclimate", "crop_farming"),
        ("4-way-ensemble(no T-GCN)", "Persistence", "crop_farming"),
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
        print(f"{model_a:26} vs {model_b:23} [{cluster:13}] (n={n_zones:2}): "
              f"diff={obs_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p~{p:.3f}  -> {sig}")

    pd.DataFrame(rows).to_csv("significance_test_full_ensemble_results_2020_2024.csv", index=False)
    print("\nwrote significance_test_full_ensemble_results_2020_2024.csv")


if __name__ == "__main__":
    main()
