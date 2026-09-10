"""NEW (2026-09-02): closes final_answers_to_cite.md gap #2 -- no
significance test previously existed comparing the RQ2 champion ensemble
(4-way, XGBoost+RandomForest+LSTM+TabICLv2, no T-GCN) directly against
Busker's architecture. Every RQ2 Experiment 5 significance test compares
ensemble configs against each other, against persistence, or against
RandomForest/XGBoost alone -- this is the missing one.

Same method as significance_test_vs_experiment1.py (which does the
equivalent single-model comparison for RQ1): unit(zone)-level block
bootstrap (500 iters, seed 42), resample zone_codes with replacement,
recompute mean-weighted-F1-across-leads for each side, paired difference.

Window: restricted to the 8 real release months both sides can actually
share (2020-02 .. 2022-06) -- Busker's own released dataset
(input_master.parquet) has a hard ceiling at 2022-12-01, and his target
is itself only observed a few times a year, so this is the maximum
possible overlap regardless of which window the ensemble side is scored
on. Confirmed empirically (see experiment_log.md's 2026-09-02 entry):
using the ensemble's already-built 2020-2024 extended predictions gains
zero additional overlap over the plain 2020-2022 default predictions,
because Busker is the binding constraint either way. Uses the
extended-window ensemble file anyway (matches every other ensemble
reference in this session), for consistency, not because it changes the
result.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
N_ITER = 500
SEED = 42

EXP1_PATH = "../../RQ1/experiment_1/busker_arch_datesplit_2020_2022/predictions.parquet"
ENSEMBLE_PATH = "all_combinations_extended/predictions/XGB_RF_LSTM_TabICL.parquet"
ZONE_NAME_LOOKUP_PATH = "../../RQ1/experiment_2/predictions_extended/predictions.parquet"

LHZ_MAP_EXP1 = {"p": "pastoral", "ap": "agropastoral", "other": "crop_farming"}


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_busker(path):
    df = pd.read_parquet(path)
    df = df[df["country"] == "Ethiopia"].copy()
    df["time"] = pd.to_datetime(df["time"])
    df["zone_id"] = df["county"]
    df["lhz"] = df["lhz"].map(LHZ_MAP_EXP1)
    assert df["lhz"].notna().all(), "unmapped lhz code in Busker data"
    return df[["time", "zone_id", "lhz", "lead", "observed", "prediction"]]


def load_ensemble(path, zone_lookup_path):
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    lookup = pd.read_parquet(zone_lookup_path)[["zone_code", "zone_name"]].drop_duplicates()
    df = df.merge(lookup, on="zone_code", how="left")
    assert df["zone_name"].notna().all(), "unmapped zone_code in ensemble predictions"
    df["zone_id"] = df["zone_name"]
    return df[["time", "zone_id", "lhz", "lead", "observed", "prediction"]]


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
    zones = df_a["zone_id"].unique()
    n_zones = len(zones)
    by_zone_a = {z: g for z, g in df_a.groupby("zone_id")}
    by_zone_b = {z: g for z, g in df_b.groupby("zone_id")}
    observed_diff = mean_f1_across_leads(df_a) - mean_f1_across_leads(df_b)
    draws = np.empty(n_iter)
    for i in range(n_iter):
        sample_zones = rng.choice(zones, size=n_zones, replace=True)
        pooled_a = pd.concat([by_zone_a[z] for z in sample_zones], axis=0)
        pooled_b = pd.concat([by_zone_b[z] for z in sample_zones], axis=0)
        draws[i] = mean_f1_across_leads(pooled_a) - mean_f1_across_leads(pooled_b)
    return observed_diff, draws, n_zones


def main():
    busker = load_busker(EXP1_PATH)
    ensemble = load_ensemble(ENSEMBLE_PATH, ZONE_NAME_LOOKUP_PATH)

    common_months = sorted(set(busker["time"].unique()) & set(ensemble["time"].unique()))
    print(f"shared release months ({len(common_months)}): "
          f"{[pd.Timestamp(m).strftime('%Y-%m') for m in common_months]}")
    busker = busker[busker["time"].isin(common_months)]
    ensemble = ensemble[ensemble["time"].isin(common_months)]

    assert set(busker["zone_id"].unique()) == set(ensemble["zone_id"].unique()), \
        "zone identifiers do not match between Busker and the ensemble"

    models = {"ensemble_4way": ensemble, "busker_architecture": busker}
    comparisons = [
        ("ensemble_4way", "busker_architecture", "all"),
        ("ensemble_4way", "busker_architecture", "pastoral"),
        ("ensemble_4way", "busker_architecture", "agropastoral"),
        ("ensemble_4way", "busker_architecture", "crop_farming"),
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
        print(f"{model_a:20} vs {model_b:20} [{cluster:13}] (n={n_zones:2}): "
              f"diff={obs_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p~{p:.3f}  -> {sig}")

    pd.DataFrame(rows).to_csv("significance_test_ensemble_vs_busker_results.csv", index=False)
    print("\nwrote significance_test_ensemble_vs_busker_results.csv")


if __name__ == "__main__":
    main()
