"""Is RQ1 Experiment 2's best variant (unhcr_delta_noclimate: Ethiopia-
enriched dataset, delta target, climate cluster removed) actually better
than experiment_1's Busker-architecture reproduction, or is the disclosed
+2.6% Ethiopia-wide / +6.6% pastoral edge (report.md SS7.4) within
zone-resampling noise?

Same method as experiments/RQ2/experiment_5/significance_test*.py (the
tests used to pick RQ2's champion ensemble): unit(zone)-level block
bootstrap (500 iters, seed 42), resample zone identifiers with
replacement, recompute mean-weighted-F1-across-leads on the resampled
multiset for each side, paired difference, 95% percentile CI.

Two schema differences from experiment_1 vs experiment_2 handled here:
- experiment_1 (busker_arch_datesplit_2020_2022) keys zones by `county`
  (HOA-wide unit name), experiment_2 keys by `zone_code`/`zone_name` --
  matched here on unit name, confirmed exact 92/92 match for Ethiopia.
- experiment_1 uses Busker's own lhz codes (p/ap/other), experiment_2
  uses this project's own (pastoral/agropastoral/crop_farming) -- mapped
  1:1 (`other` -> `crop_farming`, per CLAUDE.md's own livelihood_zones
  pipeline note).
experiment_2's test window has one extra release month (2022-10) that
experiment_1's underlying Busker input_master.parquet lacks -- restricted
to the 8 months both sides actually share so this is a true paired
comparison, not just a matched date range.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LEADS = [0, 1, 2, 3, 4, 8, 12]
N_ITER = 500
SEED = 42

EXP1_PATH = "../experiment_1/busker_arch_datesplit_2020_2022/predictions.parquet"
EXP2_PATH = "unhcr_delta_noclimate/predictions.parquet"

LHZ_MAP_EXP1 = {"p": "pastoral", "ap": "agropastoral", "other": "crop_farming"}


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_exp1(path):
    df = pd.read_parquet(path)
    df = df[df["country"] == "Ethiopia"].copy()
    df["time"] = pd.to_datetime(df["time"])
    df["zone_id"] = df["county"]
    df["lhz"] = df["lhz"].map(LHZ_MAP_EXP1)
    assert df["lhz"].notna().all(), "unmapped lhz code in experiment_1"
    return df[["time", "zone_id", "lhz", "lead", "observed", "prediction"]]


def load_exp2(path):
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
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
    exp1 = load_exp1(EXP1_PATH)
    exp2 = load_exp2(EXP2_PATH)

    common_months = sorted(set(exp1["time"].unique()) & set(exp2["time"].unique()))
    print(f"shared release months ({len(common_months)}): "
          f"{[pd.Timestamp(m).strftime('%Y-%m') for m in common_months]}")
    dropped_exp2 = sorted(set(exp2["time"].unique()) - set(common_months))
    if dropped_exp2:
        print(f"dropping exp2-only months not in exp1: "
              f"{[pd.Timestamp(m).strftime('%Y-%m') for m in dropped_exp2]}")
    exp1 = exp1[exp1["time"].isin(common_months)]
    exp2 = exp2[exp2["time"].isin(common_months)]

    assert set(exp1["zone_id"].unique()) == set(exp2["zone_id"].unique()), \
        "zone identifiers do not match between experiment_1 and experiment_2"

    models = {"exp2_unhcr_delta_noclimate": exp2, "exp1_busker_arch_reproduction": exp1}

    comparisons = [
        ("exp2_unhcr_delta_noclimate", "exp1_busker_arch_reproduction", "all"),
        ("exp2_unhcr_delta_noclimate", "exp1_busker_arch_reproduction", "pastoral"),
        ("exp2_unhcr_delta_noclimate", "exp1_busker_arch_reproduction", "agropastoral"),
        ("exp2_unhcr_delta_noclimate", "exp1_busker_arch_reproduction", "crop_farming"),
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
        print(f"{model_a:30} vs {model_b:30} [{cluster:13}] (n={n_zones:2}): "
              f"diff={obs_diff:+.4f}  95% CI=[{ci_lo:+.4f},{ci_hi:+.4f}]  p~{p:.3f}  -> {sig}")

    pd.DataFrame(rows).to_csv("significance_test_vs_experiment1_results.csv", index=False)
    print("\nwrote significance_test_vs_experiment1_results.csv")


if __name__ == "__main__":
    main()
