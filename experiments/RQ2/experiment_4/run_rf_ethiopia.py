"""RQ2 Experiment 4: RandomForestRegressor arm -- a 5th architecture added
to RQ2's original XGBoost/LSTM/TabICLv2/T-GCN comparison, at the project
owner's explicit request (2026-08-25; docs/dissertation_plan.md RQ2 updated
accordingly). Confirmed scope: same dataset/split/clusters/leads as
experiments/RQ1/experiment_2's XGBoost arm ("Same as RQ1 Exp2 / RQ2 Exp1").

This script is a direct architecture swap of
experiments/RQ1/experiment_2/run_model_ethiopia.py -- same per-cluster
(pastoral/agropastoral/crop_farming) x per-lead (0,1,2,3,4,8,12) model
grid, same feature matrix construction, same level/delta target-mode
choice, same date-based extended-window override, same persistence/
seasonality benchmarks and crisis-onset scoring -- with RandomForestRegressor
in place of XGBRegressor. Deliberately reimplemented rather than imported
(matches the existing convention: experiment_1's LSTM script and
experiment_2's TabICL script are each self-contained too, not sharing code
across experiment folders). The XGBoost-specific ACLED-lag/log1p/GLEAM-GloFAS ablation flags are NOT
carried over here -- re-run run_model_ethiopia.py directly if one of those
narrower ablations is ever wanted on RF. **`--exclude-feature-cluster` was
added 2026-08-25** (experiments/RQ2/experiment_5's ensemble build), copied
verbatim from run_model_ethiopia.py's FEATURE_CLUSTER_PREFIXES/
ALWAYS_KEEP_PREFIXES/classify_feature_cluster() -- specifically so an
RF-delta-noclimate run can be produced as RF's counterpart to XGBoost's
already-established best single-model recipe (unhcr_delta_noclimate,
report.md section 7.4) for a same-backbone ensemble blend, not for a fresh
RF ablation study of its own.

RF_PARAMS: n_estimators=400 matches XGB_PARAMS for a like-for-like ensemble
size. max_depth=10 and min_samples_leaf=3 are set (not sklearn's
unconstrained defaults) because the smallest cluster (agropastoral, 5
zones) has as few as ~150-2000 training rows depending on lead -- an
unconstrained RandomForestRegressor overfits badly at that scale (checked
empirically: unconstrained trees drove agropastoral train R2 to ~0.98 while
test R2 went negative). No hyperparameter search was run (matches this
project's standing "fixed params, not tuned" convention for XGB_PARAMS/
TabICL's own defaults) -- these are a single reasonable choice, not a
result of cross-validation.

Usage:
    python run_rf_ethiopia.py --model-tables-dir <dir with ethiopia_lead*.csv> \
        --out-dir <output directory> [--target-mode level|delta] \
        [--train-end YYYY-MM --test-start YYYY-MM --test-end YYYY-MM --exclude-years Y,Y]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]
LEADS = [0, 1, 2, 3, 4, 8, 12]

RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": 10,
    "min_samples_leaf": 3,
    "random_state": 42,
    "n_jobs": -1,
}

CRISIS_THRESHOLD = 3.0
CRISIS_BUFFER = 0.05

TARGET_COL = "ipc_continuous"
BASE1_COL = "ipc_lag1"

# Identical to experiments/RQ1/experiment_2/run_model_ethiopia.py's own
# IDENTITY_DROP/PROVENANCE_DROP -- see that file's docstring for the
# rationale on each column.
IDENTITY_DROP = [
    "zone_code", "zone_id", "month", "zone_name", "assessment_month",
    "origin_month", "lead_months", "observed", "split",
    "dominant_livelihood_zone",
    "ipc_continuous", "ipc_phase_20pct", "pct_phase3plus", "pop_coverage",
    "ipc_continuous_area", "ha_share",
]
PROVENANCE_DROP = [
    "maize_yield_status", "maize_yield_note", "wheat_yield_status", "wheat_yield_note",
    "sorghum_yield_status", "sorghum_yield_note", "teff_yield_status", "teff_yield_note",
    "headline_cpi_yoy_change_source", "food_cpi_yoy_change_source",
    "headline_cpi_chainlinked_source", "food_cpi_chainlinked_source",
    "glofas_source", "seasonal_source", "seasonal_target_season",
    "usgs_gefs_anom_units", "usgs_gefs_issue_date", "usgs_gefs_source",
]

# Copied verbatim from experiments/RQ1/experiment_2/run_model_ethiopia.py --
# see that file for the full membership rationale. Kept in sync manually
# (not imported) per this project's existing per-experiment-folder
# self-containment convention.
FEATURE_CLUSTER_PREFIXES = {
    "climate": (
        "total_rainfall_mm", "wet_days", "max_dry_spell_length", "rainfall_sum_",
        "spi_", "pet_mm", "sm_root", "precip_mm", "spei_balance_mm", "sm_mean_",
        "spei_balance_sum_", "ssmi_", "spei_", "ndvi", "iod", "mei", "nino34", "wvg",
        "glofas_", "seasonal_", "usgs_gefs_",
    ),
    "agriculture": (
        "pct_pastoral", "pct_agropastoral", "pct_crop_farming", "season_system",
        "is_harvest_season", "is_lean_season", "is_dry_season",
        "months_since_harvest_start", "months_since_lean_start", "agss_surveyed_flag",
        "maize_yield", "wheat_yield", "sorghum_yield", "teff_yield",
    ),
    "economic": (
        "maize_price", "wheat_price", "sorghum_price", "teff_price",
        "fuel_diesel_price", "fuel_petrol_price", "livestock_goat_price",
        "livestock_sheep_price", "goat_maize_tot", "sheep_maize_tot",
        "maize_n_markets", "wheat_n_markets", "sorghum_n_markets", "teff_n_markets",
        "fuel_diesel_n_markets", "fuel_petrol_n_markets", "livestock_goat_n_markets",
        "livestock_sheep_n_markets", "headline_cpi", "food_cpi", "exchange_rate",
        "gdp_per_capita",
    ),
    "conflict": (
        "acled_event_count", "acled_fatalities",
        "total_refugees_hosted", "total_asylum_seekers_hosted", "idps_ethiopia",
    ),
}
ALWAYS_KEEP_PREFIXES = (
    "ipc_lag1", "ipc_lag4", "ipc_lag8", "ipc_mean12",
    "ha_share_lag1", "ha_share_lag4", "ha_share_lag8", "ha_share_mean12",
    "months_since_assessment",
)


def classify_feature_cluster(col):
    if col in ALWAYS_KEEP_PREFIXES:
        return "always_keep"
    for group, prefixes in FEATURE_CLUSTER_PREFIXES.items():
        if col.startswith(prefixes):
            return group
    return None


def load_lead_table(model_tables_dir, lead):
    path = Path(model_tables_dir) / f"ethiopia_lead{lead:02d}.csv"
    return pd.read_csv(path, low_memory=False)


def seasonality_predictions(train_labels, test_index):
    monthly = train_labels.groupby(train_labels.index.month).mean()
    monthly = monthly.reindex([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1])
    monthly = monthly.interpolate(method="linear", axis=0)
    monthly = monthly.iloc[:-1]
    values = monthly.reindex(test_index.month).to_numpy()
    return pd.Series(values, index=test_index)


def run_cell(df, cluster, lead, verbose=True, models_dir=None, target_mode="level",
             exclude_feature_cluster=None,
             train_end=None, test_start=None, test_end=None, exclude_years=None):
    sub = df[(df["dominant_livelihood_zone"] == cluster) & (df["observed"] == True)].copy()
    sub = sub.dropna(axis=1, how="all")
    sub["time"] = pd.to_datetime(sub["month"])
    custom_split = train_end is not None or test_start is not None
    if not custom_split:
        sub = sub[sub["split"].isin(["train", "test"])]
    if target_mode == "delta":
        sub = sub[sub[BASE1_COL].notna()]
    sub = sub.sort_values(["time", "zone_code"]).reset_index(drop=True)

    meta = sub[["time", "zone_code", "zone_name", "split"]].copy()
    labels = sub[TARGET_COL]
    base1 = sub[BASE1_COL]

    frame = sub.copy()
    if exclude_feature_cluster is not None:
        excluded = ({exclude_feature_cluster} if isinstance(exclude_feature_cluster, str)
                    else set(exclude_feature_cluster))
        excl_cols = [c for c in frame.columns if classify_feature_cluster(c) in excluded]
        frame = frame.drop(columns=excl_cols)
    if "season_system" in frame.columns:
        frame = pd.get_dummies(frame, columns=["season_system"], prefix="season", prefix_sep="_")
    drop_cols = [c for c in IDENTITY_DROP + PROVENANCE_DROP if c in frame.columns]
    frame = frame.drop(columns=drop_cols)
    frame = frame.drop(columns=frame.select_dtypes(include=["object", "category"]).columns)
    features = frame.drop(columns=["time"], errors="ignore")
    bool_cols = features.select_dtypes(include="bool").columns
    if len(bool_cols):
        features[bool_cols] = features[bool_cols].astype(int)

    # RandomForestRegressor, unlike XGBRegressor, cannot take NaN directly --
    # median-impute (train-derived) per column, same disclosed model-input-
    # stage-only imputation already used for the LSTM arm (RQ2 Experiment 1
    # report.md), not a change to any pipeline output.
    if custom_split:
        if train_end is not None:
            train_mask = (meta["time"] <= pd.Timestamp(train_end)).to_numpy()
        else:
            train_mask = (meta["split"] == "train").to_numpy()
        if test_start is not None:
            test_mask = (meta["time"] >= pd.Timestamp(test_start)).to_numpy()
        else:
            test_mask = (meta["split"] == "test").to_numpy()
        if test_end is not None:
            test_mask &= (meta["time"] <= pd.Timestamp(test_end)).to_numpy()
        if exclude_years:
            test_mask &= ~meta["time"].dt.year.isin(exclude_years).to_numpy()
    else:
        train_mask = (meta["split"] == "train").to_numpy()
        test_mask = (meta["split"] == "test").to_numpy()

    train_medians = features[train_mask].median(numeric_only=True)
    features = features.fillna(train_medians)

    train_x, test_x = features[train_mask], features[test_mask]
    train_y, test_y = labels[train_mask], labels[test_mask]
    train_meta, test_meta = meta[train_mask].reset_index(drop=True), meta[test_mask].reset_index(drop=True)
    base1_train = base1[train_mask].reset_index(drop=True)
    base1_test = base1[test_mask].reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    if target_mode == "delta":
        fit_y = train_y - base1_train
        model = RandomForestRegressor(**RF_PARAMS).fit(train_x.reset_index(drop=True), fit_y)
        raw_predictions = model.predict(test_x.reset_index(drop=True))
        predictions = base1_test.to_numpy() + raw_predictions
    else:
        model = RandomForestRegressor(**RF_PARAMS).fit(train_x.reset_index(drop=True), train_y)
        predictions = model.predict(test_x.reset_index(drop=True))

    if models_dir is not None:
        import joblib
        joblib.dump(model, models_dir / f"rf_{cluster}_lead{lead}.joblib")

    base2 = pd.Series(np.nan, index=test_meta.index)
    train_labels_s = train_y.copy()
    train_labels_s.index = pd.DatetimeIndex(train_meta["time"])
    for zone in test_meta["zone_code"].unique():
        train_rows = train_labels_s[train_meta["zone_code"].to_numpy() == zone]
        test_rows = test_meta.index[test_meta["zone_code"] == zone]
        if len(train_rows) == 0:
            continue
        base2.loc[test_rows] = seasonality_predictions(
            train_rows, pd.DatetimeIndex(test_meta.loc[test_rows, "time"])
        ).to_numpy()

    out = test_meta.copy()
    out["lhz"] = cluster
    out["lead"] = lead
    out["observed"] = test_y.to_numpy()
    out["prediction"] = predictions
    out["base1_preds"] = base1_test.to_numpy()
    out["base2_preds"] = base2.to_numpy()

    if verbose:
        boundary = train_meta["time"].max()
        print(f"    {cluster:13} L{lead:<2} n_feat={features.shape[1]:3} "
              f"train={len(train_x):5} test={len(test_x):5} "
              f"split after {boundary:%Y-%m} | "
              f"MAE {mean_absolute_error(test_y, predictions):.4f} "
              f"R2 {r2_score(test_y, predictions):.4f}")

    importance = pd.DataFrame({
        "feature": features.columns,
        "importance": model.feature_importances_,
        "lhz": cluster,
        "lead": lead,
    })
    info = {
        "lhz": cluster, "lead": lead, "n_features": int(features.shape[1]),
        "n_train": int(len(train_x)), "n_test": int(len(test_x)),
        "train_end": str(train_meta["time"].max().date()),
        "test_start": str(test_meta["time"].min().date()),
    }
    return out, importance, info


def score(predictions, group_cols):
    rows = []
    for keys, group in predictions.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        truth = group["observed"]
        record = dict(zip(group_cols, keys))
        record["n"] = len(group)
        for label, column in (("", "prediction"),
                              ("_baseline", "base1_preds"),
                              ("_baseline2", "base2_preds")):
            pred = group[column]
            ok = pred.notna() & truth.notna()
            if ok.sum() < 2:
                record[f"mae{label}"] = np.nan
                record[f"rmse{label}"] = np.nan
                record[f"r2{label}"] = np.nan
                continue
            record[f"mae{label}"] = mean_absolute_error(truth[ok], pred[ok])
            record[f"rmse{label}"] = mean_squared_error(truth[ok], pred[ok]) ** 0.5
            record[f"r2{label}"] = r2_score(truth[ok], pred[ok])
        rows.append(record)
    return pd.DataFrame(rows)


def _onset_flags(series, threshold, buffer):
    flags = series.where(series >= (threshold - buffer), 0)
    flags = flags.where(flags == 0, 1)
    flags = np.where((flags == 1) & (flags != flags.shift(1)), 1, 0)
    if len(flags) > 0:
        flags[0] = 0
    return flags


def cont_calc(truth, preds, threshold=CRISIS_THRESHOLD, buffer=CRISIS_BUFFER):
    truth_bin = _onset_flags(truth, threshold, buffer)
    preds_bin = _onset_flags(preds, threshold, buffer)

    hits = truth_bin * preds_bin
    false_alarms = ((preds_bin == 1) & (truth_bin == 0)).astype(float)
    correct_negatives = ((preds_bin == 0) & (truth_bin == 0)).astype(float)
    misses = ((preds_bin == 0) & (truth_bin == 1)).astype(float)

    denominator = false_alarms.sum() + correct_negatives.sum()
    far = false_alarms.sum() / denominator if denominator > 0 else np.nan
    denominator = hits.sum() + misses.sum()
    hr = hits.sum() / denominator if denominator > 0 else np.nan

    recall = recall_score(truth_bin, preds_bin, zero_division=0)
    precision = precision_score(truth_bin, preds_bin, zero_division=0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {
        "event_count": int(truth_bin.sum()),
        "hr": hr, "far": far,
        "recall": recall, "precision": precision, "f1": f1,
    }


def crisis_onset_rates(predictions, by=("lhz", "lead")):
    by = list(by)
    rows = []
    ordered = predictions.sort_values(by + ["zone_code", "time"])
    for keys, group in ordered.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        truth = group["observed"].reset_index(drop=True)
        record = dict(zip(by, keys))
        for label, column in (("", "prediction"),
                              ("_baseline", "base1_preds"),
                              ("_baseline2", "base2_preds")):
            preds = group[column].reset_index(drop=True)
            scores = cont_calc(truth, preds.fillna(0))
            record["event_count"] = scores["event_count"]
            record[f"hr{label}"] = scores["hr"]
            record[f"far{label}"] = scores["far"]
            if label == "":
                record["f1"] = scores["f1"]
        rows.append(record)
    return pd.DataFrame(rows)


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-tables-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--leads", type=str, default=None)
    parser.add_argument("--clusters", type=str, default=None)
    parser.add_argument("--target-mode", choices=["level", "delta"], default="level")
    parser.add_argument("--exclude-feature-cluster", type=str, default=None,
                         help="comma-separated cluster name(s) from FEATURE_CLUSTER_PREFIXES "
                              "(climate/agriculture/economic/conflict) to drop before fitting.")
    parser.add_argument("--train-end", type=str, default=None)
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    parser.add_argument("--exclude-years", type=str, default=None)
    args = parser.parse_args(argv)

    leads = [int(x) for x in args.leads.split(",")] if args.leads else LEADS
    clusters = args.clusters.split(",") if args.clusters else CLUSTERS
    exclude_years = ([int(y) for y in args.exclude_years.split(",")]
                     if args.exclude_years else None)
    exclude_clusters = (args.exclude_feature_cluster.split(",")
                        if args.exclude_feature_cluster else None)
    if exclude_clusters:
        valid = set(FEATURE_CLUSTER_PREFIXES)
        bad = set(exclude_clusters) - valid
        if bad:
            raise SystemExit(f"--exclude-feature-cluster: unknown cluster(s) {bad}, "
                             f"valid options: {sorted(valid)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = None
    if args.save_models:
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nfitting {len(clusters)} x {len(leads)} = "
          f"{len(clusters) * len(leads)} RandomForestRegressor models")
    all_predictions, all_importance, infos = [], [], []
    for lead in leads:
        df = load_lead_table(args.model_tables_dir, lead)
        for cluster in clusters:
            n_units = df.loc[df["dominant_livelihood_zone"] == cluster, "zone_code"].nunique()
            if lead == leads[0]:
                print(f"  {cluster}: {n_units} zones")
            predictions, importance, info = run_cell(
                df, cluster, lead, models_dir=models_dir, target_mode=args.target_mode,
                exclude_feature_cluster=exclude_clusters,
                train_end=args.train_end, test_start=args.test_start,
                test_end=args.test_end, exclude_years=exclude_years,
            )
            all_predictions.append(predictions)
            all_importance.append(importance)
            infos.append(info)

    predictions = pd.concat(all_predictions, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)

    per_unit = score(predictions, ["lhz", "lead", "zone_code"])
    per_cluster = score(predictions, ["lhz", "lead"])
    per_lead = score(predictions, ["lead"])
    onsets = crisis_onset_rates(predictions)

    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    per_unit.to_csv(out_dir / "metrics_per_unit.csv", index=False)
    per_cluster.to_csv(out_dir / "metrics_per_cluster.csv", index=False)
    per_lead.to_csv(out_dir / "metrics_per_lead.csv", index=False)
    onsets.to_csv(out_dir / "crisis_onset_rates.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    meta = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "model_tables_dir": str(args.model_tables_dir),
        "rf_params": RF_PARAMS,
        "target_mode": args.target_mode,
        "exclude_feature_cluster": exclude_clusters,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "exclude_years": exclude_years,
        "split_rule": (
            f"custom date-based split: train <= {args.train_end}, "
            f"test >= {args.test_start}"
            + (f" and <= {args.test_end}" if args.test_end else "")
            + (f", excluding years {exclude_years}" if exclude_years else "")
        ) if (args.train_end or args.test_start) else (
            "date-based (train 2011-04..2019-12, test 2020-01..2022-12), "
            "observed==True rows only"
        ),
        "leads": leads,
        "clusters": clusters,
        "cells": infos,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nPer-lead, pooled over all zones (model | persistence | seasonality):")
    table = per_lead.set_index("lead")[
        ["n", "mae", "mae_baseline", "mae_baseline2", "r2", "r2_baseline", "r2_baseline2"]
    ]
    print(table.round(4).to_string())

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
