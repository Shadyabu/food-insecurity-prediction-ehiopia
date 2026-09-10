"""Fit and score Busker et al.'s (2024) 21 XGBoost models (3 livelihood
zones x 7 leads) against a given input_master.

Self-contained copy of the model-fitting/scoring logic kept inside this
experiment folder (not `experiments/busker_baseline/run_busker_baseline.py`)
so `experiments/RQ1/experiment_1/` stays a fully self-contained package, per
this experiment's own report.md ("Keep this experiment's code and outputs
isolated"). Same XGB_PARAMS, same positional 80:20 split, same benchmarks
and crisis-onset scoring as the original -- see `report.md` section 1 for
full methodology notes and `CLAUDE.md` section 7a for what Busker's released
code settles.

Usage:
    python run_model.py --input-master <path to input_master.parquet> \
        --out-dir <output directory> [--save-models]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

CLUSTERS = ["p", "ap", "other"]
CLUSTER_LABELS = {"p": "pastoral", "ap": "agropastoral", "other": "crop farming"}
LEADS = [0, 1, 2, 3, 4, 8, 12]

TRAIN_TEST_RATIO = 0.20

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.01,
    "random_state": 42,
}

DROP_FEATURES = ["lead", "base_forecast", "FEWS_CS"]

CRISIS_THRESHOLD = 3.0
CRISIS_BUFFER = 0.05


def load_input_master(path):
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["year"])
    wpg = [c for c in df.columns if "WPG" in c]
    df = df.drop(columns=wpg)
    print(f"input_master: {len(df):,} rows, {len(df.columns)} columns "
          f"(dropped year and {len(wpg)} WPG columns)")
    return df


def seasonality_predictions(train_labels, test_index):
    monthly = train_labels.groupby(train_labels.index.month).mean()
    monthly = monthly.reindex([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1])
    monthly = monthly.interpolate(method="linear", axis=0)
    monthly = monthly.iloc[:-1]
    values = monthly.reindex(test_index.month).to_numpy()
    return pd.Series(values, index=test_index)


def run_cell(df, cluster, lead, verbose=True, models_dir=None, split_dates=None):
    """split_dates: None for Busker's own positional 80:20 split (default,
    unchanged behavior). Or (train_end, test_start, test_end) ISO date
    strings for a date-based split matching this project's own
    train<=2019 / test 2020-2022 convention (used elsewhere via
    experiment_2's run_model_ethiopia.py) -- needed to compare the
    Busker-architecture reproduction against the enriched-dataset runs on
    the same date boundary, since the positional split lands inside
    2019-06 per-cell (varies by cluster/lead), not on a fixed date."""
    sub = df[(df["lhz"] == cluster) & (df["lead"] == lead)].copy()
    sub = sub.dropna(axis=1, how="all")
    sub = sub[sub["FEWS_CS"].notna()]
    sub = sub.sort_values(["time", "county"]).reset_index(drop=True)

    meta_cols = ["time", "county", "lhz", "country"]
    meta = sub[meta_cols].copy()
    labels = sub["FEWS_CS"]
    base1 = sub["base_forecast"]

    frame = pd.get_dummies(sub, columns=["country"], prefix="", prefix_sep="")
    frame = frame.drop(columns=frame.select_dtypes(include=["object", "category"]).columns)
    features = frame.drop(columns=DROP_FEATURES + ["time"])

    if split_dates is None:
        train_x, test_x, train_y, test_y = train_test_split(
            features, labels, test_size=TRAIN_TEST_RATIO, shuffle=False
        )
        train_meta = meta.iloc[: len(train_x)]
        test_meta = meta.iloc[len(train_x):]
    else:
        train_end, test_start, test_end = split_dates
        train_mask = sub["time"] <= pd.Timestamp(train_end)
        test_mask = (sub["time"] >= pd.Timestamp(test_start)) & (sub["time"] <= pd.Timestamp(test_end))
        train_x, train_y, train_meta = features[train_mask], labels[train_mask], meta[train_mask]
        test_x, test_y, test_meta = features[test_mask], labels[test_mask], meta[test_mask]

    model = XGBRegressor(**XGB_PARAMS).fit(train_x, train_y)
    predictions = model.predict(test_x)

    if models_dir is not None:
        model.save_model(str(models_dir / f"xgb_{cluster}_lead{lead}.json"))

    base2 = pd.Series(np.nan, index=test_meta.index)
    train_labels = train_y.copy()
    train_labels.index = pd.DatetimeIndex(train_meta["time"])
    for county in test_meta["county"].unique():
        train_rows = train_labels[train_meta["county"].to_numpy() == county]
        test_rows = test_meta.index[test_meta["county"] == county]
        if len(train_rows) == 0:
            continue
        base2.loc[test_rows] = seasonality_predictions(
            train_rows, pd.DatetimeIndex(test_meta.loc[test_rows, "time"])
        ).to_numpy()

    out = test_meta.copy()
    out["lead"] = lead
    out["observed"] = test_y.to_numpy()
    out["prediction"] = predictions
    out["base1_preds"] = base1.loc[test_meta.index].to_numpy()
    out["base2_preds"] = base2.to_numpy()

    if verbose:
        boundary = train_meta["time"].max()
        print(f"    {cluster:5} L{lead:<2} n_feat={features.shape[1]:3} "
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
    ordered = predictions.sort_values(by + ["county", "time"])
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
    parser.add_argument("--input-master", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--leads", type=str, default=None)
    parser.add_argument("--clusters", type=str, default=None)
    parser.add_argument("--train-end", type=str, default=None,
                         help="ISO date; with --test-start/--test-end, replaces the "
                              "default positional 80:20 split with a date-based one")
    parser.add_argument("--test-start", type=str, default=None)
    parser.add_argument("--test-end", type=str, default=None)
    args = parser.parse_args(argv)

    leads = [int(x) for x in args.leads.split(",")] if args.leads else LEADS
    clusters = args.clusters.split(",") if args.clusters else CLUSTERS

    split_dates = None
    if args.train_end or args.test_start or args.test_end:
        if not (args.train_end and args.test_start and args.test_end):
            raise SystemExit("--train-end/--test-start/--test-end must all be given together")
        split_dates = (args.train_end, args.test_start, args.test_end)

    df = load_input_master(args.input_master)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = None
    if args.save_models:
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nfitting {len(clusters)} x {len(leads)} = "
          f"{len(clusters) * len(leads)} models")
    all_predictions, all_importance, infos = [], [], []
    for cluster in clusters:
        n_units = df.loc[df["lhz"] == cluster, "county"].nunique()
        print(f"  {CLUSTER_LABELS.get(cluster, cluster)} ({cluster}): {n_units} units")
        for lead in leads:
            predictions, importance, info = run_cell(df, cluster, lead, models_dir=models_dir,
                                                       split_dates=split_dates)
            all_predictions.append(predictions)
            all_importance.append(importance)
            infos.append(info)

    predictions = pd.concat(all_predictions, ignore_index=True)
    importance = pd.concat(all_importance, ignore_index=True)

    per_unit = score(predictions, ["lhz", "lead", "county"])
    per_cluster = score(predictions, ["lhz", "lead"])
    per_lead = score(predictions, ["lead"])
    onsets = crisis_onset_rates(predictions)
    per_country = score(predictions, ["country", "lead"])
    per_country_cluster = score(predictions, ["country", "lhz", "lead"])
    onsets_country = crisis_onset_rates(predictions, by=["country", "lhz", "lead"])

    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    per_unit.to_csv(out_dir / "metrics_per_unit.csv", index=False)
    per_cluster.to_csv(out_dir / "metrics_per_cluster.csv", index=False)
    per_lead.to_csv(out_dir / "metrics_per_lead.csv", index=False)
    onsets.to_csv(out_dir / "crisis_onset_rates.csv", index=False)
    per_country.to_csv(out_dir / "metrics_per_country.csv", index=False)
    per_country_cluster.to_csv(out_dir / "metrics_per_country_cluster.csv", index=False)
    onsets_country.to_csv(out_dir / "crisis_onset_rates_per_country.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)

    meta = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "input_master": str(args.input_master),
        "xgb_params": XGB_PARAMS,
        "train_test_ratio": TRAIN_TEST_RATIO if split_dates is None else None,
        "split_dates": split_dates,
        "leads": leads,
        "clusters": clusters,
        "cells": infos,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nPer-lead, pooled over all units (model | persistence | seasonality):")
    table = per_lead.set_index("lead")[
        ["n", "mae", "mae_baseline", "mae_baseline2", "r2", "r2_baseline", "r2_baseline2"]
    ]
    print(table.round(4).to_string())

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
