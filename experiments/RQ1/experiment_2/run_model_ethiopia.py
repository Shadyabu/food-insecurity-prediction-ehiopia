"""Fit and score the same XGBoost architecture as experiment_1
(3 livelihood-zone clusters x 7 leads) on this project's own Ethiopia
92-zone dataset (data/processed/model_tables/ethiopia_lead*.csv), for
direct comparison against experiment_1's Busker et al. (2024) reproduction.

Same XGB_PARAMS, same crisis-onset scoring (CRISIS_THRESHOLD/CRISIS_BUFFER),
same benchmark logic (persistence, seasonality) as
experiments/RQ1/experiment_1/run_model.py -- deliberately re-implemented
here rather than imported, so this experiment folder stays self-contained
too (same convention as experiment_1, see its report.md).

Necessary differences from experiment_1/run_model.py, all because the two
datasets are shaped differently, not because the model architecture changed
-- see report.md section 1 for the full rationale on each:
  - Ethiopia's model_tables are 7 already lead-shifted files (one per lead),
    not one stacked input_master filtered by a `lead` column.
  - Train/test uses this project's own date-based `split` column (train
    2011-04..2019-12, test 2020-01..2022-12; `beyond` 2023+ dropped, not
    part of this comparison) instead of Busker's positional 80:20 slice --
    CLAUDE.md section 7 ("walk-forward validation... train through 2019,
    hold out 2019-2022"), matching what
    pipelines/model_join/build_ethiopia_model_tables.py's own docstring
    already confirmed ("reuse the Busker-parity train/test split already
    embedded in the IPC base panel").
  - Restricted to `observed == True` rows only (this project's
    ipc_continuous is forward-filled between FEWS assessments; Busker's own
    code makes the equivalent restriction to FEWS_CS.notna() -- CLAUDE.md
    section 7a). Training or scoring against forward-filled/duplicated
    target values would not be a fair like-for-like comparison.
  - Feature matrix: static provenance/status/source string columns (yield
    status/note, *_source flags, seasonal_target_season,
    usgs_gefs_issue_date, usgs_gefs_anom_units, zone_name) have no Busker
    analogue and are dropped; `season_system` (3-way, static per zone) is
    one-hot encoded, mirroring Busker's own one-hot of `country` -- the
    closest structural analogue in this dataset (a static categorical
    identity attribute, not derived from the target).
  - Persistence baseline uses `ipc_lag1` (this project's own lead-shifted
    "last known IPC value" feature) in place of Busker's `base_forecast`.
    Unlike `base_forecast`, `ipc_lag1` is NOT dropped from the feature
    matrix -- this matches Busker's own treatment exactly: his released
    code keeps an equivalent `lag1` feature IN the model and drops only the
    separately-named `base_forecast` duplicate (feature_engineering.py
    :89-109/:127); there is no separate duplicate column here, so the one
    `ipc_lag1` column serves both roles.

Usage:
    python run_model_ethiopia.py --model-tables-dir <dir with ethiopia_lead*.csv> \
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
from xgboost import XGBRegressor

CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]
LEADS = [0, 1, 2, 3, 4, 8, 12]

XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.01,
    "random_state": 42,
}

CRISIS_THRESHOLD = 3.0
CRISIS_BUFFER = 0.05

TARGET_COL = "ipc_continuous"
BASE1_COL = "ipc_lag1"

# Identity/target/metadata columns -- never fed to the model as features.
# See module docstring and report.md section 1 for the rationale on each.
IDENTITY_DROP = [
    "zone_code", "zone_id", "month", "zone_name", "assessment_month",
    "origin_month", "lead_months", "observed", "split",
    "dominant_livelihood_zone",
    "ipc_continuous", "ipc_phase_20pct", "pct_phase3plus", "pop_coverage",
    "ipc_continuous_area", "ha_share",
]
# Provenance/status/source string flags -- no Busker analogue, not features.
PROVENANCE_DROP = [
    "maize_yield_status", "maize_yield_note", "wheat_yield_status", "wheat_yield_note",
    "sorghum_yield_status", "sorghum_yield_note", "teff_yield_status", "teff_yield_note",
    "headline_cpi_yoy_change_source", "food_cpi_yoy_change_source",
    "headline_cpi_chainlinked_source", "food_cpi_chainlinked_source",
    "glofas_source", "seasonal_source", "seasonal_target_season",
    "usgs_gefs_anom_units", "usgs_gefs_issue_date", "usgs_gefs_source",
]

# Domain feature-cluster ablation (docs/dissertation_plan.md RQ1's own
# separately-named "Experiment 2": "climate / agriculture / economic /
# conflict removed in turn"). Prefix-matched against raw column names,
# checked for completeness in classify_feature_cluster() below -- every
# non-identity, non-provenance column must land in exactly one of these
# four groups or ALWAYS_KEEP_PREFIXES, or the run fails loudly rather than
# silently mis-scoping an ablation. See report.md section 6 for the full
# membership listing and rationale.
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
    # UNHCR persons-of-concern (added 2026-08-24) moved here from "economic"
    # on 2026-08-24 at the project owner's explicit request, to test a
    # "conflict = ACLED + UNHCR" grouping -- SUPERSEDES the original
    # ACLED-only framing used in report.md sections 6-10 (documented there
    # as a deliberate choice at the time). Re-running --exclude-feature-
    # cluster conflict today excludes both sources together; report.md
    # section 12 explicitly separates "conflict (ACLED only)" and
    # "conflict (ACLED+UNHCR)" results rather than silently overwriting
    # the earlier numbers.
    "conflict": (
        "acled_event_count", "acled_fatalities",
        "total_refugees_hosted", "total_asylum_seekers_hosted", "idps_ethiopia",
    ),
}
# Autoregressive/target-memory features and assessment-recency metadata --
# not a substantive data-source cluster, kept in every ablation variant
# including the "all clusters" baseline.
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


# ACLED's 4 lagged windows (1/3/6/12 months back) alongside its own lag0
# columns (acled_event_count/acled_fatalities, already the correct
# lead-shifted vintage for the target month). A user hypothesis
# (report.md section 8): with only 3 conflict-source columns worth of
# raw signal (event count, fatalities, and the lead-shift itself) spread
# across 5 time windows (lag0/1/3/6/12), the 4 lagged variants may be
# redundant/noisy rather than informative -- testable in isolation from
# the full-cluster removal already covered in section 6/7.
ACLED_LAG_COLS = [
    "acled_event_count_lag1", "acled_fatalities_lag1",
    "acled_event_count_lag3", "acled_fatalities_lag3",
    "acled_event_count_lag6", "acled_fatalities_lag6",
    "acled_event_count_lag12", "acled_fatalities_lag12",
]
# All 10 raw-count ACLED columns (lag0 + the 4 lagged windows). Both are
# extreme-right-skewed non-negative counts (78%/86% zero zone-months for
# event_count/fatalities respectively, but max 71/1,287 in a single
# zone-month -- checked directly against data/processed/model_tables/,
# see report.md section 9) -- a textbook case for a log1p transform before
# feeding a tree-based model raw counts dominated by rare outliers.
ACLED_COUNT_COLS = ["acled_event_count", "acled_fatalities"] + ACLED_LAG_COLS
# NOTE (2026-08-24): pipelines/acled_conflict/compute_acled_admin2_monthly_features.py
# now also produces 30 disorder-type-disaggregated columns
# (acled_{event_count,fatalities}_{political_violence,demonstrations,
# strategic_developments}[_lag{1,3,6,12}]) -- these are picked up
# automatically by classify_feature_cluster() (prefix match on
# "acled_event_count"/"acled_fatalities" already covers them, so
# --exclude-feature-cluster conflict removes all 40 ACLED columns), but
# ACLED_LAG_COLS/ACLED_COUNT_COLS above deliberately stay scoped to only
# the original 10 total-only columns, so --exclude-acled-lags/
# --log-transform-acled keep meaning exactly what they meant when first
# run (report.md sections 8-9) rather than silently changing scope under
# an unchanged flag name. See report.md section 10 for the disaggregation
# test itself.

# A narrower cut within "climate" (report.md section 6.4): GLEAM SSMI/SPEI
# and GloFAS are the two climate sub-sources with an already-documented
# validation weakness specific to pastoral/lowland zones (GLEAM SSMI
# r=0.54-0.71 vs. the underlying raw soil moisture's r=0.91; GloFAS's own
# magnitude sanity check flagged unexpected pastoral/dry-lowland
# behaviour) -- section 6.4 found the whole climate cluster's net-negative
# effect concentrates almost entirely in pastoral zones in 2021. This lets
# CHIRPS/SPI/NDVI/teleconnections/WVG (which all validate well) stay in
# while testing whether GLEAM+GloFAS specifically are the drag, rather
# than climate broadly. Column list taken directly from
# data/processed/features/gleam_ssmi_spei_admin2_monthly.csv's own header
# (pet_mm/sm_root/precip_mm/spei_balance_mm/sm_mean_*/spei_balance_sum_*/
# ssmi_*/spei_*) plus every glofas_ column that survives into the model
# tables (glofas_exceed_2yr/20yr, glofas_source) -- confirmed no prefix
# collision with CHIRPS (total_rainfall_mm/wet_days/rainfall_sum_*/spi_*)
# or NDVI (ndvi*) columns.
GLEAM_GLOFAS_COLS = (
    "pet_mm", "sm_root", "precip_mm", "spei_balance_mm",
    "sm_mean_", "spei_balance_sum_", "ssmi_", "spei_",
    "glofas_",
)


def load_lead_table(model_tables_dir, lead):
    path = Path(model_tables_dir) / f"ethiopia_lead{lead:02d}.csv"
    df = pd.read_csv(path, low_memory=False)
    return df


def seasonality_predictions(train_labels, test_index):
    monthly = train_labels.groupby(train_labels.index.month).mean()
    monthly = monthly.reindex([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1])
    monthly = monthly.interpolate(method="linear", axis=0)
    monthly = monthly.iloc[:-1]
    values = monthly.reindex(test_index.month).to_numpy()
    return pd.Series(values, index=test_index)


def run_cell(df, cluster, lead, verbose=True, models_dir=None, target_mode="level",
             exclude_feature_cluster=None, exclude_acled_lags=False,
             log_transform_acled=False, exclude_gleam_glofas=False,
             train_end=None, test_start=None, test_end=None, exclude_years=None):
    sub = df[(df["dominant_livelihood_zone"] == cluster) & (df["observed"] == True)].copy()
    sub = sub.dropna(axis=1, how="all")
    sub["time"] = pd.to_datetime(sub["month"])
    custom_split = train_end is not None or test_start is not None
    if not custom_split:
        sub = sub[sub["split"].isin(["train", "test"])]
    # else: keep every row with a real (observed) target, including the
    # "beyond" split (2023+, normally dropped -- see report.md section 13)
    # -- train/test membership is computed from `time` below instead of
    # the fixed Busker-parity split column.
    if target_mode == "delta":
        # A delta target needs a persistence value to subtract -- drops the
        # ~2.5% of (mostly train-only) rows that are each zone's very first
        # observed row, before any lag exists. See report.md section 7.
        sub = sub[sub[BASE1_COL].notna()]
    sub = sub.sort_values(["time", "zone_code"]).reset_index(drop=True)

    meta = sub[["time", "zone_code", "zone_name", "split"]].copy()
    labels = sub[TARGET_COL]
    base1 = sub[BASE1_COL]

    frame = sub.copy()
    if exclude_feature_cluster is not None:
        # Accepts a single cluster name or a list of them (combined
        # ablation, e.g. ["climate", "conflict"]). Dropped pre-dummy so
        # season_system's whole one-hot family goes with it when
        # "agriculture" is excluded. ipc_lag1/BASE1_COL is never in
        # FEATURE_CLUSTER_PREFIXES (it's ALWAYS_KEEP) so it stays in the
        # feature matrix regardless of which cluster(s) are excluded.
        excluded = ({exclude_feature_cluster} if isinstance(exclude_feature_cluster, str)
                    else set(exclude_feature_cluster))
        excl_cols = [c for c in frame.columns if classify_feature_cluster(c) in excluded]
        frame = frame.drop(columns=excl_cols)
    if exclude_acled_lags:
        # Keeps acled_event_count/acled_fatalities (lag0, already the
        # correct lead-shifted vintage) -- only the 4 extra lagged windows
        # are dropped. See ACLED_LAG_COLS docstring.
        frame = frame.drop(columns=[c for c in ACLED_LAG_COLS if c in frame.columns])
    if exclude_gleam_glofas:
        # Narrower cut than --exclude-feature-cluster climate -- drops only
        # GLEAM SSMI/SPEI + GloFAS, keeps CHIRPS/NDVI/teleconnections/WVG.
        # See GLEAM_GLOFAS_COLS docstring and report.md section 6.4.
        frame = frame.drop(columns=[c for c in frame.columns if c.startswith(GLEAM_GLOFAS_COLS)])
    if log_transform_acled:
        # log1p is safe here -- ACLED counts/fatalities are always >= 0
        # (validated in pipelines/acled_conflict/validate_acled_output.py),
        # so no negative-input or log(0) issue. See ACLED_COUNT_COLS docstring.
        present = [c for c in ACLED_COUNT_COLS if c in frame.columns]
        frame[present] = np.log1p(frame[present])
    if "season_system" in frame.columns:
        frame = pd.get_dummies(frame, columns=["season_system"], prefix="season", prefix_sep="_")
    drop_cols = [c for c in IDENTITY_DROP + PROVENANCE_DROP if c in frame.columns]
    frame = frame.drop(columns=drop_cols)
    frame = frame.drop(columns=frame.select_dtypes(include=["object", "category"]).columns)
    features = frame.drop(columns=["time"], errors="ignore")
    bool_cols = features.select_dtypes(include="bool").columns
    if len(bool_cols):
        features[bool_cols] = features[bool_cols].astype(int)

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

    train_x, test_x = features[train_mask], features[test_mask]
    train_y, test_y = labels[train_mask], labels[test_mask]
    train_meta, test_meta = meta[train_mask].reset_index(drop=True), meta[test_mask].reset_index(drop=True)
    base1_train = base1[train_mask].reset_index(drop=True)
    base1_test = base1[test_mask].reset_index(drop=True)
    train_y = train_y.reset_index(drop=True)
    test_y = test_y.reset_index(drop=True)

    if target_mode == "delta":
        # Fit on target - persistence (what changed), then add persistence
        # back on at prediction time -- predictions/out stay on the same
        # absolute IPC scale as level mode, so scoring/onset code is unchanged.
        fit_y = train_y - base1_train
        model = XGBRegressor(**XGB_PARAMS).fit(train_x.reset_index(drop=True), fit_y)
        raw_predictions = model.predict(test_x.reset_index(drop=True))
        predictions = base1_test.to_numpy() + raw_predictions
    else:
        model = XGBRegressor(**XGB_PARAMS).fit(train_x.reset_index(drop=True), train_y)
        predictions = model.predict(test_x.reset_index(drop=True))

    if models_dir is not None:
        model.save_model(str(models_dir / f"xgb_{cluster}_lead{lead}.json"))

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
    parser.add_argument("--target-mode", choices=["level", "delta"], default="level",
                        help="'level' fits ipc_continuous directly (default, matches "
                             "experiment_1). 'delta' fits (ipc_continuous - ipc_lag1) "
                             "and adds ipc_lag1 back at prediction time -- see report.md "
                             "section 5.")
    parser.add_argument("--exclude-feature-cluster", type=str, default=None,
                        help="comma-separated: drop these whole domain feature cluster(s) "
                             "from the feature matrix (docs/dissertation_plan.md RQ1's "
                             "feature-cluster ablation), e.g. 'climate' or "
                             "'climate,conflict'. Choices: climate, agriculture, economic, "
                             "conflict. Default: none excluded (all clusters in). "
                             "Autoregressive/target-memory features (ipc_lag*, "
                             "ha_share_lag*, months_since_assessment) are never dropped "
                             "-- see report.md section 6.")
    parser.add_argument("--exclude-acled-lags", action="store_true",
                        help="drop only ACLED's 4 lagged windows "
                             "(acled_{event_count,fatalities}_lag{1,3,6,12}), keeping the "
                             "lag0 acled_event_count/acled_fatalities columns -- a finer-"
                             "grained test than --exclude-feature-cluster conflict, see "
                             "report.md section 8.")
    parser.add_argument("--log-transform-acled", action="store_true",
                        help="apply log1p to all 10 ACLED count/fatality columns (lag0 + "
                             "the 4 lagged windows) before fitting -- see report.md "
                             "section 9.")
    parser.add_argument("--exclude-gleam-glofas", action="store_true",
                        help="drop only GLEAM SSMI/SPEI + GloFAS columns, keeping "
                             "CHIRPS/NDVI/teleconnections/WVG -- a finer-grained test than "
                             "--exclude-feature-cluster climate, see report.md section 6.4.")
    parser.add_argument("--train-end", type=str, default=None,
                        help="YYYY-MM, inclusive. Overrides the default Busker-parity split "
                             "column with a date-based train/test split instead -- train = "
                             "[start, train-end]. Requires --test-start too (or leave test on "
                             "the default split column, an unusual combination). See "
                             "report.md section 13.")
    parser.add_argument("--test-start", type=str, default=None,
                        help="YYYY-MM, inclusive start of a date-based test window "
                             "(overrides the default split column, including rows from the "
                             "'beyond' split, normally dropped entirely). See report.md "
                             "section 13.")
    parser.add_argument("--test-end", type=str, default=None,
                        help="YYYY-MM, inclusive end of the date-based test window "
                             "(default: no upper bound, i.e. every available month).")
    parser.add_argument("--exclude-years", type=str, default=None,
                        help="comma-separated calendar years to drop entirely from the "
                             "date-based test window, e.g. '2025' (only meaningful together "
                             "with --test-start). See report.md section 13.")
    args = parser.parse_args(argv)

    leads = [int(x) for x in args.leads.split(",")] if args.leads else LEADS
    clusters = args.clusters.split(",") if args.clusters else CLUSTERS
    exclude_clusters = (args.exclude_feature_cluster.split(",")
                        if args.exclude_feature_cluster else None)
    exclude_years = ([int(y) for y in args.exclude_years.split(",")]
                     if args.exclude_years else None)
    if exclude_clusters is not None:
        valid = set(FEATURE_CLUSTER_PREFIXES)
        bad = [c for c in exclude_clusters if c not in valid]
        if bad:
            raise SystemExit(f"--exclude-feature-cluster: unknown cluster(s) {bad}, "
                             f"choices are {sorted(valid)}")

    if exclude_clusters is not None:
        # Loud completeness check, mirroring
        # pipelines/model_join/build_ethiopia_model_tables.py's own
        # classify_columns() convention -- a column classify_feature_cluster()
        # can't place is a silent scoping bug in this ablation, not a
        # harmless default.
        sample = load_lead_table(args.model_tables_dir, leads[0])
        known = set(IDENTITY_DROP) | set(PROVENANCE_DROP) | {TARGET_COL}
        unclassified = [
            c for c in sample.columns
            if c not in known and classify_feature_cluster(c) is None
        ]
        if unclassified:
            raise SystemExit(
                f"FEATURE_CLUSTER_PREFIXES/ALWAYS_KEEP_PREFIXES do not cover: "
                f"{sorted(unclassified)} -- fix classify_feature_cluster() before running "
                f"the ablation, a missed column silently changes what's excluded."
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = None
    if args.save_models:
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nfitting {len(clusters)} x {len(leads)} = "
          f"{len(clusters) * len(leads)} models")
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
                exclude_acled_lags=args.exclude_acled_lags,
                log_transform_acled=args.log_transform_acled,
                exclude_gleam_glofas=args.exclude_gleam_glofas,
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
        "xgb_params": XGB_PARAMS,
        "target_mode": args.target_mode,
        "exclude_feature_cluster": exclude_clusters,
        "exclude_acled_lags": args.exclude_acled_lags,
        "log_transform_acled": args.log_transform_acled,
        "exclude_gleam_glofas": args.exclude_gleam_glofas,
        "train_end": args.train_end,
        "test_start": args.test_start,
        "test_end": args.test_end,
        "exclude_years": exclude_years,
        "split_rule": (
            f"custom date-based split: train <= {args.train_end}, "
            f"test >= {args.test_start}"
            + (f" and <= {args.test_end}" if args.test_end else "")
            + (f", excluding years {exclude_years}" if exclude_years else "")
            + " (overrides the default split column, includes 'beyond' rows)"
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
