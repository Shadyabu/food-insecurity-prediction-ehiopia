#!/usr/bin/env python3
"""
build_ipc_features.py
=====================
Stage 5: turn the admin2 IPC panel into Busker-faithful model tables.

    [3. AGGREGATE] population-weighted admin2 panel     (aggregate_fews_ipc.py)
    [5. EXPAND]    assessment-dated -> monthly           (this script)
    [5b. LAG]      four FEWS IPC memory features         (this script)
    [5c. SHIFT]    one table per lead time               (this script)

WHAT BUSKER DID, AND WHY THIS SCRIPT IS SHAPED THIS WAY
-------------------------------------------------------
From the paper:

  "Memory effects in the target variable FEWS IPC were accounted for by
   including values from 1, 4, and 8 months prior, along with the mean FEWS IPC
   value from the past 12 months."

  "Predictions were made for various lead times (0, 1, 2, 3, 4, 8, and 12
   months) using seven distinct XGBoost models. Each model was trained with
   specific lags corresponding to the respective lead time. To achieve these
   lags, the features were shifted in time by an amount of months equal to the
   lead time, before model training started."

Three consequences:

1. The modelling table is MONTHLY. A 1-month lag cannot exist otherwise. But
   FEWS NET assesses only 3-4 times a year, and Busker's released raster
   (fews_xr_CS.nc, 47 timesteps) is NOT monthly -- the expansion happened in
   code he did not release. The filling rule is therefore the one genuinely
   unknown part of the reproduction, and this script makes it explicit and
   configurable rather than burying it.

2. The TARGET stays at month t; the FEATURES are shifted back by the lead time.
   For a 3-month model, features at t-3 predict the target at t. This includes
   the IPC memory features, so a 3-month model sees IPC at t-4, t-7 and t-11,
   never anything from after t-3. Shifting the whole feature block at once is
   what keeps that guarantee.

3. Seven separate models, not one model with a lead-time column.

MONTHLY EXPANSION IS TWO DIFFERENT THINGS
------------------------------------------
Each CS record carries a validity window (projection_start..projection_end).
A February assessment covering February-April is FEWS NET's own statement about
all three months -- propagating across that window is using the data as
published, not inventing it. Propagating BEYOND the window is an assumption.

This script distinguishes them with an `observed` flag:

    observed = True   month falls inside a published validity window
    observed = False  month filled by carrying the last assessment forward

Keep both. Train on everything for the Busker reproduction; filter to
observed == True for the more honest evaluation. Same table, one flag, so the
two designs are directly comparable rather than being separate pipelines.

USAGE
-----
    python build_ipc_features.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent  # repo root (pipelines/ipc_target/ -> pipelines/ -> repo root)
PROCESSED_DIR = ROOT / "data" / "processed"
INTERIM_DIR = ROOT / "data" / "interim" / "ipc_target"

PANEL_PATH = PROCESSED_DIR / "features" / "ipc_target_admin2_panel.csv"
TIDY_PATH = INTERIM_DIR / "ipc_panel_long.csv"
OUT_DIR = INTERIM_DIR / "model_tables"

SCENARIO = "CS"                       # the target variable is the current situation
TARGET_CONTINUOUS = "ipc_continuous"  # Busker's regression target (MAE, R2)
TARGET_DISCRETE = "ipc_phase_20pct"   # for weighted F1 later

# Busker's memory features: IPC at 1, 4 and 8 months prior, plus the mean of the
# past 12 months.
LAG_MONTHS = [1, 4, 8]
ROLLING_MEAN_MONTHS = 12

# Busker's seven lead times.
LEAD_TIMES = [0, 1, 2, 3, 4, 8, 12]

# Busker trained on 2009-2019 and held out 2019-2022. The FDW extract for
# Ethiopia starts 2011-04, so the training window opens later; the hold-out
# boundary is kept identical for comparability.
TRAIN_END = "2019-12-31"
TEST_END = "2022-12-31"

# "window" propagates only across each assessment's published validity window
# and leaves the rest missing. "ffill" additionally carries the last assessment
# forward into uncovered months, which is what a monthly Busker-style table
# requires. Both are emitted; this sets which one `observed` gates on.
FILL_MODE = "ffill"

# Beyond this many months since the last assessment, a forward-filled value is
# too stale to be meaningful. Rows past it are dropped rather than fed to the
# model as if they were observations. Your 2025 gap runs 12 months.
MAX_FILL_MONTHS = 8


# --------------------------------------------------------------------------
# Load
# --------------------------------------------------------------------------

def load_panel():
    if not PANEL_PATH.exists():
        raise SystemExit(f"{PANEL_PATH} not found. Run aggregate_fews_ipc.py first.")
    df = pd.read_csv(PANEL_PATH, parse_dates=["time"])
    df = df[df["scenario"] == SCENARIO].copy()
    if df.empty:
        raise SystemExit(f"No {SCENARIO} rows in {PANEL_PATH}.")
    df["month"] = df["time"].dt.to_period("M")
    print(f"  panel: {len(df):,} rows, {df['zone_code'].nunique()} zones, "
          f"{df['month'].nunique()} assessed months "
          f"({df['month'].min()} .. {df['month'].max()})")
    return df


def load_validity_windows():
    """Map each assessment month to the last month it remains valid for.

    Taken from the tidy fnid-level panel, which retains projection_end; the
    aggregation step drops it. Uses the modal window per assessment because a
    handful of units occasionally carry an off-pattern window.
    """
    if not TIDY_PATH.exists():
        print(f"  {TIDY_PATH.name} not found -- treating every assessment as "
              f"valid for its own month only")
        return {}

    tidy = pd.read_csv(TIDY_PATH, low_memory=False,
                       parse_dates=["collection_date", "projection_end"])
    tidy = tidy[tidy["scenario"] == SCENARIO].dropna(
        subset=["collection_date", "projection_end"])
    if tidy.empty:
        return {}

    tidy["start"] = tidy["collection_date"].dt.to_period("M")
    tidy["end"] = tidy["projection_end"].dt.to_period("M")
    modal = tidy.groupby("start")["end"].agg(lambda s: s.mode().iloc[0])

    spans = (modal.index.to_timestamp().to_period("M"))
    lengths = [(modal.loc[s] - s).n + 1 for s in modal.index]
    print(f"  validity windows: {len(modal)} assessments, "
          f"length min {min(lengths)}, median {int(np.median(lengths))}, "
          f"max {max(lengths)} months")
    return modal.to_dict()


# --------------------------------------------------------------------------
# Monthly expansion
# --------------------------------------------------------------------------

def expand_to_monthly(panel, windows):
    """Build a complete zone x month grid and mark observed vs filled."""
    zones = panel[["zone_id", "zone_name", "zone_code"]].drop_duplicates()
    months = pd.period_range(panel["month"].min(), panel["month"].max(), freq="M")
    print(f"\n  expanding to {len(zones)} zones x {len(months)} months "
          f"= {len(zones) * len(months):,} rows")

    grid = pd.MultiIndex.from_product(
        [zones["zone_code"], months], names=["zone_code", "month"]
    ).to_frame(index=False)
    grid = grid.merge(zones, on="zone_code", how="left")

    value_cols = [c for c in [
        TARGET_CONTINUOUS, TARGET_DISCRETE, "pct_phase3plus", "pop_coverage",
        "ha_share", "ipc_continuous_area",
    ] if c in panel.columns]

    obs = panel[["zone_code", "month"] + value_cols].copy()
    obs["assessment_month"] = obs["month"]
    out = grid.merge(obs, on=["zone_code", "month"], how="left")

    out = out.sort_values(["zone_code", "month"]).reset_index(drop=True)

    # Carry the last assessment forward within each zone, tracking its age.
    grp = out.groupby("zone_code", sort=False)
    for col in value_cols + ["assessment_month"]:
        out[col] = grp[col].ffill()
    out["assessment_month"] = out["assessment_month"].astype("Period[M]")
    out["months_since_assessment"] = [
        (m - a).n if pd.notna(a) else np.nan
        for m, a in zip(out["month"], out["assessment_month"])
    ]

    # A month is "observed" if it falls inside the published validity window of
    # the assessment it inherited from -- i.e. FEWS NET said something about it.
    if windows:
        valid_end = out["assessment_month"].map(windows)
        out["observed"] = [
            bool(pd.notna(e) and m <= e) for m, e in zip(out["month"], valid_end)
        ]
    else:
        out["observed"] = out["months_since_assessment"] == 0

    n_obs = int(out["observed"].sum())
    n_filled = int(out[TARGET_CONTINUOUS].notna().sum()) - n_obs
    print(f"    observed (inside a published window): {n_obs:,} "
          f"({n_obs/len(out):.1%})")
    print(f"    forward-filled beyond window        : {n_filled:,} "
          f"({n_filled/len(out):.1%})")
    print(f"    never covered (before first / gaps) : "
          f"{int(out[TARGET_CONTINUOUS].isna().sum()):,}")

    if MAX_FILL_MONTHS is not None:
        stale = out["months_since_assessment"] > MAX_FILL_MONTHS
        print(f"    dropping {int(stale.sum()):,} rows more than "
              f"{MAX_FILL_MONTHS} months past their assessment (too stale)")
        out = out[~stale.fillna(True)]

    return out.reset_index(drop=True)


# --------------------------------------------------------------------------
# Lag features
# --------------------------------------------------------------------------

def _lag_and_rolling_mean(df, grp, value_col, prefix):
    """Shared shift/rolling-mean logic for a single column, used for both
    ipc_continuous and ha_share -- see add_memory_features()."""
    cols = []
    for lag in LAG_MONTHS:
        col = f"{prefix}_lag{lag}"
        df[col] = grp[value_col].shift(lag)
        cols.append(col)
    mean_col = f"{prefix}_mean{ROLLING_MEAN_MONTHS}"
    df[mean_col] = (
        grp[value_col].shift(1)
        .groupby(df["zone_code"], sort=False)
        .rolling(ROLLING_MEAN_MONTHS, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    cols.append(mean_col)
    return cols


def add_memory_features(df):
    """Busker's memory features, built on zone_code.

    zone_code is a stable panel key across all 15 years. fnid is not -- the FEWS
    NET mapping units were redrawn in 2019, 2020, 2021 and 2023, so a lag built
    at fnid level silently returns the wrong place or nothing at all across
    those boundaries. That is the whole reason the reprojection happens first.

    Two feature families, both lagged the same way:
    - ipc_lag{1,4,8} / ipc_mean12: FEWS IPC memory (the target's own history).
    - ha_share_lag{1,4,8} / ha_share_mean12: humanitarian food assistance.
      Busker et al. (2024) describe their assistance feature as the FEWS NET
      "!" flag marking areas that would be one phase worse without significant
      assistance, "aggregated to the administrative units using the same
      population-weighted averaging as deployed on the FEWS IPC current
      situation" and used as a previous-timestep (lagged) feature alongside
      the IPC memory features -- i.e. exactly ha_share (already computed in
      aggregate_fews_ipc.py from the is_allowing_for_assistance / HA layer),
      lagged the same way. This is not a separate pipeline; it is the same
      quantity already in the panel, just not yet lagged.
    """
    print("\n  building memory features on zone_code")
    df = df.sort_values(["zone_code", "month"]).reset_index(drop=True)
    grp = df.groupby("zone_code", sort=False)

    feature_cols = _lag_and_rolling_mean(df, grp, TARGET_CONTINUOUS, "ipc")

    if "ha_share" in df.columns:
        feature_cols += _lag_and_rolling_mean(df, grp, "ha_share", "ha_share")
    else:
        print("    ha_share not in panel -- skipping assistance memory features")

    for col in feature_cols:
        print(f"    {col}: {df[col].notna().mean():.1%} populated")

    # How often does the 1-month lag simply equal the target? Under monthly
    # expansion this is high by construction, which is exactly why the lagged
    # IPC dominated Busker's SHAP ranking -- and why the bar to beat is
    # persistence, not the majority class.
    same = (df["ipc_lag1"] == df[TARGET_CONTINUOUS]).mean()
    print(f"    ipc_lag1 identical to target in {same:.1%} of rows")

    return df, feature_cols


# --------------------------------------------------------------------------
# Lead-time tables
# --------------------------------------------------------------------------

def make_lead_table(df, feature_cols, lead):
    """Shift every feature back by `lead` months, keeping the target at t.

    Busker: "the features were shifted in time by an amount of months equal to
    the lead time, before model training started." Shifting the whole feature
    block together is what guarantees no post-origin information leaks in: at
    lead 3 the model sees IPC from t-4, t-7 and t-11, never t-1.
    """
    out = df.sort_values(["zone_code", "month"]).copy()
    grp = out.groupby("zone_code", sort=False)
    for col in feature_cols:
        out[col] = grp[col].shift(lead)
    out["lead_months"] = lead
    out["origin_month"] = out["month"] - lead
    return out


def split_label(month):
    ts = month.to_timestamp()
    if ts <= pd.Timestamp(TRAIN_END):
        return "train"
    if ts <= pd.Timestamp(TEST_END):
        return "test"
    return "beyond"


def persistence_baseline(df, lead):
    """MAE of predicting the target with the most recent value available.

    Reported per split, because Busker's headline MAE 0.35 is a HOLD-OUT number
    (2019-2022). A persistence figure computed over the whole panel is not
    comparable to it; only the test-split column is.

    Reported separately for observed and filled rows, because under monthly
    expansion a filled row's target is often literally equal to ipc_lag1 -- so
    including filled rows measures how well the fill rule reproduces itself, not
    how well anything predicts food insecurity.
    """
    out = {}
    sub = df.dropna(subset=[TARGET_CONTINUOUS, "ipc_lag1"])
    for split in ("train", "test"):
        s = sub[sub["split"] == split]
        o = s[s["observed"]]
        out[f"{split}_all"] = (float((s[TARGET_CONTINUOUS] - s["ipc_lag1"]).abs().mean())
                               if len(s) else np.nan)
        out[f"{split}_obs"] = (float((o[TARGET_CONTINUOUS] - o["ipc_lag1"]).abs().mean())
                               if len(o) else np.nan)
    return out


def main():
    print("=" * 74)
    print("IPC monthly expansion and Busker memory features")
    print(f"  fill mode        : {FILL_MODE}")
    print(f"  max fill months  : {MAX_FILL_MONTHS}")
    print(f"  lead times       : {LEAD_TIMES}")
    print("=" * 74)

    print("\n[1] Load")
    panel = load_panel()
    windows = load_validity_windows()

    print("\n[2] Monthly expansion")
    monthly = expand_to_monthly(panel, windows)

    print("\n[3] Memory features")
    monthly, feature_cols = add_memory_features(monthly)

    monthly["split"] = monthly["month"].map(split_label)
    print("\n  split sizes: " + ", ".join(
        f"{k}:{v:,}" for k, v in monthly["split"].value_counts().items()))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_path = OUT_DIR / "ipc_monthly_base.csv"
    monthly.to_csv(base_path, index=False)
    print(f"  wrote {base_path} ({len(monthly):,} rows)")

    print("\n[4] Lead-time tables and persistence baselines")
    print("     Busker's MAE 0.35 is a hold-out number, so compare it to the")
    print("     TEST columns only. 'obs' excludes forward-filled rows.")
    print(f"\n  {'lead':>5} {'rows':>9} {'train':>8} {'test':>8} "
          f"{'test all':>9} {'test obs':>9} {'train all':>10}")
    summary = []
    for lead in LEAD_TIMES:
        table = make_lead_table(monthly, feature_cols, lead)
        usable = table.dropna(subset=[TARGET_CONTINUOUS] + feature_cols)
        path = OUT_DIR / f"ipc_lead{lead:02d}.csv"
        usable.to_csv(path, index=False)

        p = persistence_baseline(table, lead)
        n_train = int((usable["split"] == "train").sum())
        n_test = int((usable["split"] == "test").sum())
        print(f"  {lead:>5} {len(usable):>9,} {n_train:>8,} {n_test:>8,} "
              f"{p['test_all']:>9.4f} {p['test_obs']:>9.4f} {p['train_all']:>10.4f}")
        summary.append(p)

    mean_test_all = float(np.nanmean([s["test_all"] for s in summary]))
    mean_test_obs = float(np.nanmean([s["test_obs"] for s in summary]))
    print(f"\n  persistence averaged across the seven lead times (hold-out):")
    print(f"    all rows      : MAE {mean_test_all:.4f}")
    print(f"    observed only : MAE {mean_test_obs:.4f}")
    print(f"    Busker (Horn of Africa, 2019-2022 hold-out): MAE 0.35")
    if mean_test_all < 0.35:
        print(f"    -> persistence alone is below 0.35 on the all-rows table.")
        print(f"       Different region and target distribution, so this is NOT")
        print(f"       a like-for-like refutation -- but it does mean the")
        print(f"       reproduction must report persistence alongside the model,")
        print(f"       or the model's contribution cannot be read from MAE.")

    print(f"\n  wrote {len(LEAD_TIMES)} lead-time tables to {OUT_DIR}")
    print("\n" + "=" * 74)
    print("Target and memory features are ready. Next: join the climate,")
    print("market and conflict features on (zone_code, month), then apply the")
    print("SAME per-lead shift to them before training.")
    print("Busker's reference result: MAE 0.35 average across lead times.")
    print("=" * 74)


if __name__ == "__main__":
    main()