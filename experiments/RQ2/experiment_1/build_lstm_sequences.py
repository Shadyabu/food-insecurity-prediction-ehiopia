"""Build 12-month sliding-window feature sequences per zone for the LSTM arm
of RQ2 Experiment 1, from the pre-lead-shift raw panel
(data/interim/model_join/ethiopia_raw_wide_panel.csv) -- NOT from the
already lead-shifted, lag-engineered data/processed/model_tables/ used by
XGBoost (experiments/RQ1/experiment_2/run_model_ethiopia.py).

Why a different input representation for the LSTM: the model_tables give
each row a single flattened snapshot of hand-engineered lag/rolling
features (ipc_lag1, rainfall_sum_3m, ...) -- there is no real month-to-month
sequence for a recurrent architecture to learn from. This script instead
gives the LSTM an actual sequence: for each (zone, origin_month) prediction
origin, the 12 consecutive raw monthly feature snapshots ending at
origin_month (origin_month-11 .. origin_month), each already carrying its
own natural-month feature values (including that source's own lag columns,
e.g. ndvi_lag1, computed exactly as of that snapshot's own month -- no
leakage, since every value in the window is knowable at or before
origin_month). The target (ipc_continuous at origin_month + lead) and the
train/test split are NOT re-derived here -- they are read directly from
data/processed/model_tables/ethiopia_lead{lead}.csv, so the LSTM predicts
the exact same (zone, target_month, lead) rows XGBoost does, just from a
different feature representation. This keeps the target/split leakage-safe
handling (CLAUDE.md Sec 3.2) identical between the two architectures --
only the input shape differs, and that difference is the point of RQ2's
architecture comparison, not an accident.

The raw panel is a complete, gapless (zone x month) rectangle (92 zones x
183 months, 2011-04..2026-06) -- confirmed before writing this script -- so
a 12-month window is just a positional slice, no month-arithmetic/gap
handling needed. A real, disclosed limitation of this design: some higher
-lead origin_months (lead 8/12 can reach back to 2010-04/2010-08) predate
the panel's 2011-04 start, so no full 12-month window exists for them --
those rows are dropped (not padded), which shrinks the *train* set at
higher leads (2852 -> 2116 rows by lead 12) but leaves the test set (828
rows, all >= 2020-01) completely untouched at every lead -- checked
directly before writing this script, not assumed.

Output: experiments/RQ2/experiment_1/sequences/lead{lead:02d}.npz per lead,
containing X (N, 12, F) float32 (NaN preserved -- imputation/scaling is a
train-set-dependent decision left to run_lstm_ethiopia.py, not baked in
here so the same cache can serve multiple train/test window choices),
y (N,) float32, zone_code/time/split/dominant_livelihood_zone (N,) object
arrays, plus a shared feature_names.json (identical column order across
every lead, since the feature set is fixed by the raw panel, not by lead).

Usage:
    python build_lstm_sequences.py --raw-panel <path> --model-tables-dir <dir> \
        --out-dir <dir> [--window 12]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = [0, 1, 2, 3, 4, 8, 12]
DEFAULT_WINDOW = 12

# Same identity/target/metadata exclusion as
# experiments/RQ1/experiment_2/run_model_ethiopia.py's IDENTITY_DROP, minus
# columns that don't exist in the raw (pre-lead-shift) panel and plus
# zone_id (an arbitrary numeric ID, would leak zone identity spuriously if
# left in as a "feature" -- dropped there too, just implicitly, since it's
# never in FEATURE_CLUSTER_PREFIXES and gets swept by the object/category
# drop; here it's float/int so needs an explicit drop).
IDENTITY_DROP_RAW = [
    "zone_code", "zone_id", "month", "zone_name", "assessment_month",
    "months_since_assessment", "observed", "split",
    "ipc_continuous", "ipc_phase_20pct", "pct_phase3plus", "pop_coverage",
    "ipc_continuous_area", "ha_share",
]
# Same PROVENANCE_DROP as run_model_ethiopia.py.
PROVENANCE_DROP_RAW = [
    "maize_yield_status", "maize_yield_note", "wheat_yield_status", "wheat_yield_note",
    "sorghum_yield_status", "sorghum_yield_note", "teff_yield_status", "teff_yield_note",
    "headline_cpi_yoy_change_source", "food_cpi_yoy_change_source",
    "headline_cpi_chainlinked_source", "food_cpi_chainlinked_source",
    "glofas_source_lead1", "glofas_source_lead3",
    "seasonal_target_season_lead1", "seasonal_source_lead1",
    "seasonal_target_season_lead3", "seasonal_source_lead3",
    "usgs_gefs_issue_date_lead1", "usgs_gefs_source_lead1", "usgs_gefs_anom_units_lead1",
]
# Busker-fidelity-only snapshots (pipelines/model_join/build_ethiopia_model_tables.py's
# own EXCLUDED group, 7 cols) -- never reach data/processed/model_tables/, so
# dropped here too for exact feature-set parity with what XGBoost sees.
UNDATED_EXCLUDE = [
    "headline_cpi_undated", "headline_cpi_yoy_change_undated",
    "food_cpi_undated", "food_cpi_yoy_change_undated",
    "gdp_per_capita_undated", "gdp_per_capita_yoy_change_undated",
    "wvg_undated",
]
# Static categoricals -- one-hot encoded separately (constant across the 12
# timesteps of any one window; dominant_livelihood_zone is included so the
# pooled per-lead LSTM can learn cluster-specific behaviour despite training
# on all 92 zones together, per the approved RQ2 design decision).
CATEGORICAL_COLS = ["dominant_livelihood_zone", "season_system"]


def load_raw_panel(path):
    df = pd.read_csv(path, low_memory=False)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["zone_code", "month"]).reset_index(drop=True)
    return df


# Diagnostic-only addition (2026-09-02, RQ2 Experiment 5 2026-window
# ensemble check) -- same "conflict" cluster membership as
# run_model_ethiopia.py's FEATURE_CLUSTER_PREFIXES post-2026-08-24
# broadening (ACLED + UNHCR), matched by prefix here rather than importing
# classify_feature_cluster() (that function classifies already lead-shifted
# model_tables column names, e.g. "acled_event_count_lag1"; the raw panel's
# own column names are the same base names, so the same prefixes apply).
CONFLICT_PREFIXES = (
    "acled_", "total_refugees_hosted", "total_asylum_seekers_hosted", "idps_ethiopia",
)


def build_feature_frame(raw, exclude_conflict=False):
    drop_cols = [c for c in IDENTITY_DROP_RAW + PROVENANCE_DROP_RAW + UNDATED_EXCLUDE
                 if c in raw.columns]
    if exclude_conflict:
        drop_cols += [c for c in raw.columns if c.startswith(CONFLICT_PREFIXES)]
    frame = raw.drop(columns=drop_cols)
    frame = pd.get_dummies(frame, columns=CATEGORICAL_COLS, prefix=CATEGORICAL_COLS)
    bool_cols = frame.select_dtypes(include="bool").columns
    frame[bool_cols] = frame[bool_cols].astype(np.float32)
    remaining_object = frame.select_dtypes(include=["object", "category"]).columns
    if len(remaining_object):
        raise SystemExit(f"unexpected non-numeric columns survived: {list(remaining_object)}")
    feature_cols = list(frame.columns)
    return frame.astype(np.float32), feature_cols


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-panel", type=str, required=True)
    parser.add_argument("--model-tables-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--exclude-conflict", action="store_true",
                         help="Drop acled_*/UNHCR displacement columns before windowing -- "
                              "diagnostic-only variant for the 2026-09-02 ensemble check.")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window = args.window

    print(f"loading raw panel from {args.raw_panel}")
    raw = load_raw_panel(args.raw_panel)
    feature_frame, feature_cols = build_feature_frame(raw, exclude_conflict=args.exclude_conflict)
    print(f"  {len(feature_cols)} features, panel shape {raw.shape}")

    zones = raw["zone_code"].unique()
    months_per_zone = raw.groupby("zone_code")["month"].apply(lambda s: s.reset_index(drop=True))
    ref_months = raw.loc[raw["zone_code"] == zones[0], "month"].reset_index(drop=True)
    for z in zones[1:]:
        zm = raw.loc[raw["zone_code"] == z, "month"].reset_index(drop=True)
        if not zm.equals(ref_months):
            raise SystemExit(f"zone {z} does not share the panel's common month grid -- "
                              f"positional windowing assumption is violated")
    month_index = {m: i for i, m in enumerate(ref_months)}
    print(f"  confirmed dense common month grid: {len(ref_months)} months, "
          f"{ref_months.min():%Y-%m}..{ref_months.max():%Y-%m}")

    zone_arrays = {}
    for z in zones:
        mask = (raw["zone_code"] == z).to_numpy()
        zone_arrays[z] = feature_frame.loc[mask].to_numpy(dtype=np.float32)

    (out_dir / "feature_names.json").write_text(json.dumps(feature_cols, indent=2))

    for lead in LEADS:
        mt_path = Path(args.model_tables_dir) / f"ethiopia_lead{lead:02d}.csv"
        mt = pd.read_csv(mt_path, low_memory=False)
        # Keep 'beyond' (2023+) rows too, not just train/test -- needed so a
        # wider test window (e.g. 2020-2024) has real post-2022 rows to
        # score against; run_lstm_ethiopia.py still trains on split=="train"
        # only. Excludes only the ~276 reintroduced staleness-gap rows
        # (NaN target/split, CLAUDE.md model_join section) via observed==True.
        mt = mt[(mt["observed"] == True) & (mt["split"].isin(["train", "test", "beyond"]))].copy()
        mt["origin_month"] = pd.to_datetime(mt["origin_month"])
        mt["time"] = pd.to_datetime(mt["month"])

        X_list, y_list, zone_list, time_list, split_list, lhz_list, origin_list, base1_list = (
            [], [], [], [], [], [], [], []
        )
        n_dropped_no_window = 0
        for row in mt.itertuples(index=False):
            pos = month_index.get(row.origin_month)
            if pos is None or pos < window - 1:
                n_dropped_no_window += 1
                continue
            arr = zone_arrays[row.zone_code][pos - window + 1: pos + 1]
            X_list.append(arr)
            y_list.append(row.ipc_continuous)
            zone_list.append(row.zone_code)
            time_list.append(row.time)
            split_list.append(row.split)
            lhz_list.append(row.dominant_livelihood_zone)
            origin_list.append(row.origin_month)
            base1_list.append(row.ipc_lag1)

        X = np.stack(X_list).astype(np.float32)
        y = np.array(y_list, dtype=np.float32)
        n_train = sum(1 for s in split_list if s == "train")
        n_test = sum(1 for s in split_list if s == "test")
        print(f"lead {lead:2d}: {X.shape[0]} sequences (train={n_train}, test={n_test}), "
              f"dropped {n_dropped_no_window} rows with no full {window}-month window")

        np.savez_compressed(
            out_dir / f"lead{lead:02d}.npz",
            X=X, y=y,
            zone_code=np.array(zone_list, dtype=object),
            time=np.array([str(t.date()) for t in time_list], dtype=object),
            origin_month=np.array([str(t.date()) for t in origin_list], dtype=object),
            split=np.array(split_list, dtype=object),
            dominant_livelihood_zone=np.array(lhz_list, dtype=object),
            base1=np.array(base1_list, dtype=np.float32),
        )

    print(f"\nwrote sequence caches to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
