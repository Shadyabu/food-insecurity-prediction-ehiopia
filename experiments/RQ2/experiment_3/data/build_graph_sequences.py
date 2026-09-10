"""Build (T, N, F) graph-snapshot sequences for the T-GCN arm of RQ2
Experiment 3, from the pre-lead-shift raw panel
(data/interim/model_join/ethiopia_raw_wide_panel.csv) -- same source and
same feature-exclusion logic as experiments/RQ2/experiment_1's LSTM arm
(build_lstm_sequences.py), reimplemented here rather than imported, per that
experiment's own established "self-contained per experiment folder"
precedent (see its report.md Sec 2).

Why a different shape than the LSTM's (N, 12, F) per-zone sequences: a GCN
needs every node present at every timestep so it can propagate through the
adjacency matrix, so the natural unit here is a (T, N, F) *graph snapshot*
sequence -- T consecutive months, all N zones, ending at origin_month --
not N independent per-zone sequences. Node order is fixed by
zone_order.json (alphabetical zone_code, matching adjacency.npy) so row i of
every (N, F) slice always refers to the same admin2 zone as row i of the
adjacency matrix.

The target y and train/test split are read from the matching
data/processed/model_tables/ethiopia_lead{lead}.csv rows, exactly like the
LSTM arm -- not re-derived -- so XGBoost/LSTM/T-GCN all predict the
identical (zone, target_month, lead) set (confirmed empirically before
writing this script: split is uniform across all 92 zones for any given
target month -- 0/49 months differ -- so it is stored once per snapshot,
not once per node).

Output: experiments/RQ2/experiment_3/data/sequences/lead{lead:02d}.npz per
lead:
  X                        (S, T, N, F) float32, NaN preserved
  y                        (S, N) float32, NaN where that zone has no valid
                            target that snapshot (should not happen given the
                            uniform-split check above, but kept as an explicit
                            mask rather than assumed)
  target_mask              (S, N) bool -- True where y is a real observed value
  origin_month / time      (S,) object date strings
  split                    (S,) object ("train"/"test"/"beyond")
  node_zone_code           (N,) object -- fixed node order, shared across every lead
  node_livelihood_zone     (N,) object -- static per-zone dominant_livelihood_zone
  base1                    (S, N) float32 -- ipc_lag1 per node (persistence baseline input)
plus a shared feature_names.json.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

LEADS = [0, 1, 2, 3, 4, 8, 12]
DEFAULT_WINDOW = 12

# Identical to experiments/RQ2/experiment_1/build_lstm_sequences.py -- kept
# in sync deliberately so T-GCN sees the exact same feature set as the LSTM
# arm, only reshaped differently.
IDENTITY_DROP_RAW = [
    "zone_code", "zone_id", "month", "zone_name", "assessment_month",
    "months_since_assessment", "observed", "split",
    "ipc_continuous", "ipc_phase_20pct", "pct_phase3plus", "pop_coverage",
    "ipc_continuous_area", "ha_share",
]
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
UNDATED_EXCLUDE = [
    "headline_cpi_undated", "headline_cpi_yoy_change_undated",
    "food_cpi_undated", "food_cpi_yoy_change_undated",
    "gdp_per_capita_undated", "gdp_per_capita_yoy_change_undated",
    "wvg_undated",
]
CATEGORICAL_COLS = ["dominant_livelihood_zone", "season_system"]


def load_raw_panel(path):
    df = pd.read_csv(path, low_memory=False)
    df["month"] = pd.to_datetime(df["month"])
    df = df.sort_values(["zone_code", "month"]).reset_index(drop=True)
    return df


def build_feature_frame(raw):
    drop_cols = [c for c in IDENTITY_DROP_RAW + PROVENANCE_DROP_RAW + UNDATED_EXCLUDE
                 if c in raw.columns]
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
    parser.add_argument("--zone-order", type=str, required=True,
                         help="path to zone_order.json (must match adjacency.npy's row/col order)")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    window = args.window

    zone_order = json.loads(Path(args.zone_order).read_text())
    n_nodes = len(zone_order)

    print(f"loading raw panel from {args.raw_panel}")
    raw = load_raw_panel(args.raw_panel)
    if sorted(raw["zone_code"].unique()) != sorted(zone_order):
        raise SystemExit("raw panel's zone set does not match zone_order.json -- rebuild adjacency first")

    feature_frame, feature_cols = build_feature_frame(raw)
    print(f"  {len(feature_cols)} features, panel shape {raw.shape}")

    ref_months = raw.loc[raw["zone_code"] == zone_order[0], "month"].reset_index(drop=True)
    for z in zone_order[1:]:
        zm = raw.loc[raw["zone_code"] == z, "month"].reset_index(drop=True)
        if not zm.equals(ref_months):
            raise SystemExit(f"zone {z} does not share the panel's common month grid -- "
                              f"positional windowing assumption is violated")
    month_index = {m: i for i, m in enumerate(ref_months)}
    print(f"  confirmed dense common month grid: {len(ref_months)} months, "
          f"{ref_months.min():%Y-%m}..{ref_months.max():%Y-%m}")

    # (n_zones_time_index, F) per node, in fixed zone_order -- stacked to
    # (n_months, N, F) so a snapshot window is a single positional slice.
    per_zone = []
    for z in zone_order:
        mask = (raw["zone_code"] == z).to_numpy()
        per_zone.append(feature_frame.loc[mask].to_numpy(dtype=np.float32))
    panel_TNF = np.stack(per_zone, axis=1)  # (n_months, N, F)

    node_livelihood_zone = np.array(
        [raw.loc[raw["zone_code"] == z, "dominant_livelihood_zone"].iloc[0] for z in zone_order],
        dtype=object,
    )

    (out_dir / "feature_names.json").write_text(json.dumps(feature_cols, indent=2))
    (out_dir / "zone_order.json").write_text(json.dumps(zone_order, indent=2))

    for lead in LEADS:
        mt_path = Path(args.model_tables_dir) / f"ethiopia_lead{lead:02d}.csv"
        mt = pd.read_csv(mt_path, low_memory=False)
        mt = mt[(mt["observed"] == True) & (mt["split"].isin(["train", "test", "beyond"]))].copy()
        mt["origin_month"] = pd.to_datetime(mt["origin_month"])
        mt["time"] = pd.to_datetime(mt["month"])

        # Pivot to one row per origin_month, columns = zone -> (target, split, base1).
        y_pivot = mt.pivot(index="origin_month", columns="zone_code", values="ipc_continuous")
        base1_pivot = mt.pivot(index="origin_month", columns="zone_code", values="ipc_lag1")
        split_by_origin = mt.drop_duplicates("origin_month").set_index("origin_month")["split"]
        time_by_origin = mt.drop_duplicates("origin_month").set_index("origin_month")["time"]

        y_pivot = y_pivot.reindex(columns=zone_order)
        base1_pivot = base1_pivot.reindex(columns=zone_order)

        X_list, y_list, mask_list, origin_list, time_list, split_list, base1_list = (
            [], [], [], [], [], [], []
        )
        n_dropped_no_window = 0
        for origin_month, y_row in y_pivot.iterrows():
            pos = month_index.get(origin_month)
            if pos is None or pos < window - 1:
                n_dropped_no_window += 1
                continue
            X_list.append(panel_TNF[pos - window + 1: pos + 1])  # (T, N, F)
            y_vals = y_row.to_numpy(dtype=np.float64)
            mask_list.append(~np.isnan(y_vals))
            y_list.append(np.nan_to_num(y_vals, nan=0.0).astype(np.float32))
            base1_list.append(base1_pivot.loc[origin_month].to_numpy(dtype=np.float32))
            origin_list.append(origin_month)
            time_list.append(time_by_origin.loc[origin_month])
            split_list.append(split_by_origin.loc[origin_month])

        X = np.stack(X_list).astype(np.float32)          # (S, T, N, F)
        y = np.stack(y_list).astype(np.float32)           # (S, N)
        target_mask = np.stack(mask_list)                 # (S, N) bool
        n_train = sum(1 for s in split_list if s == "train")
        n_test = sum(1 for s in split_list if s == "test")
        print(f"lead {lead:2d}: {X.shape[0]} snapshots (train={n_train}, test={n_test}), "
              f"dropped {n_dropped_no_window} snapshots with no full {window}-month window; "
              f"target coverage {target_mask.mean():.4f}")

        np.savez_compressed(
            out_dir / f"lead{lead:02d}.npz",
            X=X, y=y, target_mask=target_mask,
            base1=np.stack(base1_list).astype(np.float32),
            origin_month=np.array([str(t.date()) for t in origin_list], dtype=object),
            time=np.array([str(t.date()) for t in time_list], dtype=object),
            split=np.array(split_list, dtype=object),
            node_zone_code=np.array(zone_order, dtype=object),
            node_livelihood_zone=node_livelihood_zone,
        )

    print(f"\nwrote graph-sequence caches to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
