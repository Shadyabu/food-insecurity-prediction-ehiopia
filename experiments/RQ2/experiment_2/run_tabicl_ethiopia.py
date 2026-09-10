"""Fit a pooled-per-lead TabICLv2 classifier on the Ethiopia model tables,
for direct comparison against experiments/RQ2/experiment_1's XGBoost/LSTM
result on the same dataset (RQ2 Experiment 2: architecture comparison, third
arm -- see docs/dissertation_plan.md RQ2).

Approved design (see conversation that set this up, 2026-08-25):
  - TabICLClassifier (soda-inria/tabicl) is a pretrained in-context-learning
    classifier, not a regressor -- fit directly on `ipc_phase_20pct` (the
    existing 1-5 population-share class column), not `ipc_continuous`. This
    is a real difference from XGBoost/LSTM's regress-then-round approach,
    not a style choice -- flagged wherever numbers are compared.
  - Pooled per lead (92 zones together, dominant_livelihood_zone one-hot as
    a feature), matching experiment_1's LSTM design, not XGBoost's 21
    separate cluster models -- confirmed empirically that agropastoral's
    ~315 raw training rows sit right at TabICL's documented ~300-row
    pretrained floor, the same thinness that already ruled out per-cluster
    models for the LSTM arm.
  - Feature count: the model tables carry ~310 candidate feature columns;
    TabICL's validated pretrained range tops out at ~100 features. Rather
    than PCA (loses interpretability) or feeding the full set (untested,
    out-of-distribution regime), the top TOP_N_FEATURES=80 columns are
    selected per lead by mean gain importance from the already-fitted,
    already-trusted XGBoost run
    (../../RQ1/experiment_2/unhcr_level/feature_importance.csv, averaged
    across its 3 per-cluster models for that lead), plus the
    dominant_livelihood_zone one-hot columns always included (no importance
    score exists for them in that per-cluster file, since XGBoost drops
    dominant_livelihood_zone as an identity column when fitting separately
    per cluster).
  - Missing values: TabICLClassifier's own built-in mean-imputation handles
    NaN (its documented default behaviour) -- deliberately NOT replicating
    the LSTM arm's manual train-fit median-impute step. A disclosed
    divergence, not an oversight: torch has no native NaN handling so the
    LSTM arm needed a manual step; TabICL claims to handle NaN natively, so
    this run tests that claim as designed rather than second-guessing it.
  - Windows: same two as LSTM/XGBoost -- default (train <=2019-12, test
    2020-01..2022-12, Busker-parity split) and extended (same training
    data, test 2020-01..2024-12) -- one fit scored both ways.
  - Hardware: device="cpu" forced explicitly. This machine's MPS backend
    (Apple M3, 24GB unified memory) OOMs at this dataset's scale (confirmed
    empirically: MPS tried to allocate ~9GB on top of ~20GB already
    resident and crashed; CPU handles the same shape in ~2 minutes/lead).
    No CUDA available. TabICL's own docs note CPU inference is supported
    but slower than GPU -- this run accepts that tradeoff rather than
    reducing dataset size to fit MPS's memory ceiling.
  - Scale mismatch, disclosed: TabICL's target/prediction are the discrete
    1-5 phase class; the persistence baseline (`base1_preds`) is therefore
    computed as ipc_lag1 rounded and clipped to the same 1-5 class scale
    (not the raw continuous ipc_lag1 used by the XGBoost/LSTM arms), so
    this script's own MAE/RMSE/R2 are on a different (discretized) scale
    from theirs and not directly comparable -- only accuracy/weighted-F1
    (and crisis-onset hit/false-alarm rates, threshold-based either way)
    are apples-to-apples across all three architectures. build_comparison.py
    handles this correctly since it discretizes every source's predictions
    to the same class scale before scoring F1/accuracy.

Usage:
    python run_tabicl_ethiopia.py --model-tables-dir ../../../data/processed/model_tables \
        --out-dir <dir> [--extra-test-start 2020-01 --extra-test-end 2024-12] \
        [--top-n-features 80]
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
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from tabicl import TabICLClassifier

LEADS = [0, 1, 2, 3, 4, 8, 12]
SEED = 42
CRISIS_THRESHOLD = 3.0
CRISIS_BUFFER = 0.05
TOP_N_FEATURES = 80

TARGET_COL = "ipc_phase_20pct"
CONT_BASE1_COL = "ipc_lag1"

DEFAULT_IMPORTANCE_PATH = "../../RQ1/experiment_2/unhcr_level/feature_importance.csv"

# Same exclusion lists as experiments/RQ1/experiment_2/run_model_ethiopia.py
# (reimplemented here, not imported -- these scripts sit in different
# experiment folders and each is meant to be self-contained, matching
# experiment_1's own LSTM-script convention). dominant_livelihood_zone is
# deliberately NOT in this list, unlike the XGBoost script -- this run
# needs it one-hot-encoded as a feature, since it pools all clusters into
# one model instead of fitting one model per cluster.
IDENTITY_DROP = [
    "zone_code", "zone_id", "month", "zone_name", "assessment_month",
    "origin_month", "lead_months", "observed", "split",
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


def load_lead_table(model_tables_dir, lead):
    path = Path(model_tables_dir) / f"ethiopia_lead{lead:02d}.csv"
    return pd.read_csv(path, low_memory=False)


def load_importance_ranking(importance_path, lead):
    """Mean gain importance per feature, averaged across the 3 per-cluster
    XGBoost models fit at this lead, descending. Used only to rank/select --
    the pooled TabICL model here is not the same model those importances
    came from, so these are a proxy for "generally informative", not a
    ground truth for what TabICL itself will find useful."""
    imp = pd.read_csv(importance_path)
    imp = imp[imp["lead"] == lead]
    ranked = imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
    return ranked.index.tolist()


def build_feature_frame(df):
    frame = df.copy()
    if "season_system" in frame.columns:
        frame = pd.get_dummies(frame, columns=["season_system"], prefix="season", prefix_sep="_")
    frame = pd.get_dummies(frame, columns=["dominant_livelihood_zone"], prefix="lhz", prefix_sep="_")
    drop_cols = [c for c in IDENTITY_DROP + PROVENANCE_DROP if c in frame.columns]
    frame = frame.drop(columns=drop_cols)
    frame = frame.drop(columns=frame.select_dtypes(include=["object", "category"]).columns)
    bool_cols = frame.select_dtypes(include="bool").columns
    if len(bool_cols):
        frame[bool_cols] = frame[bool_cols].astype(int)
    return frame


def select_features(frame, ranking, top_n):
    lhz_cols = sorted(c for c in frame.columns if c.startswith("lhz_"))
    ranked_present = [c for c in ranking if c in frame.columns and c not in lhz_cols]
    selected = ranked_present[:top_n]
    return selected + lhz_cols


def to_class(x):
    """Round/clip to the 1-5 phase scale, cast to int -- only safe on
    already-NaN-filtered input (the target, filtered upstream)."""
    return np.clip(np.round(x), 1, 5).astype(int)


def to_class_or_nan(x):
    """Same rounding/clipping as to_class() but keeps NaN as float rather
    than casting to int -- used for the persistence baseline, which is
    genuinely missing for each zone's first-ever observed assessment (92
    of 4,508 observed rows at lead0, confirmed empirically), unlike the
    target itself which is always filtered to non-null before this point."""
    return np.clip(np.round(x), 1, 5)


def run_lead(model_tables_dir, lead, importance_path, top_n,
             extra_test_start=None, extra_test_end=None, verbose=True):
    df = load_lead_table(model_tables_dir, lead)
    sub = df[(df["observed"] == True) & df[TARGET_COL].notna()].copy()
    sub["time"] = pd.to_datetime(sub["month"])
    sub = sub.sort_values(["time", "zone_code"]).reset_index(drop=True)

    meta = sub[["time", "zone_code", "zone_name", "split", "dominant_livelihood_zone"]].copy()
    labels = to_class(sub[TARGET_COL])
    base1_class = to_class_or_nan(sub[CONT_BASE1_COL])

    features_frame = build_feature_frame(sub)
    ranking = load_importance_ranking(importance_path, lead)
    selected = select_features(features_frame, ranking, top_n)
    X = features_frame[selected]

    train_mask = (meta["split"] == "train").to_numpy()
    train_x = X[train_mask].reset_index(drop=True)
    train_y = labels[train_mask]

    clf = TabICLClassifier(random_state=SEED, device="cpu")
    clf.fit(train_x, train_y)

    windows = {"default": (meta["split"] == "test").to_numpy()}
    if extra_test_start is not None:
        extra_mask = (meta["time"] >= pd.Timestamp(extra_test_start)).to_numpy()
        if extra_test_end is not None:
            extra_mask &= (meta["time"] <= pd.Timestamp(extra_test_end)).to_numpy()
        windows["extended"] = extra_mask

    out_frames = []
    for window_name, mask in windows.items():
        if mask.sum() == 0:
            continue
        test_x = X[mask].reset_index(drop=True)
        preds = clf.predict(test_x)
        frame = pd.DataFrame({
            "time": meta.loc[mask, "time"].to_numpy(),
            "zone_code": meta.loc[mask, "zone_code"].to_numpy(),
            "lhz": meta.loc[mask, "dominant_livelihood_zone"].to_numpy(),
            "lead": lead,
            "window": window_name,
            "observed": labels[mask],
            "prediction": preds,
            "base1_preds": base1_class[mask],
        })
        out_frames.append(frame)
        if verbose:
            truth, pred, base = frame["observed"], frame["prediction"], frame["base1_preds"]
            base_ok = base.notna()
            base_acc = accuracy_score(truth[base_ok], base[base_ok]) if base_ok.any() else np.nan
            base_f1 = (f1_score(truth[base_ok], base[base_ok], average="weighted", zero_division=0)
                       if base_ok.any() else np.nan)
            print(f"    lead {lead:2d} [{window_name:8}] n={mask.sum():4} n_feat={len(selected):3} "
                  f"acc {accuracy_score(truth, pred):.4f} (persistence {base_acc:.4f}) "
                  f"F1w {f1_score(truth, pred, average='weighted', zero_division=0):.4f} "
                  f"(persistence {base_f1:.4f})")

    return pd.concat(out_frames, ignore_index=True), len(selected)


def score(predictions, group_cols):
    """Per (label, column) NaN-masking mirrors run_model_ethiopia.py's own
    score() -- needed here because base1_preds is genuinely NaN for each
    zone's first-ever observed row (see to_class_or_nan docstring), unlike
    `prediction`, which TabICLClassifier always returns a value for."""
    rows = []
    for keys, group in predictions.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        truth = group["observed"]
        record = dict(zip(group_cols, keys))
        record["n"] = len(group)
        for label, column in (("", "prediction"), ("_baseline", "base1_preds")):
            pred = group[column]
            ok = pred.notna() & truth.notna()
            if ok.sum() < 1:
                for metric in ("accuracy", "f1_weighted", "mae", "rmse", "r2"):
                    record[f"{metric}{label}"] = np.nan
                continue
            t, p = truth[ok], pred[ok]
            record[f"accuracy{label}"] = accuracy_score(t, p)
            record[f"f1_weighted{label}"] = f1_score(t, p, average="weighted", zero_division=0)
            record[f"mae{label}"] = mean_absolute_error(t, p)
            record[f"rmse{label}"] = mean_squared_error(t, p) ** 0.5
            record[f"r2{label}"] = r2_score(t, p) if t.nunique() > 1 else np.nan
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


def crisis_onset_rates(predictions, by=("window", "lhz", "lead")):
    by = list(by)
    rows = []
    ordered = predictions.sort_values(by + ["zone_code", "time"])
    for keys, group in ordered.groupby(by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        truth = pd.Series(group["observed"].to_numpy(), dtype=float).reset_index(drop=True)
        record = dict(zip(by, keys))
        for label, column in (("", "prediction"), ("_baseline", "base1_preds")):
            preds = pd.Series(group[column].to_numpy(), dtype=float).reset_index(drop=True)
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
    parser.add_argument("--importance-path", type=str, default=DEFAULT_IMPORTANCE_PATH)
    parser.add_argument("--top-n-features", type=int, default=TOP_N_FEATURES)
    parser.add_argument("--leads", type=str, default=None)
    parser.add_argument("--extra-test-start", type=str, default=None,
                         help="YYYY-MM, inclusive -- score the same trained model on a second, "
                              "wider test window in addition to the default split's test set.")
    parser.add_argument("--extra-test-end", type=str, default=None,
                         help="YYYY-MM, inclusive end of the extra test window.")
    args = parser.parse_args(argv)

    leads = [int(x) for x in args.leads.split(",")] if args.leads else LEADS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nfitting {len(leads)} pooled TabICLv2 classifiers (one per lead)")
    all_predictions = []
    n_features_by_lead = {}
    for lead in leads:
        print(f"  lead {lead}:")
        preds, n_feat = run_lead(
            args.model_tables_dir, lead, args.importance_path, args.top_n_features,
            extra_test_start=args.extra_test_start, extra_test_end=args.extra_test_end,
        )
        all_predictions.append(preds)
        n_features_by_lead[lead] = n_feat

    predictions = pd.concat(all_predictions, ignore_index=True)

    per_unit = score(predictions, ["window", "lhz", "lead", "zone_code"])
    per_cluster = score(predictions, ["window", "lhz", "lead"])
    per_lead = score(predictions, ["window", "lead"])
    onsets = crisis_onset_rates(predictions)

    predictions.to_parquet(out_dir / "predictions.parquet", index=False)
    per_unit.to_csv(out_dir / "metrics_per_unit.csv", index=False)
    per_cluster.to_csv(out_dir / "metrics_per_cluster.csv", index=False)
    per_lead.to_csv(out_dir / "metrics_per_lead.csv", index=False)
    onsets.to_csv(out_dir / "crisis_onset_rates.csv", index=False)

    meta = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "model_tables_dir": str(args.model_tables_dir),
        "importance_path": str(args.importance_path),
        "top_n_features": args.top_n_features,
        "n_features_by_lead": n_features_by_lead,
        "architecture": {
            "type": "pooled per-lead TabICLv2Classifier",
            "package": "tabicl",
            "device": "cpu",
            "target_col": TARGET_COL,
            "seed": SEED,
        },
        "windows": {
            "default": "train <=2019-12, test 2020-01..2022-12 (Busker-parity split)",
            "extended": (f"same training data, test {args.extra_test_start}..{args.extra_test_end}"
                         if args.extra_test_start else None),
        },
        "leads": leads,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nPer-lead, pooled over all zones (model | persistence):")
    print(per_lead.set_index(["window", "lead"])[
        ["n", "accuracy", "accuracy_baseline", "f1_weighted", "f1_weighted_baseline"]
    ].round(4).to_string())

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
