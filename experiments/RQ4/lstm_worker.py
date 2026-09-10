"""Standalone LSTM worker, run as a SEPARATE PROCESS from common.py's main
orchestration code.

Why this is a separate process rather than an in-process function in
common.py: empirically (2026-08-27), calling `nn.LSTM.load_state_dict()`
with real (pretrained, non-default) weight tensors segfaults (exit 139)
whenever xgboost/shap have already been imported earlier in the same
process -- a native OpenMP/BLAS runtime conflict between xgboost's bundled
library and PyTorch's LSTM CPU weight-loading codepath, confirmed
reproducible and independent of KMP_DUPLICATE_LIB_OK=TRUE. A fresh
nn.LSTM with randomly-initialized weights does NOT crash, which is what
made this easy to miss in a smaller smoke test. Isolating all LSTM work
(model loading, predict, SHAP) into a process that never imports
xgboost/tabicl sidesteps the conflict entirely -- this script only ever
imports numpy/pandas/torch/shap/sklearn, confirmed safe together.

Usage:
    python lstm_worker.py --lead 0 --mode predict [--mask-cols-file f.json] --out out.csv
    python lstm_worker.py --lead 0 --mode shap [--row-idx-file f.json] --out out.csv
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RQ4_DIR = Path(__file__).resolve().parent
REPO_ROOT = RQ4_DIR.parent.parent
LSTM_RUN_DIR = REPO_ROOT / "experiments" / "RQ2" / "experiment_1" / "lstm_delta"
LSTM_SEQ_DIR = REPO_ROOT / "experiments" / "RQ2" / "experiment_1" / "sequences"

SEED = 42

# 2026-09-02 addition: "extended" test window (2020-01..2024-12, date-based,
# matches run_lstm_ethiopia.py's own --extra-test-start/--extra-test-end
# convention) alongside the original "default" (2020-01..2022-12, the
# sequence file's own "split" column, unchanged behavior).
EXTENDED_TEST_START = pd.Timestamp("2020-01-01")
EXTENDED_TEST_END = pd.Timestamp("2024-12-31")


def load_feature_names():
    return json.loads((LSTM_SEQ_DIR / "feature_names.json").read_text())


def get_lstm_cell(lead, window="default"):
    seq = np.load(LSTM_SEQ_DIR / f"lead{lead:02d}.npz", allow_pickle=True)
    X, y = seq["X"], seq["y"]
    split = seq["split"]
    zone_code = seq["zone_code"]
    time = pd.to_datetime(seq["time"])
    base1 = seq["base1"]
    n_features = X.shape[2]

    train_mask = split == "train"
    if window == "extended":
        test_mask = np.asarray((time >= EXTENDED_TEST_START) & (time <= EXTENDED_TEST_END))
    else:
        test_mask = split == "test"

    train_flat = X[train_mask].reshape(-1, n_features)
    medians = np.nanmedian(train_flat, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_filled = np.where(np.isnan(X), medians, X)

    train_flat_filled = X_filled[train_mask].reshape(-1, n_features)
    mean = train_flat_filled.mean(axis=0)
    std = train_flat_filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    return {
        "X_raw": X, "y": y, "split": split, "zone_code": zone_code,
        "time": time, "base1": base1, "medians": medians, "mean": mean,
        "std": std, "train_mask": train_mask, "test_mask": test_mask,
        "feature_names": load_feature_names(),
    }


def lstm_scale(X_raw, medians, mean, std):
    X_filled = np.where(np.isnan(X_raw), medians, X_raw)
    return (X_filled - mean) / std


def mask_nan_then_scale(X_raw, feature_names, cols, medians, mean, std):
    X = X_raw.copy()
    idx = [feature_names.index(c) for c in cols if c in feature_names]
    if idx:
        X[:, :, idx] = np.nan
    return lstm_scale(X, medians, mean, std)


def load_model(lead, n_features=322):
    import torch
    import torch.nn as nn

    class LSTMRegressor(nn.Module):
        def __init__(self, n_features, hidden_size=64, dense_size=32, dropout=0.2):
            super().__init__()
            self.lstm = nn.LSTM(n_features, hidden_size, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, dense_size), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(dense_size, 1),
            )

        def forward(self, x):
            _, (h_n, _) = self.lstm(x)
            return self.head(h_n[-1]).squeeze(-1)

    model = LSTMRegressor(n_features)
    model.load_state_dict(torch.load(LSTM_RUN_DIR / "models" / f"lstm_lead{lead}.pt"))
    model.eval()
    return model


def predict(model, X_scaled, base1):
    import torch
    with torch.no_grad():
        raw = model(torch.from_numpy(X_scaled.astype(np.float32))).numpy()
    return base1 + raw


def run_predict(lead, mask_cols, row_idx, out_path, window="default"):
    lc = get_lstm_cell(lead, window=window)
    model = load_model(lead)
    X_test_raw = lc["X_raw"][lc["test_mask"]]
    time_test = lc["time"][lc["test_mask"]]
    zone_test = lc["zone_code"][lc["test_mask"]]
    y_test = lc["y"][lc["test_mask"]]
    base1_test = lc["base1"][lc["test_mask"]]
    if row_idx is not None:
        X_test_raw = X_test_raw[row_idx]
        time_test = np.asarray(time_test)[row_idx]
        zone_test = zone_test[row_idx]
        y_test = y_test[row_idx]
        base1_test = base1_test[row_idx]
    if mask_cols:
        Xs = mask_nan_then_scale(X_test_raw, lc["feature_names"], mask_cols, lc["medians"], lc["mean"], lc["std"])
    else:
        Xs = lstm_scale(X_test_raw, lc["medians"], lc["mean"], lc["std"])
    pred = predict(model, Xs, base1_test)
    out = pd.DataFrame({
        "time": pd.to_datetime(time_test), "zone_code": zone_test,
        "lead": lead, "lstm_pred": pred, "observed": y_test,
    })
    out.to_csv(out_path, index=False)


def run_shap(lead, row_idx, out_path, background_n=30, window="default"):
    import shap
    import torch

    lc = get_lstm_cell(lead, window=window)
    model = load_model(lead)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).unsqueeze(-1)

    wrapped = Wrap(model)

    X_test_raw = lc["X_raw"][lc["test_mask"]]
    if row_idx is not None:
        X_test_raw = X_test_raw[row_idx]
    X_test_scaled = lstm_scale(X_test_raw, lc["medians"], lc["mean"], lc["std"])
    X_train_scaled = lstm_scale(lc["X_raw"][lc["train_mask"]], lc["medians"], lc["mean"], lc["std"])

    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_train_scaled), size=min(background_n, len(X_train_scaled)), replace=False)
    background = torch.from_numpy(X_train_scaled[bg_idx].astype(np.float32))

    explainer = shap.GradientExplainer(wrapped, background)
    test_t = torch.from_numpy(X_test_scaled.astype(np.float32))
    sv = np.asarray(explainer.shap_values(test_t))  # (n, seq_len, n_features, 1)
    per_row = np.abs(sv[..., 0]).sum(axis=1)  # sum over 12 timesteps -> (n, n_features)
    importance = pd.Series(per_row.mean(axis=0), index=lc["feature_names"], name="importance")
    importance.index.name = "feature"
    out = importance.reset_index()
    out["n_rows"] = X_test_scaled.shape[0]
    out.to_csv(out_path, index=False)


def run_shap_batch(lead, row_idx_list, out_path, background_n=30, window="default"):
    """Like run_shap, but loops over MANY row-index draws (e.g. 100
    bootstrap resamples) inside one process -- model load + GradientExplainer
    background setup happens ONCE, not once per draw. Used by Experiment 2
    to avoid ~100x repeated subprocess-startup + setup cost per lead."""
    import shap
    import torch

    lc = get_lstm_cell(lead, window=window)
    model = load_model(lead)

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            return self.m(x).unsqueeze(-1)

    wrapped = Wrap(model)

    X_test_raw_full = lc["X_raw"][lc["test_mask"]]
    X_train_scaled = lstm_scale(lc["X_raw"][lc["train_mask"]], lc["medians"], lc["mean"], lc["std"])
    rng = np.random.default_rng(SEED)
    bg_idx = rng.choice(len(X_train_scaled), size=min(background_n, len(X_train_scaled)), replace=False)
    background = torch.from_numpy(X_train_scaled[bg_idx].astype(np.float32))
    explainer = shap.GradientExplainer(wrapped, background)

    rows = []
    for i, row_idx in enumerate(row_idx_list):
        X_test_raw = X_test_raw_full[np.asarray(row_idx)]
        X_test_scaled = lstm_scale(X_test_raw, lc["medians"], lc["mean"], lc["std"])
        test_t = torch.from_numpy(X_test_scaled.astype(np.float32))
        sv = np.asarray(explainer.shap_values(test_t))
        per_row = np.abs(sv[..., 0]).sum(axis=1)
        importance = per_row.mean(axis=0)
        for feat, val in zip(lc["feature_names"], importance):
            rows.append({"bootstrap_iter": i, "feature": feat, "importance": val, "n_rows": X_test_scaled.shape[0]})

    pd.DataFrame(rows).to_csv(out_path, index=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lead", type=int, required=True)
    p.add_argument("--mode", choices=["predict", "shap", "shap-batch"], required=True)
    p.add_argument("--window", choices=["default", "extended"], default="default")
    p.add_argument("--mask-cols-file", type=str, default=None)
    p.add_argument("--row-idx-file", type=str, default=None)
    p.add_argument("--row-idx-batch-file", type=str, default=None)
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    mask_cols = json.loads(Path(args.mask_cols_file).read_text()) if args.mask_cols_file else None
    row_idx = np.array(json.loads(Path(args.row_idx_file).read_text())) if args.row_idx_file else None

    if args.mode == "predict":
        run_predict(args.lead, mask_cols, row_idx, args.out, window=args.window)
    elif args.mode == "shap":
        run_shap(args.lead, row_idx, args.out, window=args.window)
    else:
        row_idx_list = json.loads(Path(args.row_idx_batch_file).read_text())
        run_shap_batch(args.lead, row_idx_list, args.out, window=args.window)


if __name__ == "__main__":
    main()
