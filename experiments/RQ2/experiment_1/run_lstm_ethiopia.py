"""Fit and score a pooled-per-lead LSTM on the sequence caches built by
build_lstm_sequences.py, for direct comparison against
experiments/RQ1/experiment_2/run_model_ethiopia.py's XGBoost result on the
same Ethiopia dataset (RQ2 Experiment 1: architecture comparison).

Approved design (see conversation that set this up, 2026-08-24):
  - Sequence input (12-month sliding window per zone, raw unshifted panel),
    not XGBoost's flattened lag-engineered rows -- see
    build_lstm_sequences.py's own docstring for the full rationale.
  - One pooled LSTM per lead (7 total, all 92 zones together, livelihood
    zone one-hot included as a feature), not 21 separate cluster models --
    XGBoost's pastoral/agropastoral clusters only have ~150-2000 rows each,
    too thin for a from-scratch deep net. Metrics are broken out per
    cluster after scoring for direct comparability with XGBoost's
    metrics_per_cluster.csv.
  - Same target (ipc_continuous, level, not delta -- matches XGBoost's
    plain/default run, experiments/RQ1/experiment_2/unhcr_level/, since RQ2
    is an architecture bake-off on the same dataset, not a rerun of RQ1
    Experiment 2's own best-variant tuning) and the same two test windows:
    the project's default Busker-parity split (train <=2019-12, test
    2020-01..2022-12) and an extended window (test 2020-01..2024-12,
    same training data) -- scored from ONE trained model per lead, not
    retrained per window, since the training set is identical either way.
  - Same scoring functions (score(), crisis_onset_rates(), cont_calc()) as
    run_model_ethiopia.py, reimplemented here rather than imported --
    mirrors that script's own "self-contained per experiment folder"
    convention (see its docstring) plus these two scripts sit in different
    directories.

Missing-value handling: unlike XGBoost's native NaN handling, an LSTM
cannot take NaN inputs. Per-feature median imputation, fit on the TRAIN
split only (no leakage) and applied to train+test, followed by
train-fit standardization -- the same "impute at model-input time, not in
any pipeline output" posture already established for the Busker baseline
arm (CLAUDE.md Sec 7a: "he imputes everywhere... fills what remains with
the unit mean" -- median used here instead of mean, more robust to
ACLED/price outliers, otherwise the same category of necessary, disclosed,
model-input-stage-only imputation, not a violation of Sec 3.6's
pipeline-output no-imputation rule).

Usage:
    python run_lstm_ethiopia.py --sequences-dir sequences --out-dir <dir> \
        [--extra-test-start 2020-01 --extra-test-end 2024-12] [--save-models]
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)

LEADS = [0, 1, 2, 3, 4, 8, 12]
SEED = 42
CRISIS_THRESHOLD = 3.0
CRISIS_BUFFER = 0.05

HIDDEN_SIZE = 64
DENSE_SIZE = 32
DROPOUT = 0.2
LR = 1e-3
WEIGHT_DECAY = 1e-5
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 20
VAL_FRACTION = 0.15  # chronological tail of train, held out for early stopping


class LSTMRegressor(nn.Module):
    def __init__(self, n_features, hidden_size=HIDDEN_SIZE, dense_size=DENSE_SIZE, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_size, num_layers=1, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, 1),
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1]).squeeze(-1)


def load_sequences(sequences_dir, lead):
    data = np.load(Path(sequences_dir) / f"lead{lead:02d}.npz", allow_pickle=True)
    return {k: data[k] for k in data.files}


def chronological_val_split(time_arr, val_fraction):
    order = np.argsort(time_arr)
    n_val = max(1, int(len(order) * val_fraction))
    val_idx = set(order[-n_val:].tolist())
    is_val = np.array([i in val_idx for i in range(len(time_arr))])
    return is_val


def fit_lstm(train_x, train_y, val_x, val_y, n_features, verbose=True):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = LSTMRegressor(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    train_x_t = torch.from_numpy(train_x)
    train_y_t = torch.from_numpy(train_y)
    val_x_t = torch.from_numpy(val_x)
    val_y_t = torch.from_numpy(val_y)

    n = train_x_t.shape[0]
    best_val = np.inf
    best_state = None
    epochs_since_improve = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for start in range(0, n, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            opt.zero_grad()
            pred = model(train_x_t[idx])
            loss = loss_fn(pred, train_y_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x_t)
            val_loss = loss_fn(val_pred, val_y_t).item()
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= PATIENCE:
            break

    model.load_state_dict(best_state)
    if verbose:
        print(f"    trained {epoch + 1} epochs, best val MSE {best_val:.4f}")
    return model


def run_lead(sequences_dir, lead, extra_test_start=None, extra_test_end=None,
             models_dir=None, verbose=True, target_mode="level"):
    seq = load_sequences(sequences_dir, lead)
    X, y = seq["X"], seq["y"]
    zone_code = seq["zone_code"]
    time = pd.to_datetime(seq["time"])
    split = seq["split"]
    lhz = seq["dominant_livelihood_zone"]
    base1 = seq["base1"]

    train_mask = split == "train"
    n_features = X.shape[2]

    # Median impute (train-derived) then standardize (train-derived), both
    # computed over (example, timestep) pairs flattened together.
    train_flat = X[train_mask].reshape(-1, n_features)
    medians = np.nanmedian(train_flat, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_filled = np.where(np.isnan(X), medians, X)

    train_flat_filled = X_filled[train_mask].reshape(-1, n_features)
    mean = train_flat_filled.mean(axis=0)
    std = train_flat_filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_scaled = (X_filled - mean) / std

    train_x_all = X_scaled[train_mask].astype(np.float32)
    fit_target = (y - base1) if target_mode == "delta" else y
    train_y_all = fit_target[train_mask].astype(np.float32)
    train_time_all = time[train_mask]

    is_val = chronological_val_split(train_time_all.to_numpy(), VAL_FRACTION)
    fit_x, fit_y = train_x_all[~is_val], train_y_all[~is_val]
    val_x, val_y = train_x_all[is_val], train_y_all[is_val]

    model = fit_lstm(fit_x, fit_y, val_x, val_y, n_features, verbose=verbose)

    if models_dir is not None:
        torch.save(model.state_dict(), Path(models_dir) / f"lstm_lead{lead}.pt")

    windows = {"default": split == "test"}
    if extra_test_start is not None:
        extra_mask = np.asarray(time >= pd.Timestamp(extra_test_start))
        if extra_test_end is not None:
            extra_mask &= np.asarray(time <= pd.Timestamp(extra_test_end))
        windows["extended"] = extra_mask

    model.eval()
    out_frames = []
    for window_name, mask in windows.items():
        if mask.sum() == 0:
            continue
        test_x = torch.from_numpy(X_scaled[mask].astype(np.float32))
        with torch.no_grad():
            raw_preds = model(test_x).numpy()
        preds = base1[mask] + raw_preds if target_mode == "delta" else raw_preds
        frame = pd.DataFrame({
            "time": time[mask],
            "zone_code": zone_code[mask],
            "lhz": lhz[mask],
            "lead": lead,
            "window": window_name,
            "observed": y[mask],
            "prediction": preds,
            "base1_preds": base1[mask],
        })
        out_frames.append(frame)
        if verbose:
            base1_mae = mean_absolute_error(y[mask], base1[mask])
            base1_r2 = r2_score(y[mask], base1[mask])
            print(f"    lead {lead:2d} [{window_name:8}] n={mask.sum():4} "
                  f"MAE {mean_absolute_error(y[mask], preds):.4f} (persistence {base1_mae:.4f}) "
                  f"R2 {r2_score(y[mask], preds):.4f} (persistence {base1_r2:.4f})")

    return pd.concat(out_frames, ignore_index=True)


def score(predictions, group_cols):
    rows = []
    for keys, group in predictions.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        truth = group["observed"]
        record = dict(zip(group_cols, keys))
        record["n"] = len(group)
        for label, column in (("", "prediction"), ("_baseline", "base1_preds")):
            pred = group[column]
            record[f"mae{label}"] = mean_absolute_error(truth, pred)
            record[f"rmse{label}"] = mean_squared_error(truth, pred) ** 0.5
            record[f"r2{label}"] = r2_score(truth, pred)
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
        truth = group["observed"].reset_index(drop=True)
        record = dict(zip(by, keys))
        for label, column in (("", "prediction"), ("_baseline", "base1_preds")):
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
    parser.add_argument("--sequences-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--leads", type=str, default=None)
    parser.add_argument("--extra-test-start", type=str, default=None,
                         help="YYYY-MM, inclusive -- score the same trained model on a second, "
                              "wider test window in addition to the default split's test set.")
    parser.add_argument("--extra-test-end", type=str, default=None,
                         help="YYYY-MM, inclusive end of the extra test window.")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--target-mode", choices=["level", "delta"], default="level",
                         help="'level' fits ipc_continuous directly (default -- matches "
                              "XGBoost's plain baseline for RQ2's architecture comparison). "
                              "'delta' fits (ipc_continuous - ipc_lag1) and adds ipc_lag1 back "
                              "at prediction time, mirroring RQ1 Experiment 2's own delta "
                              "variant -- a follow-up test here, not the primary comparison, "
                              "after the plain level-mode LSTM was found to badly underfit the "
                              "persistence signal (see report.md).")
    args = parser.parse_args(argv)

    leads = [int(x) for x in args.leads.split(",")] if args.leads else LEADS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = None
    if args.save_models:
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nfitting {len(leads)} pooled LSTMs (one per lead)")
    all_predictions = []
    for lead in leads:
        print(f"  lead {lead}:")
        preds = run_lead(
            args.sequences_dir, lead,
            extra_test_start=args.extra_test_start, extra_test_end=args.extra_test_end,
            models_dir=models_dir, target_mode=args.target_mode,
        )
        all_predictions.append(preds)

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
        "sequences_dir": str(args.sequences_dir),
        "target_mode": args.target_mode,
        "architecture": {
            "type": "pooled per-lead LSTM", "hidden_size": HIDDEN_SIZE,
            "dense_size": DENSE_SIZE, "dropout": DROPOUT, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "val_fraction_chronological_tail_of_train": VAL_FRACTION,
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
        ["n", "mae", "mae_baseline", "r2", "r2_baseline"]
    ].round(4).to_string())

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
