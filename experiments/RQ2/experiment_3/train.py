"""Train and score T-GCN (+ GCN-only / LSTM-only ablations) on the graph
-snapshot sequence caches built by data/build_graph_sequences.py, for RQ2
Experiment 3 -- Wubetie et al. (2026) T-GCN, directly comparable to
experiments/RQ2/experiment_1's XGBoost/LSTM results via the same discrete
lead grid (0,1,2,3,4,8,12) and the same Busker-parity default split
(train <=2019-12, test 2020-01..2022-12), decisions confirmed with the
project owner 2026-08-25.

One model trained per (lead, architecture, head) combination -- like
experiment_1's "one pooled LSTM per lead", not 21 cluster-specific models,
for the same reason (too few rows per livelihood-zone cluster for a
from-scratch deep net); metrics broken out per livelihood zone after
scoring, via evaluate.py.

## Transductive masking (spec Sec 5)

The paper's own "transductive" setup masks *nodes* within a single graph
snapshot (some counties labeled train, some test, same snapshot). This
project's split is not a node partition -- every one of the 92 admin2 zones
is present in every snapshot's forward pass, always (there is no "test
node" concept here, only a "test month") -- so the equivalent, and
stronger, guarantee implemented here is: **test/"beyond"-split snapshots
never enter a forward pass during training at all**, not just their loss.
Only train-split snapshots (minus the chronological validation tail, held
out for early stopping) are ever passed through the model with
`torch.no_grad()` off. The full graph structure (all N nodes, the fixed
adjacency) is used in every one of those forward passes, matching the
spec's "full graph structure used in every forward pass" -- just scoped to
the train split's own months, since that is what "transductive" can
actually mean once the node dimension is never partitioned.

## Missing-value handling

Same posture as experiments/RQ2/experiment_1/run_lstm_ethiopia.py: neural
nets can't take NaN. Train-split-derived per-feature median imputation,
then train-split-derived standardization, both applied to train+val+test
-- a disclosed model-input-stage-only transform, not a pipeline change
(CLAUDE.md Sec 3.6 no-imputation rule applies to pipeline outputs, not to a
model's own input layer; see CLAUDE.md Sec 7a for the precedent this
follows).

Usage:
    python train.py --sequences-dir data/sequences --adjacency data/adjacency.npy \
        --out-dir runs/tgcn_ce --architecture tgcn --head ce \
        --num-layers 4 --hidden-units 64
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate import evaluate_all, weighted_f1
from models.baselines import GCNOnly, LSTMOnly
from models.graph import load_normalized_adjacency
from models.heads import build_head, discretize_ipc
from models.tgcn import TGCN

LEADS = [0, 1, 2, 3, 4, 8, 12]
SEED = 42

DEFAULTS = dict(
    num_layers=4, hidden_units=64, dropout=0.2, lr=1e-3, weight_decay=1e-5,
    max_epochs=3000, patience=20, val_fraction=0.15,
)


def build_backbone(architecture: str, in_dim: int, hidden_units: int, num_layers: int, dropout: float):
    if architecture == "tgcn":
        return TGCN(in_dim, hidden_units=hidden_units, num_gcn_layers=num_layers, dropout=dropout)
    if architecture == "gcn_only":
        return GCNOnly(in_dim, hidden_units=hidden_units, num_layers=num_layers, dropout=dropout)
    if architecture == "lstm_only":
        return LSTMOnly(in_dim, hidden_units=hidden_units, num_layers=1, dropout=dropout)
    raise ValueError(f"unknown architecture: {architecture!r}")


def chronological_val_split(time_arr: np.ndarray, val_fraction: float) -> np.ndarray:
    order = np.argsort(time_arr)
    n_val = max(1, int(len(order) * val_fraction))
    val_idx = set(order[-n_val:].tolist())
    return np.array([i in val_idx for i in range(len(time_arr))])


def masked_forward(backbone, head, A_hat, X, y_continuous, mask):
    """X: (S, T, N, F); y_continuous/mask: (S, N). Returns (logits_flat, y_class_flat) over masked entries."""
    h = backbone(A_hat, X)               # (S, N, hidden)
    logits = head(h)                     # (S, N, C) or (S, N, K-1)
    logits_flat = logits.reshape(-1, logits.shape[-1])[mask.reshape(-1)]
    y_class_flat = discretize_ipc(y_continuous.reshape(-1))[mask.reshape(-1)]
    return logits_flat, y_class_flat


def fit(backbone, head, A_hat, train_X, train_y, train_mask, val_X, val_y, val_mask,
        lr, weight_decay, max_epochs, patience, verbose=True):
    torch.manual_seed(SEED)
    opt = torch.optim.Adam(
        list(backbone.parameters()) + list(head.parameters()), lr=lr, weight_decay=weight_decay
    )

    best_val_f1 = -np.inf
    best_state = None
    epochs_since_improve = 0
    epoch = 0
    for epoch in range(max_epochs):
        backbone.train()
        head.train()
        opt.zero_grad()
        logits, y_class = masked_forward(backbone, head, A_hat, train_X, train_y, train_mask)
        loss = head.loss(logits, y_class)
        loss.backward()
        opt.step()

        backbone.eval()
        head.eval()
        with torch.no_grad():
            val_logits, val_y_class = masked_forward(backbone, head, A_hat, val_X, val_y, val_mask)
            val_pred = head.predict_class(val_logits)
            val_f1 = weighted_f1(val_y_class.numpy(), val_pred.numpy())

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state = {
                "backbone": {k: v.clone() for k, v in backbone.state_dict().items()},
                "head": {k: v.clone() for k, v in head.state_dict().items()},
            }
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
        if epochs_since_improve >= patience:
            break

    backbone.load_state_dict(best_state["backbone"])
    head.load_state_dict(best_state["head"])
    if verbose:
        print(f"    trained {epoch + 1} epochs, best val weighted F1 {best_val_f1:.4f}")
    return backbone, head


def run_lead(sequences_dir, adjacency_path, lead, architecture, head_kind,
             num_layers, hidden_units, dropout, lr, weight_decay, max_epochs, patience,
             val_fraction, models_dir=None, verbose=True):
    A_hat, zone_order = load_normalized_adjacency(adjacency_path)

    data = np.load(Path(sequences_dir) / f"lead{lead:02d}.npz", allow_pickle=True)
    X, y, target_mask = data["X"], data["y"], data["target_mask"]
    split = data["split"]
    time = pd.to_datetime(data["time"])
    node_zone_code = data["node_zone_code"]
    node_lhz = data["node_livelihood_zone"]

    if list(node_zone_code) != zone_order:
        raise SystemExit("sequence cache node order does not match adjacency zone_order -- rebuild one of them")

    n_features = X.shape[-1]
    train_snapshot_mask = split == "train"

    train_flat = X[train_snapshot_mask].reshape(-1, n_features)
    medians = np.nanmedian(train_flat, axis=0)
    medians = np.where(np.isnan(medians), 0.0, medians)
    X_filled = np.where(np.isnan(X), medians, X)

    train_flat_filled = X_filled[train_snapshot_mask].reshape(-1, n_features)
    mean = train_flat_filled.mean(axis=0)
    std = train_flat_filled.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    X_scaled = ((X_filled - mean) / std).astype(np.float32)

    is_val = chronological_val_split(time[train_snapshot_mask].to_numpy(), val_fraction)
    train_idx = np.where(train_snapshot_mask)[0]
    fit_idx = train_idx[~is_val]
    val_idx = train_idx[is_val]

    X_t = torch.from_numpy(X_scaled)
    y_t = torch.from_numpy(y.astype(np.float32))
    mask_t = torch.from_numpy(target_mask)

    backbone = build_backbone(architecture, n_features, hidden_units, num_layers, dropout)
    head = build_head(head_kind, hidden_units)

    backbone, head = fit(
        backbone, head, A_hat,
        X_t[fit_idx], y_t[fit_idx], mask_t[fit_idx],
        X_t[val_idx], y_t[val_idx], mask_t[val_idx],
        lr=lr, weight_decay=weight_decay, max_epochs=max_epochs, patience=patience, verbose=verbose,
    )

    if models_dir is not None:
        torch.save(
            {"backbone": backbone.state_dict(), "head": head.state_dict()},
            Path(models_dir) / f"{architecture}_{head_kind}_lead{lead}.pt",
        )

    windows = {"default": split == "test", "extended": np.isin(split, ["test", "beyond"])}
    backbone.eval()
    head.eval()
    out_frames = []
    for window_name, snap_mask in windows.items():
        if snap_mask.sum() == 0:
            continue
        idx = np.where(snap_mask)[0]
        with torch.no_grad():
            h = backbone(A_hat, X_t[idx])
            logits = head(h)
            pred_class = head.predict_class(logits).numpy()  # (S_win, N)
        true_class = discretize_ipc(y_t[idx]).numpy()
        valid = target_mask[idx]  # (S_win, N)

        s_win = len(idx)
        n_nodes = len(zone_order)
        frame = pd.DataFrame({
            "time": np.repeat(time.to_numpy()[idx], n_nodes),
            "zone_code": np.tile(node_zone_code, s_win),
            "lhz": np.tile(node_lhz, s_win),
            "lead": lead,
            "architecture": architecture,
            "head": head_kind,
            "window": window_name,
            "true_class": true_class.reshape(-1),
            "pred_class": pred_class.reshape(-1),
            "valid": valid.reshape(-1),
        })
        frame = frame[frame["valid"]].drop(columns="valid")
        out_frames.append(frame)
        if verbose:
            wf1 = weighted_f1(frame["true_class"].to_numpy(), frame["pred_class"].to_numpy())
            print(f"    lead {lead:2d} [{window_name:8}] n={len(frame):4} weighted F1 {wf1:.4f}")

    return pd.concat(out_frames, ignore_index=True)


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
    parser.add_argument("--adjacency", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--leads", type=str, default=None)
    parser.add_argument("--architecture", choices=["tgcn", "gcn_only", "lstm_only"], default="tgcn")
    parser.add_argument("--head", choices=["ce", "coral"], default="ce")
    parser.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    parser.add_argument("--hidden-units", type=int, default=DEFAULTS["hidden_units"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--max-epochs", type=int, default=DEFAULTS["max_epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--val-fraction", type=float, default=DEFAULTS["val_fraction"])
    parser.add_argument("--save-models", action="store_true")
    args = parser.parse_args(argv)

    leads = [int(x) for x in args.leads.split(",")] if args.leads else LEADS
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = None
    if args.save_models:
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nfitting {len(leads)} {args.architecture} models (head={args.head}, "
          f"{args.num_layers}L/{args.hidden_units}u)")
    all_predictions = []
    for lead in leads:
        print(f"  lead {lead}:")
        preds = run_lead(
            args.sequences_dir, args.adjacency, lead, args.architecture, args.head,
            args.num_layers, args.hidden_units, args.dropout, args.lr, args.weight_decay,
            args.max_epochs, args.patience, args.val_fraction, models_dir=models_dir,
        )
        all_predictions.append(preds)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_parquet(out_dir / "predictions.parquet", index=False)

    summary_rows, per_class_frames, per_lhz_frames = [], [], []
    for (window, lead), group in predictions.groupby(["window", "lead"]):
        result = evaluate_all(group["true_class"].to_numpy(), group["pred_class"].to_numpy(), group["lhz"].to_numpy())
        summary_rows.append({"window": window, "lead": lead, **result["summary"]})
        pc = result["per_class"].copy(); pc["window"] = window; pc["lead"] = lead
        per_class_frames.append(pc)
        pl = result["per_livelihood_zone"].copy(); pl["window"] = window; pl["lead"] = lead
        per_lhz_frames.append(pl)

    metrics_per_lead = pd.DataFrame(summary_rows)
    metrics_per_class = pd.concat(per_class_frames, ignore_index=True)
    metrics_per_lhz = pd.concat(per_lhz_frames, ignore_index=True)

    metrics_per_lead.to_csv(out_dir / "metrics_per_lead.csv", index=False)
    metrics_per_class.to_csv(out_dir / "metrics_per_class.csv", index=False)
    metrics_per_lhz.to_csv(out_dir / "metrics_per_livelihood_zone.csv", index=False)

    meta = {
        "run_at": datetime.now().isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "sequences_dir": str(args.sequences_dir),
        "adjacency": str(args.adjacency),
        "architecture": args.architecture,
        "head": args.head,
        "hyperparameters": {
            "num_layers": args.num_layers, "hidden_units": args.hidden_units,
            "dropout": args.dropout, "lr": args.lr, "weight_decay": args.weight_decay,
            "max_epochs": args.max_epochs, "patience": args.patience,
            "val_fraction_chronological_tail_of_train": args.val_fraction, "seed": SEED,
        },
        "windows": {
            "default": "train <=2019-12, test 2020-01..2022-12 (Busker-parity split)",
            "extended": "same training data, test+beyond splits (2020-01 onward, incl. 2023+)",
        },
        "leads": leads,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    print("\nPer-lead weighted F1:")
    print(metrics_per_lead.set_index(["window", "lead"])[["n", "weighted_f1", "balanced_accuracy", "accuracy"]]
          .round(4).to_string())
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
