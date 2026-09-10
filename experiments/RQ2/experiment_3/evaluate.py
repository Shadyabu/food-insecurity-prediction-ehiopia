"""Evaluation metrics for RQ2 Experiment 3 (T-GCN), per the prompt's Sec 4/7
deliverable: weighted F1 as the primary metric (this project's metric of
record, CLAUDE.md Sec 7 -- not accuracy, since IPC-phase classes are
imbalanced), balanced accuracy, per-class precision/recall/F1, and a
per-livelihood-zone stratified breakdown (needed for RQ3 -- correlating
performance with data completeness per zone).

All functions take flat 1-D arrays of already-discretized 0-indexed class
labels (see models/heads.py::discretize_ipc) and an aligned
node_livelihood_zone array for the stratified breakdown -- no model-specific
logic here, so the same functions score T-GCN, GCN-only, and LSTM-only
predictions identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

IPC_CLASS_NAMES = ["Minimal", "Stressed", "Crisis", "Emergency", "Famine"]  # class 0..4 == IPC phase 1..5


def weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "n": int(len(y_true)),
        "weighted_f1": weighted_f1(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    labels = list(range(len(IPC_CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    return pd.DataFrame({
        "ipc_class": IPC_CLASS_NAMES,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def per_livelihood_zone_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                 livelihood_zone: np.ndarray) -> pd.DataFrame:
    rows = []
    for zone in sorted(pd.unique(livelihood_zone)):
        mask = livelihood_zone == zone
        if mask.sum() == 0:
            continue
        rows.append({"livelihood_zone": zone, **summary_metrics(y_true[mask], y_pred[mask])})
    return pd.DataFrame(rows)


def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray, livelihood_zone: np.ndarray) -> dict:
    """Returns {'summary': dict, 'per_class': DataFrame, 'per_livelihood_zone': DataFrame}."""
    return {
        "summary": summary_metrics(y_true, y_pred),
        "per_class": per_class_metrics(y_true, y_pred),
        "per_livelihood_zone": per_livelihood_zone_metrics(y_true, y_pred, livelihood_zone),
    }
