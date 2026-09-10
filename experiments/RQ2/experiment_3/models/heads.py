"""Prediction heads (Wubetie et al. 2026 spec Sec 4). Both map a per-node
hidden state (B, N, hidden_units) to a 5-class IPC-phase prediction --
adapted from the paper's own 3-class FCSL target, which this project does
not use.

(a) `ClassificationHead` -- standard categorical cross-entropy, what the
    paper actually used (their own finding: CE outperformed an ordinal loss
    in practice, despite the target being ordinal).
(b) `CoralHead` -- CORAL (Cao, Mirjalili & Raschka, 2020, "Rank Consistent
    Ordinal Regression for Neural Networks with Application to Age
    Estimation"): K-1 binary "is true class > threshold k" classifiers
    sharing a single weight vector (only the bias differs per threshold),
    which guarantees rank-monotonic predictions by construction -- unlike
    K independent one-vs-rest classifiers, which have no such guarantee.
    Given directly to compare against (a) empirically, per this project's
    own RQ2/RQ4 interest in whether the ordinal structure of IPC phase is
    worth modelling explicitly.

Target encoding: IPC phase is continuous in this project's own target
(`ipc_continuous`) -- discretized to 5 integer classes (1..5 -> 0..4) via
the same nearest-integer-clipped-to-[1,5] rule already used throughout
RQ1/RQ2 for weighted-F1 scoring (CLAUDE.md Sec 7a's classifier addition),
not re-derived here.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 5


def discretize_ipc(y_continuous: torch.Tensor) -> torch.Tensor:
    """y_continuous: IPC phase value(s), any shape. Returns 0-indexed class
    labels (long), nearest-integer rounded and clipped to [1, 5] then
    shifted to [0, 4]."""
    y = torch.round(y_continuous).clamp(1, NUM_CLASSES)
    return (y - 1).long()


class ClassificationHead(nn.Module):
    def __init__(self, hidden_units: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(hidden_units, num_classes)
        self.num_classes = num_classes

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden_units) -> logits (..., num_classes)."""
        return self.fc(h)

    def loss(self, logits: torch.Tensor, y_class: torch.Tensor) -> torch.Tensor:
        """logits: (M, num_classes); y_class: (M,) 0-indexed class labels."""
        return F.cross_entropy(logits, y_class)

    def predict_class(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(dim=-1)


class CoralHead(nn.Module):
    """K-1 rank-monotonic binary thresholds sharing one weight vector."""

    def __init__(self, hidden_units: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.num_thresholds = num_classes - 1
        self.fc = nn.Linear(hidden_units, 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(self.num_thresholds))
        self.num_classes = num_classes

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden_units) -> logits (..., num_thresholds), logit_k
        for the event "true class > k"."""
        return self.fc(h) + self.bias

    def _levels(self, y_class: torch.Tensor) -> torch.Tensor:
        """y_class: (M,) 0-indexed. Returns (M, num_thresholds) binary
        targets: levels[i, k] = 1 iff y_class[i] > k."""
        thresholds = torch.arange(self.num_thresholds, device=y_class.device)
        return (y_class.unsqueeze(-1) > thresholds.unsqueeze(0)).float()

    def loss(self, logits: torch.Tensor, y_class: torch.Tensor) -> torch.Tensor:
        levels = self._levels(y_class)
        return F.binary_cross_entropy_with_logits(logits, levels)

    def predict_class(self, logits: torch.Tensor) -> torch.Tensor:
        """Sum of thresholds exceeded (sigmoid(logit_k) > 0.5), per CORAL's
        own prediction rule -- rank-consistent by construction, no
        argmax-over-independent-classes ambiguity possible."""
        return (torch.sigmoid(logits) > 0.5).sum(dim=-1)


def build_head(kind: str, hidden_units: int, num_classes: int = NUM_CLASSES) -> nn.Module:
    if kind == "ce":
        return ClassificationHead(hidden_units, num_classes)
    if kind == "coral":
        return CoralHead(hidden_units, num_classes)
    raise ValueError(f"unknown head kind: {kind!r} (expected 'ce' or 'coral')")
