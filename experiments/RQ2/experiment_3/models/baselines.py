"""Two single-component ablations, reproducing the paper's own Table 2
comparison (Wubetie et al. 2026 spec Sec 6):

- `GCNOnly`  -- spatial component in isolation. Drops the GRU entirely and
  predicts from the current month's snapshot alone (the most recent
  timestep of the input window, i.e. no temporal information at all -- "no
  temporal recurrence" is taken literally, not just "no GRU cell").
- `LSTMOnly` -- temporal component in isolation. A standard `nn.LSTM` run
  independently per node over the same 12-month window T-GCN sees, sharing
  weights across all N nodes (the graph/adjacency is never touched) --
  architecturally the same pooled-LSTM design already validated in
  `experiments/RQ2/experiment_1/run_lstm_ethiopia.py` (hidden=64, one
  shared LSTM over all zones), just reshaped to batch over (snapshot x
  node) instead of (zone x origin_month) since the input here is a graph
  -snapshot sequence, not a per-zone sequence.

Both return a (B, N, hidden_units) representation with the same shape as
`TGCN`'s output, so `models/heads.py`'s CE/CORAL heads work unmodified on
all three architectures -- required for the head-choice comparison to be
apples-to-apples across T-GCN / GCN-only / LSTM-only.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gcn import GCNStack


class GCNOnly(nn.Module):
    def __init__(self, in_dim: int, hidden_units: int = 64, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gcn = GCNStack(in_dim, hidden_units=hidden_units, num_layers=num_layers, dropout=dropout)
        self.hidden_units = hidden_units

    def forward(self, A_hat: torch.sparse.Tensor, X_seq: torch.Tensor) -> torch.Tensor:
        """X_seq: (B, T, N, in_dim). Only the last (most recent, origin_month)
        timestep is used -- no temporal recurrence, by design."""
        X_last = X_seq[:, -1]  # (B, N, in_dim)
        return self.gcn(A_hat, X_last)


class LSTMOnly(nn.Module):
    def __init__(self, in_dim: int, hidden_units: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim, hidden_size=hidden_units, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.hidden_units = hidden_units

    def forward(self, A_hat: torch.sparse.Tensor, X_seq: torch.Tensor) -> torch.Tensor:
        """A_hat is accepted but unused (ignored by design -- this baseline
        never sees the adjacency matrix). X_seq: (B, T, N, in_dim)."""
        del A_hat
        b, t, n, f = X_seq.shape
        x_per_node = X_seq.permute(0, 2, 1, 3).reshape(b * n, t, f)  # (B*N, T, F)
        _, (h_n, _) = self.lstm(x_per_node)
        h_final = h_n[-1]  # (B*N, hidden_units) -- last layer's final hidden state
        return h_final.reshape(b, n, self.hidden_units)
