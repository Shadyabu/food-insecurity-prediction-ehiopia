"""T-GCN cell and sequence model (Wubetie et al. 2026 spec Sec 3):

    u_t = sigmoid(W_u [f(A,X_t), h_{t-1}] + b_u)      # update gate
    r_t = sigmoid(W_r [f(A,X_t), h_{t-1}] + b_r)      # reset gate
    c_t = tanh(W_c [f(A,X_t), r_t * h_{t-1}] + b_c)   # candidate memory
    h_t = u_t * h_{t-1} + (1 - u_t) * c_t

## Why this is NOT a direct call to torch_geometric_temporal.nn.recurrent.TGCN

Evaluated directly (its source was read in full before writing this file).
Its `f(A, X_t)` is hardcoded as a *single* `GCNConv` layer inside each gate
(`conv_z`, `conv_r`, `conv_h` -- three separate one-layer graph convs, out
-channels fixed at the cell's `out_channels`). That is a real divergence
that matters for this experiment specifically: Sec 2 of the spec requires a
*configurable-depth* spatial GCN (`num_layers`, `hidden_units`) so the
paper's own Table 1 ablation (32u/2L, 32u/4L, 64u/2L, 64u/4L) can be
reproduced before this project's own hyperparameter search -- the library
class has no `num_layers` argument and cannot represent a 4-layer GCN
feeding the gates. Using it directly would silently substitute a
architecturally-different (shallower) spatial encoder for the one the spec
and the paper's own ablation describe.

Composed instead from two library-grade, independently well-tested
primitives rather than hand-rolling either: `GCNStack` (this project's own
class, `models/gcn.py`, built to the spec's exact layer-wise propagation
rule and reused unmodified for the GCN-only baseline in `models/baselines.py`)
supplies `f(A, X_t)`, and `torch.nn.GRUCell` supplies the four gate
equations above -- `GRUCell`'s own equations (`r`, `z`, `n`, `h' = (1-z)n +
z*h`) are algebraically the same four equations with `z<->u_t` and `n<->c_t`
relabelled, so this is a faithful, not approximate, implementation of the
spec, just composed from `torch.nn` rather than
`torch_geometric_temporal.nn.recurrent`. `GRUCell` batches over any leading
dimension, so it is applied with batch dimension `(B*N,)`, i.e. every node
treated as an independent sequence element sharing the same gate weights --
exactly the "same GRU parameters at every node" assumption the spec's
per-node hidden-state matrix implies.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .gcn import GCNStack


class TGCNCell(nn.Module):
    def __init__(self, in_dim: int, hidden_units: int = 64, num_gcn_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.gcn = GCNStack(in_dim, hidden_units=hidden_units, num_layers=num_gcn_layers, dropout=dropout)
        self.gru_cell = nn.GRUCell(hidden_units, hidden_units)
        self.hidden_units = hidden_units

    def forward(self, A_hat: torch.sparse.Tensor, X_t: torch.Tensor, h_prev: torch.Tensor | None) -> torch.Tensor:
        """X_t: (B, N, in_dim). h_prev: (B, N, hidden_units) or None (zero-initialized)."""
        f_t = self.gcn(A_hat, X_t)  # f(A, X_t), shape (B, N, hidden_units)
        b, n, h = f_t.shape
        f_flat = f_t.reshape(b * n, h)
        h_flat = (torch.zeros_like(f_flat) if h_prev is None else h_prev.reshape(b * n, h))
        h_new = self.gru_cell(f_flat, h_flat)
        return h_new.reshape(b, n, h)


class TGCN(nn.Module):
    """Runs TGCNCell over a (B, T, N, in_dim) sequence, returns the final
    per-node hidden state (B, N, hidden_units) for a prediction head."""

    def __init__(self, in_dim: int, hidden_units: int = 64, num_gcn_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.cell = TGCNCell(in_dim, hidden_units=hidden_units, num_gcn_layers=num_gcn_layers, dropout=dropout)
        self.hidden_units = hidden_units

    def forward(self, A_hat: torch.sparse.Tensor, X_seq: torch.Tensor) -> torch.Tensor:
        b, t, n, _ = X_seq.shape
        h = None
        for step in range(t):
            h = self.cell(A_hat, X_seq[:, step], h)
        return h
