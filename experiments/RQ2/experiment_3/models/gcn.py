"""Spatial GCN component (Wubetie et al. 2026 spec Sec 2):

    H^l = ReLU( A_hat @ H^(l-1) @ W^(l-1) )

Implemented as W first (dense N x C_in -> N x C_out), then A_hat propagation
(sparse N x N -> N x C_out) -- mathematically identical to the spec's stated
order by associativity of matrix multiplication (A_hat @ H @ W == A_hat @
(H @ W) == (A_hat @ H) @ W, since no nonlinearity sits between the two
multiplications), and cheaper when out_channels < in_channels (the usual
case here: hundreds of raw features collapsing to 32/64 hidden units).

num_layers and hidden_units are constructor arguments, not hardcoded, so the
same class reproduces the paper's own Table 1 ablation (32u/2L, 32u/4L,
64u/2L, 64u/4L) as well as this project's own hyperparameter search.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def sparse_propagate(A_hat: torch.sparse.Tensor, H: torch.Tensor) -> torch.Tensor:
    """A_hat @ H, where H is (N, C) or (B, N, C). A_hat is a fixed (N, N)
    sparse tensor shared across the batch (a static geographic adjacency
    that does not vary per sample), so batching is done by folding the
    batch dimension into the feature/channel dimension around the single
    sparse matmul, not by looping."""
    if H.dim() == 2:
        return torch.sparse.mm(A_hat, H)
    if H.dim() != 3:
        raise ValueError(f"H must be (N, C) or (B, N, C), got shape {tuple(H.shape)}")
    b, n, c = H.shape
    h_flat = H.permute(1, 0, 2).reshape(n, b * c)
    out = torch.sparse.mm(A_hat, h_flat)
    return out.reshape(n, b, c).permute(1, 0, 2)


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, A_hat: torch.sparse.Tensor, H: torch.Tensor) -> torch.Tensor:
        return sparse_propagate(A_hat, self.linear(H))


class GCNStack(nn.Module):
    """num_layers stacked GCNLayer + ReLU + Dropout, per Sec 2 of the spec."""

    def __init__(self, in_dim: int, hidden_units: int = 64, num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        dims = [in_dim] + [hidden_units] * num_layers
        self.layers = nn.ModuleList(GCNLayer(dims[i], dims[i + 1]) for i in range(num_layers))
        self.dropout = nn.Dropout(dropout)
        self.out_dim = hidden_units

    def forward(self, A_hat: torch.sparse.Tensor, H: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            H = F.relu(layer(A_hat, H))
            H = self.dropout(H)
        return H
