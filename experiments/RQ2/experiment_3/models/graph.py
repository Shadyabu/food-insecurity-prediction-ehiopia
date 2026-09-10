"""Symmetric-normalized graph adjacency, per Wubetie et al. (2026)'s T-GCN spec:

    A_tilde = A + I                              (self-loops added)
    D       = degree matrix of A_tilde
    A_hat   = D^(-1/2) . A_tilde . D^(-1/2)       (symmetric normalization)

Stored as a torch.sparse_coo_tensor so memory scales O(|E|), not O(N^2) --
matters once this is reused on a larger admin unit set than Ethiopia's 92
zones (this project's own node count is not hardcoded here).
"""
from __future__ import annotations

import numpy as np
import torch


def normalize_adjacency(A: np.ndarray) -> torch.sparse.Tensor:
    """A: (N, N) binary/weighted adjacency, no self-loops assumed.
    Returns a sparse (N, N) torch tensor: D^(-1/2) (A + I) D^(-1/2)."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"A must be square (N, N), got shape {A.shape}")

    n = A.shape[0]
    A_tilde = A.astype(np.float64) + np.eye(n, dtype=np.float64)

    degree = A_tilde.sum(axis=1)
    if np.any(degree <= 0):
        raise ValueError("A + I has a zero-degree row -- every node has a self-loop, this should not happen")
    d_inv_sqrt = 1.0 / np.sqrt(degree)

    A_hat = d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]

    rows, cols = np.nonzero(A_hat)
    values = A_hat[rows, cols]
    indices = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    values_t = torch.tensor(values, dtype=torch.float32)
    return torch.sparse_coo_tensor(indices, values_t, size=(n, n)).coalesce()


def load_normalized_adjacency(adjacency_path: str) -> tuple[torch.sparse.Tensor, list[str]]:
    """Convenience loader for this experiment's adjacency.npy + zone_order.json."""
    import json
    from pathlib import Path

    adjacency_path = Path(adjacency_path)
    A = np.load(adjacency_path)
    zone_order = json.loads((adjacency_path.parent / "zone_order.json").read_text())
    return normalize_adjacency(A), zone_order
