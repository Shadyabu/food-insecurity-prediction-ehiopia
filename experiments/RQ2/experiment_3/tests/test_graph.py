"""Unit tests for normalize_adjacency() -- explicitly requested deliverable
(small synthetic graph, check the exact D^(-1/2)(A+I)D^(-1/2) formula, not
just that it runs).

Run with: /opt/miniconda3/bin/python -m pytest experiments/RQ2/experiment_3/tests/test_graph.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.graph import normalize_adjacency  # noqa: E402


def test_normalize_adjacency_matches_manual_computation():
    # 4-node path graph: 0-1-2-3
    A = np.array([
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
    ], dtype=np.float32)

    A_hat_sparse = normalize_adjacency(A)
    A_hat = A_hat_sparse.to_dense().numpy()

    A_tilde = A + np.eye(4)
    D = A_tilde.sum(axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(D)
    expected = d_inv_sqrt[:, None] * A_tilde * d_inv_sqrt[None, :]

    np.testing.assert_allclose(A_hat, expected, atol=1e-6)


def test_normalize_adjacency_is_symmetric_for_symmetric_input():
    rng = np.random.default_rng(0)
    n = 6
    A = (rng.random((n, n)) < 0.3).astype(np.float32)
    A = np.triu(A, k=1)
    A = A + A.T  # symmetrize, no self-loops

    A_hat = normalize_adjacency(A).to_dense().numpy()
    np.testing.assert_allclose(A_hat, A_hat.T, atol=1e-6)


def test_normalize_adjacency_diagonal_is_positive_from_self_loops():
    # Isolated node (degree 0 in A) still gets a nonzero diagonal entry
    # because A + I always has a self-loop, per the spec.
    A = np.zeros((3, 3), dtype=np.float32)
    A_hat = normalize_adjacency(A).to_dense().numpy()
    np.testing.assert_allclose(A_hat, np.eye(3), atol=1e-6)
    assert np.all(np.diag(A_hat) > 0)


def test_normalize_adjacency_is_sparse_tensor():
    A = np.eye(5, k=1, dtype=np.float32) + np.eye(5, k=-1, dtype=np.float32)
    result = normalize_adjacency(A)
    assert result.is_sparse
    assert result.shape == (5, 5)


def test_normalize_adjacency_rejects_non_square():
    with pytest.raises(ValueError):
        normalize_adjacency(np.zeros((3, 4)))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
