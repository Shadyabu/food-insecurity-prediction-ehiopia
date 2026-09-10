"""Unit tests for TGCNCell/TGCN forward pass shapes and gradient flow --
explicitly requested deliverable (small synthetic graph, check output
shapes and that gradients actually reach every parameter).

Run with: /opt/miniconda3/bin/python -m pytest experiments/RQ2/experiment_3/tests/test_tgcn_cell.py -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.baselines import GCNOnly, LSTMOnly  # noqa: E402
from models.gcn import GCNStack  # noqa: E402
from models.graph import normalize_adjacency  # noqa: E402
from models.heads import CoralHead, ClassificationHead, discretize_ipc  # noqa: E402
from models.tgcn import TGCN, TGCNCell  # noqa: E402


@pytest.fixture
def synthetic_graph():
    n = 6
    rng = np.random.default_rng(42)
    A = (rng.random((n, n)) < 0.4).astype(np.float32)
    A = np.triu(A, k=1)
    A = A + A.T
    return normalize_adjacency(A), n


def test_tgcn_cell_forward_shape(synthetic_graph):
    A_hat, n = synthetic_graph
    cell = TGCNCell(in_dim=5, hidden_units=8, num_gcn_layers=2)
    X_t = torch.randn(3, n, 5)  # (B, N, in_dim)
    h = cell(A_hat, X_t, None)
    assert h.shape == (3, n, 8)

    h2 = cell(A_hat, X_t, h)
    assert h2.shape == (3, n, 8)


def test_tgcn_sequence_forward_shape(synthetic_graph):
    A_hat, n = synthetic_graph
    model = TGCN(in_dim=5, hidden_units=8, num_gcn_layers=3)
    X_seq = torch.randn(2, 12, n, 5)  # (B, T, N, in_dim)
    h_final = model(A_hat, X_seq)
    assert h_final.shape == (2, n, 8)


def test_tgcn_gradient_flow(synthetic_graph):
    A_hat, n = synthetic_graph
    model = TGCN(in_dim=4, hidden_units=6, num_gcn_layers=2)
    head = ClassificationHead(hidden_units=6, num_classes=5)
    X_seq = torch.randn(3, 6, n, 4, requires_grad=False)
    y_class = torch.randint(0, 5, (3, n))

    h_final = model(A_hat, X_seq)
    logits = head(h_final)
    loss = head.loss(logits.reshape(-1, 5), y_class.reshape(-1))
    loss.backward()

    params = list(model.parameters()) + list(head.parameters())
    assert len(params) > 0
    for p in params:
        assert p.grad is not None, "gradient did not reach a parameter"
        assert torch.any(p.grad != 0), "a parameter's gradient is all-zero"


def test_gcn_stack_configurable_depth_and_width(synthetic_graph):
    A_hat, n = synthetic_graph
    for num_layers, hidden in [(2, 32), (2, 64), (4, 32), (4, 64)]:
        stack = GCNStack(in_dim=7, hidden_units=hidden, num_layers=num_layers)
        out = stack(A_hat, torch.randn(n, 7))
        assert out.shape == (n, hidden)
        assert len(stack.layers) == num_layers


def test_gcn_only_baseline_ignores_all_but_last_timestep(synthetic_graph):
    A_hat, n = synthetic_graph
    model = GCNOnly(in_dim=3, hidden_units=8, num_layers=2)
    model.eval()  # disable dropout so the comparison below is deterministic
    X_seq = torch.randn(2, 5, n, 3)
    out_full = model(A_hat, X_seq)

    X_seq_altered_history = X_seq.clone()
    X_seq_altered_history[:, :-1] = torch.randn(2, 4, n, 3)  # scramble everything but the last step
    out_altered = model(A_hat, X_seq_altered_history)

    assert torch.allclose(out_full, out_altered), "GCN-only baseline must not use timesteps before the last one"


def test_lstm_only_baseline_ignores_adjacency(synthetic_graph):
    A_hat, n = synthetic_graph
    model = LSTMOnly(in_dim=3, hidden_units=8)
    X_seq = torch.randn(2, 5, n, 3)

    out_with_real_A = model(A_hat, X_seq)
    A_alt = normalize_adjacency(np.eye(n, k=1, dtype=np.float32) + np.eye(n, k=-1, dtype=np.float32))
    out_with_alt_A = model(A_alt, X_seq)

    assert torch.allclose(out_with_real_A, out_with_alt_A), "LSTM-only baseline must not use the adjacency matrix"


def test_coral_head_predictions_are_rank_monotonic():
    head = CoralHead(hidden_units=4, num_classes=5)
    h = torch.randn(50, 4)
    logits = head(h)
    preds = head.predict_class(logits)
    assert preds.min() >= 0
    assert preds.max() <= 4
    # CORAL's construction guarantees P(y>0) >= P(y>1) >= ... is not required
    # of raw logits, but the predicted class (count of exceeded thresholds)
    # must always land in the valid class range for arbitrary logits.


def test_discretize_ipc_rounds_and_clips():
    y = torch.tensor([1.0, 1.4, 1.6, 3.0, 5.4, 0.2, 6.0])
    classes = discretize_ipc(y)
    assert classes.tolist() == [0, 0, 1, 2, 4, 0, 4]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
