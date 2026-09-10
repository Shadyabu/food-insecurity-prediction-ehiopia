"""RQ4 Experiment 3 -- local (per-instance) leave-top-1-out SHAP stability,
LSTM half. Run STANDALONE (not through common.py's subprocess wrapper) --
this script never imports xgboost, so it is safe to run directly; see
lstm_worker.py's own module docstring for why LSTM work must stay isolated
from any process that has already imported xgboost/shap-for-trees.

Same H0 and design as run_experiment3_trees.py (see that script's
docstring) -- mirrors timing_probe.py's approach exactly, just over the
full test set per lead instead of a 10-row sample.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

RQ4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RQ4_DIR))
import lstm_worker as w  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
LEADS = [0, 1, 2, 3, 4, 8, 12]


def per_row_shap(explainer, X_scaled_row_batch):
    import torch
    test_t = torch.from_numpy(X_scaled_row_batch.astype(np.float32))
    sv = np.asarray(explainer.shap_values(test_t))  # (n, seq_len, n_features, 1)
    return np.abs(sv[..., 0]).sum(axis=1)  # (n, n_features)


def row_local_stability(pre_row, post_row, top1_idx):
    keep = np.ones(len(pre_row), dtype=bool)
    keep[top1_idx] = False
    pre = pre_row[keep]
    post = post_row[keep]
    if len(pre) < 2 or np.all(pre == pre[0]) or np.all(post == post[0]):
        return np.nan
    rho, _ = spearmanr(pre, post)
    return rho


def main():
    import torch
    import shap

    rows = []
    t_start = time.time()

    for lead in LEADS:
        t_lead = time.time()
        lc = w.get_lstm_cell(lead, window="default")
        model = w.load_model(lead)

        class Wrap(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x).unsqueeze(-1)

        wrapped = Wrap(model)

        X_test_raw = lc["X_raw"][lc["test_mask"]]
        time_test = lc["time"][lc["test_mask"]]
        zone_test = lc["zone_code"][lc["test_mask"]]
        n = X_test_raw.shape[0]
        feature_names = lc["feature_names"]

        X_train_scaled = w.lstm_scale(lc["X_raw"][lc["train_mask"]], lc["medians"], lc["mean"], lc["std"])
        rng = np.random.default_rng(w.SEED)
        bg_idx = rng.choice(len(X_train_scaled), size=min(30, len(X_train_scaled)), replace=False)
        background = torch.from_numpy(X_train_scaled[bg_idx].astype(np.float32))
        explainer = shap.GradientExplainer(wrapped, background)

        X_test_scaled = w.lstm_scale(X_test_raw, lc["medians"], lc["mean"], lc["std"])
        pre_sv = per_row_shap(explainer, X_test_scaled)  # (n, n_features)
        top1_idx = np.abs(pre_sv).argmax(axis=1)

        for i in range(n):
            feat = feature_names[top1_idx[i]]
            row_raw = X_test_raw[i:i + 1]
            masked_scaled = w.mask_nan_then_scale(
                row_raw, feature_names, [feat], lc["medians"], lc["mean"], lc["std"]
            )
            post_sv = per_row_shap(explainer, masked_scaled)[0]
            rho = row_local_stability(pre_sv[i], post_sv, top1_idx[i])
            rows.append({
                "subject": "lstm", "lead": lead,
                "zone_code": zone_test[i], "time": pd.Timestamp(time_test[i]),
                "top1_feature": feat, "rho": rho,
            })

        print(f"[lstm] lead={lead}: n={n}, {time.time() - t_lead:.1f}s", flush=True)

    print(f"[lstm] TOTAL: {time.time() - t_start:.1f}s ({len(rows)} rows)", flush=True)
    pd.DataFrame(rows).to_csv(OUT_DIR / "local_stability_lstm.csv", index=False)
    print("Done. Wrote local_stability_lstm.csv")


if __name__ == "__main__":
    main()
