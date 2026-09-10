"""Champion-ensemble nowcast -- validation.

1. Lead=0 anchor check: target month 2026-06 is real/observed and every
   member's lead=0 prediction needs zero reconstruction (XGBoost/
   RandomForest score a real historical row; the LSTM's input window IS
   the real cached sequence; TabICLv2 predicts a real row) -- so this
   pipeline's lead=0 output should closely match the already-computed,
   independently-built `rq5ens_*_2026_baseline` 4-way average for the same
   target month (built 2026-09-02 for a different diagnostic, same
   underlying member recipes: XGBoost/RandomForest exclude_feature_cluster
   =["climate"], LSTM/TabICLv2 full-feature). Not required to match
   bit-for-bit (different code paths reaching the same models), but should
   be very close -- a large mismatch would mean a real bug in one pipeline
   or the other.
2. Structural checks: 644 rows, no NaN, ipc_class in [1,5], each member's
   raw prediction column populated for every row.
"""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
NOWCAST_DIR = Path(__file__).resolve().parent
RQ2_EXP5_DIR = REPO_ROOT / "experiments" / "RQ2" / "experiment_5"

LEADS = [0, 1, 2, 3, 4, 8, 12]
FAILURES: list[str] = []


def check(condition: bool, message: str):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def structural_checks(df: pd.DataFrame):
    print("=== Structural checks ===")
    check(len(df) == 92 * len(LEADS), f"expected {92 * len(LEADS)} rows, got {len(df)}")
    for col in ["xgb_pred", "rf_pred", "lstm_pred", "tabicl_pred", "ipc_continuous", "ipc_class"]:
        check(df[col].notna().all(), f"no NaN in {col}")
    check(df["ipc_class"].between(1, 5).all(), "every ipc_class in [1,5]")
    recomputed_mean = df[["xgb_pred", "rf_pred", "lstm_pred", "tabicl_pred"]].mean(axis=1)
    check(np.allclose(df["ipc_continuous"], recomputed_mean, atol=1e-9),
          "ipc_continuous exactly equals mean of the 4 members")


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_2026_member(name, path):
    d = pd.read_parquet(path)
    d["time"] = pd.to_datetime(d["time"])
    if "window" in d.columns:  # LSTM/TabICLv2 carry both windows in one file
        d = d[d["window"] == "extended"]
    # XGBoost/RandomForest's rq5ens_*_2026_baseline files have only the
    # 2026 diagnostic rows already (no "window" column at all).
    d = d[(d["time"] == pd.Timestamp("2026-06-01")) & (d["lead"] == 0)]
    return d.set_index("zone_code")["prediction"].rename(name)


def anchor_check_lead0(df: pd.DataFrame):
    print("\n=== Lead=0 anchor check vs. rq5ens_*_2026_baseline ===")
    paths = {
        "XGBoost": REPO_ROOT / "experiments" / "RQ1" / "experiment_2" / "rq5ens_xgb_2026_baseline" / "predictions.parquet",
        "RandomForest": RQ2_EXP5_DIR / "rq5ens_rf_2026_baseline" / "predictions.parquet",
        "LSTM": RQ2_EXP5_DIR / "rq5ens_lstm_2026_baseline" / "predictions.parquet",
        "TabICLv2": RQ2_EXP5_DIR / "rq5ens_tabicl_2026_baseline" / "predictions.parquet",
    }
    if not all(p.exists() for p in paths.values()):
        print("  [SKIP] reference files not all found")
        return

    members = [load_2026_member(name, p) for name, p in paths.items()]
    ref = pd.concat(members, axis=1)
    ref["ref_ensemble_continuous"] = ref.mean(axis=1)
    ref["ref_ensemble_class"] = to_ipc_class(ref["ref_ensemble_continuous"])

    mine = df[df["lead"] == 0].set_index("zone_code")
    merged = mine[["ipc_continuous", "ipc_class"]].join(
        ref[["ref_ensemble_continuous", "ref_ensemble_class"]], how="inner"
    )
    check(len(merged) == 92, f"92 zones overlap for the anchor check, got {len(merged)}")

    class_match = (merged["ipc_class"] == merged["ref_ensemble_class"]).mean()
    mean_abs_diff = (merged["ipc_continuous"] - merged["ref_ensemble_continuous"]).abs().mean()
    corr = merged["ipc_continuous"].corr(merged["ref_ensemble_continuous"])
    print(f"  n={len(merged)}  class exact-match={class_match:.1%}  "
          f"mean abs diff (continuous)={mean_abs_diff:.4f}  correlation={corr:.4f}")
    check(corr > 0.9, "correlation with independently-built reference > 0.9")
    check(mean_abs_diff < 0.3, "mean absolute difference vs. reference < 0.3 IPC points")


def main():
    df = pd.read_csv(NOWCAST_DIR / "champion_nowcast_predictions.csv")
    structural_checks(df)
    anchor_check_lead0(df)

    print("\n=== Summary ===")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
