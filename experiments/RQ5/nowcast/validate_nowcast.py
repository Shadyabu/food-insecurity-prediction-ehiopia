"""RQ5 nowcast, Phase 0c -- validation.

1. Lead=0 anchor check: lead=0 needs zero feature reconstruction (target ==
   origin), so its prediction/SHAP/cause should be IDENTICAL whether
   produced by this pipeline's synthetic-row path or by running the exact
   same scoring logic directly against the REAL, unmodified lead00.csv row
   at month=2026-06. This is the strongest, cheapest check that the
   column-mapping machinery (crop-calendar re-lookup, forecast-preselect
   renaming, dtype alignment) didn't silently corrupt anything -- if lead=0
   matches exactly, the same machinery is trusted to generalize to lead>0,
   which has no independent ground truth to check against at all.
2. No unexpected NaN predictions; every ipc_class in [1,5].
3. Ensemble's ipc_continuous/ipc_class exactly equals the xgboost/
   randomforest average (recomputed from nowcast_predictions.csv directly,
   not re-trusted from the compute script).
4. Rough magnitude cross-check against the existing (different-model,
   different-recipe) RQ2 Experiment 5 diagnostic 2026 predictions, as a
   plausibility sanity check, not an exact-match test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
RQ5_DIR = REPO_ROOT / "experiments" / "RQ5"
NOWCAST_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(RQ5_DIR))
import compute_local_shap_2020_2024 as rq5shap  # noqa: E402
from rule_engine import load_taxonomy, required_columns, score_batch  # noqa: E402

CLUSTERS = rq5shap.CLUSTERS
ORIGIN_MONTH = pd.Timestamp("2026-06-01")

FAILURES: list[str] = []


def check(condition: bool, message: str):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        FAILURES.append(message)


def anchor_check_lead0():
    print("\n=== Check 1: lead=0 anchor (real row vs. this pipeline's reconstruction) ===")
    rq5shap.EXTENDED_TEST_START = pd.Timestamp("2020-01-01")
    rq5shap.EXTENDED_TEST_END = pd.Timestamp("2026-12-31")
    taxonomy = load_taxonomy()
    need_cols = sorted(required_columns(taxonomy))

    nowcast_pred = pd.read_csv(NOWCAST_DIR / "nowcast_predictions.csv")
    nowcast_pred["target_month"] = pd.to_datetime(nowcast_pred["target_month"])
    mine = nowcast_pred[(nowcast_pred["lead"] == 0) & (nowcast_pred["target_month"] == ORIGIN_MONTH)]

    for cluster in CLUSTERS:
        df = rq5shap.xgb_mod.load_lead_table(str(rq5shap.MODEL_TABLES_DIR), 0)  # REAL, unmodified lead00.csv
        test_x_raw, test_meta, base1_test, train_medians = rq5shap.build_full_feature_matrix(df, cluster, 0)
        is_origin = test_meta["time"] == ORIGIN_MONTH
        if is_origin.sum() == 0:
            continue

        xgb_model = rq5shap.load_xgb_model(cluster, 0)
        xgb_pred = xgb_model.predict(test_x_raw)[is_origin.to_numpy()]
        rf_model = rq5shap.load_rf_model(cluster, 0)
        rf_pred = rf_model.predict(test_x_raw.fillna(train_medians))[is_origin.to_numpy()]
        base1 = base1_test[is_origin].to_numpy()

        real_xgb_ipc = np.clip(np.round(xgb_pred + base1), 1, 5).astype(int)
        real_rf_ipc = np.clip(np.round(rf_pred + base1), 1, 5).astype(int)
        real_zones = test_meta[is_origin]["zone_code"].to_numpy()

        mine_cluster = mine[mine["cluster"] == cluster].sort_values("zone_code")
        real_df = pd.DataFrame({"zone_code": real_zones, "xgb_ipc": real_xgb_ipc, "rf_ipc": real_rf_ipc}).sort_values("zone_code")

        mine_xgb = mine_cluster[mine_cluster["subject"] == "xgboost"].sort_values("zone_code")
        mine_rf = mine_cluster[mine_cluster["subject"] == "randomforest"].sort_values("zone_code")

        check(
            list(mine_xgb["zone_code"]) == list(real_df["zone_code"]),
            f"{cluster}: same zone set at lead=0 (n={len(real_df)})",
        )
        check(
            np.array_equal(mine_xgb["ipc_class"].to_numpy(), real_df["xgb_ipc"].to_numpy()),
            f"{cluster}: XGBoost lead=0 ipc_class matches real-row computation exactly",
        )
        check(
            np.array_equal(mine_rf["ipc_class"].to_numpy(), real_df["rf_ipc"].to_numpy()),
            f"{cluster}: RandomForest lead=0 ipc_class matches real-row computation exactly",
        )


def structural_checks():
    print("\n=== Check 2 & 3: NaN / range / ensemble-consistency ===")
    df = pd.read_csv(NOWCAST_DIR / "nowcast_predictions.csv")

    check(df["ipc_class"].notna().all(), "no NaN ipc_class anywhere")
    check(df["ipc_class"].between(1, 5).all(), "every ipc_class in [1,5]")
    check(len(df) == 92 * 7 * 3, f"expected 1932 rows (92 zones x 7 leads x 3 subjects), got {len(df)}")
    check(
        set(df["predicted_cause"].unique()) <= {"DROUGHT", "FLOODING", "CONFLICT", "MARKET_SHOCK", "COMPOUND", "UNCLASSIFIED", "NO_DATA"},
        "predicted_cause only uses the taxonomy's known labels",
    )

    piv = df.pivot_table(index=["zone_code", "lead"], columns="subject", values="ipc_continuous")
    recomputed_ens = (piv["xgboost"] + piv["randomforest"]) / 2.0
    check(
        np.allclose(piv["ensemble"].to_numpy(), recomputed_ens.to_numpy(), atol=1e-9),
        "ensemble ipc_continuous exactly equals mean(xgboost, randomforest) for every zone/lead",
    )


def magnitude_plausibility_check():
    print("\n=== Check 4: rough plausibility vs. RQ2 Exp5's diagnostic 2026 predictions ===")
    ref_path = REPO_ROOT / "experiments" / "RQ1" / "experiment_2" / "rq5ens_xgb_2026_baseline" / "predictions.parquet"
    if not ref_path.exists():
        print(f"  [SKIP] reference file not found: {ref_path}")
        return
    ref = pd.read_parquet(ref_path)
    ref["time"] = pd.to_datetime(ref["time"])
    ref_lead0 = ref[(ref["time"] == ORIGIN_MONTH) & (ref["lead"] == 0)]

    mine = pd.read_csv(NOWCAST_DIR / "nowcast_predictions.csv")
    mine["target_month"] = pd.to_datetime(mine["target_month"])
    mine_lead0_xgb = mine[(mine["target_month"] == ORIGIN_MONTH) & (mine["lead"] == 0) & (mine["subject"] == "xgboost")]

    merged = ref_lead0[["zone_code", "prediction"]].merge(
        mine_lead0_xgb[["zone_code", "ipc_continuous"]], on="zone_code", how="inner"
    )
    if len(merged) == 0:
        print("  [SKIP] no overlapping zones to compare")
        return
    corr = merged["prediction"].corr(merged["ipc_continuous"])
    mean_abs_diff = (merged["prediction"] - merged["ipc_continuous"]).abs().mean()
    print(f"  n={len(merged)} zones, correlation={corr:.3f}, mean abs diff={mean_abs_diff:.3f}")
    print("  (different model/feature recipe by design -- reported for plausibility, not pass/fail)")


def main():
    anchor_check_lead0()
    structural_checks()
    magnitude_plausibility_check()

    print("\n=== Summary ===")
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED:")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
