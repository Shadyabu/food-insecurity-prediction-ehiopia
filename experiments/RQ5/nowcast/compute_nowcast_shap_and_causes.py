"""RQ5 nowcast, Phase 0b -- score Phase 0a's constructed rows: predict IPC
class, compute per-instance SHAP, and run the RQ5 rule engine to get a
predicted disaster cause, for xgboost / randomforest / ensemble.

REUSE STRATEGY: rather than re-deriving the feature-prep transform (dummy
encoding, identity/provenance drop, dtype handling) that
`experiments/RQ5/compute_local_shap_2020_2024.py::build_full_feature_matrix()`
already implements and this project already trusts, this script
concatenates Phase 0a's synthetic rows onto the REAL historical lead table
and calls that exact same function -- with its module-level test-window
bounds temporarily widened to cover the nowcast's 2026-07..2027-06 target
months. This guarantees byte-identical column handling to every other
RQ1/RQ2/RQ5 script in this project, rather than risking a subtly different
reimplementation.
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
LEADS = rq5shap.LEADS
BASE1_COL = rq5shap.BASE1_COL

NOWCAST_FEATURE_ROWS_PATH = NOWCAST_DIR / "nowcast_feature_rows.csv"
ORIGIN_MONTH = pd.Timestamp("2026-06-01")


def widen_test_window():
    """Nowcast target months (2026-07..2027-06) fall outside the 2020-2024
    window compute_local_shap_2020_2024.py's build_full_feature_matrix()
    hardcodes. Monkeypatch the module-level bounds it reads at call time --
    documented here, not silent -- rather than reimplementing the function."""
    rq5shap.EXTENDED_TEST_START = pd.Timestamp("2020-01-01")
    rq5shap.EXTENDED_TEST_END = pd.Timestamp("2027-12-31")


def build_concat_panel(lead: int, nowcast_rows: pd.DataFrame) -> pd.DataFrame:
    """Real lead table + this lead's synthetic nowcast rows, schema-aligned
    (see build_nowcast_features.py's docstring).

    At lead=0 the target month (2026-06) already has a real row in the
    historical lead table (it's the origin month itself) -- drop it before
    concatenating the synthetic row so the two don't duplicate each other.
    A no-op at every other lead, since their target months (2026-07
    onward) don't exist in the historical table at all."""
    historical = rq5shap.xgb_mod.load_lead_table(str(rq5shap.MODEL_TABLES_DIR), lead)
    target_month_str = (ORIGIN_MONTH + pd.DateOffset(months=lead)).strftime("%Y-%m")
    historical = historical[historical["month"] != target_month_str]
    synthetic = nowcast_rows[nowcast_rows["lead"] == lead].drop(columns=["lead", "target_month"])
    # Column sets must match exactly for a clean concat -- align, don't hope.
    missing_in_synthetic = set(historical.columns) - set(synthetic.columns)
    missing_in_historical = set(synthetic.columns) - set(historical.columns)
    if missing_in_synthetic:
        raise ValueError(f"lead {lead}: synthetic rows missing columns present in real lead table: {sorted(missing_in_synthetic)}")
    if missing_in_historical:
        # extra bookkeeping columns on the synthetic side are fine, just drop before concat
        synthetic = synthetic.drop(columns=sorted(missing_in_historical))
    synthetic = synthetic[historical.columns]

    # Align dtypes to the real lead table's own dtype per column before
    # concatenating. Needed for e.g. seasonal_overlaps_{kiremt,belg,gu,deyr}:
    # at lead in {1,3} the historical column is a real (never-null) `bool`,
    # but the synthetic row round-trips through CSV as float64 (1.0/0.0) --
    # left unaligned, pd.concat upcasts the combined column to `object`,
    # which build_full_feature_matrix()'s own object/category-dtype drop
    # then silently removes, even though the model expects it.
    for col in historical.columns:
        if historical[col].dtype != synthetic[col].dtype:
            try:
                synthetic[col] = synthetic[col].astype(historical[col].dtype)
            except (ValueError, TypeError):
                pass  # leave as-is; a genuine incompatibility, not this bool/float case

    return pd.concat([historical, synthetic], ignore_index=True)


def main():
    widen_test_window()
    taxonomy = load_taxonomy()
    need_cols = sorted(required_columns(taxonomy))

    nowcast_rows = pd.read_csv(NOWCAST_FEATURE_ROWS_PATH, low_memory=False)

    raw_records = []
    shap_records = []

    for lead in LEADS:
        panel = build_concat_panel(lead, nowcast_rows)
        target_month = ORIGIN_MONTH + pd.DateOffset(months=lead)

        for cluster in CLUSTERS:
            test_x_raw, test_meta, base1_test, train_medians = rq5shap.build_full_feature_matrix(panel, cluster, lead)

            # Isolate just the nowcast row(s) -- everything at exactly the
            # target month within this (widened) test window that ISN'T
            # part of the original 2020-2024 rebuild.
            is_nowcast = test_meta["time"] == target_month
            if is_nowcast.sum() == 0:
                print(f"  cluster={cluster:13s} lead={lead:2d}  no zones in this cluster at this lead (n=0) -- skipping")
                continue

            avail_cols = [c for c in need_cols if c in test_x_raw.columns]
            missing_cols = [c for c in need_cols if c not in test_x_raw.columns]

            xgb_model = rq5shap.load_xgb_model(cluster, lead)
            xgb_explainer = shap.TreeExplainer(xgb_model)
            xgb_pred_full = xgb_model.predict(test_x_raw)
            xgb_shap_full = xgb_explainer.shap_values(test_x_raw)
            xgb_shap = pd.DataFrame(xgb_shap_full, columns=test_x_raw.columns, index=test_x_raw.index)[avail_cols]

            test_x_imputed = test_x_raw.fillna(train_medians)
            rf_model = rq5shap.load_rf_model(cluster, lead)
            rf_explainer = shap.TreeExplainer(rf_model)
            rf_pred_full = rf_model.predict(test_x_imputed)
            rf_shap_full = rf_explainer.shap_values(test_x_imputed)
            rf_shap = pd.DataFrame(rf_shap_full, columns=test_x_imputed.columns, index=test_x_raw.index)[avail_cols]

            for missing_col in missing_cols:
                xgb_shap[missing_col] = np.nan
                rf_shap[missing_col] = np.nan
            xgb_shap = xgb_shap[need_cols]
            rf_shap = rf_shap[need_cols]
            ens_shap = (xgb_shap + rf_shap) / 2.0

            xgb_ipc = np.clip(np.round(xgb_pred_full + base1_test.to_numpy()), 1, 5).astype(int)
            rf_ipc = np.clip(np.round(rf_pred_full + base1_test.to_numpy()), 1, 5).astype(int)
            ens_ipc_continuous = ((xgb_pred_full + rf_pred_full) / 2.0) + base1_test.to_numpy()
            ens_ipc = np.clip(np.round(ens_ipc_continuous), 1, 5).astype(int)

            base_meta = test_meta.assign(
                cluster=cluster, lead=lead, base1=base1_test.to_numpy(),
                xgb_ipc_continuous=xgb_pred_full + base1_test.to_numpy(),
                rf_ipc_continuous=rf_pred_full + base1_test.to_numpy(),
                ens_ipc_continuous=ens_ipc_continuous,
                xgb_ipc_class=xgb_ipc, rf_ipc_class=rf_ipc, ens_ipc_class=ens_ipc,
            )
            raw_sub = test_x_raw.reindex(columns=need_cols)

            for subject, shap_sub in [("xgboost", xgb_shap), ("randomforest", rf_shap), ("ensemble", ens_shap)]:
                meta_out = base_meta[is_nowcast].copy()
                meta_out["subject"] = subject
                raw_records.append(pd.concat([meta_out.reset_index(drop=True), raw_sub[is_nowcast].reset_index(drop=True)], axis=1))
                shap_records.append(pd.concat([meta_out.reset_index(drop=True), shap_sub[is_nowcast].reset_index(drop=True)], axis=1))

            print(f"  cluster={cluster:13s} lead={lead:2d}  n_nowcast_zones={int(is_nowcast.sum()):3d}  done")

    raw_all = pd.concat(raw_records, ignore_index=True)
    shap_all = pd.concat(shap_records, ignore_index=True)

    # Run the rule engine (unchanged) on the nowcast rows only.
    id_cols = ["time", "zone_code", "zone_name", "cluster", "lead", "subject",
               "base1", "xgb_ipc_continuous", "rf_ipc_continuous", "ens_ipc_continuous",
               "xgb_ipc_class", "rf_ipc_class", "ens_ipc_class"]
    scored = score_batch(raw_all, shap_all[need_cols], taxonomy, id_cols=id_cols, include_breakdown=True)

    # Pick the IPC-class/continuous columns matching each row's own subject.
    ipc_class_map = {"xgboost": "xgb_ipc_class", "randomforest": "rf_ipc_class", "ensemble": "ens_ipc_class"}
    ipc_cont_map = {"xgboost": "xgb_ipc_continuous", "randomforest": "rf_ipc_continuous", "ensemble": "ens_ipc_continuous"}
    scored["ipc_class"] = scored.apply(lambda r: r[ipc_class_map[r["subject"]]], axis=1)
    scored["ipc_continuous"] = scored.apply(lambda r: r[ipc_cont_map[r["subject"]]], axis=1)
    scored = scored.drop(columns=list(ipc_class_map.values()) + list(ipc_cont_map.values()))
    scored = scored.rename(columns={"time": "target_month"})

    out_path = NOWCAST_DIR / "nowcast_predictions.csv"
    scored.to_csv(out_path, index=False)
    print(f"\nwrote {len(scored)} rows to {out_path}")
    print(f"subjects: xgboost, randomforest, ensemble | clusters: {CLUSTERS} | leads: {LEADS}")
    print(f"predicted_cause distribution:\n{scored['predicted_cause'].value_counts()}")


if __name__ == "__main__":
    main()
