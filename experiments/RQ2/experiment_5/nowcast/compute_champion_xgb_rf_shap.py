"""Champion-ensemble explanation, part 1/2 -- per-instance SHAP for the
champion's two no-climate tree members (XGBoost, RandomForest), on the
same 644 nowcast rows build_xgb_rf_nowcast.py already scored.

These two members structurally cannot ever produce DROUGHT/FLOODING
evidence (exclude_feature_cluster=["climate"] means spi_/ssmi_/spei_/
ndvi/glofas_ columns are absent from their feature matrix entirely) --
rule_engine.py's own `col not in raw.index` handling already degrades
those two categories to a score of 0 gracefully, no special-casing needed
here. CONFLICT (acled_*) and MARKET_SHOCK (price/CPI/exchange-rate) are
NOT climate-excluded, so those two categories remain fully explainable
from these two members alone.

Mirrors run_model_ethiopia.py / run_rf_ethiopia.py's run_cell() feature-
prep logic exactly (same IDENTITY_DROP/PROVENANCE_DROP/
classify_feature_cluster, same delta-target construction) but stops short
of fit-and-score so the trained model object is available for SHAP,
instead of retraining from scratch a second time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RQ5_NOWCAST_DIR = REPO_ROOT / "experiments" / "RQ5" / "nowcast"
NOWCAST_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ1" / "experiment_2"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ2" / "experiment_4"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ5"))
import run_model_ethiopia as xgb_mod  # noqa: E402
import run_rf_ethiopia as rf_mod  # noqa: E402
from rule_engine import load_taxonomy, required_columns  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ2" / "experiment_5" / "nowcast"))
from build_xgb_rf_nowcast import build_concat_panel, CLUSTERS, LEADS, ORIGIN_MONTH  # noqa: E402

TRAIN_END = pd.Timestamp("2019-12-31")


def build_features(mod, df, cluster, target_month: pd.Timestamp):
    """Mirrors run_cell()'s feature-prep prefix (exclude_feature_cluster=
    ["climate"], target_mode="delta") up through the train/test split, but
    returns the raw matrices instead of fitting -- so the caller can fit
    once and reuse the model object for SHAP."""
    sub = df[(df["dominant_livelihood_zone"] == cluster) & (df["observed"] == True)].copy()
    sub = sub.dropna(axis=1, how="all")
    sub["time"] = pd.to_datetime(sub["month"])
    sub = sub[sub[mod.BASE1_COL].notna()]
    sub = sub.sort_values(["time", "zone_code"]).reset_index(drop=True)

    meta = sub[["time", "zone_code", "zone_name", "split"]].copy()
    labels = sub[mod.TARGET_COL]
    base1 = sub[mod.BASE1_COL]

    frame = sub.copy()
    excl_cols = [c for c in frame.columns if mod.classify_feature_cluster(c) == "climate"]
    frame = frame.drop(columns=excl_cols)
    if "season_system" in frame.columns:
        frame = pd.get_dummies(frame, columns=["season_system"], prefix="season", prefix_sep="_")
    drop_cols = [c for c in mod.IDENTITY_DROP + mod.PROVENANCE_DROP if c in frame.columns]
    frame = frame.drop(columns=drop_cols)
    frame = frame.drop(columns=frame.select_dtypes(include=["object", "category"]).columns)
    features = frame.drop(columns=["time"], errors="ignore")
    bool_cols = features.select_dtypes(include="bool").columns
    if len(bool_cols):
        features[bool_cols] = features[bool_cols].astype(int)

    train_mask = (meta["time"] <= TRAIN_END).to_numpy()
    test_mask = (meta["time"] == target_month).to_numpy()

    train_x = features[train_mask].reset_index(drop=True)
    train_y = labels[train_mask].reset_index(drop=True)
    base1_train = base1[train_mask].reset_index(drop=True)
    test_x = features[test_mask].reset_index(drop=True)
    test_meta = meta[test_mask].reset_index(drop=True)
    base1_test = base1[test_mask].reset_index(drop=True)

    fit_y = train_y - base1_train
    return train_x, fit_y, test_x, test_meta, base1_test


def main():
    taxonomy = load_taxonomy()
    need_cols = sorted(required_columns(taxonomy))

    raw_records = []
    shap_records = []

    for lead in LEADS:
        target_month = ORIGIN_MONTH + pd.DateOffset(months=lead)
        # build_concat_panel() is shared verbatim with build_xgb_rf_nowcast.py
        panel_xgb = build_concat_panel(lead, pd.read_csv(RQ5_NOWCAST_DIR / "nowcast_feature_rows.csv", low_memory=False))

        for cluster in CLUSTERS:
            train_x, fit_y, test_x, test_meta, base1_test = build_features(xgb_mod, panel_xgb, cluster, target_month)
            if len(test_x) == 0:
                continue

            xgb_model = XGBRegressor(**xgb_mod.XGB_PARAMS).fit(train_x, fit_y)
            xgb_explainer = shap.TreeExplainer(xgb_model)
            xgb_shap_full = xgb_explainer.shap_values(test_x)
            xgb_shap = pd.DataFrame(xgb_shap_full, columns=test_x.columns, index=test_x.index)
            avail_cols_xgb = [c for c in need_cols if c in test_x.columns]
            missing_xgb = [c for c in need_cols if c not in test_x.columns]

            train_x2, fit_y2, test_x2, test_meta2, base1_test2 = build_features(rf_mod, panel_xgb, cluster, target_month)
            train_medians = train_x2.median(numeric_only=True)
            rf_model = RandomForestRegressor(**rf_mod.RF_PARAMS).fit(train_x2.fillna(train_medians), fit_y2)
            rf_explainer = shap.TreeExplainer(rf_model)
            rf_shap_full = rf_explainer.shap_values(test_x2.fillna(train_medians))
            rf_shap = pd.DataFrame(rf_shap_full, columns=test_x2.columns, index=test_x2.index)
            avail_cols_rf = [c for c in need_cols if c in test_x2.columns]
            missing_rf = [c for c in need_cols if c not in test_x2.columns]

            for subject, shap_df, raw_df, avail, missing, meta in [
                ("champion_xgboost", xgb_shap, test_x, avail_cols_xgb, missing_xgb, test_meta),
                ("champion_randomforest", rf_shap, test_x2, avail_cols_rf, missing_rf, test_meta2),
            ]:
                shap_sub = shap_df[avail].copy()
                raw_sub = raw_df[avail].copy()
                for col in missing:
                    shap_sub[col] = np.nan
                    raw_sub[col] = np.nan
                shap_sub = shap_sub[need_cols]
                raw_sub = raw_sub[need_cols]

                meta_out = meta.assign(cluster=cluster, lead=lead, subject=subject,
                                        base1=(base1_test if "xgboost" in subject else base1_test2).to_numpy())
                raw_records.append(pd.concat([meta_out.reset_index(drop=True), raw_sub.reset_index(drop=True)], axis=1))
                shap_records.append(pd.concat([meta_out.reset_index(drop=True), shap_sub.reset_index(drop=True)], axis=1))

            print(f"  cluster={cluster:13s} lead={lead:2d}  done (champion xgb+rf)")

    raw_all = pd.concat(raw_records, ignore_index=True)
    shap_all = pd.concat(shap_records, ignore_index=True)
    raw_all.to_parquet(NOWCAST_DIR / "champion_xgb_rf_raw.parquet", index=False)
    shap_all.to_parquet(NOWCAST_DIR / "champion_xgb_rf_shap.parquet", index=False)
    print(f"\nwrote {len(raw_all)} rows to champion_xgb_rf_raw.parquet / champion_xgb_rf_shap.parquet")


if __name__ == "__main__":
    main()
