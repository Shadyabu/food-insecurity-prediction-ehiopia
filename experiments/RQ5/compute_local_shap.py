"""RQ5 Phase 1 infrastructure — compute per-instance (local) SHAP values and
raw feature values for the taxonomy-required columns, on the full default
test window (2020-01..2022-12, 828 zone-months per lead), for the three
model subjects RQ5 runs the rule engine against.

MODEL CHOICE, deviating from RQ2/RQ4's "best model" (confirmed with project
owner 2026-08-31, "run multiple subjects, compare"):
  RQ2 Experiment 5's best single/ensemble models (unhcr_delta_noclimate,
  rf_delta_noclimate, and the 4-way ensemble built from them) all use
  --exclude-feature-cluster climate. FEATURE_CLUSTER_PREFIXES["climate"] in
  run_model_ethiopia.py covers every spi_/ssmi_/spei_/ndvi/glofas_ column --
  i.e. every single feature this taxonomy's DROUGHT and FLOODING categories
  read from. Those two subjects therefore have ZERO possible SHAP evidence
  for drought/flood under the no-climate models -- structurally incapable
  of ever producing the two cause labels RQ5 exists to test (the project
  owner's own illustrative cases are literally "Borena/Afder/Dawa/Liban
  (drought)"). This is a hard architectural conflict, not a style choice.
  RQ5 therefore uses the full-feature (climate-included) delta-target
  models instead:
    - XGBoost:      experiments/RQ1/experiment_2/delta_variant/
                     (target_mode=delta, exclude_feature_cluster=None,
                     confirmed via its own run_meta.json)
    - RandomForest:  experiments/RQ2/experiment_4/rf_delta_default_withmodels/
                     (same recipe; rf_delta_default itself had no saved
                     model objects -- re-run once with --save-models added,
                     confirmed numerically identical metrics_per_lead.csv to
                     the original rf_delta_default, same "re-run once,
                     deterministic" convention RQ4 already used for the
                     no-climate variants)
  These are each individually WORSE predictors than the RQ2 champion
  ensemble (that is precisely why the no-climate ablation was adopted as
  the champion in RQ1 Experiment 2 section 6.1) -- flag this tradeoff
  explicitly in the RQ5 write-up: better local-explanation coverage,
  weaker predictive accuracy, than the "best model" language in the
  original RQ5 spec would suggest.

Ensemble subject: instance-level fusion of XGBoost + RandomForest SHAP by
simple average (both share an identical engineered feature vocabulary,
confirmed via matching column sets below) -- the natural per-instance
analogue of RQ4 common.py's own global rank-percentile tree fusion. LSTM/
TabICLv2 are excluded from this local layer, same rationale RQ4 already
established for its own SHAP-ranking layer (different/no fast per-instance
SHAP path) -- not repeated here since RQ5 doesn't touch either model.

RAW VALUES used for the rule engine's polarity gate are taken PRE-impute
(NaN preserved) even for the RandomForest subject, whose SHAP is computed
on the train-median-imputed matrix (what the model actually saw) -- an
imputed/fabricated median must not be read by the taxonomy as if it were a
real observed anomaly. See rule_engine.py's polarity-gate docstring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RQ5_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ1" / "experiment_2"))
sys.path.insert(0, str(RQ5_DIR))
import run_model_ethiopia as xgb_mod  # noqa: E402
from rule_engine import load_taxonomy, required_columns  # noqa: E402

MODEL_TABLES_DIR = REPO_ROOT / "data" / "processed" / "model_tables"
XGB_RUN_DIR = REPO_ROOT / "experiments" / "RQ1" / "experiment_2" / "delta_variant_current"
RF_RUN_DIR = REPO_ROOT / "experiments" / "RQ2" / "experiment_4" / "rf_delta_default_withmodels"
OUT_DIR = RQ5_DIR / "shap_cache"

CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]
LEADS = [0, 1, 2, 3, 4, 8, 12]
TARGET_COL = "ipc_continuous"
BASE1_COL = "ipc_lag1"


def build_full_feature_matrix(df: pd.DataFrame, cluster: str, lead: int):
    """Mirrors run_model_ethiopia.py::run_cell's feature-prep block exactly
    (target_mode='delta', exclude_feature_cluster=None, default Busker-
    parity split) but does not fit -- just reconstructs the matrices the
    saved delta_variant/rf_delta_default_withmodels models were trained on,
    so a persisted model can be loaded and explained directly."""
    sub = df[(df["dominant_livelihood_zone"] == cluster) & (df["observed"] == True)].copy()
    sub = sub.dropna(axis=1, how="all")
    sub["time"] = pd.to_datetime(sub["month"])
    sub = sub[sub["split"].isin(["train", "test"])]
    sub = sub[sub[BASE1_COL].notna()]
    sub = sub.sort_values(["time", "zone_code"]).reset_index(drop=True)

    meta = sub[["time", "zone_code", "zone_name", "split"]].copy()
    base1 = sub[BASE1_COL]

    frame = sub.copy()
    if "season_system" in frame.columns:
        frame = pd.get_dummies(frame, columns=["season_system"], prefix="season", prefix_sep="_")
    drop_cols = [c for c in xgb_mod.IDENTITY_DROP + xgb_mod.PROVENANCE_DROP if c in frame.columns]
    frame = frame.drop(columns=drop_cols)
    frame = frame.drop(columns=frame.select_dtypes(include=["object", "category"]).columns)
    features = frame.drop(columns=["time"], errors="ignore")
    bool_cols = features.select_dtypes(include="bool").columns
    if len(bool_cols):
        features[bool_cols] = features[bool_cols].astype(int)

    train_mask = (meta["split"] == "train").to_numpy()
    test_mask = (meta["split"] == "test").to_numpy()

    test_x_raw = features[test_mask].reset_index(drop=True)
    test_meta = meta[test_mask].reset_index(drop=True)
    base1_test = base1[test_mask].reset_index(drop=True)
    train_medians = features[train_mask].median(numeric_only=True)

    return test_x_raw, test_meta, base1_test, train_medians


def load_xgb_model(cluster, lead):
    m = XGBRegressor()
    m.load_model(str(XGB_RUN_DIR / "models" / f"xgb_{cluster}_lead{lead}.json"))
    return m


def load_rf_model(cluster, lead):
    return joblib.load(RF_RUN_DIR / "models" / f"rf_{cluster}_lead{lead}.joblib")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    taxonomy = load_taxonomy()
    need_cols = sorted(required_columns(taxonomy))

    raw_records = []
    shap_records = []

    for lead in LEADS:
        df = xgb_mod.load_lead_table(str(MODEL_TABLES_DIR), lead)
        for cluster in CLUSTERS:
            test_x_raw, test_meta, base1_test, train_medians = build_full_feature_matrix(df, cluster, lead)
            # Forecast-product columns (GloFAS/IRI-CPC/USGS-GEFS, incl.
            # glofas_exceed_2yr) are pre-selected per lead at model-table
            # build time and only populated at the leads their source
            # product actually reaches (lead1/lead3 for GloFAS) -- CLAUDE.md
            # section 2. Absent at other leads is expected, not an error;
            # missing columns are added back as all-NaN so every lead's
            # output has the same schema and the rule engine's own
            # `col not in raw.index` / NaN handling applies uniformly.
            avail_cols = [c for c in need_cols if c in test_x_raw.columns]
            missing_cols = [c for c in need_cols if c not in test_x_raw.columns]
            if missing_cols:
                print(f"  ({cluster} lead{lead}: {missing_cols} not in feature matrix at this lead -- filled NaN)")

            xgb_model = load_xgb_model(cluster, lead)
            xgb_explainer = shap.TreeExplainer(xgb_model)
            xgb_shap_full = xgb_explainer.shap_values(test_x_raw)
            xgb_shap = pd.DataFrame(xgb_shap_full, columns=test_x_raw.columns)[avail_cols]

            test_x_imputed = test_x_raw.fillna(train_medians)
            rf_model = load_rf_model(cluster, lead)
            rf_explainer = shap.TreeExplainer(rf_model)
            rf_shap_full = rf_explainer.shap_values(test_x_imputed)
            rf_shap = pd.DataFrame(rf_shap_full, columns=test_x_imputed.columns)[avail_cols]

            for missing_col in missing_cols:
                xgb_shap[missing_col] = np.nan
                rf_shap[missing_col] = np.nan
            xgb_shap = xgb_shap[need_cols]
            rf_shap = rf_shap[need_cols]

            ens_shap = (xgb_shap + rf_shap) / 2.0

            n = len(test_meta)
            base_meta = test_meta.assign(cluster=cluster, lead=lead, base1=base1_test.to_numpy())
            raw_sub = test_x_raw.reindex(columns=need_cols).reset_index(drop=True)

            for subject, shap_sub in [("xgboost", xgb_shap), ("randomforest", rf_shap), ("ensemble", ens_shap)]:
                meta_out = base_meta.copy()
                meta_out["subject"] = subject
                raw_records.append(pd.concat([meta_out.reset_index(drop=True), raw_sub], axis=1))
                shap_records.append(pd.concat([meta_out.reset_index(drop=True), shap_sub.reset_index(drop=True)], axis=1))

            print(f"cluster={cluster:13s} lead={lead:2d}  n_test={n:4d}  done (xgb+rf+ensemble)")

    raw_all = pd.concat(raw_records, ignore_index=True)
    shap_all = pd.concat(shap_records, ignore_index=True)
    raw_all.to_parquet(OUT_DIR / "raw_values.parquet", index=False)
    shap_all.to_parquet(OUT_DIR / "shap_values.parquet", index=False)
    print(f"\nwrote {len(raw_all)} rows to {OUT_DIR}/raw_values.parquet and shap_values.parquet")
    print(f"subjects: xgboost, randomforest, ensemble | clusters: {CLUSTERS} | leads: {LEADS}")


if __name__ == "__main__":
    main()
