"""Shared loading/masking/prediction/SHAP plumbing for RQ4 (faithfulness +
stability of the SHAP explanation layer), reused by both experiment_1/ and
experiment_2/.

Three subjects, per the project-owner-confirmed scope (2026-08-27):
  - XGBoost:      experiments/RQ1/experiment_2/unhcr_delta_noclimate  (best single XGB)
  - RandomForest: experiments/RQ2/experiment_4/rf_delta_noclimate     (best single RF)
  - Ensemble:     the best-performing combination found in
                   experiments/RQ2/experiment_5/all_combinations/ (XGBoost +
                   RandomForest + LSTM + TabICLv2, F1=0.7218 default window) --
                   T-GCN excluded per that experiment's own leave-one-out finding.

None of these are retrained here -- XGBoost/RandomForest were re-run once
with --save-models added (deterministic, fixed random_state=42, numerically
identical to the already-logged results) purely to get persisted model
objects; LSTM already had persisted .pt weights from RQ2 Experiment 1;
TabICLv2 has no weights to persist (in-context foundation model -- .fit()
just conditions on the training set, ~0.4s, not gradient optimization), so
it is fit once per lead and that same fitted object is reused for every
masked/bootstrapped predict() call, matching "do not retrain" in spirit.

Scope (confirmed 2026-08-27): all 7 leads (0,1,2,3,4,8,12), pooled across
all 3 livelihood-zone clusters, on the default test window (train
<=2019-12, test 2020-01..2022-12, 828 rows/lead, Busker-parity split).

Ensemble SHAP-ranking scope (confirmed 2026-08-27, after an empirical
timing probe showed permutation SHAP for TabICLv2 costs ~75s/row --
~500 days for the literal 100-bootstrap x 7-lead spec): the ensemble's
*explanation layer* (SHAP ranking, used for top-k/bottom-k selection in
Experiment 1 and for bootstrap stability in Experiment 2) is built from
XGBoost + RandomForest + LSTM only -- all three are SHAP-tractable in
minutes, not days (TreeExplainer for the two tree models, GradientExplainer
for LSTM, both exact/fast). TabICLv2 is EXCLUDED from the SHAP-ranking
layer but STAYS a full voting member of the ensemble's actual *predictions*
in Experiment 1 (masking only needs its cheap predict() call, ~37s for a
full 828-row lead, not its explainer) -- consistent with this project's own
existing precedent that TabICLv2 never produces an independent SHAP ranking
anywhere else either (run_tabicl_ethiopia.py already selects its own top-80
input features from XGBoost's feature_importance.csv, not its own SHAP).

Masking convention (documented per model type, per the standard approach
for how each already handles missing data in this project):
  - XGBoost: mask with NaN. XGBoost's native missing-value handling (a
    learned default split direction per node) is the standard, no-injected-
    value way to represent "this feature is absent" for a tree booster.
  - RandomForest: mask with the TRAIN-derived median. sklearn's
    RandomForestRegressor cannot take NaN -- this project's own RF pipeline
    (run_rf_ethiopia.py) already uses train-derived median imputation as
    its established missing-value convention, reused here for consistency.
  - LSTM: mask with the TRAIN-derived median (in raw units, before
    standardization), then re-apply the same train-derived
    standardization -- matches run_lstm_ethiopia.py's own established
    impute-then-standardize convention exactly. Applied across all 12
    sequence timesteps for the named raw-panel column being masked.
  - TabICLv2: mask with NaN. TabICLClassifier's own built-in mean
    imputation is its documented, standard way of handling missing
    features (see CLAUDE.md's TabICLv2 entry) -- the model-native
    mechanism, same logic as XGBoost's NaN handling.

Combined ("ensemble") SHAP ranking construction: XGBoost and RandomForest
share an IDENTICAL engineered feature vocabulary (confirmed via run_meta.json
-- n_features matches exactly per cluster/lead between the two), so their
rankings are averaged directly by rank-percentile. LSTM uses a different
vocabulary (raw, pre-lead-shift panel columns, one value per sequence
timestep rather than pre-engineered lag columns) -- its ranking is
rank-percentile-averaged into the combined ranking wherever a feature NAME
coincides with the tree vocabulary, and kept as its own separate entry
otherwise (a partial-coverage Borda-style fusion, not a forced 1:1 merge).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RQ4_DIR = Path(__file__).resolve().parent
REPO_ROOT = RQ4_DIR.parent.parent
MODEL_TABLES_DIR = REPO_ROOT / "data" / "processed" / "model_tables"

XGB_RUN_DIR = REPO_ROOT / "experiments" / "RQ1" / "experiment_2" / "unhcr_delta_noclimate"
RF_RUN_DIR = REPO_ROOT / "experiments" / "RQ2" / "experiment_4" / "rf_delta_noclimate"
LSTM_RUN_DIR = REPO_ROOT / "experiments" / "RQ2" / "experiment_1" / "lstm_delta"
LSTM_SEQ_DIR = REPO_ROOT / "experiments" / "RQ2" / "experiment_1" / "sequences"
sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ1" / "experiment_2"))
import run_model_ethiopia as xgb_mod  # noqa: E402
# NOTE: run_tabicl_ethiopia is intentionally NOT imported here (nor is
# tabicl/torch) -- TabICLv2 runs entirely inside tabicl_worker.py, a
# separate process, to avoid the xgboost+pretrained-torch-weights segfault
# documented above. Only tabicl_worker.py itself imports it.

CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]
LEADS = [0, 1, 2, 3, 4, 8, 12]
TARGET_MODE = "delta"
EXCLUDE_FEATURE_CLUSTER = "climate"

# Test-window toggle, added 2026-09-02 for the RQ3/RQ4/RQ5 rebuild onto the
# project's revised primary window (docs/dissertation_plan.md section 6).
# "default" (2020-01..2022-12, the split column, unchanged) or "extended"
# (2020-01..2024-12, date-based, includes "beyond"-split rows). Set once via
# set_window() before any of this module's cell/ranking functions are
# called -- a module global, not threaded as a parameter through the whole
# call chain, matching this file's existing style (MODEL_TABLES_DIR etc.
# are already globals) and this script's one-shot-batch-job usage pattern
# (no concurrent/interleaved window use within a single run).
WINDOW = "default"
EXTENDED_TEST_START = pd.Timestamp("2020-01-01")
EXTENDED_TEST_END = pd.Timestamp("2024-12-31")


def set_window(window):
    global WINDOW
    assert window in ("default", "extended"), window
    WINDOW = window
SEED = 42
TABICL_TOP_N = 80

TARGET_COL = "ipc_continuous"
BASE1_COL = "ipc_lag1"


def to_ipc_class(x):
    return np.clip(np.round(np.asarray(x, dtype=float)), 1, 5).astype(int)


def weighted_f1(y_true_class, y_pred_class):
    from sklearn.metrics import f1_score
    return f1_score(y_true_class, y_pred_class, average="weighted", zero_division=0)


# ---------------------------------------------------------------------------
# XGBoost / RandomForest: shared feature-matrix construction (identical
# preprocessing between the two scripts except RF's extra median-impute
# step -- see run_model_ethiopia.py::run_cell / run_rf_ethiopia.py::run_cell).
# ---------------------------------------------------------------------------

def build_tabular_matrix(df, cluster, lead, impute):
    sub = df[(df["dominant_livelihood_zone"] == cluster) & (df["observed"] == True)].copy()
    sub = sub.dropna(axis=1, how="all")
    sub["time"] = pd.to_datetime(sub["month"])
    if WINDOW == "extended":
        sub = sub[sub["split"].isin(["train", "test", "beyond"])]
    else:
        sub = sub[sub["split"].isin(["train", "test"])]
    sub = sub[sub[BASE1_COL].notna()]
    sub = sub.sort_values(["time", "zone_code"]).reset_index(drop=True)

    meta = sub[["time", "zone_code", "zone_name", "split"]].copy()
    labels = sub[TARGET_COL]
    base1 = sub[BASE1_COL]

    frame = sub.copy()
    excl_cols = [c for c in frame.columns if xgb_mod.classify_feature_cluster(c) == EXCLUDE_FEATURE_CLUSTER]
    frame = frame.drop(columns=excl_cols)
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
    if WINDOW == "extended":
        test_mask = ((meta["time"] >= EXTENDED_TEST_START) & (meta["time"] <= EXTENDED_TEST_END)).to_numpy()
    else:
        test_mask = (meta["split"] == "test").to_numpy()

    train_medians = features[train_mask].median(numeric_only=True)
    if impute:
        features = features.fillna(train_medians)

    train_x = features[train_mask].reset_index(drop=True)
    test_x = features[test_mask].reset_index(drop=True)
    train_y = labels[train_mask].reset_index(drop=True)
    test_y = labels[test_mask].reset_index(drop=True)
    test_meta = meta[test_mask].reset_index(drop=True)
    base1_test = base1[test_mask].reset_index(drop=True)

    return {
        "train_x": train_x, "test_x": test_x,
        "train_y": train_y, "test_y": test_y,
        "test_meta": test_meta, "base1_test": base1_test,
        "train_medians": train_medians,
    }


_TABULAR_CACHE = {}


def get_tabular_cell(cluster, lead, impute):
    key = (cluster, lead, impute, WINDOW)
    if key not in _TABULAR_CACHE:
        df = xgb_mod.load_lead_table(str(MODEL_TABLES_DIR), lead)
        _TABULAR_CACHE[key] = build_tabular_matrix(df, cluster, lead, impute)
    return _TABULAR_CACHE[key]


def load_xgb_model(cluster, lead):
    from xgboost import XGBRegressor
    m = XGBRegressor()
    m.load_model(str(XGB_RUN_DIR / "models" / f"xgb_{cluster}_lead{lead}.json"))
    return m


def load_rf_model(cluster, lead):
    import joblib
    return joblib.load(RF_RUN_DIR / "models" / f"rf_{cluster}_lead{lead}.joblib")


def predict_delta(model, test_x, base1_test):
    return base1_test.to_numpy() + model.predict(test_x)


def mask_nan(test_x, cols):
    out = test_x.copy()
    present = [c for c in cols if c in out.columns]
    out[present] = np.nan
    return out


def mask_median(test_x, cols, train_medians):
    out = test_x.copy()
    present = [c for c in cols if c in out.columns]
    for c in present:
        out[c] = train_medians[c]
    return out


def xgb_shap_values(model, X):
    import shap
    explainer = shap.TreeExplainer(model)
    return explainer.shap_values(X)


def rf_shap_values(model, X):
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    return sv


# ---------------------------------------------------------------------------
# LSTM -- run out-of-process via lstm_worker.py.
#
# Empirically (2026-08-27), loading real (pretrained) nn.LSTM weights via
# load_state_dict() segfaults whenever xgboost/shap have already been
# imported in the same process (a native OpenMP/BLAS conflict, reproducible,
# unaffected by KMP_DUPLICATE_LIB_OK=TRUE) -- and this module imports
# xgboost at the top (via run_model_ethiopia). So every LSTM
# predict/SHAP call below shells out to a subprocess that never imports
# xgboost/tabicl (lstm_worker.py, confirmed safe standalone), passing
# masks/row-indices via a temp JSON file and reading results back from a
# temp CSV. Slower than an in-process call by a fixed per-call subprocess
# startup cost (~1-2s), but that is negligible next to the actual predict/
# SHAP compute time at every scale this project runs LSTM at.
# ---------------------------------------------------------------------------

import json
import subprocess
import tempfile

LSTM_WORKER = RQ4_DIR / "lstm_worker.py"


def load_lstm_feature_names():
    return json.loads((LSTM_SEQ_DIR / "feature_names.json").read_text())


def lstm_test_size(lead):
    """Pure-numpy read of the test-set row count for a lead -- safe to call
    in-process (no torch), unlike the actual LSTM model/predict/SHAP work
    which is isolated in lstm_worker.py (see that module's docstring)."""
    seq = np.load(LSTM_SEQ_DIR / f"lead{lead:02d}.npz", allow_pickle=True)
    if WINDOW == "extended":
        time = pd.to_datetime(seq["time"])
        return int(((time >= EXTENDED_TEST_START) & (time <= EXTENDED_TEST_END)).sum())
    return int((seq["split"] == "test").sum())


def _run_lstm_worker(lead, mode, mask_cols=None, row_idx=None):
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out_path = td / "out.csv"
        cmd = [sys.executable, str(LSTM_WORKER), "--lead", str(lead), "--mode", mode,
               "--window", WINDOW, "--out", str(out_path)]
        if mask_cols:
            mc_path = td / "mask_cols.json"
            mc_path.write_text(json.dumps(list(mask_cols)))
            cmd += ["--mask-cols-file", str(mc_path)]
        if row_idx is not None:
            ri_path = td / "row_idx.json"
            ri_path.write_text(json.dumps(np.asarray(row_idx).tolist()))
            cmd += ["--row-idx-file", str(ri_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"lstm_worker.py failed (lead={lead}, mode={mode}):\n{result.stderr}")
        return pd.read_csv(out_path)


def lstm_predictions(lead, mask_cols=None, row_idx=None):
    """Returns a DataFrame: time, zone_code, lead, lstm_pred, observed."""
    df = _run_lstm_worker(lead, "predict", mask_cols=mask_cols, row_idx=row_idx)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ---------------------------------------------------------------------------
# TabICLv2 -- run out-of-process via tabicl_worker.py, for the same reason
# as LSTM (see the LSTM section above): TabICLv2 is a pretrained-weight
# torch model internally, and this module has xgboost imported at the top.
# Only predict() is needed (no SHAP -- see module docstring for scope).
# ---------------------------------------------------------------------------

TABICL_WORKER = RQ4_DIR / "tabicl_worker.py"


def tabicl_predictions(lead, mask_cols=None):
    """Returns a DataFrame: time, zone_code, lead, tabicl_pred, observed_class."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out_path = td / "out.csv"
        cmd = [sys.executable, str(TABICL_WORKER), "--lead", str(lead), "--window", WINDOW, "--out", str(out_path)]
        if mask_cols:
            mc_path = td / "mask_cols.json"
            mc_path.write_text(json.dumps(list(mask_cols)))
            cmd += ["--mask-cols-file", str(mc_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"tabicl_worker.py failed (lead={lead}):\n{result.stderr}")
        df = pd.read_csv(out_path)
        df["time"] = pd.to_datetime(df["time"])
        return df


# ---------------------------------------------------------------------------
# Global SHAP-importance ranking construction (shared by experiment_1's
# once-off ranking and experiment_2's per-bootstrap ranking -- both call
# these with an optional row_idx to select/resample rows per cell).
# ---------------------------------------------------------------------------

def xgb_shap_importance_by_cell(cluster, lead, row_idx=None):
    cell = get_tabular_cell(cluster, lead, impute=False)
    model = load_xgb_model(cluster, lead)
    X = cell["test_x"] if row_idx is None else cell["test_x"].iloc[row_idx].reset_index(drop=True)
    sv = xgb_shap_values(model, X)
    sv = np.asarray(sv)
    importance = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    return importance, len(X)


def rf_shap_importance_by_cell(cluster, lead, row_idx=None):
    cell = get_tabular_cell(cluster, lead, impute=True)
    model = load_rf_model(cluster, lead)
    X = cell["test_x"] if row_idx is None else cell["test_x"].iloc[row_idx].reset_index(drop=True)
    sv = rf_shap_values(model, X)
    sv = np.asarray(sv)
    importance = pd.Series(np.abs(sv).mean(axis=0), index=X.columns)
    return importance, len(X)


def lstm_shap_importance_by_lead(lead, row_idx=None):
    df = _run_lstm_worker(lead, "shap", row_idx=row_idx)
    importance = pd.Series(df["importance"].to_numpy(), index=df["feature"])
    n = int(df["n_rows"].iloc[0])
    return importance, n


def lstm_shap_importance_batch_by_lead(lead, row_idx_list):
    """One subprocess call computes SHAP for MANY bootstrap draws at once
    (model + GradientExplainer background loaded once, not once per draw)
    -- see lstm_worker.py::run_shap_batch. Returns a DataFrame with columns
    bootstrap_iter, feature, importance, n_rows."""
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        out_path = td / "out.csv"
        batch_path = td / "row_idx_batch.json"
        batch_path.write_text(json.dumps([np.asarray(r).tolist() for r in row_idx_list]))
        cmd = [sys.executable, str(LSTM_WORKER), "--lead", str(lead), "--mode", "shap-batch",
               "--window", WINDOW, "--row-idx-batch-file", str(batch_path), "--out", str(out_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"lstm_worker.py shap-batch failed (lead={lead}):\n{result.stderr}")
        return pd.read_csv(out_path)


def rank_percentile(importance):
    """Higher importance -> higher percentile (1.0 = most important)."""
    ranks = importance.rank(method="average", ascending=True)
    return (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else ranks * 0 + 1.0


def weighted_average_importance(cell_importances):
    """cell_importances: list of (pd.Series feature->importance, n_rows).
    Returns one pd.Series, weighted-averaged by n_rows, over the union of
    feature names (features absent from a given cell simply don't
    contribute to that cell's weight for that feature)."""
    all_names = sorted(set().union(*[imp.index for imp, _ in cell_importances]))
    num = pd.Series(0.0, index=all_names)
    den = pd.Series(0.0, index=all_names)
    for imp, n in cell_importances:
        num.loc[imp.index] += imp.to_numpy() * n
        den.loc[imp.index] += n
    return (num / den.replace(0, np.nan)).dropna()


def xgb_global_ranking(row_idx_by_cell=None):
    """row_idx_by_cell: optional dict {(cluster, lead): row_idx array} for
    bootstrap resampling; None => use full test set for every cell."""
    cells = []
    for cluster in CLUSTERS:
        for lead in LEADS:
            ridx = None if row_idx_by_cell is None else row_idx_by_cell[(cluster, lead)]
            imp, n = xgb_shap_importance_by_cell(cluster, lead, ridx)
            cells.append((imp, n))
    return weighted_average_importance(cells)


def rf_global_ranking(row_idx_by_cell=None):
    cells = []
    for cluster in CLUSTERS:
        for lead in LEADS:
            ridx = None if row_idx_by_cell is None else row_idx_by_cell[(cluster, lead)]
            imp, n = rf_shap_importance_by_cell(cluster, lead, ridx)
            cells.append((imp, n))
    return weighted_average_importance(cells)


def lstm_global_ranking(row_idx_by_lead=None):
    cells = []
    for lead in LEADS:
        ridx = None if row_idx_by_lead is None else row_idx_by_lead[lead]
        imp, n = lstm_shap_importance_by_lead(lead, ridx)
        cells.append((imp, n))
    return weighted_average_importance(cells)


def combined_tree_ranking(xgb_imp, rf_imp):
    """XGBoost + RandomForest share an identical engineered vocabulary --
    average by rank-percentile (robust to the two models' very different
    raw |SHAP| magnitude scales, e.g. RF's deeper trees vs XGB's shallow
    ones)."""
    common = sorted(set(xgb_imp.index) & set(rf_imp.index))
    px = rank_percentile(xgb_imp.loc[common])
    pr = rank_percentile(rf_imp.loc[common])
    return ((px + pr) / 2).sort_values(ascending=False)


def combined_ensemble_ranking(xgb_imp, rf_imp, lstm_imp):
    """Ensemble SHAP-ranking scope per project-owner decision (2026-08-27):
    XGBoost + RandomForest (shared vocabulary, rank-percentile-averaged)
    fused with LSTM (separate raw-panel vocabulary) wherever a feature NAME
    coincides; features present in only one side keep that side's own
    percentile alone (partial-coverage Borda-style fusion). TabICLv2 is
    excluded from this ranking layer (see module docstring)."""
    tree = combined_tree_ranking(xgb_imp, rf_imp)
    tree_pct = rank_percentile(tree)
    lstm_pct = rank_percentile(lstm_imp)
    all_names = sorted(set(tree_pct.index) | set(lstm_pct.index))
    combined = pd.Series(index=all_names, dtype=float)
    for name in all_names:
        vals = []
        if name in tree_pct.index:
            vals.append(tree_pct[name])
        if name in lstm_pct.index:
            vals.append(lstm_pct[name])
        combined[name] = np.mean(vals)
    return combined.sort_values(ascending=False)
