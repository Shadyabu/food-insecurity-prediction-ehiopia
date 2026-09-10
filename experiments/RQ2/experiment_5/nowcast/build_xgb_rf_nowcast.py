"""Champion-ensemble nowcast, part 1/3 -- XGBoost-no-climate and
RandomForest-no-climate members, scored on genuinely future target months
(2026-07 .. 2027-06) from the fixed 2026-06 origin.

Reuses experiments/RQ5/nowcast/build_nowcast_features.py's already-built,
architecture-agnostic synthetic feature rows (experiments/RQ5/nowcast/
nowcast_feature_rows.csv) -- those rows carry the FULL raw feature set;
which columns actually get used is decided downstream by each model's own
selection logic (here: run_cell()'s exclude_feature_cluster=["climate"]),
so no separate no-climate row-construction is needed.

Rather than loading the champion's saved model checkpoints and manually
reindexing to their feature lists (the approach used for RQ5's own models,
where SHAP explanation required a persisted model object), this reuses
run_model_ethiopia.py / run_rf_ethiopia.py's own run_cell() function
directly -- the exact function that produced unhcr_delta_noclimate/
rf_delta_noclimate in the first place. Retraining is deterministic
(random_state=42, identical train data/split), so this is numerically
identical to those saved models, and reusing run_cell() end-to-end
guarantees the feature-selection logic can never drift from what
production-scored those two variants everywhere else in this project.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RQ5_NOWCAST_DIR = REPO_ROOT / "experiments" / "RQ5" / "nowcast"
NOWCAST_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ1" / "experiment_2"))
sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ2" / "experiment_4"))
import run_model_ethiopia as xgb_mod  # noqa: E402
import run_rf_ethiopia as rf_mod  # noqa: E402

MODEL_TABLES_DIR = REPO_ROOT / "data" / "processed" / "model_tables"
NOWCAST_FEATURE_ROWS_PATH = RQ5_NOWCAST_DIR / "nowcast_feature_rows.csv"

CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]
LEADS = [0, 1, 2, 3, 4, 8, 12]
ORIGIN_MONTH = pd.Timestamp("2026-06-01")
TRAIN_END = "2019-12"


def build_concat_panel(lead: int, nowcast_rows: pd.DataFrame) -> pd.DataFrame:
    """Same schema-alignment approach as experiments/RQ5/nowcast/
    compute_nowcast_shap_and_causes.py::build_concat_panel() -- see that
    module's docstring for why the historical target-month row is dropped
    before concatenating the synthetic one (lead=0 only)."""
    historical = xgb_mod.load_lead_table(str(MODEL_TABLES_DIR), lead)
    target_month_str = (ORIGIN_MONTH + pd.DateOffset(months=lead)).strftime("%Y-%m")
    historical = historical[historical["month"] != target_month_str]

    synthetic = nowcast_rows[nowcast_rows["lead"] == lead].drop(columns=["lead", "target_month"])
    missing_in_synthetic = set(historical.columns) - set(synthetic.columns)
    missing_in_historical = set(synthetic.columns) - set(historical.columns)
    if missing_in_synthetic:
        raise ValueError(f"lead {lead}: synthetic rows missing columns: {sorted(missing_in_synthetic)}")
    if missing_in_historical:
        synthetic = synthetic.drop(columns=sorted(missing_in_historical))
    synthetic = synthetic[historical.columns]

    for col in historical.columns:
        if historical[col].dtype != synthetic[col].dtype:
            try:
                synthetic[col] = synthetic[col].astype(historical[col].dtype)
            except (ValueError, TypeError):
                pass

    return pd.concat([historical, synthetic], ignore_index=True)


def main():
    nowcast_rows = pd.read_csv(NOWCAST_FEATURE_ROWS_PATH, low_memory=False)

    records = []
    for lead in LEADS:
        panel = build_concat_panel(lead, nowcast_rows)
        target_month = ORIGIN_MONTH + pd.DateOffset(months=lead)
        target_month_str = target_month.strftime("%Y-%m")

        for cluster in CLUSTERS:
            common_kwargs = dict(
                cluster=cluster, lead=lead, verbose=False,
                target_mode="delta", exclude_feature_cluster=["climate"],
                train_end=TRAIN_END, test_start=target_month_str, test_end=target_month_str,
            )
            xgb_out, _, _ = xgb_mod.run_cell(panel, **common_kwargs)
            rf_out, _, _ = rf_mod.run_cell(panel, **common_kwargs)

            if len(xgb_out) == 0:
                print(f"  cluster={cluster:13s} lead={lead:2d}  no zones in this cluster -- skipping")
                continue

            assert list(xgb_out["zone_code"]) == list(rf_out["zone_code"]), \
                f"lead {lead} {cluster}: XGBoost/RandomForest zone order mismatch"

            for i in range(len(xgb_out)):
                records.append({
                    "zone_code": xgb_out["zone_code"].iloc[i],
                    "lead": lead,
                    "target_month": target_month_str,
                    "cluster": cluster,
                    "xgb_pred": xgb_out["prediction"].iloc[i],
                    "rf_pred": rf_out["prediction"].iloc[i],
                })
            print(f"  cluster={cluster:13s} lead={lead:2d}  n={len(xgb_out):3d}  done")

    out = pd.DataFrame(records)
    out_path = NOWCAST_DIR / "xgb_rf_nowcast.csv"
    out.to_csv(out_path, index=False)
    print(f"\nwrote {len(out)} rows to {out_path}")


if __name__ == "__main__":
    main()
