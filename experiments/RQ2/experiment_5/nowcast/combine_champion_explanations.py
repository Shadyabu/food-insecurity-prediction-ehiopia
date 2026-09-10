"""Champion-ensemble explanation, final step -- run the rule engine on
each explainable member (champion_xgboost, champion_randomforest,
champion_lstm) and build a combined "champion_ensemble" explanation
subject (TabICLv2 excluded -- no tractable per-instance SHAP path exists
anywhere in this project, per experiments/RQ4/common.py's own established
precedent for its SHAP-ranking layer).

Raw feature values are NOT identical across the 3 members here (unlike
RQ5's own script, where all subjects shared one full-feature input) --
XGBoost-noclimate/RandomForest-noclimate structurally have NO climate
columns at all, while LSTM (full-feature) does. The combined subject's
raw value per taxonomy feature is the first available value across
[xgboost, randomforest, lstm] (they agree whenever more than one has it);
its SHAP value is the mean across whichever members actually have that
feature (pandas .mean(skipna=True)) -- for climate features this reduces
to "LSTM's own contribution" (the only member that can see them), for
conflict/market features it's a true 3-way average. This is the
project-owner-confirmed "whatever explanation is possible" design: DROUGHT
and FLOODING CAN surface as a champion_ensemble cause via LSTM's
contribution alone, just with 2 of 3 members contributing nothing to that
category, rather than the null result get from the two no-climate members.
"""

from pathlib import Path

import pandas as pd
import sys

NOWCAST_DIR = Path(__file__).resolve().parent
REPO_ROOT = NOWCAST_DIR.parent.parent.parent.parent
RQ5_DIR = REPO_ROOT / "experiments" / "RQ5"
sys.path.insert(0, str(RQ5_DIR))
from rule_engine import load_taxonomy, required_columns, score_batch  # noqa: E402

ID_COLS = ["zone_code", "lead", "target_month"]


def load_member(raw_path, shap_path, subject_name, need_cols):
    raw = pd.read_parquet(raw_path)
    shap = pd.read_parquet(shap_path)
    raw = raw[raw["subject"] == subject_name] if "subject" in raw.columns else raw
    shap = shap[shap["subject"] == subject_name] if "subject" in shap.columns else shap
    # compute_champion_xgb_rf_shap.py carries "time" (Timestamp); lstm_explain_worker.py
    # already writes "target_month" (string) -- standardize on the latter.
    for df in (raw, shap):
        if "target_month" not in df.columns:
            df["target_month"] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m")
    raw = raw.sort_values(["lead", "zone_code"]).reset_index(drop=True)
    shap = shap.sort_values(["lead", "zone_code"]).reset_index(drop=True)
    assert list(raw["zone_code"]) == list(shap["zone_code"]) and list(raw["lead"]) == list(shap["lead"]), \
        f"{subject_name}: raw/shap row order mismatch"
    return raw[ID_COLS + need_cols].set_index(["zone_code", "lead"]), shap[ID_COLS + need_cols].set_index(["zone_code", "lead"])


def main():
    taxonomy = load_taxonomy()
    need_cols = sorted(required_columns(taxonomy))

    xgb_raw, xgb_shap = load_member(
        NOWCAST_DIR / "champion_xgb_rf_raw.parquet", NOWCAST_DIR / "champion_xgb_rf_shap.parquet",
        "champion_xgboost", need_cols,
    )
    rf_raw, rf_shap = load_member(
        NOWCAST_DIR / "champion_xgb_rf_raw.parquet", NOWCAST_DIR / "champion_xgb_rf_shap.parquet",
        "champion_randomforest", need_cols,
    )
    lstm_raw, lstm_shap = load_member(
        NOWCAST_DIR / "champion_lstm_raw.parquet", NOWCAST_DIR / "champion_lstm_shap.parquet",
        "champion_lstm", need_cols,
    )

    common_index = xgb_raw.index
    assert set(rf_raw.index) == set(common_index) and set(lstm_raw.index) == set(common_index), \
        "index mismatch across champion explanation members"
    rf_raw, rf_shap = rf_raw.loc[common_index], rf_shap.loc[common_index]
    lstm_raw, lstm_shap = lstm_raw.loc[common_index], lstm_shap.loc[common_index]

    # Raw: first available value across [xgboost, randomforest, lstm].
    ens_raw = xgb_raw[need_cols].combine_first(rf_raw[need_cols]).combine_first(lstm_raw[need_cols])
    # SHAP: mean across whichever members actually have that feature.
    stacked = pd.concat([xgb_shap[need_cols], rf_shap[need_cols], lstm_shap[need_cols]], keys=["x", "r", "l"])
    ens_shap = stacked.groupby(level=[1, 2]).mean()  # group back to (zone_code, lead)
    ens_shap = ens_shap.loc[common_index]

    meta = xgb_raw.reset_index()[ID_COLS]

    all_raw = []
    all_shap = []
    for subject, raw_df, shap_df in [
        ("champion_xgboost", xgb_raw, xgb_shap),
        ("champion_randomforest", rf_raw, rf_shap),
        ("champion_lstm", lstm_raw, lstm_shap),
        ("champion_ensemble", ens_raw, ens_shap),
    ]:
        raw_df = raw_df.reset_index()[["zone_code", "lead"]].join(raw_df[need_cols].reset_index(drop=True))
        shap_df = shap_df.reset_index()[["zone_code", "lead"]].join(shap_df[need_cols].reset_index(drop=True))
        raw_df = raw_df.merge(meta, on=["zone_code", "lead"])
        shap_df = shap_df.merge(meta, on=["zone_code", "lead"])
        raw_df["subject"] = subject
        shap_df["subject"] = subject
        all_raw.append(raw_df)
        all_shap.append(shap_df)

    raw_all = pd.concat(all_raw, ignore_index=True)
    shap_all = pd.concat(all_shap, ignore_index=True)

    id_cols = ["zone_code", "lead", "target_month", "subject"]
    scored = score_batch(raw_all, shap_all[need_cols], taxonomy, id_cols=id_cols, include_breakdown=True)

    out_path = NOWCAST_DIR / "champion_explanations.csv"
    scored.to_csv(out_path, index=False)
    print(f"wrote {len(scored)} rows to {out_path}")
    print(f"\npredicted_cause by subject:")
    print(scored.groupby("subject")["predicted_cause"].value_counts())


if __name__ == "__main__":
    main()
