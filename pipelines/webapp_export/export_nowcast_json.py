"""Web export, Phase 1b -- join experiments/RQ5/nowcast/nowcast_predictions.csv,
experiments/RQ2/experiment_5/nowcast/champion_nowcast_predictions.csv, and
experiments/RQ2/experiment_5/nowcast/champion_explanations.csv with
boundaries/livelihood_zones_admin2.csv into one compact JSON the frontend
fetches directly.

UNIFIED SELECTOR DESIGN (confirmed with the project owner 2026-09-04,
superseding the earlier split design): one dropdown drives BOTH the map's
IPC phase AND the "Reason for prediction" panel, using that same selected
model's own prediction and its own explanation -- 3 exposed subjects
(RQ5's own 2-way "ensemble" is computed and validated but deliberately
NOT exposed in the app, per the project owner's 2026-09-04 request --
xgboost/randomforest alone already cover the self-consistent, fully-
explainable option; the exported `nowcast_predictions.csv` upstream still
has all 3, this script just filters at the export boundary):
  - xgboost, randomforest: RQ5's own full-feature, climate-included
    models (unchanged) -- fully self-consistent, fully explainable at
    every phase.
  - champion_ensemble: RQ2 Experiment 5's 4-way champion (XGBoost-no-
    climate + RandomForest-no-climate + LSTM + TabICLv2) for the
    PREDICTION (this project's highest-accuracy combination), paired with
    the best explanation actually achievable for it -- SHAP averaged
    across the 3 members with a tractable per-instance SHAP path
    (XGBoost-no-climate, RandomForest-no-climate, LSTM; TabICLv2 has no
    tractable per-instance SHAP path anywhere in this project, same
    exclusion precedent as experiments/RQ4/common.py's SHAP-ranking
    layer). Two of those three members have zero climate features, so
    DROUGHT/FLOODING evidence is real but structurally diluted for this
    subject -- surfaced to the frontend via a `limitation` string, shown
    as an on-page asterisk/footnote rather than hidden.

This is the single re-run seam for the project owner's stated long-term
goal (monthly automated updates): once experiments/RQ5/nowcast/'s 3
scripts AND experiments/RQ2/experiment_5/nowcast/'s 7 scripts are re-run
against fresher data, re-running this script is the only other step
needed -- no frontend code changes required.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RQ5_NOWCAST_DIR = REPO_ROOT / "experiments" / "RQ5" / "nowcast"
CHAMPION_NOWCAST_DIR = REPO_ROOT / "experiments" / "RQ2" / "experiment_5" / "nowcast"
LIVELIHOOD_ZONES_PATH = REPO_ROOT / "boundaries" / "livelihood_zones_admin2.csv"
OUT_PATH = REPO_ROOT / "frontend" / "data" / "nowcast.json"

LEADS = [0, 1, 2, 3, 4, 8, 12]
RQ5_SUBJECTS = ["xgboost", "randomforest", "ensemble"]  # all 3 present upstream in nowcast_predictions.csv
EXPORTED_RQ5_SUBJECTS = ["xgboost", "randomforest"]  # "ensemble" deliberately not exposed in the app
ALL_SUBJECTS = EXPORTED_RQ5_SUBJECTS + ["champion_ensemble"]
TOP_SHAP_N = 5

CHAMPION_LIMITATION = (
    "* Prediction from the 4-way champion ensemble (highest accuracy in this "
    "project). Explanation averages only the 3 of 4 members with a tractable "
    "per-instance SHAP path (XGBoost-no-climate, RandomForest-no-climate, "
    "LSTM) -- TabICLv2 excluded. Two of those three have NO climate features "
    "at all, so drought/flood evidence is real (from LSTM) but diluted, and "
    "under-represented versus the other explanation models."
)

# SHAP breakdown columns are named shap_<CATEGORY>_<feature> -- split back
# into (category, feature) for the frontend rather than exposing the raw
# concatenated column name. MARKET_SHOCK itself contains an underscore, so
# a naive partition("_") on the first underscore is wrong (it would split
# "MARKET_SHOCK_price_volatility" into category="MARKET" instead of
# "MARKET_SHOCK") -- must match against the taxonomy's own known category
# names, longest-first so "MARKET_SHOCK" wins over any shorter prefix.
SHAP_PREFIX = "shap_"
CATEGORY_NAMES = sorted(["DROUGHT", "FLOODING", "CONFLICT", "MARKET_SHOCK"], key=len, reverse=True)


def shap_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(SHAP_PREFIX)]


def top_shap_for_row(row: pd.Series, shap_cols: list[str]) -> list[dict]:
    contribs = []
    for col in shap_cols:
        val = row[col]
        if pd.isna(val) or val <= 0:
            continue
        rest = col[len(SHAP_PREFIX):]
        category = next(c for c in CATEGORY_NAMES if rest.startswith(c + "_"))
        feature = rest[len(category) + 1:]
        contribs.append({"category": category, "feature": feature, "shap": round(float(val), 6)})
    contribs.sort(key=lambda d: d["shap"], reverse=True)
    return contribs[:TOP_SHAP_N]


def to_ipc_class(x):
    return int(min(5, max(1, round(x))))


def main():
    rq5_preds = pd.read_csv(RQ5_NOWCAST_DIR / "nowcast_predictions.csv")
    champion_preds = pd.read_csv(CHAMPION_NOWCAST_DIR / "champion_nowcast_predictions.csv")
    champion_expl = pd.read_csv(CHAMPION_NOWCAST_DIR / "champion_explanations.csv")
    champion_expl = champion_expl[champion_expl["subject"] == "champion_ensemble"].copy()
    lz = pd.read_csv(LIVELIHOOD_ZONES_PATH)

    assert set(rq5_preds["zone_code"]) == set(lz["zone_code"]), "zone_code mismatch: RQ5 predictions vs livelihood zones"
    assert set(rq5_preds["lead"].unique()) == set(LEADS), f"RQ5: expected leads {LEADS}, got {sorted(rq5_preds['lead'].unique())}"
    assert set(rq5_preds["subject"].unique()) == set(RQ5_SUBJECTS), \
        f"RQ5: expected subjects {RQ5_SUBJECTS}, got {sorted(rq5_preds['subject'].unique())}"
    assert set(champion_preds["zone_code"]) == set(lz["zone_code"]), "zone_code mismatch: champion predictions vs livelihood zones"
    assert set(champion_preds["lead"].unique()) == set(LEADS), \
        f"champion: expected leads {LEADS}, got {sorted(champion_preds['lead'].unique())}"
    assert len(champion_expl) == 92 * len(LEADS), f"champion_ensemble explanation: expected {92 * len(LEADS)} rows, got {len(champion_expl)}"

    origin_month = pd.to_datetime(rq5_preds[rq5_preds["lead"] == 0]["target_month"]).iloc[0].strftime("%Y-%m")
    champion_origin_month = pd.to_datetime(champion_preds[champion_preds["lead"] == 0]["target_month"]).iloc[0].strftime("%Y-%m")
    assert origin_month == champion_origin_month, \
        f"RQ5 nowcast origin ({origin_month}) and champion ensemble origin ({champion_origin_month}) must match"

    rq5_shap_cols = shap_columns(rq5_preds)
    champion_shap_cols = shap_columns(champion_expl)
    lz_by_zone = lz.set_index("zone_code")
    champion_preds_by_zone_lead = champion_preds.set_index(["zone_code", "lead"])
    champion_expl_by_zone_lead = champion_expl.set_index(["zone_code", "lead"])

    zone_names = rq5_preds.drop_duplicates("zone_code").set_index("zone_code")["zone_name"]

    zones_out = {}
    for zone_code in lz["zone_code"]:
        lz_row = lz_by_zone.loc[zone_code]
        zone_entry = {
            "zone_name": zone_names.loc[zone_code],
            "region": lz_row["region"],
            "dominant_livelihood_zone": lz_row["dominant_livelihood_zone"],
            "pct_pastoral": round(float(lz_row["pct_pastoral"]), 4),
            "pct_agropastoral": round(float(lz_row["pct_agropastoral"]), 4),
            "pct_crop_farming": round(float(lz_row["pct_crop_farming"]), 4),
            "predictions": {},
        }
        zone_rq5 = rq5_preds[rq5_preds["zone_code"] == zone_code]
        for lead, lead_preds in zone_rq5.groupby("lead"):
            target_month = pd.to_datetime(lead_preds["target_month"].iloc[0]).strftime("%Y-%m")
            lead_entry = {"target_month": target_month}

            for _, row in lead_preds[lead_preds["subject"].isin(EXPORTED_RQ5_SUBJECTS)].iterrows():
                lead_entry[row["subject"]] = {
                    "ipc_class": int(row["ipc_class"]),
                    "ipc_continuous": round(float(row["ipc_continuous"]), 3),
                    "predicted_cause": row["predicted_cause"],
                    "is_compound": bool(row["is_compound"]),
                    "top_shap": top_shap_for_row(row, rq5_shap_cols),
                }

            champ_pred_row = champion_preds_by_zone_lead.loc[(zone_code, lead)]
            champ_expl_row = champion_expl_by_zone_lead.loc[(zone_code, lead)]
            lead_entry["champion_ensemble"] = {
                "ipc_class": to_ipc_class(champ_pred_row["ipc_continuous"]),
                "ipc_continuous": round(float(champ_pred_row["ipc_continuous"]), 3),
                "predicted_cause": champ_expl_row["predicted_cause"],
                "is_compound": bool(champ_expl_row["is_compound"]),
                "top_shap": top_shap_for_row(champ_expl_row, champion_shap_cols),
                "limitation": CHAMPION_LIMITATION,
            }

            zone_entry["predictions"][str(int(lead))] = lead_entry
        zones_out[zone_code] = zone_entry

    out = {
        "origin_month": origin_month,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "leads": LEADS,
        "subjects": ALL_SUBJECTS,
        "default_subject": "champion_ensemble",
        "zones": zones_out,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=None, separators=(",", ":"))

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"Wrote {len(zones_out)} zones x {len(LEADS)} leads x {len(ALL_SUBJECTS)} subjects to {OUT_PATH} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
