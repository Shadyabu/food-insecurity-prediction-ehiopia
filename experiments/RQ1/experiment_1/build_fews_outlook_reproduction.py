"""FEWS NET outlook (ML1/ML2) benchmark, independently reproduced -- not
transcribed from the paper's Figure 5. Scores weighted F1 (this project's
primary metric) and R2 (a sanity check against the paper) for Ethiopia's
92 admin2 zones, 2020-2024, by livelihood-zone cluster x lead.

WHY THIS DOESN'T NEED A NEW ACQUISITION STEP
----------------------------------------------
`pipelines/ipc_target/` already fetches ML1/ML2 from FDW's live `ipcphase`
API (not just CS), already rasterizes them onto the CHIRPS grid indexed by
`collection_date` (`fetch_fews_ipc.py`, RUN_TIME_KEY="collection" -- the
same indexing convention Busker's own released fews_xr_ML1/ML2.nc files
use), and already population-weight-aggregates them to admin2
(`aggregate_fews_ipc.py`) with the same validation Busker et al.'s own
grids were checked against (99.5%/100% agreement). FDW's live archive runs
to 2026-06, well past Busker's cached snapshot (issue months only to
2023-06). So `data/processed/features/ipc_target_admin2_panel.csv` already
holds everything needed at admin2 level for ML1/ML2 -- the only piece
missing is the per-record lead time, which lived in the fnid-level tidy
panel (`data/interim/ipc_target/ipc_panel_long.csv`) before rasterization
collapsed it away. This script joins that lead metadata back on, matches
each outlook to the nearest real observed CS release, and scores it.

LEAD DEFINITION
----------------
Each ML1/ML2 record is a single map covering a multi-month window
(period_start..projection_end), not one map per target month -- confirmed
empirically (period_start/projection_end are constant within an issue,
0 fnid collisions per issue). `lead_end` (months from collection_date to
projection_end, already computed in ipc_panel_long.csv) is used as THE
lead, and the outlook's phase is scored against the CS release nearest to
projection_end -- the most conservative (longest-horizon) reading of what
the map is actually claiming. This gives ML1 leads {0,1,2,3} and ML2
leads {4,5,6,7} -- NOT Busker's own {1,2,3,4}/{5,6,7,8} month-tuple
scheme (his was a heuristic reconstruction from raw shapefiles with no
issue-date field per feature; FDW's lead_end is the direct, principled
number and is used as-is rather than force-fit to his numbering). Also
NOT this project's usual {0,1,2,3,4,8,12} grid -- FEWS's own outlook
horizon never reaches lead 8 or 12, matching the already-documented
GloFAS/IRI-CPC/USGS-GEFS pattern of forecast products with a shorter
reach than the model's own lead grid.

VERIFICATION MATCH
--------------------
projection_end does not always land exactly on a CS release month (CS is
tri-annual, Feb/June/Oct-ish; projection_end months are more scattered).
Each outlook is matched to the NEAREST real CS observation for the same
zone within a 31-day tolerance; anything further is dropped as
unverifiable (never forward-filled -- consistent with this project's
"observed rows only" rule for the target).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent.parent
FEATURES_DIR = ROOT / "data" / "processed" / "features"
INTERIM_DIR = ROOT / "data" / "interim" / "ipc_target"
MODEL_TABLE = ROOT / "data" / "processed" / "model_tables" / "ethiopia_lead00.csv"
OUT_DIR = SCRIPT_DIR
MATCH_TOLERANCE_DAYS = 31
TEST_START = "2020-01-01"
TEST_END = "2024-12-31"


def load_lead_lookup() -> pd.DataFrame:
    """(scenario, collection_date) -> (projection_end, lead_end). One row per
    issue -- period_start/projection_end/lead_end are constant within an
    issue (verified empirically, 0 exceptions across 94,603 ML1 records)."""
    long = pd.read_csv(INTERIM_DIR / "ipc_panel_long.csv", low_memory=False,
                        usecols=["scenario", "collection_date", "projection_end", "lead_end"])
    long = long[long["scenario"].isin(["ML1", "ML2"])].dropna(subset=["collection_date"])
    long["collection_date"] = pd.to_datetime(long["collection_date"])
    long["projection_end"] = pd.to_datetime(long["projection_end"])
    lookup = long.drop_duplicates(subset=["scenario", "collection_date"])
    dup_check = long.groupby(["scenario", "collection_date"])[["projection_end", "lead_end"]].nunique()
    bad = dup_check[(dup_check["projection_end"] > 1) | (dup_check["lead_end"] > 1)]
    if len(bad):
        raise SystemExit(f"projection_end/lead_end not constant within {len(bad)} issues -- "
                          f"lead-lookup assumption violated, do not proceed silently")
    return lookup[["scenario", "collection_date", "projection_end", "lead_end"]]


def load_outlook_and_observed():
    panel = pd.read_csv(FEATURES_DIR / "ipc_target_admin2_panel.csv", parse_dates=["time"])

    observed = panel[panel["scenario"] == "CS"][
        ["zone_code", "time", "ipc_continuous", "ipc_phase_20pct"]
    ].rename(columns={"time": "cs_date", "ipc_continuous": "obs_continuous",
                       "ipc_phase_20pct": "obs_phase"}).dropna(subset=["obs_phase"])

    outlook = panel[panel["scenario"].isin(["ML1", "ML2"])][
        ["zone_code", "scenario", "time", "ipc_continuous", "ipc_phase_20pct"]
    ].rename(columns={"time": "collection_date", "ipc_continuous": "pred_continuous",
                       "ipc_phase_20pct": "pred_phase"}).dropna(subset=["pred_phase"])

    return outlook, observed


def match_to_nearest_cs(outlook: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """merge_asof per zone: nearest CS release to each outlook's projection_end,
    within MATCH_TOLERANCE_DAYS. Unmatched rows are dropped, not filled."""
    matched = []
    n_dropped = 0
    for zone, sub_outlook in outlook.groupby("zone_code"):
        sub_obs = observed[observed["zone_code"] == zone].sort_values("cs_date")
        if sub_obs.empty:
            n_dropped += len(sub_outlook)
            continue
        sub_outlook = sub_outlook.sort_values("projection_end")
        m = pd.merge_asof(
            sub_outlook, sub_obs.drop(columns=["zone_code"]),
            left_on="projection_end", right_on="cs_date",
            direction="nearest", tolerance=pd.Timedelta(days=MATCH_TOLERANCE_DAYS),
        )
        n_dropped += m["cs_date"].isna().sum()
        matched.append(m.dropna(subset=["cs_date"]))
    result = pd.concat(matched, ignore_index=True)
    print(f"  matched {len(result)} outlook records to a CS release within "
          f"{MATCH_TOLERANCE_DAYS}d; dropped {n_dropped} unverifiable")
    return result


def main():
    print("[1] Loading lead lookup and outlook/observed panels")
    lookup = load_lead_lookup()
    outlook, observed = load_outlook_and_observed()

    print("[2] Joining lead_end/projection_end onto outlook rows")
    outlook = outlook.merge(lookup, on=["scenario", "collection_date"], how="inner")
    print(f"  {len(outlook)} outlook rows carry a resolved lead")

    print("[3] Matching each outlook to the nearest real CS observation")
    df = match_to_nearest_cs(outlook, observed)

    print(f"[4] Restricting to target window {TEST_START}..{TEST_END}")
    df = df[(df["cs_date"] >= TEST_START) & (df["cs_date"] <= TEST_END)].copy()
    print(f"  {len(df)} rows remain")

    print("[5] Attaching livelihood-zone cluster")
    zones = pd.read_csv(MODEL_TABLE, usecols=["zone_code", "dominant_livelihood_zone"]).drop_duplicates()
    df = df.merge(zones, on="zone_code", how="left")
    if df["dominant_livelihood_zone"].isna().any():
        raise SystemExit("unmatched zone_code against dominant_livelihood_zone lookup")

    df["pred_phase"] = df["pred_phase"].astype(int)
    df["obs_phase"] = df["obs_phase"].astype(int)
    df["pred_phase_3plus"] = np.clip(df["pred_phase"], 1, 3)
    df["obs_phase_3plus"] = np.clip(df["obs_phase"], 1, 3)

    print("[6] Scoring weighted F1 / R2 by cluster x lead")
    rows = []
    for (cluster, lead), grp in df.groupby(["dominant_livelihood_zone", "lead_end"]):
        if len(grp) < 5:
            continue
        rows.append({
            "cluster": cluster, "lead_end": lead, "scenario": grp["scenario"].iloc[0],
            "n": len(grp), "n_zones": grp["zone_code"].nunique(),
            "weighted_f1": f1_score(grp["obs_phase"], grp["pred_phase"], average="weighted", zero_division=0),
            "weighted_f1_3plus": f1_score(grp["obs_phase_3plus"], grp["pred_phase_3plus"], average="weighted", zero_division=0),
            "r2": r2_score(grp["obs_continuous"], grp["pred_continuous"]) if grp["obs_continuous"].nunique() > 1 else np.nan,
        })
    result = pd.DataFrame(rows).sort_values(["cluster", "lead_end"])

    also_all = []
    for lead, grp in df.groupby("lead_end"):
        if len(grp) < 5:
            continue
        also_all.append({
            "cluster": "all_zones", "lead_end": lead, "scenario": grp["scenario"].iloc[0],
            "n": len(grp), "n_zones": grp["zone_code"].nunique(),
            "weighted_f1": f1_score(grp["obs_phase"], grp["pred_phase"], average="weighted", zero_division=0),
            "weighted_f1_3plus": f1_score(grp["obs_phase_3plus"], grp["pred_phase_3plus"], average="weighted", zero_division=0),
            "r2": r2_score(grp["obs_continuous"], grp["pred_continuous"]) if grp["obs_continuous"].nunique() > 1 else np.nan,
        })
    result = pd.concat([result, pd.DataFrame(also_all)], ignore_index=True)

    out_path = OUT_DIR / "fews_outlook_f1_2020_2024.csv"
    result.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(result.to_string(index=False))

    df[["zone_code", "dominant_livelihood_zone", "scenario", "collection_date",
        "projection_end", "lead_end", "cs_date", "pred_phase", "obs_phase",
        "pred_continuous", "obs_continuous"]].to_csv(OUT_DIR / "fews_outlook_matched_rows.csv", index=False)


if __name__ == "__main__":
    main()
