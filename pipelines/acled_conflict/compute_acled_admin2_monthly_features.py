"""
compute_acled_admin2_monthly_features.py

AGGREGATE + ENGINEER stage. Point-in-polygon join of the cached ACLED
point events (fetch_acled_admin2.py) to admin2, per CLAUDE.md Sec 3.4 --
point-event data does NOT go through the raster zonal-stats pipeline.

SCOPE (confirmed with the project owner 2026-08-09): all ACLED disorder
types count -- Political violence (Battles, Violence against civilians,
Explosions/Remote violence), Strategic developments, and Demonstrations
(Riots, Protests). Not restricted to political-violence-only, since
civil-unrest signal (riots/protests) was judged still relevant to food
access disruption for this project.

FEATURES:
  - acled_event_count: count of ACLED events in the zone that month
  - acled_fatalities: sum of reported fatalities in the zone that month
  - Disorder-type disaggregation (added 2026-08-24, additive -- the two
    columns above are unchanged and still the single total): each event's
    `event_type` (not the coarser `disorder_type`, which has a 504-event
    "Political violence; Demonstrations" combined category that can't be
    cleanly split) is bucketed into exactly one of three groups, matching
    this file's own scope note above --
      political_violence: Battles, Violence against civilians,
        Explosions/Remote violence
      demonstrations: Protests, Riots
      strategic_developments: Strategic developments (ACLED's own
        residual category -- arrests, non-violent territory transfers,
        looting, etc; kept separate rather than folded into either of the
        other two since it's neither violent nor a demonstration)
    `acled_event_count_{bucket}` / `acled_fatalities_{bucket}` for each of
    the 3 buckets, sum to the totals above by construction (every
    event_type maps to exactly one bucket) -- checked at the end of
    main(). Motivation: an ablation this session found the pooled ACLED
    total contributes ~zero predictive value; a log1p transform (provably
    a no-op for a tree model) ruled out scale/skew as the cause, leaving
    "disorder types pooled into one count" as the remaining live
    hypothesis -- see experiments/RQ1/experiment_2/report.md sections 8-9.

ZERO vs. GAP: ACLED has continuous Ethiopia coverage from 2000-01 with no
documented reporting gap (unlike locust's 2022 gap between two disjoint
sources) -- a zone-month with no matching event is a genuine, confirmed
zero, not a missing-data placeholder. No NaN scaffold rows are produced.

BOUNDARY FALLBACK for points that don't land inside any admin2 polygon:
snap to the nearest zone if within 5 km, else drop and log the count --
same rule and tolerance as pipelines/locust/compute_locust_admin2_monthly_features.py.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "acled_conflict")
ACLED_CSV = os.path.join(RAW_DIR, "acled_ethiopia_2000_present.csv")

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "acled_admin2_monthly.csv")

LAGS = [1, 3, 6, 12]  # same lag set as every other price/index column (WFP prices, CPI, exchange rate)

EQUAL_AREA_CRS = "ESRI:102022"  # Africa Albers Equal Area Conic -- used only for the snap-distance check
SNAP_TOLERANCE_KM = 5.0

EVENT_TYPE_BUCKET = {
    "Battles": "political_violence",
    "Violence against civilians": "political_violence",
    "Explosions/Remote violence": "political_violence",
    "Protests": "demonstrations",
    "Riots": "demonstrations",
    "Strategic developments": "strategic_developments",
}
DISORDER_BUCKETS = ["political_violence", "demonstrations", "strategic_developments"]


def load_admin2():
    return gpd.read_file(ADMIN2_SHP_PATH)


def point_in_polygon_join(points_df, gdf_admin2, lon_col="longitude", lat_col="latitude"):
    """Point-in-polygon join with a nearest-zone fallback (Sec 3.4). Returns
    (joined_df_with_zone_code, n_dropped).

    points_df is reset to a contiguous RangeIndex first: building the
    GeoSeries separately and assigning it as a column aligns by index, so a
    non-contiguous input index (e.g. after the caller did `df[mask]`
    upstream) would silently pair each point with the wrong geometry."""
    points_df = points_df.reset_index(drop=True)
    geom = gpd.GeoSeries(
        [Point(xy) for xy in zip(points_df[lon_col], points_df[lat_col])],
        crs="EPSG:4326",
    )
    pts_gdf = gpd.GeoDataFrame(points_df.copy(), geometry=geom, crs="EPSG:4326")

    joined = gpd.sjoin(pts_gdf, gdf_admin2[[ADMIN2_PCODE_FIELD, "geometry"]],
                        how="left", predicate="within")
    joined = joined.drop(columns=["index_right"])

    missing_mask = joined[ADMIN2_PCODE_FIELD].isna()
    n_missing_initial = missing_mask.sum()

    if n_missing_initial > 0:
        gdf_proj = gdf_admin2.to_crs(EQUAL_AREA_CRS)
        missing_pts = pts_gdf.loc[missing_mask].to_crs(EQUAL_AREA_CRS)

        nearest = gpd.sjoin_nearest(
            missing_pts, gdf_proj[[ADMIN2_PCODE_FIELD, "geometry"]],
            how="left", distance_col="dist_m",
        )
        nearest = nearest[~nearest.index.duplicated(keep="first")]
        within_tolerance = nearest["dist_m"] <= (SNAP_TOLERANCE_KM * 1000)

        snapped = nearest.loc[within_tolerance, ADMIN2_PCODE_FIELD]
        joined.loc[snapped.index, ADMIN2_PCODE_FIELD] = snapped.values

        n_snapped = within_tolerance.sum()
        n_dropped = n_missing_initial - n_snapped
        print(f"  {n_missing_initial} points outside every polygon: "
              f"{n_snapped} snapped to nearest zone (<= {SNAP_TOLERANCE_KM} km), "
              f"{n_dropped} dropped (beyond tolerance).")
    else:
        n_dropped = 0

    joined = joined[joined[ADMIN2_PCODE_FIELD].notna()]
    return joined.rename(columns={ADMIN2_PCODE_FIELD: "zone_code"}), n_dropped


def main():
    gdf_admin2 = load_admin2()
    all_zone_codes = gdf_admin2[ADMIN2_PCODE_FIELD].tolist()

    events = pd.read_csv(ACLED_CSV, encoding="utf-8-sig", parse_dates=["event_date"])
    events["month"] = events["event_date"].dt.to_period("M")

    print("Point-in-polygon join: ACLED events")
    joined, n_dropped = point_in_polygon_join(events, gdf_admin2)

    unmapped = sorted(set(joined["event_type"].unique()) - set(EVENT_TYPE_BUCKET))
    if unmapped:
        raise SystemExit(
            f"EVENT_TYPE_BUCKET does not cover event_type(s): {unmapped} -- "
            f"fix the mapping before running, an unmapped event_type would "
            f"silently drop out of every bucketed column and break the "
            f"bucket-sums-to-total invariant checked at the end of main()."
        )
    joined["disorder_bucket"] = joined["event_type"].map(EVENT_TYPE_BUCKET)

    agg = joined.groupby(["zone_code", "month"]).agg(
        acled_event_count=("event_id_cnty", "size"),
        acled_fatalities=("fatalities", "sum"),
    ).reset_index()

    for bucket in DISORDER_BUCKETS:
        bucket_events = joined[joined["disorder_bucket"] == bucket]
        bucket_agg = bucket_events.groupby(["zone_code", "month"]).agg(
            **{f"acled_event_count_{bucket}": ("event_id_cnty", "size"),
               f"acled_fatalities_{bucket}": ("fatalities", "sum")}
        ).reset_index()
        agg = agg.merge(bucket_agg, on=["zone_code", "month"], how="left")

    # ------------------------------------------------------------------
    # Full zone x month scaffold across the observed date range --
    # no-event zone-months are a genuine, confirmed zero (see docstring),
    # so fillna(0) here is correct, not a gap-masking shortcut.
    # ------------------------------------------------------------------
    all_months = pd.period_range(events["month"].min(), events["month"].max(), freq="M").tolist()
    scaffold = pd.MultiIndex.from_product(
        [all_zone_codes, all_months], names=["zone_code", "month"]
    ).to_frame(index=False)

    bucket_cols = [f"acled_event_count_{b}" for b in DISORDER_BUCKETS] + \
                  [f"acled_fatalities_{b}" for b in DISORDER_BUCKETS]

    out = scaffold.merge(agg, on=["zone_code", "month"], how="left")
    for col in ["acled_event_count", "acled_fatalities"] + bucket_cols:
        out[col] = out[col].fillna(0).astype(int)

    out["month"] = out["month"].dt.to_timestamp()
    out = out.sort_values(["zone_code", "month"]).reset_index(drop=True)

    # Bucket columns must sum to the totals by construction (every event
    # maps to exactly one bucket) -- a real check, not a formality: if it
    # ever fails, EVENT_TYPE_BUCKET has stopped covering every event_type
    # ACLED reports (e.g. a new sub_event_type added upstream).
    event_sum = out[[f"acled_event_count_{b}" for b in DISORDER_BUCKETS]].sum(axis=1)
    fatal_sum = out[[f"acled_fatalities_{b}" for b in DISORDER_BUCKETS]].sum(axis=1)
    assert (event_sum == out["acled_event_count"]).all(), \
        "BUG: bucketed event counts do not sum to acled_event_count"
    assert (fatal_sum == out["acled_fatalities"]).all(), \
        "BUG: bucketed fatalities do not sum to acled_fatalities"

    # ------------------------------------------------------------------
    # Lag features (1/3/6/12 months). Computed on the complete, zero-filled
    # zone x month scaffold above -- since every zone has a true monthly
    # calendar with no gaps (unlike WFP prices), shift(n) per zone already
    # equals "n calendar months ago", no additional scaffold needed. Backward
    # -looking only (pandas .shift() with a positive lag pulls from strictly
    # earlier rows within the same zone_code group) -- no data leakage.
    # ------------------------------------------------------------------
    for lag in LAGS:
        out[f"acled_event_count_lag{lag}"] = out.groupby("zone_code")["acled_event_count"].shift(lag)
        out[f"acled_fatalities_lag{lag}"] = out.groupby("zone_code")["acled_fatalities"].shift(lag)
        for bucket in DISORDER_BUCKETS:
            out[f"acled_event_count_{bucket}_lag{lag}"] = (
                out.groupby("zone_code")[f"acled_event_count_{bucket}"].shift(lag)
            )
            out[f"acled_fatalities_{bucket}_lag{lag}"] = (
                out.groupby("zone_code")[f"acled_fatalities_{bucket}"].shift(lag)
            )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(out)} zone-month rows x {len(out.columns)} columns -> {OUTPUT_PATH}")
    print(f"Zones: {out['zone_code'].nunique()} (expected {len(all_zone_codes)})")
    print(f"Months: {out['month'].min().date()} to {out['month'].max().date()}")
    print(f"Zone-months with any event: {(out['acled_event_count'] > 0).sum()} / {len(out)} "
          f"({(out['acled_event_count'] > 0).mean():.2%})")
    print(f"Total events joined: {out['acled_event_count'].sum()} (raw file had {len(events)}, "
          f"{n_dropped} dropped beyond {SNAP_TOLERANCE_KM}km tolerance)")
    print(f"Total fatalities: {out['acled_fatalities'].sum()}")
    print("Bucket breakdown (events / fatalities):")
    for bucket in DISORDER_BUCKETS:
        print(f"  {bucket}: {out[f'acled_event_count_{bucket}'].sum()} / "
              f"{out[f'acled_fatalities_{bucket}'].sum()}")
    print("PASS: bucketed event counts and fatalities sum to the totals.")


if __name__ == "__main__":
    main()
