"""
compute_locust_admin2_monthly_features.py

AGGREGATE + ENGINEER stage. Takes the two cached raw point sets from
fetch_locust_admin2.py and produces the final zone-month feature table.

METHOD (matches Busker et al.'s stated approach: "the total area affected
each month was calculated as a percentage of the overall area within the
defined administrative division"):

  1. Point-in-polygon join each source's points to zone_code (admin2),
     per CLAUDE.md Sec 3.4 -- point-event data does NOT go through the
     raster zonal-stats pipeline.
  2. Sum reported swarm area (hectares) per zone-month, divide by the
     zone's own area (computed via an equal-area reprojection, since the
     admin2 shapefile's native Shape_Area column is in raw degrees^2 and
     not usable directly), cap at 100%.
  3. Build a complete zone x month scaffold across the full 1986-2026
     range so every zone-month has a row -- no silent drops for
     zero-sighting zones (Sec 3.3 / Sec 5 of the locust task).

THREE-WAY ROW STATE (not a plain 0/NaN choice):
  - Busker-archive period (1986-07 to 2021-12) and RAMSES-live period
    (2023-01 to present): a zone-month with no matching swarm point gets
    an explicit 0.0 -- both sources are presence-event archives that
    would have recorded a report had one occurred, so "no report" is
    informative here.
  - The 2022 gap (no source covers this window at all, see
    fetch_locust_admin2.py's docstring): NaN, not 0 -- we have no
    evidence either way. Carried via locust_data_source ==
    "no_source_2022_gap" so this is never silently conflated with a
    confirmed-zero month downstream.

BOUNDARY FALLBACK for points that don't land inside any admin2 polygon
(coastline/border precision, or a coordinate that's just outside a zone's
edge): snap to the nearest zone if within 5 km, else drop and log the
count -- decided up front per CLAUDE.md Sec 3.4, not defaulted silently.
"""

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "locust")
BUSKER_CSV = os.path.join(RAW_DIR, "busker_swarms_ethiopia_1986_2021.csv")
RAMSES_CSV = os.path.join(RAW_DIR, "ramses_swarms_ethiopia_live.csv")

ADMIN2_SHP_PATH = os.path.join(REPO_ROOT, "boundaries", "eth_admbnda_adm2_csa_bofedb_2021.shp")
ADMIN2_PCODE_FIELD = "ADM2_PCODE"

OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "locust_admin2_monthly.csv")

# Equal-area projection for zone-area computation -- Africa Albers Equal
# Area Conic. The shapefile's own Shape_Area column is raw lon/lat
# degrees^2 (sums to ~92.7, not a usable area), confirmed by inspection.
EQUAL_AREA_CRS = "ESRI:102022"

SNAP_TOLERANCE_KM = 5.0  # boundary fallback for points outside every polygon

BUSKER_COVERAGE_START = "1986-07"  # first actual Busker record; keep the scaffold to real coverage
BUSKER_COVERAGE_END = "2021-12"
GAP_START = "2022-01"
GAP_END = "2022-12"
RAMSES_COVERAGE_START = "2023-01"
FEATURE_END_MONTH = "2026-06"  # keep in sync with other pipelines' FEATURE_END_YEAR


def load_admin2():
    gdf = gpd.read_file(ADMIN2_SHP_PATH)
    gdf_proj = gdf.to_crs(EQUAL_AREA_CRS)
    gdf["area_km2"] = gdf_proj.geometry.area / 1e6
    return gdf


def point_in_polygon_join(points_df, gdf_admin2, lon_col="longitude", lat_col="latitude"):
    """Point-in-polygon join with a nearest-zone fallback (Sec 3.4). Returns
    (joined_df_with_zone_code, n_dropped)."""
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
        # sjoin_nearest can return duplicate matches on exact ties -- keep first
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


def ha_to_km2(area_ha):
    return area_ha / 100.0


def swarm_size_to_km2(row):
    unit = row["Swarm_Size_Unit"]
    size = row["Maximum_Size_Sum"]
    if pd.isna(unit) or unit in ("N/A", None) or pd.isna(size) or size == 0:
        return np.nan
    if unit == "ha":
        return ha_to_km2(size)
    if unit in ("km²", "km2"):
        return float(size)
    # Unrecognised unit -- surface it rather than silently guessing a conversion.
    raise ValueError(f"Unhandled Swarm_Size_Unit value: {unit!r}")


def main():
    gdf_admin2 = load_admin2()
    all_zone_codes = gdf_admin2[ADMIN2_PCODE_FIELD].tolist()
    zone_area = gdf_admin2.set_index(ADMIN2_PCODE_FIELD)["area_km2"]

    # ------------------------------------------------------------------
    # 1. Load + join both raw sources
    # ------------------------------------------------------------------
    busker = pd.read_csv(BUSKER_CSV, parse_dates=["event_date"])
    busker["area_km2"] = ha_to_km2(busker["area_ha"])

    ramses = pd.read_csv(RAMSES_CSV, parse_dates=["event_date"])
    ramses["area_km2"] = ramses.apply(swarm_size_to_km2, axis=1)

    print("Point-in-polygon join: Busker archive")
    busker_joined, busker_dropped = point_in_polygon_join(busker, gdf_admin2)
    print("Point-in-polygon join: RAMSES live")
    ramses_joined, ramses_dropped = point_in_polygon_join(ramses, gdf_admin2)

    busker_joined["month"] = busker_joined["event_date"].dt.to_period("M")
    ramses_joined["month"] = ramses_joined["event_date"].dt.to_period("M")

    points = pd.concat([
        busker_joined[["zone_code", "month", "area_km2"]],
        ramses_joined[["zone_code", "month", "area_km2"]],
    ], ignore_index=True)

    # ------------------------------------------------------------------
    # 2. Aggregate to zone-month
    # ------------------------------------------------------------------
    agg = points.groupby(["zone_code", "month"]).agg(
        locust_area_km2_affected=("area_km2", lambda s: s.sum(skipna=True)),
        locust_sightings_surveyed=("area_km2", "size"),
    ).reset_index()

    # ------------------------------------------------------------------
    # 3. Full zone x month scaffold, with explicit gap handling
    # ------------------------------------------------------------------
    covered_months = (
        pd.period_range(BUSKER_COVERAGE_START, BUSKER_COVERAGE_END, freq="M").tolist()
        + pd.period_range(RAMSES_COVERAGE_START, FEATURE_END_MONTH, freq="M").tolist()
    )
    gap_months = pd.period_range(GAP_START, GAP_END, freq="M").tolist()
    all_months = sorted(covered_months + gap_months)

    scaffold = pd.MultiIndex.from_product(
        [all_zone_codes, all_months], names=["zone_code", "month"]
    ).to_frame(index=False)

    out = scaffold.merge(agg, on=["zone_code", "month"], how="left")

    is_gap = out["month"].isin(gap_months)
    out.loc[is_gap, ["locust_area_km2_affected", "locust_sightings_surveyed"]] = np.nan
    out.loc[~is_gap, "locust_area_km2_affected"] = out.loc[~is_gap, "locust_area_km2_affected"].fillna(0.0)
    out.loc[~is_gap, "locust_sightings_surveyed"] = out.loc[~is_gap, "locust_sightings_surveyed"].fillna(0).astype(int)

    out["zone_area_km2"] = out["zone_code"].map(zone_area)
    out["locust_pct_area_affected"] = (out["locust_area_km2_affected"] / out["zone_area_km2"] * 100).clip(upper=100)
    out.loc[is_gap, "locust_pct_area_affected"] = np.nan

    out["locust_data_source"] = np.select(
        [out["month"].isin(covered_months) & (out["month"] <= pd.Period(BUSKER_COVERAGE_END, "M")),
         out["month"].isin(gap_months),
         out["month"] >= pd.Period(RAMSES_COVERAGE_START, "M")],
        ["busker_archive_1986_2021", "no_source_2022_gap", "ramses_live_2023_present"],
        default="unknown",
    )

    out["month"] = out["month"].dt.to_timestamp()
    out = out[["zone_code", "month", "locust_pct_area_affected",
               "locust_sightings_surveyed", "locust_data_source"]]
    out = out.sort_values(["zone_code", "month"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nWrote {len(out)} zone-month rows -> {OUTPUT_PATH}")
    print(f"Zones: {out['zone_code'].nunique()} (expected {len(all_zone_codes)})")
    print(f"Months: {out['month'].min().date()} to {out['month'].max().date()}")
    print(out["locust_data_source"].value_counts())
    print(f"Non-gap rows with locust_pct_area_affected > 0: "
          f"{(out['locust_pct_area_affected'] > 0).sum()}")
    print(f"Points dropped (outside {SNAP_TOLERANCE_KM}km of any zone): "
          f"{busker_dropped + ramses_dropped} of {len(busker) + len(ramses)}")


if __name__ == "__main__":
    main()
