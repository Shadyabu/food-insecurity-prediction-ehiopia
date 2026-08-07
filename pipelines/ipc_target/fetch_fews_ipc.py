#!/usr/bin/env python3
"""
fetch_fews_ipc.py
=================
Acquisition and preparation of the FEWS NET IPC target variable, following
Busker et al. (2024).

This covers Stages 1-2 of the four-stage pipeline pattern:

    [1. ACQUIRE]   FDW REST API -> cached raw CSV/GeoJSON
    [2. RASTERIZE] fsc_admin polygons -> IPC phase on the CHIRPS 0.05deg grid
    ---- (this script ends here) ----
    [3. AGGREGATE] population-weighted mean -> 92-zone admin2 panel   (aggregate_fews_ipc.py)
    [4. VALIDATE]  compare against Busker's published rasters          (`validate` below)

WHY RASTERIZE ONTO THE CHIRPS GRID
----------------------------------
Busker's own released files (fews_xr_CS.nc / _ML1.nc / _ML2.nc) are NOT
admin-aggregated tables. They are rasters on exactly the CHIRPS p05 grid:

    lon 33.025 .. 51.375  (368 cells)   = CHIRPS global index 4260..4627
    lat 14.875 .. -4.675  (392 cells)   = CHIRPS global index 1297.. (descending)

Verified: (lon - 0.025 + 180)/0.05 and (lat - 0.025 + 50)/0.05 are both exact
integers. So his IPC grid aligns cell-for-cell with the CHIRPS rainfall grid
already built for this project. Reproducing that intermediate gives a direct,
cell-level validation reference *before* any population weighting is applied,
which isolates rasterization errors from aggregation errors. Do not skip it.

BUSKER FILE STRUCTURE (measured, not assumed)
---------------------------------------------
  fews_xr_CS.nc   47 timesteps, 2009-07-01 .. 2022-06-01
                  vars: CS (phases 1-5), HA (humanitarian assistance, 0/1/2)
                  cadence: quarterly Jan/Apr/Jul/Oct to 2015, then
                           tri-annual Feb/Jun/Oct from 2016 (+ one 2018-12)
                  CS contains 0 at exactly one timestep (2017-02-01, 3756
                  cells) - treat 0 as missing, not as a phase.
                  HA is all-NaN before 2012 (assistance only recorded from 2012).
  fews_xr_ML1.nc  51 timesteps, 2019-02-01 .. 2023-06-01, MONTHLY
  fews_xr_ML2.nc  (missing 2023-02 and 2023-03), phases 1-5 only

Note the asymmetry: CS is tri-annual over 13 years; ML1/ML2 are monthly but
only from 2019. Any benchmark comparison is therefore restricted to 2019+.

USAGE
-----
    python fetch_fews_ipc.py inspect-busker
    python fetch_fews_ipc.py fetch --countries ET
    python fetch_fews_ipc.py tidy
    python fetch_fews_ipc.py rasterize --extent hoa --like busker_comparison/fews_xr_CS.nc
    python fetch_fews_ipc.py validate

Run them in order; each stage is independently re-runnable and caches its
output, so fixing step 3 never forces a re-download of step 1.

KNOWN GAP: humanitarian food assistance (HA)
--------------------------------------------
Busker's CS file carries an HA layer and he aggregates it with the same
population weighting. The FDW `ipcphase` endpoint exposes scenario=CS/ML1/ML2
only; HA is not obviously available there. It IS present in the downloadable
per-period shapefile bundles from https://fews.net/data/acute-food-insecurity
as a separate layer. See fetch_ha() below - deliberately left unimplemented
rather than guessed at.
"""

from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

BASE_URL = "https://fdw.fews.net/api"

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent  # repo root (pipelines/ipc_target/ -> pipelines/ -> repo root)
RAW_DIR = ROOT / "data" / "raw" / "ipc_target"
INTERIM_DIR = ROOT / "data" / "interim" / "ipc_target"
BUSKER_DIR = SCRIPT_DIR / "busker_comparison"

SCENARIOS = ["CS", "ML1", "ML2"]

# Horn of Africa countries used by Busker et al. (for the RQ1 baseline
# reproduction). Ethiopia alone for the RQ2 purpose-built dataset.
HOA_COUNTRIES = ["ET", "KE", "SO", "SD", "SS", "UG"]

# FEWS NET food-security classification mapping units. Ethiopia is mapped on
# fsc_admin; some HoA countries (notably Somalia) use the admin x livelihood
# zone intersection, so request both.
UNIT_TYPES = ["fsc_admin", "fsc_admin_lhz"]

# Values that are NOT IPC phases and must be masked before any arithmetic.
# FEWS NET encodes water / protected areas / missing with high sentinels;
# 0 also appears (one timestep in Busker's CS) and is not a phase either.
# The script logs every distinct value it sees so this list can be corrected
# against reality rather than trusted blindly.
NON_PHASE_CODES = {0, 66, 77, 88, 99}
VALID_PHASES = {1, 2, 3, 4, 5}

# Bounding boxes (west, south, east, north) in degrees.
EXTENTS = {
    # Busker's published extent, reconstructed exactly from fews_xr_CS.nc.
    "hoa": (33.0, -4.7, 51.4, 14.9),
    # Ethiopia with a small margin.
    "ethiopia": (32.9, 3.3, 48.1, 15.0),
}

CHIRPS_RES = 0.05
REQUEST_TIMEOUT = 600  # these extracts are large; be generous


# --------------------------------------------------------------------------
# Grid construction
# --------------------------------------------------------------------------

def chirps_grid(bbox):
    """Return (lon, lat) cell centres snapped to the CHIRPS p05 global grid.

    CHIRPS global cell centres are at -180 + 0.05*i + 0.025 (lon) and
    -50 + 0.05*j + 0.025 (lat). Snapping to that grid rather than building a
    fresh linspace over the bbox is what guarantees cell-for-cell alignment
    with the rainfall features and with Busker's own rasters. Latitude is
    returned DESCENDING (north first), matching both CHIRPS convention and
    rasterio.transform.from_origin's implicit assumption - see CHIRPS
    pipeline documentation section 4.2 for the ~0.55deg offset bug this avoids.
    """
    west, south, east, north = bbox

    i0 = int(np.ceil((west + 180 - 0.025) / CHIRPS_RES))
    i1 = int(np.floor((east + 180 - 0.025) / CHIRPS_RES))
    j0 = int(np.ceil((south + 50 - 0.025) / CHIRPS_RES))
    j1 = int(np.floor((north + 50 - 0.025) / CHIRPS_RES))

    lon = -180 + CHIRPS_RES * np.arange(i0, i1 + 1) + 0.025
    lat = -50 + CHIRPS_RES * np.arange(j0, j1 + 1) + 0.025
    lat = lat[::-1]  # descending: row 0 = north

    return lon, lat


def grid_from_reference(nc_path):
    """Copy the exact lon/lat coordinates from a reference NetCDF.

    Preferred over chirps_grid() when validating against Busker, because it
    removes any float32 rounding difference (his coords carry ~4e-7 artefacts
    inherited from CHIRPS) and makes the comparison a straight array subtraction
    with no reindexing.
    """
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        return ds["lon"].values.copy(), ds["lat"].values.copy()


def transform_from_grid(lon, lat):
    """Affine transform for a descending-latitude regular grid."""
    from rasterio.transform import from_origin

    res_x = float(abs(lon[1] - lon[0]))
    res_y = float(abs(lat[0] - lat[1]))
    return from_origin(
        float(lon[0]) - res_x / 2.0,
        float(lat[0]) + res_y / 2.0,
        res_x,
        res_y,
    )


# --------------------------------------------------------------------------
# Stage 1: acquisition
# --------------------------------------------------------------------------

def _verify_csv(path):
    """Open and parse the file. A truncated download still 'exists' on disk."""
    df = pd.read_csv(path, nrows=5)
    if df.empty and path.stat().st_size < 200:
        raise ValueError(f"{path} parsed but is effectively empty")
    return True


def _verify_geojson(path):
    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)
    if not isinstance(obj, dict):
        raise ValueError(f"{path}: expected a JSON object")
    feats = obj.get("features")
    if feats is None and isinstance(obj.get("results"), dict):
        feats = obj["results"].get("features")
    if not feats:
        raise ValueError(f"{path}: no features (top-level keys: {sorted(obj)})")
    if not feats[0].get("geometry"):
        raise ValueError(f"{path}: first feature has no geometry")
    if obj.get("next"):
        # Not fatal - the file is valid, just partial. Flag it loudly here so it
        # is caught at download time rather than after rasterizing a subset.
        print(f"  WARNING: {Path(path).name} is paginated and incomplete "
              f"({len(feats)} features on this page)")
    return True


def download_with_cache(url, out_path, verifier, max_retries=3, force=False):
    """Download once, cache locally, verify before trusting the cache.

    The verify-before-trust step is not optional: a failed download leaves a
    truncated file that still passes os.path.exists(), and a later run will
    silently treat it as complete. This exact bug bit the CHIRPS pipeline.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        try:
            verifier(out_path)
            print(f"  cached: {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
            return out_path
        except Exception as exc:
            print(f"  cache corrupt ({exc}); re-downloading {out_path.name}")
            out_path.unlink()

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  GET {url}")
            with requests.get(url, timeout=REQUEST_TIMEOUT, stream=True) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        fh.write(chunk)
            verifier(out_path)
            print(f"  saved:  {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
            return out_path
        except Exception as exc:
            if out_path.exists():
                out_path.unlink()  # never leave a partial file behind
            if attempt == max_retries:
                raise
            print(f"  attempt {attempt} failed ({exc}); retrying")

    raise RuntimeError("unreachable")


def _country_query(countries):
    return "&".join(f"country_code={c}" for c in countries)


def probe_record_count(scenario, countries):
    """Ask the JSON endpoint how many records exist, to verify the CSV extract.

    The CSV extract is a single bulk request with no built-in completeness
    signal. Comparing its row count against the JSON endpoint's `count` field
    is a cheap guard against a silently truncated response.
    """
    url = (
        f"{BASE_URL}/ipcphase.json?{_country_query(countries)}"
        f"&scenario={scenario}&page_size=1&offset=0"
    )
    try:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        return int(resp.json().get("count", -1))
    except Exception as exc:
        print(f"  (count probe failed for {scenario}: {exc})")
        return -1


def fetch_ipcphase(countries, scenarios=SCENARIOS, force=False):
    """Stage 1a: pull the IPC phase time series for each scenario."""
    print("\n[1a] IPC phase records")
    paths = {}
    for scenario in scenarios:
        url = (
            f"{BASE_URL}/ipcphase.csv?{_country_query(countries)}"
            f"&scenario={scenario}"
        )
        out = RAW_DIR / f"ipcphase_{scenario}_{'-'.join(countries)}.csv"
        paths[scenario] = download_with_cache(url, out, _verify_csv, force=force)

        expected = probe_record_count(scenario, countries)
        # Count with the CSV parser, NOT by counting newlines: several fields
        # (description, data_usage_policy) contain embedded newlines, so a line
        # count overstates the row count and produces a spurious "truncated"
        # warning.
        actual = len(pd.read_csv(paths[scenario], usecols=[0], low_memory=False))
        if expected >= 0:
            status = "OK" if actual == expected else "MISMATCH -- extract may be truncated"
            print(f"  {scenario}: {actual} rows vs API count {expected}  [{status}]")
        else:
            print(f"  {scenario}: {actual} rows (count unverified)")
    return paths


def fetch_features(countries, force=False):
    """Stage 1b: pull the fsc_admin geometries.

    Fetched WITHOUT as_of_date so that every boundary vintage comes back in one
    request; the join to phase records is on fnid, which is vintage-specific,
    so per-date geometry requests are unnecessary. This is the same
    vintage-overlap situation documented in the CHIRPS work - the difference is
    that here it is confined to the target variable and cannot contaminate the
    feature pipeline.
    """
    print("\n[1b] fsc_admin geometries")
    unit_query = "&".join(f"unit_type={u}" for u in UNIT_TYPES)
    url = f"{BASE_URL}/feature.geojson?{_country_query(countries)}&{unit_query}"
    out = RAW_DIR / f"features_{'-'.join(countries)}.geojson"
    return download_with_cache(url, out, _verify_geojson, force=force)


def fetch_ha(countries):
    """NOT IMPLEMENTED - humanitarian food assistance layer.

    Busker's CS file carries an HA variable (0/1/2, present from 2012) which he
    aggregates with the same population weighting as the phase data. The FDW
    ipcphase endpoint does not appear to expose it.

    To resolve, in order of cost:
      1. Inspect the ipcphase CSV columns after `fetch` - HA may ride along as
         an extra field rather than a separate scenario.
      2. Check https://fdw.fews.net/api/schema/swagger-ui/ for an HA-specific
         endpoint or filter.
      3. Fall back to the per-period shapefile bundles at
         https://fews.net/data/acute-food-insecurity, which contain an HA layer
         alongside CS/ML1/ML2.

    Left unimplemented deliberately: a guessed endpoint that silently returns
    an empty frame is worse than an explicit gap.
    """
    raise NotImplementedError(fetch_ha.__doc__)


# --------------------------------------------------------------------------
# Stage 2a: tidy
# --------------------------------------------------------------------------

def resolve_column(df, candidates, label):
    """Find the first matching column name, case-insensitively.

    The FDW CSV schema was not directly inspectable when this script was
    written, so column names are resolved defensively rather than hard-coded.
    On failure this prints the actual columns, which is the fastest possible
    path to a fix.
    """
    lookup = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lookup:
            return lookup[cand.lower()]
    raise KeyError(
        f"Could not find a column for '{label}'.\n"
        f"  tried: {candidates}\n"
        f"  available: {sorted(df.columns)}\n"
        f"Edit the candidate list in resolve_column() calls to match."
    )


def tidy_scenario(csv_path, scenario):
    """Normalise one scenario's raw extract into a long panel."""
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"\n  {scenario}: {len(df)} raw rows, {len(df.columns)} columns")

    fnid_col = resolve_column(df, ["fnid", "geographic_unit_fnid", "unit_fnid"], "fnid")
    value_col = resolve_column(df, ["value", "ipc_phase", "phase"], "phase value")

    # Confirmed FDW schema: the target period is projection_start/projection_end
    # (used for CS as well as ML1/ML2), and the date the analysis was published
    # is reporting_date. Without reporting_date, lead time cannot be derived and
    # competing vintages for the same period cannot be ordered.
    start_col = resolve_column(
        df, ["projection_start", "start_date", "period_start", "period_date"], "period start"
    )
    try:
        collect_col = resolve_column(
            df, ["reporting_date", "collection_date", "collected_date", "document_date"],
            "collection date",
        )
    except KeyError:
        collect_col = None
        print("    WARNING: no reporting_date column; lead times cannot be derived")

    keep = {fnid_col: "fnid", value_col: "phase_raw", start_col: "period_start"}
    if collect_col:
        keep[collect_col] = "collection_date"

    # Discriminator columns. These decide which of several records for the same
    # (fnid, period) is the one to rasterize, so they must survive into the panel
    # rather than being dropped here.
    for extra in [
        "projection_end", "country_code", "unit_type", "geographic_unit_name",
        "classification_scale", "is_allowing_for_assistance", "scenario_name",
        "collection_schedule", "status", "pct_phase3", "pct_phase4", "pct_phase5",
        # Provenance fields. If FDW has revised an analysis since Busker
        # downloaded his shapefiles, or if two documents cover the same period,
        # these are the only columns that can show it.
        "datasourcedocument", "source_document", "dataseries", "created",
        "modified", "preference_rating", "datacollection",
    ]:
        if extra in df.columns:
            keep[extra] = extra

    out = df[list(keep)].rename(columns=keep)
    out["scenario"] = scenario

    out["period_start"] = pd.to_datetime(out["period_start"], errors="coerce")
    if "projection_end" in out.columns:
        out["projection_end"] = pd.to_datetime(out["projection_end"], errors="coerce")
    if "collection_date" in out.columns:
        out["collection_date"] = pd.to_datetime(out["collection_date"], errors="coerce")
        # Lead measured to the START of the projection window understates ML1:
        # an Outlook published in month M sets ML1 = M..M+3, so its start-lead is
        # always 0 while its actual horizon is the window END. Report both.
        out["lead_start"] = (
            (out["period_start"].dt.year - out["collection_date"].dt.year) * 12
            + (out["period_start"].dt.month - out["collection_date"].dt.month)
        )
        if "projection_end" in out.columns:
            out["lead_end"] = (
                (out["projection_end"].dt.year - out["collection_date"].dt.year) * 12
                + (out["projection_end"].dt.month - out["collection_date"].dt.month)
            )

    # Report the structure of the discriminators BEFORE any filtering, so the
    # choice of filter is made against what the data actually contains.
    for col in ["classification_scale", "is_allowing_for_assistance", "unit_type",
                "scenario_name", "status"]:
        if col in out.columns:
            counts = out[col].value_counts(dropna=False).head(6)
            rendered = ", ".join(f"{k}={v}" for k, v in counts.items())
            print(f"    {col}: {rendered}")

    # Mask sentinels. Log first, mask second - never the other way round.
    seen = sorted(pd.unique(out["phase_raw"].dropna()))
    print(f"    distinct raw values: {seen}")
    unexpected = set(seen) - VALID_PHASES - NON_PHASE_CODES
    if unexpected:
        print(f"    WARNING: unrecognised codes {sorted(unexpected)} -- "
              f"check FEWS NET encoding before trusting these as phases")

    out["phase"] = out["phase_raw"].where(out["phase_raw"].isin(list(VALID_PHASES)))
    n_masked = int(out["phase"].isna().sum() - out["phase_raw"].isna().sum())
    print(f"    masked {n_masked} non-phase records ({n_masked/max(len(out),1):.2%})")

    # How much collision is there per (fnid, period)? If this is large, a filter
    # is required before rasterizing rather than a blind keep-last.
    dup = int(out.duplicated(subset=["fnid", "period_start"]).sum())
    print(f"    records sharing an (fnid, period) key: {dup} "
          f"({dup/max(len(out),1):.1%}) -- resolved at rasterize time")

    return out


def tidy(scenarios=SCENARIOS, countries=None):
    """Stage 2a: raw CSVs -> one tidy long panel."""
    print("\n[2a] Tidying raw extracts")
    countries = countries or HOA_COUNTRIES
    frames = []
    for scenario in scenarios:
        path = RAW_DIR / f"ipcphase_{scenario}_{'-'.join(countries)}.csv"
        if not path.exists():
            print(f"  skipping {scenario}: {path.name} not found (run `fetch` first)")
            continue
        frames.append(tidy_scenario(path, scenario))

    if not frames:
        raise SystemExit("No raw extracts found. Run `fetch` first.")

    panel = pd.concat(frames, ignore_index=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INTERIM_DIR / "ipc_panel_long.csv"
    panel.to_csv(out_path, index=False)

    print(f"\n  wrote {out_path} ({len(panel)} rows)")
    print("\n  coverage by scenario:")
    for scenario, grp in panel.groupby("scenario"):
        dates = grp["period_start"].dropna()
        print(f"    {scenario}: {len(grp):>7} rows, "
              f"{dates.min():%Y-%m} .. {dates.max():%Y-%m}, "
              f"{grp['fnid'].nunique()} units, "
              f"{dates.dt.to_period('M').nunique()} distinct months")
        for label in ["lead_start", "lead_end"]:
            if label in grp.columns:
                lead = grp[label].dropna()
                if len(lead):
                    print(f"      {label} (months): min {lead.min():.0f}, "
                          f"median {lead.median():.0f}, max {lead.max():.0f}")
    return panel


# --------------------------------------------------------------------------
# Stage 2b: rasterize
# --------------------------------------------------------------------------

def load_features(path):
    """Load an FDW feature extract into a GeoDataFrame.

    Deliberately does NOT use gpd.read_file(). The FDW endpoint paginates, so
    the response is a FeatureCollection carrying extra top-level keys, or one
    wrapped in a DRF envelope ({"count":..., "next":..., "results": {...}}).
    Fiona's GeoJSON driver reads zero features from that and hands back an empty
    frame with only a geometry column -- which looks like an empty download
    rather than a parsing problem, and fails much later than it should.

    Parsing the JSON directly also surfaces `next`, which is the only signal
    that the extract is incomplete.
    """
    import geopandas as gpd
    from shapely.geometry import shape

    with open(path, "r", encoding="utf-8") as fh:
        obj = json.load(fh)

    if isinstance(obj, dict):
        print(f"    top-level keys: {sorted(obj.keys())}")
        if obj.get("next"):
            print(f"    WARNING: `next` is set -- this extract is INCOMPLETE.\n"
                  f"      continue from: {obj['next']}")
        if "features" not in obj and isinstance(obj.get("results"), dict):
            obj = obj["results"]
        elif "features" not in obj and isinstance(obj.get("results"), list):
            obj = {"features": obj["results"]}

    feats = obj.get("features", []) if isinstance(obj, dict) else obj
    if not feats:
        raise SystemExit(
            f"No features parsed from {path}. Inspect its actual structure with:\n"
            f"  python -c \"import json; d=json.load(open('{path}')); "
            f"print(type(d), list(d)[:12] if isinstance(d, dict) else len(d))\""
        )

    records, geoms = [], []
    for ft in feats:
        props = dict(ft.get("properties") or {})
        # Some endpoints carry the identifier on the feature rather than inside
        # properties; keep both routes open.
        if "fnid" not in props and ft.get("id") is not None:
            props.setdefault("fnid", ft["id"])
        geom = ft.get("geometry")
        records.append(props)
        geoms.append(shape(geom) if geom else None)

    gdf = gpd.GeoDataFrame(records, geometry=geoms, crs="EPSG:4326")
    print(f"    parsed {len(gdf)} features; fields: {sorted(c for c in gdf.columns)[:12]}")

    n_null = int(gdf.geometry.isna().sum())
    if n_null:
        print(f"    dropping {n_null} features with null geometry")
        gdf = gdf[gdf.geometry.notna()].copy()

    return gdf


def prepare_panel(time_key="collection", classification_scale=None,
                  unit_types=("fsc_admin", "fsc_admin_lhz"), verbose=True):
    """Load the tidy panel and apply the standard filters. Shared by rasterize
    and diagnose so both always see exactly the same records."""
    panel_path = INTERIM_DIR / "ipc_panel_long.csv"
    if not panel_path.exists():
        raise SystemExit("Run `tidy` first.")
    panel = pd.read_csv(panel_path, low_memory=False,
                        parse_dates=["period_start", "collection_date", "projection_end"])

    time_col = {"period": "period_start", "collection": "collection_date"}[time_key]
    if time_col not in panel.columns:
        raise SystemExit(f"time_key={time_key!r} needs column {time_col!r}, "
                         f"which is not in the panel. Re-run `tidy`.")
    if verbose:
        print(f"  time axis: {time_col} (time_key={time_key})")

    if unit_types and "unit_type" in panel.columns:
        before = len(panel)
        dropped = panel.loc[~panel["unit_type"].isin(unit_types), "unit_type"]
        panel = panel[panel["unit_type"].isin(unit_types)]
        if before != len(panel) and verbose:
            print(f"  unit_type in {list(unit_types)}: {before} -> {len(panel)} rows "
                  f"(dropped {dropped.value_counts().to_dict()})")

    if "classification_scale" in panel.columns:
        if classification_scale:
            before = len(panel)
            panel = panel[panel["classification_scale"].astype(str) == classification_scale]
            if verbose:
                print(f"  classification_scale == {classification_scale!r}: "
                      f"{before} -> {len(panel)} rows")
        else:
            household = panel["classification_scale"].astype(str).str.contains(
                "Household", case=False, na=False
            )
            if household.any():
                if verbose:
                    print(f"  dropping {int(household.sum())} household-scale records "
                          f"(area-level classification only)")
                panel = panel[~household]

    return panel, time_col


def load_geometry_gdf(verbose=True):
    """Load, de-duplicate and repair the fsc_admin geometries."""
    geo_paths = sorted(RAW_DIR.glob("features_*.geojson"))
    if not geo_paths:
        raise SystemExit("No geometry file found. Run `fetch` first.")
    if verbose:
        print(f"  reading geometries from {geo_paths[0].name}")
    gdf = load_features(geo_paths[0])

    fnid_col = resolve_column(gdf, ["fnid", "FNID", "id", "unit_fnid"], "fnid (geometry)")
    cols = [fnid_col, "geometry"]
    # admin_1 lets `diagnose` test whether disagreement is regionally clustered
    # (e.g. peripheral lowland zones) rather than scattered.
    for extra in ["admin_1", "admin_2", "ar_name"]:
        if extra in gdf.columns:
            cols.append(extra)
    gdf = gdf[cols].rename(columns={fnid_col: "fnid"})
    gdf = gdf.drop_duplicates(subset="fnid")
    if verbose:
        print(f"  {len(gdf)} unique fnids with geometry")

    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        if verbose:
            print(f"  repairing {invalid} invalid geometries with buffer(0)")
        gdf.loc[~gdf.geometry.is_valid, "geometry"] = gdf.loc[
            ~gdf.geometry.is_valid, "geometry"
        ].buffer(0)

    return gdf


def rasterize(extent="hoa", like=None, scenarios=SCENARIOS, out_name=None,
              classification_scale=None, unit_types=("fsc_admin", "fsc_admin_lhz"),
              vintage="latest", emit_ha=True, time_key="period"):
    """Stage 2b: burn IPC phases onto the CHIRPS grid, one raster per period.

    Output is a NetCDF with the same (time, lat, lon) layout as Busker's files,
    so `validate` can subtract them directly.

    `time_key` decides what the time axis MEANS, and the two options are not
    interchangeable:

    * "collection" -- index by reporting_date, i.e. when the Outlook was
      published. This is what Busker does: his ML1 and ML2 files share an
      identical time axis (51 steps, 2019-02..2023-06), which is only possible
      under publication-date indexing, since the two scenarios' target windows
      never coincide. Use this to reproduce his files.
    * "period" -- index by projection_start, i.e. the period being forecast.
      Use this for benchmarking, where each projection must be paired with the
      CS that later verified it.

    For CS and ML1 the two are equivalent (projection_start == reporting_date,
    so lead_start is 0). They diverge only for ML2, whose window opens 1-4
    months after publication.

    Other filtering, and why each filter exists:

    * `unit_types` -- the extract also contains `admin0` (whole-country) and
      `idp_camp` (point) records. An admin0 polygon burned onto the grid paints
      a single phase across the entire country and silently overwrites every
      zone in that timestep. This filter is NOT optional.
    * `classification_scale` -- "IPC Highest Household" is a household-level
      measure, not the area-level classification Busker uses (highest phase
      faced by >=20% of the population). Excluded by default. The IPC 2.0 / 3.0
      / 3.1 values are successive protocol versions that do not overlap in time,
      so all three are kept.
    * `is_allowing_for_assistance` is deliberately NOT filtered on. Each
      (fnid, period) appears exactly once, so the flag is an attribute of that
      single record rather than a competing variant; filtering would punch holes
      in the target. It is emitted as a separate HA layer instead (`emit_ha`),
      which is the analogue of Busker's HA variable.
    * `vintage` -- under period indexing, overlapping ML2 windows mean one
      target period is projected several times at different leads. "latest"
      keeps the shortest-lead (most informed) projection; "earliest" keeps the
      longest-lead one, which is the harder and more operationally honest test.
      Under collection indexing this is mostly moot, since one publication
      issues one projection per unit.
    """
    import geopandas as gpd
    import xarray as xr
    from rasterio.features import rasterize as rio_rasterize

    print("\n[2b] Rasterizing onto the CHIRPS grid")

    panel, time_col = prepare_panel(time_key, classification_scale, unit_types)
    gdf = load_geometry_gdf()

    if like:
        lon, lat = grid_from_reference(like)
        print(f"  grid copied from {Path(like).name}: {len(lon)} x {len(lat)}")
    else:
        lon, lat = chirps_grid(EXTENTS[extent])
        print(f"  grid snapped to CHIRPS p05 [{extent}]: {len(lon)} x {len(lat)}")
    print(f"    lon {lon[0]:.4f} .. {lon[-1]:.4f} | lat {lat[0]:.4f} .. {lat[-1]:.4f}")

    transform = transform_from_grid(lon, lat)
    shape = (len(lat), len(lon))
    FILL = -999

    datasets = {}
    for scenario in scenarios:
        sub = panel[(panel["scenario"] == scenario) & panel["phase"].notna()]
        if sub.empty:
            print(f"  {scenario}: no records, skipping")
            continue

        periods = sorted(sub[time_col].dropna().unique())
        stack = np.full((len(periods), *shape), np.nan, dtype="float32")

        has_ha = emit_ha and "is_allowing_for_assistance" in sub.columns
        ha_stack = (np.full((len(periods), *shape), np.nan, dtype="float32")
                    if has_ha else None)

        geo_fnids = set(gdf["fnid"])
        panel_fnids = set(sub["fnid"])
        unmatched = panel_fnids - geo_fnids
        if unmatched:
            print(f"  {scenario}: {len(unmatched)} of {len(panel_fnids)} fnids have no "
                  f"geometry -- e.g. {sorted(unmatched)[:3]}")

        collisions = 0
        for t, period in enumerate(periods):
            slice_ = sub[sub[time_col] == period]

            # Overlapping projection windows mean one target period can be
            # projected several times at different leads; pick one vintage.
            if "collection_date" in slice_.columns:
                n_before = len(slice_)
                slice_ = (
                    slice_.sort_values("collection_date")
                    .drop_duplicates(subset="fnid",
                                     keep="last" if vintage == "latest" else "first")
                )
                collisions += n_before - len(slice_)

            cols = ["fnid", "phase"] + (["is_allowing_for_assistance"] if has_ha else [])
            merged = gdf.merge(slice_[cols], on="fnid", how="inner")
            if merged.empty:
                print(f"    {scenario} {pd.Timestamp(period):%Y-%m}: no fnid matches -- skipped")
                continue

            burned = rio_rasterize(
                ((geom, int(val)) for geom, val in zip(merged.geometry, merged["phase"])),
                out_shape=shape,
                transform=transform,
                fill=FILL,
                all_touched=False,
                dtype="int16",
            )
            arr = burned.astype("float32")
            arr[burned == FILL] = np.nan
            stack[t] = arr

            if has_ha:
                flag = merged["is_allowing_for_assistance"].astype(str).str.lower().isin(
                    ["true", "1", "yes"]
                ).astype(int)
                # Burn 1/2 rather than 0/1 so that "no assistance" is a real
                # value and only outside-coverage becomes NaN; shifted back after.
                ha_burn = rio_rasterize(
                    ((geom, int(v) + 1) for geom, v in zip(merged.geometry, flag)),
                    out_shape=shape,
                    transform=transform,
                    fill=FILL,
                    all_touched=False,
                    dtype="int16",
                )
                ha_arr = ha_burn.astype("float32") - 1.0
                ha_arr[ha_burn == FILL] = np.nan
                ha_stack[t] = ha_arr

        da = xr.DataArray(
            stack,
            dims=("time", "lat", "lon"),
            coords={"time": pd.to_datetime(periods), "lat": lat, "lon": lon},
            name=scenario,
        )
        ds_out = da.to_dataset()

        if has_ha:
            ds_out["HA"] = xr.DataArray(
                ha_stack,
                dims=("time", "lat", "lon"),
                coords={"time": pd.to_datetime(periods), "lat": lat, "lon": lon},
            )
            ha_rate = np.nanmean(ha_stack) if np.isfinite(ha_stack).any() else float("nan")
            print(f"  {scenario}: HA layer emitted, "
                  f"{ha_rate:.1%} of classified cells flagged assistance-inclusive")

        datasets[scenario] = ds_out

        covered = np.isfinite(stack).mean()
        print(f"  {scenario}: {len(periods)} timesteps, {covered:.1%} of cells classified, "
              f"{collisions} duplicate (fnid, period) records resolved "
              f"by {vintage} reporting_date")

    if not datasets:
        raise SystemExit("Nothing rasterized.")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for scenario, ds_out in datasets.items():
        name = out_name or f"ipc_grid_{scenario}.nc"
        if out_name:
            name = out_name.replace("{scenario}", scenario)
        path = INTERIM_DIR / name
        ds_out.to_netcdf(path)
        written.append(path)
        print(f"  wrote {path}")
    return written


# --------------------------------------------------------------------------
# Stage 4 (prep): inspect and validate against Busker
# --------------------------------------------------------------------------

def inspect_busker(busker_dir=None):
    """Report the structure of Busker's released rasters.

    Run this before anything else. It defines the target format and confirms
    the grid the rest of the pipeline must reproduce.
    """
    import xarray as xr

    busker_dir = Path(busker_dir or BUSKER_DIR)
    print(f"\n[0] Busker reference files in {busker_dir}")

    for scenario in SCENARIOS:
        path = busker_dir / f"fews_xr_{scenario}.nc"
        if not path.exists():
            print(f"  {path.name}: NOT FOUND")
            continue

        with xr.open_dataset(path) as ds:
            print(f"\n  {path.name}")
            print(f"    dims: {dict(ds.sizes)}")
            print(f"    vars: {list(ds.data_vars)}")

            times = pd.to_datetime(ds["time"].values)
            print(f"    time: {times[0]:%Y-%m-%d} .. {times[-1]:%Y-%m-%d} ({len(times)} steps)")
            gaps = times.to_series().diff().dt.days.dropna()
            if len(gaps):
                print(f"    step spacing (days): min {gaps.min():.0f}, "
                      f"median {gaps.median():.0f}, max {gaps.max():.0f}")

            lon, lat = ds["lon"].values, ds["lat"].values
            i = (lon[0] + 180 - 0.025) / CHIRPS_RES
            j = (lat[0] + 50 - 0.025) / CHIRPS_RES
            on_grid = abs(i - round(i)) < 1e-4 and abs(j - round(j)) < 1e-4
            print(f"    grid: lon {lon[0]:.4f}..{lon[-1]:.4f}, lat {lat[0]:.4f}..{lat[-1]:.4f}")
            print(f"    on CHIRPS p05 grid: {on_grid}  (lon idx {i:.1f}, lat idx {j:.1f})")

            for var in ds.data_vars:
                arr = ds[var].values
                finite = arr[np.isfinite(arr)]
                vals = np.unique(finite)
                print(f"    {var}: values {vals.tolist()}, "
                      f"{np.isnan(arr).mean():.1%} NaN")
                odd = set(vals.tolist()) - VALID_PHASES
                if var == scenario and odd:
                    per_time = [
                        (f"{pd.Timestamp(t):%Y-%m}", int(np.isin(a, list(odd)).sum()))
                        for t, a in zip(ds["time"].values, arr)
                        if np.isin(a, list(odd)).any()
                    ]
                    print(f"      non-phase codes {sorted(odd)} at: {per_time}")


def diagnose(scenario="CS", times=None, busker_dir=None, time_key="collection",
             classification_scale=None, unit_types=("fsc_admin", "fsc_admin_lhz"),
             like=None, extent="hoa", top=12):
    """Attribute disagreement with Busker to specific zones and phase pairs.

    `validate` tells you a timestep is wrong; this tells you which polygons are
    wrong, which way, and what distinguishes them. It re-burns an fnid-index
    raster for the timestep so every disagreeing cell can be traced back to the
    record that produced it, then reports the attributes of those records
    (classification_scale, assistance flag, vintage dates) so a hypothesis can
    be tested rather than assumed.
    """
    import xarray as xr
    from rasterio.features import rasterize as rio_rasterize

    busker_dir = Path(busker_dir or BUSKER_DIR)
    print(f"\n[D] Diagnosing {scenario} against Busker")

    ref_path = busker_dir / f"fews_xr_{scenario}.nc"
    own_path = INTERIM_DIR / f"ipc_grid_{scenario}.nc"
    if not ref_path.exists() or not own_path.exists():
        raise SystemExit(f"Need both {ref_path} and {own_path}.")

    ref = xr.open_dataset(ref_path)[scenario]
    own = xr.open_dataset(own_path)[scenario]

    shared = sorted(set(pd.to_datetime(ref.time.values)) & set(pd.to_datetime(own.time.values)))
    if times:
        wanted = {pd.Timestamp(t) for t in times}
        shared = [t for t in shared if t in wanted or t.strftime("%Y-%m") in
                  {pd.Timestamp(w).strftime("%Y-%m") for w in wanted}]
    if not shared:
        raise SystemExit("No matching timesteps.")

    panel, time_col = prepare_panel(time_key, classification_scale, unit_types, verbose=False)
    panel = panel[(panel["scenario"] == scenario) & panel["phase"].notna()]
    gdf = load_geometry_gdf(verbose=False)

    if like:
        lon, lat = grid_from_reference(like)
    else:
        lon, lat = grid_from_reference(ref_path)
    transform = transform_from_grid(lon, lat)
    shape = (len(lat), len(lon))

    for t in shared:
        a = ref.sel(time=t).values
        b = own.sel(time=t).values
        mask = np.isfinite(a) & np.isfinite(b)
        if not mask.any():
            continue
        bad = mask & (a != b)
        print(f"\n  {pd.Timestamp(t):%Y-%m}: {int(mask.sum()):,} co-located cells, "
              f"{int(bad.sum()):,} disagree ({bad.sum()/mask.sum():.2%})")
        if not bad.any():
            continue

        # Phase confusion: which substitutions are happening?
        conf = pd.crosstab(a[bad].astype(int), b[bad].astype(int),
                           rownames=["busker"], colnames=["ours"])
        print("    phase confusion (disagreeing cells only):")
        for line in conf.to_string().splitlines():
            print(f"      {line}")

        # Burn an fnid index so disagreeing cells can be traced to records.
        slice_ = panel[panel[time_col] == t]
        if "collection_date" in slice_.columns:
            slice_ = (slice_.sort_values("collection_date")
                            .drop_duplicates(subset="fnid", keep="last"))
        merged = gdf.merge(slice_, on="fnid", how="inner").reset_index(drop=True)
        if merged.empty:
            print("    (no records for this timestep; cannot attribute)")
            continue

        idx = rio_rasterize(
            ((geom, i + 1) for i, geom in enumerate(merged.geometry)),
            out_shape=shape, transform=transform, fill=0,
            all_touched=False, dtype="int32",
        )
        hits = idx[bad]
        counts = pd.Series(hits[hits > 0]).value_counts().head(top)
        n_orphan = int((hits == 0).sum())

        # Zone-level table: is disagreement whole-zone (value substitution) or
        # partial (geometry misalignment)? These have completely different causes.
        name_col = ("geographic_unit_name" if "geographic_unit_name" in merged.columns
                    else "fnid")
        zone_rows = []
        for i in range(1, len(merged) + 1):
            cells = idx == i
            if not cells.any():
                continue
            ours_z, ref_z = b[cells], a[cells]
            m = np.isfinite(ours_z) & np.isfinite(ref_z)
            if not m.any():
                continue
            zone_rows.append({
                "i": i,
                "fnid": merged.iloc[i - 1]["fnid"],
                "name": merged.iloc[i - 1][name_col],
                "ours": float(pd.Series(ours_z[m]).mode().iloc[0]),
                "busker": float(pd.Series(ref_z[m]).mode().iloc[0]),
                "frac_bad": float((ours_z[m] != ref_z[m]).mean()),
                "n_cells": int(m.sum()),
            })
        zt = pd.DataFrame(zone_rows)

        if not zt.empty:
            disagreeing = zt[zt["frac_bad"] > 0]
            whole = int((disagreeing["frac_bad"] > 0.99).sum())
            print(f"    zones: {len(zt)} compared, {len(disagreeing)} disagree, "
                  f"{whole} of those disagree on >99% of their cells")
            if len(disagreeing):
                print(f"      -> {'whole-zone value substitution' if whole/len(disagreeing) > 0.8 else 'partial/edge disagreement (geometry)'}")

        print(f"    top zones by disagreeing cells "
              f"({n_orphan:,} cells fall outside any of our polygons):")
        for i, n in counts.items():
            row = merged.iloc[i - 1]
            zone_cells = int((idx == i).sum())
            attrs = []
            for col in ["classification_scale", "is_allowing_for_assistance",
                        "unit_type", "geographic_unit_name"]:
                if col in merged.columns:
                    attrs.append(f"{col}={row[col]}")
            print(f"      {row['fnid']}: {n:,}/{zone_cells:,} cells disagree, "
                  f"our phase {row['phase']:.0f} | {', '.join(attrs)}")

        # Ordering-shift test. If one dataset's values were joined to zones via a
        # sorted list with an off-by-k error, then Busker's value at position p
        # equals OUR value at position p+k under that ordering. Recompute against
        # the alternative rather than inferring it -- if agreement genuinely
        # improves under a shift, the ordering is the cause.
        if len(zt) > 10:
            print("    ordering-shift test (share of zones matching under shift k):")
            for label, key in [("by name", "name"), ("by fnid", "fnid"),
                               ("by file order", "i")]:
                s = zt.sort_values(key).reset_index(drop=True)
                base = float((s["busker"] == s["ours"]).mean())
                shifts = {
                    k: float((s["busker"] == s["ours"].shift(-k)).mean())
                    for k in (-2, -1, 1, 2)
                }
                best_k, best_v = max(shifts.items(), key=lambda kv: kv[1])
                flag = "  <-- BETTER THAN UNSHIFTED" if best_v > base else ""
                rendered = ", ".join(f"k={k}: {v:.1%}" for k, v in shifts.items())
                print(f"      {label:14s} k=0: {base:.1%} | {rendered}{flag}")

        # What actually distinguishes the disagreeing zones? Rather than testing
        # one hypothesis at a time, scan every available metadata column and
        # report where the disagreeing group's distribution departs from the
        # agreeing group's. A column that separates the two is the mechanism.
        if not zt.empty and len(zt) > 10:
            zt = zt.assign(bad=zt["frac_bad"] > 0.5)
            meta = merged.set_index(merged.index + 1)
            candidates = [c for c in [
                "classification_scale", "is_allowing_for_assistance", "status",
                "collection_schedule", "datasourcedocument", "source_document",
                "dataseries", "created", "modified", "preference_rating",
                "admin_1", "unit_type",
            ] if c in meta.columns]

            print("    discriminator scan (disagreeing vs agreeing zones):")
            zt_meta = zt.join(meta[candidates], on="i", rsuffix="_m")
            if not candidates:
                print(f"      no metadata columns available. merged has: "
                      f"{sorted(merged.columns)}")
            for col in candidates:
                vals = zt_meta[col]
                n_uniq = vals.nunique(dropna=False)
                bad_vals, ok_vals = vals[zt_meta["bad"]], vals[~zt_meta["bad"]]

                if n_uniq < 2:
                    print(f"      {col:22s} constant ({vals.iloc[0]!r}) -- cannot discriminate")
                    continue

                if n_uniq > 40:
                    # High-cardinality (document ids, timestamps): a distribution
                    # comparison is meaningless, but a set comparison is not. If
                    # the disagreeing zones come from documents the agreeing ones
                    # never use, that is the mechanism.
                    bad_set, ok_set = set(bad_vals.dropna()), set(ok_vals.dropna())
                    overlap = len(bad_set & ok_set) / max(len(bad_set | ok_set), 1)
                    only_bad = len(bad_set - ok_set)
                    marker = "  <-- DISJOINT SETS" if overlap < 0.2 else ""
                    print(f"      {col:22s} {n_uniq} distinct | disagreeing use "
                          f"{len(bad_set)} values, {only_bad} of them unused by "
                          f"agreeing zones (Jaccard {overlap:.2f}){marker}")
                    continue

                bad_dist = bad_vals.value_counts(normalize=True, dropna=False)
                ok_dist = ok_vals.value_counts(normalize=True, dropna=False)
                gaps = (bad_dist - ok_dist.reindex(bad_dist.index).fillna(0)).abs()
                if gaps.empty:
                    print(f"      {col:22s} no disagreeing zones to compare")
                    continue
                worst = gaps.idxmax()
                sep = float(gaps.max())
                marker = "  <-- SEPARATES" if sep > 0.35 else ""
                print(f"      {col:22s} max gap {sep:.0%} on {worst!r} "
                      f"(bad {bad_dist.get(worst, 0):.0%} vs ok {ok_dist.get(worst, 0):.0%})"
                      f"{marker}")

            # Zone size: are the disagreeing zones systematically the large,
            # sparsely-gridded peripheral ones?
            big_bad = zt.loc[zt["bad"], "n_cells"].median()
            big_ok = zt.loc[~zt["bad"], "n_cells"].median()
            print(f"      median zone size (cells): disagreeing {big_bad:.0f} "
                  f"vs agreeing {big_ok:.0f}")



def audit(scenario="CS", time_key="collection", classification_scale=None,
          unit_types=("fsc_admin", "fsc_admin_lhz")):
    """Audit the target variable for the defects that actually damage a model.

    Agreement with Busker answers whether the extraction is faithful. It does
    NOT answer whether the target is fit to train on. These are different
    questions with different failure modes, and the ones below are the ones that
    silently degrade results:

      1. Temporal density -- the target is assessed a few times a year while
         features are monthly. Whatever fills the gaps IS a modelling decision.
      2. Class balance -- if crisis phases are rare, weighted F1 and the
         decision threshold matter far more than architecture choice.
      3. Spatial coverage -- zones that appear and disappear produce ragged
         panels and biased per-zone evaluation.
      4. Vintage discontinuity -- fnids are vintage-specific. A lagged IPC
         feature (Busker's single strongest predictor) breaks silently across a
         boundary redraw, because the "same place" has a new identifier.
      5. Implausible transitions -- large phase jumps between consecutive
         assessments indicate either genuine shocks or data problems.
    """
    print(f"\n[A] Target-variable audit: {scenario}")

    panel, time_col = prepare_panel(time_key, classification_scale, unit_types)
    sub = panel[(panel["scenario"] == scenario) & panel["phase"].notna()].copy()
    if sub.empty:
        raise SystemExit(f"No {scenario} records. Run `tidy` first.")
    sub["month"] = sub[time_col].dt.to_period("M")

    # --- 1. Temporal density ------------------------------------------------
    months = sorted(sub["month"].dropna().unique())
    print(f"\n  1. TEMPORAL DENSITY: {len(months)} assessed months, "
          f"{months[0]} .. {months[-1]}")
    per_year = sub.groupby(sub[time_col].dt.year)["month"].nunique()
    print("     assessments per year: " + ", ".join(
        f"{y}:{n}" for y, n in per_year.items()))

    gaps = []
    for a, b in zip(months[:-1], months[1:]):
        n = (b - a).n
        if n > 4:
            gaps.append((str(a), str(b), n))
    if gaps:
        print(f"     gaps longer than 4 months ({len(gaps)}):")
        for a, b, n in gaps:
            print(f"       {a} -> {b} ({n} months)")
    else:
        print("     no gaps longer than 4 months")
    span = (months[-1] - months[0]).n + 1
    print(f"     -> {len(months)}/{span} months assessed ({len(months)/span:.0%}). "
          f"The other {span - len(months)} months must be filled by an explicit "
          f"rule; forward-fill implies the phase held constant.")

    # --- 2. Class balance ---------------------------------------------------
    dist = sub["phase"].value_counts(normalize=True).sort_index()
    print("\n  2. CLASS BALANCE (share of zone-month observations)")
    for ph, share in dist.items():
        n = int((sub["phase"] == ph).sum())
        print(f"     phase {ph:.0f}: {share:6.2%}  (n={n:,})")
    crisis = float((sub["phase"] >= 3).mean())
    print(f"     crisis (phase 3+): {crisis:.2%}")
    print(f"     -> majority-class baseline accuracy would be {dist.max():.1%}; "
          f"any model must beat that to be meaningful.")
    by_year = sub.groupby(sub[time_col].dt.year)["phase"].apply(lambda s: (s >= 3).mean())
    print("     crisis rate by year: " + ", ".join(
        f"{y}:{v:.0%}" for y, v in by_year.items()))

    # --- 3. Spatial coverage ------------------------------------------------
    per_month = sub.groupby("month")["fnid"].nunique()
    print(f"\n  3. SPATIAL COVERAGE: zones per assessed month -- "
          f"min {per_month.min()}, median {per_month.median():.0f}, max {per_month.max()}")
    coverage = sub.groupby("fnid")["month"].nunique()
    full = int((coverage == len(months)).sum())
    print(f"     {len(coverage)} distinct fnids; {full} appear in every assessed month")
    print(f"     -> fnid is NOT a stable panel key. This is expected (vintages) "
          f"and is what the admin2 reprojection exists to fix.")

    # --- 4. Vintage discontinuity ------------------------------------------
    sub["vintage"] = sub["fnid"].astype(str).str.slice(2, 6)
    vint = sub.groupby([sub[time_col].dt.year, "vintage"]).size().unstack(fill_value=0)
    print("\n  4. BOUNDARY VINTAGE BY YEAR (rows=year, cols=vintage)")
    for line in vint.to_string().splitlines():
        print(f"     {line}")
    switches = []
    for year in sorted(sub[time_col].dt.year.unique()):
        vs = set(sub.loc[sub[time_col].dt.year == year, "vintage"])
        if len(vs) > 1:
            switches.append((year, sorted(vs)))
    if switches:
        print(f"     years spanning multiple vintages: "
              + ", ".join(f"{y}({'/'.join(v)})" for y, v in switches))
        print(f"     -> ANY lagged-IPC feature crossing these years is unreliable "
              f"at fnid level. Build lags AFTER reprojecting to admin2, never before.")

    # --- 5. Implausible transitions ----------------------------------------
    seq = sub.sort_values([ "fnid", time_col])
    seq["prev"] = seq.groupby("fnid")["phase"].shift(1)
    seq["delta"] = (seq["phase"] - seq["prev"]).abs()
    jumps = seq[seq["delta"] >= 2]
    print(f"\n  5. TRANSITIONS: {int(seq['delta'].notna().sum()):,} consecutive pairs")
    if seq["delta"].notna().any():
        vc = seq["delta"].value_counts(normalize=True).sort_index()
        print("     |change| distribution: " + ", ".join(
            f"{d:.0f}:{v:.2%}" for d, v in vc.items()))
    print(f"     jumps of 2+ phases: {len(jumps):,} "
          f"({len(jumps)/max(int(seq['delta'].notna().sum()),1):.2%})")
    if len(jumps):
        worst = jumps.groupby(jumps[time_col].dt.to_period("M")).size().nlargest(5)
        print("     months with most large jumps: " + ", ".join(
            f"{m}:{n}" for m, n in worst.items()))

    print("\n  VERDICT GUIDE")
    print("     Temporal density and vintage discontinuity are the two that")
    print("     change model results. Class balance sets the metric. Coverage")
    print("     and transitions are usually descriptive, not defects.")
    return sub



def validate(busker_dir=None, scenarios=SCENARIOS):
    """Stage 4: cell-by-cell comparison against Busker's rasters.

    Reports agreement on the overlapping (time, cell) set. Following the
    replication guide: check every available match, not a handful, and report
    a distribution rather than examples.
    """
    import xarray as xr

    busker_dir = Path(busker_dir or BUSKER_DIR)
    print("\n[4] Validation against Busker et al. rasters")

    rows = []
    for scenario in scenarios:
        ref_path = busker_dir / f"fews_xr_{scenario}.nc"
        own_path = INTERIM_DIR / f"ipc_grid_{scenario}.nc"
        if not ref_path.exists() or not own_path.exists():
            print(f"  {scenario}: missing "
                  f"{'reference' if not ref_path.exists() else 'own output'}, skipping")
            continue

        ref = xr.open_dataset(ref_path)[scenario]
        own = xr.open_dataset(own_path)[scenario]

        shared = sorted(set(pd.to_datetime(ref.time.values))
                        & set(pd.to_datetime(own.time.values)))
        if not shared:
            print(f"  {scenario}: no overlapping timesteps "
                  f"(ref {pd.Timestamp(ref.time.values[0]):%Y-%m}.."
                  f"{pd.Timestamp(ref.time.values[-1]):%Y-%m}, "
                  f"own {pd.Timestamp(own.time.values[0]):%Y-%m}.."
                  f"{pd.Timestamp(own.time.values[-1]):%Y-%m})")
            continue

        ref_s = ref.sel(time=shared)
        own_s = own.sel(time=shared).reindex_like(ref_s, method="nearest", tolerance=1e-6)

        a, b = ref_s.values, own_s.values
        both = np.isfinite(a) & np.isfinite(b)
        n = int(both.sum())
        if n == 0:
            print(f"  {scenario}: no co-located valid cells -- check grid alignment")
            continue

        exact = float((a[both] == b[both]).mean())
        within1 = float((np.abs(a[both] - b[both]) <= 1).mean())
        bias = float(np.mean(b[both] - a[both]))

        only_ref = int((np.isfinite(a) & ~np.isfinite(b)).sum())
        only_own = int((~np.isfinite(a) & np.isfinite(b)).sum())

        print(f"\n  {scenario}: {len(shared)} shared timesteps, {n:,} co-located cells")
        print(f"    exact phase agreement : {exact:.2%}")
        print(f"    within 1 phase        : {within1:.2%}")
        print(f"    mean signed diff      : {bias:+.4f} (own - Busker)")
        print(f"    classified only by Busker: {only_ref:,}   only by us: {only_own:,}")

        per_time = []
        for t in shared:
            aa = ref.sel(time=t).values
            bb = own.sel(time=t).values
            m = np.isfinite(aa) & np.isfinite(bb)
            if m.sum():
                per_time.append((t, float((aa[m] == bb[m]).mean())))
        if per_time:
            worst = sorted(per_time, key=lambda x: x[1])[:5]
            print("    worst timesteps: " + ", ".join(
                f"{pd.Timestamp(t):%Y-%m} {v:.1%}" for t, v in worst))

        rows.append({"scenario": scenario, "timesteps": len(shared), "cells": n,
                     "exact": exact, "within_1": within1, "bias": bias})

    if rows:
        summary = pd.DataFrame(rows)
        out = INTERIM_DIR / "validation_vs_busker.csv"
        summary.to_csv(out, index=False)
        print(f"\n  wrote {out}")
        print("\n  Interpretation: exact agreement below ~95% points at a "
              "rasterization or vintage-selection difference, NOT at population "
              "weighting - that has not been applied yet at this stage.")
    return rows


# --------------------------------------------------------------------------
# Pipeline runner
# --------------------------------------------------------------------------
#
# Edit the settings below, then run:   python fetch_fews_ipc.py
#
# Every stage caches its output, so re-running is cheap: `fetch` skips files
# already downloaded and verified, and the later stages simply recompute from
# the cache. Set FORCE_DOWNLOAD = True to re-pull from the API.

RUN_COUNTRIES = ["ET"]           # ["ET", "KE", "SO", "SD", "SS", "UG"] for the full HoA baseline
RUN_SCENARIOS = ["CS", "ML1", "ML2"]
FORCE_DOWNLOAD = False

# "collection" indexes rasters by publication date and reproduces Busker's
# files. Switch to "period" to index by the period being forecast, which is
# what benchmarking needs. See rasterize() for why these differ only for ML2.
RUN_TIME_KEY = "collection"

# Copy the grid from Busker's file so validation is a straight array
# subtraction. Set to None to snap to the CHIRPS p05 grid over RUN_EXTENT.
RUN_GRID_LIKE = BUSKER_DIR / "fews_xr_CS.nc"
RUN_EXTENT = "hoa"

RUN_AUDIT = True
RUN_DIAGNOSE = False             # investigative; off by default
DIAGNOSE_SCENARIO = "CS"
DIAGNOSE_TIMES = ["2012-04", "2012-07"]


def main():
    print("=" * 74)
    print("FEWS NET IPC pipeline")
    print(f"  countries : {', '.join(RUN_COUNTRIES)}")
    print(f"  scenarios : {', '.join(RUN_SCENARIOS)}")
    print(f"  time key  : {RUN_TIME_KEY}")
    print("=" * 74)

    inspect_busker()

    fetch_ipcphase(RUN_COUNTRIES, RUN_SCENARIOS, force=FORCE_DOWNLOAD)
    fetch_features(RUN_COUNTRIES, force=FORCE_DOWNLOAD)

    tidy(RUN_SCENARIOS, RUN_COUNTRIES)

    like = str(RUN_GRID_LIKE) if RUN_GRID_LIKE and Path(RUN_GRID_LIKE).exists() else None
    if RUN_GRID_LIKE and not like:
        print(f"\n  NOTE: {RUN_GRID_LIKE} not found; snapping to the CHIRPS grid instead")
    rasterize(extent=RUN_EXTENT, like=like, scenarios=RUN_SCENARIOS,
              time_key=RUN_TIME_KEY)

    validate(scenarios=RUN_SCENARIOS)

    if RUN_AUDIT:
        for scenario in RUN_SCENARIOS:
            audit(scenario=scenario, time_key=RUN_TIME_KEY)

    if RUN_DIAGNOSE:
        diagnose(scenario=DIAGNOSE_SCENARIO, times=DIAGNOSE_TIMES,
                 time_key=RUN_TIME_KEY)

    print("\n" + "=" * 74)
    print("Done. Rasters written to:", INTERIM_DIR)
    print("Next: aggregate_fews_ipc.py -- population-weighted reprojection")
    print("      onto the fixed 92-zone admin2 grid.")
    print("=" * 74)


if __name__ == "__main__":
    main()