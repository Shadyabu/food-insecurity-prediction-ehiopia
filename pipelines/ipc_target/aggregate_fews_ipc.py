#!/usr/bin/env python3
"""
aggregate_fews_ipc.py
=====================
Stage 3: population-weighted reprojection of the IPC rasters onto the fixed
92-zone admin2 grid.

    [1. ACQUIRE]   FDW REST API -> cached CSV/GeoJSON          (fetch_fews_ipc.py)
    [2. RASTERIZE] fsc_admin polygons -> CHIRPS 0.05deg grid   (fetch_fews_ipc.py)
    [3. AGGREGATE] population-weighted mean -> admin2 panel    (this script)
    [4. VALIDATE]  against Busker / World Bank references      (this script)

WHY THIS STEP EXISTS
--------------------
The features and the target live on incompatible geographies:

    CHIRPS features   92 admin2 zones      key ADM2_PCODE    one fixed set
    IPC target        ~4,769 fsc_admin     key fnid          five vintages

They cannot be joined as-is. Worse, fnid is not a stable panel key: FEWS NET
redraws its food-security mapping units and reissues identifiers (vintage
transitions land in 2019, 2020, 2021 and 2023 for Ethiopia). A lagged IPC
feature -- Busker's single strongest SHAP predictor -- silently breaks across
those boundaries, because the "same place" acquires a new id. Reprojecting onto
the fixed admin2 set makes zone_id a genuine panel key across the full period,
which is why this must happen BEFORE any lag or rolling feature is built.

METHOD
------
Busker et al. state they used area-level classifications (the highest phase
faced by at least 20% of the population) and computed the population-weighted
spatial mean per administrative unit, using gridded WorldPop unconstrained data
adjusted to match official UN country totals.

Population weighting rather than area weighting because IPC is a
population-based measure: the classification is defined by the share of PEOPLE
in each phase, so an empty desert should not outvote a dense town.

    IPC_zone,t = sum_i(pop_i * phase_i,t) / sum_i(pop_i)     over cells i in zone

The IPC rasters produced by stage 2 already sit on the CHIRPS 0.05deg grid, so
the zone raster from the rainfall pipeline applies unchanged -- no reprojection
and no vector overlay is required.

THREE TARGETS ARE EMITTED, deliberately
---------------------------------------
* ipc_continuous  -- the population-weighted mean. Busker's target, evaluated
                     by MAE. Needed for like-for-like RQ1 comparison.
* ipc_phase_20pct -- the highest phase reached by >=20% of the zone's
                     population. This is IPC's OWN area-classification rule, so
                     it is reproducible and defensible. Use it for weighted F1.
                     Do NOT simply round ipc_continuous: rounding averages away
                     exactly the crisis signal the intervention thresholds
                     depend on (a zone that is 30% Phase 4 and 70% Phase 1 has a
                     mean of 1.9, which rounds to "no intervention").
* pct_phase3plus  -- share of zone population in Phase 3 or above. Falls out for
                     free and is arguably the most decision-relevant of the
                     three, since humanitarian caseload is a headcount.

USAGE
-----
    python aggregate_fews_ipc.py

Edit the configuration block below first -- it needs paths to your admin2
shapefile and your WorldPop rasters.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent  # repo root (pipelines/ipc_target/ -> pipelines/ -> repo root)
INTERIM_DIR = ROOT / "data" / "interim" / "ipc_target"
OUTPUT_DIR = ROOT / "data" / "processed" / "features"
BUSKER_DIR = SCRIPT_DIR / "busker_comparison"

SCENARIOS = ["CS", "ML1", "ML2"]

# Ethiopia's official 92-zone admin2 boundaries (CSA/BoFED, OCHA COD format) --
# the same file the CHIRPS pipeline uses. Reusing it is the whole point: it is
# what makes the features and the target joinable.
ADMIN2_SHAPEFILE = "boundaries/eth_admbnda_adm2_csa_bofedb_2021.shp"
ADMIN2_SEARCH = ["**/*adm2*.shp", "**/*ADM2*.shp", "**/*admin2*.shp"]
ADMIN2_NAME_FIELD = "ADM2_EN"
ADMIN2_CODE_FIELD = "ADM2_PCODE"

# WorldPop rasters, one per year. Two products exist and they are NOT
# interchangeable:
#   *_ppp_*  people per pixel  (a COUNT -- aggregate by summing)
#   *_pd_*   people per km2    (a DENSITY -- must be multiplied by cell area
#                               before summing, or the result is meaningless)
# Busker uses the unconstrained, UN-adjusted product. Either form works here
# because POPULATION_IS_DENSITY converts density to counts before aggregation.
WORLDPOP_DIR = "worldpop"
WORLDPOP_PATTERN = "*UNadj*.tif"
POPULATION_IS_DENSITY = "auto"   # "auto" | True | False

# Year-matched weights track real population change but let a zone's score drift
# slightly even when no phase changed; a fixed year holds weights constant and
# isolates phase change. Year-matched is the better default, but run both and
# compare -- it is a cheap sensitivity check and a defensible paragraph.
POPULATION_YEAR_MODE = "matched"   # "matched" | "fixed"
POPULATION_FIXED_YEAR = 2015

# The IPC area-classification threshold. 0.20 is IPC's own convention.
AREA_RULE_THRESHOLD = 0.20

# Zones whose population coverage falls below this are flagged, not dropped --
# a partially-covered zone is a real observation with a caveat, and silently
# discarding it would bias evaluation toward well-monitored areas.
MIN_COVERAGE = 0.50

# Supersampling factor for fractional zone coverage. Binary rasterization
# assigns equal weight to every cell a polygon touches at all, which is badly
# wrong for small zones (Dire Dawa urban: 8 cells; verified in the CHIRPS work
# to give 50.0mm against a correct area-weighted 28.6mm). Supersampling by 10
# resolves each 0.05deg cell into 100 subcells and measures actual overlap.
SUPERSAMPLE = 10


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def _resolve(path_like):
    """Resolve a configured path against the REPO ROOT, not the cwd.

    A relative path like "boundaries/x.shp" is otherwise interpreted relative to
    wherever the script happened to be launched from, which is the usual reason
    a correctly-configured path still reports as missing.
    """
    p = Path(path_like)
    if p.is_absolute():
        return p
    if (ROOT / p).exists():
        return ROOT / p
    if p.exists():
        return p.resolve()
    return ROOT / p


def find_admin2():
    """Locate the admin2 shapefile, searching if the configured path misses."""
    if ADMIN2_SHAPEFILE:
        p = _resolve(ADMIN2_SHAPEFILE)
        if p.exists():
            return p
        print(f"  configured path not found: {p}")

    search_roots = [ROOT]
    for root in search_roots:
        for pattern in ADMIN2_SEARCH:
            hits = sorted(root.glob(pattern))
            if hits:
                print(f"  found admin2 shapefile by search: {hits[0]}")
                return hits[0]

    # Diagnostic: show what actually exists, rather than just failing.
    nearby = []
    for root in search_roots:
        nearby += [str(p.relative_to(root)) for p in list(root.rglob("*.shp"))[:20]]
    raise SystemExit(
        "Could not find the admin2 shapefile.\n"
        f"  repo root        : {ROOT}\n"
        f"  working directory: {Path.cwd()}\n"
        f"  configured value : {ADMIN2_SHAPEFILE!r}\n"
        f"  .shp files found nearby: {nearby or 'none'}\n\n"
        "Relative paths are resolved against the REPO ROOT. If your\n"
        "boundaries/ folder sits somewhere else, give an absolute path instead."
    )


def load_admin2():
    import geopandas as gpd

    path = find_admin2()
    gdf = gpd.read_file(path)
    print(f"  admin2: {len(gdf)} zones from {path.name}")

    if gdf.crs is None:
        print("  WARNING: shapefile has no CRS; assuming EPSG:4326")
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_epsg() != 4326:
        print(f"  reprojecting {gdf.crs.to_string()} -> EPSG:4326")
        gdf = gdf.to_crs("EPSG:4326")

    name_col = ADMIN2_NAME_FIELD if ADMIN2_NAME_FIELD in gdf.columns else None
    code_col = ADMIN2_CODE_FIELD if ADMIN2_CODE_FIELD in gdf.columns else None
    if name_col is None:
        cands = [c for c in gdf.columns if re.search(r"adm2.*(en|name)", c, re.I)]
        name_col = cands[0] if cands else gdf.columns[0]
        print(f"  name field not found; using {name_col!r}")
    if code_col is None:
        cands = [c for c in gdf.columns if re.search(r"adm2.*(pcode|code)", c, re.I)]
        code_col = cands[0] if cands else name_col
        print(f"  code field not found; using {code_col!r}")

    gdf = gdf[[name_col, code_col, "geometry"]].rename(
        columns={name_col: "zone_name", code_col: "zone_code"}
    )
    gdf["zone_id"] = np.arange(1, len(gdf) + 1)

    invalid = int((~gdf.geometry.is_valid).sum())
    if invalid:
        print(f"  repairing {invalid} invalid geometries with buffer(0)")
        gdf.loc[~gdf.geometry.is_valid, "geometry"] = gdf.loc[
            ~gdf.geometry.is_valid, "geometry"
        ].buffer(0)

    if len(gdf) != 92:
        print(f"  NOTE: expected 92 zones, found {len(gdf)}. Confirm this is the "
              f"same boundary set the CHIRPS features were built on.")
    return gdf


def find_worldpop():
    """Map year -> WorldPop raster path."""
    search_roots = [_resolve(WORLDPOP_DIR)] if WORLDPOP_DIR else []
    search_roots += [ROOT]

    found = {}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob(WORLDPOP_PATTERN):
            m = re.search(r"(19|20)\d{2}", path.name)
            if m:
                found.setdefault(int(m.group(0)), path)
        if found:
            break

    if not found:
        raise SystemExit(
            "No WorldPop rasters found.\n"
            f"  repo root        : {ROOT}\n"
            f"  configured dir   : {WORLDPOP_DIR!r} -> "
            f"{_resolve(WORLDPOP_DIR) if WORLDPOP_DIR else 'unset'}\n"
            f"  pattern          : {WORLDPOP_PATTERN!r}\n"
            "Busker uses the UNCONSTRAINED, UN-ADJUSTED product. The constrained "
            "variant masks unbuilt areas and gives different weights."
        )
    print(f"  WorldPop: {len(found)} raster(s), years {sorted(found)}")
    if len(found) == 1:
        print(f"    only one year available -- population weights will be held "
              f"constant across the whole period regardless of "
              f"POPULATION_YEAR_MODE={POPULATION_YEAR_MODE!r}")
    return found


# --------------------------------------------------------------------------
# Grid machinery
# --------------------------------------------------------------------------

def grid_from_nc(path):
    import xarray as xr

    with xr.open_dataset(path) as ds:
        return ds["lon"].values.copy(), ds["lat"].values.copy()


def transform_from_grid(lon, lat):
    from rasterio.transform import from_origin

    res_x = float(abs(lon[1] - lon[0]))
    res_y = float(abs(lat[0] - lat[1]))
    return from_origin(float(lon[0]) - res_x / 2.0,
                       float(lat[0]) + res_y / 2.0, res_x, res_y)


def zone_fraction_grid(gdf, lon, lat, supersample=SUPERSAMPLE):
    """Fractional coverage of each grid cell by each zone.

    Returns an array of shape (n_zones, ny, nx) giving, for every cell, the
    share of its area falling inside each zone. Computed by rasterizing at
    `supersample`x resolution and block-averaging, which handles small and
    oddly-shaped zones correctly without a separate special case, and which also
    resolves the overlap ambiguity at shared borders.
    """
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_origin

    ny, nx = len(lat), len(lon)
    res_x = float(abs(lon[1] - lon[0]))
    res_y = float(abs(lat[0] - lat[1]))

    fine_transform = from_origin(
        float(lon[0]) - res_x / 2.0, float(lat[0]) + res_y / 2.0,
        res_x / supersample, res_y / supersample,
    )
    fine = rio_rasterize(
        ((geom, int(zid)) for geom, zid in zip(gdf.geometry, gdf["zone_id"])),
        out_shape=(ny * supersample, nx * supersample),
        transform=fine_transform, fill=0, all_touched=False, dtype="int32",
    )

    n_zones = len(gdf)
    frac = np.zeros((n_zones, ny, nx), dtype="float32")
    blocks = fine.reshape(ny, supersample, nx, supersample)
    for k, zid in enumerate(gdf["zone_id"].values):
        frac[k] = (blocks == zid).mean(axis=(1, 3))

    covered = frac.sum(axis=0)
    print(f"  zone coverage: {(covered > 0).sum():,} cells touched, "
          f"max overlap sum {covered.max():.3f}")
    tiny = [(gdf.iloc[k]["zone_name"], float(frac[k].sum()))
            for k in range(n_zones) if frac[k].sum() < 2.0]
    if tiny:
        print(f"  {len(tiny)} zones smaller than 2 whole cells "
              f"(fractional weighting matters here): "
              f"{', '.join(n for n, _ in tiny[:5])}")
    return frac


def worldpop_to_grid(pop_path, lon, lat):
    """Resample a WorldPop raster onto the coarse IPC grid as population COUNTS.

    Two corrections that are easy to get silently wrong:

    1. Density vs count. A *_pd_* raster is people per km2. Summing density
       values is meaningless -- it produces a number with no units that varies
       with resolution. Density must be multiplied by each pixel's true area
       first, and in EPSG:4326 that area shrinks with cos(latitude), so a fixed
       factor is also wrong. For Ethiopia (3.3N-14.9N) cos(lat) ranges 1.00 to
       0.97, a ~3% north-south gradient -- small, but it biases exactly the
       northern highland zones where most of the population lives.

    2. Sum, not average, when going from fine to coarse. Population is an
       extensive quantity: each coarse cell must receive the TOTAL of the fine
       cells inside it. Averaging silently rescales by the resolution ratio.

    Uses rasterio's warp machinery rather than manual index arithmetic so that
    partial cells at the grid edge and any CRS difference are handled properly.
    """
    import rasterio
    from rasterio.warp import Resampling, reproject

    is_density = POPULATION_IS_DENSITY
    if is_density == "auto":
        name = Path(pop_path).name.lower()
        is_density = ("_pd_" in name or "_dens" in name)

    with rasterio.open(pop_path) as src:
        band = src.read(1).astype("float64")
        if src.nodata is not None:
            band[band == src.nodata] = 0.0
        band = np.nan_to_num(band, nan=0.0, posinf=0.0, neginf=0.0)
        band[band < 0] = 0.0

        if is_density:
            res_x = abs(src.transform.a)
            res_y = abs(src.transform.e)
            rows = np.arange(src.height)
            # Cell-centre latitude for every row.
            lats = src.transform.f + (rows + 0.5) * src.transform.e
            km_per_deg = 111.32
            area_km2 = (
                (res_x * km_per_deg * np.cos(np.radians(lats)))[:, None]
                * (res_y * km_per_deg)
            )
            band = band * area_km2
            label = "density -> counts"
        else:
            label = "counts"

        dst = np.zeros((len(lat), len(lon)), dtype="float64")
        reproject(
            source=band,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs or "EPSG:4326",
            dst_transform=transform_from_grid(lon, lat),
            dst_crs="EPSG:4326",
            resampling=Resampling.sum,
        )

    print(f"    {Path(pop_path).name} [{label}]: "
          f"{band.sum()/1e6:.1f}M national, {dst.sum()/1e6:.1f}M on grid")
    return dst


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------

def aggregate_scenario(scenario, gdf, frac, pop_by_year, lon, lat):
    """Population-weighted aggregation of one scenario onto the admin2 zones."""
    import xarray as xr

    nc_path = INTERIM_DIR / f"ipc_grid_{scenario}.nc"
    if not nc_path.exists():
        print(f"  {scenario}: {nc_path.name} not found, skipping")
        return None

    ds = xr.open_dataset(nc_path)
    phase = ds[scenario]
    has_ha = "HA" in ds.data_vars
    times = pd.to_datetime(phase["time"].values)
    print(f"\n  {scenario}: {len(times)} timesteps, HA layer {'present' if has_ha else 'absent'}")

    pop_cache = {}
    warned = set()

    def population_for(year):
        key = POPULATION_FIXED_YEAR if POPULATION_YEAR_MODE == "fixed" else year
        available = sorted(pop_by_year)
        nearest = min(available, key=lambda y: abs(y - key))
        path = pop_by_year[nearest]
        # Cache on the resolved PATH, not the requested year: with a single
        # raster available, every year resolves to the same file, and caching by
        # year would reload and re-warp it once per year for no benefit.
        if path not in pop_cache:
            if nearest != key and nearest not in warned:
                print(f"    population year {key} unavailable; using {nearest} "
                      f"for all years that resolve to it")
                warned.add(nearest)
            pop_cache[path] = worldpop_to_grid(path, lon, lat)
        return pop_cache[path]

    records = []
    for t_idx, t in enumerate(times):
        arr = phase.isel(time=t_idx).values.astype("float64")
        valid = np.isfinite(arr)
        pop = population_for(int(t.year))
        ha_arr = ds["HA"].isel(time=t_idx).values.astype("float64") if has_ha else None

        for k in range(len(gdf)):
            w = frac[k] * pop                       # area-fraction x population
            total_w = float(w.sum())
            if total_w <= 0:
                continue
            w_valid = w * valid
            covered_w = float(w_valid.sum())
            coverage = covered_w / total_w
            if covered_w <= 0:
                continue

            phases = arr[valid]
            weights = w_valid[valid]

            mean_phase = float((phases * weights).sum() / covered_w)

            # Same aggregation with area weights only (no population). Kept so
            # the effect of the population weighting can be measured directly:
            # there is no external admin-level reference for this step, so an
            # internal contrast is the available evidence.
            wa = frac[k] * valid
            wa_sum = float(wa.sum())
            mean_phase_area = (float((arr[valid] * wa[valid]).sum() / wa_sum)
                               if wa_sum > 0 else np.nan)

            # Population share per phase -> IPC's own area-classification rule.
            shares = {}
            for ph in (1, 2, 3, 4, 5):
                shares[ph] = float(weights[phases == ph].sum() / covered_w)
            above = [ph for ph in (5, 4, 3, 2, 1) if shares[ph] >= AREA_RULE_THRESHOLD]
            area_phase = above[0] if above else int(round(mean_phase))
            pct3plus = shares[3] + shares[4] + shares[5]

            rec = {
                "zone_id": int(gdf.iloc[k]["zone_id"]),
                "zone_name": gdf.iloc[k]["zone_name"],
                "zone_code": gdf.iloc[k]["zone_code"],
                "time": t,
                "scenario": scenario,
                "ipc_continuous": mean_phase,
                "ipc_continuous_area": mean_phase_area,
                "ipc_phase_20pct": int(area_phase),
                "pct_phase3plus": pct3plus,
                "pop_coverage": coverage,
                "pop_total": total_w,
                "low_coverage": coverage < MIN_COVERAGE,
            }
            for ph in (1, 2, 3, 4, 5):
                rec[f"pct_phase{ph}"] = shares[ph]
            if ha_arr is not None:
                ha_valid = np.isfinite(ha_arr) & valid
                if (w * ha_valid).sum() > 0:
                    rec["ha_share"] = float(
                        (ha_arr[ha_valid] * w[ha_valid]).sum() / (w * ha_valid).sum()
                    )
            records.append(rec)

    out = pd.DataFrame(records)
    print(f"    -> {len(out):,} zone-timestep rows, "
          f"{out['low_coverage'].sum():,} flagged low coverage")
    return out


# --------------------------------------------------------------------------
# Validation and reporting
# --------------------------------------------------------------------------

def report(panel):
    """Sanity-check the aggregated panel before it is used for modelling."""
    print("\n[4] Aggregated panel report")

    for scenario, grp in panel.groupby("scenario"):
        print(f"\n  {scenario}: {len(grp):,} rows, {grp['zone_id'].nunique()} zones, "
              f"{grp['time'].nunique()} timesteps")

        # The panel key must now be complete -- that is the entire point of
        # reprojecting. A ragged panel here means something went wrong.
        expected = grp["zone_id"].nunique() * grp["time"].nunique()
        print(f"    panel completeness: {len(grp)/expected:.1%} of "
              f"{expected:,} zone-timestep cells")

        dist = grp["ipc_phase_20pct"].value_counts(normalize=True).sort_index()
        print("    ipc_phase_20pct distribution: " + ", ".join(
            f"{int(p)}:{v:.1%}" for p, v in dist.items()))
        print(f"    ipc_continuous: mean {grp['ipc_continuous'].mean():.3f}, "
              f"range {grp['ipc_continuous'].min():.2f}-{grp['ipc_continuous'].max():.2f}")
        print(f"    crisis rate: 20pct-rule {(grp['ipc_phase_20pct']>=3).mean():.1%}, "
              f"pop-weighted {grp['pct_phase3plus'].mean():.1%}")

        # Does the area rule diverge from naive rounding? If it does, that is
        # the crisis signal rounding would have destroyed.
        rounded = grp["ipc_continuous"].round().astype(int).clip(1, 5)
        differs = float((rounded != grp["ipc_phase_20pct"]).mean())
        harsher = float((grp["ipc_phase_20pct"] > rounded).mean())
        print(f"    area rule differs from rounding in {differs:.1%} of rows "
              f"({harsher:.1%} of rows classified MORE severe than rounding)")

        low = grp[grp["low_coverage"]]
        if len(low):
            worst = low.groupby("zone_name").size().nlargest(5)
            print(f"    low-coverage zones: " + ", ".join(
                f"{n}({c})" for n, c in worst.items()))


def validate_against_busker(panel):
    """Internal validation of the weighting step.

    Busker released only gridded rasters (fews_xr_CS/ML1/ML2.nc), not an
    admin-level table, so there is no external reference for THIS step. Cell
    agreement was already established at stage 2 (99.5% CS, 100% ML1/ML2), which
    means any error here must originate in the weighting itself -- and that can
    be checked internally, without a reference, by three arguments:

      1. Degenerate zones. Where every classified cell in a zone carries the
         same phase, the weighted mean MUST equal that phase exactly, whatever
         the weights are. This is the sharpest available test: it isolates the
         aggregation arithmetic from any question about the weights, and any
         failure is unambiguous rather than a matter of degree.
      2. Population conservation. Zone populations must sum to the national
         total, which is independently known from UN estimates.
      3. Weighting sensitivity. Comparing population- against area-weighted
         scores bounds how much the weighting choice can possibly matter.
    """
    print("\n[4b] Internal validation of the weighting step")
    print("     (Busker released rasters only, no admin-level table, so this")
    print("      step has no external reference -- these checks stand in.)")

    share_cols = [f"pct_phase{p}" for p in (1, 2, 3, 4, 5) if f"pct_phase{p}" in panel]
    if share_cols:
        shares = panel[share_cols].to_numpy()
        phases = np.array([int(c.replace("pct_phase", "")) for c in share_cols])

        # The strongest available test, and it applies to EVERY row rather than
        # only the single-phase ones: the weighted mean must equal the
        # share-weighted sum of phases, by definition of both quantities. Any
        # deviation beyond floating-point noise means the aggregation and the
        # share computation disagree, which cannot happen if both are correct.
        expected = shares @ phases
        err = np.abs(panel["ipc_continuous"].to_numpy() - expected)
        sums = shares.sum(axis=1)
        print(f"\n  1. AGGREGATION IDENTITY (all {len(panel):,} rows)")
        print(f"     max |continuous - sum(share_p * p)| : {err.max():.2e}")
        print(f"     max |sum of shares - 1|             : {np.abs(sums - 1).max():.2e}")
        ok = err.max() < 1e-9 and np.abs(sums - 1).max() < 1e-9
        print(f"     -> {'PASS' if ok else 'FAIL -- aggregation arithmetic is wrong'}")

        # Single-phase zones, tested against their own impurity rather than an
        # absolute tolerance. A zone counted as uniform at a 0.9999 threshold may
        # still hold 1e-4 of a neighbouring phase, and its mean is then off by
        # exactly that much -- which is correct behaviour, not an error.
        max_share = shares.max(axis=1)
        uniform = max_share > 0.9999
        n_uniform = int(uniform.sum())
        if n_uniform:
            dominant = phases[np.argmax(shares[uniform], axis=1)]
            dev = np.abs(panel.loc[uniform, "ipc_continuous"].to_numpy() - dominant)
            impurity = 1.0 - max_share[uniform]
            headroom = impurity * (phases.max() - phases.min())
            rule_ok = (panel.loc[uniform, "ipc_phase_20pct"].to_numpy() == dominant)
            within = dev <= headroom + 1e-9
            print(f"\n  1b. NEAR-UNIFORM ZONES: {n_uniform:,} single-phase "
                  f"({n_uniform/len(panel):.0%} of panel)")
            print(f"      max deviation {dev.max():.2e}, max explainable by "
                  f"residual impurity {headroom.max():.2e}")
            print(f"      deviations within their own impurity bound: {within.mean():.2%}")
            print(f"      area rule returns the dominant phase: {rule_ok.mean():.2%}")
            print(f"      -> {'PASS' if within.all() and rule_ok.all() else 'FAIL'}")

    first = panel[panel["time"] == panel["time"].min()]
    total = first.groupby("scenario")["pop_total"].sum().max()
    print(f"\n  2. POPULATION CONSERVATION: zone totals sum to {total/1e6:.1f}M")
    print(f"     UN estimate for Ethiopia 2020: ~114.9M")
    print(f"     -> {abs(total/1e6 - 114.9)/114.9:.1%} difference "
          f"(residual is cells outside any zone: coastline and border slivers)")

    if "ipc_continuous_area" in panel.columns:
        both = panel.dropna(subset=["ipc_continuous", "ipc_continuous_area"])
        diff = (both["ipc_continuous"] - both["ipc_continuous_area"]).abs()
        corr = both["ipc_continuous"].corr(both["ipc_continuous_area"])
        print(f"\n  3. WEIGHTING SENSITIVITY (population vs area weights)")
        print(f"     correlation {corr:.4f} | mean |diff| {diff.mean():.4f} phases "
              f"| 95th pct {diff.quantile(0.95):.4f} | max {diff.max():.4f}")
        big = both.loc[diff > 0.5]
        if len(big):
            worst = big.groupby("zone_name").size().nlargest(5)
            print(f"     {len(big):,} rows differ by >0.5 phase, concentrated in: "
                  + ", ".join(f"{n}({c})" for n, c in worst.items()))
            print(f"     -> these are the internally heterogeneous zones where the "
                  f"weighting choice is load-bearing; worth naming in the writeup")
        else:
            print(f"     -> no row differs by more than 0.5 phase: the weighting "
                  f"choice cannot materially change results")


def main():
    print("=" * 74)
    print("FEWS NET IPC -> admin2 population-weighted aggregation")
    print(f"  population weights: {POPULATION_YEAR_MODE}"
          + (f" ({POPULATION_FIXED_YEAR})" if POPULATION_YEAR_MODE == "fixed" else ""))
    print(f"  area rule threshold: {AREA_RULE_THRESHOLD:.0%}")
    print("=" * 74)

    print("\n[1] Inputs")
    gdf = load_admin2()
    pop_by_year = find_worldpop()

    reference = INTERIM_DIR / "ipc_grid_CS.nc"
    if not reference.exists():
        raise SystemExit(f"{reference} not found. Run fetch_fews_ipc.py first.")
    lon, lat = grid_from_nc(reference)
    print(f"  grid: {len(lon)} x {len(lat)}, "
          f"lon {lon[0]:.4f}..{lon[-1]:.4f}, lat {lat[0]:.4f}..{lat[-1]:.4f}")

    print("\n[2] Zone fractional coverage")
    frac = zone_fraction_grid(gdf, lon, lat)

    print("\n[3] Population-weighted aggregation")
    frames = []
    for scenario in SCENARIOS:
        out = aggregate_scenario(scenario, gdf, frac, pop_by_year, lon, lat)
        if out is not None and len(out):
            frames.append(out)

    if not frames:
        raise SystemExit("Nothing aggregated.")

    panel = pd.concat(frames, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "ipc_target_admin2_panel.csv"
    panel.to_csv(out_path, index=False)

    report(panel)
    validate_against_busker(panel)

    print(f"\n  wrote {out_path} ({len(panel):,} rows)")
    print("\n" + "=" * 74)
    print("Done. Join to the CHIRPS features on (zone_code, time).")
    print("Build lagged IPC features from THIS panel, never from fnid-level data.")
    print("=" * 74)


if __name__ == "__main__":
    main()