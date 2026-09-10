"""
compute_wvg_monthly_features.py

Engineering stage for the Western V Gradient (WVG) pipeline. Derives WVG
from the raw NOAA ERSSTv5 gridded SST file (fetch_wvg_ersst.py) and writes
data/processed/features/wvg_monthly.csv.

WHY THIS IS A DERIVATION, NOT A FETCH (see
docs/WVG_Pipeline_Documentation.md Sec 1-2 for the full investigation):
unlike IOD/MEI/NINO3.4, no agency publishes an updated, ongoing WVG index
file. Funk et al. (2023)'s own data release is a one-time snapshot. This
pipeline instead reproduces their published formula against the live
ERSSTv5 grid, using the formula recovered from two independent sources
that agree word-for-word: Busker et al.'s own staged reference
(pipelines/teleconnections/busker_comparison/SST.xlsx, sheet "WVG" header
block) and Funk et al.'s Dryad dataset's own methods text.

FORMULA
  WVG(year) = standardize[ NINO3.4_MAM(year) - WesternV_MAM(year) ],
  standardized on a fixed 1950-2020 baseline (subtract that window's mean,
  divide by that window's std -- NOT re-fit per year, matching Busker's
  header text "Standardized Using a 1950-2020 baseline" and this project's
  "match the paper's definition, don't re-standardize" default).

  WesternV_MAM(year) = area-weighted (cos-latitude) mean SST, March-April-May,
  pooled over all ocean pixels in the union of 3 boxes:
    110-140E, 15S-15N | 160E-160W (=160-200E), 20N-35N | 155E-160W (=155-200E), 15S-30S
  NINO3.4_MAM(year) = area-weighted mean SST, March-April-May, box
    170E-120W (=170-240E), 5S-5N

SIGN CONVENTION -- confirmed empirically, not just from the text: Busker's
header literally reads "difference between the Western V region and the
NINO3.4 region" (WV - NINO3.4 order), but that produces a WVG anti-
correlated with Busker's own staged values (r=-0.98 against their 103-year
series). NINO3.4 - WesternV reproduces their sign (r=+0.93). This project's
output uses the NINO3.4-minus-WesternV convention because it is the one
that actually matches the published numbers -- narrative phrasing in a
methods sentence is not a reliable sign source, matching the numbers is.
This also matches Funk et al.'s own narrative (WV warming -> WVG more
negative -> more frequent MAM drought).

FORMULA-VARIANT SEARCH (cheapest-to-most-expensive per CLAUDE.md Sec 3.5,
full numbers in docs/WVG_Pipeline_Documentation.md Sec 3): tested
area-weighted vs unweighted regional means (no material difference),
monthly-climatology anomaly vs MAM-seasonal-series baseline standardization
(no material difference), and difference-of-two-independently-standardized-
z-scores vs standardize-the-raw-difference (material difference -- the
former inflates variance ~1.4x relative to Busker's file; the latter
matches Busker's std almost exactly, 1.00 vs 1.04, and is what's
implemented here). Even after this search, an OLS-optimal rescaling of the
raw regional-difference series against Busker's own numbers tops out at
r=0.93 / R^2=0.87 -- there is a real, unresolved ~13% variance gap not
fixable by further rescaling, most likely from a sub-box weighting or
edge-inclusion convention in Funk et al.'s own (unavailable) source code
that this project's plain area-weighted-pixel-pooling can't recover
exactly. This is the DERIVED-INDEX VALIDATION LIMITATION flagged in
CLAUDE.md and the WVG doc -- weaker than every fetch-based pipeline in this
project, and weaker than the task's own "compare to a paper figure" floor,
since it's a full 103-year numeric comparison, just not an exact one.

CADENCE MISMATCH WITH IOD/MEI/NINO3.4 (confirmed with the project owner
2026-08-09, after checking Busker et al.'s own text: "we included the WVG
as observed during the MAM season... We assigned all SST indices to all
administrative units" -- i.e. Busker's own use is ALSO one value per year,
not a monthly series): WVG is fundamentally a MAM-seasonal, one-value-per-
year index -- there is no monthly variation to report, so this pipeline
follows the ANNUAL/GDP pattern (pipelines/imf_gdp/), not the monthly-lag
pattern used for IOD/MEI/NINO3.4:
  - wvg_undated: year Y's WVG value repeated across all 12 months of year Y
    (Busker-exact -- no real-time cutoff at all, RQ1 fidelity only).
  - wvg: RQ2+ default. Year Y's value is not visible until June of year Y
    (ERSSTv5 is a monthly product with ~1-month publish lag -- May's grid
    cell is typically available by early June, confirmed by this file's
    own "data_modified" attribute lagging its latest time step by about a
    month). Before June of year Y, still shows year Y-1's value.
  - wvg_lag1yr / wvg_lag2yr / wvg_lag3yr: prior years' WVG (built off the
    lag-safe annual series), giving the model visibility into recent WVG
    trajectory -- the annual analogue of IOD/MEI/NINO3.4's lag1/3/6-month
    columns, in the only unit that means anything for an index that
    updates once a year.
"""

import os

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_PATH = os.path.join(REPO_ROOT, "data", "raw", "wvg", "sst.mnmean.nc")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "wvg_monthly.csv")
ANNUAL_OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "wvg_annual_mam.csv")

WESTERN_V_BOXES = [  # (lon_lo, lon_hi, lat_lo, lat_hi), 0-360E / -90..90
    (110, 140, -15, 15),
    (160, 200, 20, 35),
    (155, 200, -30, -15),
]
NINO34_BOX = [(170, 240, -5, 5)]

BASELINE_START, BASELINE_END = 1950, 2020
MAM_MONTHS = [3, 4, 5]
FEATURE_START_YEAR = 2009  # matches the other feature pipelines' feature-year window
PUBLICATION_LAG_CUTOFF_MONTH = 6  # ERSSTv5 May grid available by ~June; year Y visible from June Y
LAG_YEARS = [1, 2, 3]


def _region_mask(lat, lon, boxes):
    combined = None
    for lon_lo, lon_hi, lat_lo, lat_hi in boxes:
        m = (lon >= lon_lo) & (lon <= lon_hi) & (lat >= lat_lo) & (lat <= lat_hi)
        combined = m if combined is None else (combined | m)
    return combined


def _region_mean_series(sst, lat, lon, coslat, boxes):
    """NaN-safe (land-masked), cos-latitude area-weighted regional mean SST
    per month, pooling all ocean pixels across every box in `boxes` (per
    CLAUDE.md Sec 3.3's weight-renormalization rule -- applied here to a
    box-region mean rather than a zonal raster mean, same principle)."""
    mask = _region_mask(lat, lon, boxes)
    weights_2d = (coslat * xr.ones_like(lon)).where(mask, 0.0)
    valid = sst.notnull() & mask
    weights_valid = xr.where(valid, weights_2d, 0.0)
    numerator = (sst.fillna(0) * weights_valid).sum(dim=["lat", "lon"])
    denominator = weights_valid.sum(dim=["lat", "lon"])
    return (numerator / denominator).to_series()


def compute_regional_monthly_means():
    with xr.open_dataset(RAW_PATH) as ds:
        sst = ds["sst"].load()
        lat, lon = sst["lat"], sst["lon"]
        coslat = np.cos(np.deg2rad(lat))

        wv = _region_mean_series(sst, lat, lon, coslat, WESTERN_V_BOXES)
        n34 = _region_mean_series(sst, lat, lon, coslat, NINO34_BOX)

    df = pd.DataFrame({"western_v_sst": wv, "nino34_sst": n34})
    df.index = pd.to_datetime(df.index)
    df["year"] = df.index.year
    df["month"] = df.index.month
    return df.reset_index(drop=True)


def compute_annual_wvg(monthly_regional):
    """MAM seasonal mean per region per year, differenced (NINO3.4 - Western
    V, see module docstring for the sign-convention evidence), then
    standardized on the fixed 1950-2020 baseline."""
    mam = (
        monthly_regional[monthly_regional["month"].isin(MAM_MONTHS)]
        .groupby("year")[["western_v_sst", "nino34_sst"]]
        .mean()
    )
    # A year needs all 3 of Mar/Apr/May present -- groupby.mean() silently
    # averages over however many are present, so check the count explicitly.
    mam_counts = (
        monthly_regional[monthly_regional["month"].isin(MAM_MONTHS)]
        .groupby("year")[["western_v_sst", "nino34_sst"]]
        .count()
    )
    complete_years = mam_counts[(mam_counts["western_v_sst"] == 3) & (mam_counts["nino34_sst"] == 3)].index
    mam = mam.loc[mam.index.isin(complete_years)]

    raw_diff = mam["nino34_sst"] - mam["western_v_sst"]
    baseline = raw_diff.loc[BASELINE_START:BASELINE_END]
    wvg = (raw_diff - baseline.mean()) / baseline.std()

    annual = pd.DataFrame({
        "year": wvg.index,
        "western_v_sst_mam": mam["western_v_sst"].values,
        "nino34_sst_mam": mam["nino34_sst"].values,
        "wvg": wvg.values,
    }).reset_index(drop=True)
    return annual


def known_year_asof(row_year, row_month):
    """Publication-lag rule: year Y's MAM value isn't visible until June of
    year Y itself (Busker's own text confirms it's used "as observed during
    the MAM season" -- there's no meaningful annual value before MAM has
    actually happened and been published)."""
    if row_month >= PUBLICATION_LAG_CUTOFF_MONTH:
        return row_year
    return row_year - 1


def build_monthly_panel(annual):
    wvg_by_year = annual.set_index("year")["wvg"]

    today = pd.Timestamp.now().normalize().replace(day=1)
    full_range = pd.date_range(f"{FEATURE_START_YEAR}-01-01", today, freq="MS")
    grid = pd.DataFrame({"date": full_range, "year": full_range.year, "month": full_range.month})

    # Undated (Busker-exact, RQ1-parity): each row's own calendar year, no lag.
    grid["wvg_undated"] = grid["year"].map(wvg_by_year)

    # Publication-lag-safe (RQ2+ default).
    known_year = grid.apply(lambda r: known_year_asof(r["year"], r["month"]), axis=1)
    grid["wvg"] = known_year.map(wvg_by_year)
    for lag in LAG_YEARS:
        grid[f"wvg_lag{lag}yr"] = (known_year - lag).map(wvg_by_year)

    cols = ["year", "month", "wvg", "wvg_undated"] + [f"wvg_lag{lag}yr" for lag in LAG_YEARS]
    return grid[cols].reset_index(drop=True)


def main():
    monthly_regional = compute_regional_monthly_means()
    annual = compute_annual_wvg(monthly_regional)

    os.makedirs(os.path.dirname(ANNUAL_OUTPUT_PATH), exist_ok=True)
    annual.to_csv(ANNUAL_OUTPUT_PATH, index=False)
    print(f"Wrote {len(annual)} annual MAM WVG rows ({annual['year'].min()}-{annual['year'].max()}) "
          f"to {ANNUAL_OUTPUT_PATH}")

    monthly = build_monthly_panel(annual)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    monthly.to_csv(OUTPUT_PATH, index=False)
    first, last = monthly.iloc[0], monthly.iloc[-1]
    n_nan = monthly["wvg"].isna().sum()
    print(f"Wrote {len(monthly)} monthly rows ({int(first['year'])}-{int(first['month']):02d} to "
          f"{int(last['year'])}-{int(last['month']):02d}) to {OUTPUT_PATH}")
    print(f"wvg (lag-safe) NaN rows: {n_nan} (expected only if FEATURE_START_YEAR predates the "
          f"first usable prior-year value)")


if __name__ == "__main__":
    main()
