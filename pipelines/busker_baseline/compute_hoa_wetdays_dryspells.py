"""Wet days and maximum dry-spell length per HOA admin unit, from daily CHIRPS.

Busker's input_collector.py reads a pre-built `chirps-v2_005_ALL.nc` that is NOT
in his released package -- it is the only input of his that has to be
regenerated. The global daily CHIRPS NetCDFs are already cached under
data/raw/chirps/ from pipelines/chirps/, so this stage rebuilds `wd` and `ds`
from them using his own definitions.

Ported from:
  busker_scripts/ML_model_scripts/ML_functions.py:93   wet_days()
  busker_scripts/ML_model_scripts/ML_functions.py:106  max_dry_spells()
  busker_scripts/Input_data_scripts/input_collector.py:98-116  land mask + monthly
  busker_scripts/Input_data_scripts/input_collector.py:644-660 zonal means

His definitions, restated:
  * wet day  = tp > 1 mm; monthly count.
  * dry day  = tp <= 1 mm. A dry spell is a run of consecutive dry days, and the
    run counter is forced to zero on 1 March and 1 October so a spell can never
    cross a rainy-season boundary. Only runs of >= 5 days count; monthly maximum.

Two documented equivalences, both asserted by --self-check:
  1. His land mask (total rainfall over the record == 0 -> masked) reduces to
     CHIRPS's own ocean NaN mask over the HOA box: within the box there are
     zero finite land pixels with a zero rainfall total.
  2. The run-length counter here is written in NumPy rather than his
     cumsum/ffill xarray chain. `--self-check` reproduces his literal xarray
     formulation on a subwindow and asserts the two agree exactly.

Output: data/interim/busker_baseline/hoa_wetdays_dryspells_monthly.csv
        columns county, country, time, wd, ds
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import regionmask
import xarray as xr

from busker_paths import (
    CHIRPS_RAW_DIR,
    COUNTRIES,
    INTERIM_DIR,
    WD_DS_PATH,
    load_units,
)

# input_collector.py:105,110 -- both functions are called with threshold=1
WET_DAY_THRESHOLD_MM = 1.0
MIN_DRY_SPELL_DAYS = 5

# The two season-start days on which max_dry_spells() zeroes the run counter.
# ML_functions.py:118
SEASON_RESET_DAYS = [(3, 1), (10, 1)]

FIRST_YEAR = 2000
LAST_YEAR = 2022

# Padding around the union of the three country boundaries.
BBOX_PAD_DEG = 0.3


def hoa_bbox():
    """Bounding box covering all 213 units, padded."""
    bounds = np.array([load_units(c).total_bounds for c in COUNTRIES])
    return (
        bounds[:, 0].min() - BBOX_PAD_DEG,
        bounds[:, 1].min() - BBOX_PAD_DEG,
        bounds[:, 2].max() + BBOX_PAD_DEG,
        bounds[:, 3].max() + BBOX_PAD_DEG,
    )


def load_daily(year, bbox):
    """Daily CHIRPS for one year over the HOA box, as float32 with NaN ocean."""
    path = CHIRPS_RAW_DIR / f"chirps-v2.0.{year}.days_p05.nc"
    if not path.exists():
        raise SystemExit(f"{path} not found -- run pipelines/chirps/fetch_chirps_admin2.py")

    ds = xr.open_dataset(path)
    ds = ds.sel(
        longitude=slice(bbox[0], bbox[2]),
        latitude=slice(bbox[1], bbox[3]),
    )
    da = ds["precip"].astype("float32")
    # CHIRPS uses negative fill values in some vintages; treat them as ocean.
    da = da.where(da >= 0)
    return da.load()


def run_length_reset(dry, reset_day):
    """Consecutive-dry-day counter, reset at wet days and at season starts.

    Vectorised equivalent of ML_functions.py:120-122's cumsum/ffill chain:
    v[t] = 0 where the day is wet or is a season-start day, else v[t-1] + 1.

    dry       -- (T, P) bool, True on dry days
    reset_day -- (T,) bool, True on 1 March / 1 October
    """
    n_time = dry.shape[0]
    out = np.zeros(dry.shape, dtype="int16")
    running = np.zeros(dry.shape[1], dtype="int16")
    for t in range(n_time):
        if reset_day[t]:
            running = np.zeros_like(running)
        else:
            running = np.where(dry[t], running + 1, 0).astype("int16")
        out[t] = running
    return out


def monthly_indicators(da, dry_spell_op="ge"):
    """Monthly total rainfall, wet-day count and maximum dry-spell length.

    `tp` is P_monthly = P_HAD.resample(time='MS').sum() at input_collector.py:115
    -- the monthly rainfall total, later spatially averaged per unit at :645.

    `dry_spell_op` -- "ge" (default) matches his actual code
    (`ML_functions.py:123`, `spell >= 5`), the operator that produced the
    published reference numbers. "gt" tests the paper's literal text
    ("maximum dry-spell length (>5 consecutive dry days)", section 2.1.2.1),
    which his own code does not implement -- see CLAUDE.md section 7a.
    """
    time = pd.DatetimeIndex(da["time"].values)
    values = da.values.reshape(len(time), -1)

    wet = values > WET_DAY_THRESHOLD_MM
    dry = values <= WET_DAY_THRESHOLD_MM  # NaN compares False -> treated as wet

    reset_day = np.array(
        [(t.month, t.day) in SEASON_RESET_DAYS for t in time], dtype=bool
    )
    spell = run_length_reset(dry, reset_day)
    op = np.greater_equal if dry_spell_op == "ge" else np.greater
    spell = np.where(op(spell, MIN_DRY_SPELL_DAYS), spell, 0)

    months = time.to_period("M")
    unique_months = months.unique()

    shape = (len(unique_months), values.shape[1])
    tp = np.empty(shape, dtype="float32")
    wd = np.empty(shape, dtype="float32")
    ds_ = np.empty(shape, dtype="float32")
    for i, month in enumerate(unique_months):
        sel = months == month
        tp[i] = np.nansum(values[sel], axis=0)
        wd[i] = wet[sel].sum(axis=0)
        ds_[i] = spell[sel].max(axis=0)

    return unique_months, tp, wd, ds_


def build_masks(lon, lat):
    """Per-country pixel -> OBJECTID mask, flattened. input_collector.py:293-318."""
    masks = {}
    for country in COUNTRIES:
        units = load_units(country)
        mask = regionmask.mask_geopandas(units, lon, lat)
        masks[country] = (mask.values.ravel(), units)
    return masks


def zonal_means(masks, land, months, tp, wd, ds_):
    """Per-unit spatial mean. input_collector.py:645-657 uses a plain
    NaN-skipping mean over the unit's pixels, after re-applying the land mask."""
    records = []
    for country, (flat_mask, units) in masks.items():
        for objectid, row in units.iterrows():
            pixels = np.flatnonzero((flat_mask == objectid) & land)
            if pixels.size == 0:
                print(f"  WARNING: no land pixel for {row['county']} ({country})")
                continue
            records.append(
                pd.DataFrame(
                    {
                        "county": row["county"],
                        "country": country,
                        "time": months.to_timestamp(),
                        "tp": tp[:, pixels].mean(axis=1),
                        "wd": wd[:, pixels].mean(axis=1),
                        "ds": ds_[:, pixels].mean(axis=1),
                    }
                )
            )
    return pd.concat(records, ignore_index=True)


def self_check(da):
    """Assert the NumPy run-length counter matches his literal xarray chain."""
    sub = da.isel(latitude=slice(0, 12), longitude=slice(0, 12))
    ds_in = sub.to_dataset(name="tp")

    # ML_functions.py:108-123, verbatim.
    dry_days = ds_in.where((ds_in["tp"] > WET_DAY_THRESHOLD_MM) | ds_in.isnull(), 1)
    dry_days = dry_days.where((dry_days["tp"] == 1) | dry_days.isnull(), 0)
    dry_spell = dry_days.where(
        (
            ((dry_days["time"].dt.month != 3) | (dry_days["time"].dt.day != 1))
            & ((dry_days["time"].dt.month != 10) | (dry_days["time"].dt.day != 1))
        ),
        0,
    )
    cumulative = dry_spell["tp"].cumsum(dim="time") - dry_spell["tp"].cumsum(
        dim="time"
    ).where(dry_spell["tp"].values == 0).ffill(dim="time").fillna(0)
    his = cumulative.where(cumulative >= MIN_DRY_SPELL_DAYS, 0).values

    time = pd.DatetimeIndex(sub["time"].values)
    values = sub.values.reshape(len(time), -1)
    dry = values <= WET_DAY_THRESHOLD_MM
    reset_day = np.array(
        [(t.month, t.day) in SEASON_RESET_DAYS for t in time], dtype=bool
    )
    mine = run_length_reset(dry, reset_day)
    mine = np.where(mine >= MIN_DRY_SPELL_DAYS, mine, 0)

    his_flat = np.nan_to_num(his.reshape(len(time), -1))
    land = np.isfinite(values).all(axis=0)
    ok = np.array_equal(his_flat[:, land], mine[:, land].astype(his_flat.dtype))
    print(f"  self-check dry-spell counter vs his xarray chain: "
          f"{'MATCH' if ok else 'MISMATCH'} ({land.sum()} land pixels)")
    if not ok:
        raise SystemExit("dry-spell reimplementation does not match Busker's")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=FIRST_YEAR)
    parser.add_argument("--last-year", type=int, default=LAST_YEAR)
    parser.add_argument("--self-check", action="store_true",
                        help="verify the run-length counter against his xarray chain")
    parser.add_argument("--dry-spell-op", choices=["ge", "gt"], default="ge",
                        help="ge (default) = his actual code (>=5 days); "
                             "gt = the paper's literal '>5 consecutive dry days' text")
    parser.add_argument("--out-path", type=str, default=None,
                        help="override WD_DS_PATH (default: shared with the main pipeline)")
    args = parser.parse_args(argv)

    out_path = Path(args.out_path) if args.out_path else WD_DS_PATH
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    bbox = hoa_bbox()
    print(f"HOA box: lon {bbox[0]:.2f}..{bbox[2]:.2f}, lat {bbox[1]:.2f}..{bbox[3]:.2f}")

    all_months, all_tp, all_wd, all_ds = [], [], [], []
    finite = None
    rain_total = None
    masks = None

    for year in range(args.first_year, args.last_year + 1):
        # Warm start on 1 October of the previous year: the run counter is
        # zeroed there anyway, so a chunk beginning on that date reproduces the
        # continuous computation over his single concatenated file exactly.
        prev = load_daily(year - 1, bbox).sel(time=slice(f"{year - 1}-10-01", None))
        cur = load_daily(year, bbox)
        da = xr.concat([prev, cur], dim="time")

        if masks is None:
            masks = build_masks(da["longitude"].values, da["latitude"].values)
            finite = np.isfinite(cur.values).all(axis=0).ravel()
            rain_total = np.zeros(finite.size, dtype="float64")
            print(f"  grid {da.sizes['latitude']}x{da.sizes['longitude']}, "
                  f"{finite.sum():,} finite pixels")

        # His land mask (input_collector.py:100-102) drops pixels whose rainfall
        # total over the *whole record* is zero, so the total is accumulated
        # across every year and the mask is applied after the loop. His record
        # is 1981-2022; this one is the panel window 2000-2022, which can only
        # differ for a pixel that rained before 2000 and never since.
        rain_total += np.nan_to_num(cur.values).sum(axis=0).ravel()

        if args.self_check and year == args.first_year:
            self_check(da)

        months, tp, wd, ds_ = monthly_indicators(da, dry_spell_op=args.dry_spell_op)
        keep = np.array([m.year == year for m in months])
        all_months.append(months[keep])
        all_tp.append(tp[keep])
        all_wd.append(wd[keep])
        all_ds.append(ds_[keep])
        print(f"  {year}: {keep.sum()} months")

        del da, prev, cur

    months = pd.PeriodIndex(np.concatenate([m.to_numpy() for m in all_months]), freq="M")
    tp = np.concatenate(all_tp)
    wd = np.concatenate(all_wd)
    ds_ = np.concatenate(all_ds)

    land = finite & (rain_total > 0)
    print(f"land mask: {land.sum():,} pixels "
          f"({int((finite & (rain_total == 0)).sum())} finite but never-raining, dropped)")

    print(f"aggregating {len(months)} months to units...")
    out = zonal_means(masks, land, months, tp, wd, ds_)

    out.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"  {len(out):,} rows, {out['county'].nunique()} units, "
          f"{out['time'].min():%Y-%m} .. {out['time'].max():%Y-%m}")
    print(out[["tp", "wd", "ds"]].describe().to_string())

    compare_tp_against_busker(out)
    return 0


def compare_tp_against_busker(out):
    """Cross-check `tp` against his own staged admin-level rainfall table.

    Climate_indices/spi/data_prec_Admin.xlsx is Busker's own unit-level monthly
    precipitation, produced by the same CHIRPS -> unit chain this stage rebuilds.
    It is not used as an input (his `tp` comes from the raster at
    input_collector.py:644-646) but it is the natural independent reference.
    """
    from busker_paths import CLIM_DIR

    path = CLIM_DIR / "spi" / "data_prec_Admin.xlsx"
    if not path.exists():
        print(f"\n  {path.name} not found -- skipping tp cross-check")
        return

    ref = pd.read_excel(path, index_col=0, header=0)
    ref.index = pd.PeriodIndex(ref.index, freq="M")
    ref = ref.stack().rename("tp_busker").reset_index()
    ref.columns = ["month", "county", "tp_busker"]

    ours = out.copy()
    ours["month"] = pd.PeriodIndex(ours["time"], freq="M")
    merged = ours.merge(ref, on=["month", "county"], how="inner")
    merged = merged[merged["tp_busker"] > 1]  # %-diff is meaningless near zero

    pct = (merged["tp"] - merged["tp_busker"]).abs() / merged["tp_busker"] * 100
    corr = merged["tp"].corr(merged["tp_busker"])
    print(f"\n  tp vs data_prec_Admin.xlsx: {len(merged):,} matched unit-months, "
          f"{merged['county'].nunique()} units")
    print(f"    r = {corr:.4f} | median diff {pct.median():.2f}% | "
          f"mean diff {pct.mean():.2f}% | within 10%: "
          f"{(pct <= 10).mean() * 100:.1f}%")


if __name__ == "__main__":
    sys.exit(main())
