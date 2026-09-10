"""Assemble Busker et al.'s `data_master` for all 213 Horn of Africa units.

Port of busker_scripts/Input_data_scripts/input_collector.py. His script loops
unit by unit and appends; this one builds one frame per source and merges, which
is materially faster and produces the same panel.

Output: data/interim/busker_baseline/data_master.csv
        index (county, country, time) monthly 2000-01 .. 2022-12, plus his
        36 numeric columns and `lhz`.

Four substitutions, where his script reads an intermediate he did not release:

  1. `chirps-v2_005_ALL.nc` -> rebuilt by compute_hoa_wetdays_dryspells.py from
     the cached global daily CHIRPS. Supplies tp/wd/ds. `tp` reproduces his own
     staged data_prec_Admin.xlsx at r = 1.0000 (median 0.00% difference over
     50,516 unit-months), which validates the whole grid -> unit chain.
  2. `SST_index.xlsx` sheets `WVG_processed`/`WPG_processed` -> parsed from the
     released `SST.xlsx` sheet `WVG` (header block, then year | MAM WVG |
     OND WPG), broadcast to monthly, then his .shift(1).ffill().
  3. `market_data_{county}_{country}_{good}.xlsx` -> derived from the released
     bulk WFP CSVs. See ETHIOPIA_WFP_UNITS for the one place this reproduction
     is inexact: his per-unit price extracts are not in the package, so the
     FEWS-style abbreviations in the bulk file had to be crosswalked by hand.
  4. `fews_xr.nc` (his script rasterizes the shapefiles when absent) -> the
     released FEWS/fews_xr_CS.nc, which already carries CS and HA.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import regionmask
import xarray as xr

from busker_paths import (
    CLIM_DIR,
    CONFLICT_DIR,
    COUNTRIES,
    COUNTRY_SPECS,
    DATA_MASTER_PATH,
    FEWS_DIR,
    GOODS,
    INTERIM_DIR,
    LHZ_PATH,
    LOCUST_DIR,
    NDVI_DIR,
    PANEL_END,
    PANEL_START,
    POP_DIR,
    PRICE_INDICATOR,
    SE_DIR,
    SST_DIR,
    WD_DS_PATH,
    WFP_DIR,
    load_all_units,
    load_units,
)

ACCUMULATIONS = [1, 3, 6, 12, 24]

# input_collector.py:681-682. His climate-index workbooks carry Kenya's former
# county names; the released shapefile already carries the current ones.
INDEX_RENAMES = {
    "Keiyo-Marakwet": "Elgeyo Marakwet",
    "Tharaka": "Tharaka-Nithi",
    "Murang'a": "Muranga",
}

# input_collector.py:371-375
ACLED_RENAMES = {
    "North Shewa": "North Shewa (AM)",
    "Kemashi": "Kamashi",
    "Kembata Tibaro": "Kembata Tembaro",
}

# Crosswalk from the bulk WFP file's FEWS-style abbreviations to the Ethiopia
# shapefile names. Kenya and Somalia need none -- their `Admin 2` values match
# the shapefile exactly. Ambiguous entries were resolved against the file's own
# `Admin 1` and `Market Name` columns, not guessed:
#   AA ZONE1 -> Addis Ababa / market "Addis Ababa"      -> Region 14
#   GODE     -> Somali / market "Gode"                  -> Shabelle
#   GODERE   -> Gambela / market "Meti"                 -> Majang
#   ZONE 1   -> Gambela / market "Gambela"              -> Agnewak
#   "Administrative unit not available" -> Amhara / market "Baher Dar" -> West Gojam
# Gambela "ZONE 2" (Punido) and "ZONE 3" (Kowerneng/Korgang) are deliberately
# left unmapped: no confident zone assignment, ~30 rows between them.
ETHIOPIA_WFP_UNITS = {
    "AA ZONE1": ["Region 14"],
    "AFDER": ["Afder"],
    "ALABA SW": ["Halaba"],
    "AMARO SW": ["Amaro"],
    "ARSI": ["Arsi"],
    "ASOSA": ["Asosa"],
    "Administrative unit not available": ["West Gojam"],
    "BALE": ["Bale"],
    "BORENA": ["Borena"],
    "C. TIGRAY": ["Central Tigray"],
    "DIRE DAWA": ["Dire Dawa urban"],
    "E. GOJAM": ["East Gojam"],
    "E. HARERGE": ["East Hararge"],
    "E. SHEWA": ["East Shewa"],
    "E. TIGRAY": ["Eastern Tigray"],
    "E. WELLEGA": ["East Wellega"],
    "GAMO GOFA": ["Gamo", "Gofa"],  # historical zone, since split
    "GEDEO": ["Gedeo"],
    "GODE": ["Shabelle"],
    "GODERE": ["Majang"],
    "GUJI": ["Guji"],
    "GURAGE": ["Guraghe"],
    "HADIYA": ["Hadiya"],
    "HARAR/HUNDENE": ["Harari"],
    "JIJIGA": ["Fafan"],  # Jijiga zone renamed Fafan
    "JIMMA": ["Jimma"],
    "KAMASHI": ["Kamashi"],
    "KONSO SW": ["Konso"],
    "KORAHE": ["Korahe"],
    "LIBEN": ["Liban"],
    "MEKELE": ["Mekelle Tigray"],
    "N. GONDER": ["North Gondar"],
    "N. SHEWA (R3)": ["North Shewa (AM)"],  # R3 = Amhara
    "N. SHEWA (R4)": ["North Shewa (OR)"],  # R4 = Oromia
    "N. WELLO": ["North Wello"],
    "NW. TIGRAY": ["North Western Tigray"],
    "OROMIYA": ["Oromia"],  # the Amhara-region Oromia zone
    "S. TIGRAY": ["Southern Tigray"],
    "S. WELLO": ["South Wello"],
    "S.W. SHEWA": ["South West Shewa"],
    "SELTI": ["Siltie"],
    "SIDAMA": ["Sidama"],
    "SOUTH OMO": ["South Omo"],
    "W. GOJAM": ["West Gojam"],
    "W. HAMRA": ["Wag Hamra"],
    "W. HARERGE": ["West Hararge"],
    "WELAYITA": ["Welayta"],
    "WEST SHEWA": ["West Shewa"],
    "ZONE 1": ["Agnewak"],  # Gambela
    "ZONE1": ["Awsi-Zone 1"],  # Afar
    "ZONE2": ["Kilbati-Zone 2"],
    "ZONE3": ["Gabi-Zone 3"],
    "ZONE4": ["Fanti-Zone 4"],
    "ZONE5": ["Hari-Zone 5"],
}


def month_index():
    return pd.period_range(PANEL_START, PANEL_END, freq="M")


def to_month(series):
    return pd.PeriodIndex(pd.to_datetime(series), freq="M")


# ---------------------------------------------------------------------------
# FEWS target -- input_collector.py:155-215, 219-231, 593-633
# ---------------------------------------------------------------------------

def load_fews(units):
    """Population-weighted mean CS and HA per unit."""
    fews = xr.open_dataset(FEWS_DIR / "fews_xr_CS.nc")
    # input_collector.py:210 -- FEWS maps carry sporadic very high values.
    fews = fews.where(fews < 10)

    lon = fews["lon"].values
    lat = fews["lat"].values

    pop = xr.open_dataset(POP_DIR / "pop_hoa.tif", engine="rasterio")
    pop = pop.rename({"x": "lon", "y": "lat"}).squeeze("band", drop=True)["band_data"]
    pop = pop.interp_like(fews, method="nearest")
    weights = np.nan_to_num(pop.values.ravel()).astype("float64")

    times = pd.PeriodIndex(pd.to_datetime(fews["time"].values), freq="M")
    cs = fews["CS"].values.reshape(len(times), -1).astype("float64")
    ha = fews["HA"].values.reshape(len(times), -1).astype("float64")

    frames = []
    for country in COUNTRIES:
        gdf = load_units(country)
        mask = regionmask.mask_geopandas(gdf, lon, lat).values.ravel()
        for objectid, row in gdf.iterrows():
            pix = np.flatnonzero(mask == objectid)
            if pix.size == 0:
                print(f"  WARNING: no FEWS pixel for {row['county']} ({country})")
                continue
            w = weights[pix]
            out = {}
            for name, arr in (("FEWS_CS", cs), ("FEWS_HA", ha)):
                vals = arr[:, pix]
                ok = np.isfinite(vals)
                wsum = (ok * w).sum(axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    out[name] = np.where(
                        wsum > 0, (np.nan_to_num(vals) * w).sum(axis=1) / wsum, np.nan
                    )
            frames.append(
                pd.DataFrame({"county": row["county"], "country": country,
                              "month": times, **out})
            )
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# CHIRPS-derived tp / wd / ds -- rebuilt upstream
# ---------------------------------------------------------------------------

def load_chirps(ds_path=None):
    ds_path = ds_path or WD_DS_PATH
    if not ds_path.exists():
        raise SystemExit(
            f"{ds_path} not found -- run compute_hoa_wetdays_dryspells.py first"
        )
    df = pd.read_csv(ds_path, parse_dates=["time"])
    df["month"] = to_month(df["time"])
    return df[["county", "country", "month", "tp", "wd", "ds"]]


# ---------------------------------------------------------------------------
# Standardised drought indices -- input_collector.py:688-739
# ---------------------------------------------------------------------------

def load_indices():
    """SPI/SPEI/SSMI at 1/3/6/12/24 months, from his own unit-level workbooks."""
    frames = []
    for name, folder, stem in (("SPI", "spi", "spi"), ("SPEI", "spei", "spei"),
                               ("SSMI", "ssmi", "ssmi")):
        for acc in ACCUMULATIONS:
            wide = pd.read_excel(CLIM_DIR / folder / f"{stem}_{acc}.xlsx",
                                 index_col=0, header=0)
            wide = wide.rename(columns=INDEX_RENAMES)
            # input_collector.py:701 -- his workbooks are month-end stamped.
            wide.index = pd.PeriodIndex(wide.index, freq="M")
            long = wide.stack(future_stack=True).rename(f"{name}_{acc}").reset_index()
            long.columns = ["month", "county", f"{name}_{acc}"]
            frames.append(long.set_index(["county", "month"]))
    return pd.concat(frames, axis=1).reset_index()


# ---------------------------------------------------------------------------
# NDVI anomalies -- input_collector.py:122-152, 746-769
# ---------------------------------------------------------------------------

def xr_anomaly(dataset, baseline_end_year=None):
    """input_collector.py:134-144. His literal code computes the climatology
    mean/std over whatever years are in `dataset` -- no upper-year bound is
    ever applied, so with the released `>=2000` filter and no ceiling, the
    baseline his own trained model actually used is 2000-2022 (confirmed
    against the file's real extent), not the paper's stated "2000-2021"
    (section 2.1.2.1) or even his own code comment's "2000-2020". Default
    (`baseline_end_year=None`) reproduces his actual code exactly. Passing
    `baseline_end_year=2021` instead computes the climatology from only
    2000-2021 and applies it to the full 2000-2022 series -- a test of
    whether matching the paper's literal text (rather than his code)
    reproduces the published numbers better. See CLAUDE.md section 7a.
    """
    baseline = (dataset if baseline_end_year is None
                else dataset.where(dataset.time.dt.year <= baseline_end_year, drop=True))
    climatology_mean = baseline.groupby("time.month").mean("time")
    climatology_std = baseline.groupby("time.month").std("time")
    return xr.apply_ufunc(
        lambda x, m, s: (x - m) / s,
        dataset.groupby("time.month"),
        climatology_mean,
        climatology_std,
    )


def load_ndvi(units, baseline_end_year=None):
    sources = {
        "NDVI_anom": "NDVI_NOA_STAR_1981_2022.nc",
        "NDVI_anom_range": "NDVI_rangemask.nc",
        "NDVI_anom_crop": "NDVI_cropmask.nc",
    }
    out = {}
    masks = None
    times = None
    for column, filename in sources.items():
        ds = xr.open_dataset(NDVI_DIR / filename).load()
        if "band" in ds.sizes:
            ds = ds.squeeze("band", drop=True)
        ds = ds.drop_vars("spatial_ref", errors="ignore")
        # input_collector.py:127-129 -- keep 2000 onwards only.
        ds = ds.where(ds.time.dt.year >= 2000).dropna(how="all", dim="time")

        anom = xr_anomaly(ds, baseline_end_year).drop_vars("month", errors="ignore")["NDVI"]

        if masks is None:
            lon = anom["longitude"].values
            lat = anom["latitude"].values
            masks = {c: regionmask.mask_geopandas(load_units(c), lon, lat).values.ravel()
                     for c in COUNTRIES}
            times = pd.PeriodIndex(pd.to_datetime(anom["time"].values), freq="M")

        values = anom.values.reshape(len(times), -1)
        for country in COUNTRIES:
            gdf = load_units(country)
            for objectid, row in gdf.iterrows():
                pix = np.flatnonzero(masks[country] == objectid)
                key = (row["county"], country)
                series = (np.full(len(times), np.nan) if pix.size == 0
                          else np.nanmean(values[:, pix], axis=1))
                out.setdefault(key, {})[column] = series

    frames = []
    for (county, country), cols in out.items():
        frames.append(pd.DataFrame({"county": county, "country": country,
                                    "month": times, **cols}))
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# SST indices -- input_collector.py:325-352
# ---------------------------------------------------------------------------

def load_sst(teleconnection_lag_months=None):
    """teleconnection_lag_months: leakage-ablation test only (report.md
    section 5b), not part of Busker's actual code -- his MEI/NINA34/IOD
    columns use final/revised values with no lag (unlike WVG/WPG, which
    already get .shift(1).ffill() below). None reproduces his real
    behavior exactly; pass an int to test what a publication-lag control
    on those three indices would have done instead."""
    months = month_index()
    out = pd.DataFrame(index=months)

    # His script reads pre-processed WVG_processed / WPG_processed sheets that
    # are not in the release. The raw `WVG` sheet holds a header block, then
    # year | MAM WVG | OND WPG.
    raw = pd.read_excel(SST_DIR / "SST.xlsx", sheet_name="WVG", header=None)
    numeric = raw[pd.to_numeric(raw[0], errors="coerce").notna()].copy()
    numeric.columns = ["year", "WVG", "WPG"]
    numeric["year"] = numeric["year"].astype(int)
    annual = numeric.set_index("year")[["WVG", "WPG"]].astype(float)

    for column in ("WVG", "WPG"):
        series = pd.Series(
            [annual[column].get(m.year, np.nan) for m in months], index=months
        )
        # input_collector.py:329-331 / 337-339: a MAM average cannot be known in
        # March, so shift one month forward, then forward-fill.
        out[column] = series.shift(1).ffill()

    for sheet in ("MEI", "NINA34", "IOD"):
        series = pd.read_excel(SST_DIR / "SST.xlsx", sheet_name=sheet,
                               index_col=0, header=0)
        series.index = pd.PeriodIndex(pd.to_datetime(series.index), freq="M")
        col = series.columns[0]
        # input_collector.py:344/348 -- -99.99 missing-data sentinel.
        values = series[col].where(series[col] > -10)
        values = values.reindex(months)
        if teleconnection_lag_months:
            values = values.shift(teleconnection_lag_months).ffill()
        out[sheet] = values

    return out.rename_axis("month").reset_index()


# ---------------------------------------------------------------------------
# WFP market prices -- input_collector.py:786-831
# ---------------------------------------------------------------------------

def _wfp_unit_column(country):
    # Kenya's `Admin 2` matches the county shapefile exactly (its `Admin 1`
    # holds the seven former provinces, which do not); Somalia's matches too.
    return "Admin 2"


def load_prices(units):
    frames = []
    for country in COUNTRIES:
        parts = []
        for kind in ("retail", "wholesale"):
            path = WFP_DIR / country / f"WFP_{country}_FoodPrices_new_tool_{kind}.csv"
            parts.append(pd.read_csv(path, sep=";", low_memory=False))
        df = pd.concat(parts, ignore_index=True)
        df = df[df["Commodity"].isin(GOODS)].copy()

        df["month"] = pd.PeriodIndex(
            pd.to_datetime(df["Price Date"], format="%d-%m-%Y", errors="coerce"),
            freq="M",
        )
        value = (df[PRICE_INDICATOR].astype(str).str.replace(",", ".", regex=False))
        df["value"] = pd.to_numeric(value, errors="coerce")
        df = df.dropna(subset=["month", "value"])

        unit_col = _wfp_unit_column(country)
        if country == "Ethiopia":
            df["units"] = df[unit_col].map(ETHIOPIA_WFP_UNITS)
            df = df.dropna(subset=["units"]).explode("units")
        else:
            df["units"] = df[unit_col]

        valid = set(units[units["country"] == country]["county"])
        df = df[df["units"].isin(valid)]

        # input_collector.py:829 -- average over duplicates at the same date
        # (retail vs wholesale, and multiple markets in a unit).
        grouped = (df.groupby(["units", "month", "Commodity"])["value"]
                     .mean().unstack("Commodity"))
        grouped.columns = [f"{c}_{PRICE_INDICATOR}" for c in grouped.columns]
        grouped = grouped.reset_index().rename(columns={"units": "county"})
        grouped["country"] = country
        frames.append(grouped)

    out = pd.concat(frames, ignore_index=True)
    for good in GOODS:
        col = f"{good}_{PRICE_INDICATOR}"
        if col not in out.columns:
            out[col] = np.nan
    return out


# ---------------------------------------------------------------------------
# ACLED -- input_collector.py:362-417, 844-858
# ---------------------------------------------------------------------------

def load_acled(units):
    path = next(CONFLICT_DIR.glob("acled*.csv"))
    df = pd.read_csv(path, index_col=0, low_memory=False)
    for old, new in ACLED_RENAMES.items():
        df.loc[df["admin2"] == old, "admin2"] = new

    frames = []
    for country in COUNTRIES:
        level = COUNTRY_SPECS[country]["admin_level"]
        sub = df[df["country"] == country].copy()
        sub["month"] = to_month(sub["event_date"])
        # input_collector.py:845 -- every event counts once.
        sub["acled_count"] = 1
        grouped = (sub.groupby([level, "month"])[["acled_count", "fatalities"]]
                      .sum().reset_index())
        grouped = grouped.rename(columns={level: "county",
                                          "fatalities": "acled_fatalities"})
        grouped["country"] = country
        frames.append(grouped)

    out = pd.concat(frames, ignore_index=True)
    # input_collector.py:852-856 -- a month with no event is a zero, not a gap,
    # and units absent from ACLED entirely are filled with zeros throughout.
    grid = units[["county", "country"]].merge(
        pd.DataFrame({"month": month_index()}), how="cross"
    )
    out = grid.merge(out, on=["county", "country", "month"], how="left")
    out[["acled_count", "acled_fatalities"]] = (
        out[["acled_count", "acled_fatalities"]].fillna(0)
    )
    return out


# ---------------------------------------------------------------------------
# Desert locust -- input_collector.py:423-424, 883-909
# ---------------------------------------------------------------------------

def load_locust(units):
    df = pd.read_excel(LOCUST_DIR / "desert_locust_dataset.xlsx", index_col=0)
    df.index = pd.PeriodIndex(pd.to_datetime(df.index), freq="M")
    df = df.rename_axis("month").reset_index()
    df["unit"] = df["unit"].replace(INDEX_RENAMES)

    # input_collector.py:897-904 -- affected hectares as a share of unit area,
    # computed in EPSG:3857 exactly as he does.
    area_ha = units.to_crs("EPSG:3857").area / 10_000
    areas = dict(zip(units["county"], area_ha))

    df = df[df["unit"].isin(areas)].copy()
    df["DL_area"] = df["AREAHA"] / df["unit"].map(areas) * 100
    grouped = df.groupby(["unit", "month"])["DL_area"].sum().reset_index()
    grouped = grouped.rename(columns={"unit": "county"})

    grid = units[["county", "country"]].merge(
        pd.DataFrame({"month": month_index()}), how="cross"
    )
    out = grid.merge(grouped, on=["county", "month"], how="left")
    # input_collector.py:890 -- reindexed with fill_value=0.
    out["DL_area"] = out["DL_area"].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# CPI and GDP -- input_collector.py:435-546
# ---------------------------------------------------------------------------

def _wb_inflation(sheet, country):
    df = pd.read_excel(SE_DIR / "Inflation_WB.xlsx", header=0, sheet_name=sheet)
    df = df[df["Country"] == country]
    date_cols = [c for c in df.columns
                 if isinstance(c, (int, np.integer)) and 190000 < int(c) < 210000]
    series = df[date_cols].T.iloc[:, 0]
    series.index = pd.PeriodIndex(
        pd.to_datetime(series.index.astype(int).astype(str), format="%Y%m"), freq="M"
    )
    return pd.to_numeric(series, errors="coerce")


def load_cpi(cpi_lag_months=None):
    """cpi_lag_months: leakage-ablation test only (report.md section 5b),
    not part of Busker's actual code -- his CPI is used as if each
    month's value were known instantly. None reproduces his real
    behavior; pass an int to test a publication-lag control instead."""
    out = {}
    for country in COUNTRIES:
        if country in ("Ethiopia", "Kenya"):
            headline = _wb_inflation("hcpi_m", country)
            food = _wb_inflation("fcpi_m", country)
        else:
            # input_collector.py:473-493
            som = pd.read_excel(SE_DIR / "CPI_SOM_data_portal.xlsx", header=6)
            som.iloc[:, 0] = som.iloc[:, 0].astype(str).str.replace("M", "", regex=False)
            som = som.set_index(som.columns[0])
            som.index = pd.PeriodIndex(
                pd.to_datetime(som.index, format="%Y%m", errors="coerce"), freq="M"
            )
            som = som[som.index.notna()]
            headline = pd.to_numeric(som.iloc[:, 0], errors="coerce")
            food = pd.to_numeric(som.iloc[:, 1], errors="coerce")
        months = month_index()
        headline = headline.reindex(months)
        food = food.reindex(months)
        if cpi_lag_months:
            headline = headline.shift(cpi_lag_months).ffill()
            food = food.shift(cpi_lag_months).ffill()
        out[country] = pd.DataFrame({"CPI_H": headline, "CPI_F": food}, index=months)
    return out


def load_gdp(gdp_lag_months=None):
    """gdp_lag_months: leakage-ablation test only (report.md section 5b),
    not part of Busker's actual code. None reproduces his real behavior
    (annual value forward-filled to monthly, visible instantly); pass an
    int to test a publication-lag control instead."""
    df = pd.read_excel(SE_DIR / "GDP_PER_CAPITA_IMF.xlsx", header=0)
    label = df.columns[0]
    out = {}
    for country in COUNTRIES:
        row = df[df[label] == country]
        if row.empty:
            raise SystemExit(f"{country} not found in GDP_PER_CAPITA_IMF.xlsx")
        series = row.iloc[0, 1:]
        series.index = pd.to_datetime(series.index.astype(str), format="%Y")
        series = pd.to_numeric(series.replace("no data", np.nan), errors="coerce")
        if gdp_lag_months:
            series = series.copy()
            series.index = series.index + pd.DateOffset(months=gdp_lag_months)
        # input_collector.py:522 -- annual value forward-filled to monthly.
        monthly = series.resample("MS").ffill()
        monthly.index = pd.PeriodIndex(monthly.index, freq="M")
        monthly = monthly.reindex(month_index()).rename("GDP")
        out[country] = monthly
    return out


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def impute_per_unit(df, columns):
    """input_collector.py:923-932 -- linear interpolation within each unit,
    then that unit's own mean for anything still missing."""
    out = df.sort_values(["county", "country", "month"]).copy()
    grouped = out.groupby(["county", "country"], sort=False)
    for column in columns:
        out[column] = grouped[column].transform(
            lambda s: s.interpolate(method="linear").fillna(s.mean())
        )
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-impute", action="store_true",
                        help="write the raw panel without his interpolation/mean fill")
    parser.add_argument("--ndvi-baseline-end-year", type=int, default=None,
                        help="cap the NDVI anomaly climatology at this year "
                             "(default: no cap, matching his actual code). "
                             "Pass 2021 to test the paper's literal stated "
                             "reference period instead -- see CLAUDE.md §7a.")
    parser.add_argument("--ds-path", type=str, default=None,
                        help="override the wet-day/dry-spell CSV read "
                             "(default: WD_DS_PATH)")
    parser.add_argument("--out-path", type=str, default=None,
                        help="override DATA_MASTER_PATH")
    parser.add_argument("--gdp-publication-lag-months", type=int, default=None,
                        help="leakage-ablation test (report.md §5b), not part of "
                             "Busker's actual code -- default None matches his real "
                             "instant-visibility behavior")
    parser.add_argument("--cpi-publication-lag-months", type=int, default=None,
                        help="leakage-ablation test (report.md §5b), see --gdp-publication-lag-months")
    parser.add_argument("--teleconnection-lag-months", type=int, default=None,
                        help="leakage-ablation test (report.md §5b) -- lag applied to "
                             "MEI/NINA34/IOD only (WVG/WPG already get his own .shift(1))")
    args = parser.parse_args(argv)

    out_path = Path(args.out_path) if args.out_path else DATA_MASTER_PATH
    ds_path = Path(args.ds_path) if args.ds_path else None

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    units = load_all_units()
    print(f"units: {len(units)} "
          f"({', '.join(f'{c} {(units.country == c).sum()}' for c in COUNTRIES)})")

    grid = units[["county", "country"]].merge(
        pd.DataFrame({"month": month_index()}), how="cross"
    )
    master = grid

    print("  FEWS CS/HA ...")
    master = master.merge(load_fews(units), on=["county", "country", "month"], how="left")

    print("  CHIRPS tp/wd/ds ...")
    master = master.merge(load_chirps(ds_path), on=["county", "country", "month"], how="left")

    print("  SPI/SPEI/SSMI ...")
    master = master.merge(load_indices(), on=["county", "month"], how="left")

    print("  NDVI anomalies ...")
    master = master.merge(load_ndvi(units, args.ndvi_baseline_end_year),
                          on=["county", "country", "month"], how="left")

    print("  SST indices ...")
    master = master.merge(load_sst(args.teleconnection_lag_months), on=["month"], how="left")

    print("  WFP prices ...")
    master = master.merge(load_prices(units), on=["county", "country", "month"],
                          how="left")

    print("  ACLED ...")
    master = master.merge(load_acled(units), on=["county", "country", "month"],
                          how="left")

    print("  desert locust ...")
    master = master.merge(load_locust(units), on=["county", "country", "month"],
                          how="left")

    print("  CPI / GDP ...")
    cpi = load_cpi(args.cpi_publication_lag_months)
    gdp = load_gdp(args.gdp_publication_lag_months)
    master = master.set_index(["country", "month"])
    for column, table in (("CPI_H", cpi), ("CPI_F", cpi)):
        master[column] = [table[c].loc[m, column] for c, m in master.index]
    master["GDP"] = [gdp[c].loc[m] for c, m in master.index]
    master = master.reset_index()

    print("  livelihood zones ...")
    lhz = pd.read_excel(LHZ_PATH, index_col=0)["max"]
    master["lhz"] = master["county"].map(lhz)
    missing_lhz = master.loc[master["lhz"].isna(), "county"].nunique()
    assert missing_lhz == 0, f"{missing_lhz} units have no livelihood zone"

    numeric = [c for c in master.columns
               if c not in ("county", "country", "month", "lhz")]
    print(f"\nraw panel: {len(master):,} rows, {len(numeric)} numeric columns")
    coverage = master[numeric].notna().mean().sort_values()
    print("  lowest coverage before imputation:")
    print(coverage.head(8).mul(100).round(1).to_string())

    if not args.skip_impute:
        print("\n  imputing (linear interpolation, then unit mean) ...")
        impute_cols = [c for c in numeric if c != "FEWS_CS"]
        master = impute_per_unit(master, impute_cols)
        still_missing = master[impute_cols].isna().mean()
        still_missing = still_missing[still_missing > 0]
        if len(still_missing):
            print("  columns still holding NaN after his rule "
                  "(no value anywhere in that unit):")
            print(still_missing.mul(100).round(1).to_string())
        else:
            print("  no NaN remains outside FEWS_CS")

    master = master.sort_values(["country", "county", "month"])
    master.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")
    print(f"  {len(master):,} rows x {len(master.columns)} columns, "
          f"{master['county'].nunique()} units, "
          f"{master['month'].min()} .. {master['month'].max()}")
    print(f"  FEWS_CS observed on {master['FEWS_CS'].notna().sum():,} rows "
          f"({master.loc[master['FEWS_CS'].notna(), 'month'].nunique()} distinct months)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
