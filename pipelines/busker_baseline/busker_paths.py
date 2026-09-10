"""Shared paths and unit catalogue for the Busker et al. (2024) baseline reproduction.

This reproduction runs on Busker's own 213-unit Horn of Africa catalogue (Kenya
admin-1, Somalia and Ethiopia admin-2), NOT on this project's fixed 92-zone
Ethiopia admin2 key. That is deliberate and self-contained: nothing here ever
joins to the Ethiopia feature panel, so CLAUDE.md section 3.1 is untouched.

Source package: Busker et al.'s Zenodo replication set, unzipped at BUSKER_DATA.
A byte-identical copy is cached at data/raw/busker_replication/input_data.zip.
"""

from pathlib import Path

import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parents[2]

# Busker's unzipped replication package, now vendored into the repo root.
# Falls back to the original Downloads location if the in-repo copy is absent.
BUSKER_DATA = REPO_ROOT / "busker_dataset"
if not BUSKER_DATA.exists():
    BUSKER_DATA = Path("/Users/shadyabushady/Downloads/input_data/busker_dataset")

VECTOR_DIR = BUSKER_DATA / "Vector"
CLIM_DIR = BUSKER_DATA / "Climate_indices"
FEWS_DIR = BUSKER_DATA / "FEWS"
NDVI_DIR = BUSKER_DATA / "NDVI"
POP_DIR = BUSKER_DATA / "Population"
SST_DIR = BUSKER_DATA / "SST_indices"
SE_DIR = BUSKER_DATA / "Macro_economic_indicators"
LOCUST_DIR = BUSKER_DATA / "Locust"
CONFLICT_DIR = BUSKER_DATA / "Conflicts"
WFP_DIR = BUSKER_DATA / "WFP_price_data"
LHZ_PATH = BUSKER_DATA / "Livelihood_zones" / "livelihood_zones.xlsx"

# Daily CHIRPS is the one input his package omits (his scripts read a
# pre-built chirps-v2_005_ALL.nc). The global daily NetCDFs are already
# cached here from pipelines/chirps/.
CHIRPS_RAW_DIR = REPO_ROOT / "data" / "raw" / "chirps"

INTERIM_DIR = REPO_ROOT / "data" / "interim" / "busker_baseline"
RESULTS_DIR = REPO_ROOT / "experiments" / "results" / "busker_baseline"

WD_DS_PATH = INTERIM_DIR / "hoa_wetdays_dryspells_monthly.csv"
DATA_MASTER_PATH = INTERIM_DIR / "data_master.csv"
# His feature_engineering.py writes input_master.csv; parquet is used here
# because the stacked lead panel is ~411k rows x ~85 columns and round-trips
# through CSV both slowly and lossily for float precision.
INPUT_MASTER_PATH = INTERIM_DIR / "input_master.parquet"

# input_collector.py starts every unit's panel at 2000-01-01; the FEWS target
# runs 2009-07..2022-06, so 2022 is the last year that needs covering.
PANEL_START = "2000-01-01"
PANEL_END = "2022-12-31"

# busker_scripts/Input_data_scripts/input_collector.py:265-291
COUNTRY_SPECS = {
    "Kenya": {
        "shapefile": VECTOR_DIR / "Kenya" / "County.shp",
        "name_col": "COUNTY",
        "objectid_col": "OBJECTID",
        "admin_level": "admin1",
    },
    "Somalia": {
        "shapefile": VECTOR_DIR / "Somalia" / "Som_Admbnda_Adm2_UNDP.shp",
        "name_col": "admin2Name",
        "objectid_col": "OBJECTID_1",
        "admin_level": "admin2",
    },
    "Ethiopia": {
        "shapefile": VECTOR_DIR / "Ethiopia" / "eth_admbnda_adm2_csa_bofedb_2021.shp",
        "name_col": "ADM2_EN",
        "objectid_col": None,  # built from the row index, per input_collector.py:286-288
        "admin_level": "admin2",
    },
}

COUNTRIES = ["Kenya", "Somalia", "Ethiopia"]

# The two WFP commodities Busker uses, and the price indicator he reads.
# input_collector.py:44, 272/281/291
GOODS = ["Maize (white)", "Fuel (diesel)"]
PRICE_INDICATOR = "Pewi"

# busker_scripts/Input_data_scripts/feature_engineering.py:38
LEADS = [0, 1, 2, 3, 4, 8, 12]


def load_units(country):
    """Return a GeoDataFrame indexed by OBJECTID with a 'county' name column.

    Mirrors the per-country blocks in input_collector.py:265-291, including
    Ethiopia's OBJECTID being derived from the row index rather than a field.
    """
    spec = COUNTRY_SPECS[country]
    gdf = gpd.read_file(spec["shapefile"])
    gdf = gdf.rename(columns={spec["name_col"]: "county"})

    if spec["objectid_col"] is None:
        gdf = gdf.reset_index().rename(columns={"index": "OBJECTID"})
    else:
        gdf = gdf.rename(columns={spec["objectid_col"]: "OBJECTID"})

    return gdf.set_index("OBJECTID")


def load_all_units():
    """All 213 units as a single frame: OBJECTID, county, country, geometry."""
    import pandas as pd

    frames = []
    for country in COUNTRIES:
        gdf = load_units(country).reset_index()
        gdf["country"] = country
        frames.append(gdf[["OBJECTID", "county", "country", "geometry"]])
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
