"""
prepare_persons_of_concern.py

Cleaning stage for the UNHCR "persons of concern" displacement table
(data/raw/UNHCR/persons_of_concern.csv). This source does NOT follow the
admin2 zone-month Acquire->Aggregate->Engineer->Validate pattern used by
every other pipeline in this repo (CLAUDE.md Sec 3-6) -- UNHCR reports at
the country-of-asylum / country-of-origin level, not by Ethiopian zone, so
there is no admin2 spatial join to build and no zone_code column in the
output. Treat this as a standalone reference table, not a
data/processed/features/<source>_admin2_monthly.csv-shaped feature.

SCOPE OF THE RAW FILE (checked 2026-08-24): every one of the 326 rows has
Country of Asylum == Ethiopia -- this is UNHCR's count of people Ethiopia
HOSTS, broken out by their country of origin, not a table of Ethiopians
displaced abroad. Per UNHCR convention, the row(s) where
Country of Origin == "Ethiopia" (Country of Asylum == Country of Origin)
report Ethiopia's own IDP figure; every other row is a refugee/
asylum-seeker inflow from a third country (29 distinct origin countries,
2008-2025). `country_of_asylum` is dropped in the cleaned output because
it is constant (always Ethiopia) and carries no information.

NOT WIRED INTO pipelines/model_join/: this table has no zone_code, so
joining it onto the 92-admin2 model tables needs an explicit join-key
decision (e.g. national broadcast-by-year like cpi/imf_gdp, vs. keeping
country_of_origin as a per-row categorical context feature for a
differently-shaped model) that has not been made. Flag this before using
the output below inside pipelines/model_join/.

CATEGORICAL ENCODING CHOICE: country_of_origin_iso3 is cast to pandas'
native 'category' dtype, not one-hot or label-encoded. XGBoost's
tree_method="hist" with enable_categorical=True consumes pandas
categoricals directly (splitting on category subsets internally) -- more
memory-efficient than one-hot (29 origin countries would otherwise mean 29
near-zero-variance dummy columns) and avoids the ordinal bias a plain
label-encode would introduce (an arbitrary integer ordering would let a
tree threshold-split as if code 3 < code 17 meant something).

CSV CAVEAT: pandas' category dtype does not survive a CSV round-trip --
pd.read_csv() always brings country_of_origin_iso3 back as a plain
object/string column. This script writes BOTH a .parquet (category dtype
preserved -- use this for anything feeding straight into model training)
and a .csv (human inspection / diffing only). If you load the CSV output,
re-cast before training:

    df["country_of_origin_iso3"] = df["country_of_origin_iso3"].astype("category")

Run standalone:
    python -m pipelines.unhcr_displacement.prepare_persons_of_concern
Or import:
    from pipelines.unhcr_displacement.prepare_persons_of_concern import prepare_persons_of_concern
    df = prepare_persons_of_concern()
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = REPO_ROOT / "data" / "raw" / "UNHCR" / "persons_of_concern.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "UNHCR"
OUTPUT_CSV = OUTPUT_DIR / "persons_of_concern_clean.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "persons_of_concern_clean.parquet"

COLUMN_RENAME = {
    "Year": "year",
    "Country of Origin": "country_of_origin",
    "Country of Origin ISO": "country_of_origin_iso3",
    "Refugees": "refugees",
    "Asylum-seekers": "asylum_seekers",
    "IDPs": "idps",
}
OUTPUT_COLUMNS = list(COLUMN_RENAME.values())
KEY_COLUMNS = ["year", "country_of_origin_iso3"]
INT_COLUMNS = ["refugees", "asylum_seekers", "idps"]


def load_raw(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"UNHCR raw file not found: {raw_path}")
    return pd.read_csv(raw_path)


def validate_no_nulls(df: pd.DataFrame, columns: list[str] = OUTPUT_COLUMNS) -> None:
    null_counts = df[columns].isnull().sum()
    bad = null_counts[null_counts > 0]
    if not bad.empty:
        raise ValueError(f"Nulls found in key columns:\n{bad}")


def validate_no_duplicate_keys(df: pd.DataFrame, key_columns: list[str] = KEY_COLUMNS) -> None:
    dup_mask = df.duplicated(subset=key_columns, keep=False)
    if dup_mask.any():
        raise ValueError(
            f"Duplicate {tuple(key_columns)} rows found:\n{df.loc[dup_mask, key_columns]}"
        )


def validate_iso3_to_name_mapping(df: pd.DataFrame) -> None:
    counts = df.groupby("country_of_origin_iso3", observed=True)["country_of_origin"].nunique()
    bad = counts[counts > 1]
    if not bad.empty:
        raise ValueError(f"ISO3 codes mapping to more than one country name: {bad.to_dict()}")


def validate_output(df: pd.DataFrame) -> None:
    """Full validation suite -- also usable standalone against an already-loaded output."""
    validate_no_nulls(df)
    validate_no_duplicate_keys(df)
    validate_iso3_to_name_mapping(df)


def prepare_persons_of_concern(
    raw_path: Path = RAW_PATH,
    output_dir: Path = OUTPUT_DIR,
    write_output: bool = True,
) -> pd.DataFrame:
    raw = load_raw(raw_path)
    df = raw.rename(columns=COLUMN_RENAME)[OUTPUT_COLUMNS].copy()

    validate_no_nulls(df)

    df["year"] = df["year"].astype(int)
    for col in INT_COLUMNS:
        df[col] = df[col].astype(int)

    df = df.sort_values(KEY_COLUMNS).reset_index(drop=True)

    validate_no_duplicate_keys(df)
    validate_iso3_to_name_mapping(df)

    df["country_of_origin_iso3"] = df["country_of_origin_iso3"].astype("category")

    if write_output:
        output_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_dir / "persons_of_concern_clean.parquet", index=False)
        df.to_csv(output_dir / "persons_of_concern_clean.csv", index=False)

    return df


if __name__ == "__main__":
    result = prepare_persons_of_concern()
    print(f"Wrote {len(result)} rows to:\n  {OUTPUT_PARQUET}\n  {OUTPUT_CSV}")
    print(result.dtypes)
