"""
validate_persons_of_concern_output.py

Validates data/processed/UNHCR/persons_of_concern_clean.parquet, per
CLAUDE.md Sec 3.5. There is no independent third-party reference to diff
against here (UNHCR's own Refugee Data Finder is already the authoritative
source this pipeline reads from, unlike e.g. CHIRPS/NDVI which are checked
against Busker et al.'s published values) -- this is a structural /
business-rule validation only:

1. Schema: exactly the expected columns, in the expected dtypes
   (year=int, country_of_origin_iso3=category, refugees/asylum_seekers/
   idps=int).
2. No duplicate (year, country_of_origin_iso3) rows.
3. No nulls in any column.
4. Every country_of_origin_iso3 maps to exactly one country_of_origin name.
5. Sorted by year, then country_of_origin_iso3.
6. Sanity check: Country of Asylum was Ethiopia for 100% of raw rows
   (confirmed at build time, re-checked here against the raw file so a
   future re-fetch that changes scope doesn't silently invalidate the
   module docstring's stated scope).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_persons_of_concern import (  # noqa: E402
    INT_COLUMNS,
    KEY_COLUMNS,
    OUTPUT_COLUMNS,
    OUTPUT_PARQUET,
    RAW_PATH,
    validate_output,
)


def main() -> None:
    if not OUTPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"{OUTPUT_PARQUET} not found -- run "
            "`python -m pipelines.unhcr_displacement.prepare_persons_of_concern` first."
        )

    df = pd.read_parquet(OUTPUT_PARQUET)
    print(f"Loaded {len(df)} rows from {OUTPUT_PARQUET}")

    missing_cols = set(OUTPUT_COLUMNS) - set(df.columns)
    assert not missing_cols, f"Missing expected columns: {missing_cols}"
    assert list(df.columns) == OUTPUT_COLUMNS, (
        f"Column order mismatch: got {list(df.columns)}, expected {OUTPUT_COLUMNS}"
    )

    assert pd.api.types.is_integer_dtype(df["year"]), "year is not an integer dtype"
    for col in INT_COLUMNS:
        assert pd.api.types.is_integer_dtype(df[col]), f"{col} is not an integer dtype"
    assert isinstance(df["country_of_origin_iso3"].dtype, pd.CategoricalDtype), (
        "country_of_origin_iso3 did not survive the parquet round-trip as 'category'"
    )
    print("Schema/dtype checks: PASS")

    validate_output(df)
    print("No-duplicate-keys / no-nulls / ISO3-to-name-mapping checks: PASS")

    expected_sort = df.sort_values(KEY_COLUMNS).reset_index(drop=True)
    assert df.reset_index(drop=True).equals(expected_sort), (
        f"Output is not sorted by {KEY_COLUMNS}"
    )
    print(f"Sort order ({', '.join(KEY_COLUMNS)}): PASS")

    raw = pd.read_csv(RAW_PATH)
    asylum_countries = raw["Country of Asylum"].unique()
    if list(asylum_countries) != ["Ethiopia"]:
        print(
            f"WARNING: raw file's Country of Asylum is no longer Ethiopia-only "
            f"({asylum_countries}) -- the module docstring's stated scope "
            "('Ethiopia hosts, by origin country') needs re-checking."
        )
    else:
        print("Country of Asylum == Ethiopia for 100% of raw rows: PASS")

    print("\nAll validation checks passed.")


if __name__ == "__main__":
    main()
