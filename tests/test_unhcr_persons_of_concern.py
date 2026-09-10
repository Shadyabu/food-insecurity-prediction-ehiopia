"""Structural checks for the UNHCR persons-of-concern cleaning pipeline.

Run with:  /opt/miniconda3/bin/python -m pytest tests/test_unhcr_persons_of_concern.py -v
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "unhcr_displacement"))

from prepare_persons_of_concern import (  # noqa: E402
    INT_COLUMNS,
    KEY_COLUMNS,
    OUTPUT_COLUMNS,
    prepare_persons_of_concern,
    validate_iso3_to_name_mapping,
    validate_no_duplicate_keys,
    validate_no_nulls,
)


@pytest.fixture(scope="module")
def clean_df():
    return prepare_persons_of_concern(write_output=False)


def test_schema(clean_df):
    assert list(clean_df.columns) == OUTPUT_COLUMNS


def test_dtypes(clean_df):
    assert pd.api.types.is_integer_dtype(clean_df["year"])
    for col in INT_COLUMNS:
        assert pd.api.types.is_integer_dtype(clean_df[col])
    assert isinstance(clean_df["country_of_origin_iso3"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_object_dtype(clean_df["country_of_origin"]) or pd.api.types.is_string_dtype(
        clean_df["country_of_origin"]
    )


def test_no_duplicate_keys(clean_df):
    assert not clean_df.duplicated(subset=KEY_COLUMNS).any()
    # also confirm the validator itself catches an injected duplicate
    dupe = pd.concat([clean_df, clean_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_no_duplicate_keys(dupe)


def test_no_nulls(clean_df):
    assert not clean_df[OUTPUT_COLUMNS].isnull().any().any()
    with_null = clean_df.copy()
    with_null.loc[0, "refugees"] = None
    with pytest.raises(ValueError):
        validate_no_nulls(with_null)


def test_iso3_maps_to_exactly_one_country_name(clean_df):
    counts = clean_df.groupby("country_of_origin_iso3", observed=True)["country_of_origin"].nunique()
    assert (counts == 1).all()

    conflicting = clean_df.copy()
    conflicting.loc[0, "country_of_origin"] = "___not_a_real_country___"
    with pytest.raises(ValueError):
        validate_iso3_to_name_mapping(conflicting)


def test_sorted_by_year_then_iso3(clean_df):
    expected = clean_df.sort_values(KEY_COLUMNS).reset_index(drop=True)
    pd.testing.assert_frame_equal(clean_df.reset_index(drop=True), expected)


def test_idempotent(clean_df):
    second_run = prepare_persons_of_concern(write_output=False)
    pd.testing.assert_frame_equal(clean_df, second_run)


def test_ethiopia_row_present_as_idp_figure(clean_df):
    # Country of Asylum == Ethiopia for every raw row (checked at build time,
    # re-verified by validate_persons_of_concern_output.py); the
    # country_of_origin == "Ethiopia" rows are therefore Ethiopia's own IDP
    # count, per UNHCR convention (origin == asylum country).
    eth_rows = clean_df[clean_df["country_of_origin"] == "Ethiopia"]
    assert len(eth_rows) > 0
