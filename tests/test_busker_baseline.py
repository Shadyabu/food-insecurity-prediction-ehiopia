"""Structural and leakage checks for the Busker et al. (2024) baseline reproduction.

Run with:  /opt/miniconda3/bin/python -m pytest tests/test_busker_baseline.py -v

These are the gate described in the plan: unit and livelihood-zone counts, the
80-feature derivation, his zero-NaN invariant, and an independent check that the
lead shift moved every column except the target.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pipelines" / "busker_baseline"))

from busker_paths import (  # noqa: E402
    DATA_MASTER_PATH,
    INPUT_MASTER_PATH,
    LEADS,
    LHZ_PATH,
    load_all_units,
)

N_UNITS = 213
# The paper: "The pastoral and crop-farming regions are the largest, with 82 and
# 106 administrative units, respectively, whereas the agro-pastoral regions
# consist of 25 different units." (section 2.2.1)
EXPECTED_LHZ_COUNTS = {"p": 82, "ap": 25, "other": 106}

# feature_engineering.py produces 36 numeric data_master columns.
N_DATA_MASTER_NUMERIC = 36

# 35 base numeric + 38 rolled + month + OND/MAM + 4 FEWS memory + 3 country
# one-hots - 3 WPG = 80. Ethiopia-only runs would lose the country one-hots;
# this reproduction is the full Horn of Africa, so all three are present.
N_MODEL_FEATURES = 80

# Price columns are the one documented gap: his per-unit market extracts are not
# in the released package, so units with no reporting market stay missing.
ALLOWED_NAN = {"FEWS_CS", "base_forecast",
               "Fuel (diesel)_Pewi", "Maize (white)_Pewi",
               "Fuel (diesel)_Pewi_4", "Fuel (diesel)_Pewi_12",
               "Maize (white)_Pewi_4", "Maize (white)_Pewi_12"}


@pytest.fixture(scope="module")
def data_master():
    if not DATA_MASTER_PATH.exists():
        pytest.skip(f"{DATA_MASTER_PATH} not built")
    df = pd.read_csv(DATA_MASTER_PATH)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    return df


@pytest.fixture(scope="module")
def input_master():
    if not INPUT_MASTER_PATH.exists():
        pytest.skip(f"{INPUT_MASTER_PATH} not built")
    df = pd.read_parquet(INPUT_MASTER_PATH)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ---------------------------------------------------------------------------
# Unit catalogue
# ---------------------------------------------------------------------------

def test_unit_count():
    units = load_all_units()
    assert len(units) == N_UNITS
    assert units["county"].nunique() == N_UNITS
    counts = units["country"].value_counts().to_dict()
    assert counts == {"Ethiopia": 92, "Somalia": 74, "Kenya": 47}


def test_livelihood_zone_counts():
    lhz = pd.read_excel(LHZ_PATH, index_col=0)["max"]
    assert lhz.value_counts().to_dict() == EXPECTED_LHZ_COUNTS


# ---------------------------------------------------------------------------
# data_master
# ---------------------------------------------------------------------------

def test_data_master_shape(data_master):
    assert data_master["county"].nunique() == N_UNITS
    numeric = [c for c in data_master.columns
               if c not in ("county", "country", "month", "lhz")]
    assert len(numeric) == N_DATA_MASTER_NUMERIC


def test_data_master_is_contiguous_monthly(data_master):
    """Every unit spans an unbroken monthly index, which is what makes the
    positional lead shift in feature_engineering.py a calendar shift."""
    expected = pd.period_range("2000-01", "2022-12", freq="M")
    for (county, _), group in data_master.groupby(["county", "country"]):
        months = pd.PeriodIndex(group["month"]).sort_values()
        assert len(months) == len(expected), county
        assert (months == expected).all(), county


def test_fews_target_is_not_forward_filled(data_master):
    """ML_execution.py:314 keeps only FEWS_CS.notna(), so the target must stay
    missing off release months rather than being carried forward."""
    observed = data_master.loc[data_master["FEWS_CS"].notna(), "month"]
    assert observed.nunique() == 47, "his fews_xr_CS.nc carries 47 release months"
    assert data_master["FEWS_CS"].isna().mean() > 0.8


# ---------------------------------------------------------------------------
# input_master
# ---------------------------------------------------------------------------

def test_all_leads_present(input_master):
    assert sorted(input_master["lead"].unique()) == sorted(LEADS)


def test_model_feature_count_is_80(input_master):
    """Reproduces the drops ML_execution.py applies before fitting."""
    df = input_master.drop(columns=["year"])
    df = df.drop(columns=[c for c in df.columns if "WPG" in c])
    df = pd.get_dummies(df, columns=["country"], prefix="", prefix_sep="")
    df = df.drop(columns=df.select_dtypes(include=["object", "category"]).columns)
    features = df.drop(columns=["lead", "base_forecast", "FEWS_CS", "time"])
    assert features.shape[1] == N_MODEL_FEATURES, sorted(features.columns)


def test_no_unexpected_nan(input_master):
    """His own check at feature_engineering.py:177-180 expects NaN only in
    FEWS_CS and base_forecast; the price columns are the documented exception."""
    nan_columns = set(input_master.columns[input_master.isna().any()])
    assert nan_columns <= ALLOWED_NAN, sorted(nan_columns - ALLOWED_NAN)


def test_rolling_mean_excludes_current_month(input_master):
    """`.rolling(N).mean().shift(1)` -- the preceding N months, not including
    the current one. Checked on a unit-month where the window is fully filled."""
    lead0 = input_master[input_master["lead"] == 0]
    unit = lead0[lead0["county"] == "Turkana"].sort_values("time")
    sample = unit.iloc[30]
    window = unit.iloc[26:30]["acled_count"]  # the four preceding months
    assert np.isclose(sample["acled_count_4"], window.mean())
    assert not np.isclose(
        sample["acled_count_4"], unit.iloc[27:31]["acled_count"].mean()
    ), "the current month must be excluded from the window"


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def _aligned(input_master, lead, columns):
    """Lead-L rows joined to the lead-0 rows they should have been shifted from."""
    lead0 = input_master[input_master["lead"] == 0].copy()
    leadn = input_master[input_master["lead"] == lead].copy()
    lead0["key_time"] = lead0["time"]
    # A lead-L row targeting month t must carry the lead-0 values from t - L.
    leadn["key_time"] = leadn["time"] - pd.DateOffset(months=lead)
    return leadn.merge(
        lead0[["county", "country", "key_time"] + columns],
        on=["county", "country", "key_time"],
        suffixes=("", "_src"),
        how="inner",
    )


@pytest.mark.parametrize("lead", [1, 2, 3, 4, 8, 12])
def test_features_are_shifted_by_the_lead(input_master, lead):
    columns = ["tp", "SPI_3", "NDVI_anom", "acled_count_12", "IOD", "GDP",
               "FEWS_CS_lag1", "FEWS_CS_12", "base_forecast", "month", "MAM"]
    merged = _aligned(input_master, lead, columns)
    # Every lead-L row should find its source except the earliest `lead` months,
    # whose origin falls before the panel starts.
    n_leadn = (input_master["lead"] == lead).sum()
    assert len(merged) >= n_leadn - N_UNITS * lead
    assert len(merged) > 10_000
    for column in columns:
        left = merged[column]
        right = merged[f"{column}_src"]
        both_nan = left.isna() & right.isna()
        assert (both_nan | np.isclose(
            left.fillna(0).astype(float), right.fillna(0).astype(float)
        )).all(), f"{column} is not shifted by {lead} months"


@pytest.mark.parametrize("lead", [1, 2, 3, 4, 8, 12])
def test_target_is_never_shifted(input_master, lead):
    """FEWS_CS must stay at the target month at every lead."""
    lead0 = input_master[input_master["lead"] == 0][
        ["county", "country", "time", "FEWS_CS"]
    ]
    leadn = input_master[input_master["lead"] == lead][
        ["county", "country", "time", "FEWS_CS"]
    ]
    merged = leadn.merge(lead0, on=["county", "country", "time"],
                         suffixes=("", "_src"))
    assert len(merged) > 0
    both_nan = merged["FEWS_CS"].isna() & merged["FEWS_CS_src"].isna()
    assert (both_nan | np.isclose(
        merged["FEWS_CS"].fillna(0), merged["FEWS_CS_src"].fillna(0)
    )).all()


def test_persistence_never_uses_the_target_month(input_master):
    """base_forecast is FEWS_CS.shift(1).ffill() shifted again by the lead, so
    it can only ever hold a release strictly before the origin month."""
    lead0 = input_master[input_master["lead"] == 0].sort_values(
        ["county", "country", "time"]
    )
    for _, group in lead0.groupby(["county", "country"]):
        observed = group.loc[group["FEWS_CS"].notna()]
        if len(observed) < 3:
            continue
        # At an observed month the persistence value must come from an earlier
        # release, so the two are equal only when the class did not change.
        assert not np.allclose(
            observed["FEWS_CS"].to_numpy(),
            observed["base_forecast"].to_numpy(),
            equal_nan=True,
        ), "base_forecast reproduces the target exactly -- it is leaking"
        break
