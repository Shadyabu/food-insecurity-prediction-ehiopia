"""Busker et al.'s feature engineering, applied to data_master.

Port of busker_scripts/Input_data_scripts/feature_engineering.py. His script
loops unit by unit; this one uses groupby transforms, which is equivalent
because every operation he performs is already within-unit.

The four steps, and the details that matter:

  1. Rolling means (:60-73). `_4` and `_12` are `.rolling(N).mean().shift(1)` --
     the *preceding* N months, current month excluded -- and each is then filled
     with its own per-unit mean. They are applied to every column EXCEPT
     `do_nothing_cols`: FEWS_CS, tp, and all fifteen SPI/SPEI/SSMI columns are
     deliberately not rolled (they are already accumulations). 19 columns are.
  2. Season flags (:77-83). `month`/`year` plus boolean OND and MAM.
  3. FEWS memory (:89-109). `base_ini = FEWS_CS.shift(1).ffill()`, then lag1 =
     base_ini, lag4 = base_ini.shift(3), lag8 = base_ini.shift(7), and
     FEWS_CS_12 = base_ini.rolling(12).mean(). Every one is filled with the mean
     of *base_ini*, not with its own mean.
  4. Lead times (:120-145). For each lead, every column EXCEPT FEWS_CS is
     shifted back by the lead and the first `lead` rows are dropped -- so
     `month`, `OND`, `MAM` and `FEWS_HA` all shift too, and the model sees the
     origin month's calendar position rather than the target's. `base_ini` is
     renamed `base_forecast` and becomes the persistence benchmark.

Finally (:166-170) the NDVI cropland and rangeland columns are filled with the
dataset-wide mean, which is how units holding no cropland or rangeland pixels
are handled.

Output: data/interim/busker_baseline/input_master.parquet
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from busker_paths import (
    DATA_MASTER_PATH,
    INPUT_MASTER_PATH,
    INTERIM_DIR,
    LEADS,
)

UNIT_KEYS = ["county", "country"]

# feature_engineering.py:63, expanded.
DO_NOTHING_COLS = (
    ["FEWS_CS", "tp", "lhz", "county", "country"]
    + [f"SPI_{a}" for a in (1, 3, 6, 12, 24)]
    + [f"SPEI_{a}" for a in (1, 3, 6, 12, 24)]
    + [f"SSMI_{a}" for a in (1, 3, 6, 12, 24)]
)

ROLL_WINDOWS = [4, 12]


def add_rolling_means(df, columns):
    """feature_engineering.py:66-71."""
    grouped = df.groupby(UNIT_KEYS, sort=False)
    new = {}
    for column in columns:
        for window in ROLL_WINDOWS:
            rolled = grouped[column].transform(
                lambda s, w=window: s.rolling(window=w).mean().shift(1)
            )
            name = f"{column}_{window}"
            # Filled with the per-unit mean of the rolled column itself.
            new[name] = rolled.groupby(
                [df["county"], df["country"]], sort=False
            ).transform(lambda s: s.fillna(s.mean()))
    return df.assign(**new)


def add_season_flags(df):
    """feature_engineering.py:77-83."""
    month = df["month"].dt.month
    return df.assign(
        month=month,
        year=df["month"].dt.year,
        OND=month.isin([10, 11, 12]),
        MAM=month.isin([3, 4, 5]),
    )


def add_fews_memory(df):
    """feature_engineering.py:89-109."""
    grouped = df.groupby(UNIT_KEYS, sort=False)
    base = grouped["FEWS_CS"].transform(lambda s: s.shift(1).ffill())
    df = df.assign(base_ini=base)

    grouped = df.groupby(UNIT_KEYS, sort=False)
    derived = {
        "FEWS_CS_lag1": grouped["base_ini"].transform(lambda s: s),
        "FEWS_CS_lag4": grouped["base_ini"].transform(lambda s: s.shift(3)),
        "FEWS_CS_lag8": grouped["base_ini"].transform(lambda s: s.shift(7)),
        "FEWS_CS_12": grouped["base_ini"].transform(
            lambda s: s.rolling(window=12).mean()
        ),
    }
    # Every one is filled with the mean of base_ini, not with its own mean.
    base_mean = grouped["base_ini"].transform("mean")
    for name, series in derived.items():
        derived[name] = series.fillna(base_mean)
    return df.assign(**derived)


def make_lead_panel(df, lead):
    """feature_engineering.py:120-145.

    Shifts every column except FEWS_CS back by `lead` and drops the first
    `lead` rows of each unit. The panel is a contiguous monthly index per unit,
    so the positional shift is a calendar shift.
    """
    out = df.sort_values(UNIT_KEYS + ["time"]).copy()
    shift_columns = [c for c in out.columns if c not in ("FEWS_CS", "time")]

    if lead > 0:
        grouped = out.groupby(UNIT_KEYS, sort=False)
        out[shift_columns] = grouped[shift_columns].shift(lead)
        out = out.groupby(UNIT_KEYS, sort=False, group_keys=False).apply(
            lambda g: g.iloc[lead:], include_groups=True
        )

    out["lead"] = lead
    return out.rename(columns={"base_ini": "base_forecast"})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leads", type=str, default=None,
                        help="comma-separated subset of lead times, for smoke runs")
    parser.add_argument("--in-path", type=str, default=None,
                        help="override DATA_MASTER_PATH (read)")
    parser.add_argument("--out-path", type=str, default=None,
                        help="override INPUT_MASTER_PATH (write)")
    args = parser.parse_args(argv)

    leads = ([int(x) for x in args.leads.split(",")] if args.leads else LEADS)
    in_path = Path(args.in_path) if args.in_path else DATA_MASTER_PATH
    out_path = Path(args.out_path) if args.out_path else INPUT_MASTER_PATH

    if not in_path.exists():
        raise SystemExit(f"{in_path} not found -- run build_data_master.py")

    df = pd.read_csv(in_path)
    df["month"] = pd.PeriodIndex(df["month"], freq="M")
    df = df.sort_values(UNIT_KEYS + ["month"]).reset_index(drop=True)
    print(f"data_master: {len(df):,} rows, {df['county'].nunique()} units, "
          f"{df['month'].min()} .. {df['month'].max()}")

    roll_columns = [c for c in df.columns
                    if c not in DO_NOTHING_COLS and c != "month"]
    print(f"  rolling {len(roll_columns)} columns "
          f"({len(roll_columns) * len(ROLL_WINDOWS)} new): "
          f"{', '.join(roll_columns)}")
    df = add_rolling_means(df, roll_columns)

    # `time` keeps the target month; `month` becomes the calendar-month feature
    # and is shifted with everything else, exactly as in his script.
    df["time"] = df["month"]
    df = add_season_flags(df)
    df = add_fews_memory(df)

    print(f"  building {len(leads)} lead panels: {leads}")
    panels = [make_lead_panel(df, lead) for lead in leads]
    master = pd.concat(panels, ignore_index=True)

    # feature_engineering.py:166-170 -- units without cropland or rangeland get
    # the dataset-wide mean of the corresponding base column.
    for kind in ("crop", "range"):
        base = f"NDVI_anom_{kind}"
        fill = master[base].mean()
        columns = [c for c in master.columns if kind in c]
        before = int(master[columns].isna().sum().sum())
        master[columns] = master[columns].fillna(fill)
        print(f"  NDVI {kind}: filled {before:,} cells with dataset mean {fill:.4f} "
              f"across {len(columns)} columns")

    nan_columns = master.columns[master.isna().any()].tolist()
    print(f"\ncolumns still holding NaN: {nan_columns}")
    expected = {"FEWS_CS", "base_forecast"}
    unexpected = sorted(set(nan_columns) - expected)
    if unexpected:
        rates = master[unexpected].isna().mean().mul(100).round(1)
        print("  NOT in his expected {FEWS_CS, base_forecast} set:")
        print(rates.to_string())

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    master["time"] = master["time"].dt.to_timestamp()
    master.to_parquet(out_path, index=False)

    print(f"\nwrote {out_path}")
    print(f"  {len(master):,} rows x {len(master.columns)} columns")
    print(f"  leads: {sorted(master['lead'].unique())}")
    print(f"  rows with an observed FEWS_CS: {master['FEWS_CS'].notna().sum():,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
