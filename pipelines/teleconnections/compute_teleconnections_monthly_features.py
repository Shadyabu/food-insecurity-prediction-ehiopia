"""
compute_teleconnections_monthly_features.py

Engineering stage for the teleconnections pipeline. Parses the three raw
NOAA series (see fetch_teleconnections.py) into a single month-indexed
table and writes
data/processed/features/teleconnections_monthly.csv.

No spatial join anywhere in this file -- IOD/MEI/NINO3.4 are single global
values per month, not per-zone. The eventual multi-source feature join
must broadcast this table's rows to every zone_code for the matching
month (a plain many-to-one merge on `month` alone, executed at join time
in the model-table build step, NOT baked into this file -- see
docs/Teleconnections_Pipeline_Documentation.md Sec 3).

No re-standardization: DMI, MEI.v2 and the CPC NINO3.4 anomaly series are
already anomaly/standardized-index products as published by NOAA -- fitting
our own climatology here would double-standardize an already-standardized
value (confirmed by cross-checking the raw file contents and NOAA's own
"Climate Indices" reference page description). See CLAUDE.md Sec 3.6 /
Sec 4.2 of pipeline_replication_guide.md for why this matters generally --
here it means Stage 3 is just a lag-shift, not a baseline fit.

REVISION-LEAKAGE CAVEAT (see docs/Teleconnections_Pipeline_Documentation.md
Sec 4 for full reasoning): NOAA does not publish a bulk archive of
historical revision vintages for these indices, so the *_lag0 columns
below are each source's current (possibly revised) value, not necessarily
what would have been available for real-time prediction at that time. This
is confirmed empirically -- MEI.v2 in particular drifts from Busker et
al.'s 2023-vintage reference file by up to ~0.3 in places (e.g. 1987-06:
2.07 here vs 1.78 in their file). Decision (confirmed with the project
owner 2026-08-08): keep the lag0 (concurrent-month) column in the output
for RQ1 fidelity (Busker's own methodology used final values with no
vintage control) and for general reference-table completeness, but do NOT
use lag0 in any Ethiopia-specific (RQ2+) model -- use lag1/lag3/lag6
instead, which are far less exposed to the still-provisional most-recent
months.

Lag windows for the enriched Ethiopia-specific feature set (lag1, lag3,
lag6; confirmed with the project owner 2026-08-08): a defensible
ENSO/IOD-to-Horn-of-Africa-rainfall response window without over-
proliferating columns. This is IN ADDITION TO the uniform lead-time shift
Busker et al. and this project both apply to the whole feature block at
model-build time (CLAUDE.md Sec 3.2, make_lead_table()) -- these lags give
the model direct visibility into recent SST trajectory that the uniform
shift alone would not.
"""

import os

import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "teleconnections")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "teleconnections_monthly.csv")

MEI_PATH = os.path.join(RAW_DIR, "meiv2.data")
DMI_PATH = os.path.join(RAW_DIR, "dmi_had_long.data")
NINO34_PATH = os.path.join(RAW_DIR, "ersst5_nino_mth_91-20.ascii")

MEI_MISSING = -999.00
DMI_MISSING = -9999.000

FEATURE_START_YEAR = 2009  # matches the other feature pipelines (NDVI, GLEAM) feature-year window
LAGS = [1, 3, 6]


def _read_lines(path):
    with open(path, "r") as f:
        return f.readlines()


def parse_year_month_grid(path, missing_value, value_name, tolerance=1e-6):
    """Parses NOAA's standard 'YEAR M1 M2 ... M12' row format (used by
    both the MEI.v2 and DMI/HadISST files) -- skips header/footer lines
    by requiring exactly 13 whitespace-separated tokens and a plausible
    year in the first column."""
    rows = []
    for line in _read_lines(path):
        tokens = line.split()
        if len(tokens) != 13:
            continue
        try:
            year = int(tokens[0])
        except ValueError:
            continue
        if not (1800 <= year <= 2100):
            continue
        try:
            values = [float(t) for t in tokens[1:]]
        except ValueError:
            continue
        for month, val in enumerate(values, start=1):
            rows.append({
                "year": year,
                "month": month,
                value_name: np.nan if abs(val - missing_value) < tolerance else val,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No data rows parsed from {path} -- check format hasn't changed")
    return df


def parse_nino34_cpc(path, value_name="nino34"):
    """Parses NOAA CPC's 'YR MON NINO1+2 ANOM NINO3 ANOM NINO4 ANOM
    NINO3.4 ANOM' fixed-column format -- keeps only the NINO3.4 anomaly
    (last column), not the raw SST or the other three regions."""
    rows = []
    for line in _read_lines(path):
        tokens = line.split()
        if len(tokens) != 10:
            continue
        try:
            year = int(tokens[0])
            month = int(tokens[1])
        except ValueError:
            continue
        if not (1 <= month <= 12):
            continue
        try:
            anom = float(tokens[9])
        except ValueError:
            continue
        rows.append({"year": year, "month": month, value_name: anom})
    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"No data rows parsed from {path} -- check format hasn't changed")
    return df


def build_monthly_table():
    mei = parse_year_month_grid(MEI_PATH, MEI_MISSING, "mei")
    iod = parse_year_month_grid(DMI_PATH, DMI_MISSING, "iod")
    nino34 = parse_nino34_cpc(NINO34_PATH, "nino34")

    merged = mei.merge(iod, on=["year", "month"], how="outer").merge(nino34, on=["year", "month"], how="outer")
    merged["date"] = pd.to_datetime(dict(year=merged["year"], month=merged["month"], day=1))
    merged = merged.sort_values("date").reset_index(drop=True)

    # Reindex onto a continuous monthly grid so lag shifts are never
    # silently misaligned by a gap in any one source's raw file.
    full_range = pd.date_range(merged["date"].min(), merged["date"].max(), freq="MS")
    merged = merged.set_index("date").reindex(full_range)
    merged.index.name = "date"

    for col in ["iod", "mei", "nino34"]:
        for lag in LAGS:
            merged[f"{col}_lag{lag}"] = merged[col].shift(lag)

    merged["year"] = merged.index.year
    merged["month"] = merged.index.month
    merged = merged.reset_index(drop=True)
    merged = merged[merged["year"] >= FEATURE_START_YEAR].reset_index(drop=True)

    return merged


def trim_to_common_coverage(df):
    """Drop trailing rows where any lag0 index hasn't been published yet
    -- each source's release cadence differs (checked 2026-08-08: MEI
    reaches 2026-07, NINO3.4 reaches 2026-06, DMI only reaches 2026-05)."""
    last_common_pos = min(
        df.index[df["iod"].notna()].max(),
        df.index[df["mei"].notna()].max(),
        df.index[df["nino34"].notna()].max(),
    )
    return df.loc[:last_common_pos].copy()


def main():
    df = build_monthly_table()
    df = trim_to_common_coverage(df)

    cols = ["year", "month", "iod", "mei", "nino34"]
    for c in ["iod", "mei", "nino34"]:
        cols += [f"{c}_lag{lag}" for lag in LAGS]
    df = df[cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    first, last = df.iloc[0], df.iloc[-1]
    print(f"Wrote {len(df)} rows ({int(first['year'])}-{int(first['month']):02d} to "
          f"{int(last['year'])}-{int(last['month']):02d}) to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
