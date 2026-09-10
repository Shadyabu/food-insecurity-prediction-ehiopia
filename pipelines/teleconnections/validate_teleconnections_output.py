"""
validate_teleconnections_output.py

Validates the teleconnections output two ways:

1. Against Busker et al.'s own staged reference file
   (busker_comparison/SST.xlsx, sheets IOD/MEI/NINA34) -- the closest
   thing to an independent published reference for these three series,
   per CLAUDE.md Sec 3.5.
2. Against NOAA's raw published files directly, re-parsed independently
   of compute_teleconnections_monthly_features.py -- this is the most
   exact validation possible for this pipeline (no aggregation or
   reprojection step exists that could introduce genuine disagreement,
   per the task brief), so any mismatch here is a fetch/parsing bug, not
   noise.

MEI is expected to show a real, non-zero difference against Busker's file
(documented revision drift -- see
docs/Teleconnections_Pipeline_Documentation.md Sec 4) -- this is reported
as a distribution, not treated as a failure. IOD and NINO3.4 are expected
to match Busker's file almost exactly.
"""

import os

import numpy as np
import openpyxl
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "teleconnections_monthly.csv")
BUSKER_XLSX = os.path.join(SCRIPT_DIR, "busker_comparison", "SST.xlsx")
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "teleconnections")


def load_busker_sheet(sheet_name, missing_value=None, tolerance=1e-6):
    wb = openpyxl.load_workbook(BUSKER_XLSX, data_only=True)
    ws = wb[sheet_name]
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):  # row 1 is the header
        date_val, value = row[0], row[1]
        if date_val is None or value is None:
            continue
        if isinstance(date_val, str):
            date_val = pd.to_datetime(date_val)
        if missing_value is not None and abs(value - missing_value) < tolerance:
            continue
        rows.append({"year": date_val.year, "month": date_val.month, "busker_value": value})
    return pd.DataFrame(rows)


def report_distribution(label, diff):
    diff = diff.dropna()
    n = len(diff)
    if n == 0:
        print(f"  {label}: no overlapping rows to compare")
        return
    abs_diff = diff.abs()
    print(f"  {label}: n={n}, median|diff|={abs_diff.median():.4f}, mean|diff|={abs_diff.mean():.4f}, "
          f"max|diff|={abs_diff.max():.4f}, exact match (<1e-6) = {(abs_diff < 1e-6).sum()}/{n} "
          f"({100 * (abs_diff < 1e-6).mean():.1f}%)")


def validate_against_busker(ours):
    print("=== 1. Validation against Busker et al.'s staged reference (SST.xlsx) ===")

    busker_iod = load_busker_sheet("IOD")
    m = ours[["year", "month", "iod"]].merge(busker_iod, on=["year", "month"], how="inner")
    report_distribution("IOD vs Busker", m["iod"] - m["busker_value"])

    busker_mei = load_busker_sheet("MEI", missing_value=-999)
    m = ours[["year", "month", "mei"]].merge(busker_mei, on=["year", "month"], how="inner")
    report_distribution("MEI vs Busker (revision drift expected, see doc)", m["mei"] - m["busker_value"])

    busker_nino34 = load_busker_sheet("NINA34", missing_value=-99.99)
    m = ours[["year", "month", "nino34"]].merge(busker_nino34, on=["year", "month"], how="inner")
    report_distribution("NINO3.4 vs Busker", m["nino34"] - m["busker_value"])


def independent_reparse_check(ours):
    print("\n=== 2. Independent re-parse of raw NOAA files (fetch/parse-bug check) ===")

    # MEI: raw grid file, independently re-parsed with a plain line scan
    mei_rows = []
    with open(os.path.join(RAW_DIR, "meiv2.data")) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) == 13 and tokens[0].isdigit() and 1800 < int(tokens[0]) < 2100:
                year = int(tokens[0])
                for i, tok in enumerate(tokens[1:], start=1):
                    val = float(tok)
                    if val != -999.00:
                        mei_rows.append({"year": year, "month": i, "raw_value": val})
    mei_raw = pd.DataFrame(mei_rows)
    m = ours[["year", "month", "mei"]].merge(mei_raw, on=["year", "month"], how="inner")
    report_distribution("MEI vs independent re-parse", m["mei"] - m["raw_value"])

    # NINO3.4: raw CPC file, independently re-parsed by column position
    nino_rows = []
    with open(os.path.join(RAW_DIR, "ersst5_nino_mth_91-20.ascii")) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) == 10 and tokens[0].isdigit():
                nino_rows.append({"year": int(tokens[0]), "month": int(tokens[1]), "raw_value": float(tokens[9])})
    nino_raw = pd.DataFrame(nino_rows)
    m = ours[["year", "month", "nino34"]].merge(nino_raw, on=["year", "month"], how="inner")
    report_distribution("NINO3.4 vs independent re-parse", m["nino34"] - m["raw_value"])

    # DMI: raw grid file, independently re-parsed with a plain line scan
    dmi_rows = []
    with open(os.path.join(RAW_DIR, "dmi_had_long.data")) as f:
        for line in f:
            tokens = line.split()
            if len(tokens) == 13 and tokens[0].isdigit() and 1800 < int(tokens[0]) < 2100:
                year = int(tokens[0])
                for i, tok in enumerate(tokens[1:], start=1):
                    val = float(tok)
                    if val != -9999.000:
                        dmi_rows.append({"year": year, "month": i, "raw_value": val})
    dmi_raw = pd.DataFrame(dmi_rows)
    m = ours[["year", "month", "iod"]].merge(dmi_raw, on=["year", "month"], how="inner")
    report_distribution("IOD vs independent re-parse", m["iod"] - m["raw_value"])


def check_lag_alignment(ours):
    print("\n=== 3. Lag-shift alignment check ===")
    # For any row with a full 6 months of history in the output, lag1 of
    # month t should equal the lag0 value at month t-1 (checked directly
    # off the loaded table, not re-derived from source files).
    ours_indexed = ours.set_index(pd.to_datetime(dict(year=ours.year, month=ours.month, day=1)))
    shifted = ours_indexed["iod"].shift(1)
    mismatch = (ours_indexed["iod_lag1"] - shifted).abs().dropna()
    n_bad = (mismatch > 1e-9).sum()
    print(f"  iod_lag1 vs iod.shift(1): {n_bad} mismatches out of {len(mismatch)} rows "
          f"({'PASS' if n_bad == 0 else 'FAIL'})")


def main():
    ours = pd.read_csv(OUTPUT_PATH)
    print(f"Loaded {len(ours)} rows from {OUTPUT_PATH}\n")

    validate_against_busker(ours)
    independent_reparse_check(ours)
    check_lag_alignment(ours)


if __name__ == "__main__":
    main()
