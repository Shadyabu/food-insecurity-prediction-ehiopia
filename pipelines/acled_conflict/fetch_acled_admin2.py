"""
fetch_acled_admin2.py

ACQUIRE stage for the ACLED conflict-event pipeline.

ACLED has no anonymous bulk-download endpoint (unlike FAO RAMSES for
locust) -- their Access Tool / API requires a registered account and
acceptance of ACLED's terms of use, so this stage is a manual export
rather than a scripted download:

  1. Register / log in at acleddata.com, use the Data Export Tool, filter
     to country = Ethiopia, event_date 2000-01-01 to present, export CSV.
  2. Drop the exported file in this pipeline's directory (matched below by
     the `ACLED Data_*.csv` glob ACLED's export tool names its files with).

This script's only job is to cache that manually-exported file into
data/raw/ (gitignored, per repo convention -- every other pipeline's raw
data lives there, not under pipelines/<source>/) and verify it actually
opens before anything downstream trusts it (CLAUDE.md Sec 3.6 -- a
truncated/interrupted export still "exists" on disk).

Compared against Busker et al.'s own staged ACLED extract
(busker_comparison/acled_1900-01-01-2023-04-17-Eastern_Africa-Ethiopia-Kenya-Somalia.csv,
Ethiopia/Kenya/Somalia bundled together, stops 2023-04-17): this file is
Ethiopia-only already, identical 31-column schema, and current through
2025-08-08. See docs/data_sources_master_table.md (ACLED row) for the full
comparison. Re-run the manual export periodically to extend coverage --
this script does not re-fetch automatically.
"""

import glob
import os
import shutil

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/acled_conflict/ -> pipelines/ -> repo root

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "acled_conflict")
CACHE_PATH = os.path.join(RAW_DIR, "acled_ethiopia_2000_present.csv")

SOURCE_GLOB = os.path.join(SCRIPT_DIR, "ACLED Data_*.csv")

REQUIRED_COLUMNS = [
    "event_id_cnty", "event_date", "year", "disorder_type", "event_type",
    "sub_event_type", "country", "admin1", "admin2", "latitude", "longitude",
    "fatalities",
]


def find_latest_export():
    matches = sorted(glob.glob(SOURCE_GLOB))
    if not matches:
        raise FileNotFoundError(
            f"No ACLED export found matching {SOURCE_GLOB}. "
            "Export from acleddata.com's Data Export Tool (country=Ethiopia) "
            "and drop the CSV in this directory first."
        )
    return matches[-1]  # ACLED's export filename embeds the request date -- last one is newest


def main():
    os.makedirs(RAW_DIR, exist_ok=True)

    source_path = find_latest_export()
    print(f"Found export: {source_path}")

    shutil.copyfile(source_path, CACHE_PATH)

    # Verify the cached file actually opens and has the expected shape --
    # not just that it exists on disk (Sec 3.6).
    df = pd.read_csv(CACHE_PATH, encoding="utf-8-sig")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Cached ACLED file is missing expected columns: {missing_cols}")

    non_ethiopia = df.loc[df["country"] != "Ethiopia", "country"].unique()
    if len(non_ethiopia) > 0:
        raise ValueError(f"Expected Ethiopia-only export, found other countries: {non_ethiopia}")

    print(f"Cached -> {CACHE_PATH}")
    print(f"Rows: {len(df)}")
    print(f"Date range: {df['event_date'].min()} to {df['event_date'].max()}")
    print(f"Admin1 units: {df['admin1'].nunique()}, Admin2 labels: {df['admin2'].nunique()}")


if __name__ == "__main__":
    main()
