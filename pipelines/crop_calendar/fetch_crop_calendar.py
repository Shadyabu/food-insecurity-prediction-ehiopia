#!/usr/bin/env python3
"""
fetch_crop_calendar.py
=======================
Stage 1 (Acquire): cache FEWS NET's own seasonal calendar for Ethiopia from
the FDW `season` endpoint.

    GET https://fdw.fews.net/api/season/?country_code=ET&format=json

This is a small (44-row) reference table, not a per-time-step download, so
there is no year-by-year acquisition loop like the raster pipelines -- one
request, cached and verified-to-open before use (CLAUDE.md §3.6).

USAGE
-----
    python fetch_crop_calendar.py
"""

from __future__ import annotations

import json
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent
RAW_PATH = ROOT / "data" / "raw" / "crop_calendar" / "fdw_season_ET.json"

URL = "https://fdw.fews.net/api/season/"
PARAMS = {"country_code": "ET", "format": "json", "limit": 500}


def main():
    print("=" * 74)
    print("Fetching FEWS NET seasonal calendar (FDW `season` endpoint)")
    print("=" * 74)

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = RAW_PATH.with_suffix(".json.part")

    resp = requests.get(URL, params=PARAMS, headers={"Accept": "application/json"},
                        timeout=60)
    resp.raise_for_status()
    data = resp.json()

    tmp_path.write_text(json.dumps(data, indent=2))
    # Verify it actually opens as valid JSON with the expected shape before
    # trusting it -- a truncated download still "exists" (CLAUDE.md §3.6).
    reloaded = json.loads(tmp_path.read_text())
    if not isinstance(reloaded, list) or not reloaded:
        tmp_path.unlink()
        raise SystemExit(f"Unexpected response shape from {URL}: {type(reloaded)}")
    tmp_path.rename(RAW_PATH)

    print(f"  wrote {RAW_PATH.relative_to(ROOT)} ({len(reloaded)} season records)")
    print(f"  purposes: {sorted(set(r['purpose'] for r in reloaded))}")
    print(f"  season_types: {sorted(set(r['season_type'] for r in reloaded))}")


if __name__ == "__main__":
    main()
