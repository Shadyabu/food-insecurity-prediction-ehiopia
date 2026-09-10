#!/usr/bin/env python3
"""
build_crop_calendar_admin2.py
==============================
Stage 2-3 (Engineer): turn FEWS NET's own seasonal calendar (fetched by
fetch_crop_calendar.py) into per-admin2, per-month-of-year season flags.

WHY THIS IS A boundaries/-LEVEL BUILD, NOT features/
-------------------------------------------------------
This is a fixed annual pattern (month 1-12), not a 2011-2026 time series, so
it follows the same convention as livelihood_zones: output lands at
boundaries/crop_calendar_admin2_month.csv, keyed on (zone_code, month_of_year)
rather than (zone_code, month). Join to any real calendar month at
feature-build time via `calendar_month = real_month.month`.

WHICH SEASON SYSTEM APPLIES TO WHICH ZONE
---------------------------------------------
FDW's `season` endpoint (purpose=NutritionIndicator) publishes THREE distinct,
non-overlapping regional calendars for Ethiopia, keyed by geographicunit:

  - Admin 0 "Ethiopia" (national default): Belg/Meher two-harvest highland
    system -- Lean Apr-Jun, Belg harvest Jul-Aug, Meher harvest Oct-Jan.
  - Admin 1 "Somali": Gu/Deyr pastoral system -- Lean(pastoral) Feb-Apr,
    Lean(agricultural) May-Jun, Gu harvest Jul, Deyr harvest Jan.
  - Admin 1 "Benishangul Gumuz": Meher-only variant -- Lean Jun-Sep, Meher
    harvest Oct-Jan (no Belg season in this region).

Zones are assigned a system by ADM1_EN region: Somali -> Gu/Deyr, Benishangul
Gumz -> its Meher-only variant, everything else -> the national default.

KNOWN GAP: AFAR
------------------
Afar is a pastoral region like Somali, but FDW's season endpoint does not
publish an Afar-specific Admin 1 record in this extract (only Somali and
Benishangul Gumuz have region-specific NutritionIndicator entries; Afar's
CropProductionIndicator entries give Karan/Sugum harvest MONTHS only, with no
corresponding Lean-season definition). Afar zones therefore fall back to the
national Belg/Meher default here, which is agronomically wrong for a
pastoral region -- flagged explicitly rather than fabricating an Afar lean
season FEWS NET hasn't published. See validate_crop_calendar_output.py.

`is_planting_season` WAS NOT BUILT
--------------------------------------
The task specification proposed `is_planting_season` and
`months_since_planting_start` as candidate columns, for confirmation before
writing code. FDW's `season` endpoint publishes exactly three season_type
values for Ethiopia: Harvest, Lean, Dry -- no Planting type exists in the
source at all (confirmed: `fetch_crop_calendar.py` prints the full set).
Deriving a planting window from the published Dry-season boundaries was
attempted and rejected: the gap between one harvest's Dry period and the
next Harvest window does not reliably locate the OTHER season's planting
(e.g. the Post-Belg-Harvest Dry period, Sep, is immediately followed by the
Meher Harvest window in Oct -- Meher's actual growing season overlaps
earlier in the year, so "the month after a Dry window ends" would silently
mislabel Meher's planting as September). Building the column would mean
inventing a lead time not present in the source. Built instead:
`is_harvest_season`, `is_lean_season`, `is_dry_season` (all three season
types FEWS NET actually publishes) and `months_since_harvest_start` /
`months_since_lean_start` (cyclic distance since the most recent start of
that season type, useful as an LSTM positional feature without fabricating
a planting date).

USAGE
-----
    python build_crop_calendar_admin2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent

RAW_PATH = ROOT / "data" / "raw" / "crop_calendar" / "fdw_season_ET.json"
ADMIN2_SHAPEFILE = ROOT / "boundaries" / "eth_admbnda_adm2_csa_bofedb_2021.shp"
OUT_PATH = ROOT / "boundaries" / "crop_calendar_admin2_month.csv"

# geographicunit id -> which zones this calendar applies to, matched on ADM1_EN
SOMALI_GEOUNIT = 5218
BENISHANGUL_GEOUNIT = 5221
NATIONAL_GEOUNIT = 8042

REGION_TO_GEOUNIT = {
    "Somali": SOMALI_GEOUNIT,
    "Benishangul Gumz": BENISHANGUL_GEOUNIT,
}
DEFAULT_GEOUNIT = NATIONAL_GEOUNIT


def month_in_window(month, start, end):
    """True if `month` (1-12) falls in [start, end], wrapping across Dec/Jan."""
    if start <= end:
        return start <= month <= end
    return month >= start or month <= end


def cyclic_dist_since_start(month, starts):
    """Months elapsed (0-11) since the most recent window start, cyclically.

    E.g. for a start of month 10 (October): October itself -> 0,
    November -> 1, ..., September -> 11.
    """
    if not starts:
        return None
    return min((month - s) % 12 for s in starts)


def load_season_windows():
    if not RAW_PATH.exists():
        raise SystemExit(f"{RAW_PATH} not found. Run fetch_crop_calendar.py first.")
    records = json.loads(RAW_PATH.read_text())
    df = pd.DataFrame(records)
    df = df[df["purpose"] == "NutritionIndicator"]
    return df


def windows_for_geounit(df, geounit_id):
    sub = df[df["geographicunit"] == geounit_id]
    if sub.empty:
        raise SystemExit(f"No season windows found for geographicunit={geounit_id}")
    return sub[["name", "season_type", "start_month", "end_month"]].to_dict("records")


def month_flags(windows, month):
    harvest_starts = [w["start_month"] for w in windows if w["season_type"] == "Harvest"]
    lean_starts = [w["start_month"] for w in windows if w["season_type"] == "Lean"]

    is_harvest = any(month_in_window(month, w["start_month"], w["end_month"])
                      for w in windows if w["season_type"] == "Harvest")
    is_lean = any(month_in_window(month, w["start_month"], w["end_month"])
                  for w in windows if w["season_type"] == "Lean")
    is_dry = any(month_in_window(month, w["start_month"], w["end_month"])
                 for w in windows if w["season_type"] == "Dry")

    return {
        "is_harvest_season": is_harvest,
        "is_lean_season": is_lean,
        "is_dry_season": is_dry,
        "months_since_harvest_start": cyclic_dist_since_start(month, harvest_starts),
        "months_since_lean_start": cyclic_dist_since_start(month, lean_starts),
    }


def main():
    print("=" * 74)
    print("Crop calendar: admin2 x month-of-year season flags")
    print("=" * 74)

    print("\n[1] Load season windows")
    season_df = load_season_windows()
    system_windows = {
        "national_belg_meher": windows_for_geounit(season_df, NATIONAL_GEOUNIT),
        "somali_gu_deyr": windows_for_geounit(season_df, SOMALI_GEOUNIT),
        "benishangul_meher_only": windows_for_geounit(season_df, BENISHANGUL_GEOUNIT),
    }
    for name, windows in system_windows.items():
        print(f"    {name}: " + "; ".join(
            f"{w['name']}({w['season_type']}) {w['start_month']}->{w['end_month']}"
            for w in windows))

    print("\n[2] Assign a season system per zone by region")
    gdf = gpd.read_file(ADMIN2_SHAPEFILE)[["ADM2_EN", "ADM2_PCODE", "ADM1_EN"]].rename(
        columns={"ADM2_EN": "zone_name", "ADM2_PCODE": "zone_code", "ADM1_EN": "region"})

    def system_for_region(region):
        if region == "Somali":
            return "somali_gu_deyr"
        if region == "Benishangul Gumz":
            return "benishangul_meher_only"
        return "national_belg_meher"

    gdf["season_system"] = gdf["region"].map(system_for_region)
    print(gdf["season_system"].value_counts().to_string())
    n_afar = (gdf["region"] == "Afar").sum()
    print(f"    NOTE: {n_afar} Afar zones fall back to national_belg_meher "
          f"(no Afar-specific FDW record -- see script docstring)")

    print("\n[3] Expand to zone x month-of-year (1-12) with season flags")
    rows = []
    for _, zone in gdf.iterrows():
        windows = system_windows[zone["season_system"]]
        for month in range(1, 13):
            flags = month_flags(windows, month)
            rows.append({
                "zone_code": zone["zone_code"],
                "zone_name": zone["zone_name"],
                "region": zone["region"],
                "season_system": zone["season_system"],
                "month": month,
                **flags,
            })
    out = pd.DataFrame(rows)

    print(f"\n[4] Write {OUT_PATH.relative_to(ROOT)}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"    wrote {len(out)} rows ({gdf.shape[0]} zones x 12 months)")

    print("\n" + "=" * 74)
    print("Done. Run validate_crop_calendar_output.py for the spot-check.")
    print("=" * 74)


if __name__ == "__main__":
    main()
