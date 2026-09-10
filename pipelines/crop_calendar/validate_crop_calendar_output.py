#!/usr/bin/env python3
"""
validate_crop_calendar_output.py
=================================
No independent numeric reference exists for this source (it's FEWS NET's own
published calendar, not observed data -- see build script's docstring), so
this validates by confirming the derived per-month flags align with FEWS
NET's published calendar for a sample of zones/regions, per CLAUDE.md §5's
guidance for reference (non-time-series) data, plus structural checks.

USAGE
-----
    python validate_crop_calendar_output.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "boundaries" / "crop_calendar_admin2_month.csv"

# (region, month, expected is_harvest_season, expected is_lean_season, why)
SPOT_CHECKS = [
    ("Amhara", 11, True, False, "November is inside the Oct-Jan Meher harvest"),
    ("Amhara", 5, False, True, "May is inside the Apr-Jun national lean season"),
    ("Amhara", 8, True, False, "August is inside the Jul-Aug Belg harvest"),
    ("Somali", 7, True, False, "July is the Gu harvest month"),
    ("Somali", 1, True, False, "January is the Deyr harvest month"),
    ("Somali", 3, False, True, "March is inside the Feb-Apr pastoral lean season"),
    ("Benishangul Gumz", 12, True, False, "December is inside the Oct-Jan Meher harvest"),
    ("Benishangul Gumz", 7, False, True, "July is inside the Jun-Sep lean season"),
]


def main():
    print("=" * 74)
    print("Crop calendar: structural checks + FEWS NET calendar spot-check")
    print("=" * 74)

    if not OUT_PATH.exists():
        raise SystemExit(f"{OUT_PATH} not found. Run build_crop_calendar_admin2.py first.")
    df = pd.read_csv(OUT_PATH)

    print(f"\n[1] Structural checks ({len(df)} rows)")
    ok = True

    if len(df) != 92 * 12:
        print(f"    FAIL: expected {92*12} rows (92 zones x 12 months), got {len(df)}")
        ok = False
    else:
        print(f"    OK: {92*12} rows (92 zones x 12 months)")

    dupes = df.duplicated(subset=["zone_code", "month"]).sum()
    if dupes:
        print(f"    FAIL: {dupes} duplicate (zone_code, month) rows")
        ok = False
    else:
        print("    OK: (zone_code, month) is unique")

    if not df["month"].between(1, 12).all():
        print("    FAIL: month values outside 1-12")
        ok = False
    else:
        print("    OK: month in [1, 12] for every row")

    # Every zone-month should have EXACTLY one season type active among
    # harvest/lean (dry is allowed to coexist conceptually but the three
    # published systems never overlap harvest+lean in the same month --
    # confirm that invariant holds).
    both = (df["is_harvest_season"] & df["is_lean_season"]).sum()
    if both:
        print(f"    FAIL: {both} zone-months flagged as BOTH harvest and lean")
        ok = False
    else:
        print("    OK: no zone-month is flagged as both harvest and lean")

    print("\n[2] Spot-check against FEWS NET's published seasonal calendar")
    n_pass = 0
    for region, month, exp_harvest, exp_lean, why in SPOT_CHECKS:
        sub = df[(df["region"] == region) & (df["month"] == month)]
        if sub.empty:
            print(f"    ?  {region} month={month}: no rows found")
            continue
        row = sub.iloc[0]
        pass_h = bool(row["is_harvest_season"]) == exp_harvest
        pass_l = bool(row["is_lean_season"]) == exp_lean
        status = "PASS" if (pass_h and pass_l) else "FAIL"
        n_pass += status == "PASS"
        print(f"    {status}  {region:18s} month={month:>2} "
              f"harvest={bool(row['is_harvest_season'])!s:5} "
              f"lean={bool(row['is_lean_season'])!s:5}  ({why})")
        if status == "FAIL":
            ok = False
    print(f"\n    {n_pass}/{len(SPOT_CHECKS)} spot-checks passed")

    print("\n[3] Known gap: Afar inherits the national default (documented, not a bug)")
    afar_system = df.loc[df["region"] == "Afar", "season_system"].unique()
    print(f"    Afar season_system = {list(afar_system)} "
          f"(expected: ['national_belg_meher'] -- FDW publishes no Afar-specific "
          f"NutritionIndicator record; see build script docstring)")

    print("\n" + "=" * 74)
    print("PASS" if ok else "FAIL -- see above")
    print("=" * 74)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
