#!/usr/bin/env python3
"""
validate_livelihood_zones_output.py
====================================
No admin2-level numeric reference exists for this source (see build script's
docstring -- FDW's fsc_lhz endpoint returns 0 features for Ethiopia), so this
does not follow the median/mean-% pattern used elsewhere (CLAUDE.md §3.5).
Instead: structural checks + a spot-check against zones whose livelihood
system is unambiguous in the FEWS NET / food-security literature independent
of this pipeline (e.g. Borena and the Afar/Somali lowland zones as classic
pastoral areas; Gondar as classic highland cereal-farming), per this
project's own §3 pattern for reference (non-time-series) data.

USAGE
-----
    python validate_livelihood_zones_output.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_PATH = ROOT / "boundaries" / "livelihood_zones_admin2.csv"

# (substring match on zone_name, expected dominant_livelihood_zone, why)
SPOT_CHECKS = [
    ("Borena", "pastoral", "Borana pastoralist heartland, southern Oromia"),
    ("Doolo", "pastoral", "Somali region pastoral zone"),
    ("Kilbati-Zone 2", "pastoral", "Afar pastoral lowlands"),
    ("Hari-Zone 5", "pastoral", "Afar pastoral lowlands"),
    ("Central Gondar", "crop_farming", "Amhara highland cereal-farming belt"),
    ("Eastern Tigray", "crop_farming", "Tigray highland crop-farming"),
    ("Region 14", "crop_farming", "Addis Ababa (ADM2_EN 'Region 14') -- urban, no pastoral base"),
]


def main():
    print("=" * 74)
    print("Livelihood zones: structural checks + literature spot-check")
    print("=" * 74)

    if not OUT_PATH.exists():
        raise SystemExit(f"{OUT_PATH} not found. Run build_livelihood_zones_admin2.py first.")
    df = pd.read_csv(OUT_PATH)

    print(f"\n[1] Structural checks ({len(df)} rows)")
    ok = True

    n = len(df)
    if n != 92:
        print(f"    FAIL: expected 92 zones, got {n}")
        ok = False
    else:
        print("    OK: 92 rows")

    dupes = df["zone_code"].duplicated().sum()
    if dupes:
        print(f"    FAIL: {dupes} duplicate zone_code values")
        ok = False
    else:
        print("    OK: zone_code is unique")

    # Source rows commonly sum to <1.0 rather than exactly 1.0 -- the residual
    # is land Busker's classification left unassigned to p/ap/other (urban,
    # water, protected areas), not a computation error. Report the
    # distribution (per §3.5) rather than asserting exact-sum-to-1; only flag
    # a row if the residual is implausibly large (>25%, i.e. more than a
    # quarter of the zone unclassified) or shares exceed 1.0 outright.
    shares = df[["pct_pastoral", "pct_agropastoral", "pct_crop_farming"]]
    total = shares.sum(axis=1)
    print(f"    share-sum distribution: min {total.min():.3f}, median "
          f"{total.median():.3f}, max {total.max():.3f}")
    over = (total > 1.0 + 1e-6).sum()
    far_under = (total < 0.75).sum()
    if over:
        print(f"    FAIL: {over} rows sum to more than 1.0")
        ok = False
    elif far_under:
        print(f"    FAIL: {far_under} rows sum to less than 0.75 (implausibly "
              f"large unclassified residual)")
        ok = False
    else:
        print("    OK: no row exceeds 1.0 or falls below a 0.75 floor")

    out_of_range = ((shares < 0) | (shares > 1)).any(axis=1).sum()
    if out_of_range:
        print(f"    FAIL: {out_of_range} rows with a share outside [0, 1]")
        ok = False
    else:
        print("    OK: all shares within [0, 1]")

    print("\n[2] Spot-check against literature-known livelihood systems")
    n_pass = 0
    for substr, expected, why in SPOT_CHECKS:
        match = df[df["zone_name"].str.contains(substr, case=False, na=False)]
        if match.empty:
            print(f"    ?  {substr}: zone not found in output")
            continue
        row = match.iloc[0]
        actual = row["dominant_livelihood_zone"]
        status = "PASS" if actual == expected else "FAIL"
        n_pass += status == "PASS"
        print(f"    {status}  {row['zone_name']:20s} expected={expected:13s} "
              f"actual={actual:13s} ({why})")
        if status == "FAIL":
            ok = False

    print(f"\n    {n_pass}/{len(SPOT_CHECKS)} spot-checks passed")

    print("\n[3] Regional aggregate sanity (pastoral share should be near-zero "
          "in the historically settled highland regions)")
    for region in ["Amhara", "Tigray"]:
        mean_pastoral = df.loc[df["region"] == region, "pct_pastoral"].mean()
        print(f"    {region:10s} mean pct_pastoral = {mean_pastoral:.3f}")
        if mean_pastoral > 0.10:
            print(f"    FAIL: unexpectedly high pastoral share in {region}")
            ok = False

    print("\n" + "=" * 74)
    print("PASS" if ok else "FAIL -- see above")
    print("=" * 74)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
