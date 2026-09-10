#!/usr/bin/env python3
"""
build_livelihood_zones_admin2.py
=================================
Static reference build: per-admin2 livelihood-system shares (pastoral /
agropastoral / crop-farming), used for RQ3 stratification (CLAUDE.md §7) and
to select the correct crop-calendar season system per zone.

WHY THIS ISN'T A features/ PIPELINE
------------------------------------
Every other source in data/processed/features/ is a monthly time series.
Livelihood zone composition is static reference geography, not a panel, so
this follows CLAUDE.md §5's instruction to treat it as a boundaries/-level
build: output lands at boundaries/livelihood_zones_admin2.csv, keyed on the
same zone_code as the admin2 shapefile, not on (zone_code, month).

SOURCE: BUSKER'S DERIVED TABLE, NOT A FRESH FEWS NET SHAPEFILE
-----------------------------------------------------------------
FEWS NET's FDW exposes a `fsc_lhz` ("FSC Livelihood Zone") geographic-unit
type, which would be the rigorous route (polygon geometry + fractional-area
overlap onto admin2, matching the fractional-coverage pattern already used
for the IPC target and every raster pipeline). Checked directly:

    GET https://fdw.fews.net/api/feature.geojson?country_code=ET&unit_type=fsc_lhz
    -> 200 OK, 0 features

FDW does not serve populated livelihood-zone polygons for Ethiopia through
this endpoint. In the absence of a machine-readable FEWS NET livelihood-zone
shapefile, this script uses Busker et al.'s own derived table instead
(`busker_comparison/livelihood_zones.xlsx`), which the paper's replication
package provides pre-aggregated to admin-unit population shares across all
three Horn-of-Africa countries. This is a *derived classification*, not
FEWS NET's original polygons -- flagged as a limitation, not silently
upgraded to look more authoritative than it is.

WHAT THE SOURCE FILE ACTUALLY CONTAINS
---------------------------------------
213 rows (Kenya + Ethiopia + Somalia), one row per admin-unit name, columns
`p` / `ap` / `other` (population shares) and `max` (dominant class letter).
Filtered here to the 92 Ethiopian admin2 zones by exact name match against
`ADM2_EN` -- verified 92/92 exact matches, so no fuzzy name-reconciliation
or fnid-vintage handling is needed for this file (unlike the FEWS NET IPC
data, this table is already keyed by zone name at the current vintage).

THE "other" = CROP-FARMING INFERENCE
--------------------------------------
The source file labels its third category "other", not "crop-farming". This
project's own stratification scheme (CLAUDE.md §7) is explicitly a 3-way
pastoral / agropastoral / crop-farming split, and empirically "other" is
dominant almost exclusively in Ethiopia's settled highland/agrarian regions
(Amhara, Tigray, SNNP, Oromia highlands, Gambela, Benishangul-Gumuz, Addis
Ababa, Harari, Dire Dawa: 67/72 non-pastoral-dominant zones), while "p"
(pastoral) dominates almost exclusively in Somali, Afar, and lowland
Oromia (Borena/Guji) -- Ethiopia's known pastoral regions. This is strong
but not certain evidence; Busker et al.'s paper does not spell out the
category's exact definition. Treated here as crop-farming, flagged as an
inference rather than a confirmed FEWS NET definition.

WEALTH-GROUP BREAKDOWNS: SEARCHED, NOT FOUND
------------------------------------------------
Checked FDW's full API root (138 endpoints) for anything wealth-group or
livelihood-profile shaped: no `livelihoodzone`, `wealthgroup`, or profile
endpoint exists. Checked the Busker source file for other sheets: single
sheet, no wealth-group columns. FEWS NET's household wealth-group
(poor/middle/better-off) breakdowns come from HEA baseline surveys published
as narrative livelihood-zone-profile PDFs, not structured/bulk data --
extracting them would mean hand-transcribing dozens of PDF profiles, which
is out of proportion to an admin2-level feature. Left out of scope; not
silently dropped -- documented here so a future session doesn't re-search
the same 138 endpoints.

USAGE
-----
    python build_livelihood_zones_admin2.py
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent  # pipelines/livelihood_zones/ -> pipelines/ -> repo root

SOURCE_XLSX = SCRIPT_DIR / "busker_comparison" / "livelihood_zones.xlsx"
ADMIN2_SHAPEFILE = ROOT / "boundaries" / "eth_admbnda_adm2_csa_bofedb_2021.shp"
OUT_PATH = ROOT / "boundaries" / "livelihood_zones_admin2.csv"

CLASS_MAP = {"p": "pastoral", "ap": "agropastoral", "other": "crop_farming"}


def main():
    print("=" * 74)
    print("Livelihood zones: admin2 pastoral / agropastoral / crop-farming shares")
    print("=" * 74)

    print(f"\n[1] Load Busker's admin-unit livelihood shares: {SOURCE_XLSX.name}")
    src = pd.read_excel(SOURCE_XLSX)
    src = src.rename(columns={src.columns[0]: "zone_name",
                               "p": "pct_pastoral",
                               "ap": "pct_agropastoral",
                               "other": "pct_crop_farming",
                               "max": "dominant_class"})
    print(f"    {len(src)} rows (Horn of Africa: Ethiopia + Kenya + Somalia)")

    print(f"\n[2] Load admin2 boundary: {ADMIN2_SHAPEFILE.name}")
    gdf = gpd.read_file(ADMIN2_SHAPEFILE)[
        ["ADM2_EN", "ADM2_PCODE", "ADM1_EN"]
    ].rename(columns={"ADM2_EN": "zone_name", "ADM2_PCODE": "zone_code",
                       "ADM1_EN": "region"})
    print(f"    {len(gdf)} Ethiopian admin2 zones")

    print("\n[3] Match by exact zone name")
    merged = gdf.merge(src, on="zone_name", how="left")
    n_matched = merged["dominant_class"].notna().sum()
    print(f"    matched {n_matched}/{len(gdf)} zones by exact ADM2_EN name")
    if n_matched != len(gdf):
        missing = sorted(merged.loc[merged["dominant_class"].isna(), "zone_name"])
        print(f"    UNMATCHED ({len(missing)}): {missing}")
        raise SystemExit(
            "Not all 92 zones matched by exact name -- previously verified as "
            "92/92; investigate before proceeding rather than silently dropping."
        )

    merged["dominant_livelihood_zone"] = merged["dominant_class"].map(CLASS_MAP)
    if merged["dominant_livelihood_zone"].isna().any():
        bad = sorted(merged.loc[merged["dominant_livelihood_zone"].isna(),
                                 "dominant_class"].unique())
        raise SystemExit(f"Unrecognised dominant_class value(s): {bad}")

    out = merged[["zone_code", "zone_name", "region", "pct_pastoral",
                  "pct_agropastoral", "pct_crop_farming",
                  "dominant_livelihood_zone"]].sort_values("zone_code")

    print("\n[4] Dominant class by region (sanity check -- see script docstring)")
    print(out.groupby("region")["dominant_livelihood_zone"].value_counts()
          .unstack(fill_value=0).to_string())

    print(f"\n[5] Write {OUT_PATH.relative_to(ROOT)}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"    wrote {len(out)} rows")
    print(f"\n  dominant class counts: "
          f"{out['dominant_livelihood_zone'].value_counts().to_dict()}")

    print("\n" + "=" * 74)
    print("Done. Run validate_livelihood_zones_output.py for the spot-check.")
    print("=" * 74)


if __name__ == "__main__":
    main()
