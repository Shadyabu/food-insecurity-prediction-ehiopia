"""
compute_agss_yields_admin2_monthly_features.py

AGGREGATE + ENGINEER stage for the AgSS yield pipeline.

    [1. ACQUIRE]   FDW cropproductionfacts, crop:yield, ET   (fetch_agss_yields_admin2.py)
    [2. AGGREGATE] admin_2 name crosswalk -> zone_code       (this script)
    [3. ENGINEER]  anomaly, publication-lag visibility,      (this script)
                   monthly broadcast, surveyed/status flags
    [4. VALIDATE]  FAOSTAT/World Bank cross-check            (validate_agss_yields_output.py)

ZONE CROSSWALK -- WHY THIS IS NOT A SIMPLE JOIN
------------------------------------------------
FDW's `admin_2` field is a free-text name tied to whichever fnid vintage
reported that record, not this project's `zone_code`. Checked directly
against boundaries/livelihood_zones_admin2.csv: of ~105 distinct admin_2
names seen across the 4 staple crops, only 46 match one of the 92 zone
names exactly. The other 59 fall into three genuinely different cases,
investigated and resolved individually (cheapest-to-most-expensive, per
CLAUDE.md Sec 3.5) rather than fuzzy-matched:

  1. SPELLING / RENAMING VARIANTS (the majority) -- e.g. "North Wollo" vs.
     this project's "North Wello", "Sitti" vs. "Siti", "Zone 1 (AF)" vs.
     "Awsi-Zone 1". One admin unit, one name change or transliteration
     difference. Mapped 1:1 in ZONE_ALIASES.

  2. GENUINE ADMINISTRATIVE SPLITS -- an FDW-era admin_2 unit whose
     territory was later divided into two or more of this project's 92
     zones (e.g. "Gamo Gofa" -> "Gamo" + "Gofa"; "Kembata Alaba Tembaro" ->
     "Kembata Tembaro" + "Halaba"; "Keficho Shekicho" -> "Kefa" + "Sheka";
     "Dire Dawa" -> "Dire Dawa urban" + "Dire Dawa rural"). The single
     historical value is the correct reported figure for the UNDIVIDED
     area at the time it was collected, so it is attributed to every child
     zone -- but each such row is explicitly flagged via
     `{crop}_yield_note = "inherited_from_split_parent:<FDW name>"` so a
     later reader can see two zones sharing an identical figure is a real,
     documented administrative-history artefact, not a join bug. This is
     NOT the same kind of decision as the pastoral-zone no-imputation rule
     below: it re-attributes a genuinely collected value to the areas it
     validly covered, rather than inventing a value where none exists.

  3. GENUINELY AMBIGUOUS / UNRESOLVABLE -- names alone couldn't resolve
     these 7, so each was checked geometrically against FDW's
     `geographicunit` endpoint (2026-08-09): FDW does NOT expose full
     polygon geometry via its public API (only centroid + area), so this
     is a centroid-in-polygon + area-ratio check against the fixed 92-zone
     boundary, not the fractional-area-overlap method `pipelines/ipc_target/`
     uses against Busker's own bulk-downloaded fsc_admin shapefile -- a
     real methodological downgrade, noted rather than glossed over. Result:

       name                        contains-zone   hist-area/zone-area
       Moyale                      Daawa           2.45x  -> real split
       North Omo                   Gamo            3.20x  -> real split
       Segen Area Peoples Zone     Burji           3.26x  -> real split
       Argoba Special Woreda (AM)  South Wello      0.02x -> tiny enclave
       Pawe Special Woreda         Metekel          0.03x -> tiny enclave
       Zone 1 (GM)                 Nuwer           2.61x  -> real split
       Zone 2 (GM)                 Agnewak          0.66x -> genuine match

     A ratio far from 1.0 in EITHER direction means forcing a 1:1 match
     would manufacture precision, just in two different ways: >2x means
     the historical unit truly covered multiple current zones (the same
     problem case 2 handles, but with too many plausible children to pick
     defensibly); <0.05x means the historical unit is a tiny enclave whose
     single yield figure would misrepresent a current zone 30-50x its own
     size if attributed wholesale -- the false-precision problem in
     reverse. Only Zone 2 (GM)'s ratio (0.66x, a plausible boundary
     redraw rather than a split) was close enough to add to ZONE_ALIASES.
     The other 6 (Moyale, North Omo, Segen Area Peoples Zone, Argoba
     Special Woreda, Pawe Special Woreda, Zone 1 (GM)) remain UNMATCHED
     and DROPPED -- confirmed by geometry, not just left as an unresolved
     guess. Combined cost of dropping these 6: ~93 Meher-season records
     across the 4 crops (~2% of ~5,100 fetched), concentrated in Argoba
     Special Woreda and Segen Area Peoples Zone.

PASTORAL-ZONE COVERAGE GAP -- DELIBERATE NO-IMPUTATION
---------------------------------------------------------
An early ~2,800-record PARTIAL sample (pulled during source investigation,
before the full fetch worked around FDW's date-range-filter no-op --
see fetch_agss_yields_admin2.py) suggested a starker gap than the full
dataset actually shows. Checked again against the COMPLETE 1995-2024
zone-year table before finalizing this script: 8 of this project's 92
zones -- 7 pastoral (Doolo, Korahe, Fanti-Zone 4, Daawa, Nogob, Erer,
Afder) and 1 agropastoral (Alle) -- have ZERO real (`Collected`, non-null)
yield values for any of the 4 crops across the entire 30-year window,
despite some of them nominally appearing in FDW's table via
`Not Collected`/`Not Available` placeholder rows in some years (which is
why `agss_surveyed_flag` below is defined on real values only, not mere
row presence -- a looser definition would silently hide this finding).
Beyond those 8, coverage among the remaining pastoral/agropastoral zones is
NOT uniform: a middle tier (Kilbati-Zone 2, East Bale, Jarar: 3 collected
years; Shabelle, West Guji: 4; Hari-Zone 5: 5; Nuwer: 15) has real but
sparse, intermittent data, while another tier (Borena, Bale, Siti,
Awsi-Zone 1, Guji, East Hararge, South Omo, Fafan: 18-25 years) is nearly
as well covered as the median crop-farming zone (mean 20.3 years). This is
a genuine gradient tied to how much cropland actually exists within each
zone's boundary, not a clean pastoral/crop-farming binary -- worth stating
precisely in the write-up rather than the coarser "pastoral zones are
unsurveyed" framing the initial partial sample suggested.

Whatever the exact zone, missing years are AgSS's real sampling frame (a
crop-sector survey has no reason to visit -- or found nothing worth
tabulating in -- a zone with negligible cropland that year), not a
processing gap -- so every gap is carried as an explicit `not_surveyed` /
`not_collected_or_available` status, never imputed via a regional average
or model-based fill. Imputing here would manufacture false precision in
exactly the zones this dissertation's fairness framing already commits to
surfacing, not smoothing over (CLAUDE.md Sec 7, "Historical bias &
geographic feedback loops").

FDW's own `status` field additionally distinguishes, for zones that WERE
in a given year's survey round: `Collected` (real value) from
`Not Collected` / `Not Available` (round covered the zone, this crop's
figure wasn't reported/published). Both are mapped to
`not_collected_or_available` here -- genuinely different from
`not_surveyed` (no record for that zone at all, any year), which is the
signal that matters for RQ3.

RECENCY -- NO FORWARD-FILL PAST THE CONFIRMED 2022 CEILING
---------------------------------------------------------------
Confirmed in fetch_agss_yields_admin2.py's docstring: no AgSS record of any
kind exists past 2022-07-31 for Ethiopia in FDW (same ceiling as
HarvestStat Africa -- a real limit, not a pipeline gap). A Meher-season
value for harvest year Y is broadcast forward monthly ONLY within its own
normal annual visibility window (see PUBLICATION_LAG_MONTHS below) -- once
that window ends with no newer harvest year available to supersede it,
the feature reverts to explicit NaN with status `no_data_post_<Y+1>`,
never held constant indefinitely. This was a confirmed decision (2026-08-09
project-owner exchange): AgSS is silently absent for exactly 2023-2026,
which CLAUDE.md Sec 7 already flags as a higher-crisis-rate era distinct
from 2011-2021 -- worth restating wherever this feature is first used in
an RQ2+ experiment.

PUBLICATION LAG -- A DOCUMENTED ASSUMPTION, NOT A VERIFIED VINTAGE
-----------------------------------------------------------------------
No machine-readable AgSS release-date record could be found (ESS's own PDF
upload timestamps reflect site migrations, not original publication; FDW's
`created` timestamps reflect a 2026-04 bulk re-ingestion of the whole
historical archive, not per-record original publication). Same
unresolvable-vintage situation as pipelines/imf_gdp/. Applying the same
kind of documented default: Meher harvest completes ~January of year Y+1,
so a Meher-Y value becomes visible from **October of Y+1** (~9 months
post-harvest -- slightly more conservative than GDP's April-of-Y+1, given
AgSS's heavier household-survey tabulation process). Flagged here as an
ASSUMPTION for anyone citing this feature's timing in the write-up.

SEASON SCOPE -- MEHER ONLY
------------------------------
Only `season_name == "Meher"` records are used (the main season, ~90-95%
of Ethiopia's cereal production, and the season these 4 staple grains are
overwhelmingly reported under). Belg-season records exist for some
zone-years but are a minor secondary season for these crops and are out of
scope for this feature -- not merged in, to avoid conflating two different
growing seasons' yields into one series.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
RAW_DIR = REPO_ROOT / "data" / "raw" / "agss_yields"
INTERIM_DIR = REPO_ROOT / "data" / "interim" / "agss_yields"
OUTPUT_DIR = REPO_ROOT / "data" / "processed" / "features"
LIVELIHOOD_CSV = REPO_ROOT / "boundaries" / "livelihood_zones_admin2.csv"

CROPS = ["maize", "wheat", "sorghum", "teff"]

# Publication-lag assumption -- see docstring. Meher-Y visible from
# October of Y+1 through September of Y+2 (12 months), unless superseded
# earlier by a newer harvest year.
PUBLICATION_LAG_MONTH = 10  # October
PUBLICATION_LAG_YEAR_OFFSET = 1  # of harvest_year + 1

# Full monthly panel range this feature is broadcast across, matching the
# other broadcast-pattern sources (imf_gdp, cpi, teleconnections, wvg).
PANEL_START = (1995, 1)
PANEL_END = (2026, 12)

MIN_ANOMALY_HISTORY_YEARS = 3  # backward-looking baseline needs some history first

# --------------------------------------------------------------------------
# Zone crosswalk -- see module docstring for how each case was decided.
# --------------------------------------------------------------------------

ZONE_ALIASES = {
    # Afar
    "Awusi": "Awsi-Zone 1", "Zone 1 (AF)": "Awsi-Zone 1",
    "Kilbati": "Kilbati-Zone 2", "Zone 2 (AF)": "Kilbati-Zone 2",
    "Gabi": "Gabi-Zone 3", "Zone 3 (AF)": "Gabi-Zone 3",
    "Fanti": "Fanti-Zone 4", "Zone 4 (AF)": "Fanti-Zone 4",
    "Khari": "Hari-Zone 5", "Hari": "Hari-Zone 5", "Zone 5 (AF)": "Hari-Zone 5",
    # Tigray
    "East Tigray": "Eastern Tigray",
    "South Tigray": "Southern Tigray",
    "West Tigray": "Western Tigray",
    "Northwest Tigray": "North Western Tigray",
    "Southeast Tigray": "South Eastern Tigray",
    # Amhara
    "North Wollo": "North Wello",
    "South Wollo": "South Wello",
    "Wag Himra": "Wag Hamra",
    "Oromia Zone": "Oromia",
    # Oromia / Somali pastoral zones
    "Gujii": "Guji",
    "West Gujii": "West Guji",
    "Dollo": "Doolo",
    "Liben": "Liban",
    "Shebelle": "Shabelle",
    "Sitti": "Siti",
    "Shinile": "Siti",  # Shinile Zone was renamed Siti Zone (Somali region reorg)
    "Jijiga": "Fafan",  # Jijiga Zone was renamed Fafan Zone (Somali region reorg)
    "Ilubabor": "Ilu Aba Bora",
    "Kelem Wellega": "Kellem Wollega",
    "Horo Guduru Wellega": "Horo Gudru Wellega",
    "Oromia Special Zone": "Finfine Special",
    "Southwest Shewa": "South West Shewa",
    # Benishangul-Gumuz
    "Kemashi": "Kamashi",
    "Mao-Komo Special Woreda": "Mao Komo Special",
    # SNNP / South West Ethiopia
    "West Omo": "Mirab Omo",  # "Mirab" = "West" in Amharic; confirmed via
        # FDW geographicunit centroid falling exactly inside our Mirab Omo
        # boundary (2026-08-09). Does NOT change coverage -- "West Omo" has
        # zero maize/wheat/sorghum/teff yield records in the raw data
        # regardless -- added for correctness, not to fill a gap.
    "Bench Maji": "Bench Sheko",  # Bench Maji Zone was renamed Bench Sheko Zone
    "Keffa": "Kefa",
    "Wolayita": "Welayta",
    "Gurage": "Guraghe",
    "Dawro": "Dawuro",
    "Alaba Special Woreda": "Halaba",
    "Amaro Special Woreda": "Amaro",
    "Basketo Special Woreda": "Basketo",
    "Burji Special Woreda": "Burji",
    "Derashe Special Woreda": "Derashe",
    "Konso Special Woreda": "Konso",
    "Konta Special Woreda": "Konta Special",
    "Yem Special Woreda": "Yem Special",
    # Gambela
    "Mezhenger": "Majang",
    "Nuer": "Nuwer",
    "Etang Special Woreda": "Itang Special woreda",
    # Confirmed via FDW geographicunit centroid + area check (2026-08-09,
    # see docstring "GEOMETRIC CONFIRMATION" section): "Zone 2 (GM)"'s
    # centroid falls inside Agnewak and its historical area (14,432 km2) is
    # 0.66x Agnewak's current area (21,998 km2) -- close enough to be a
    # genuine boundary adjustment, not a multi-way split.
    "Zone 2 (GM)": "Agnewak",
}

# One historical FDW admin_2 name -> multiple current zones. The reported
# value is attributed to every child zone, flagged via a *_note column.
ZONE_SPLITS = {
    "Gamo Gofa": ["Gamo", "Gofa"],
    "Kembata Alaba Tembaro": ["Kembata Tembaro", "Halaba"],
    "Keficho Shekicho": ["Kefa", "Sheka"],
    "Dire Dawa": ["Dire Dawa urban", "Dire Dawa rural"],
}

# Investigated and deliberately left unmatched -- see docstring case 3.
# Not an exhaustive list of every possible FDW name; anything not in
# ZONE_ALIASES/ZONE_SPLITS/our 92 names simply falls through to "unmatched"
# and is reported (not silently dropped) by the aggregation step below.
KNOWN_UNRESOLVABLE = {
    "Moyale", "North Omo", "Segen Area Peoples Zone",
    "Argoba Special Woreda (AM)", "Pawe Special Woreda",
    "Zone 1 (GM)",
}


def load_zone_lookup():
    df = pd.read_csv(LIVELIHOOD_CSV)
    return df[["zone_code", "zone_name", "dominant_livelihood_zone"]].copy()


def resolve_zone(admin_2_name, our_names):
    """Return a list of zone_names this FDW admin_2 name maps to (usually
    one, sometimes >1 for a documented historical split, sometimes empty
    for an unmatched name), plus a note string for split attributions."""
    if admin_2_name in our_names:
        return [admin_2_name], None
    if admin_2_name in ZONE_ALIASES:
        return [ZONE_ALIASES[admin_2_name]], None
    if admin_2_name in ZONE_SPLITS:
        children = ZONE_SPLITS[admin_2_name]
        return children, f"inherited_from_split_parent:{admin_2_name}"
    return [], None


# --------------------------------------------------------------------------
# Load + resolve raw records
# --------------------------------------------------------------------------

def load_crop_records(crop):
    path = RAW_DIR / f"cropproductionfacts_{crop}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run fetch_agss_yields_admin2.py first.")
    with open(path) as f:
        return json.load(f)


def harvest_year_from_season(season_year):
    """'Meher 2001' -> 2001. Robust to any stray whitespace/casing."""
    m = re.search(r"(\d{4})", season_year or "")
    return int(m.group(1)) if m else None


def build_zone_year_table(crop, our_names):
    records = load_crop_records(crop)
    meher = [r for r in records if r.get("season_name") == "Meher"]

    unmatched_names = set()
    rows = []
    for r in meher:
        admin_2 = r.get("admin_2")
        if not admin_2:
            continue  # national/regional-aggregate rows, not zone-level
        zone_names, note = resolve_zone(admin_2, our_names)
        if not zone_names:
            unmatched_names.add(admin_2)
            continue
        harvest_year = harvest_year_from_season(r.get("season_year"))
        if harvest_year is None:
            continue
        status = r.get("status")
        value = r.get("value")
        for zn in zone_names:
            rows.append({
                "zone_name": zn,
                "harvest_year": harvest_year,
                "value": value,
                "status": status,
                "note": note,
                "fdw_admin_2": admin_2,
            })

    df = pd.DataFrame(rows)
    print(f"  {crop}: {len(meher)} Meher records -> {len(df)} zone-year rows "
          f"({len(unmatched_names)} unmatched admin_2 names, "
          f"{df['zone_name'].nunique() if len(df) else 0} zones resolved)")
    if unmatched_names:
        print(f"    unmatched (dropped): {sorted(unmatched_names)}")

    if df.empty:
        return df, unmatched_names

    # Deduplicate per (zone_name, harvest_year): multiple fnid vintages can
    # report the same zone-year (see docstring). Prefer a Collected row with
    # a real value; fall back to any row otherwise. Sort so the preferred
    # row comes first within each group, then keep only that first row --
    # avoids groupby().apply()'s column-dropping behavior in pandas 3.0.
    df = df.assign(_priority=~((df["status"] == "Collected") & df["value"].notna()))
    df = df.sort_values(["zone_name", "harvest_year", "_priority"])
    dedup = df.drop_duplicates(subset=["zone_name", "harvest_year"], keep="first")
    dedup = dedup.drop(columns="_priority").reset_index(drop=True)
    n_dupes = len(df) - len(dedup)
    if n_dupes:
        print(f"    collapsed {n_dupes} duplicate zone-year records "
              f"(multiple fnid vintages reporting the same zone-year)")
    return dedup, unmatched_names


# --------------------------------------------------------------------------
# Engineer: status categories, anomaly, publication-lag monthly broadcast
# --------------------------------------------------------------------------

def classify_status(row):
    if pd.notna(row.get("value")):
        return "collected"
    return "not_collected_or_available"


def compute_anomaly(panel_years, values):
    """Backward-looking % deviation from the zone-crop's own expanding
    mean of PRIOR years only (never including the current or future year --
    CLAUDE.md Sec 3.2 leakage rule). Requires at least
    MIN_ANOMALY_HISTORY_YEARS of prior collected values; NaN before that."""
    order = np.argsort(panel_years)
    years_sorted = np.array(panel_years)[order]
    vals_sorted = np.array(values, dtype="float64")[order]

    anomalies = np.full(len(vals_sorted), np.nan)
    history = []
    for i in range(len(vals_sorted)):
        if len(history) >= MIN_ANOMALY_HISTORY_YEARS:
            baseline = np.mean(history)
            if baseline > 0 and not np.isnan(vals_sorted[i]):
                anomalies[i] = (vals_sorted[i] - baseline) / baseline
        if not np.isnan(vals_sorted[i]):
            history.append(vals_sorted[i])

    out = np.full(len(vals_sorted), np.nan)
    out[order] = anomalies
    return out


def month_range(start, end):
    y, m = start
    ey, em = end
    months = []
    while (y, m) <= (ey, em):
        months.append((y, m))
        m += 1
        if m == 13:
            m = 1
            y += 1
    return months


def visibility_window(harvest_year):
    """First (year, month) this harvest_year's value is visible, and the
    (year, month) AFTER which it expires if no newer harvest year exists
    to supersede it (12-month natural window)."""
    start_year = harvest_year + PUBLICATION_LAG_YEAR_OFFSET
    start = (start_year, PUBLICATION_LAG_MONTH)
    end_year, end_month = start_year, PUBLICATION_LAG_MONTH - 1
    end_year += 1
    if end_month == 0:
        end_month = 12
        end_year -= 1
    return start, (end_year, end_month)


def broadcast_crop_to_monthly(crop, zone_year_df, zone_code, zone_name):
    """Build the monthly panel for one (crop, zone), applying the
    publication-lag visibility rule and the no-forward-fill-past-ceiling
    rule (see module docstring)."""
    months = month_range(PANEL_START, PANEL_END)
    out = pd.DataFrame(months, columns=["year", "month"])
    out["zone_code"] = zone_code

    zdf = zone_year_df[zone_year_df["zone_name"] == zone_name].sort_values("harvest_year")
    if zdf.empty:
        out[f"{crop}_yield"] = np.nan
        out[f"{crop}_yield_anomaly"] = np.nan
        out[f"{crop}_yield_status"] = "not_surveyed"
        out[f"{crop}_yield_note"] = None
        return out

    anomalies = compute_anomaly(zdf["harvest_year"].tolist(), zdf["value"].tolist())
    zdf = zdf.assign(anomaly=anomalies, status_cat=zdf.apply(classify_status, axis=1))

    # For each month, find the harvest year whose 12-month visibility
    # window covers it, preferring the MOST RECENT harvest year that has
    # already become visible by that month (a later report always
    # supersedes an earlier one once both are visible).
    yield_col = np.full(len(out), np.nan)
    anomaly_col = np.full(len(out), np.nan)
    status_col = np.array(["not_surveyed"] * len(out), dtype=object)
    note_col = np.array([None] * len(out), dtype=object)

    records = []
    for _, r in zdf.iterrows():
        vis_start, vis_end = visibility_window(int(r["harvest_year"]))
        records.append((vis_start, vis_end, r))
    records.sort(key=lambda x: x[0])  # by visibility start

    for idx, (yr, mo) in enumerate(zip(out["year"], out["month"])):
        cur = (yr, mo)
        candidate = None
        for vis_start, vis_end, r in records:
            if vis_start <= cur <= vis_end:
                candidate = r  # later entries in sorted order override earlier ones
        if candidate is not None:
            yield_col[idx] = candidate["value"] if pd.notna(candidate["value"]) else np.nan
            anomaly_col[idx] = candidate["anomaly"]
            status_col[idx] = candidate["status_cat"]
            note_col[idx] = candidate["note"]
        else:
            # No harvest year's window covers this month. Before the first
            # window: genuinely not yet in scope (not_surveyed-style NaN,
            # but this zone DOES have AgSS history -- use a distinct
            # pre-history status so it isn't confused with a structural
            # non-survey zone).
            first_start = records[0][0] if records else None
            last_end = records[-1][1] if records else None
            if first_start is not None and cur < first_start:
                status_col[idx] = "before_agss_history"
            elif last_end is not None and cur > last_end:
                status_col[idx] = f"no_data_post_{last_end[0]}_{last_end[1]:02d}"
            else:
                status_col[idx] = "not_surveyed"

    out[f"{crop}_yield"] = yield_col
    out[f"{crop}_yield_anomaly"] = anomaly_col
    out[f"{crop}_yield_status"] = status_col
    out[f"{crop}_yield_note"] = note_col
    return out


def main():
    print("=" * 74)
    print("AgSS yield AGGREGATE + ENGINEER")
    print("=" * 74)

    zones = load_zone_lookup()
    our_names = set(zones["zone_name"])

    print("\n[1] Loading + resolving zone-year tables per crop")
    zone_year_tables = {}
    all_unmatched = set()
    for crop in CROPS:
        df, unmatched = build_zone_year_table(crop, our_names)
        zone_year_tables[crop] = df
        all_unmatched |= unmatched

    print(f"\n  total distinct unmatched admin_2 names across all crops: {len(all_unmatched)}")

    print("\n[2] Broadcasting to monthly panel per zone (publication-lag rule applied)")
    panels = []
    for _, zrow in zones.iterrows():
        zone_code, zone_name = zrow["zone_code"], zrow["zone_name"]
        crop_frames = []
        for crop in CROPS:
            cf = broadcast_crop_to_monthly(crop, zone_year_tables[crop], zone_code, zone_name)
            crop_frames.append(cf)
        merged = crop_frames[0]
        for cf in crop_frames[1:]:
            merged = merged.merge(cf, on=["year", "month", "zone_code"], how="left")
        panels.append(merged)

    panel = pd.concat(panels, ignore_index=True)

    # agss_surveyed_flag: static per zone -- 1 if this zone has AT LEAST ONE
    # REAL (Collected, non-null) yield value for any of the 4 crops in any
    # year, 0 otherwise. Deliberately NOT "appears in the crop-facts table
    # at all" -- FDW's own "Not Collected"/"Not Available" rows mean the
    # zone was nominally on that round's roster but no figure was ever
    # obtained, which is functionally indistinguishable from "not surveyed"
    # for a modeler deciding whether this zone's yield features are usable.
    # Checked directly: 8 zones (7 pastoral + 1 agropastoral -- Doolo,
    # Korahe, Fanti-Zone 4, Daawa, Nogob, Erer, Afder, Alle) have ZERO
    # collected values across the full 1995-2024 window despite sometimes
    # appearing in the raw table via placeholder rows -- this is the
    # structural coverage gap the task asked to quantify, and this flag
    # would silently hide it if defined more loosely.
    ever_collected = set()
    for crop in CROPS:
        zdf = zone_year_tables[crop]
        ever_collected |= set(zdf.loc[zdf["value"].notna(), "zone_name"])
    zone_to_code = dict(zip(zones["zone_name"], zones["zone_code"]))
    surveyed_codes = {zone_to_code[zn] for zn in ever_collected if zn in zone_to_code}
    panel["agss_surveyed_flag"] = panel["zone_code"].isin(surveyed_codes).astype(int)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "agss_yields_admin2_monthly.csv"
    col_order = ["zone_code", "year", "month", "agss_surveyed_flag"]
    for crop in CROPS:
        col_order += [f"{crop}_yield", f"{crop}_yield_anomaly",
                      f"{crop}_yield_status", f"{crop}_yield_note"]
    panel = panel[col_order]
    panel.to_csv(out_path, index=False)

    print(f"\n[3] Wrote {out_path} ({len(panel):,} rows, {panel['zone_code'].nunique()} zones)")

    print("\n[4] Coverage summary")
    print(f"  zones with agss_surveyed_flag=1: {panel.groupby('zone_code')['agss_surveyed_flag'].first().sum()} / 92")
    livelihood = zones.set_index("zone_code")["dominant_livelihood_zone"]
    flag_by_zone = panel.groupby("zone_code")["agss_surveyed_flag"].first()
    for lz_type in livelihood.unique():
        zcs = livelihood[livelihood == lz_type].index
        n = len(zcs)
        surveyed = flag_by_zone.reindex(zcs).sum()
        print(f"    {lz_type}: {surveyed}/{n} zones surveyed")

    for crop in CROPS:
        status_counts = panel[f"{crop}_yield_status"].value_counts()
        print(f"\n  {crop}_yield_status distribution (zone-months):")
        for status, count in status_counts.items():
            print(f"    {status}: {count:,}")

    print("\nDone. Run validate_agss_yields_output.py next.")


if __name__ == "__main__":
    main()
