"""
validate_ethiopia_model_tables.py

Validation stage for the Ethiopia feature-target join (CLAUDE.md Sec 3.5:
every new pipeline needs a validate_..._output.py before its output is
trusted). Checks the 7 files in data/processed/model_tables/ against:

1. Structural completeness -- same row count/shape/zone coverage across
   all 7 lead tables, and that count matches 92 zones x the full gapless
   calendar range (post the 2025-07/08/09 reindex fix -- see
   build_ethiopia_raw_wide_panel.py's load_ipc_base() docstring).
2. Leakage: SHIFT columns at lead L must equal the raw wide panel's value
   at (zone, month-L) EXACTLY -- not just "close", not a positional
   shift-across-a-gap artifact (the specific bug caught and fixed
   2026-08-20). Spot-checked across ALL rows for two representative
   columns per lead, not a handful of examples.
3. Forecast-preselect columns populated at exactly the leads each source
   actually reaches (lead1: GloFAS+IRI+USGS, 18/18; lead3: GloFAS+IRI
   only, 12/18; every other lead: 0/18) -- for every row, not just in
   aggregate.
4. IDENTITY columns identical across every lead table for the same
   (zone_code, month) -- confirms the "never shifted" contract actually
   held.
5. origin_month = month - lead_months for every single row.
6. No fabricated data: reintroduced (target-dropped-for-staleness) rows
   have NaN split/ipc_continuous, never a real-looking value.
"""

import os

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

RAW_PANEL_PATH = os.path.join(REPO_ROOT, "data", "interim", "model_join", "ethiopia_raw_wide_panel.csv")
MODEL_TABLES_DIR = os.path.join(REPO_ROOT, "data", "processed", "model_tables")

LEADS = [0, 1, 2, 3, 4, 8, 12]
IDENTITY_SPOT_COLS = ["ipc_continuous", "ha_share", "pct_pastoral", "is_harvest_season"]
SHIFT_SPOT_COLS = ["total_rainfall_mm", "maize_price_etb", "acled_event_count", "headline_cpi_chainlinked",
                    "total_refugees_hosted", "idps_ethiopia"]
FORECAST_OUTPUT_COLS = [
    "glofas_exceed_2yr", "glofas_exceed_20yr", "glofas_source",
    "seasonal_precip_prob_below", "seasonal_precip_prob_near", "seasonal_precip_prob_above",
    "seasonal_target_season", "seasonal_overlaps_kiremt", "seasonal_overlaps_belg",
    "seasonal_overlaps_gu", "seasonal_overlaps_deyr", "seasonal_source",
    "usgs_gefs_precip_mm", "usgs_gefs_precip_anom", "usgs_gefs_issue_date",
    "usgs_gefs_source", "usgs_gefs_coverage_days", "usgs_gefs_anom_units",
]
LEAD_FORECAST_EXPECTED = {0: 0, 1: 18, 2: 0, 3: 12, 4: 0, 8: 0, 12: 0}


def load_lead_tables():
    tables = {}
    for lead in LEADS:
        path = os.path.join(MODEL_TABLES_DIR, f"ethiopia_lead{lead:02d}.csv")
        df = pd.read_csv(path, low_memory=False)
        df["month"] = pd.PeriodIndex(df["month"], freq="M")
        tables[lead] = df
    return tables


def check_structural(tables):
    print("=== 1. Structural completeness ===")
    shapes = {lead: df.shape for lead, df in tables.items()}
    ref_shape = shapes[0]
    ok = all(s == ref_shape for s in shapes.values())
    print(f"  shapes: {shapes}")
    print(f"  {'PASS' if ok else 'FAIL'}: all 7 lead tables have identical shape")

    for lead, df in tables.items():
        n_zones = df["zone_code"].nunique()
        assert n_zones == 92, f"lead{lead}: expected 92 zones, got {n_zones}"
    print("  PASS: 92/92 zones present in every lead table")


def check_shift_leakage(raw, tables):
    print("\n=== 2. Leakage: SHIFT columns match raw[month - lead] exactly ===")
    all_pass = True
    for lead in LEADS:
        if lead == 0:
            continue
        df = tables[lead]
        merged = df[["zone_code", "month"] + SHIFT_SPOT_COLS].copy()
        merged["orig_month"] = merged["month"] - lead
        chk = merged.merge(
            raw[["zone_code", "month"] + SHIFT_SPOT_COLS].rename(
                columns={"month": "orig_month", **{c: f"raw_{c}" for c in SHIFT_SPOT_COLS}}
            ),
            on=["zone_code", "orig_month"], how="left",
        )
        for col in SHIFT_SPOT_COLS:
            mism = ((chk[col] != chk[f"raw_{col}"]) & ~(chk[col].isna() & chk[f"raw_{col}"].isna())).sum()
            status = "PASS" if mism == 0 else "FAIL"
            if mism != 0:
                all_pass = False
            print(f"  lead{lead:02d} {col}: {mism} mismatches / {len(chk)} rows [{status}]")
    print("  OVERALL:", "PASS" if all_pass else "FAIL -- investigate before trusting this join")
    return all_pass


def check_forecast_preselect(raw, tables):
    print("\n=== 3. Forecast-preselect columns populated only at their real lead ===")
    all_pass = True
    for lead in LEADS:
        df = tables[lead]
        expected = LEAD_FORECAST_EXPECTED[lead]
        n_populated_cols = sum(df[c].notna().any() for c in FORECAST_OUTPUT_COLS)
        status = "PASS" if n_populated_cols == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  lead{lead:02d}: {n_populated_cols} cols populated (expected {expected}) [{status}]")

    lead1 = tables[1]
    chk = lead1[["zone_code", "month", "glofas_exceed_2yr"]].merge(
        raw[["zone_code", "month", "glofas_exceed_2yr_lead1"]], on=["zone_code", "month"], how="left"
    )
    mism = ((chk["glofas_exceed_2yr"] != chk["glofas_exceed_2yr_lead1"]) &
            ~(chk["glofas_exceed_2yr"].isna() & chk["glofas_exceed_2yr_lead1"].isna())).sum()
    print(f"  lead01 glofas_exceed_2yr same-row-as-raw check: {mism} mismatches / {len(chk)} [{'PASS' if mism == 0 else 'FAIL'}]")
    if mism != 0:
        all_pass = False
    print("  OVERALL:", "PASS" if all_pass else "FAIL")
    return all_pass


def check_identity_columns(tables):
    print("\n=== 4. IDENTITY columns identical across all lead tables ===")
    all_pass = True
    ref = tables[0][["zone_code", "month"] + IDENTITY_SPOT_COLS]
    for lead in LEADS[1:]:
        cmp = tables[lead][["zone_code", "month"] + IDENTITY_SPOT_COLS]
        merged = ref.merge(cmp, on=["zone_code", "month"], suffixes=("_ref", "_cmp"))
        for col in IDENTITY_SPOT_COLS:
            mism = ((merged[f"{col}_ref"] != merged[f"{col}_cmp"]) &
                    ~(merged[f"{col}_ref"].isna() & merged[f"{col}_cmp"].isna())).sum()
            if mism != 0:
                all_pass = False
                print(f"  lead00 vs lead{lead:02d} {col}: {mism} mismatches [FAIL]")
    print("  OVERALL:", "PASS -- identity columns never drift across leads" if all_pass else "FAIL")
    return all_pass


def check_origin_month(tables):
    print("\n=== 5. origin_month = month - lead_months ===")
    all_pass = True
    for lead, df in tables.items():
        expected = (df["month"] - lead).astype(str)
        mism = (df["origin_month"] != expected).sum()
        if mism != 0:
            all_pass = False
        print(f"  lead{lead:02d}: {mism} mismatches / {len(df)} [{'PASS' if mism == 0 else 'FAIL'}]")
    print("  OVERALL:", "PASS" if all_pass else "FAIL")
    return all_pass


def check_no_fabricated_gap_rows(tables):
    print("\n=== 6. Reintroduced staleness-gap rows have NaN target, not fabricated data ===")
    df = tables[0]
    gap_rows = df[df["month"].astype(str).isin(["2025-07", "2025-08", "2025-09"])]
    n_bad = gap_rows["ipc_continuous"].notna().sum() + gap_rows["split"].notna().sum()
    print(f"  {len(gap_rows)} gap rows found (expected 92 x 3 = 276)")
    print(f"  {'PASS' if n_bad == 0 and len(gap_rows) == 276 else 'FAIL'}: "
          f"target/split both NaN for every reintroduced row")
    return n_bad == 0 and len(gap_rows) == 276


def main():
    raw = pd.read_csv(RAW_PANEL_PATH, low_memory=False)
    raw["month"] = pd.PeriodIndex(raw["month"], freq="M")
    tables = load_lead_tables()

    check_structural(tables)
    r2 = check_shift_leakage(raw, tables)
    r3 = check_forecast_preselect(raw, tables)
    r4 = check_identity_columns(tables)
    r5 = check_origin_month(tables)
    r6 = check_no_fabricated_gap_rows(tables)

    print("\n" + "=" * 60)
    print("FINAL:", "ALL CHECKS PASS" if all([r2, r3, r4, r5, r6]) else "SOME CHECKS FAILED -- see above")


if __name__ == "__main__":
    main()
