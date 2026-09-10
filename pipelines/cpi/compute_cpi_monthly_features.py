"""
compute_cpi_monthly_features.py

Engineering stage for the headline + food CPI feature. Merges the two raw
sources fetched by fetch_cpi.py into a single Ethiopia-only monthly table
and writes data/processed/features/cpi_monthly.csv.

NATIONAL-ONLY, no spatial join anywhere in this file -- same broadcast
pattern as imf_gdp/teleconnections/wvg. The eventual multi-source feature
join must broadcast this table's rows to every zone_code for the matching
month (plain many-to-one merge on year+month at join time, NOT baked into
this file).

WHAT EACH SOURCE CONTRIBUTES (see docs/CPI_Pipeline_Documentation.md for
the full reasoning):
  - Ha et al. (World Bank Global Database of Inflation): the ONLY source
    of the raw index LEVEL (headline_cpi, food_cpi). Ethiopia headline
    reaches further (~2025-03) than food (~2023-06) in the fetched
    vintage -- both are real published figures, on their own base period
    (confirmed distinct even within Ethiopia -- headline base 2010, food
    base 2016-12 in the earlier vintage checked; the live vintage no
    longer documents a base date for headline at all, i.e. it may have
    been rebased -- yet another reason a raw level should never be
    compared *across* countries, only within Ethiopia's own series, and
    even then, only via % change, never a raw-level ratio spanning a
    silent rebasing).
  - ESS monthly bulletins: extend the YoY % change past Ha et al.'s
    coverage, through the present. ESS bulletins report inflation RATES,
    never an index level -- there is nothing to read a level off directly.
    Chain-linking a level off these rates was originally deferred
    (confirmed with the project owner 2026-08-09) because it would
    silently splice two potentially different base periods if ESS's own
    index was rebased at any point after Ha et al.'s last known level --
    undetectable from a rate alone. **Un-deferred 2026-08-20**: all four
    cached ESS bulletins (Nov 2023, Dec 2024, Dec 2025, Jun 2026 --
    spanning both the food and headline cutoffs end-to-end) explicitly
    state the identical base period ("December EFY2009 = 100", i.e.
    December 2016 = 100 -- the same base Ha et al.'s live vintage is
    already confirmed to use, see Sec 5 below), so the previously-
    undetectable risk is now directly checked and ruled out for this
    window. `headline_cpi_chainlinked`/`food_cpi_chainlinked` (see
    build_monthly_table()) reconstruct the level past cutoff via
    level(t) = level(t-12) x (1 + yoy_change(t)), kept as separate columns
    from headline_cpi/food_cpi (which still stay NaN past cutoff,
    unchanged) so a reconstructed figure is never mistaken for an
    original Ha et al. one.

VINTAGE/LEAKAGE (mirrors the imf_gdp *_undated / teleconnections
lag0-vs-lag1/3/6 pattern, confirmed with the project owner 2026-08-09):
CPI is periodically revised (Ha et al.'s database is updated twice a
year; empirically, Ethiopia's own headline series was REBASED --
re-spliced from an internal legacy 2010=100 base onto Ethiopia's actual
official base period, December 2016=100 -- and its published history
shortened, between the May-2023 and April-2025 vintages checked during
build; see docs/CPI_Pipeline_Documentation.md Sec 5 for the confirmed
mechanism). There is no bulk historical-vintage archive (same
imf.org-style dead end as GDP), so this file ships THREE tiers, not two:

  - headline_cpi_undated, headline_cpi_yoy_change_undated,
    food_cpi_undated, food_cpi_yoy_change_undated: Busker et al.'s own
    May-2023-vintage figures, straight from their staged
    busker_comparison/Inflation_WB.xlsx (load_busker_undated()). This is
    what RQ1's "Busker et al. dataset" reproduction run should use for
    CPI -- NOT the current-release columns below, which are on a
    different (rebased) base for headline and therefore NOT
    RQ1-comparable. No lag variants (matches imf_gdp's undated columns --
    a static snapshot, not an RQ2+ modeling input).
  - headline_cpi, headline_cpi_yoy_change, food_cpi, food_cpi_yoy_change:
    the CURRENT-release value for each column's own month. Needed because
    Busker's own vintage has no data past ~2023 at all, and this
    project's target extends through 2026 -- reference-table completeness
    only. Do NOT use these bare (lag0) columns in any RQ2+ model, and do
    NOT use headline_cpi (the level) in any RQ1 comparison either.
  - {col}_lag{1,3,6,12} (current-release columns only): the RQ2+ default,
    far less exposed to still-provisional recent months or an undetected
    rebasing right at the series boundary. Lag windows match the WFP
    prices pipeline's granularity (1/3/6/12), since CPI is monthly
    market-adjacent economic data like WFP prices, not a coarser
    teleconnection index.
"""

import os
import re

import numpy as np
import openpyxl
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
RAW_DIR = os.path.join(REPO_ROOT, "data", "raw", "cpi")
HA_ET_AL_PATH = os.path.join(RAW_DIR, "Inflation-data.xlsx")
BUSKER_XLSX = os.path.join(SCRIPT_DIR, "busker_comparison", "Inflation_WB.xlsx")
ESS_DIR = os.path.join(RAW_DIR, "ess_bulletins")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "cpi_monthly.csv")

FEATURE_START_YEAR = 2009  # matches the other feature pipelines' (teleconnections, NDVI, GLEAM) feature-year window
LAGS = [1, 3, 6, 12]

ESS_BULLETIN_FILES = ["cpi_nov_2023.pdf", "cpi_dec_2024.pdf", "cpi_dec_2025.pdf", "cpi_jun_2026.pdf"]

MONTH_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}
MONTH_NAMES_PATTERN = "|".join(MONTH_NUM.keys())

# ESS Table 1 data row: "<Month>[-EFY <year>]   <general>   <food>   <nonfood>"
# EFY suffix only appears on the first and last row of each bulletin's
# trailing-12-month block; middle rows are bare month names. Exactly 3
# numeric columns after the month label is what distinguishes this table
# from ESS's other tables (Table 3 month-to-month has 1 column, Tables
# 4/5 12-month-moving-average have 2) -- see docs/CPI_Pipeline_Documentation.md
# Sec 3 for why this, not a "Table 1:" title-line search, is the robust
# anchor (the title line itself was observed corrupted by PDF text-flow
# interleaving in more than one bulletin).
ROW_RE = re.compile(
    rf"^\s*({MONTH_NAMES_PATTERN})(?:-\s*EFY\s*(\d+))?\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s*$"
)


# --------------------------------------------------------------------------
# Ha et al. (World Bank Global Database of Inflation) -- index levels
# --------------------------------------------------------------------------

def _load_ha_et_al_series(path, sheet_name):
    """Extracts Ethiopia's monthly index level from one Ha et al. sheet
    (hcpi_m or fcpi_m) in the given workbook -- shared by the live-fetched
    file and the older Busker-staged snapshot, since both are the exact
    same database format (see docs/CPI_Pipeline_Documentation.md Sec 2).
    Columns are 'Country Code', ..., then integer YYYYMM date columns,
    then trailing metadata columns ('Data source', 'Base date', 'Note',
    ...) -- date columns are identified by dtype (int, in a plausible
    YYYYMM range), not by position, since the exact trailing metadata
    columns differ between the two sheets AND between vintages (the live
    vintage dropped the 'Base date' column entirely for hcpi_m -- see
    Sec 5)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet_name]
    rows = list(ws.iter_rows(values_only=True))
    header = rows[0]
    date_cols = [i for i, h in enumerate(header) if isinstance(h, int) and 190000 < h < 300000]
    eth_matches = [r for r in rows[1:] if r[0] == "ETH"]
    if len(eth_matches) != 1:
        raise ValueError(f"{path} {sheet_name}: expected exactly one ETH row, found {len(eth_matches)}")
    eth = eth_matches[0]

    records = []
    for i in date_cols:
        val = eth[i]
        if val is not None:
            yyyymm = header[i]
            records.append({"year": yyyymm // 100, "month": yyyymm % 100, "level": float(val)})
    df = pd.DataFrame(records).sort_values(["year", "month"]).reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{path} {sheet_name}: no ETH observations parsed -- format may have changed")
    return df


def load_ha_et_al():
    headline = _load_ha_et_al_series(HA_ET_AL_PATH, "hcpi_m").rename(columns={"level": "headline_cpi"})
    food = _load_ha_et_al_series(HA_ET_AL_PATH, "fcpi_m").rename(columns={"level": "food_cpi"})
    merged = headline.merge(food, on=["year", "month"], how="outer")
    return merged.sort_values(["year", "month"]).reset_index(drop=True)


def load_busker_undated():
    """Busker et al.'s own May-2023-vintage snapshot of the SAME Ha et al.
    database (busker_comparison/Inflation_WB.xlsx) -- for RQ1 fidelity
    ONLY, mirroring imf_gdp's gdp_per_capita_undated /
    teleconnections' lag0 pattern. NOT the current-release headline_cpi/
    food_cpi columns above, which are rebased/extended and NOT
    RQ1-comparable (see Sec 5 of the doc) -- this is what an RQ1 "Busker
    et al. dataset" reproduction run should actually use for CPI."""
    headline = _load_ha_et_al_series(BUSKER_XLSX, "hcpi_m").rename(columns={"level": "headline_cpi_undated"})
    food = _load_ha_et_al_series(BUSKER_XLSX, "fcpi_m").rename(columns={"level": "food_cpi_undated"})
    merged = headline.merge(food, on=["year", "month"], how="outer")
    return merged.sort_values(["year", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# ESS bulletins -- YoY % change only, extends past Ha et al.'s coverage
# --------------------------------------------------------------------------

def _efy_month_to_gregorian_year(month_name, efy_year):
    """Ethiopian New Year falls ~Sept 11: a bulletin's Sep-Dec rows share
    the SAME EFY year as the following Jan-Aug rows (EFY year increments
    at the Ethiopian new year, not at the Gregorian one), so the Gregorian
    year offset differs by whether the row's month falls before or after
    that boundary. Confirmed empirically against all 4 fetched bulletins'
    own filename/issue-date anchors (e.g. "November EFY2016" bulletin ==
    the cpi_nov_2023 download == Gregorian November 2023 == 2016+7)."""
    month_num = MONTH_NUM[month_name]
    return efy_year + 7 if month_num >= 9 else efy_year + 8


def _pdftotext_layout(pdf_path):
    import subprocess

    result = subprocess.run(
        ["pdftotext", "-layout", pdf_path, "-"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def parse_ess_bulletin(pdf_path):
    """Extracts Table 1's trailing-12(-13)-month General/Food YoY rates.
    Returns a DataFrame with year, month, ess_headline_yoy, ess_food_yoy
    (as fractions, matching pandas .pct_change() convention -- ESS reports
    percentages, so values are divided by 100)."""
    text = _pdftotext_layout(pdf_path)
    matches = []
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            matches.append(m.groups())

    # Table 1 is the FIRST run of 3-numeric-column month rows in the
    # document -- Table 3 (month-to-month, 1 column) and Tables 4/5
    # (12-month moving average, 2 columns) never match ROW_RE at all, and
    # the "July EFYxx - June EFYyy" multi-year summary rows above Table 1
    # don't match either (they don't start with a bare month name), so no
    # explicit "Table 1:" title search is needed -- confirmed robust
    # against the one bulletin where that title line was itself corrupted
    # by PDF text-flow interleaving.
    if len(matches) < 12:
        raise ValueError(f"{pdf_path}: only found {len(matches)} Table-1-shaped rows, expected >=12")

    # Anchor: the LAST matched row always carries an explicit EFY suffix
    # (confirmed across all 4 bulletins) -- walk backward assigning each
    # prior row the preceding calendar month, rather than parsing the EFY
    # rollover mid-table (most rows have no EFY suffix at all).
    last_month_name, last_efy, _, _, _ = matches[-1]
    if last_efy is None:
        raise ValueError(f"{pdf_path}: last Table 1 row has no EFY-year anchor -- cannot date this table")
    anchor_year = _efy_month_to_gregorian_year(last_month_name, int(last_efy))
    anchor_month = MONTH_NUM[last_month_name]

    records = []
    n = len(matches)
    for offset, (month_name, efy, general, food, nonfood) in enumerate(matches):
        months_before_anchor = (n - 1) - offset
        total_month_index = anchor_year * 12 + (anchor_month - 1) - months_before_anchor
        year, month = divmod(total_month_index, 12)
        month += 1

        if efy is not None:
            # Cross-check every explicitly-labeled row (not just the
            # anchor) against the same conversion rule, independent of the
            # backward walk -- catches any off-by-one in the walk itself.
            expected_year = _efy_month_to_gregorian_year(month_name, int(efy))
            if expected_year != year or MONTH_NUM[month_name] != month:
                raise ValueError(
                    f"{pdf_path}: EFY-labeled row '{month_name}-EFY{efy}' resolved to {year}-{month:02d} "
                    f"by backward walk but {expected_year}-{MONTH_NUM[month_name]:02d} by direct conversion"
                )

        records.append({
            "year": year, "month": month,
            "ess_headline_yoy": float(general) / 100.0,
            "ess_food_yoy": float(food) / 100.0,
        })

    return pd.DataFrame(records)


def load_ess_bulletins():
    frames = [parse_ess_bulletin(os.path.join(ESS_DIR, f)) for f in ESS_BULLETIN_FILES]
    combined = pd.concat(frames, ignore_index=True)

    # Multiple bulletins deliberately overlap (e.g. Dec-2025 and Jun-2026
    # both cover Jun-Dec-2025) as an internal cross-check -- assert they
    # actually agree before silently collapsing to one value per month.
    dupes = combined.groupby(["year", "month"])
    for (year, month), grp in dupes:
        if len(grp) > 1:
            for col in ["ess_headline_yoy", "ess_food_yoy"]:
                if grp[col].max() - grp[col].min() > 1e-9:
                    raise ValueError(
                        f"ESS bulletins disagree for {year}-{month:02d} on {col}: {grp[col].tolist()}"
                    )
    return combined.drop_duplicates(["year", "month"]).sort_values(["year", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Merge + feature engineering
# --------------------------------------------------------------------------

def build_monthly_table():
    ha = load_ha_et_al()
    ess = load_ess_bulletins()
    busker = load_busker_undated()

    start = pd.Timestamp(FEATURE_START_YEAR, 1, 1)
    end = pd.Timestamp(max(ha["year"].max(), ess["year"].max()), 12, 1)
    full_range = pd.date_range(start, end, freq="MS")
    grid = pd.DataFrame({"date": full_range, "year": full_range.year, "month": full_range.month})

    grid = grid.merge(ha, on=["year", "month"], how="left")
    grid = grid.merge(ess, on=["year", "month"], how="left")
    grid = grid.merge(busker, on=["year", "month"], how="left")
    grid = grid.sort_values("date").reset_index(drop=True)

    # RQ1-fidelity companion columns (mirrors imf_gdp's *_undated pattern):
    # Busker et al.'s own May-2023-vintage figures, straight from their
    # staged file, no lag protection -- for feeding an RQ1 "Busker et al.
    # dataset" reproduction run, NOT for RQ2+ modeling. Computed the same
    # explicit-division way as the current-release columns below (see that
    # comment for why, not pandas' .pct_change()).
    grid["headline_cpi_yoy_change_undated"] = (
        grid["headline_cpi_undated"] / grid["headline_cpi_undated"].shift(12) - 1
    )
    grid["food_cpi_yoy_change_undated"] = grid["food_cpi_undated"] / grid["food_cpi_undated"].shift(12) - 1

    # YoY change: computed from the Ha et al. level wherever both m and
    # m-12 have a level (standard index(m)/index(m-12)-1 definition);
    # ESS's own directly-published rate fills in months past that
    # coverage. The two never actually compete for the same month in
    # practice (ESS bulletins were chosen to start right where Ha et al.
    # leaves off), but Ha et al. wins on conflict since it's the more
    # precise, index-level-derived figure.
    #
    # Deliberately NOT pandas' .pct_change(12): older pandas (pre-2.1
    # default-change) silently forward-fills NaNs before computing the
    # ratio unless fill_method=None is passed, which would fabricate a
    # rate from a stale, frozen level for every month past food_cpi's
    # real coverage -- confirmed to actually happen on at least one
    # environment used against this pipeline (2026-08-09). The explicit
    # division is unambiguous regardless of pandas version/defaults.
    grid["headline_cpi_yoy_change"] = grid["headline_cpi"] / grid["headline_cpi"].shift(12) - 1
    grid["food_cpi_yoy_change"] = grid["food_cpi"] / grid["food_cpi"].shift(12) - 1

    grid["headline_cpi_yoy_change_source"] = np.where(grid["headline_cpi_yoy_change"].notna(), "ha_et_al", None)
    grid["food_cpi_yoy_change_source"] = np.where(grid["food_cpi_yoy_change"].notna(), "ha_et_al", None)

    fill_headline = grid["headline_cpi_yoy_change"].isna() & grid["ess_headline_yoy"].notna()
    grid.loc[fill_headline, "headline_cpi_yoy_change"] = grid.loc[fill_headline, "ess_headline_yoy"]
    grid.loc[fill_headline, "headline_cpi_yoy_change_source"] = "ess_bulletin"

    fill_food = grid["food_cpi_yoy_change"].isna() & grid["ess_food_yoy"].notna()
    grid.loc[fill_food, "food_cpi_yoy_change"] = grid.loc[fill_food, "ess_food_yoy"]
    grid.loc[fill_food, "food_cpi_yoy_change_source"] = "ess_bulletin"

    grid = grid.drop(columns=["ess_headline_yoy", "ess_food_yoy"])

    # ------------------------------------------------------------------
    # Chain-linked level reconstruction past Ha et al.'s coverage cutoff
    # (un-deferred 2026-08-20 -- see module docstring for why this is now
    # considered safe: all 4 cached ESS bulletins state the identical base
    # period across the entire gap window). level(t) = level(t-12) x
    # (1 + yoy_change(t)), walking forward chronologically so a
    # reconstructed value can itself seed a later month's reconstruction
    # (standard chain-linking). Uses headline_cpi_yoy_change/
    # food_cpi_yoy_change, which are already exact (Ha-et-al-level-derived)
    # wherever Ha et al. has real coverage and ESS-bulletin-filled past
    # that -- so this column exactly equals the real level for every month
    # already covered by Ha et al., and only actually reconstructs the
    # gap. Kept as a SEPARATE column from headline_cpi/food_cpi (which
    # stay NaN past cutoff, unchanged) -- do not use for RQ1 (Busker never
    # reconstructs this; use the _undated columns for RQ1).
    # ------------------------------------------------------------------
    def chain_link_level(levels, yoy_change):
        out = levels.to_numpy(dtype=float).copy()
        yoy = yoy_change.to_numpy(dtype=float)
        for i in range(12, len(out)):
            if np.isnan(out[i]) and not np.isnan(out[i - 12]) and not np.isnan(yoy[i]):
                out[i] = out[i - 12] * (1 + yoy[i])
        return out

    grid["headline_cpi_chainlinked"] = chain_link_level(grid["headline_cpi"], grid["headline_cpi_yoy_change"])
    grid["food_cpi_chainlinked"] = chain_link_level(grid["food_cpi"], grid["food_cpi_yoy_change"])
    grid["headline_cpi_chainlinked_source"] = np.where(
        grid["headline_cpi"].notna(), "ha_et_al",
        np.where(grid["headline_cpi_chainlinked"].notna(), "chainlinked_ess", None),
    )
    grid["food_cpi_chainlinked_source"] = np.where(
        grid["food_cpi"].notna(), "ha_et_al",
        np.where(grid["food_cpi_chainlinked"].notna(), "chainlinked_ess", None),
    )

    for col in ["headline_cpi", "headline_cpi_yoy_change", "food_cpi", "food_cpi_yoy_change",
                "headline_cpi_chainlinked", "food_cpi_chainlinked"]:
        for lag in LAGS:
            grid[f"{col}_lag{lag}"] = grid[col].shift(lag)

    grid = grid[grid["year"] >= FEATURE_START_YEAR].reset_index(drop=True)
    return grid


def main():
    df = build_monthly_table()

    cols = ["year", "month"]
    for col in ["headline_cpi", "headline_cpi_yoy_change", "food_cpi", "food_cpi_yoy_change",
                "headline_cpi_chainlinked", "food_cpi_chainlinked"]:
        cols += [col, f"{col}_lag1", f"{col}_lag3", f"{col}_lag6", f"{col}_lag12"]
    cols += ["headline_cpi_yoy_change_source", "food_cpi_yoy_change_source",
             "headline_cpi_chainlinked_source", "food_cpi_chainlinked_source"]
    cols += [
        "headline_cpi_undated", "headline_cpi_yoy_change_undated",
        "food_cpi_undated", "food_cpi_yoy_change_undated",
    ]
    df = df[cols]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    first, last = df.iloc[0], df.iloc[-1]
    print(f"Wrote {len(df)} rows ({int(first['year'])}-{int(first['month']):02d} to "
          f"{int(last['year'])}-{int(last['month']):02d}) to {OUTPUT_PATH}")
    for col in ["headline_cpi", "food_cpi", "headline_cpi_yoy_change", "food_cpi_yoy_change",
                "headline_cpi_undated", "food_cpi_undated",
                "headline_cpi_chainlinked", "food_cpi_chainlinked"]:
        n_nan = df[col].isna().sum()
        last_valid = df.loc[df[col].notna(), ["year", "month"]].iloc[-1]
        print(f"  {col}: {n_nan} NaN rows, last valid = {int(last_valid['year'])}-{int(last_valid['month']):02d}")
    print("  headline_cpi_yoy_change_source counts:")
    print(df["headline_cpi_yoy_change_source"].value_counts(dropna=False).to_string())
    print("  food_cpi_yoy_change_source counts:")
    print(df["food_cpi_yoy_change_source"].value_counts(dropna=False).to_string())
    print("  headline_cpi_chainlinked_source counts:")
    print(df["headline_cpi_chainlinked_source"].value_counts(dropna=False).to_string())
    print("  food_cpi_chainlinked_source counts:")
    print(df["food_cpi_chainlinked_source"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
