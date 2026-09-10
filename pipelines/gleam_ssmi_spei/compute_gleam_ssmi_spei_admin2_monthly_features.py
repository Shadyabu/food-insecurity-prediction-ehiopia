"""
compute_gleam_ssmi_spei_admin2_monthly_features.py

Engineer stage: takes the raw GLEAM admin2-monthly table (sm_root, pet_mm)
plus CHIRPS's already-validated daily precipitation, and computes SSMI and
SPEI at 5 accumulation windows (1/3/6/12/24 months), following Busker et al.
(2024) / Odongo et al. (2023): rainfall for SPI, rainfall+PET for SPEI,
root-zone soil moisture for SSMI, accumulated over each window, then
standardized against a fixed historical climatology.

Deliberately SEPARATE from fetch_gleam_admin2.py, same reason as CHIRPS's
own compute_..._monthly_features.py: this step changes far more often
during development than the SFTP download does, and re-running it must
never require re-downloading ~7GB of GLEAM NetCDFs.

SPEI's precipitation input reuses CHIRPS's own validated daily series
(re-aggregated to monthly here from the interim daily CSV, not the
processed CHIRPS feature file, which only exports 2009+ -- the full
1981+ record is needed for the standardization baseline). This matches
Busker et al.'s own methodology: their SPI is "based on CHIRPS" and their
SPEI is precipitation minus GLEAM PET, using the *same* precipitation
input as SPI, not a second precipitation source.

--------------------------------------------------------------------------
Leakage note (standardization baseline)
--------------------------------------------------------------------------
Per docs/CHIRPS_Pipeline_Documentation.md Sec 4.5 and
docs/Pipeline_Replication_Guide.md Sec 4.2 (existing project convention,
already applied to SPI): the gamma/log-logistic distribution parameters
are fit ONCE per zone per calendar month using the full available record
(1981-FEATURE_END_YEAR), not an expanding/rolling window. This is a fixed
historical-baseline approach, matching Busker et al.'s own stated method
("long-term climatology (1981-2022)") and is standard practice for
SPI/SPEI/SSMI-family indices in the drought-monitoring literature. It is
a deliberate, documented divergence from strict per-row backward-only
leakage prevention (a value's standardization technically depends on the
whole-record distribution, including years after that row) -- flagged
here explicitly per CLAUDE.md Sec 3.2, exactly as SPI's own baseline
choice already is.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))  # pipelines/gleam_ssmi_spei/ -> pipelines/ -> repo root

GLEAM_RAW_PATH = os.path.join(REPO_ROOT, "data", "interim", "gleam_ssmi_spei", "gleam_admin2_monthly_raw.csv")
CHIRPS_DAILY_PATH = os.path.join(REPO_ROOT, "data", "interim", "chirps", "chirps_admin2_daily_combined.csv")
OUTPUT_PATH = os.path.join(REPO_ROOT, "data", "processed", "features", "gleam_ssmi_spei_admin2_monthly.csv")

ZONE_FIELD = "pcode"
FEATURE_START_YEAR = 2009
FEATURE_END_YEAR = 2025  # GLEAM v4.3a's latest available year; one year short of CHIRPS's 2026

WINDOWS = [1, 3, 6, 12, 24]
DIST_MIN_YEARS = 5

# --------------------------------------------------------------------------
# 1. Load GLEAM raw (sm_root, pet_mm) and CHIRPS precip (re-aggregated from
#    the validated daily series, full record -- needed for the baseline fit)
# --------------------------------------------------------------------------

gleam_df = pd.read_csv(GLEAM_RAW_PATH)
print(f"Loaded GLEAM raw: {gleam_df.shape[0]} rows, {gleam_df[ZONE_FIELD].nunique()} zones, "
      f"{gleam_df['year'].min()}-{gleam_df['year'].max()}.")

chirps_daily = pd.read_csv(CHIRPS_DAILY_PATH)
chirps_daily["date"] = pd.to_datetime(chirps_daily["date"])
chirps_daily["year"] = chirps_daily["date"].dt.year
chirps_daily["month"] = chirps_daily["date"].dt.month
chirps_monthly = (
    chirps_daily.groupby([ZONE_FIELD, "year", "month"])["precipitation_mm"]
    .sum()
    .rename("precip_mm")
    .reset_index()
)
print(f"Loaded CHIRPS monthly precip (re-aggregated): {chirps_monthly.shape[0]} rows, "
      f"{chirps_monthly['year'].min()}-{chirps_monthly['year'].max()}.")

df = gleam_df.merge(chirps_monthly, on=[ZONE_FIELD, "year", "month"], how="left")
df = df.sort_values([ZONE_FIELD, "year", "month"]).reset_index(drop=True)

# --------------------------------------------------------------------------
# 2. SPEI input: water balance D = precipitation - potential evaporation
# --------------------------------------------------------------------------

df["spei_balance_mm"] = df["precip_mm"] - df["pet_mm"]

# --------------------------------------------------------------------------
# 3. Rolling accumulation windows
#    - sm_root is a state variable (volumetric fraction) -> mean over window
#    - pet_mm / spei_balance_mm are fluxes (mm/month) -> sum over window,
#      same convention as CHIRPS's rainfall_sum_Nm
# --------------------------------------------------------------------------

print(f"Computing rolling accumulation windows: {WINDOWS} ...")
for window in WINDOWS:
    df[f"sm_mean_{window}m"] = (
        df.groupby(ZONE_FIELD)["sm_root"].transform(lambda s: s.rolling(window, min_periods=window).mean())
    )
    df[f"spei_balance_sum_{window}m"] = (
        df.groupby(ZONE_FIELD)["spei_balance_mm"].transform(lambda s: s.rolling(window, min_periods=window).sum())
    )

# --------------------------------------------------------------------------
# 4. SSMI: best-fit distribution per zone per calendar month, selected via
#    Kolmogorov-Smirnov goodness-of-fit -- per Odongo et al. (2023), the
#    methodology Busker et al. (2024) cite ("multiple distributions were
#    tested for each of the indices per period and administrative unit...
#    the distribution with the best fit based on the Kolmogorov best-fit
#    test was selected"). Candidate set (gamma / lognormal / Weibull) is a
#    reasonable choice for a non-negative, moderately-skewed state variable
#    like root-zone soil moisture -- the paper doesn't enumerate its own
#    candidate list, so this is documented as our own choice, not a
#    reproduction of theirs. No zero-correction needed: root-zone soil
#    moisture is continuous and never exactly zero, unlike raw daily rainfall.
# --------------------------------------------------------------------------

SSMI_CANDIDATE_DISTS = [stats.gamma, stats.lognorm, stats.weibull_min]


def fit_best_distribution(valid_vals):
    """Fits each candidate (floc=0, non-negative support) and returns the
    (dist, params) with the lowest KS statistic -- i.e. the best fit."""
    best_stat, best_dist, best_params = None, None, None
    for dist in SSMI_CANDIDATE_DISTS:
        try:
            params = dist.fit(valid_vals, floc=0)
            stat, _ = stats.kstest(valid_vals, dist.name, args=params)
        except Exception:
            continue
        if best_stat is None or stat < best_stat:
            best_stat, best_dist, best_params = stat, dist, params
    return best_dist, best_params


def compute_ssmi_for_zone(zone_df, value_col, month_col="month"):
    zone_df = zone_df.copy()
    out_col = f"ssmi_from_{value_col}"
    zone_df[out_col] = np.nan
    for m in range(1, 13):
        mask = zone_df[month_col] == m
        vals = zone_df.loc[mask, value_col].to_numpy()
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < DIST_MIN_YEARS:
            continue
        valid_vals = vals[valid_mask]
        if len(valid_vals) < 3:
            continue
        dist, params = fit_best_distribution(valid_vals)
        if dist is None:
            continue
        cdf = np.full_like(vals, np.nan, dtype=float)
        cdf[valid_mask] = dist.cdf(valid_vals, *params)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        ssmi_vals = np.full_like(vals, np.nan, dtype=float)
        ssmi_vals[valid_mask] = stats.norm.ppf(cdf[valid_mask])
        zone_df.loc[mask, out_col] = ssmi_vals
    return zone_df


# --------------------------------------------------------------------------
# 5. SPEI: log-logistic fit per zone per calendar month (the water balance
#    D = P - PET can be negative, unlike gamma's non-negative support) --
#    per docs/data_sources_master_table.md's explicit distribution note and
#    Vicente-Serrano et al. (2010), the original SPEI paper Busker et al.
#    cite. No zero-correction: D is continuous and effectively never exactly
#    zero, unlike raw daily rainfall.
# --------------------------------------------------------------------------


def compute_spei_for_zone(zone_df, value_col, month_col="month"):
    zone_df = zone_df.copy()
    out_col = f"spei_from_{value_col}"
    zone_df[out_col] = np.nan
    for m in range(1, 13):
        mask = zone_df[month_col] == m
        vals = zone_df.loc[mask, value_col].to_numpy()
        valid_mask = ~np.isnan(vals)
        if valid_mask.sum() < DIST_MIN_YEARS:
            continue
        valid_vals = vals[valid_mask]
        try:
            shape, loc, scale = stats.fisk.fit(valid_vals)
        except Exception:
            continue
        cdf = np.full_like(vals, np.nan, dtype=float)
        cdf[valid_mask] = stats.fisk.cdf(valid_vals, shape, loc=loc, scale=scale)
        cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
        spei_vals = np.full_like(vals, np.nan, dtype=float)
        spei_vals[valid_mask] = stats.norm.ppf(cdf[valid_mask])
        zone_df.loc[mask, out_col] = spei_vals
    return zone_df


print("Fitting SSMI (gamma) per zone, per accumulation window ...")
for window in WINDOWS:
    value_col = f"sm_mean_{window}m"
    pieces = [compute_ssmi_for_zone(zone_df, value_col=value_col) for _, zone_df in df.groupby(ZONE_FIELD)]
    df = pd.concat(pieces, ignore_index=True)
    df = df.rename(columns={f"ssmi_from_{value_col}": f"ssmi_{window}"})

print("Fitting SPEI (log-logistic) per zone, per accumulation window ...")
for window in WINDOWS:
    value_col = f"spei_balance_sum_{window}m"
    pieces = [compute_spei_for_zone(zone_df, value_col=value_col) for _, zone_df in df.groupby(ZONE_FIELD)]
    df = pd.concat(pieces, ignore_index=True)
    df = df.rename(columns={f"spei_from_{value_col}": f"spei_{window}"})

df = df.sort_values([ZONE_FIELD, "year", "month"]).reset_index(drop=True)

# --------------------------------------------------------------------------
# 6. Restrict to feature years and save
# --------------------------------------------------------------------------

final_df = df[(df["year"] >= FEATURE_START_YEAR) & (df["year"] <= FEATURE_END_YEAR)].copy()
final_df = final_df.sort_values([ZONE_FIELD, "year", "month"]).reset_index(drop=True)

final_df.to_csv(OUTPUT_PATH, index=False)

print(f"\nFinal monthly feature table: {final_df.shape[0]} rows ({FEATURE_START_YEAR}-{FEATURE_END_YEAR})")
print(f"Saved to {OUTPUT_PATH}")
print(f"Columns: {list(final_df.columns)}")
for window in WINDOWS:
    print(f"  ssmi_{window} NaN rate: {final_df[f'ssmi_{window}'].isna().mean():.2%}, "
          f"spei_{window} NaN rate: {final_df[f'spei_{window}'].isna().mean():.2%}")
