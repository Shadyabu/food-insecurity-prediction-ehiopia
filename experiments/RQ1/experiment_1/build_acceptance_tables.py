"""Build the reference-vs-reproduction comparison and 1-SD acceptance tables
required by RQ1_Experiment1_Busker_Reproduction.md sections 2 and 3.

Reads the metrics already produced by experiments/busker_baseline/run_busker_baseline.py
(re-run into this experiment's own metrics/ and predictions/ dirs -- same
XGB_PARAMS, same random_state=42, same HOA/213-unit config, so numbers are
identical to the previously-validated `final-hoa` run; see report.md for why
this counts as this experiment's own run rather than a stale reuse).

Reference numbers below are transcribed verbatim from
RQ1_Experiment1_Busker_Reproduction.md sections 2.1-2.5 (Busker et al. 2024
paper text/figures, some marked by the source doc itself as "approximate --
read from published figure").
"""

import numpy as np
import pandas as pd

METRICS_DIR = "metrics"
CLUSTER_LABELS = {"p": "Pastoral", "ap": "Agro-pastoral", "other": "Crop farming"}

# --- Section 2.3: R2 by livelihood zone and lead time (Fig 5, approximate) ---
REF_R2_BY_ZONE_LEAD = {
    ("p", 0): 0.49, ("ap", 0): 0.48, ("other", 0): 0.54,
    ("p", 1): 0.55, ("ap", 1): 0.48, ("other", 1): 0.65,
    ("p", 2): 0.44, ("ap", 2): 0.48, ("other", 2): 0.61,
    ("p", 3): 0.53, ("ap", 3): 0.37, ("other", 3): 0.57,
    ("p", 4): 0.35, ("ap", 4): 0.40, ("other", 4): 0.46,
    ("p", 8): 0.28, ("ap", 8): 0.23, ("other", 8): 0.48,
    ("p", 12): 0.30, ("ap", 12): 0.05, ("other", 12): 0.40,
}

# --- Section 2.2: per-unit R2 at 3-month lead (Fig 3, exact values) ---
REF_R2_PER_UNIT_LEAD3 = {
    "Afmadow": 0.94, "Burco": 0.77, "Hari-Zone 5": 0.59, "Xudur": 0.57,
    "Tana River": 0.45, "Waajid": 0.45, "Wajir": 0.44, "Garissa": 0.39,
    "Western Tigray": 0.29,
}

# --- Section 2.1: headline pooled metrics ---
REF_HEADLINE = {
    "mae_lead3_213units": 0.35,
    "r2_lead1_wholeregion": 0.72,
    "observed_ipc_std": 0.8,
}

# --- Section 2.5: HR / FAR by zone and lead (Fig 6, approximate table) ---
REF_HR_FAR = {
    ("p", 0): (0.12, 0.10), ("ap", 0): (0.40, 0.04), ("other", 0): (0.0, 0.001),
    ("p", 1): (0.20, 0.08), ("ap", 1): (0.50, 0.04), ("other", 1): (0.0, 0.0),
    ("p", 2): (0.18, 0.08), ("ap", 2): (0.30, 0.06), ("other", 2): (0.0, 0.0),
    ("p", 3): (0.18, 0.07), ("ap", 3): (0.20, 0.08), ("other", 3): (0.0, 0.0),
    ("p", 4): (0.05, 0.06), ("ap", 4): (0.05, 0.05), ("other", 4): (0.0, 0.0),
    ("p", 8): (0.04, 0.07), ("ap", 8): (0.09, 0.08), ("other", 8): (0.0, 0.0),
    ("p", 12): (0.02, 0.02), ("ap", 12): (0.31, 0.14), ("other", 12): (0.0, 0.0),
}


def _mad_sd(x):
    """Median-absolute-deviation-based SD estimate (1.4826*MAD), robust to
    the extreme per-unit R2 outliers described in report.md section 3 --
    a handful of units with a near-zero-variance test-set target (as few as
    9 test months) produce R2 as low as -4e8, which inflates the raw SD by
    up to 8 orders of magnitude and makes a naive 1-SD test vacuous."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))


def _pct_diff(reproduced, reference):
    """Relative % difference vs. the published reference value, matching the
    'median/mean % difference, % of zones within X% agreement' convention
    used to validate every other pipeline in this project (CLAUDE.md §3.5).
    Undefined (NaN) when the reference itself is 0 (no ap/lead12-style case
    in REF_R2_BY_ZONE_LEAD hits exactly 0, but reference values near 0, e.g.
    0.05, make this figure large/noisy by construction -- flagged per-row via
    n_units, not hidden)."""
    if reference == 0:
        return np.nan
    return (reproduced - reference) / abs(reference) * 100


def build_r2_acceptance():
    unit = pd.read_csv(f"{METRICS_DIR}/metrics_per_unit.csv")
    cluster = pd.read_csv(f"{METRICS_DIR}/metrics_per_cluster.csv")

    grouped = unit.groupby(["lhz", "lead"])["r2"]
    sd = grouped.std(ddof=1).rename("per_unit_r2_sd")
    robust_sd = grouped.apply(_mad_sd).rename("per_unit_r2_robust_sd")
    n_extreme = grouped.apply(lambda s: int((s < -10).sum())).rename("n_units_r2_below_neg10")
    n_units = grouped.count().rename("n_units")
    cluster_r2 = cluster.set_index(["lhz", "lead"])["r2"].rename("reproduced_zone_r2")

    rows = []
    for (lhz, lead), ref_r2 in REF_R2_BY_ZONE_LEAD.items():
        repro = cluster_r2.get((lhz, lead), np.nan)
        s = sd.get((lhz, lead), np.nan)
        rs = robust_sd.get((lhz, lead), np.nan)
        n = n_units.get((lhz, lead), np.nan)
        n_ext = n_extreme.get((lhz, lead), 0)
        abs_diff = abs(repro - ref_r2)
        pct_diff = _pct_diff(repro, ref_r2)
        within_raw = bool(abs_diff <= s) if pd.notna(s) else None
        within_robust = bool(abs_diff <= rs) if pd.notna(rs) else None
        within_5pct = bool(abs(pct_diff) <= 5) if pd.notna(pct_diff) else None
        rows.append({
            "livelihood_zone": CLUSTER_LABELS[lhz], "lhz_code": lhz, "lead": lead,
            "reference_r2": ref_r2, "reproduced_r2": round(repro, 4),
            "per_unit_r2_sd": round(s, 4), "per_unit_r2_robust_sd_mad": round(rs, 4),
            "n_units": int(n), "n_units_r2_below_neg10": n_ext,
            "abs_diff": round(abs_diff, 4),
            "pct_diff": round(pct_diff, 2) if pd.notna(pct_diff) else np.nan,
            "within_1sd_raw": within_raw, "within_1sd_robust": within_robust,
            "within_5pct": within_5pct,
        })
    out = pd.DataFrame(rows).sort_values(["lhz_code", "lead"])
    out.to_csv(f"{METRICS_DIR}/statistical_acceptance_r2.csv", index=False)

    n_pass_raw = out["within_1sd_raw"].sum()
    n_pass_robust = out["within_1sd_robust"].sum()
    n_pass_5pct = out["within_5pct"].sum()
    n_total = out["within_1sd_raw"].notna().sum()
    n_total_5pct = out["within_5pct"].notna().sum()
    print(f"R2 acceptance (regular per-unit SD, not bootstrapped): {n_pass_raw}/{n_total} within 1 SD")
    print(f"R2 acceptance (robust MAD-based SD, supplementary): {n_pass_robust}/{n_total} within 1 SD")
    print(f"R2 acceptance (+-5% rule): {n_pass_5pct}/{n_total_5pct} within 5%")
    return out


def build_mae_acceptance_lead3():
    """MAE has only one published reference point (0.35, pooled over all 213
    units at lead 3) -- no zone x lead grid exists to test against, unlike R2.
    Apply the same 1-SD test per zone at lead 3 only, against that single
    pooled reference, and report the pooled-213-unit reading alongside it.
    """
    unit = pd.read_csv(f"{METRICS_DIR}/metrics_per_unit.csv")
    cluster = pd.read_csv(f"{METRICS_DIR}/metrics_per_cluster.csv")
    lead = pd.read_csv(f"{METRICS_DIR}/metrics_per_lead.csv")

    u3 = unit[unit["lead"] == 3]
    sd = u3.groupby("lhz")["mae"].std(ddof=1)
    c3 = cluster[cluster["lead"] == 3].set_index("lhz")["mae"]

    ref = REF_HEADLINE["mae_lead3_213units"]
    rows = []
    for lhz in ["p", "ap", "other"]:
        repro = c3.get(lhz, np.nan)
        s = sd.get(lhz, np.nan)
        abs_diff = abs(repro - ref)
        pct_diff = _pct_diff(repro, ref)
        rows.append({
            "livelihood_zone": CLUSTER_LABELS[lhz], "lhz_code": lhz, "lead": 3,
            "reference_mae": ref, "reproduced_zone_mae": round(repro, 4),
            "per_unit_mae_sd": round(s, 4), "abs_diff": round(abs_diff, 4),
            "pct_diff": round(pct_diff, 2), "within_1sd": bool(abs_diff <= s),
            "within_5pct": bool(abs(pct_diff) <= 5),
            "note": "reference is the single pooled 213-unit value, not zone-specific",
        })

    # Three whole-region readings of "average MAE" (unit mean, cluster mean, pooled).
    unit_mean = u3["mae"].mean()
    cluster_mean = c3.mean()
    pooled = lead.set_index("lead").loc[3, "mae"]
    rows.append({
        "livelihood_zone": "ALL (unit-weighted mean, 213 units)", "lhz_code": "all_unit_mean",
        "lead": 3, "reference_mae": ref, "reproduced_zone_mae": round(unit_mean, 4),
        "per_unit_mae_sd": round(u3["mae"].std(ddof=1), 4),
        "abs_diff": round(abs(unit_mean - ref), 4),
        "pct_diff": round(_pct_diff(unit_mean, ref), 2),
        "within_1sd": bool(abs(unit_mean - ref) <= u3["mae"].std(ddof=1)),
        "within_5pct": bool(abs(_pct_diff(unit_mean, ref)) <= 5),
        "note": "unit-weighted mean over all 213 per-unit MAEs",
    })
    rows.append({
        "livelihood_zone": "ALL (mean of 3 zone models)", "lhz_code": "all_cluster_mean",
        "lead": 3, "reference_mae": ref, "reproduced_zone_mae": round(cluster_mean, 4),
        "per_unit_mae_sd": np.nan, "abs_diff": round(abs(cluster_mean - ref), 4),
        "pct_diff": round(_pct_diff(cluster_mean, ref), 2),
        "within_1sd": None, "within_5pct": bool(abs(_pct_diff(cluster_mean, ref)) <= 5),
        "note": "mean of the 3 livelihood-zone pooled MAEs (n=3, SD not meaningful)",
    })
    rows.append({
        "livelihood_zone": "ALL (pooled over all predictions)", "lhz_code": "all_pooled",
        "lead": 3, "reference_mae": ref, "reproduced_zone_mae": round(pooled, 4),
        "per_unit_mae_sd": round(u3["mae"].std(ddof=1), 4),
        "abs_diff": round(abs(pooled - ref), 4),
        "pct_diff": round(_pct_diff(pooled, ref), 2),
        "within_1sd": bool(abs(pooled - ref) <= u3["mae"].std(ddof=1)),
        "within_5pct": bool(abs(_pct_diff(pooled, ref)) <= 5),
        "note": "pooled over all individual lead-3 predictions region-wide",
    })

    out = pd.DataFrame(rows)
    out.to_csv(f"{METRICS_DIR}/statistical_acceptance_mae_lead3.csv", index=False)
    return out


def build_per_unit_r2_comparison():
    unit = pd.read_csv(f"{METRICS_DIR}/metrics_per_unit.csv")
    u3 = unit[unit["lead"] == 3].set_index("county")["r2"]

    rows = []
    for county, ref_r2 in REF_R2_PER_UNIT_LEAD3.items():
        repro = u3.get(county, np.nan)
        rows.append({
            "admin_unit": county, "reference_r2": ref_r2,
            "reproduced_r2": round(repro, 4) if pd.notna(repro) else np.nan,
            "abs_diff": round(abs(repro - ref_r2), 4) if pd.notna(repro) else np.nan,
            "matched": pd.notna(repro),
        })
    out = pd.DataFrame(rows)
    out.to_csv(f"{METRICS_DIR}/per_unit_r2_comparison_lead3.csv", index=False)
    print(f"per-unit R2 comparison: {out['matched'].sum()}/{len(out)} named units matched")
    return out


def build_hr_far_comparison():
    onsets = pd.read_csv(f"{METRICS_DIR}/crisis_onset_rates.csv")
    onsets = onsets.set_index(["lhz", "lead"])

    rows = []
    for (lhz, lead), (ref_hr, ref_far) in REF_HR_FAR.items():
        if (lhz, lead) not in onsets.index:
            continue
        row = onsets.loc[(lhz, lead)]
        rows.append({
            "livelihood_zone": CLUSTER_LABELS[lhz], "lhz_code": lhz, "lead": lead,
            "event_count": int(row["event_count"]),
            "reference_hr": ref_hr, "reproduced_hr": round(row["hr"], 4),
            "hr_abs_diff": round(abs(row["hr"] - ref_hr), 4),
            "reference_far": ref_far, "reproduced_far": round(row["far"], 4),
            "far_abs_diff": round(abs(row["far"] - ref_far), 4),
            "reproduced_f1": round(row["f1"], 4),
        })
    out = pd.DataFrame(rows).sort_values(["lhz_code", "lead"])
    out.to_csv(f"{METRICS_DIR}/hr_far_comparison.csv", index=False)
    return out


if __name__ == "__main__":
    r2_out = build_r2_acceptance()
    mae_out = build_mae_acceptance_lead3()
    unit_out = build_per_unit_r2_comparison()
    hrfar_out = build_hr_far_comparison()
    print("\nwrote statistical_acceptance_r2.csv, statistical_acceptance_mae_lead3.csv, "
          "per_unit_r2_comparison_lead3.csv, hr_far_comparison.csv to metrics/")
