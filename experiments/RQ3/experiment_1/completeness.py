"""RQ3 data-completeness scoring.

Completeness is computed over `data/interim/model_join/ethiopia_raw_wide_panel.csv`
(model_join Stage 1 output: every source joined at its natural, un-shifted
month -- not a lead-shifted `model_tables/` file). This is the right table for
a *feature availability* question: `model_tables/` re-expresses some columns
per lead (e.g. GloFAS/IRI-CPC/USGS-GEFS forecast-preselect columns are only
populated at the leads their product actually reaches, NaN elsewhere by
construction -- CLAUDE.md Sec 2), which would make completeness an artifact of
lead choice rather than a property of the region's data.

Feature-column selection reuses experiments/RQ1/experiment_2/run_model_ethiopia.py's
own IDENTITY_DROP / PROVENANCE_DROP / FEATURE_CLUSTER_PREFIXES / classify_feature_cluster
rather than re-deriving a column list -- these are the exact definitions the
project's own modeling code uses to decide "is this a feature". Reusing them
means the completeness score is scored over the same feature set the models
are actually trained on, and stays in sync automatically if that set changes.

Forecast-preselect columns (GloFAS/IRI-CPC/USGS-GEFS) appear in the raw panel
twice, suffixed `_lead1`/`_lead3` (the two lead times those products are
pre-aligned to -- see CLAUDE.md's model_join Sec 2 entry). Using both would
double-count the same underlying source; only `_lead1` is kept (the one lead
all three products reach; USGS-GEFS never reaches lead3), matching
`classify_feature_cluster`'s prefix rules, which key off the un-suffixed
prefix (e.g. "glofas_") and classify the suffixed columns correctly without
needing the suffix stripped.

Aggregation choice (documented per CLAUDE.md Sec 4's "comment the statistical
choices" convention): completeness per zone = unweighted mean of per-feature
completeness across every selected feature column. RQ4's SHAP analysis (which
could motivate an importance-weighted alternative) is not yet built (CLAUDE.md
Sec 2, "Not yet built"), so an unweighted mean is the only defensible default
right now -- not a simplification made for convenience. The required
sensitivity check (min-of-cluster-means instead of mean-of-all-features) is in
`sensitivity_alt_completeness()` below, run separately by run_rq3_analysis.py.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_PANEL_PATH = REPO_ROOT / "data" / "interim" / "model_join" / "ethiopia_raw_wide_panel.csv"
LIVELIHOOD_ZONES_PATH = REPO_ROOT / "boundaries" / "livelihood_zones_admin2.csv"

sys.path.insert(0, str(REPO_ROOT / "experiments" / "RQ1" / "experiment_2"))
from run_model_ethiopia import (  # noqa: E402
    IDENTITY_DROP,
    PROVENANCE_DROP,
    TARGET_COL,
    BASE1_COL,
    FEATURE_CLUSTER_PREFIXES,
    ALWAYS_KEEP_PREFIXES,
    classify_feature_cluster,
)

NON_FEATURE_COLS = set(IDENTITY_DROP) | set(PROVENANCE_DROP) | {TARGET_COL, BASE1_COL, "observed", "split"}
CLUSTER_NAMES = list(FEATURE_CLUSTER_PREFIXES.keys())  # ["climate", "agriculture", "economic", "conflict"]


def load_raw_panel():
    """Load Stage 1's un-shifted, all-sources-joined panel (92 zones x 183 gapless months)."""
    df = pd.read_csv(RAW_PANEL_PATH, low_memory=False)
    df["month"] = pd.to_datetime(df["month"])
    return df


def select_feature_columns(columns):
    """Return {raw_column_name: display_name, ...} for every column that counts
    toward completeness, and {raw_column_name: cluster_name}. Drops target/
    identity/provenance columns, drops the redundant `_lead3` forecast-preselect
    duplicate, and classifies every survivor via the model's own
    classify_feature_cluster() (never left unclassified -- run_model_ethiopia.py
    already asserts every real feature lands in exactly one group or
    ALWAYS_KEEP_PREFIXES; reused here, not re-verified).
    """
    lead3_cols = {c for c in columns if c.endswith("_lead3")}
    display_names = {}
    clusters = {}
    for col in columns:
        if col in lead3_cols:
            continue
        base = col[:-len("_lead1")] if col.endswith("_lead1") else col
        if base in NON_FEATURE_COLS:
            continue
        group = classify_feature_cluster(col)
        if group is None:
            continue  # not expected to happen; classify_feature_cluster covers every real feature
        display_names[col] = base
        clusters[col] = "target_memory" if group == "always_keep" else group
    return display_names, clusters


def per_feature_completeness(df, feature_cols):
    """(non-missing months) / (total months in the panel) per zone per feature.

    The panel is a confirmed gapless zone x month rectangle (92 zones x 183
    months each, model_join Sec 2), so "expected observation-months" is just
    the row count per zone -- no separate calendar reconstruction needed.
    A single denominator (the full panel span, 2011-04 to 2026-06) is used for
    every zone and every feature, deliberately: a feature that only became
    available late for a project-wide reason (e.g. WFP teff pricing starting
    2020) then depresses every zone's score by the same amount, which cannot
    manufacture a spurious cross-zone difference -- it only compresses the
    scale. Only differences in a feature's *zone-specific* missingness
    (a market that never reports, a pastoral zone AgSS never surveyed, no
    GloFAS river-network pixel, etc.) can move one zone's score relative to
    another's, which is exactly the RQ3 completeness signal.
    """
    n_months = df.groupby("zone_code")["month"].transform("size")
    frac_present = df[feature_cols].notna().astype(float).div(n_months, axis=0)
    frac_present = pd.concat([df[["zone_code"]], frac_present], axis=1)
    return frac_present.groupby("zone_code")[feature_cols].sum()  # sum of (1/n_months) per present month == fraction present


def compute_zone_completeness(df, feature_cols, clusters):
    """Per-zone completeness: overall (mean across all features) plus a
    per-cluster mean (climate/agriculture/economic/conflict/target_memory).
    """
    per_feature = per_feature_completeness(df, feature_cols)
    out = pd.DataFrame(index=per_feature.index)
    out["completeness_mean"] = per_feature.mean(axis=1)
    for cluster in CLUSTER_NAMES + ["target_memory"]:
        cols = [c for c in feature_cols if clusters[c] == cluster]
        if cols:
            out[f"completeness_{cluster}"] = per_feature[cols].mean(axis=1)
    out["n_features_scored"] = len(feature_cols)
    return out.reset_index(), per_feature


def sensitivity_alt_completeness(per_feature, clusters, feature_cols):
    """Sensitivity check requested by the project owner: min-of-cluster-means
    instead of mean-of-all-features. A zone that is well covered on 3 clusters
    but has zero conflict/market coverage gets a low score here even though its
    mean-of-all-features score might look fine -- tests whether the primary
    completeness definition's averaging is hiding a single-cluster gap that
    drives the correlation result.
    """
    cluster_means = pd.DataFrame(index=per_feature.index)
    for cluster in CLUSTER_NAMES:  # target_memory excluded -- not one of the 4 named clusters
        cols = [c for c in feature_cols if clusters[c] == cluster]
        if cols:
            cluster_means[cluster] = per_feature[cols].mean(axis=1)
    return cluster_means.min(axis=1).rename("completeness_min_cluster")


def attach_livelihood_zone(zone_df):
    lhz = pd.read_csv(LIVELIHOOD_ZONES_PATH)[["zone_code", "zone_name", "dominant_livelihood_zone"]]
    return zone_df.merge(lhz, on="zone_code", how="left", validate="one_to_one")


def aggregate_to_livelihood_zone(zone_completeness):
    """Zone-level completeness scores averaged (equal zone weight) up to the
    3 livelihood-zone clusters. Equal-zone-weight, not population-weighted --
    matches how every other per-cluster metric in this project (RQ1/RQ2's
    metrics_per_cluster.csv) is pooled.
    """
    score_cols = [c for c in zone_completeness.columns if c.startswith("completeness_")]
    return (
        zone_completeness.groupby("dominant_livelihood_zone")[score_cols]
        .mean()
        .reset_index()
        .rename(columns={"dominant_livelihood_zone": "lhz"})
    )
