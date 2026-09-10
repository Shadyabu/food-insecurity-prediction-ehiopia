"""RQ5 Phase 1 — SHAP-to-disaster-cause rule engine.

Reads experiments/RQ5/taxonomy_config.yaml (the single source of truth for
every feature/sign/threshold used here — do not hardcode a feature list in
this file). Turns a single instance's (raw feature values, local SHAP
values) into a predicted disaster-cause label, per-category SHAP
breakdown, and the top-two-category margin used for compound detection.

Scoring rule (see taxonomy_config.yaml header for the full derivation):
  qualifies(f, C, instance) = ( polarity(f,C) * raw_value(f,instance) > 0 )
                               and ( shap_value(f,instance) > 0 )
  score(C, instance) = sum of shap_value(f,instance) over qualifying f
  predicted_cause = argmax(score) if max(score) > 0 else "unclassified"

A feature only counts toward a category if its raw value is actually on
that category's side (this is what disambiguates DROUGHT vs FLOODING
sharing rainfall_anomaly/soil_moisture at opposite polarity) AND the
model's own SHAP attribution for that instance is actually pushing the
prediction worse (positive, in this project's delta-target convention) --
a feature that happens to sit on the "drought side" but that the model
isn't crediting with worsening this instance's prediction does not count.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RQ5_DIR = Path(__file__).resolve().parent
DEFAULT_TAXONOMY_PATH = RQ5_DIR / "taxonomy_config.yaml"

CATEGORY_NAMES = ["DROUGHT", "FLOODING", "CONFLICT", "MARKET_SHOCK"]
UNCLASSIFIED = "UNCLASSIFIED"
COMPOUND = "COMPOUND"


def load_taxonomy(path: Path | str = DEFAULT_TAXONOMY_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for cat in CATEGORY_NAMES:
        if cat not in cfg["categories"]:
            raise ValueError(f"taxonomy_config.yaml missing category {cat}")
    return cfg


def required_columns(taxonomy: dict) -> set[str]:
    """Every raw/SHAP column this taxonomy needs present in the input frames."""
    cols: set[str] = set()
    for cat in CATEGORY_NAMES:
        for feat in taxonomy["categories"][cat]["features"]:
            if feat.get("derived"):
                cols.update(feat["components"])
            else:
                cols.add(feat["column"])
    return cols


def _feature_contribution(feat: dict, raw: pd.Series, shap: pd.Series) -> tuple[float, bool]:
    """Returns (contribution, had_any_data) for one taxonomy feature entry."""
    polarity = feat.get("polarity", 1)

    if not feat.get("derived"):
        col = feat["column"]
        if col not in raw.index or col not in shap.index:
            return 0.0, False
        r, s = raw[col], shap[col]
        if pd.isna(r) or pd.isna(s):
            return 0.0, False
        qualifies = (polarity * r > 0) and (s > 0)
        return (float(s) if qualifies else 0.0), True

    rule = feat["combination_rule"]
    components = feat["components"]

    if rule == "mean_of_available":
        qualifying_sum = 0.0
        n_available = 0
        for col in components:
            if col not in raw.index or col not in shap.index:
                continue
            r, s = raw[col], shap[col]
            if pd.isna(r) or pd.isna(s):
                continue
            n_available += 1
            if (polarity * r > 0) and (s > 0):
                qualifying_sum += float(s)
        if n_available == 0:
            return 0.0, False
        return qualifying_sum / n_available, True

    if rule == "sum_shap_no_raw_gate":
        total = 0.0
        n_available = 0
        for col in components:
            if col not in shap.index:
                continue
            s = shap[col]
            if pd.isna(s):
                continue
            n_available += 1
            if s > 0:
                total += float(s)
        return total, (n_available > 0)

    raise ValueError(f"unknown combination_rule {rule!r}")


def score_instance(
    raw: pd.Series,
    shap: pd.Series,
    taxonomy: dict,
    margin_threshold: float | None = None,
) -> dict[str, Any]:
    """Score a single (zone, month, lead, subject) instance.

    raw / shap: pandas Series indexed by feature column name (one instance's
    row of raw feature values and, respectively, local SHAP values from the
    same trained model).
    """
    if margin_threshold is None:
        margin_threshold = taxonomy["compound"]["margin_threshold"]

    category_scores: dict[str, float] = {}
    breakdown: dict[str, dict[str, float]] = {}
    any_data = False

    for cat in CATEGORY_NAMES:
        total = 0.0
        feat_breakdown = {}
        for feat in taxonomy["categories"][cat]["features"]:
            contrib, had_data = _feature_contribution(feat, raw, shap)
            any_data = any_data or had_data
            total += contrib
            key = feat.get("column")
            feat_breakdown[key] = contrib
        category_scores[cat] = total
        breakdown[cat] = feat_breakdown

    ranked = sorted(CATEGORY_NAMES, key=lambda c: category_scores[c], reverse=True)
    top1, top2 = ranked[0], ranked[1]
    top1_score, top2_score = category_scores[top1], category_scores[top2]

    if not any_data:
        predicted_cause = "NO_DATA"
        margin = math.nan
        is_compound = False
    elif top1_score <= 0:
        predicted_cause = UNCLASSIFIED
        margin = math.nan
        is_compound = False
    else:
        margin = (top1_score - max(top2_score, 0.0)) / top1_score
        is_compound = top2_score > 0 and margin <= margin_threshold
        predicted_cause = COMPOUND if is_compound else top1

    return {
        "predicted_cause": predicted_cause,
        "is_compound": is_compound,
        "top1_category": top1,
        "top1_score": top1_score,
        "top2_category": top2,
        "top2_score": top2_score,
        "margin": margin,
        **{f"score_{cat}": category_scores[cat] for cat in CATEGORY_NAMES},
        "_breakdown": breakdown,
    }


def score_batch(
    raw_df: pd.DataFrame,
    shap_df: pd.DataFrame,
    taxonomy: dict,
    margin_threshold: float | None = None,
    id_cols: list[str] | None = None,
    include_breakdown: bool = False,
) -> pd.DataFrame:
    """Vectorized-by-row wrapper around score_instance for a dataframe of
    instances. raw_df and shap_df must share the same row index (same
    zone/month/lead ordering) -- checked, not assumed."""
    if not raw_df.index.equals(shap_df.index):
        raise ValueError("raw_df and shap_df must share the same row index")

    records = []
    for idx in raw_df.index:
        result = score_instance(raw_df.loc[idx], shap_df.loc[idx], taxonomy, margin_threshold)
        rec = {k: v for k, v in result.items() if k != "_breakdown"}
        if include_breakdown:
            for cat, feats in result["_breakdown"].items():
                for feat_name, val in feats.items():
                    rec[f"shap_{cat}_{feat_name}"] = val
        records.append(rec)

    out = pd.DataFrame(records, index=raw_df.index)
    if id_cols:
        meta_cols = [c for c in id_cols if c in raw_df.columns]
        out = pd.concat([raw_df[meta_cols], out], axis=1)
    return out
