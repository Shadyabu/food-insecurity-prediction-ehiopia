"""RQ5 Phase 1 batch runner. Loads the cached per-instance raw/SHAP values
(compute_local_shap.py) and applies rule_engine.score_batch per (subject,
cluster, lead) group, writing one predicted-cause row per zone-month per
subject to predicted_causes.csv.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from rule_engine import load_taxonomy, score_batch

RQ5_DIR = Path(__file__).resolve().parent
SHAP_CACHE = RQ5_DIR / "shap_cache_2020_2024"
ID_COLS = ["time", "zone_code", "zone_name", "split", "cluster", "lead", "base1", "subject"]


def main():
    taxonomy = load_taxonomy()
    raw_all = pd.read_parquet(SHAP_CACHE / "raw_values.parquet")
    shap_all = pd.read_parquet(SHAP_CACHE / "shap_values.parquet")

    feature_cols = [c for c in raw_all.columns if c not in ID_COLS]

    results = []
    for (subject, cluster, lead), raw_g in raw_all.groupby(["subject", "cluster", "lead"]):
        shap_g = shap_all.loc[raw_g.index]
        raw_features = raw_g[feature_cols]
        shap_features = shap_g[feature_cols]
        scored = score_batch(raw_features, shap_features, taxonomy, include_breakdown=True)
        scored = pd.concat([raw_g[ID_COLS].reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
        results.append(scored)
        print(f"subject={subject:13s} cluster={cluster:13s} lead={lead:2d}  n={len(scored)}  "
              f"causes={scored['predicted_cause'].value_counts().to_dict()}")

    out = pd.concat(results, ignore_index=True)
    out.to_csv(RQ5_DIR / "predicted_causes_2020_2024.csv", index=False)
    print(f"\nwrote {len(out)} rows to {RQ5_DIR}/predicted_causes_2020_2024.csv")

    print("\n=== predicted_cause distribution by subject ===")
    print(pd.crosstab(out["subject"], out["predicted_cause"]))


if __name__ == "__main__":
    main()
