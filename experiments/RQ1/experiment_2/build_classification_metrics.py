"""F1 / accuracy for the Ethiopia-dataset run, same discretization method as
experiments/RQ1/experiment_1/build_classification_metrics.py (nearest-
integer IPC phase, clipped to [1, 5]) so the two documents are directly
comparable. This experiment is Ethiopia-only already, so subsets are by
livelihood zone rather than by country.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
)

PRED_PATH = "predictions/predictions.parquet"
OUT_PATH = "ethiopia_dataset_f1_accuracy.md"
LEADS = [0, 1, 2, 3, 4, 8, 12]


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def to_ipc_class_3plus(x):
    """1 / 2 / 3+ -- collapses Crisis/Emergency/Famine into a single
    '3+' class matching FEWS NET's own humanitarian-response threshold
    (IPC Phase 3+ is the standard "Crisis or worse" cutoff). Same
    round-then-clip logic as to_ipc_class(), just clipped at 3 instead of
    5 -- a raw value of 3.6 still rounds to 4 first, then clips to 3, so
    the class boundaries at .5 are unchanged, only the top bucket
    collapses."""
    return np.clip(np.round(x), 1, 3).astype(int)


def score_subset(df, lead, discretize=to_ipc_class):
    sub = df[df["lead"] == lead]
    if len(sub) == 0:
        return None
    y_true = discretize(sub["observed"])
    y_pred = discretize(sub["prediction"])
    return {
        "lead": lead,
        "n": len(sub),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "mae": mean_absolute_error(sub["observed"], sub["prediction"]),
        "r2": r2_score(sub["observed"], sub["prediction"]) if sub["observed"].nunique() > 1 else np.nan,
    }


def build_table(df, subset_mask, label, discretize=to_ipc_class):
    sub = df[subset_mask]
    rows = [score_subset(sub, lead, discretize=discretize) for lead in LEADS]
    rows = [r for r in rows if r is not None]
    out = pd.DataFrame(rows)
    out.insert(0, "subset", label)
    return out


def to_md_table(df, cols, headers):
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    df = pd.read_parquet(PRED_PATH)

    whole = build_table(df, pd.Series(True, index=df.index), "Ethiopia (all 3 livelihood zones, 92 zones)")
    pastoral = build_table(df, df["lhz"] == "pastoral", "Pastoral (20 zones)")
    agropastoral = build_table(df, df["lhz"] == "agropastoral", "Agro-pastoral (5 zones)")
    crop_farming = build_table(df, df["lhz"] == "crop_farming", "Crop farming (67 zones)")

    cols = ["lead", "n", "accuracy", "f1_weighted", "f1_macro",
            "precision_weighted", "recall_weighted", "mae", "r2"]
    headers = ["lead", "n", "accuracy", "F1 (weighted)", "F1 (macro)",
               "precision (weighted)", "recall (weighted)", "MAE", "R2"]

    def summary_row(out, label):
        return (f"- **{label}**: mean accuracy across leads = "
                f"{out['accuracy'].mean():.4f}, mean weighted F1 = "
                f"{out['f1_weighted'].mean():.4f}\n")

    md = []
    md.append("# Ethiopia-dataset F1 / accuracy — RQ1 Experiment 2\n")
    md.append(
        "Companion to `report.md`. Same nearest-integer discretization as "
        "`experiments/RQ1/experiment_1/ethiopia_pastoral_f1_accuracy.md`, so "
        "the two are directly comparable — see `report.md` section 3.\n"
    )
    md.append(
        "**Discretization.** `ipc_continuous` and predictions are rounded to "
        "the nearest integer IPC phase and clipped to [1, 5] before scoring "
        "accuracy/F1/precision/recall. MAE and R2 are computed on the "
        "original continuous values (same numbers as "
        "`metrics/metrics_per_cluster.csv`), included here for reference "
        "alongside the classification metrics, not recomputed differently.\n"
    )

    md.append("\n## Ethiopia, all 3 livelihood zones (92 zones)\n")
    md.append(to_md_table(whole, cols, headers))
    md.append("\n" + summary_row(whole, "Ethiopia (all zones)"))

    md.append("\n## Pastoral (20 zones)\n")
    md.append(to_md_table(pastoral, cols, headers))
    md.append("\n" + summary_row(pastoral, "Pastoral"))

    md.append("\n## Agro-pastoral (5 zones)\n")
    md.append(to_md_table(agropastoral, cols, headers))
    md.append("\n" + summary_row(agropastoral, "Agro-pastoral"))

    md.append("\n## Crop farming (67 zones)\n")
    md.append(to_md_table(crop_farming, cols, headers))
    md.append("\n" + summary_row(crop_farming, "Crop farming"))

    # 1/2/3+ discretization (added 2026-08-24, per project owner request):
    # same predictions/observed values, collapsed to FEWS NET's own
    # "Crisis or worse" humanitarian-response threshold instead of the
    # full 5-point IPC scale. Not directly comparable to the 1-5 table
    # above -- a coarser task is mechanically easier -- see the note
    # printed alongside it below.
    whole_3 = build_table(df, pd.Series(True, index=df.index), "Ethiopia (all 3 livelihood zones, 92 zones)",
                           discretize=to_ipc_class_3plus)
    pastoral_3 = build_table(df, df["lhz"] == "pastoral", "Pastoral (20 zones)", discretize=to_ipc_class_3plus)
    agropastoral_3 = build_table(df, df["lhz"] == "agropastoral", "Agro-pastoral (5 zones)",
                                  discretize=to_ipc_class_3plus)
    crop_farming_3 = build_table(df, df["lhz"] == "crop_farming", "Crop farming (67 zones)",
                                  discretize=to_ipc_class_3plus)

    md.append("\n---\n")
    md.append("\n# 1 / 2 / 3+ discretization (added 2026-08-24)\n")
    md.append(
        "Same predictions and observed values as above, collapsed to 3 "
        "classes instead of 5: **1** (Minimal), **2** (Stressed), **3+** "
        "(Crisis/Emergency/Famine combined) -- FEWS NET's own standard "
        "\"Crisis or worse\" humanitarian-response threshold "
        "(`to_ipc_class_3plus()`: same round-then-clip logic as the 1-5 "
        "version, clipped at 3 instead of 5). **Not directly comparable to "
        "the 1-5 table above -- collapsing 3 classes into 1 mechanically "
        "raises F1/accuracy regardless of model quality**, since there are "
        "fewer, more separated decision boundaries to get right. Reported "
        "because it answers a different, policy-relevant question (\"did "
        "the model call the crisis threshold correctly?\") rather than the "
        "exact IPC phase. Directly comparable to "
        "`experiments/RQ1/experiment_1/ethiopia_pastoral_f1_accuracy.md`'s "
        "own 1/2/3+ section, same discretization function.\n"
    )

    md.append("\n## Ethiopia, all 3 livelihood zones (92 zones)\n")
    md.append(to_md_table(whole_3, cols, headers))
    md.append("\n" + summary_row(whole_3, "Ethiopia (all zones)"))

    md.append("\n## Pastoral (20 zones)\n")
    md.append(to_md_table(pastoral_3, cols, headers))
    md.append("\n" + summary_row(pastoral_3, "Pastoral"))

    md.append("\n## Agro-pastoral (5 zones)\n")
    md.append(to_md_table(agropastoral_3, cols, headers))
    md.append("\n" + summary_row(agropastoral_3, "Agro-pastoral"))

    md.append("\n## Crop farming (67 zones)\n")
    md.append(to_md_table(crop_farming_3, cols, headers))
    md.append("\n" + summary_row(crop_farming_3, "Crop farming"))

    with open(OUT_PATH, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {OUT_PATH}")

    for name, out in [("Ethiopia (all zones)", whole), ("Pastoral", pastoral),
                       ("Agro-pastoral", agropastoral), ("Crop farming", crop_farming)]:
        print(f"{name}: mean accuracy={out['accuracy'].mean():.4f}, "
              f"mean weighted F1={out['f1_weighted'].mean():.4f}")
    print("\n1/2/3+ discretization:")
    for name, out in [("Ethiopia (all zones)", whole_3), ("Pastoral", pastoral_3),
                       ("Agro-pastoral", agropastoral_3), ("Crop farming", crop_farming_3)]:
        print(f"{name}: mean accuracy={out['accuracy'].mean():.4f}, "
              f"mean weighted F1={out['f1_weighted'].mean():.4f}")


if __name__ == "__main__":
    main()
