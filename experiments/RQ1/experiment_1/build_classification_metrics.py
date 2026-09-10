"""RQ1_Experiment1_Busker_Reproduction.md section 6 (additional notes):
track F1 score and average accuracy for (a) Ethiopia only and (b) pastoral
livelihood-zone regions, across all lead times, as a separate document from
the main regression-metric reproduction.

Busker's own target (FEWS_CS) is continuous (population-weighted mean IPC
value). There is no classification metric anywhere in his released code
(confirmed in CLAUDE.md section 7a: "There is no F1 anywhere in his work.").
To compute accuracy/F1 here, each continuous value is discretized to the
nearest integer IPC phase (1-5, the FEWS IPC scale), clipped to [1, 5], for
both observed and predicted values -- the same discretization used nowhere
else in this experiment, so it is confined to this document.
"""

import argparse

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-path", type=str, default="predictions/predictions.parquet")
    parser.add_argument("--out-path", type=str, default="ethiopia_pastoral_f1_accuracy.md")
    parser.add_argument("--csv-dir", type=str, default=None,
                         help="if given, also write the 1-5 ethiopia/pastoral_all/"
                              "ethiopia_pastoral tables as CSVs here (for figure-building)")
    args = parser.parse_args()

    df = pd.read_parquet(args.pred_path)

    ethiopia = build_table(df, df["country"] == "Ethiopia", "Ethiopia (all livelihood zones)")
    pastoral_all = build_table(df, df["lhz"] == "p", "Pastoral (all 3 countries)")
    ethiopia_pastoral = build_table(
        df, (df["country"] == "Ethiopia") & (df["lhz"] == "p"), "Ethiopia pastoral only"
    )
    whole_region = build_table(df, pd.Series(True, index=df.index), "Whole region (all 213 units, all zones)")

    if args.csv_dir:
        import os
        os.makedirs(args.csv_dir, exist_ok=True)
        ethiopia.to_csv(os.path.join(args.csv_dir, "ethiopia.csv"), index=False)
        pastoral_all.to_csv(os.path.join(args.csv_dir, "pastoral_all.csv"), index=False)
        ethiopia_pastoral.to_csv(os.path.join(args.csv_dir, "ethiopia_pastoral.csv"), index=False)

    cols = ["lead", "n", "accuracy", "f1_weighted", "f1_macro",
            "precision_weighted", "recall_weighted", "mae", "r2"]
    headers = ["lead", "n", "accuracy", "F1 (weighted)", "F1 (macro)",
               "precision (weighted)", "recall (weighted)", "MAE", "R2"]

    def summary_row(out, label):
        return (f"- **{label}**: mean accuracy across leads = "
                f"{out['accuracy'].mean():.4f}, mean weighted F1 = "
                f"{out['f1_weighted'].mean():.4f}\n")

    md = []
    md.append("# Ethiopia-only and pastoral-region F1 / accuracy — RQ1 Experiment 1\n")
    md.append(
        "Companion to `report.md`, per the experiment spec's additional notes "
        "(section 6): F1 score and accuracy, tracked separately for Ethiopia "
        "and for pastoral regions, across all lead times. Not part of the main "
        "reproduction criterion (section 3), which is regression (R2/MAE) "
        "only — Busker et al. never report a classification metric "
        "(see CLAUDE.md section 7a).\n"
    )
    md.append(
        "**Discretization.** Busker's target and predictions are continuous "
        "(population-weighted mean IPC). Each value here is rounded to the "
        "nearest integer IPC phase and clipped to [1, 5] before scoring "
        "accuracy/F1/precision/recall. MAE and R2 are computed on the "
        "original continuous values (same numbers as `metrics/metrics_per_country.csv` "
        "and `metrics/metrics_per_cluster.csv`), included here for reference "
        "alongside the classification metrics, not recomputed differently.\n"
    )
    md.append(
        "**\"Pastoral regions\" is ambiguous in the source instruction "
        "(Ethiopia's pastoral zone only, vs. all pastoral-zone units region-wide). "
        "Both readings are reported below, plus the whole-region figure for scale.**\n"
    )

    md.append("\n## Ethiopia (all 3 livelihood zones, 92 units)\n")
    md.append(to_md_table(ethiopia, cols, headers))
    md.append("\n" + summary_row(ethiopia, "Ethiopia"))

    md.append("\n## Pastoral livelihood zone, all 3 countries (82 units)\n")
    md.append(to_md_table(pastoral_all, cols, headers))
    md.append("\n" + summary_row(pastoral_all, "Pastoral (all countries)"))

    md.append("\n## Ethiopia pastoral zone only (18 units)\n")
    md.append(to_md_table(ethiopia_pastoral, cols, headers))
    md.append("\n" + summary_row(ethiopia_pastoral, "Ethiopia pastoral"))

    md.append("\n## Whole region, for scale (213 units, all zones)\n")
    md.append(to_md_table(whole_region, cols, headers))
    md.append("\n" + summary_row(whole_region, "Whole region"))

    # 1/2/3+ discretization (added 2026-08-24, per project owner request):
    # same predictions/observed values, same 4 subsets, just collapsed to
    # FEWS NET's own "Crisis or worse" humanitarian-response threshold
    # instead of the full 5-point IPC scale. A coarser classification task
    # is mechanically easier (fewer, more separated decision boundaries),
    # so these numbers are not directly comparable to the 1-5 table above
    # -- they answer a different, policy-relevant question ("did we call
    # the crisis threshold right?") rather than "did we call the exact
    # phase right?".
    ethiopia_3 = build_table(df, df["country"] == "Ethiopia", "Ethiopia (all livelihood zones)",
                              discretize=to_ipc_class_3plus)
    pastoral_all_3 = build_table(df, df["lhz"] == "p", "Pastoral (all 3 countries)",
                                  discretize=to_ipc_class_3plus)
    ethiopia_pastoral_3 = build_table(
        df, (df["country"] == "Ethiopia") & (df["lhz"] == "p"), "Ethiopia pastoral only",
        discretize=to_ipc_class_3plus
    )
    whole_region_3 = build_table(df, pd.Series(True, index=df.index), "Whole region (all 213 units, all zones)",
                                  discretize=to_ipc_class_3plus)

    md.append("\n---\n")
    md.append("\n# 1 / 2 / 3+ discretization (added 2026-08-24)\n")
    md.append(
        "Same predictions and observed values as above, collapsed to 3 "
        "classes instead of 5: **1** (Minimal), **2** (Stressed), **3+** "
        "(Crisis/Emergency/Famine combined) -- FEWS NET's own standard "
        "\"Crisis or worse\" humanitarian-response threshold "
        "(`to_ipc_class_3plus()`: same round-then-clip logic as the 1-5 "
        "version, clipped at 3 instead of 5, so the .5 rounding boundaries "
        "are unchanged). **Not directly comparable to the 1-5 table above "
        "-- collapsing 3 classes into 1 mechanically raises F1/accuracy "
        "regardless of model quality**, since there are fewer, more "
        "separated decision boundaries to get right. Reported because it "
        "answers a different, policy-relevant question (\"did the model "
        "call the crisis threshold correctly?\") rather than the exact IPC "
        "phase.\n"
    )

    md.append("\n## Ethiopia (all 3 livelihood zones, 92 units)\n")
    md.append(to_md_table(ethiopia_3, cols, headers))
    md.append("\n" + summary_row(ethiopia_3, "Ethiopia"))

    md.append("\n## Pastoral livelihood zone, all 3 countries (82 units)\n")
    md.append(to_md_table(pastoral_all_3, cols, headers))
    md.append("\n" + summary_row(pastoral_all_3, "Pastoral (all countries)"))

    md.append("\n## Ethiopia pastoral zone only (18 units)\n")
    md.append(to_md_table(ethiopia_pastoral_3, cols, headers))
    md.append("\n" + summary_row(ethiopia_pastoral_3, "Ethiopia pastoral"))

    md.append("\n## Whole region, for scale (213 units, all zones)\n")
    md.append(to_md_table(whole_region_3, cols, headers))
    md.append("\n" + summary_row(whole_region_3, "Whole region"))

    with open(args.out_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {args.out_path}")

    for name, out in [("Ethiopia", ethiopia), ("Pastoral (all)", pastoral_all),
                       ("Ethiopia pastoral", ethiopia_pastoral)]:
        print(f"{name}: mean accuracy={out['accuracy'].mean():.4f}, "
              f"mean weighted F1={out['f1_weighted'].mean():.4f}")
    print("\n1/2/3+ discretization:")
    for name, out in [("Ethiopia", ethiopia_3), ("Pastoral (all)", pastoral_all_3),
                       ("Ethiopia pastoral", ethiopia_pastoral_3)]:
        print(f"{name}: mean accuracy={out['accuracy'].mean():.4f}, "
              f"mean weighted F1={out['f1_weighted'].mean():.4f}")


if __name__ == "__main__":
    main()
