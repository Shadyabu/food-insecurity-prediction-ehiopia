"""Cross-model 1-5 vs. 1/2/3+ (Crisis-or-worse) weighted F1 comparison --
closes the "no figure compares our own models at the 3+ threshold" gap
flagged 2026-09-02. Reads the already-scored numbers from
score_1_2_3plus_coverage.py (RQ2 architectures/ensembles) plus the
already-existing RQ1/FEWS NET classification-metrics outputs -- no new
model runs, pure rescoring/plotting.

Form: grouped bar, 2 color slots (1-5 vs 3+), fixed order -- NOT one hue
per model (11+ configs would violate the "never cycle past ~5 categorical
slots" rule). Models are x-axis identity, not color identity. This
project's own architectures use the validated categorical palette's slot
1/2 pair (#2a78d6 / #eb6834, same pair already used in
all_combinations/build_figures.py); Busker/Persistence/FEWS NET are
external reference points, shown desaturated (gray family) so they read
as "not this project's own model" at a glance -- same convention as
PERSISTENCE_COLOR/BUSKER_COLOR in the existing all_combinations figure.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

COLOR_1TO5 = "#2a78d6"
COLOR_3PLUS = "#eb6834"
COLOR_REF_1TO5 = "#b7b5b0"
COLOR_REF_3PLUS = "#6f6d68"

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})

# (label, f1_1to5, f1_3plus, is_reference)
# All-zones/Ethiopia-wide mean weighted F1 across leads, default/2020-2024
# window (matching every RQ1/RQ2 headline number already reported) except
# where noted. Sourced from:
#   - Busker/XGBoost/FEWS NET: experiments/RQ1/experiment_1 and
#     experiment_2's own ethiopia_*_f1_accuracy.md / fews_outlook_f1_*.csv
#   - RQ2 architectures/ensembles: rq2_1_2_3plus_by_lead.csv (this
#     experiment folder, built 2026-09-02)
DATA = [
    ("Busker architecture\n(RQ1 Exp1)", 0.6895, 0.7257, True),
    ("XGBoost, Ethiopia\n(plain baseline)", 0.6706, 0.7115, False),
    ("XGBoost, Ethiopia\n(best variant, RQ1 §7.4)", 0.7074, 0.7422, False),
    ("RandomForest\n(delta+noclimate)", 0.7093, 0.7486, False),
    ("LSTM (delta)", 0.6641, 0.7003, False),
    ("TabICLv2", 0.6336, 0.7023, False),
    ("T-GCN (CE)", 0.3358, 0.3403, False),
    ("Ensemble, 2-way\n(XGB+RF blend)", 0.7098, 0.7458, False),
    ("Ensemble, 5-way\n(all architectures)", 0.6798, 0.7389, False),
    ("Ensemble, 4-way\nchampion (RQ2 Exp5)", 0.7218, 0.7581, False),
    ("Persistence", 0.7009, 0.7297, True),
    ("FEWS NET outlook\n(2020-2024)", 0.7395, 0.7887, True),
]


def main():
    df = pd.DataFrame(DATA, columns=["label", "f1_1to5", "f1_3plus", "is_ref"])
    df = df.sort_values("f1_1to5", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8.5))
    y = range(len(df))
    height = 0.35

    c15 = [COLOR_REF_1TO5 if r else COLOR_1TO5 for r in df["is_ref"]]
    c3p = [COLOR_REF_3PLUS if r else COLOR_3PLUS for r in df["is_ref"]]

    ax.barh([i + height / 2 for i in y], df["f1_1to5"], height=height, color=c15, label="Weighted F1 (1-5 phase)")
    ax.barh([i - height / 2 for i in y], df["f1_3plus"], height=height, color=c3p, label="Weighted F1 (1/2/3+ crisis threshold)")

    for i, (v15, v3p) in enumerate(zip(df["f1_1to5"], df["f1_3plus"])):
        ax.text(v15 + 0.008, i + height / 2, f"{v15:.3f}", va="center", fontsize=8, color="#3a3936")
        ax.text(v3p + 0.008, i - height / 2, f"{v3p:.3f}", va="center", fontsize=8, color="#3a3936")

    ax.set_yticks(list(y))
    ax.set_yticklabels(df["label"], fontsize=9)
    ax.set_xlabel("Mean weighted F1 across leads (Ethiopia, all zones)")
    ax.set_xlim(0, 0.92)
    ax.set_ylim(-0.9, len(df) - 0.3)
    ax.set_title("Weighted F1 at both discretizations: exact IPC phase (1-5) vs.\nCrisis-or-worse threshold (1/2/3+)")

    # Legend: 2 series + a note distinguishing this project's own models
    # from external reference points (gray), per the color-by-job rule.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_1TO5),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_3PLUS),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_REF_1TO5),
    ]
    labels = ["1-5 phase (this project's models)", "1/2/3+ crisis threshold (this project's models)",
              "External reference (Busker / Persistence / FEWS NET), both scales shown desaturated"]
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=1, fontsize=8, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "all_models_1to5_vs_3plus.png"), dpi=200, bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "all_models_1to5_vs_3plus.svg"), bbox_inches="tight")
    print(f"wrote {FIG_DIR}/all_models_1to5_vs_3plus.{{png,svg}}")


if __name__ == "__main__":
    main()
