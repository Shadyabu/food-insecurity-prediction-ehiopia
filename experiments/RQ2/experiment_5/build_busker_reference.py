"""Busker-reproduced-baseline reference series for RQ2 Experiment 5's
top-5 ensemble figures -- did not exist before the 2026-08-26 figure
audit (only Persistence was plotted as a reference; the dissertation
plan explicitly asks for the Busker comparison here too).

Writes `busker_detailed_metrics.csv` (same schema as the existing
`persistence_detailed_metrics.csv`: combo/subset/lead/n/accuracy/
f1_weighted/mae/r2) into both `all_combinations/` (default window) and
`all_combinations_extended/` (extended window) -- the Busker-architecture
run itself (`busker_arch_datesplit_2020_2022`) is identical for both,
since its own target data ends 2022-06 regardless of which ensemble
window it's being compared against.
"""

import os

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADS = [0, 1, 2, 3, 4, 8, 12]
SUBSETS = ["all", "pastoral", "agropastoral", "crop_farming"]
LHZ_RENAME = {"p": "pastoral", "ap": "agropastoral", "other": "crop_farming"}
BUSKER_PATH = os.path.join(SCRIPT_DIR, "..", "..", "RQ1", "experiment_1",
                            "busker_arch_datesplit_2020_2022", "predictions.parquet")


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def main():
    df = pd.read_parquet(BUSKER_PATH)
    df = df[df["country"] == "Ethiopia"].copy()
    df["lhz"] = df["lhz"].map(LHZ_RENAME)

    rows = []
    for subset in SUBSETS:
        sub_all = df if subset == "all" else df[df["lhz"] == subset]
        for lead in LEADS:
            g = sub_all[sub_all["lead"] == lead]
            if len(g) == 0:
                continue
            yt, yp = to_ipc_class(g["observed"]), to_ipc_class(g["prediction"])
            rows.append({
                "combo": "Busker (reference)", "subset": subset, "lead": lead, "n": len(g),
                "accuracy": accuracy_score(yt, yp),
                "f1_weighted": f1_score(yt, yp, average="weighted", zero_division=0),
                "mae": mean_absolute_error(g["observed"], g["prediction"]),
                "r2": r2_score(g["observed"], g["prediction"]) if g["observed"].nunique() > 1 else np.nan,
            })
    out = pd.DataFrame(rows)

    for target_dir in ["all_combinations", "all_combinations_extended"]:
        out_path = os.path.join(SCRIPT_DIR, target_dir, "busker_detailed_metrics.csv")
        out.to_csv(out_path, index=False)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
