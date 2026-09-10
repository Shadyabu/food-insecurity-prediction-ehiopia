"""Cheap follow-up to the exhaustive 26-combination search: does the
*aggregation rule* matter, not just *which members* are in the ensemble?
Rescores all 26 combinations (both test windows) under two alternatives
to the plain mean already used everywhere else in RQ2 Experiment 5:

  - median: element-wise median of members' continuous `prediction`
    columns, rounded to class at scoring time (same round+clip convention
    as everywhere else) -- more robust to one member being far off than
    a mean is.
  - majority-vote: each member's own prediction is discretized to an IPC
    class FIRST, then the modal class across members wins; ties (e.g. a
    2-2 split in a 4-member combo, or 3 members disagreeing 3 ways in a
    3-member combo) are broken by whichever tied class is closest to the
    row's mean continuous prediction -- a simple, defensible rule, not a
    new source of leakage (uses only the same members' own predictions).

No retraining, no new validation split, no new leakage risk -- this only
changes how already-fitted predictions are combined, which is why it's
worth checking before investing in a weighted-average or stacked
meta-learner (both of which DO need a held-out validation split to avoid
overfitting ensemble weights to the test set -- not attempted here).

Usage: python3 compare_aggregation_methods.py --window {default,extended}
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LEADS = [0, 1, 2, 3, 4, 8, 12]
CLUSTERS = ["pastoral", "agropastoral", "crop_farming"]
EXTENDED_CUTOFF = pd.Timestamp("2024-10-01")

MEMBER_PATHS = {
    "default": {
        "XGB": "../../RQ1/experiment_2/unhcr_delta_noclimate/predictions.parquet",
        "RF": "../experiment_4/rf_delta_noclimate/predictions.parquet",
        "LSTM": "../experiment_1/lstm_delta/predictions.parquet",
        "TabICL": "../experiment_2/tabicl_level/predictions.parquet",
        "TGCN": "../experiment_3/tgcn_ce/predictions.parquet",
    },
    "extended": {
        "XGB": "../../RQ1/experiment_2/unhcr_delta_noclimate_extended/predictions.parquet",
        "RF": "../experiment_4/rf_delta_noclimate_extended/predictions.parquet",
        "LSTM": "../experiment_1/lstm_delta/predictions.parquet",
        "TabICL": "../experiment_2/tabicl_level/predictions.parquet",
        "TGCN": "../experiment_3/tgcn_ce/predictions.parquet",
    },
}
WINDOW_FILTER = {"default": "default", "extended": "extended"}


def to_ipc_class(x):
    return np.clip(np.round(x), 1, 5).astype(int)


def load_member(script_dir, name, rel_path, window_key):
    path = os.path.join(script_dir, rel_path)
    df = pd.read_parquet(path)
    if "window" in df.columns:
        df = df[df["window"] == WINDOW_FILTER[window_key]].copy()
    df["time"] = pd.to_datetime(df["time"])
    if window_key == "extended":
        df = df[df["time"] <= EXTENDED_CUTOFF]

    if name == "TGCN":
        df["observed"] = df["true_class"] + 1
        df["prediction"] = df["pred_class"] + 1
    key = ["time", "zone_code", "lead"]
    return df.set_index(key).sort_index()[["lhz", "observed", "prediction"]]


def load_all_members(window_key):
    paths = MEMBER_PATHS[window_key]
    members = {name: load_member(SCRIPT_DIR, name, path, window_key) for name, path in paths.items()}
    ref_key = members["XGB"].index
    for name, df in members.items():
        assert set(df.index) == set(ref_key), f"key mismatch: {name}"
        members[name] = df.loc[ref_key]

    observed = members["XGB"]["observed"]
    lhz = members["XGB"]["lhz"]
    pred_matrix = pd.DataFrame({name: df["prediction"] for name, df in members.items()})
    return pred_matrix, observed, lhz


def majority_vote(class_matrix, continuous_mean):
    """Row-wise modal class across members; ties broken by proximity to
    the row's mean continuous prediction."""
    def vote_row(row):
        counts = row.value_counts()
        top = counts[counts == counts.max()].index.tolist()
        if len(top) == 1:
            return top[0]
        cmean = continuous_mean.loc[row.name]
        return min(top, key=lambda c: abs(c - cmean))
    return class_matrix.apply(vote_row, axis=1)


def score(observed, lhz, prediction_class_or_continuous, already_class=False):
    """mean weighted F1 across 7 leads, for 'all' + 3 clusters."""
    frame = pd.DataFrame({"lhz": lhz, "observed": observed, "prediction": prediction_class_or_continuous})
    frame = frame.reset_index()
    out = {}
    for subset_name, mask in [("all", pd.Series(True, index=frame.index))] + \
                              [(c, frame["lhz"] == c) for c in CLUSTERS]:
        sub_all = frame[mask]
        vals = []
        for lead in LEADS:
            sub = sub_all[sub_all["lead"] == lead]
            if len(sub) == 0:
                continue
            y_true = to_ipc_class(sub["observed"])
            y_pred = sub["prediction"] if already_class else to_ipc_class(sub["prediction"])
            vals.append(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        out[f"mean_f1_{subset_name}"] = np.mean(vals) if vals else np.nan
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window", choices=["default", "extended"], default="default")
    args = parser.parse_args()

    pred_matrix, observed, lhz = load_all_members(args.window)
    class_matrix = pred_matrix.apply(to_ipc_class)
    names = list(MEMBER_PATHS[args.window].keys())

    combos = []
    for size in range(2, 6):
        combos.extend(itertools.combinations(names, size))
    assert len(combos) == 26

    rows = []
    for combo in combos:
        label = "+".join(combo)
        combo_pred_matrix = pred_matrix[list(combo)]
        combo_class_matrix = class_matrix[list(combo)]

        mean_pred = combo_pred_matrix.mean(axis=1)
        median_pred = combo_pred_matrix.median(axis=1)
        vote_pred = majority_vote(combo_class_matrix, mean_pred)

        for agg_name, pred, already_class in [
            ("mean", mean_pred, False),
            ("median", median_pred, False),
            ("majority_vote", vote_pred, True),
        ]:
            s = score(observed, lhz, pred, already_class=already_class)
            s["combo"] = label
            s["size"] = len(combo)
            s["agg_method"] = agg_name
            s["includes_tgcn"] = "TGCN" in combo
            rows.append(s)

    result = pd.DataFrame(rows)
    cols = ["combo", "size", "agg_method", "includes_tgcn", "mean_f1_all",
            "mean_f1_pastoral", "mean_f1_agropastoral", "mean_f1_crop_farming"]
    result = result[cols].sort_values(["agg_method", "mean_f1_all"], ascending=[True, False])
    out_path = os.path.join(SCRIPT_DIR, f"aggregation_comparison_{args.window}.csv")
    result.to_csv(out_path, index=False)

    print(f"=== Aggregation method comparison, {args.window} window ===\n")
    for agg in ["mean", "median", "majority_vote"]:
        top = result[result["agg_method"] == agg].head(5)
        print(f"--- top 5 under {agg} ---")
        print(top[["combo", "mean_f1_all"]].to_string(index=False))
        print()

    # Direct same-combo comparison: does the aggregation rule change the score
    # for the two combos we most care about -- the current best (no T-GCN) and
    # the full 5-way (with T-GCN, currently the worst-affected group)?
    print("--- same-combo comparison: mean vs median vs majority_vote ---")
    pivot = result.pivot_table(index="combo", columns="agg_method", values="mean_f1_all")
    pivot["best_agg"] = pivot[["mean", "median", "majority_vote"]].idxmax(axis=1)
    pivot = pivot.sort_values("mean", ascending=False)
    print(pivot.to_string())
    pivot.to_csv(os.path.join(SCRIPT_DIR, f"aggregation_pivot_{args.window}.csv"))

    n_median_wins = (pivot["median"] > pivot["mean"]).sum()
    n_vote_wins = (pivot["majority_vote"] > pivot["mean"]).sum()
    print(f"\nmedian beats mean on {n_median_wins}/26 combos; "
          f"majority_vote beats mean on {n_vote_wins}/26 combos")

    print(f"\nwrote {out_path}, aggregation_pivot_{args.window}.csv")


if __name__ == "__main__":
    main()
