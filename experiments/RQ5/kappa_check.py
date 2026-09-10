"""RQ5 Phase 3 — inter-rater reliability between the project owner's two
labeling passes over the full 14-case curated set (per project owner
confirmation 2026-08-31: relabel ALL cases blind to the first pass, not
just a subsample, since the full set is small enough to do twice).

Usage:
    python3 kappa_check.py labeling_worksheet_pass1.csv labeling_worksheet_pass2.csv

Both files must have the same case_id set and a filled-in ground_truth_cause
column (values from {DROUGHT, FLOODING, CONFLICT, MARKET_SHOCK, COMPOUND}).
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

VALID_LABELS = {"DROUGHT", "FLOODING", "CONFLICT", "MARKET_SHOCK", "COMPOUND"}


def normalize_label(x: str) -> str:
    """Any label starting with COMPOUND (e.g. 'COMPOUND (DROUGHT + CONFLICT)',
    an annotated variant naming which categories compound) is treated as
    COMPOUND for validation/agreement purposes -- the full original string
    is preserved everywhere else (CSV output, disagreement listing)."""
    x = str(x).strip()
    return "COMPOUND" if x.upper().startswith("COMPOUND") else x


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    pass1 = pd.read_csv(sys.argv[1])
    pass2 = pd.read_csv(sys.argv[2])

    for name, df in [("pass1", pass1), ("pass2", pass2)]:
        blank = df["ground_truth_cause"].isna() | (df["ground_truth_cause"].astype(str).str.strip() == "")
        if blank.any():
            print(f"ERROR: {name} has {blank.sum()} blank ground_truth_cause rows "
                  f"({df.loc[blank, 'case_id'].tolist()}) -- fill in every case before running kappa.")
            sys.exit(1)
        bad_labels = {v for v in df["ground_truth_cause"].unique() if normalize_label(v) not in VALID_LABELS}
        if bad_labels:
            print(f"ERROR: {name} has labels outside the taxonomy: {bad_labels}. "
                  f"Valid labels are {sorted(VALID_LABELS)} (COMPOUND may be annotated, "
                  f"e.g. 'COMPOUND (DROUGHT + CONFLICT)').")
            sys.exit(1)

    merged = pass1[["case_id", "ground_truth_cause"]].merge(
        pass2[["case_id", "ground_truth_cause"]], on="case_id", suffixes=("_pass1", "_pass2"))
    if len(merged) != len(pass1) or len(merged) != len(pass2):
        missing1 = set(pass2["case_id"]) - set(pass1["case_id"])
        missing2 = set(pass1["case_id"]) - set(pass2["case_id"])
        print(f"ERROR: case_id sets differ. Only in pass2: {missing1}. Only in pass1: {missing2}.")
        sys.exit(1)

    norm1 = merged["ground_truth_cause_pass1"].apply(normalize_label)
    norm2 = merged["ground_truth_cause_pass2"].apply(normalize_label)
    kappa = cohen_kappa_score(norm1, norm2, labels=sorted(VALID_LABELS))
    agree = (norm1 == norm2).mean()
    disagreements = merged[(norm1 != norm2).to_numpy()]

    print(f"n cases: {len(merged)}")
    print(f"raw agreement (COMPOUND-normalized): {agree:.3f} ({(norm1 == norm2).sum()}/{len(merged)})")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"\n{len(disagreements)} disagreement(s):")
    if len(disagreements):
        print(disagreements.to_string(index=False))
    else:
        print("  (none)")

    print("\nNote (n=14, indicative not inferential): with this few cases, a single "
          "flipped label moves kappa substantially -- report the disagreement list "
          "itself alongside the kappa value, not the kappa value alone.")

    out_path = Path(__file__).resolve().parent / "kappa_results.csv"
    merged["agree"] = (norm1 == norm2).to_numpy()
    merged.to_csv(out_path, index=False)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
