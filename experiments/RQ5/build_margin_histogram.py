"""RQ5 Phase 1 — histogram of top1-vs-top2 category margins across the full
scored sample, to sanity-check taxonomy_config.yaml's compound.margin_threshold
(default 0.15) before treating it as final. Only rows with a real predicted
cause (top1_score > 0) are included -- NO_DATA/UNCLASSIFIED rows have no
meaningful margin.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

RQ5_DIR = Path(__file__).resolve().parent


def main():
    df = pd.read_csv(RQ5_DIR / "predicted_causes.csv")
    scored = df[df["predicted_cause"].isin(["DROUGHT", "FLOODING", "CONFLICT", "MARKET_SHOCK", "COMPOUND"])]
    margins = scored["margin"].dropna()

    print(f"n scored rows: {len(scored)}, n with a defined margin: {len(margins)}")
    print(margins.describe())
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        pct = (margins <= thresh).mean() * 100
        print(f"  share COMPOUND-flagged at margin_threshold={thresh:.2f}: {pct:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(margins, bins=50, color="#4C72B0", edgecolor="white")
    axes[0].axvline(0.15, color="#C44E52", linestyle="--", label="default threshold (0.15)")
    axes[0].set_xlabel("top1-vs-top2 category margin")
    axes[0].set_ylabel("count")
    axes[0].set_title("All subjects/clusters/leads pooled")
    axes[0].legend()

    for subject, g in scored.groupby("subject"):
        m = g["margin"].dropna()
        axes[1].hist(m, bins=50, alpha=0.5, label=subject, density=True)
    axes[1].axvline(0.15, color="#C44E52", linestyle="--")
    axes[1].set_xlabel("top1-vs-top2 category margin")
    axes[1].set_ylabel("density")
    axes[1].set_title("By subject")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(RQ5_DIR / "margin_histogram.png", dpi=150)
    print(f"\nwrote {RQ5_DIR}/margin_histogram.png")


if __name__ == "__main__":
    main()
