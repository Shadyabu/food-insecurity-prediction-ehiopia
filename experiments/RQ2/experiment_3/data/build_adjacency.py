"""Build a queen-contiguity adjacency matrix over the project's fixed 92-zone
admin2 boundary (boundaries/eth_admbnda_adm2_csa_bofedb_2021.shp, per
CLAUDE.md Sec 3.1 -- the same boundary/key every other pipeline in this
project uses, not a new catalog).

Two zones are adjacent if their polygons share any boundary point (edge or
vertex) -- queen contiguity, the standard choice for admin-boundary GNNs and
the more permissive of the two conventions (rook contiguity requires a shared
edge, not just a corner point). Output is node-count-agnostic: any admin2
zone count works, nothing here is hardcoded to Ethiopia's 92.

Outputs (this experiment's own data dir, not a new pipelines/ directory --
mirrors experiments/RQ2/experiment_1/build_lstm_sequences.py's precedent of
keeping experiment-specific data construction inside the experiment folder):
  - adjacency.npy       : (N, N) float32 binary contiguity matrix (no self-loops)
  - zone_order.json     : ordered list of zone_code (ADM2_PCODE), index i <-> row/col i
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
BOUNDARY_SHP = REPO_ROOT / "boundaries" / "eth_admbnda_adm2_csa_bofedb_2021.shp"
OUT_DIR = Path(__file__).resolve().parent
ZONE_KEY = "ADM2_PCODE"


def build_queen_adjacency(gdf: gpd.GeoDataFrame, zone_key: str = ZONE_KEY) -> tuple[np.ndarray, list[str]]:
    """Return (adjacency, zone_order). adjacency[i, j] == 1 iff zone i and
    zone j share a boundary point and i != j. No self-loops (added later by
    normalize_adjacency, per the T-GCN spec's A_tilde = A + I)."""
    gdf = gdf.sort_values(zone_key).reset_index(drop=True)
    zone_order = gdf[zone_key].tolist()
    n = len(gdf)
    assert len(set(zone_order)) == n, "zone_key must be unique per row"

    # sjoin with predicate='touches' finds all polygon pairs sharing a
    # boundary point (edge or vertex) -- queen contiguity.
    joined = gpd.sjoin(gdf[[zone_key, "geometry"]], gdf[[zone_key, "geometry"]],
                        predicate="touches", how="inner")

    adjacency = np.zeros((n, n), dtype=np.float32)
    idx_of = {z: i for i, z in enumerate(zone_order)}
    left_col = f"{zone_key}_left"
    right_col = f"{zone_key}_right"
    for left, right in zip(joined[left_col], joined[right_col]):
        i, j = idx_of[left], idx_of[right]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0

    np.fill_diagonal(adjacency, 0.0)
    return adjacency, zone_order


def main() -> None:
    gdf = gpd.read_file(BOUNDARY_SHP)
    adjacency, zone_order = build_queen_adjacency(gdf)

    n = len(zone_order)
    degree = adjacency.sum(axis=1)
    isolated = [zone_order[i] for i in range(n) if degree[i] == 0]

    print(f"zones: {n}")
    print(f"edges (undirected): {int(adjacency.sum()) // 2}")
    print(f"degree: min={degree.min():.0f} max={degree.max():.0f} mean={degree.mean():.2f}")
    if isolated:
        print(f"WARNING: {len(isolated)} zone(s) with zero neighbors (contiguity-graph islands): {isolated}")

    np.save(OUT_DIR / "adjacency.npy", adjacency)
    with open(OUT_DIR / "zone_order.json", "w") as f:
        json.dump(zone_order, f, indent=2)
    print(f"wrote {OUT_DIR / 'adjacency.npy'} and {OUT_DIR / 'zone_order.json'}")


if __name__ == "__main__":
    main()
