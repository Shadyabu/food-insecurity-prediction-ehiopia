"""Web export, Phase 1a -- convert the admin2 boundary shapefile
(boundaries/eth_admbnda_adm2_csa_bofedb_2021.shp, the fixed 92-zone key
every pipeline in this project joins on, per CLAUDE.md section 3.1) to a
simplified GeoJSON the frontend map can load directly.

Geometry is simplified (topology-preserving) purely to shrink file size for
a browser fetch -- the CSA/BOFEDB shapefile is already the project's
official boundary catalog, this does not touch which catalog is used.
"""

from pathlib import Path

import geopandas as gpd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SHAPEFILE_PATH = REPO_ROOT / "boundaries" / "eth_admbnda_adm2_csa_bofedb_2021.shp"
OUT_PATH = REPO_ROOT / "frontend" / "data" / "admin2_boundaries.geojson"

# Degrees, not meters -- CRS is EPSG:4326. ~0.01 deg is a small enough
# simplification to keep zone shapes recognizable while cutting file size.
SIMPLIFY_TOLERANCE_DEG = 0.01


def main():
    gdf = gpd.read_file(SHAPEFILE_PATH)
    assert gdf.crs is not None and gdf.crs.to_epsg() == 4326, f"expected EPSG:4326, got {gdf.crs}"
    assert gdf["ADM2_PCODE"].is_unique, "ADM2_PCODE must be unique"
    assert len(gdf) == 92, f"expected 92 admin2 zones, got {len(gdf)}"

    keep = gdf[["ADM2_PCODE", "ADM2_EN", "ADM1_EN", "geometry"]].copy()
    keep["geometry"] = keep["geometry"].simplify(SIMPLIFY_TOLERANCE_DEG, preserve_topology=True)
    # Simplification must not drop a zone's geometry entirely.
    assert keep["geometry"].notna().all() and (~keep["geometry"].is_empty).all(), \
        "simplification produced an empty/null geometry for at least one zone"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep.to_file(OUT_PATH, driver="GeoJSON")

    before_mb = SHAPEFILE_PATH.stat().st_size / 1e6
    after_mb = OUT_PATH.stat().st_size / 1e6
    print(f"Wrote {len(keep)} zones to {OUT_PATH}")
    print(f"Shapefile size: {before_mb:.2f} MB -> GeoJSON size: {after_mb:.2f} MB")


if __name__ == "__main__":
    main()
