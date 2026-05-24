#!/usr/bin/env python3
"""
==============================================================================
STEP 2: CREATE EMPTY DIGITIZATION SHAPEFILES
==============================================================================
Creates three empty shapefiles with proper schemas that you'll fill in by
drawing polygons in QGIS:

  1. haors_manual.shp    — haor boundaries (polygons)
  2. beels_manual.shp    — beel boundaries (polygons, smaller, inside haors)
  3. khals_manual.shp    — khal/channel network (lines, connecting features)

Why three separate files?
  - Different geometry types (polygon vs line)
  - Different attributes per feature type
  - Different rendering symbology in QGIS
  - Cleaner analysis later (you can query "all beels in Tanguar Haor")

Why this schema design?
  - haor_id / beel_id link parent-child relationships
  - flow_to lets you build a connectivity graph from khals
  - notes captures uncertainty (digitization confidence)
==============================================================================
"""

import os
import geopandas as gpd
from shapely.geometry import Polygon, LineString
import pandas as pd

# Output directory — matches your existing structure
OUT_DIR = "/work/a06/wasif/haor_flood_analysis/digitization"
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# SCHEMA 1: HAORS (polygons)
# ============================================================================
# Each row = one haor's maximum seasonal extent
haor_schema = {
    "haor_id": "str",         # unique ID e.g. "TGR" for Tanguar
    "name_en": "str",         # English name e.g. "Tanguar Haor"
    "name_bn": "str",         # Bengali name (optional)
    "area_km2": "float",      # computed area (fill after digitizing)
    "ramsar": "int",          # 1 = Ramsar site, 0 = not
    "district": "str",        # admin district
    "upazila": "str",         # sub-district
    "extent_type": "str",     # "dry" or "wet" — which extent you traced
    "source": "str",          # "S2_dry" / "S2_wet" / "JRC" / "FABDEM" — what you traced from
    "confidence": "str",      # "high" / "medium" / "low" — your confidence
    "notes": "str",           # any caveats
}

# ============================================================================
# SCHEMA 2: BEELS (polygons inside haors)
# ============================================================================
# Each row = one beel (permanent water core)
beel_schema = {
    "beel_id": "str",         # unique ID e.g. "TGR_B01"
    "name_en": "str",         # English name if known
    "haor_id": "str",         # parent haor ID (links to haors_manual.shp)
    "area_km2": "float",      # computed area
    "permanence": "str",      # "permanent" / "semi_permanent"
    "notes": "str",
}

# ============================================================================
# SCHEMA 3: KHALS (lines connecting water bodies)
# ============================================================================
# Each row = one khal/channel
khal_schema = {
    "khal_id": "str",         # unique ID e.g. "K001"
    "name_en": "str",         # if named
    "flow_from": "str",       # source haor/beel ID
    "flow_to": "str",         # destination haor/beel ID
    "length_km": "float",     # computed length
    "khal_type": "str",       # "primary" / "secondary" / "minor"
    "perennial": "int",       # 1 = year-round flow, 0 = seasonal
    "notes": "str",
}

# ============================================================================
# CREATE EMPTY SHAPEFILES WITH SAMPLE ROWS (so QGIS recognizes the schema)
# ============================================================================
# We need at least one feature to write a shapefile. We'll add a dummy feature
# with the AOI center point so the schema is established. You'll DELETE this
# dummy feature in QGIS when you start digitizing.

CRS = "EPSG:4326"
AOI_CENTER = (91.15, 25.05)
dummy_polygon = Polygon([
    (AOI_CENTER[0]-0.01, AOI_CENTER[1]-0.01),
    (AOI_CENTER[0]+0.01, AOI_CENTER[1]-0.01),
    (AOI_CENTER[0]+0.01, AOI_CENTER[1]+0.01),
    (AOI_CENTER[0]-0.01, AOI_CENTER[1]+0.01),
])
dummy_line = LineString([
    (AOI_CENTER[0]-0.01, AOI_CENTER[1]),
    (AOI_CENTER[0]+0.01, AOI_CENTER[1]),
])

# --- HAORS shapefile ---
haor_dummy = {
    "haor_id": "DUMMY",
    "name_en": "DELETE ME",
    "name_bn": "",
    "area_km2": 0.0,
    "ramsar": 0,
    "district": "",
    "upazila": "",
    "extent_type": "wet",
    "source": "S2_wet",
    "confidence": "low",
    "notes": "Delete this dummy row in QGIS before starting",
    "geometry": dummy_polygon,
}
gdf_haors = gpd.GeoDataFrame([haor_dummy], crs=CRS)
gdf_haors.to_file(os.path.join(OUT_DIR, "haors_manual.shp"))
print(f"✓ Created: haors_manual.shp ({len(haor_schema)} fields)")

# --- BEELS shapefile ---
beel_dummy = {
    "beel_id": "DUMMY",
    "name_en": "DELETE ME",
    "haor_id": "",
    "area_km2": 0.0,
    "permanence": "permanent",
    "notes": "Delete this dummy row in QGIS before starting",
    "geometry": dummy_polygon,
}
gdf_beels = gpd.GeoDataFrame([beel_dummy], crs=CRS)
gdf_beels.to_file(os.path.join(OUT_DIR, "beels_manual.shp"))
print(f"✓ Created: beels_manual.shp ({len(beel_schema)} fields)")

# --- KHALS shapefile ---
khal_dummy = {
    "khal_id": "DUMMY",
    "name_en": "DELETE ME",
    "flow_from": "",
    "flow_to": "",
    "length_km": 0.0,
    "khal_type": "primary",
    "perennial": 0,
    "notes": "Delete this dummy row in QGIS before starting",
    "geometry": dummy_line,
}
gdf_khals = gpd.GeoDataFrame([khal_dummy], crs=CRS)
gdf_khals.to_file(os.path.join(OUT_DIR, "khals_manual.shp"))
print(f"✓ Created: khals_manual.shp ({len(khal_schema)} fields)")

print(f"\nAll shapefiles created in: {OUT_DIR}")
print(f"Open these in QGIS and start digitizing.")