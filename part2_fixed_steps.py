#!/usr/bin/env python3

# %%
# =============================================================================
# STEP 0: IMPORTS AND GEE INITIALIZATION
# =============================================================================

import ee
import geemap
import csv
from datetime import datetime, timedelta

try:
    ee.Initialize()
    print("✓ GEE initialized successfully")
except Exception:
    try:
        geemap.ee_initialize()
        ee.Initialize()
        print("✓ GEE initialized via geemap")
    except Exception:
        ee.Authenticate()
        ee.Initialize()
        print("✓ GEE initialized after authentication")

# --- Paths to Part 1 GeoTIFF exports (update these to your local paths) ---
# These are OPTIONAL — the script works without them, but having them
# enables offline analysis in matplotlib/QGIS later.
# Download from Google Drive → 'haor_flood_analysis' folder
GEOTIFF_DIR = "/work/a06/wasif/haor_flood_analysis"  # ← UPDATE THIS
FIRST_WET_DATE_TIF = f"{GEOTIFF_DIR}/first_wet_date_2025.tif"
WET_FRACTION_TIF   = f"{GEOTIFF_DIR}/wet_fraction_2025.tif"
TOTAL_WET_COUNT_TIF = f"{GEOTIFF_DIR}/total_wet_count_2025.tif"


# =============================================================================
# STEP 1: RECREATE PART 1 CORE OBJECTS IN GEE
# =============================================================================
# Why: Even though Part 2 runs independently, we need the same GEE objects
#      (roi, flood_masks, first_wet_date) to compute per-haor statistics.
#      This step recreates them using the exact same parameters as Part 1.
#      It does NOT re-download or re-export anything — it just tells GEE
#      "here's the same computation I did before, keep it ready."

# --- Same parameters as Part 1 ---
BBOX = [89.8, 23.5, 92.8, 25.5]
roi = ee.Geometry.Rectangle(BBOX)

BASELINE_START = "2025-01-01"
BASELINE_END   = "2025-03-31"
MONITOR_START  = "2025-04-01"
MONITOR_END    = "2025-09-30"
Z_THRESHOLD    = -2.5
SLOPE_MAX      = 5

# --- Recreate S-1 collection and flood masks ---
def get_s1_collection(roi, start, end, pol="VV"):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
            .select(pol))

# Baseline stats
baseline_col = get_s1_collection(roi, BASELINE_START, BASELINE_END)
baseline_col = baseline_col.map(lambda img: img.updateMask(img.gt(-30)))
baseline_mean = baseline_col.mean().rename("baseline_mean")
baseline_std = baseline_col.reduce(ee.Reducer.stdDev()).rename("baseline_std")
baseline_std = baseline_std.where(baseline_std.lte(0), 0.5)

# Slope mask
slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
slope_mask = slope.lt(SLOPE_MAX)

# Flood masks
def compute_zscore_flood_mask(image):
    masked = image.updateMask(image.gt(-30))
    zscore = masked.subtract(baseline_mean).divide(baseline_std)
    flood = zscore.lt(Z_THRESHOLD).rename("flood_clean")
    flood = flood.updateMask(slope_mask)
    flood = flood.focal_mode(radius=60, units="meters").rename("flood_clean")
    return flood.copyProperties(image, ["system:time_start"])

monitor_col = get_s1_collection(roi, MONITOR_START, MONITOR_END)
flood_masks = monitor_col.map(compute_zscore_flood_mask)

# First-wet-date
def add_date_band(img):
    date_ms = img.date().millis()
    epoch_ms = ee.Date("2025-01-01").millis()
    days = ee.Number(date_ms).subtract(epoch_ms).divide(86400000)
    date_img = ee.Image.constant(days).float().rename("date_days")
    return date_img.updateMask(img.select("flood_clean"))

first_wet_date = flood_masks.map(add_date_band).min().rename("first_wet_date")

# Get monitoring dates
monitor_dates_ms = flood_masks.aggregate_array("system:time_start").getInfo()
monitor_dates = sorted(set([
    datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in monitor_dates_ms
]))

print("✓ Part 1 objects recreated in GEE")
print(f"  ROI: {BBOX}")
print(f"  Flood masks: {flood_masks.size().getInfo()} images")
print(f"  Monitoring dates: {len(monitor_dates)}")


"""
==============================================================================
PART 2 — FIXED STEPS (2, 4, 5)
==============================================================================
Replace the corresponding steps in your haor_flood_timeseries_part2.py
Steps 1, 3, 6, 7, 8, 9, 10 are UNCHANGED.

Changes:
  STEP 2: Complete rewrite of CoCoAH using JRC Global Surface Water +
          Sentinel-1 May composite + proper shape filtering
  STEP 4: Better barrier detection combining HAND with FABDEM local relief
  STEP 5: FABDEM now produces a "barrier likelihood" layer instead of raw elevation

==============================================================================
"""


# %%
# =============================================================================
# STEP 2 (FIXED): CoCoAH — HAOR BOUNDARY DETECTION
# =============================================================================
# PROBLEM in v1: Used reduceToVectors on raw S-1 water mask → fragmented,
#   noisy polygons scattered everywhere. Missed many haors, included noise.
#
# FIX: Use a HYBRID approach combining two data sources:
#
#   A) JRC Global Surface Water (Pekel et al. 2016)
#      - Pre-computed water occurrence from 1984-2021
#      - "occurrence" band: % of time each pixel was water (0-100)
#      - Pixels with occurrence 10-90% are SEASONAL water → likely haors
#      - Pixels with occurrence >90% are PERMANENT water → rivers, beels
#      - This gives us a clean, noise-free baseline of where haors typically form
#
#   B) Sentinel-1 May 2025 composite
#      - Shows the CURRENT state of haors in the pre-monsoon season
#      - Confirms which JRC-identified haors actually have water in 2025
#
#   By combining both, we get much cleaner boundaries than either alone:
#   - JRC removes SAR speckle noise (uses 36 years of Landsat data)
#   - S-1 confirms current-year water presence (JRC stops at 2021)
#
# SHAPE FILTERING: We implement Ahmad et al.'s eccentricity filter to
#   remove rivers (long thin features) from the haor candidates.

print("\n--- CoCoAH Haor Boundary Detection (Improved) ---")

# --- Source A: JRC Global Surface Water ---
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")

# "occurrence" = percentage of time water was present (0-100)
# "seasonality" = number of months water is present per year
water_occurrence = jrc.select("occurrence").clip(roi)
water_seasonality = jrc.select("seasonality").clip(roi)

# Seasonal water bodies: present 10-90% of the time → haor candidates
# This excludes permanent rivers (>90%) and rare puddles (<10%)
seasonal_water = water_occurrence.gte(10).And(water_occurrence.lte(90))

# Also include areas with clear seasonal pattern (present 4-10 months/year)
# Haors are typically inundated 6-8 months, dry 4-6 months
seasonal_by_months = water_seasonality.gte(4).And(water_seasonality.lte(10))

# Combine: pixel is a haor candidate if EITHER criterion is met
haor_candidate_jrc = seasonal_water.Or(seasonal_by_months).rename("haor_candidate")

print(f"  JRC seasonal water pixels identified")

# --- Source B: Sentinel-1 May 2025 composite ---
cocoah_col = get_s1_collection(roi, "2025-05-01", "2025-05-31")
cocoah_col = cocoah_col.map(lambda img: img.updateMask(img.gt(-30)))
cocoah_composite = cocoah_col.median().clip(roi)
n_cocoah = cocoah_col.size().getInfo()
print(f"  S-1 May 2025 images: {n_cocoah}")

# Water classification from S-1 (same -13dB threshold as Ahmad et al.)
s1_water_may = cocoah_composite.lt(-13).rename("s1_water")

# --- Combine: JRC haor candidate AND currently has water in May 2025 ---
# This is more conservative (fewer false positives) than either alone
haor_combined = haor_candidate_jrc.And(s1_water_may).selfMask()

# --- Morphological cleaning ---
# Use PROPER opening (erosion then dilation) to:
#   1. Remove thin connecting channels between haors (breaks river connections)
#   2. Remove small noise patches
#   3. Smooth haor boundaries
# Radius of 200m is appropriate: smaller than haors, larger than channels
eroded = haor_combined.focal_min(
    radius=200, units="meters", kernelType="circle"
)
dilated = eroded.focal_max(
    radius=200, units="meters", kernelType="circle"
)

# Fill internal holes (closing operation: dilation then erosion)
# This fills small dry patches inside haors (e.g., islands, missed pixels)
closed = dilated.focal_max(
    radius=150, units="meters", kernelType="circle"
).focal_min(
    radius=150, units="meters", kernelType="circle"
)

# --- Convert to vectors ---
COCOAH_SCALE = 100  # 100m for vectorization

print("  Vectorizing haor polygons...")
water_for_vector = closed.rename("water").selfMask()

haor_vectors_raw = water_for_vector.reduceToVectors(
    reducer=ee.Reducer.countEvery(),
    geometry=roi,
    scale=COCOAH_SCALE,
    maxPixels=1e12,
    geometryType="polygon",
    eightConnected=True,
    bestEffort=True
)

n_raw = haor_vectors_raw.size().getInfo()
print(f"  Raw polygons: {n_raw}")

# --- Size and shape filtering ---
# At 100m resolution, 1 pixel = 0.01 km²
# Minimum haor: 1 km² = 100 pixels (exclude small puddles)
# Maximum haor: 500 km² = 50000 pixels
MIN_HAOR_PIXELS = 100   # 1 km²
MAX_HAOR_PIXELS = 50000  # 500 km²

haor_vectors_sized = haor_vectors_raw.filter(
    ee.Filter.And(
        ee.Filter.gte("count", MIN_HAOR_PIXELS),
        ee.Filter.lte("count", MAX_HAOR_PIXELS)
    )
)

# --- Shape filtering: remove rivers using perimeter/area ratio ---
# Rivers are long and thin → high perimeter relative to area
# Haors are compact → lower perimeter relative to area
# We compute a "compactness" score: 4π × area / perimeter²
# Perfect circle = 1.0, long thin river ≈ 0.01-0.1
# Haors typically > 0.15, rivers typically < 0.1

def add_compactness(feature):
    """Compute compactness score for shape filtering."""
    area = feature.geometry().area(100)  # area in m²
    perimeter = feature.geometry().perimeter(100)  # perimeter in m
    # Compactness = 4π × area / perimeter²
    # Protect against zero perimeter
    compactness = ee.Number(4).multiply(3.14159).multiply(area).divide(
        perimeter.multiply(perimeter).max(1)
    )
    return feature.set("compactness", compactness)

haor_vectors_with_shape = haor_vectors_sized.map(add_compactness)

# Filter: keep only compact features (compactness > 0.08)
# This removes rivers and narrow channels while keeping haors
# 0.08 is lenient — adjust upward (e.g., 0.12) if rivers still sneak through
COMPACTNESS_THRESHOLD = 0.08

haor_vectors = haor_vectors_with_shape.filter(
    ee.Filter.gte("compactness", COMPACTNESS_THRESHOLD)
)

n_haors = haor_vectors.size().getInfo()
print(f"✓ CoCoAH detected {n_haors} haors (after size + shape filtering)")

# Get size distribution
haor_areas = haor_vectors.aggregate_array("count").getInfo()
if haor_areas:
    pixel_area_km2 = (COCOAH_SCALE / 1000.0) ** 2
    haor_areas_km2 = [a * pixel_area_km2 for a in haor_areas]
    print(f"  Size range: {min(haor_areas_km2):.1f} - {max(haor_areas_km2):.1f} km²")
    print(f"  Median: {sorted(haor_areas_km2)[len(haor_areas_km2)//2]:.1f} km²")
    print(f"  Total area: {sum(haor_areas_km2):.1f} km²")
    small = len([a for a in haor_areas_km2 if a < 5])
    medium = len([a for a in haor_areas_km2 if 5 <= a < 50])
    large = len([a for a in haor_areas_km2 if a >= 50])
    print(f"  Small (<5 km²): {small}, Medium (5-50): {medium}, Large (>50): {large}")

# Print compactness stats for inspection
compactness_vals = haor_vectors.aggregate_array("compactness").getInfo()
if compactness_vals:
    print(f"  Compactness range: {min(compactness_vals):.3f} - {max(compactness_vals):.3f}")


# %%
# =============================================================================
# STEP 4 (FIXED): MERIT HYDRO HAND + IMPROVED BARRIER DETECTION
# =============================================================================
# PROBLEM in v1: Simple HAND threshold of 2-8m captured almost everything
#   in this flat region → magenta dots everywhere, no useful information.
#
# FIX: Use a RELATIVE approach instead of absolute thresholds.
#   Instead of "HAND > 2m = barrier", we identify barriers as:
#   1. Pixels that are LOCALLY ELEVATED compared to their neighborhood
#   2. AND form LINEAR features (not random elevated patches)
#   3. AND are located BETWEEN haor polygons (not inside haors or on uplands)
#
# We combine MERIT Hydro HAND with FABDEM local relief for this.

print("\n--- MERIT Hydro HAND + Barrier Detection (Improved) ---")

merit_hydro = ee.Image("MERIT/Hydro/v1_0_1")
hand = merit_hydro.select("hnd").clip(roi)
flow_dir = merit_hydro.select("dir").clip(roi)
elev = merit_hydro.select("elv").clip(roi)
upa = merit_hydro.select("upa").clip(roi)

# HAND classification with TIGHTER thresholds for haor region
# In this extremely flat terrain:
#   HAND < 1m  → definitely in drainage/haor (very low-lying)
#   HAND 1-3m  → transitional (could be haor margin or low embankment)
#   HAND 3-6m  → likely elevated feature (embankment, road, settlement)
#   HAND > 6m  → definitely upland or hill
hand_classes = (ee.Image(0)
    .where(hand.lt(1), 1)          # Very low-lying (haor interior)
    .where(hand.gte(1).And(hand.lt(3)), 2)   # Transitional
    .where(hand.gte(3).And(hand.lt(6)), 3)   # Likely barrier
    .where(hand.gte(6), 4)         # Upland
    .rename("hand_class")
    .selfMask())

# River network from upstream area
river_network = upa.gt(100).rename("river_network")

# --- HAND-based barrier detection (more refined) ---
# Barriers should be:
# 1. Moderately elevated (HAND 1.5-6m — not too high, not too low)
# 2. Adjacent to low-lying areas on BOTH sides
# 3. Not part of the river network

# Step 1: Identify moderately elevated pixels
moderate_elevation = hand.gte(1.5).And(hand.lt(6))

# Step 2: Check if they're adjacent to low-lying areas
# A barrier should have low HAND values nearby (within ~500m)
nearby_min_hand = hand.focal_min(radius=500, units="meters")
near_low_areas = nearby_min_hand.lt(1.5)  # There's a low area within 500m

# Step 3: Exclude river network pixels
not_river = upa.lt(50)  # Not a major channel

# Combine: barrier candidate = elevated AND near low areas AND not river
hand_barriers = moderate_elevation.And(near_low_areas).And(not_river).rename("hand_barrier")

print("✓ MERIT Hydro layers loaded with refined barrier detection")
hand_stats = hand.reduceRegion(
    reducer=ee.Reducer.percentile([10, 25, 50, 75, 90]),
    geometry=roi, scale=90, maxPixels=1e10
).getInfo()
print(f"  HAND percentiles: {hand_stats}")


# %%
# =============================================================================
# STEP 5 (FIXED): FABDEM 30m — BARRIER LIKELIHOOD LAYER
# =============================================================================
# PROBLEM in v1: Just showed raw FABDEM elevation, which is useless for
#   analysis — everything in the haor region is 2-15m elevation.
#
# FIX: Compute LOCAL RELIEF from FABDEM.
#   Local relief = how much higher is this pixel than its neighborhood?
#   Embankments and roads appear as narrow ridges: the pixel is 1-3m
#   HIGHER than pixels 200-500m away in all directions.
#
#   We compute this at two scales:
#   - Fine scale (200m radius): detects narrow embankments and roads
#   - Medium scale (500m radius): detects broader elevated features
#
#   Then combine with HAND-based barriers for a "barrier likelihood" score.

print("\n--- FABDEM 30m — Barrier Likelihood Layer ---")

fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
fabdem_mosaic = fabdem.filterBounds(roi).mosaic().clip(roi).rename("fabdem_elev")

# --- Local relief at fine scale (200m) ---
# This detects narrow features like embankments and roads
# The embankment is higher than the surrounding paddy/haor by 1-3m
focal_mean_200 = fabdem_mosaic.focal_mean(radius=200, units="meters")
local_relief_200 = fabdem_mosaic.subtract(focal_mean_200).rename("local_relief_200m")

# --- Local relief at medium scale (500m) ---
# This detects broader elevated features and settlement platforms
focal_mean_500 = fabdem_mosaic.focal_mean(radius=500, units="meters")
local_relief_500 = fabdem_mosaic.subtract(focal_mean_500).rename("local_relief_500m")

# --- Barrier detection from FABDEM ---
# A pixel is a potential barrier if:
# 1. It's at least 0.5m higher than its 200m neighborhood (fine scale)
# 2. OR at least 0.8m higher than its 500m neighborhood (medium scale)
# These thresholds are conservative for flat haor terrain where
# embankments are typically 1-4m above surrounding fields
fabdem_barrier_fine = local_relief_200.gt(0.5)
fabdem_barrier_medium = local_relief_500.gt(0.8)
fabdem_barrier = fabdem_barrier_fine.Or(fabdem_barrier_medium).rename("fabdem_barrier")

# --- Combined barrier likelihood ---
# Combine HAND-based and FABDEM-based barrier detection
# A pixel is a "high confidence barrier" if BOTH methods flag it
# A pixel is a "moderate confidence barrier" if EITHER method flags it
barrier_both = hand_barriers.And(fabdem_barrier).rename("barrier_high_conf")
barrier_either = hand_barriers.Or(fabdem_barrier).rename("barrier_any_conf")

# --- Linear feature enhancement ---
# Embankments and roads are LINEAR features. We can enhance them by
# checking if the elevated pixels form connected lines rather than
# isolated blobs. Use focal operations to detect linearity.
# A simple proxy: if a barrier pixel has barrier neighbors in a narrow
# band (not a wide blob), it's more likely a linear feature.
# We use: count barrier pixels in a small window vs. medium window
# Linear features: high density in small window, lower in medium
barrier_count_small = barrier_either.focal_sum(radius=100, units="meters")
barrier_count_large = barrier_either.focal_sum(radius=300, units="meters")

# Ratio > 0.3 means the barrier pixels are concentrated (linear-ish)
# Ratio < 0.1 means they're spread out (probably a settlement or upland)
# This is a simple heuristic — not perfect but helps
linearity_ratio = barrier_count_small.divide(barrier_count_large.max(1))
linear_barriers = barrier_either.And(linearity_ratio.gt(0.15)).rename("linear_barriers")

print("✓ FABDEM barrier likelihood computed")
print("  Layers created:")
print("    - local_relief_200m: fine-scale elevation anomaly")
print("    - local_relief_500m: medium-scale elevation anomaly")
print("    - fabdem_barrier: FABDEM-only barrier detection")
print("    - barrier_high_conf: HAND AND FABDEM agree = barrier")
print("    - barrier_any_conf: HAND OR FABDEM = potential barrier")
print("    - linear_barriers: barriers that form linear features")

# Quick stats on barrier coverage
barrier_stats = barrier_both.selfMask().reduceRegion(
    reducer=ee.Reducer.count(),
    geometry=roi, scale=100, maxPixels=1e10
).getInfo()
print(f"  High-confidence barrier pixels: {barrier_stats}")


# %%
# =============================================================================
# STEP 8 (UPDATED): COMBINED ANALYSIS MAP
# =============================================================================
# Updated to use the new improved layers from Steps 2, 4, 5.

print("\n--- Creating Combined Analysis Map (Improved) ---")
m = geemap.Map()
m.centerObject(roi, 8)

# 1. First-wet-date (from Part 1)
first_wet_vis = {
    "min": 90, "max": 272,
    "palette": [
        "023858", "045a8d", "0570b0", "3690c0",
        "74a9cf", "66c2a4", "41ae76", "238b45",
        "006d2c", "31a354", "78c679", "addd8e",
        "d9f0a3", "f7fcb1", "ffeda0", "fed976",
        "feb24c", "fd8d3c", "fc4e2a", "e31a1c",
        "d7301f", "bd0026", "a50026", "800026",
    ]
}
m.addLayer(first_wet_date.clip(roi), first_wet_vis,
           "1. First Wet Date", True)

# 2. MERIT Hydro HAND
hand_vis = {
    "min": 0, "max": 10,
    "palette": ["0D0887", "5B02A3", "9A179B", "CB4678",
                "EB7852", "FBB32F", "F0F921"]
}
m.addLayer(hand.clip(roi), hand_vis, "2. HAND (m)", False)

# 3. HAND classification (refined)
hand_class_vis = {
    "min": 1, "max": 4,
    "palette": ["2166AC", "92C5DE", "F4A582", "B2182B"]
}
m.addLayer(hand_classes.clip(roi), hand_class_vis,
           "3. HAND classes (blue=low → red=high)", False)

# 4. FABDEM local relief (fine scale) — NEW
relief_vis = {
    "min": -1, "max": 2,
    "palette": ["0571B0", "92C5DE", "F7F7F7", "F4A582", "CA0020"]
}
m.addLayer(local_relief_200.clip(roi), relief_vis,
           "4. FABDEM local relief 200m (m)", False)

# 5. High-confidence barriers (HAND + FABDEM agree) — NEW
m.addLayer(barrier_both.selfMask().clip(roi),
           {"palette": ["FF00FF"]},
           "5. Barriers: HIGH confidence (HAND+FABDEM)", False)

# 6. Linear barriers — NEW
m.addLayer(linear_barriers.selfMask().clip(roi),
           {"palette": ["FFFF00"]},
           "6. Barriers: LINEAR features", False)

# 7. River network
m.addLayer(river_network.selfMask().clip(roi),
           {"palette": ["00BFFF"]},
           "7. River network (upa>100km²)", False)

# 8. CoCoAH haor boundaries — IMPROVED
haor_outline = ee.Image().paint(haor_vectors, 1, 2)
m.addLayer(haor_outline, {"palette": ["00FF00"]},
           "8. CoCoAH haor boundaries", True)

# 9. JRC water occurrence (reference)
jrc_vis = {"min": 0, "max": 100, "palette": ["FFFFFF", "0000FF"]}
m.addLayer(water_occurrence, jrc_vis,
           "9. JRC water occurrence (%)", False)

# 10. FABDEM raw elevation (for reference only)
fabdem_vis = {
    "min": 0, "max": 30,
    "palette": ["000004", "1B0C41", "4A0C6B", "781C6D",
                "A52C60", "CF4446", "ED6925", "FB9B06",
                "F7D03C", "FCFFA4"]
}
m.addLayer(fabdem_mosaic.clip(roi), fabdem_vis,
           "10. FABDEM elevation (m, reference)", False)

# Transect lines
transects = [
    {"name": "Transect 1: Sunamganj N-S", "start": [91.0, 25.1], "end": [91.0, 24.7]},
    {"name": "Transect 2: Tanguar-Dekhar", "start": [91.1, 25.15], "end": [91.5, 25.0]},
    {"name": "Transect 3: E-W central", "start": [90.5, 24.8], "end": [91.8, 24.8]},
]
for t in transects:
    line = ee.Geometry.LineString([t["start"], t["end"]])
    m.addLayer(ee.Image().paint(ee.FeatureCollection([ee.Feature(line)]), 1, 3),
               {"palette": ["FFFFFF"]}, f"T: {t['name']}", False)

# Study area
m.addLayer(ee.Image().paint(roi, 1, 2), {"palette": ["FFFFFF"]}, "Study Area", True)

print("✓ Combined analysis map created — run 'm' to display")
print("""
Layer guide:
  1. First Wet Date — flood propagation pattern (blue=early, red=late)
  2. HAND — height above nearest drainage (meters)
  3. HAND classes — 4 levels from very low to upland
  4. FABDEM local relief — how much higher than 200m neighborhood
     (red = elevated features like embankments, blue = depressions)
  5. HIGH confidence barriers — where BOTH HAND and FABDEM agree
     (most reliable barrier locations — magenta)
  6. LINEAR barriers — elevated features that form lines/ridges
     (likely roads or embankments — yellow)
  7. River network — major channels
  8. CoCoAH haor boundaries — improved using JRC+S-1 (green)
  9. JRC water occurrence — historical water frequency (reference)
  10. FABDEM elevation — raw terrain height (reference only)

Key analysis:
  - Toggle layers 1 + 8 together: do haor boundaries align with
    first-wet-date patterns?
  - Toggle layers 1 + 5 or 6: do barriers align with sharp changes
    in flood timing?
  - Toggle layers 4 + 8: do local relief ridges correspond to
    haor boundaries?
""")

m
