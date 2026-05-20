#!/usr/bin/env python3
"""
==============================================================================
HAOR FLOOD PROPAGATION ANALYSIS — PART 2: Boundaries, Barriers & Integration
==============================================================================

This script builds on Part 1 results to:
  1. Detect individual haor boundaries using CoCoAH (Ahmad et al. 2020)
  2. Compute per-haor flood area time series
  3. Overlay MERIT Hydro HAND to identify low vs. elevated terrain
  4. Overlay FABDEM (30m) for barrier crest elevation estimation
  5. Overlay OSM road network as potential barrier locations
  6. Select corridors for future SWOT WSE transect analysis
  7. Produce combined analysis maps

CONNECTION TO PART 1:
  This script loads the GeoTIFF exports from Part 1 (first-wet-date map,
  wet fraction, total wet count). You need to have exported these from
  Part 1 Step 10 before running this script.

  HOWEVER, this script also re-creates the flood_masks collection from
  GEE (same parameters as Part 1) so it can compute per-haor area
  time series. The GeoTIFFs are used for offline analysis and visualization.

DATA SOURCES (all from GEE — NO manual downloads needed):
  - Sentinel-1 GRD: flood mask computation (same as Part 1)
  - MERIT Hydro: ee.Image("MERIT/Hydro/v1_0_1") → HAND, flow direction
  - FABDEM: ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
  - OSM roads: derived from road features available in GEE

FILES FROM PART 1 (exported GeoTIFFs from Google Drive):
  You need to download these from Google Drive and place them in a folder.
  Update the paths below to point to your local copies.

  File 1: first_wet_date_2025.tif  — First-wet-date map (days since Jan 1)
  File 2: wet_fraction_2025.tif    — Fraction of dates each pixel was wet
  File 3: total_wet_count_2025.tif — Number of dates each pixel was wet

  If you haven't exported these yet, this script will still work — it just
  won't have the offline GeoTIFF analysis sections. All GEE-based analysis
  runs independently.

Author: Wasif Shahriar (M1, Yamazaki Lab, UTokyo)
==============================================================================
"""

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
GEOTIFF_DIR = "/path/to/your/geotiff/folder"  # ← UPDATE THIS
FIRST_WET_DATE_TIF = f"{GEOTIFF_DIR}/first_wet_date_2025.tif"
WET_FRACTION_TIF   = f"{GEOTIFF_DIR}/wet_fraction_2025.tif"
TOTAL_WET_COUNT_TIF = f"{GEOTIFF_DIR}/total_wet_count_2025.tif"


# %%
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


# %%
# =============================================================================
# STEP 2: CoCoAH — DETECT INDIVIDUAL HAOR BOUNDARIES
# =============================================================================
# Why: To compute per-haor statistics (area time series, first-wet-date per haor),
#      we need to know where each individual haor is. CoCoAH (Connected Components
#      Analysis for Haors, Ahmad et al. 2020) does this automatically.
#
# How it works:
#   1. Take a pre-monsoon Sentinel-1 image (May 2025) when haors are forming
#      but haven't merged yet
#   2. Classify water using backscatter threshold (<-13 dB)
#   3. Clean up with morphological operations (erosion + dilation)
#   4. Find connected components (groups of adjacent water pixels)
#   5. Filter out rivers/channels using shape properties (circularity, eccentricity)
#   6. Result: individual haor polygons with IDs
#
# IMPORTANT: We use MAY 2025 because:
#   - Earlier (Jan-Apr): haors too small or dry → can't detect them
#   - Later (Jul-Sep): haors have MERGED into one big water body → can't
#     distinguish individual haors
#   - May is the sweet spot: haors are forming but still separate
#
# We implement this in GEE using server-side connected component labeling.

print("\n--- CoCoAH Haor Boundary Detection ---")

# Step 2a: Get pre-monsoon Sentinel-1 composite (May 2025)
# Using a full-month composite to get complete spatial coverage
cocoah_col = get_s1_collection(roi, "2025-05-01", "2025-05-31")
cocoah_col = cocoah_col.map(lambda img: img.updateMask(img.gt(-30)))
cocoah_composite = cocoah_col.median().clip(roi)

print(f"  Pre-monsoon S-1 images (May 2025): {cocoah_col.size().getInfo()}")

# Step 2b: Classify water using -13 dB threshold (Ahmad et al. 2019, 2020)
# Standing water has backscatter between -24.3 and -12.6 dB
water_threshold = -13  # dB
water_binary = cocoah_composite.lt(water_threshold).rename("water").selfMask()

# Step 2c: Morphological operations (erosion + dilation = "opening")
# Erosion removes small noise blobs; dilation fills small holes
# Ahmad et al. used a disk kernel with radius 10 pixels (~100m at 10m resolution)
# In GEE, we use focal operations:
#   - focal_min = erosion (shrinks water bodies)
#   - focal_max = dilation (expands them back)
# The net effect: noise removed, real features preserved
eroded = water_binary.focal_min(radius=100, units="meters", kernelType="circle")
opened = eroded.focal_max(radius=100, units="meters", kernelType="circle")

# Step 2d: Connected component labeling
# GEE's connectedComponents requires a "seed" image and labels each
# connected group of pixels with a unique ID
# maxSize limits computation — 1024 pixels means haors up to ~1 km² at 100m
# We use a coarser scale (100m) for the labeling to handle the large AOI
connected = opened.selfMask().connectedComponents(
    connectedness=ee.Kernel.plus(1),  # 4-connected (simpler, faster)
    maxSize=4096  # Max pixels per component — increase if needed
)

# The result has a 'labels' band where each connected component has a unique ID
haor_labels = connected.select("labels")

# Step 2e: Filter out non-haor water bodies
# Ahmad et al. used circularity, eccentricity, and area thresholds.
# In GEE, we can use connectedPixelCount and reduceConnectedComponents
# to compute per-component properties.
#
# However, full CoCoAH filtering (circularity, eccentricity) is complex in GEE.
# For a practical first pass, we filter by SIZE only:
#   - Remove components smaller than 0.5 km² (noise/puddles)
#   - Remove components larger than 5000 km² (merged mega-water bodies)
#   - This catches most haors while excluding rivers (which are long and thin
#     and typically don't form large connected blobs at -13dB threshold)

# Count pixels per component
pixel_count = haor_labels.connectedPixelCount(maxSize=4096)

# At 100m resolution, 1 pixel = 0.01 km². So:
#   0.5 km² = 50 pixels minimum
#   5000 km² = 500000 pixels maximum (unlikely at this scale)
MIN_HAOR_PIXELS = 50    # ~0.5 km² at 100m
MAX_HAOR_PIXELS = 50000  # ~500 km² at 100m

# Create size mask
size_mask = pixel_count.gte(MIN_HAOR_PIXELS).And(pixel_count.lte(MAX_HAOR_PIXELS))

# Apply size filter to labels
haor_labels_filtered = haor_labels.updateMask(size_mask)

# Convert to vector (polygons) for per-haor analysis
# Note: vectorizing the entire region at high resolution is expensive.
# We do it at 100m resolution and limit the number of features.
print("  Converting haor labels to polygons (this may take a moment)...")

haor_vectors = haor_labels_filtered.reduceToVectors(
    reducer=ee.Reducer.countEvery(),
    geometry=roi,
    scale=100,  # 100m resolution for vectorization
    maxPixels=1e12,
    geometryType="polygon",
    eightConnected=True,
    bestEffort=True
)

n_haors = haor_vectors.size().getInfo()
print(f"✓ CoCoAH detected {n_haors} individual haors")

# Get size distribution
haor_areas = haor_vectors.aggregate_array("count").getInfo()
if haor_areas:
    haor_areas_km2 = [a * 0.01 for a in haor_areas]  # pixels to km² at 100m
    print(f"  Size range: {min(haor_areas_km2):.1f} - {max(haor_areas_km2):.1f} km²")
    print(f"  Median size: {sorted(haor_areas_km2)[len(haor_areas_km2)//2]:.1f} km²")
    print(f"  Total haor area: {sum(haor_areas_km2):.1f} km²")


# %%
# =============================================================================
# STEP 3: PER-HAOR FLOOD AREA TIME SERIES
# =============================================================================
# Why: The total area time series from Part 1 shows the overall flood hydrograph.
#      But to understand CONNECTIVITY between haors, you need to see WHEN each
#      individual haor floods. If Haor A peaks on May 20 and Haor B peaks on
#      June 5, that 16-day lag is evidence of propagation (or isolation).
#
# How: For each haor polygon from CoCoAH, compute the flooded area on each
#      monitoring date by masking the flood mask to that haor's boundary
#      and summing the wet pixels.
#
# This is computationally expensive for 700+ haors × 30+ dates.
# We do it for a SUBSET of the largest/most interesting haors first.

print("\n--- Per-Haor Flood Area Time Series ---")

# Select the top N haors by size for detailed analysis
# (Computing for ALL 700+ haors would take too long interactively)
TOP_N_HAORS = 20

# Sort haors by size (largest first) and take top N
haor_list = haor_vectors.toList(haor_vectors.size())

# We'll compute area time series for each selected haor
# First, let's get the geometries and IDs
print(f"  Computing area time series for top {TOP_N_HAORS} haors by size...")

# Create 6-day composite dates (same as Part 1 v2)
COMPOSITE_WINDOW_DAYS = 6
def generate_composite_dates(start_date, end_date, interval_days):
    dates = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current < end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=interval_days)
    return dates

composite_dates = generate_composite_dates(MONITOR_START, MONITOR_END, COMPOSITE_WINDOW_DAYS)

# For each of the top N haors, compute flooded area per composite date
per_haor_results = []

# Sort by area (count) descending
sorted_indices = sorted(range(len(haor_areas)), key=lambda i: haor_areas[i], reverse=True)

for rank, idx in enumerate(sorted_indices[:TOP_N_HAORS]):
    haor_feature = ee.Feature(haor_list.get(idx))
    haor_geom = haor_feature.geometry()
    haor_area_km2 = haor_areas[idx] * 0.01  # approximate total area in km²

    print(f"  Haor {rank+1}/{TOP_N_HAORS} (area ~{haor_area_km2:.1f} km²)...", end="", flush=True)

    haor_timeseries = []

    for date_str in composite_dates:
        end_dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=COMPOSITE_WINDOW_DAYS)
        end_str = end_dt.strftime("%Y-%m-%d")

        # Get flood masks in this window
        window_masks = flood_masks.filterDate(date_str, end_str)
        n_imgs = window_masks.size().getInfo()

        if n_imgs > 0:
            # Union of flood detections in window
            composite = window_masks.select("flood_clean").max()
            # Compute flooded area within this haor
            flood_area = composite.multiply(ee.Image.pixelArea()).reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=haor_geom,
                scale=100,
                maxPixels=1e10
            ).get("flood_clean")

            area_val = ee.Number(flood_area).divide(1e6).getInfo()  # m² → km²
            haor_timeseries.append({
                "date": date_str,
                "flood_area_km2": round(area_val, 2) if area_val else 0
            })
        else:
            haor_timeseries.append({"date": date_str, "flood_area_km2": 0})

    per_haor_results.append({
        "haor_rank": rank + 1,
        "total_area_km2": haor_area_km2,
        "timeseries": haor_timeseries
    })
    print(" done")

print(f"\n✓ Per-haor area time series computed for {len(per_haor_results)} haors")

# Save to CSV
csv_file = "per_haor_area_timeseries_2025.csv"
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    header = ["haor_rank", "total_area_km2"] + composite_dates
    writer.writerow(header)
    for h in per_haor_results:
        row = [h["haor_rank"], h["total_area_km2"]]
        ts_dict = {t["date"]: t["flood_area_km2"] for t in h["timeseries"]}
        row.extend([ts_dict.get(d, 0) for d in composite_dates])
        writer.writerow(row)
print(f"✓ Saved to {csv_file}")


# %%
# =============================================================================
# STEP 4: MERIT HYDRO HAND OVERLAY
# =============================================================================
# Why: HAND (Height Above Nearest Drainage) tells you which pixels are in
#      low-lying haor interiors (HAND = 0-2m) vs. on elevated features like
#      embankments and roads (HAND = 3-5m+). This helps identify WHERE the
#      barriers are between haors.
#
# MERIT Hydro is from YOUR LAB (Yamazaki et al. 2019, WRR).
# It's available on GEE as ee.Image("MERIT/Hydro/v1_0_1").
#
# Layers we use:
#   - "hnd": HAND — height above nearest drainage (meters)
#   - "dir": Flow direction (D8 encoding)
#   - "elv": Adjusted elevation (meters above EGM96 geoid)
#   - "upa": Upstream drainage area (km²)
#
# At 90m resolution, MERIT Hydro can't resolve individual embankments (3-5m wide),
# but HAND effectively separates haor interiors from elevated ground.

print("\n--- MERIT Hydro HAND Overlay ---")

merit_hydro = ee.Image("MERIT/Hydro/v1_0_1")

# Extract layers for our AOI
hand = merit_hydro.select("hnd").clip(roi)  # Height Above Nearest Drainage
flow_dir = merit_hydro.select("dir").clip(roi)  # Flow direction
elev = merit_hydro.select("elv").clip(roi)  # Adjusted elevation
upa = merit_hydro.select("upa").clip(roi)  # Upstream area

# Classify terrain using HAND
# HAND < 2m → low-lying (haor interior, river floodplain)
# HAND 2-5m → transitional (potential embankment/road zone)
# HAND > 5m → elevated (upland, settlement)
hand_classes = (hand.where(hand.lt(2), 1)    # Low-lying
                    .where(hand.gte(2).And(hand.lt(5)), 2)   # Transitional
                    .where(hand.gte(5), 3)    # Elevated
                    .rename("hand_class"))

# Identify potential barrier zones: HAND 2-5m between two low-lying areas
# These are the locations where embankments/roads likely sit
potential_barriers = hand.gte(2).And(hand.lt(8)).rename("potential_barrier")

# River network from upstream area
# Pixels with upa > 100 km² are significant river channels
river_network = upa.gt(100).rename("river_network")

print("✓ MERIT Hydro layers loaded")
print("  HAND range in AOI: checking...")
hand_stats = hand.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=roi, scale=90, maxPixels=1e10
).getInfo()
print(f"  HAND min: {hand_stats.get('hnd_min', 'N/A')}, max: {hand_stats.get('hnd_max', 'N/A')}")


# %%
# =============================================================================
# STEP 5: FABDEM 30m OVERLAY
# =============================================================================
# Why: FABDEM has 3x higher resolution than MERIT Hydro (30m vs 90m).
#      This gives a better chance of detecting narrow embankments and roads
#      that act as barriers between haors. We use it for:
#      1. Elevation transects across barriers (Step 7)
#      2. Barrier crest elevation estimation
#      3. Comparison with SWOT WSE (future work)
#
# FABDEM is available in GEE as a community dataset.
# No download needed!

print("\n--- FABDEM 30m Overlay ---")

fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")

# Mosaic all tiles covering our AOI into a single image
fabdem_mosaic = fabdem.filterBounds(roi).mosaic().clip(roi).rename("fabdem_elev")

print("✓ FABDEM loaded from GEE community catalog")

fabdem_stats = fabdem_mosaic.reduceRegion(
    reducer=ee.Reducer.minMax(),
    geometry=roi, scale=30, maxPixels=1e10
).getInfo()
print(f"  Elevation range: {fabdem_stats}")


# %%
# =============================================================================
# STEP 6: OSM ROAD NETWORK AS BARRIER PROXY
# =============================================================================
# Why: In the haor region, major roads and raised embankments act as physical
#      barriers that block or delay flood water. Roads in haor areas are
#      typically built on raised earth platforms (1-4m above the surrounding
#      paddy fields) and function as de facto levees.
#
# GEE doesn't have a dedicated "embankment" layer from OSM, but we can
# use a global road dataset. We use the GRIP (Global Roads Inventory Project)
# dataset, which is available on GEE and includes road classifications.
#
# For haor areas, even minor roads can act as significant barriers because
# the terrain is so flat — a 1-meter raised road bed is enough to block
# shallow flood water.

print("\n--- Road Network (Barrier Proxy) ---")

# Option 1: Use GRIP global roads dataset (available on GEE)
# This has good coverage of Bangladesh including rural roads
try:
    grip_roads = ee.FeatureCollection("projects/sat-io/open-datasets/GRIP4/GRIP4-region5")
    roads_in_roi = grip_roads.filterBounds(roi)
    n_roads = roads_in_roi.size().getInfo()
    print(f"✓ GRIP roads loaded: {n_roads} road segments in AOI")
    has_roads = True
except Exception as e:
    print(f"  GRIP roads not available: {e}")
    print("  Trying alternative road source...")

    # Option 2: Use a simplified approach — rasterize roads from
    # the FABDEM itself by detecting linear elevated features
    # (Roads appear as narrow ridges in the DEM)
    has_roads = False

if not has_roads:
    # Fallback: detect linear elevated features from FABDEM
    # Roads in flat haor terrain create detectable ridges
    print("  Using FABDEM-derived elevated linear features as barrier proxy")

    # Simple edge detection: where FABDEM elevation is locally higher
    # than its neighbors, there might be a road/embankment
    focal_mean = fabdem_mosaic.focal_mean(radius=150, units="meters")
    local_relief = fabdem_mosaic.subtract(focal_mean).rename("local_relief")

    # Pixels where local relief > 0.5m are potentially raised features
    raised_features = local_relief.gt(0.5).rename("raised_features")
    print("✓ Elevated linear features detected from FABDEM")


# %%
# =============================================================================
# STEP 7: ELEVATION TRANSECTS ACROSS BARRIERS
# =============================================================================
# Why: To validate the levee-overtopping mechanism, you need to show that
#      there IS an elevation barrier between two haors. An elevation transect
#      is a cross-section showing ground elevation along a straight line
#      from Haor A across the barrier into Haor B.
#
# The transect should show: low (haor A) → high (barrier) → low (haor B)
#
# We define 2-3 sample transects across known barrier locations.
# You can adjust these coordinates after inspecting the first-wet-date map
# to target areas where you see abrupt changes in flood timing.

print("\n--- Elevation Transects ---")

# Define sample transect lines
# Each transect is defined by a start point and end point [lon, lat]
# These are placed across areas where roads/embankments separate haors
# ADJUST THESE after inspecting your first-wet-date map!
transects = [
    {
        "name": "Transect 1: Sunamganj N-S",
        "start": [91.0, 25.1],  # North of Sunamganj
        "end": [91.0, 24.7],    # South toward haor interior
        "description": "Crosses multiple haors and roads N-S through Sunamganj"
    },
    {
        "name": "Transect 2: Tanguar-Dekhar corridor",
        "start": [91.1, 25.15],  # Near Tanguar haor
        "end": [91.5, 25.0],     # Toward Dekhar haor
        "description": "Connects two well-known haors with embankments between"
    },
    {
        "name": "Transect 3: E-W across central haor belt",
        "start": [90.5, 24.8],   # Western haor region
        "end": [91.8, 24.8],     # Eastern haor region
        "description": "Long E-W transect crossing many haors and barriers"
    },
]

# Extract elevation profiles from FABDEM along each transect
transect_results = []

for t in transects:
    line = ee.Geometry.LineString([t["start"], t["end"]])

    # Sample FABDEM elevation along the transect at 30m intervals
    # Also sample HAND and first-wet-date for comparison
    profile_points = ee.Image.cat([
        fabdem_mosaic.rename("elevation"),
        hand.rename("hand"),
        first_wet_date.rename("first_wet_date")
    ]).sample(
        region=line,
        scale=30,
        numPixels=500,  # Sample up to 500 points along the line
        geometries=True  # Keep point locations
    )

    # Extract results
    profile_data = profile_points.getInfo()
    n_points = len(profile_data["features"])
    print(f"  {t['name']}: {n_points} points sampled")

    transect_results.append({
        "name": t["name"],
        "description": t["description"],
        "start": t["start"],
        "end": t["end"],
        "data": profile_data
    })

# Save transect data to CSV for plotting in matplotlib
for i, t in enumerate(transect_results):
    csv_name = f"transect_{i+1}_profile.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lon", "lat", "elevation_m", "hand_m", "first_wet_day"])
        for feat in t["data"]["features"]:
            props = feat["properties"]
            coords = feat["geometry"]["coordinates"]
            writer.writerow([
                round(coords[0], 6),
                round(coords[1], 6),
                round(props.get("elevation", -999), 2),
                round(props.get("hand", -999), 2),
                round(props.get("first_wet_date", -999), 1)
            ])
    print(f"  Saved to {csv_name}")

print(f"\n✓ {len(transect_results)} transect profiles extracted and saved")


# %%
# =============================================================================
# STEP 8: COMBINED ANALYSIS MAP
# =============================================================================
# Why: This is your main presentation figure — everything overlaid on one map.
#      You can toggle layers on/off to explore relationships between:
#      - Flood propagation (first-wet-date)
#      - Terrain (HAND, FABDEM elevation)
#      - Barriers (roads, elevated features)
#      - Haor boundaries (CoCoAH)
#      - Transect locations (for SWOT corridor analysis)

print("\n--- Creating Combined Analysis Map ---")
m = geemap.Map()
m.centerObject(roi, 8)

# --- First-wet-date (from Part 1, recreated here) ---
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
           "1. First Wet Date (propagation)", True)

# --- MERIT Hydro HAND ---
hand_vis = {
    "min": 0, "max": 10,
    "palette": ["0D0887", "5B02A3", "9A179B", "CB4678",
                "EB7852", "FBB32F", "F0F921"]
}
m.addLayer(hand.clip(roi), hand_vis, "2. MERIT Hydro HAND (m)", False)

# --- HAND classification ---
hand_class_vis = {
    "min": 1, "max": 3,
    "palette": ["2166AC", "F4A582", "B2182B"]  # Blue=low, pink=transition, red=high
}
m.addLayer(hand_classes.clip(roi), hand_class_vis,
           "3. HAND classes (blue=low, red=high)", False)

# --- Potential barriers from HAND ---
m.addLayer(potential_barriers.selfMask().clip(roi),
           {"palette": ["FF00FF"]},
           "4. Potential barriers (HAND 2-8m)", False)

# --- FABDEM elevation ---
fabdem_vis = {
    "min": 0, "max": 30,
    "palette": ["000004", "1B0C41", "4A0C6B", "781C6D",
                "A52C60", "CF4446", "ED6925", "FB9B06",
                "F7D03C", "FCFFA4"]
}
m.addLayer(fabdem_mosaic.clip(roi), fabdem_vis, "5. FABDEM elevation (m)", False)

# --- MERIT Hydro river network ---
m.addLayer(river_network.selfMask().clip(roi),
           {"palette": ["00BFFF"]},
           "6. River network (upa>100km²)", False)

# --- CoCoAH haor boundaries ---
# Style as outlines (not filled)
haor_outline = ee.Image().paint(haor_vectors, 1, 1)  # value=1, width=1
m.addLayer(haor_outline, {"palette": ["00FF00"]},
           "7. CoCoAH haor boundaries", False)

# --- Roads (if available) ---
if has_roads:
    road_img = ee.Image().paint(roads_in_roi, 1, 1)
    m.addLayer(road_img, {"palette": ["FFFF00"]},
               "8. Roads (GRIP)", False)
else:
    m.addLayer(raised_features.selfMask().clip(roi),
               {"palette": ["FFFF00"]},
               "8. Raised features (FABDEM-derived)", False)

# --- Transect lines ---
for t in transects:
    line = ee.Geometry.LineString([t["start"], t["end"]])
    m.addLayer(ee.Image().paint(ee.FeatureCollection([ee.Feature(line)]), 1, 3),
               {"palette": ["FFFFFF"]},
               f"T: {t['name']}", False)

# --- Study area outline ---
m.addLayer(ee.Image().paint(roi, 1, 2), {"palette": ["FFFFFF"]}, "Study Area", True)

print("✓ Combined analysis map created — run 'm' to display")
print("""
Layer guide (toggle in panel):
  1. First Wet Date — blue=early, red=late flood arrival
  2. MERIT Hydro HAND — height above nearest drainage
  3. HAND classes — blue=haor interior, red=elevated ground
  4. Potential barriers — magenta = HAND 2-8m between haors
  5. FABDEM elevation — 30m bare-earth terrain
  6. River network — major channels (upstream area >100km²)
  7. CoCoAH boundaries — detected haor outlines (green)
  8. Roads / raised features — yellow = potential barriers
  T: Transect lines — white lines for elevation profiles
""")

m


# %%
# =============================================================================
# STEP 9: EXPORT COCOAH BOUNDARIES AND COMBINED LAYERS
# =============================================================================
# Why: Exporting the CoCoAH haor boundaries as a shapefile/GeoJSON lets you
#      use them in QGIS, Python, or your SWOT analysis pipeline.

# Export CoCoAH haor boundaries to Google Drive as shapefile
task_haors = ee.batch.Export.table.toDrive(
    collection=haor_vectors,
    description="cocoah_haor_boundaries_2025",
    folder="haor_flood_analysis",
    fileNamePrefix="cocoah_haor_boundaries",
    fileFormat="SHP"
)
# Uncomment to start: task_haors.start()

# Export HAND classification
task_hand = ee.batch.Export.image.toDrive(
    image=hand.clip(roi).float(),
    description="merit_hydro_hand_haor",
    folder="haor_flood_analysis",
    fileNamePrefix="merit_hydro_hand",
    region=roi,
    scale=90,
    crs="EPSG:4326",
    maxPixels=1e12
)
# Uncomment to start: task_hand.start()

# Export FABDEM
task_fabdem = ee.batch.Export.image.toDrive(
    image=fabdem_mosaic.clip(roi).float(),
    description="fabdem_haor_30m",
    folder="haor_flood_analysis",
    fileNamePrefix="fabdem_30m",
    region=roi,
    scale=30,
    crs="EPSG:4326",
    maxPixels=1e12
)
# Uncomment to start: task_fabdem.start()

print("\n--- Export Tasks ---")
print("Uncomment task.start() lines to begin exports:")
print("  - cocoah_haor_boundaries.shp (haor polygons)")
print("  - merit_hydro_hand.tif (HAND at 90m)")
print("  - fabdem_30m.tif (FABDEM elevation at 30m)")
print("All will appear in Google Drive → 'haor_flood_analysis' folder")


# %%
# =============================================================================
# STEP 10: SUMMARY AND NEXT STEPS
# =============================================================================

print("""
╔══════════════════════════════════════════════════════════════╗
║                    PART 2 COMPLETE                           ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  What you now have:                                          ║
║                                                              ║
║  From Part 1:                                                ║
║    ✓ First-wet-date map (flood propagation pattern)          ║
║    ✓ Total wet count (flood duration per pixel)              ║
║    ✓ Wet fraction (flood frequency per pixel)                ║
║    ✓ Area time series CSV (total flooded area over time)     ║
║                                                              ║
║  From Part 2:                                                ║
║    ✓ CoCoAH haor boundaries (individual haor polygons)       ║
║    ✓ Per-haor area time series (top 20 haors)                ║
║    ✓ MERIT Hydro HAND overlay (low vs. elevated terrain)     ║
║    ✓ FABDEM 30m elevation (barrier crest estimation)         ║
║    ✓ Road network / raised features (barrier locations)      ║
║    ✓ Elevation transect profiles (3 corridors)               ║
║    ✓ Combined analysis map (all layers together)             ║
║                                                              ║
║  What to do next:                                            ║
║                                                              ║
║  1. INSPECT the combined map:                                ║
║     - Toggle first-wet-date + HAND together                  ║
║     - Do barrier locations (HAND 2-8m) align with abrupt     ║
║       changes in flood timing?                               ║
║     - Do CoCoAH boundaries match the first-wet-date pattern? ║
║                                                              ║
║  2. ADJUST transect locations:                               ║
║     - Based on what you see, move the transect start/end     ║
║       points to cross the most interesting barriers          ║
║     - Look for places where flood timing changes sharply     ║
║                                                              ║
║  3. SWOT WSE analysis (future):                              ║
║     - Extract SWOT WSE along the same transect lines         ║
║     - Compare SWOT WSE with FABDEM barrier crest elevation   ║
║     - Check if WSE > barrier height when downstream haor     ║
║       first appears flooded in the S-1 time series           ║
║                                                              ║
║  4. PRESENT to your supervisor:                              ║
║     - First-wet-date map = propagation evidence              ║
║     - Per-haor area curves = quantitative flood dynamics     ║
║     - Elevation transects = barrier identification           ║
║     - Combined map = integrated analysis framework           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")
