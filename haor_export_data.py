#!/usr/bin/env python3
"""
==============================================================================
HAOR FLOOD ANALYSIS — DATA EXPORT SCRIPT
==============================================================================
Run this as a batch job on your HPC server. It exports all analysis layers
from GEE to your local directory as GeoTIFFs, and computes per-haor area
time series saved as CSV.

Usage:
    python haor_export_data.py

Output directory: /work/a06/wasif/haor_flood_analysis/
    (change OUTPUT_DIR below if needed)

All GeoTIFFs are exported to Google Drive first, then you download them.
The per-haor CSV is computed directly and saved locally.

Settings:
    TOP_N_HAORS: set to None for ALL haors, or a number (e.g., 10) for quick test
==============================================================================
"""
import sys
import os

# Point to your venv's packages
venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)

# Also ensure the venv bin is in PATH (for any subprocesses)
venv_bin = "/work/a06/wasif/.venv/bin"
os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")

# Force correct HOME for PBS jobs
os.environ["HOME"] = "/home/wasif"

import ee
import csv
import json
from datetime import datetime, timedelta

# --- SETTINGS (edit these) ---
OUTPUT_DIR = "/work/a06/wasif/haor_flood_analysis"
GDRIVE_FOLDER = "haor_flood_analysis"
EXPORT_SCALE_COARSE = 100   # meters — for flood maps, HAND
EXPORT_SCALE_FINE = 30      # meters — for FABDEM, barriers
TOP_N_HAORS = None          # None = ALL haors; set to 10 or 20 for quick test
COMPOSITE_WINDOW_DAYS = 6

# --- GEE parameters (same as Part 1) ---
BBOX = [89.8, 23.5, 92.8, 25.5]
BASELINE_START = "2025-01-01"
BASELINE_END   = "2025-03-31"
MONITOR_START  = "2025-04-01"
MONITOR_END    = "2025-09-30"
Z_THRESHOLD    = -2.5
SLOPE_MAX      = 5

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize GEE ---
try:
    ee.Initialize(project="79803231644")
    print("✓ GEE initialized")
except Exception:
    try:
        from google.oauth2.credentials import Credentials
        credentials = Credentials(
            token=None,
            refresh_token="YOUR_REFRESH_TOKEN",
            token_uri="https://oauth2.googleapis.com/token",
            client_id="YOUR_CLIENT_ID",
            client_secret="YOUR_CLIENT_SECRET",
        )
        ee.Initialize(credentials=credentials, project="79803231644")
        print("✓ GEE initialized from hardcoded credentials")
    except Exception as e:
        print(f"✗ GEE initialization failed: {e}")
        sys.exit(1)

roi = ee.Geometry.Rectangle(BBOX)

# ===========================================================================
# SECTION A: FLOOD MASKS AND FIRST-WET-DATE (same as Part 1)
# ===========================================================================
print("\n=== Section A: Flood masks ===")

def get_s1_collection(roi, start, end, pol="VV"):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
            .select(pol))

baseline_col = get_s1_collection(roi, BASELINE_START, BASELINE_END)
baseline_col = baseline_col.map(lambda img: img.updateMask(img.gt(-30)))
baseline_mean = baseline_col.mean().rename("baseline_mean")
baseline_std = baseline_col.reduce(ee.Reducer.stdDev()).rename("baseline_std")
baseline_std = baseline_std.where(baseline_std.lte(0), 0.5)

slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
slope_mask = slope.lt(SLOPE_MAX)

def compute_flood_mask(image):
    masked = image.updateMask(image.gt(-30))
    zscore = masked.subtract(baseline_mean).divide(baseline_std)
    flood = zscore.lt(Z_THRESHOLD).rename("flood_clean")
    flood = flood.updateMask(slope_mask)
    flood = flood.focal_mode(radius=60, units="meters").rename("flood_clean")
    return flood.copyProperties(image, ["system:time_start"])

monitor_col = get_s1_collection(roi, MONITOR_START, MONITOR_END)
flood_masks = monitor_col.map(compute_flood_mask)
print(f"  Flood masks: {flood_masks.size().getInfo()} images")

# First-wet-date
def add_date_band(img):
    date_ms = img.date().millis()
    epoch_ms = ee.Date("2025-01-01").millis()
    days = ee.Number(date_ms).subtract(epoch_ms).divide(86400000)
    return ee.Image.constant(days).float().rename("date_days").updateMask(img.select("flood_clean"))

first_wet_date = flood_masks.map(add_date_band).min().rename("first_wet_date")

# Flood metrics
total_wet = flood_masks.select("flood_clean").sum().rename("total_wet_count")
total_obs = flood_masks.select("flood_clean").count().rename("total_obs_count")
wet_fraction = total_wet.divide(total_obs).rename("wet_fraction")

print("✓ Flood layers computed")


# ===========================================================================
# SECTION B: IMPROVED CoCoAH HAOR BOUNDARIES
# ===========================================================================
print("\n=== Section B: CoCoAH haor boundaries ===")

# Use JRC Global Surface Water + S-1 May 2025 hybrid approach
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater")
water_occurrence = jrc.select("occurrence").clip(roi)
water_seasonality = jrc.select("seasonality").clip(roi)

# Seasonal water: occurrence 10-90% OR seasonality 4-10 months
seasonal_water = water_occurrence.gte(10).And(water_occurrence.lte(90))
seasonal_months = water_seasonality.gte(4).And(water_seasonality.lte(10))
haor_candidate_jrc = seasonal_water.Or(seasonal_months)

# S-1 May 2025 confirmation
cocoah_col = get_s1_collection(roi, "2025-05-01", "2025-05-31")
cocoah_col = cocoah_col.map(lambda img: img.updateMask(img.gt(-30)))
s1_water_may = cocoah_col.median().clip(roi).lt(-13)

# Combine: JRC seasonal AND S-1 confirms water
haor_combined = haor_candidate_jrc.And(s1_water_may).selfMask()

# Morphological opening (erosion→dilation) to break channel connections
eroded = haor_combined.focal_min(radius=200, units="meters", kernelType="circle")
dilated = eroded.focal_max(radius=200, units="meters", kernelType="circle")

# Morphological closing (dilation→erosion) to fill internal holes
closed = dilated.focal_max(radius=150, units="meters", kernelType="circle")
closed = closed.focal_min(radius=150, units="meters", kernelType="circle")

# Vectorize
print("  Vectorizing haor polygons...")
water_for_vector = closed.rename("water").selfMask()
haor_vectors_raw = water_for_vector.reduceToVectors(
    reducer=ee.Reducer.countEvery(), geometry=roi, scale=100,
    maxPixels=1e12, geometryType="polygon", eightConnected=True, bestEffort=True
)

# Size filter: 1-500 km² (at 100m: 100-50000 pixels)
haor_vectors_sized = haor_vectors_raw.filter(
    ee.Filter.And(ee.Filter.gte("count", 100), ee.Filter.lte("count", 50000))
)

# Shape filter: compactness > 0.08 removes rivers
def add_compactness(feature):
    area = feature.geometry().area(100)
    perimeter = feature.geometry().perimeter(100)
    compactness = ee.Number(4).multiply(3.14159).multiply(area).divide(
        perimeter.multiply(perimeter).max(1))
    return feature.set("compactness", compactness)

haor_vectors = haor_vectors_sized.map(add_compactness).filter(
    ee.Filter.gte("compactness", 0.08))

n_haors = haor_vectors.size().getInfo()
haor_areas = haor_vectors.aggregate_array("count").getInfo()
haor_areas_km2 = [a * 0.01 for a in haor_areas]
print(f"✓ CoCoAH: {n_haors} haors detected")
print(f"  Size range: {min(haor_areas_km2):.1f} - {max(haor_areas_km2):.1f} km²")

# Rasterize for export
haor_raster = haor_vectors.reduceToImage(
    properties=["count"], reducer=ee.Reducer.first()
).rename("haor_id").clip(roi)


# ===========================================================================
# SECTION C: MERIT HYDRO HAND
# ===========================================================================
print("\n=== Section C: MERIT Hydro HAND ===")

merit_hydro = ee.Image("MERIT/Hydro/v1_0_1")
hand = merit_hydro.select("hnd").clip(roi)
upa = merit_hydro.select("upa").clip(roi)
river_network = upa.gt(100).rename("river_network")

# HAND classes
hand_classes = (ee.Image(0)
    .where(hand.lt(1), 1)
    .where(hand.gte(1).And(hand.lt(3)), 2)
    .where(hand.gte(3).And(hand.lt(6)), 3)
    .where(hand.gte(6), 4)
    .rename("hand_class").selfMask())

print("✓ MERIT Hydro loaded")

# ===========================================================================
# SECTION D: FABDEM + IMPROVED BARRIER DETECTION (v7 - simple + working)
# ===========================================================================
# Going back to what worked in the original interactive notebook:
#   - FABDEM: simple local relief (mean-based), no Khanh 4-parameter
#   - HAND: v3 settings (1.5-6m range, 500m buffer) → proven 1.94%
#   - Spatial constraint: 1km buffer near JRC seasonal water
#   - Verification at 200m scale to avoid timeouts
print("\n=== Section D: FABDEM barrier detection (v7) ===")

fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
fabdem_mosaic = fabdem.filterBounds(roi).mosaic().clip(roi).rename("fabdem_elev")

# --- Compute FABDEM derivatives (kept for export, useful for transect analysis) ---

# Relative elevation: pixel - local minimum
local_min = fabdem_mosaic.focal_min(radius=200, units="meters")
relative_elev = fabdem_mosaic.subtract(local_min).rename("relative_elev")

# Slope
fabdem_slope = ee.Terrain.slope(fabdem_mosaic).rename("fabdem_slope")

# Aspect (kept for export only — not used in detection)
aspect = ee.Terrain.aspect(fabdem_mosaic)
kernel_offset = 90
aspect_east = aspect.focal_mean(radius=kernel_offset, units="meters",
    kernel=ee.Kernel.rectangle(kernel_offset, 1, "meters"))
aspect_west = aspect.focal_mean(radius=kernel_offset, units="meters",
    kernel=ee.Kernel.rectangle(1, kernel_offset, "meters"))
aspect_diff_raw = aspect_east.subtract(aspect_west).abs()
aspect_diff = aspect_diff_raw.where(aspect_diff_raw.gt(180),
    ee.Image.constant(360).subtract(aspect_diff_raw)).rename("aspect_diff")

# Curvature (kept for export only — not used in detection)
kernel_curv = ee.Kernel.circle(radius=60, units="meters")
focal_mean_curv = fabdem_mosaic.focal_mean(kernel=kernel_curv)
curvature = focal_mean_curv.subtract(fabdem_mosaic).multiply(-1).rename("curvature")

# Local relief at two scales (PRIMARY detection signal)
focal_mean_200 = fabdem_mosaic.focal_mean(radius=200, units="meters")
local_relief_200 = fabdem_mosaic.subtract(focal_mean_200).rename("local_relief_200m")

focal_mean_500 = fabdem_mosaic.focal_mean(radius=500, units="meters")
local_relief_500 = fabdem_mosaic.subtract(focal_mean_500).rename("local_relief_500m")

# --- FABDEM BARRIER DETECTION (simple local relief approach) ---
# This was the original logic from the interactive Part 2 notebook that worked.
# A pixel is a barrier if it's locally elevated at fine OR medium scale:
#   - Fine scale (200m): catches narrow embankments
#   - Medium scale (500m): catches broader elevated features
# Without spatial constraint, this detects ~10-15% (too many — includes
# settlements, uplands, DEM artifacts).
# WITH spatial constraint (near haor edge), reduces to realistic levels.
fabdem_barrier_fine = local_relief_200.gt(0.5)
fabdem_barrier_medium = local_relief_500.gt(0.8)
fabdem_barriers_raw = fabdem_barrier_fine.Or(fabdem_barrier_medium)

# --- SPATIAL CONSTRAINT (1km buffer near JRC seasonal water) ---
jrc_seasonal = water_occurrence.gte(10).And(water_occurrence.lte(90))
jrc_seasonal_buffered = jrc_seasonal.focal_max(radius=1000, units="meters")
near_haor_edge = jrc_seasonal_buffered.rename("near_edge")

fabdem_barriers = fabdem_barriers_raw.And(near_haor_edge).rename("fabdem_barrier")

# --- HAND-based barriers (EXACT v3 settings — proven 1.94% coverage) ---
hand_moderate = hand.gte(1.5).And(hand.lt(6))
nearby_low = hand.focal_min(radius=500, units="meters").lt(1.5)
not_river = upa.lt(50)
hand_barriers_raw = hand_moderate.And(nearby_low).And(not_river)

# HAND uses tighter 500m buffer (90m native resolution)
jrc_seasonal_500 = jrc_seasonal.focal_max(radius=500, units="meters")
near_haor_edge_500 = jrc_seasonal_500.And(jrc_seasonal.Not())
hand_barriers = hand_barriers_raw.And(near_haor_edge_500).rename("hand_barrier")

# --- Combined barriers ---
barrier_high = hand_barriers.And(fabdem_barriers).rename("barrier_high_conf")
barrier_any = hand_barriers.Or(fabdem_barriers).rename("barrier_any_conf")

# --- VERIFICATION at 200m scale to avoid timeouts ---
# 30m verification times out on 82M pixels; 200m is fast and accurate enough
# (area calculations don't need pixel-perfect resolution to be representative)
print("  Verifying barrier coverage at 200m scale (target: 0.5-3%)...")
roi_area_km2 = roi.area(maxError=10).getInfo() / 1e6

for name, layer in [("FABDEM only", fabdem_barriers),
                    ("HAND only", hand_barriers),
                    ("HIGH conf (both)", barrier_high),
                    ("ANY conf (either)", barrier_any)]:
    try:
        barrier_area = layer.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi,
            scale=200, maxPixels=1e10, bestEffort=True
        ).getInfo()
        val_key = list(barrier_area.keys())[0]
        barrier_km2 = (barrier_area[val_key] or 0) / 1e6
        pct = 100 * barrier_km2 / roi_area_km2
        status = "✓" if 0.3 <= pct <= 5.0 else "⚠"
        print(f"  {status} {name}: {barrier_km2:.1f} km² ({pct:.2f}% of AOI)")
    except Exception as e:
        print(f"  ⚠ {name}: verification failed ({str(e)[:60]})")

print("✓ FABDEM barrier detection computed (v7: local relief + spatial constraint)")

# ===========================================================================
# SECTION E: EXPORT ALL LAYERS TO GOOGLE DRIVE
# ===========================================================================
print("\n=== Section E: Exporting layers to Google Drive ===")

exports = {
    "first_wet_date": (first_wet_date.clip(roi).float(), EXPORT_SCALE_COARSE),
    "total_wet_count": (total_wet.clip(roi).int16(), EXPORT_SCALE_COARSE),
    "wet_fraction": (wet_fraction.clip(roi).float(), EXPORT_SCALE_COARSE),
    "hand": (hand.float(), EXPORT_SCALE_COARSE),
    "hand_classes": (hand_classes.int8(), EXPORT_SCALE_COARSE),
    "river_network": (river_network.int8(), EXPORT_SCALE_COARSE),
    "haor_boundaries": (haor_raster.int32(), EXPORT_SCALE_COARSE),
    "fabdem_elevation": (fabdem_mosaic.float(), EXPORT_SCALE_FINE),
    "fabdem_relative_elev": (relative_elev.float(), EXPORT_SCALE_FINE),
    "fabdem_slope": (fabdem_slope.float(), EXPORT_SCALE_FINE),
    "fabdem_local_relief": (local_relief_200.float(), EXPORT_SCALE_FINE),
    "barrier_high_conf": (barrier_high.int8(), EXPORT_SCALE_FINE),
    "barrier_any_conf": (barrier_any.int8(), EXPORT_SCALE_FINE),
    "fabdem_barriers": (fabdem_barriers.int8(), EXPORT_SCALE_FINE),
    "hand_barriers": (hand_barriers.int8(), EXPORT_SCALE_COARSE),
    "jrc_water_occurrence": (water_occurrence.float(), EXPORT_SCALE_COARSE),
}

tasks = []
for name, (image, scale) in exports.items():
    task = ee.batch.Export.image.toDrive(
        image=image, description=f"haor_{name}",
        folder=GDRIVE_FOLDER, fileNamePrefix=name,
        region=roi, scale=scale, crs="EPSG:4326", maxPixels=1e13
    )
    task.start()
    tasks.append((name, task))
    print(f"  Started export: {name} (scale={scale}m)")

# Export CoCoAH haor boundaries as shapefile
task_shp = ee.batch.Export.table.toDrive(
    collection=haor_vectors,
    description="haor_cocoah_boundaries_shp",
    folder=GDRIVE_FOLDER,
    fileNamePrefix="cocoah_haor_boundaries",
    fileFormat="SHP"
)
task_shp.start()
print(f"  Started export: cocoah_haor_boundaries (SHP)")

print(f"\n✓ {len(tasks) + 1} export tasks started")
print(f"  Check Google Drive → '{GDRIVE_FOLDER}' folder")
print(f"  Download all files to: {OUTPUT_DIR}")


# ===========================================================================
# SECTION F: PER-HAOR AREA TIME SERIES (with error handling + incremental save)
# ===========================================================================
print("\n=== Section F: Per-haor area time series ===")

def generate_composite_dates(start, end, interval):
    dates = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while current < end_dt:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=interval)
    return dates

composite_dates = generate_composite_dates(MONITOR_START, MONITOR_END, COMPOSITE_WINDOW_DAYS)

# Determine how many haors to process
if TOP_N_HAORS is not None:
    sorted_indices = sorted(range(len(haor_areas)), key=lambda i: haor_areas[i], reverse=True)
    process_indices = sorted_indices[:TOP_N_HAORS]
    print(f"  Processing top {TOP_N_HAORS} haors by size")
else:
    process_indices = list(range(len(haor_areas)))
    print(f"  Processing ALL {len(process_indices)} haors")

haor_list = haor_vectors.toList(haor_vectors.size())

# --- Check if partial results exist from a previous run ---
csv_path = os.path.join(OUTPUT_DIR, "per_haor_area_timeseries_2025.csv")
completed_indices = set()

if os.path.exists(csv_path):
    # Load previously completed haors to skip them
    import csv as csv_mod
    with open(csv_path, "r") as f:
        reader = csv_mod.reader(f)
        header = next(reader)  # skip header
        for row in reader:
            completed_indices.add(int(row[0]))  # haor_idx is first column
    print(f"  Found {len(completed_indices)} previously completed haors — resuming")

# --- Open CSV in append mode (or write mode if starting fresh) ---
write_mode = "a" if completed_indices else "w"
csv_file = open(csv_path, write_mode, newline="")
writer = csv.writer(csv_file)

if not completed_indices:
    # Write header only if starting fresh
    header = ["haor_idx", "total_area_km2"] + composite_dates
    writer.writerow(header)

# --- Pre-compute flood composites for each date window ---
# This avoids rebuilding the computation chain for every haor
print("  Pre-computing flood composites per date window...")
flood_composites = []
for date_str in composite_dates:
    end_str = (datetime.strptime(date_str, "%Y-%m-%d") +
               timedelta(days=COMPOSITE_WINDOW_DAYS)).strftime("%Y-%m-%d")
    window = flood_masks.filterDate(date_str, end_str)
    # Create composite once, reuse for all haors
    composite = window.select("flood_clean").max()
    # Pre-multiply by pixel area
    composite_area = composite.multiply(ee.Image.pixelArea())
    flood_composites.append(composite_area)
print(f"  {len(flood_composites)} composites ready")

# --- Process each haor ---
failed_count = 0
processed_count = 0
import time

for rank, idx in enumerate(process_indices):
    # Skip if already completed in a previous run
    if idx in completed_indices:
        continue

    haor_feature = ee.Feature(haor_list.get(idx))
    haor_geom = haor_feature.geometry()
    haor_area_km2 = haor_areas[idx] * 0.01

    if (processed_count) % 10 == 0:
        print(f"  Processing haor {rank+1}/{len(process_indices)} "
              f"(idx={idx}, area ~{haor_area_km2:.1f} km², "
              f"done={processed_count}, failed={failed_count})...")

    haor_ts = []
    haor_failed = False

    for comp_area in flood_composites:
        try:
            area_val = comp_area.reduceRegion(
                reducer=ee.Reducer.sum(),
                geometry=haor_geom,
                scale=200,  # Use 200m instead of 100m to reduce computation
                maxPixels=1e8
            ).get("flood_clean")
            area_km2 = ee.Number(area_val).divide(1e6).getInfo()
            haor_ts.append(round(area_km2, 2) if area_km2 else 0)
        except Exception as e:
            # If one date fails, mark it as -1 and continue
            haor_ts.append(-1)
            if not haor_failed:
                print(f"    ⚠ Haor idx={idx} had an error: {str(e)[:80]}")
                haor_failed = True

    # Write this haor's results immediately (incremental save)
    row = [idx, round(haor_area_km2, 2)] + haor_ts
    writer.writerow(row)
    csv_file.flush()  # Force write to disk immediately

    processed_count += 1
    if haor_failed:
        failed_count += 1

    # Small delay to avoid overwhelming GEE API
    time.sleep(0.1)

csv_file.close()
print(f"\n✓ Per-haor time series saved to {csv_path}")
print(f"  Processed: {processed_count}, Failed: {failed_count}, "
      f"Skipped (from previous run): {len(completed_indices)}")

# ===========================================================================
# SECTION G: TOTAL AREA TIME SERIES
# ===========================================================================
print("\n=== Section G: Total area time series ===")

total_area_ts = []
for i, date_str in enumerate(composite_dates):
    try:
        end_str = (datetime.strptime(date_str, "%Y-%m-%d") +
                   timedelta(days=COMPOSITE_WINDOW_DAYS)).strftime("%Y-%m-%d")
        window = flood_masks.filterDate(date_str, end_str)
        composite = window.select("flood_clean").max()
        area_val = composite.multiply(ee.Image.pixelArea()).reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi,
            scale=200, maxPixels=1e12
        ).get("flood_clean")
        area_km2 = ee.Number(area_val).divide(1e6).getInfo()
        total_area_ts.append({"date": date_str, "area_km2": round(area_km2, 1)})
        print(f"  {date_str}: {area_km2:.1f} km²")
    except Exception as e:
        total_area_ts.append({"date": date_str, "area_km2": -1})
        print(f"  {date_str}: ERROR - {str(e)[:60]}")

csv_total = os.path.join(OUTPUT_DIR, "total_flood_area_timeseries_2025.csv")
with open(csv_total, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "area_km2"])
    writer.writeheader()
    writer.writerows(total_area_ts)
print(f"✓ Total area time series saved to {csv_total}")