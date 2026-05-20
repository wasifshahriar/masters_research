#!/usr/bin/env python3
"""
==============================================================================
HAOR FLOOD PROPAGATION ANALYSIS — PART 1: Sentinel-1 Inundation Time Series
==============================================================================
Version 2 — Fixes applied:
  1. Individual flood maps now use ±3 day windows to combine ascending +
     descending passes → full AOI coverage instead of half-empty maps
  2. First-wet-date map uses a smooth 30-level color gradient instead of
     6 monthly colors → much better for analysis
  3. Added composite approach for date browsing: each "date" is actually
     a 6-day window matching S-1 revisit cycle

Author : Wasif Shahriar (M1, Yamazaki Lab, UTokyo)
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


# %%
# =============================================================================
# STEP 1: STUDY AREA AND TIME PERIODS
# =============================================================================

BBOX = [89.8, 23.5, 92.8, 25.5]
roi = ee.Geometry.Rectangle(BBOX)

BASELINE_START = "2025-01-01"
BASELINE_END   = "2025-03-31"
MONITOR_START  = "2025-04-01"
MONITOR_END    = "2025-09-30"

# Z-score threshold
Z_THRESHOLD = -2.5

# Slope mask threshold (degrees)
SLOPE_MAX = 5

# Date window for compositing individual flood maps
# WHY 6 DAYS: Sentinel-1 revisit is ~6 days when combining A+B satellites.
# Using a 6-day window ensures we capture BOTH ascending AND descending passes
# so the entire AOI is covered. Your previous code used 1-day windows, which
# only captured ONE pass → only half the AOI was visible.
COMPOSITE_WINDOW_DAYS = 6

MIN_BASELINE_IMAGES = 5

print(f"Study area: {BBOX}")
print(f"Baseline: {BASELINE_START} to {BASELINE_END}")
print(f"Monitoring: {MONITOR_START} to {MONITOR_END}")
print(f"Z-score threshold: {Z_THRESHOLD}")
print(f"Composite window: {COMPOSITE_WINDOW_DAYS} days")


# %%
# =============================================================================
# STEP 2: CHECK SENTINEL-1 DATA AVAILABILITY
# =============================================================================

def get_s1_collection(roi, start, end, pol="VV"):
    """Get filtered Sentinel-1 GRD collection."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
            .select(pol))

baseline_col = get_s1_collection(roi, BASELINE_START, BASELINE_END)
n_baseline = baseline_col.size().getInfo()
print(f"\n--- Data Availability ---")
print(f"Baseline images (Jan-Mar 2025): {n_baseline}")

monitor_col = get_s1_collection(roi, MONITOR_START, MONITOR_END)
n_monitor = monitor_col.size().getInfo()
print(f"Monitoring images (Apr-Sep 2025): {n_monitor}")

def get_image_dates(collection):
    """Extract unique acquisition dates from an image collection."""
    dates = collection.aggregate_array("system:time_start").getInfo()
    return sorted(set([datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dates]))

monitor_dates = get_image_dates(monitor_col)
print(f"\nMonitoring dates ({len(monitor_dates)} unique dates):")
for i, d in enumerate(monitor_dates):
    print(f"  [{i+1:2d}] {d}")

asc = monitor_col.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING")).size().getInfo()
desc = monitor_col.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")).size().getInfo()
print(f"\nAscending: {asc}, Descending: {desc}")

if n_baseline < MIN_BASELINE_IMAGES:
    print(f"\n⚠ WARNING: Only {n_baseline} baseline images.")
else:
    print(f"\n✓ Sufficient baseline images ({n_baseline})")


# %%
# =============================================================================
# STEP 3: COMPUTE BASELINE STATISTICS
# =============================================================================
# UNCHANGED from v1 — this step was working correctly.

def compute_baseline_stats(roi, start, end, pol="VV"):
    """Compute per-pixel mean and std of VV backscatter during baseline."""
    base = get_s1_collection(roi, start, end, pol)
    base = base.map(lambda img: img.updateMask(img.gt(-30)))
    mean_img = base.mean().rename("baseline_mean")
    std_img = base.reduce(ee.Reducer.stdDev()).rename("baseline_std")
    std_img = std_img.where(std_img.lte(0), 0.5)
    n_images = base.count().rename("baseline_count")
    return mean_img, std_img, n_images

print("Computing baseline statistics...")
baseline_mean, baseline_std, baseline_count = compute_baseline_stats(
    roi, BASELINE_START, BASELINE_END
)
print("✓ Baseline mean and std computed")

sample_point = ee.Geometry.Point([91.0, 24.8])
stats = baseline_mean.reduceRegion(
    reducer=ee.Reducer.first(), geometry=sample_point, scale=10
).getInfo()
print(f"  Sample baseline mean at (91.0, 24.8): {stats}")


# %%
# =============================================================================
# STEP 4: COMPUTE Z-SCORE FLOOD MASKS
# =============================================================================
# UNCHANGED from v1 — the per-image flood detection logic was correct.

def compute_zscore_flood_mask(image, baseline_mean, baseline_std, z_threshold, slope_mask):
    """Compute Z-score flood mask for a single Sentinel-1 image."""
    masked = image.updateMask(image.gt(-30))
    zscore = masked.subtract(baseline_mean).divide(baseline_std).rename("zscore")
    flood = zscore.lt(z_threshold).rename("flood")
    flood = flood.updateMask(slope_mask)
    flood_clean = flood.focal_mode(radius=60, units="meters").rename("flood_clean")
    return flood_clean.copyProperties(image, ["system:time_start"])

slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
slope_mask = slope.lt(SLOPE_MAX)

monitor_col_masked = monitor_col.map(lambda img: img.updateMask(img.gt(-30)))

print("Computing Z-score flood masks for each monitoring date...")
flood_masks = monitor_col_masked.map(
    lambda img: compute_zscore_flood_mask(
        img, baseline_mean, baseline_std, Z_THRESHOLD, slope_mask
    )
)
n_masks = flood_masks.size().getInfo()
print(f"✓ Generated {n_masks} flood masks")


# %%
# =============================================================================
# STEP 5: COMPUTE FIRST-WET-DATE MAP AND FLOOD METRICS
# =============================================================================
# UNCHANGED from v1 — the computation logic was correct.

def compute_first_wet_date(flood_mask_collection):
    """Compute per-pixel first-wet-date (days since 2025-01-01)."""
    def add_date_band(img):
        date_ms = img.date().millis()
        epoch_ms = ee.Date("2025-01-01").millis()
        days = ee.Number(date_ms).subtract(epoch_ms).divide(86400000)
        date_img = ee.Image.constant(days).float().rename("date_days")
        return date_img.updateMask(img.select("flood_clean"))
    date_images = flood_mask_collection.map(add_date_band)
    first_wet = date_images.min().rename("first_wet_date")
    return first_wet

def compute_flood_metrics(flood_mask_collection):
    """Compute per-pixel flood duration and frequency."""
    total_wet = flood_mask_collection.select("flood_clean").sum().rename("total_wet_count")
    total_obs = flood_mask_collection.select("flood_clean").count().rename("total_obs_count")
    wet_fraction = total_wet.divide(total_obs).rename("wet_fraction")
    return total_wet, total_obs, wet_fraction

def compute_area_time_series(flood_mask_collection, roi, scale=100):
    """Compute total flooded area (km²) per date."""
    def get_flood_area(img):
        pixel_area = img.select("flood_clean").multiply(ee.Image.pixelArea())
        area_m2 = pixel_area.reduceRegion(
            reducer=ee.Reducer.sum(), geometry=roi, scale=scale, maxPixels=1e12
        ).get("flood_clean")
        area_km2 = ee.Number(area_m2).divide(1e6)
        return img.set("flood_area_km2", area_km2)
    return flood_mask_collection.map(get_flood_area)

print("Computing first-wet-date map...")
first_wet_date = compute_first_wet_date(flood_masks)
print("✓ First-wet-date map computed")

print("Computing flood metrics...")
total_wet, total_obs, wet_fraction = compute_flood_metrics(flood_masks)
print("✓ Flood metrics computed")

print("Computing area time series...")
flood_masks_with_area = compute_area_time_series(flood_masks, roi, scale=100)

area_list = flood_masks_with_area.aggregate_array("flood_area_km2").getInfo()
date_list = flood_masks_with_area.aggregate_array("system:time_start").getInfo()

area_dates = []
for d, a in zip(date_list, area_list):
    date_str = datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d")
    area_dates.append({"date": date_str, "area_km2": round(a, 1) if a else 0})

print(f"\n--- Flood Area Time Series ---")
print(f"{'Date':<14} {'Area (km²)':>12}")
print("-" * 28)
for item in sorted(area_dates, key=lambda x: x["date"]):
    print(f"{item['date']:<14} {item['area_km2']:>12.1f}")

print(f"\n✓ Area time series extracted for {len(area_dates)} dates")


# %%
# =============================================================================
# STEP 6: VISUALIZE — FIRST-WET-DATE MAP WITH SMOOTH GRADIENT
# =============================================================================
# CHANGED: Now uses a 30-level smooth color gradient instead of 6 monthly colors.
#
# Why this matters:
#   With 6 colors, most of the haor region showed up as one uniform green
#   (= June), making it impossible to see propagation within that month.
#   With 30+ levels, you can distinguish flooding that happened on June 1
#   vs June 12 vs June 20 — which is exactly the temporal resolution
#   you need for propagation analysis.
#
# Color scheme: a perceptually uniform gradient from dark blue (earliest)
# through cyan, green, yellow, orange to dark red (latest).
# This is similar to the 'turbo' colormap used in scientific visualization.

print("\nCreating main analysis map...")
m = geemap.Map()
m.centerObject(roi, 8)

# --- CHANGED: Smooth 30-level color gradient for first-wet-date ---
# Day 90 = April 1, Day 272 = September 30
# Each "step" in the palette ≈ 6 days (matching S-1 revisit cycle)
first_wet_vis = {
    "min": 90,
    "max": 272,
    "palette": [
        # April (days 90-120): deep blue → blue
        "023858", "045a8d", "0570b0", "3690c0",
        # May (days 121-151): cyan → teal
        "74a9cf", "66c2a4", "41ae76", "238b45",
        # June (days 152-181): green → yellow-green
        "006d2c", "31a354", "78c679", "addd8e",
        # July (days 182-212): yellow → yellow-orange
        "d9f0a3", "f7fcb1", "ffeda0", "fed976",
        # August (days 213-243): orange
        "feb24c", "fd8d3c", "fc4e2a", "e31a1c",
        # September (days 244-272): red → dark red
        "d7301f", "bd0026", "a50026", "800026",
    ]
}

m.addLayer(
    first_wet_date.clip(roi),
    first_wet_vis,
    "First Wet Date (smooth gradient)",
    True
)

# Duration
duration_vis = {
    "min": 0,
    "max": 20,
    "palette": ["FFFFFF", "C6DBEF", "6BAED6", "2171B5", "08306B"]
}
m.addLayer(total_wet.clip(roi), duration_vis, "Total Wet Count (duration)", False)

# Frequency
frequency_vis = {
    "min": 0,
    "max": 1,
    "palette": ["FFFFFF", "FED976", "FEB24C", "FD8D3C", "FC4E2A", "E31A1C", "B10026"]
}
m.addLayer(wet_fraction.clip(roi), frequency_vis, "Wet Fraction (frequency)", False)

# Baseline mean for context
m.addLayer(baseline_mean.clip(roi), {"min": -20, "max": -5}, "Baseline Mean VV (dB)", False)

# ROI outline
m.addLayer(ee.Image().paint(roi, 1, 2), {"palette": ["FFFFFF"]}, "Study Area", True)

print("✓ Main map created — run 'm' to display")
print("\nColor guide for First Wet Date:")
print("  Dark blue  = April (earliest flooding)")
print("  Cyan/teal  = May")
print("  Green      = Early June")
print("  Yellow-grn = Late June")
print("  Yellow     = July")
print("  Orange     = August")
print("  Red        = September (latest flooding)")

m


# %%
# =============================================================================
# STEP 7: INDIVIDUAL DATE FLOOD MAPS — FIXED FOR FULL COVERAGE
# =============================================================================
# CHANGED: Two fixes for the half-empty map problem:
#
# Fix 1: Use 6-day composite windows instead of 1-day windows.
#   Sentinel-1 passes your area from different directions on different days.
#   On any single day, only ONE orbital track is active → covers ~half your AOI.
#   By compositing over 6 days, we capture BOTH ascending and descending passes,
#   giving full spatial coverage of the entire study area.
#
# Fix 2: Generate composites at regular 6-day intervals.
#   Instead of picking specific dates (which might miss some), we create
#   a regular grid of 6-day windows from April 1 to September 30.
#   This gives ~30 time steps — one per S-1 revisit cycle.

def generate_composite_dates(start_date, end_date, interval_days):
    """Generate a list of start dates for composite windows."""
    dates = []
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    while current < end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=interval_days)
    return dates

def create_composite_flood_map(flood_masks, start_date, window_days):
    """
    Create a composite flood map over a time window.
    Uses .max() to take the union of all flood detections in the window.
    If ANY image in the window shows flooding at a pixel, it's marked as flooded.
    """
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + 
                timedelta(days=window_days)).strftime("%Y-%m-%d")
    
    window_masks = flood_masks.filterDate(start_date, end_date)
    # Use .max() → pixel is flooded if it was flooded in ANY image in the window
    composite = window_masks.select("flood_clean").max().rename("flood_composite")
    return composite, window_masks.size()


# Generate regular 6-day composite dates
composite_dates = generate_composite_dates(MONITOR_START, MONITOR_END, COMPOSITE_WINDOW_DAYS)
print(f"\n--- Generating {len(composite_dates)} composite flood maps ---")
print(f"  Window size: {COMPOSITE_WINDOW_DAYS} days per composite")

# Create the date browser map
print("\nCreating date browser map with full-coverage composites...")
m2 = geemap.Map()
m2.centerObject(roi, 8)

# We'll add every composite as a toggleable layer
# But to avoid overwhelming the layer panel, we group by month
# and only make the first of each month visible by default
composites_added = 0

for date_str in composite_dates:
    end_dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=COMPOSITE_WINDOW_DAYS)
    end_str = end_dt.strftime("%Y-%m-%d")
    
    # Create composite
    window_masks = flood_masks.filterDate(date_str, end_str)
    n_imgs = window_masks.size().getInfo()
    
    if n_imgs > 0:
        composite = window_masks.select("flood_clean").max()
        
        # Determine month for color coding
        month = datetime.strptime(date_str, "%Y-%m-%d").month
        month_colors = {
            4: "0570b0",  # April: blue
            5: "41ae76",  # May: teal
            6: "78c679",  # June: green
            7: "fed976",  # July: yellow
            8: "fc4e2a",  # August: orange
            9: "bd0026",  # September: red
        }
        color = month_colors.get(month, "FF0000")
        
        # Make first composite of each month visible by default
        day = datetime.strptime(date_str, "%Y-%m-%d").day
        visible = (day <= COMPOSITE_WINDOW_DAYS)  # First composite of month
        
        m2.addLayer(
            composite.selfMask().clip(roi),
            {"palette": [color]},
            f"{date_str} to {end_str} (n={n_imgs})",
            visible
        )
        composites_added += 1

# Add first-wet-date as reference layer
m2.addLayer(first_wet_date.clip(roi), first_wet_vis, "First Wet Date", False)

# Add ROI
m2.addLayer(ee.Image().paint(roi, 1, 2), {"palette": ["FFFFFF"]}, "Study Area", True)

print(f"✓ Added {composites_added} composite flood maps")
print("  Each layer covers a 6-day window with full AOI coverage")
print("  Toggle layers on/off in the panel to see flood progression")
print("  Colors: blue=Apr, teal=May, green=Jun, yellow=Jul, orange=Aug, red=Sep")
print("\n  Display with 'm2'")

m2


# %%
# =============================================================================
# STEP 8: SIDE-BY-SIDE COMPARISON MAP FOR KEY DATES
# =============================================================================
# NEW: Creates a focused comparison of pre-flood, peak-flood, and post-peak.
# This is useful for your D-Lab presentation — shows the dramatic change.

print("\nCreating key-dates comparison map...")
m3 = geemap.Map()
m3.centerObject(roi, 8)

# Pre-flood: April (before significant flooding)
pre_flood = flood_masks.filterDate("2025-04-01", "2025-04-30")
n_pre = pre_flood.size().getInfo()
if n_pre > 0:
    pre_composite = pre_flood.select("flood_clean").max()
    m3.addLayer(pre_composite.selfMask().clip(roi), 
                {"palette": ["0570b0"]}, f"Pre-flood: April (n={n_pre})", True)

# Peak flood: late May to early June (confirmed peak event)
peak_flood = flood_masks.filterDate("2025-05-25", "2025-06-10")
n_peak = peak_flood.size().getInfo()
if n_peak > 0:
    peak_composite = peak_flood.select("flood_clean").max()
    m3.addLayer(peak_composite.selfMask().clip(roi),
                {"palette": ["FF0000"]}, f"Peak flood: 25 May-10 Jun (n={n_peak})", True)

# Post-peak: July
post_peak = flood_masks.filterDate("2025-07-01", "2025-07-31")
n_post = post_peak.size().getInfo()
if n_post > 0:
    post_composite = post_peak.select("flood_clean").max()
    m3.addLayer(post_composite.selfMask().clip(roi),
                {"palette": ["FFA500"]}, f"Post-peak: July (n={n_post})", False)

# Late monsoon: September
late_monsoon = flood_masks.filterDate("2025-09-01", "2025-09-30")
n_late = late_monsoon.size().getInfo()
if n_late > 0:
    late_composite = late_monsoon.select("flood_clean").max()
    m3.addLayer(late_composite.selfMask().clip(roi),
                {"palette": ["800026"]}, f"Late monsoon: Sep (n={n_late})", False)

m3.addLayer(ee.Image().paint(roi, 1, 2), {"palette": ["FFFFFF"]}, "Study Area", True)

print("✓ Key-dates comparison map created — run 'm3'")
print("  Blue = pre-flood (April)")
print("  Red = peak flood (late May - early June)")
print("  Orange = post-peak (July)")
print("  Dark red = late monsoon (September)")

m3


# %%
# =============================================================================
# STEP 9: SAVE AREA TIME SERIES TO CSV
# =============================================================================
# UNCHANGED from v1 — this was working correctly.

csv_filename = "haor_flood_area_timeseries_2025.csv"
area_dates_sorted = sorted(area_dates, key=lambda x: x["date"])

with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "area_km2"])
    writer.writeheader()
    writer.writerows(area_dates_sorted)

print(f"✓ Area time series saved to {csv_filename}")

areas = [x["area_km2"] for x in area_dates_sorted if x["area_km2"] > 0]
if areas:
    max_area = max(areas)
    max_idx = [x["area_km2"] for x in area_dates_sorted].index(max_area)
    print(f"\n--- Flood Area Summary ---")
    print(f"  Observation dates: {len(area_dates_sorted)}")
    print(f"  Dates with flooding: {len(areas)}")
    print(f"  Maximum flooded area: {max_area:.1f} km²")
    print(f"  Date of maximum: {area_dates_sorted[max_idx]['date']}")
    print(f"  Min flooded area (>0): {min(areas):.1f} km²")


# %%
# =============================================================================
# STEP 10: EXPORT RESULTS
# =============================================================================
# UNCHANGED from v1.

EXPORT_SCALE = 100
EXPORT_CRS = "EPSG:4326"

task_fwd = ee.batch.Export.image.toDrive(
    image=first_wet_date.clip(roi).float(),
    description="haor_first_wet_date_2025",
    folder="haor_flood_analysis",
    fileNamePrefix="first_wet_date_2025",
    region=roi,
    scale=EXPORT_SCALE,
    crs=EXPORT_CRS,
    maxPixels=1e12
)
# Uncomment to start: task_fwd.start()

task_wf = ee.batch.Export.image.toDrive(
    image=wet_fraction.clip(roi).float(),
    description="haor_wet_fraction_2025",
    folder="haor_flood_analysis",
    fileNamePrefix="wet_fraction_2025",
    region=roi,
    scale=EXPORT_SCALE,
    crs=EXPORT_CRS,
    maxPixels=1e12
)
# Uncomment to start: task_wf.start()

task_twc = ee.batch.Export.image.toDrive(
    image=total_wet.clip(roi).int16(),
    description="haor_total_wet_count_2025",
    folder="haor_flood_analysis",
    fileNamePrefix="total_wet_count_2025",
    region=roi,
    scale=EXPORT_SCALE,
    crs=EXPORT_CRS,
    maxPixels=1e12
)
# Uncomment to start: task_twc.start()

print("\n--- Export Tasks ---")
print("Uncomment task.start() lines above to begin exports.")
print("Files will appear in Google Drive → 'haor_flood_analysis' folder")


# %%
# =============================================================================
# SUMMARY OF CHANGES FROM VERSION 1
# =============================================================================
print("""
╔══════════════════════════════════════════════════════════════╗
║                    CHANGES FROM VERSION 1                    ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  STEP 6 — First-wet-date color gradient:                     ║
║    OLD: 6 colors (one per month) → most areas = same green   ║
║    NEW: 24-color smooth gradient with ~6-day resolution       ║
║         Dark blue → cyan → green → yellow → orange → red     ║
║         Now you can distinguish June 1 vs June 12 vs June 20 ║
║                                                              ║
║  STEP 7 — Individual flood maps:                             ║
║    OLD: 1-day windows → only one S-1 pass → half-empty maps  ║
║    NEW: 6-day composite windows → captures both ASC + DESC   ║
║         passes → FULL spatial coverage of entire AOI          ║
║         ~30 composites generated (one per S-1 cycle)          ║
║                                                              ║
║  STEP 8 — NEW: Key-dates comparison map                      ║
║    Pre-flood (April) vs Peak (late May-June) vs Post (July)   ║
║    Useful for D-Lab presentation to show dramatic change      ║
║                                                              ║
║  Steps 0-5, 9-10: UNCHANGED (working correctly)              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝

PART 1 COMPLETE — Next steps:
  1. Inspect the first-wet-date map for propagation patterns
  2. Look for sharp color boundaries (= potential embankments)
  3. Note which haors flood early vs late
  4. Part 2 will add: CoCoAH boundaries, MERIT Hydro HAND, FABDEM
""")
