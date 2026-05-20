#!/usr/bin/env python3
"""
==============================================================================
HAOR FLOOD PROPAGATION ANALYSIS — PART 1: Sentinel-1 Inundation Time Series
==============================================================================

Author : Wasif Shahriar (M1, Yamazaki Lab, UTokyo)
Purpose: Build a Sentinel-1 SAR time series of binary flood masks over the
         NE Bangladesh haor region for the 2025 monsoon season, then compute
         per-pixel flood propagation metrics (first-wet-date, duration,
         frequency) to understand how flooding spreads across haors.

Method : Z-score approach (DeVries et al. 2020, RSE) — instead of a fixed
         backscatter threshold, we compute how many standard deviations
         each pixel's backscatter deviates from its dry-season baseline.
         This adapts automatically to local land cover and viewing geometry.

Why this matters for your thesis:
  - Your committee asked: "Are haors connected or isolated during flooding?"
  - To answer this, you need to see the SEQUENCE of flooding — which areas
    flood first, which flood later, and whether the pattern is spatially
    coherent (suggesting connectivity) or random (suggesting isolation).
  - A single flood map can't show this. You need a TIME SERIES of flood maps.
  - The first-wet-date map is your primary evidence for connectivity.

Verified 2025 flood timeline:
  - Dry baseline: January–March 2025 (no significant flooding)
  - Flood onset: mid-May 2025
  - Peak event: 29 May – 2 June 2025 (deep depression, 405mm/24hr on 1 June)
  - Affected: Sylhet, Sunamganj, Moulvibazar, Habiganj, Netrokona
  (Source: BDRCS Situation Report 1, 2 June 2025; Copernicus Sentinel-2 9 June 2025)

How to run:
  - On your UTokyo HPC Jupyter server, or any environment with GEE access
  - Requires: earthengine-api, geemap
  - pip install earthengine-api geemap (if not installed)

==============================================================================
"""

# %%
# =============================================================================
# STEP 0: IMPORTS AND GEE INITIALIZATION
# =============================================================================
# Why: We need Google Earth Engine for satellite data access and geemap for
#      visualization. All processing happens on Google's servers — your HPC
#      just sends commands and receives results.

import ee
import geemap
import json
from datetime import datetime, timedelta

# Initialize GEE — use whichever method works on your system
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
# STEP 1: DEFINE YOUR STUDY AREA AND TIME PERIODS
# =============================================================================
# Why: We need to tell GEE exactly where and when to look for data.
#
# The AOI covers the entire haor belt of NE Bangladesh.
# The baseline period (Jan-Mar) is when haors are dry — this establishes
# "normal" backscatter for each pixel.
# The monitoring period (Apr-Sep) covers the full flood season.

# --- Study Area ---
# This is your core analysis bounding box (BBOX_BD)
BBOX = [89.8, 23.5, 92.8, 25.5]  # [west, south, east, north]
roi = ee.Geometry.Rectangle(BBOX)

# --- Time Periods ---
# Baseline: dry season, no flooding expected
# We use 3 months to get enough images for robust statistics
BASELINE_START = "2025-01-01"
BASELINE_END   = "2025-03-31"

# Monitoring: the flood season we want to track
MONITOR_START  = "2025-04-01"
MONITOR_END    = "2025-09-30"

# --- Parameters ---
# Z-score threshold: how many standard deviations below normal = "flooded"
# DeVries et al. tested -2.0 to -3.0; -2.0 catches more floods but more noise
# -3.0 is conservative (fewer false positives). We start with -2.5 as a balance.
Z_THRESHOLD = -2.5

# Slope mask: remove hillsides where SAR shadows can look like water
SLOPE_MAX = 5  # degrees — haors are flat, so this removes hill artifacts

# Minimum pixels for baseline: need enough images for reliable statistics
MIN_BASELINE_IMAGES = 5

print(f"Study area: {BBOX}")
print(f"Baseline: {BASELINE_START} to {BASELINE_END}")
print(f"Monitoring: {MONITOR_START} to {MONITOR_END}")
print(f"Z-score threshold: {Z_THRESHOLD}")


# %%
# =============================================================================
# STEP 2: CHECK SENTINEL-1 DATA AVAILABILITY
# =============================================================================
# Why: Before doing any analysis, we need to know how many Sentinel-1 images
#      are available over our area. This tells us:
#      - Whether we have enough baseline images for statistics
#      - How many flood-period images we'll get (= temporal resolution)
#      - Which orbit directions (ascending/descending) are available
#
# Sentinel-1 acquires data in IW (Interferometric Wide) mode over land,
# with VV and VH polarization. Over Bangladesh, we typically get an image
# every 12 days from each orbit direction, so ~6 days combined.

def get_s1_collection(roi, start, end, pol="VV"):
    """Get filtered Sentinel-1 GRD collection."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi)
            .filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
            .select(pol))

# Check baseline availability
baseline_col = get_s1_collection(roi, BASELINE_START, BASELINE_END)
n_baseline = baseline_col.size().getInfo()
print(f"\n--- Data Availability ---")
print(f"Baseline images (Jan-Mar 2025): {n_baseline}")

# Check monitoring period availability
monitor_col = get_s1_collection(roi, MONITOR_START, MONITOR_END)
n_monitor = monitor_col.size().getInfo()
print(f"Monitoring images (Apr-Sep 2025): {n_monitor}")

# Get the dates of monitoring images so we know our temporal resolution
def get_image_dates(collection):
    """Extract acquisition dates from an image collection."""
    dates = collection.aggregate_array("system:time_start").getInfo()
    return sorted(set([datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d") for d in dates]))

monitor_dates = get_image_dates(monitor_col)
print(f"\nMonitoring dates ({len(monitor_dates)} unique dates):")
for d in monitor_dates:
    print(f"  {d}")

# Check orbit directions
asc = monitor_col.filter(ee.Filter.eq("orbitProperties_pass", "ASCENDING")).size().getInfo()
desc = monitor_col.filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")).size().getInfo()
print(f"\nAscending: {asc}, Descending: {desc}")

if n_baseline < MIN_BASELINE_IMAGES:
    print(f"\n⚠ WARNING: Only {n_baseline} baseline images. Need at least {MIN_BASELINE_IMAGES}.")
    print("  Consider extending baseline period or using both ASC+DESC orbits.")
else:
    print(f"\n✓ Sufficient baseline images for Z-score computation.")


# %%
# =============================================================================
# STEP 3: COMPUTE BASELINE STATISTICS (MEAN AND STANDARD DEVIATION)
# =============================================================================
# Why: The Z-score method requires knowing what "normal" looks like for each
#      pixel. We compute the mean and standard deviation of VV backscatter
#      during the dry season (Jan-Mar 2025).
#
# How Z-scores work (from DeVries et al. 2020):
#   Z = (current_backscatter - mean_baseline) / std_baseline
#
#   If Z is very negative (e.g., -3), the pixel is MUCH darker than normal
#   → likely flooded (smooth water reflects radar away = very low backscatter)
#
#   If Z is near 0, the pixel looks normal → not flooded
#
# Why this is better than your current -3dB threshold:
#   Your current code compares a flood image to a reference image (delta < -3dB).
#   But the -3dB threshold doesn't adapt to local conditions. A pixel that's
#   always dark (e.g., a river) won't show a big change even when flooded.
#   The Z-score adapts: it asks "is this pixel unusually dark for THIS pixel?"
#
# We compute separate baselines for ascending and descending orbits because
# they look at the ground from different angles, producing different backscatter.

def compute_baseline_stats(roi, start, end, pol="VV"):
    """
    Compute per-pixel mean and std of VV backscatter during baseline period.
    Returns separate stats for ascending and descending orbits.
    """
    base = get_s1_collection(roi, start, end, pol)

    # Apply border noise mask (remove very low values at swath edges)
    base = base.map(lambda img: img.updateMask(img.gt(-30)))

    # Compute stats for ALL orbits combined
    # (simpler and gives more images per pixel for robust stats)
    mean_img = base.mean().rename("baseline_mean")
    std_img = base.reduce(ee.Reducer.stdDev()).rename("baseline_std")

    # Ensure std is not zero (would cause division by zero in Z-score)
    # Replace zero std with a small value (0.5 dB)
    std_img = std_img.where(std_img.lte(0), 0.5)

    n_images = base.count().rename("baseline_count")

    return mean_img, std_img, n_images

print("Computing baseline statistics...")
baseline_mean, baseline_std, baseline_count = compute_baseline_stats(
    roi, BASELINE_START, BASELINE_END
)
print("✓ Baseline mean and std computed")

# Quick diagnostic: check the baseline stats make sense
# For land in Bangladesh, typical VV backscatter is -8 to -15 dB
# For water, it's -18 to -25 dB
sample_point = ee.Geometry.Point([91.0, 24.8])  # A point in the haor region
stats = baseline_mean.reduceRegion(
    reducer=ee.Reducer.first(),
    geometry=sample_point,
    scale=10
).getInfo()
print(f"\nSample baseline mean at (91.0, 24.8): {stats}")

stats_std = baseline_std.reduceRegion(
    reducer=ee.Reducer.first(),
    geometry=sample_point,
    scale=10
).getInfo()
print(f"Sample baseline std at (91.0, 24.8): {stats_std}")


# %%
# =============================================================================
# STEP 4: COMPUTE Z-SCORE FLOOD MASKS FOR EACH MONITORING DATE
# =============================================================================
# Why: This is the core of the time series analysis. For each Sentinel-1 image
#      during the flood season, we compute the Z-score and classify flooded pixels.
#
# What happens for each image:
#   1. Take the VV backscatter image for that date
#   2. Subtract the baseline mean → anomaly (how different from normal?)
#   3. Divide by baseline std → Z-score (how many std devs below normal?)
#   4. If Z < threshold → classify as flooded
#   5. Apply slope mask to remove false positives on hillsides
#
# The result is a stack of binary flood masks — one per S-1 acquisition date.
# Each pixel is either 1 (flooded) or 0 (not flooded) on each date.

def compute_zscore_flood_mask(image, baseline_mean, baseline_std, z_threshold, slope_mask):
    """
    Compute Z-score flood mask for a single Sentinel-1 image.

    Parameters:
    -----------
    image : ee.Image — single S-1 VV backscatter image
    baseline_mean : ee.Image — per-pixel mean from dry season
    baseline_std : ee.Image — per-pixel standard deviation from dry season
    z_threshold : float — Z-score cutoff (e.g., -2.5)
    slope_mask : ee.Image — binary mask (1 = flat terrain, 0 = steep)

    Returns:
    --------
    ee.Image with bands:
      - 'zscore': the Z-score value
      - 'flood': binary flood mask (1=flooded, 0=not flooded)
    """
    # Mask border noise
    masked = image.updateMask(image.gt(-30))

    # Compute Z-score: (observed - mean) / std
    zscore = masked.subtract(baseline_mean).divide(baseline_std).rename("zscore")

    # Classify: Z < threshold = flooded
    flood = zscore.lt(z_threshold).rename("flood")

    # Apply slope mask (only keep flood detections on flat terrain)
    flood = flood.updateMask(slope_mask)

    # Apply focal mode smoothing to remove salt-and-pepper noise
    # radius=60m covers ~6 pixels at 10m resolution
    flood_clean = flood.focal_mode(radius=60, units="meters").rename("flood_clean")

    # Preserve the timestamp
    return flood_clean.copyProperties(image, ["system:time_start"])


# Prepare slope mask from SRTM
slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
slope_mask = slope.lt(SLOPE_MAX)  # 1 where flat, 0 where steep

# Get the monitoring collection
monitor_col = get_s1_collection(roi, MONITOR_START, MONITOR_END)
monitor_col = monitor_col.map(lambda img: img.updateMask(img.gt(-30)))

# Apply Z-score flood detection to each image
print("Computing Z-score flood masks for each monitoring date...")
flood_masks = monitor_col.map(
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
# Why: The first-wet-date map is YOUR MOST IMPORTANT FIGURE.
#      It shows, for each pixel, the FIRST date that pixel was classified as
#      flooded during the 2025 monsoon season.
#
# If haors are connected:
#   → You'll see a spatial gradient: upstream/river areas flood first (early dates),
#     then flooding spreads outward into adjacent haors (later dates).
#     The map will show a smooth color transition from early (blue) to late (red).
#
# If haors are isolated:
#   → Each haor will flood independently based on local rainfall, not propagation.
#     The map will look patchy — adjacent haors might flood at very different times
#     with no spatial pattern.
#
# Additional metrics we compute:
#   - total_wet_count: number of dates each pixel was wet (= duration proxy)
#   - wet_fraction: proportion of dates wet (= flood frequency)
#   - total_flood_area_per_date: total flooded area across the region per date

def compute_first_wet_date(flood_mask_collection):
    """
    Compute per-pixel first-wet-date from a collection of binary flood masks.

    Logic: For each pixel, find the earliest image date where flood=1.
    We do this by converting each image's timestamp to a "days since epoch"
    band, masking it where flood=0, and taking the minimum across all dates.
    """
    def add_date_band(img):
        """Add a band with the image date as days since 2025-01-01."""
        # Get the date as milliseconds since epoch
        date_ms = img.date().millis()
        # Convert to days since 2025-01-01
        epoch_ms = ee.Date("2025-01-01").millis()
        days = ee.Number(date_ms).subtract(epoch_ms).divide(86400000)  # ms to days
        # Create a constant image with this date value
        date_img = ee.Image.constant(days).float().rename("date_days")
        # Mask: only keep date where flood = 1
        return date_img.updateMask(img.select("flood_clean"))

    # Apply to each flood mask
    date_images = flood_mask_collection.map(add_date_band)

    # First wet date = minimum date across all images (earliest flood detection)
    first_wet = date_images.min().rename("first_wet_date")

    return first_wet


def compute_flood_metrics(flood_mask_collection):
    """
    Compute per-pixel flood duration and frequency metrics.
    """
    # Total number of dates this pixel was classified as wet
    total_wet = flood_mask_collection.select("flood_clean").sum().rename("total_wet_count")

    # Total number of valid observations (not masked)
    total_obs = flood_mask_collection.select("flood_clean").count().rename("total_obs_count")

    # Wet fraction = proportion of dates that were wet
    wet_fraction = total_wet.divide(total_obs).rename("wet_fraction")

    return total_wet, total_obs, wet_fraction


def compute_area_time_series(flood_mask_collection, roi, scale=100):
    """
    Compute total flooded area (km²) for each date in the collection.
    This gives you the flood hydrograph in area-space.

    Parameters:
    -----------
    scale : int — pixel size in meters for area calculation (100m is fast, 10m is precise)
    """
    def get_flood_area(img):
        """Compute flooded area in km² for a single image."""
        # Pixel area in m²
        pixel_area = img.select("flood_clean").multiply(ee.Image.pixelArea())
        # Sum over ROI
        area_m2 = pixel_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi,
            scale=scale,
            maxPixels=1e12
        ).get("flood_clean")
        # Convert m² to km²
        area_km2 = ee.Number(area_m2).divide(1e6)
        return img.set("flood_area_km2", area_km2)

    return flood_mask_collection.map(get_flood_area)


print("Computing first-wet-date map...")
first_wet_date = compute_first_wet_date(flood_masks)
print("✓ First-wet-date map computed")

print("\nComputing flood duration and frequency metrics...")
total_wet, total_obs, wet_fraction = compute_flood_metrics(flood_masks)
print("✓ Flood metrics computed")

print("\nComputing area time series (this may take a moment)...")
flood_masks_with_area = compute_area_time_series(flood_masks, roi, scale=100)

# Extract the area values
area_list = flood_masks_with_area.aggregate_array("flood_area_km2").getInfo()
date_list = flood_masks_with_area.aggregate_array("system:time_start").getInfo()

# Convert to readable format
area_dates = []
for d, a in zip(date_list, area_list):
    date_str = datetime.utcfromtimestamp(d/1000).strftime("%Y-%m-%d")
    area_dates.append({"date": date_str, "area_km2": round(a, 1) if a else 0})

print("\n--- Flood Area Time Series ---")
print(f"{'Date':<14} {'Area (km²)':>12}")
print("-" * 28)
for item in sorted(area_dates, key=lambda x: x["date"]):
    print(f"{item['date']:<14} {item['area_km2']:>12.1f}")

print(f"\n✓ Area time series extracted for {len(area_dates)} dates")


# %%
# =============================================================================
# STEP 6: VISUALIZE THE RESULTS
# =============================================================================
# Why: You need to SEE the results to understand what they mean.
#      The first-wet-date map should show a spatial pattern if haors are connected.
#      The area time series should show the flood hydrograph shape.

# --- Interactive Map ---
print("\nCreating interactive map...")
m = geemap.Map()
m.centerObject(roi, 8)

# Add the first-wet-date map
# Color scale: blue (early flood, ~day 100=Apr) → red (late flood, ~day 270=Sep)
# Day 0 = Jan 1, so April 1 = day 90, May 15 = day 134, Sep 30 = day 272
first_wet_vis = {
    "min": 90,    # April 1
    "max": 270,   # September 30
    "palette": [
        "0000FF",  # Blue = earliest flooding (April)
        "00FFFF",  # Cyan = May
        "00FF00",  # Green = June
        "FFFF00",  # Yellow = July
        "FF8800",  # Orange = August
        "FF0000",  # Red = latest flooding (September)
    ]
}
m.addLayer(first_wet_date.clip(roi), first_wet_vis, "First Wet Date (day of year)", True)

# Add total wet count (flood duration)
duration_vis = {
    "min": 0,
    "max": 20,  # adjust based on number of monitoring images
    "palette": ["FFFFFF", "C6DBEF", "6BAED6", "2171B5", "08306B"]
}
m.addLayer(total_wet.clip(roi), duration_vis, "Total Wet Count", False)

# Add wet fraction (flood frequency)
frequency_vis = {
    "min": 0,
    "max": 1,
    "palette": ["FFFFFF", "FED976", "FEB24C", "FD8D3C", "FC4E2A", "E31A1C", "B10026"]
}
m.addLayer(wet_fraction.clip(roi), frequency_vis, "Wet Fraction", False)

# Add a specific flood mask for peak event (~June 1, 2025)
# Find the image closest to June 1
peak_date = "2025-06-01"
peak_mask = flood_masks.filterDate("2025-05-28", "2025-06-05")
n_peak = peak_mask.size().getInfo()
if n_peak > 0:
    peak_mosaic = peak_mask.select("flood_clean").max()  # Union of all flood masks in window
    m.addLayer(peak_mosaic.selfMask().clip(roi), {"palette": ["FF0000"]}, f"Peak Flood (~{peak_date})", False)
    print(f"  Added peak flood layer ({n_peak} images near {peak_date})")

# Add baseline mean for reference
m.addLayer(baseline_mean.clip(roi), {"min": -20, "max": -5}, "Baseline Mean VV (dB)", False)

# Add ROI outline
m.addLayer(ee.Image().paint(roi, 1, 2), {"palette": ["FFFFFF"]}, "Study Area", True)

# Add a legend for first-wet-date
# (geemap may support legend; if not, you'll see the color ramp in the layer controls)

print("✓ Map created — display it by running 'm' in a Jupyter cell")
print("\nLayer guide:")
print("  - 'First Wet Date': YOUR KEY FIGURE — color = when each pixel first flooded")
print("    Blue=April, Cyan=May, Green=June, Yellow=July, Orange=Aug, Red=Sep")
print("  - 'Total Wet Count': how many S-1 dates each pixel was classified as wet")
print("  - 'Wet Fraction': proportion of monitoring dates that were wet")
print("  - 'Peak Flood': flood extent around June 1 (peak of 2025 event)")

# Display the map (in Jupyter, this will render the interactive map)
m


# %%
# =============================================================================
# STEP 7: EXPORT RESULTS (OPTIONAL BUT RECOMMENDED)
# =============================================================================
# Why: Exporting to GeoTIFF lets you:
#   - Work with the data in QGIS or Python (matplotlib/rasterio) offline
#   - Overlay with MERIT Hydro HAND and FABDEM later (Part 2)
#   - Share results with your supervisor
#   - Not have to re-run GEE computations every time
#
# We export at 100m resolution (not 10m) to keep file sizes manageable.
# For publication figures, you can re-export specific areas at 10m.

EXPORT_SCALE = 100  # meters per pixel
EXPORT_CRS = "EPSG:4326"

# --- Export first-wet-date map ---
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
# Uncomment the next line to actually start the export:
# task_fwd.start()
# print("✓ First-wet-date export started → check Google Drive 'haor_flood_analysis' folder")

# --- Export wet fraction map ---
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

# --- Export total wet count ---
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
print("Export tasks are defined but NOT started yet.")
print("To start exports, uncomment the task.start() lines above and re-run this cell.")
print("Exports will appear in your Google Drive under 'haor_flood_analysis' folder.")
print(f"At {EXPORT_SCALE}m resolution, each file will be ~50-200 MB for the full haor region.")


# %%
# =============================================================================
# STEP 8: INDIVIDUAL DATE FLOOD MAPS (BROWSE THROUGH TIME)
# =============================================================================
# Why: Sometimes you want to look at individual flood maps to understand
#      what happened on specific dates (e.g., before vs. during vs. after peak).
#      This creates a slider widget to browse through dates interactively.

def create_date_slider_map(flood_masks, roi, monitor_dates):
    """
    Create a map with a date slider to browse individual flood masks.
    This is useful for visual inspection and understanding flood progression.
    """
    m2 = geemap.Map()
    m2.centerObject(roi, 8)

    # Add a selection of key dates as separate layers
    # Pick ~5-8 dates spanning the flood season for manual inspection
    key_dates = []
    if len(monitor_dates) > 8:
        # Pick evenly spaced dates
        step = len(monitor_dates) // 7
        key_dates = [monitor_dates[i] for i in range(0, len(monitor_dates), step)]
    else:
        key_dates = monitor_dates

    for date_str in key_dates[:10]:  # Limit to 10 layers for performance
        # Filter to this date (allow 1-day window)
        start = date_str
        end_dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)
        end = end_dt.strftime("%Y-%m-%d")

        date_mask = flood_masks.filterDate(start, end)
        n = date_mask.size().getInfo()
        if n > 0:
            mosaic = date_mask.select("flood_clean").max()
            m2.addLayer(
                mosaic.selfMask().clip(roi),
                {"palette": ["FF0000"]},
                f"Flood {date_str} (n={n})",
                False  # Not visible by default — toggle in layer panel
            )

    # Add first-wet-date as reference
    m2.addLayer(first_wet_date.clip(roi), first_wet_vis, "First Wet Date", True)

    # Add ROI
    m2.addLayer(ee.Image().paint(roi, 1, 2), {"palette": ["FFFFFF"]}, "Study Area", True)

    return m2

print("Creating date browser map...")
m2 = create_date_slider_map(flood_masks, roi, monitor_dates)
print("✓ Date browser map created — display with 'm2' in Jupyter")
print("  Toggle individual date layers on/off in the layer control panel")


# %%
# =============================================================================
# STEP 9: PRINT SUMMARY AND SAVE AREA TIME SERIES TO CSV
# =============================================================================
# Why: Having the area time series as a CSV lets you plot it in matplotlib
#      or Excel, and you'll need it for the area-stage analysis in Part 2.

import csv

csv_filename = "haor_flood_area_timeseries_2025.csv"

# Sort by date
area_dates_sorted = sorted(area_dates, key=lambda x: x["date"])

# Write CSV
with open(csv_filename, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "area_km2"])
    writer.writeheader()
    writer.writerows(area_dates_sorted)

print(f"✓ Area time series saved to {csv_filename}")

# Print summary statistics
areas = [x["area_km2"] for x in area_dates_sorted if x["area_km2"] > 0]
if areas:
    print(f"\n--- Flood Area Summary ---")
    print(f"  Number of observation dates: {len(area_dates_sorted)}")
    print(f"  Dates with detected flooding: {len(areas)}")
    print(f"  Maximum flooded area: {max(areas):.1f} km²")
    print(f"  Date of maximum: {area_dates_sorted[[x['area_km2'] for x in area_dates_sorted].index(max(areas))]['date']}")
    print(f"  Minimum flooded area (when >0): {min(areas):.1f} km²")


# %%
# =============================================================================
# STEP 10: WHAT TO DO NEXT (PART 2 PREVIEW)
# =============================================================================
# 
# You now have:
#   ✓ First-wet-date map — shows flood propagation pattern
#   ✓ Total wet count — shows flood duration per pixel
#   ✓ Wet fraction — shows flood frequency per pixel
#   ✓ Area time series — shows total flooded area over time (CSV)
#   ✓ Individual date flood masks — for visual inspection
#
# In Part 2, we will:
#   1. Run CoCoAH to detect individual haor boundaries from a pre-monsoon image
#   2. Use those boundaries to compute per-haor area time series
#   3. Overlay MERIT Hydro HAND to identify barriers between haors
#   4. Overlay FABDEM (30m) to estimate barrier crest elevations
#   5. Overlay OSM embankment/road data
#   6. Select 2-3 corridors for SWOT WSE transect analysis
#
# For now, look at your first-wet-date map and answer these questions:
#   - Do you see a spatial gradient? (blue on one side, red on the other?)
#   - Do adjacent haors flood at similar times? (similar colors?)
#   - Can you see any linear features where the flood timing changes abruptly?
#     (These might be embankments or roads acting as barriers)
#
# These visual observations will guide your corridor selection in Part 2.
#
print("\n" + "="*60)
print("PART 1 COMPLETE")
print("="*60)
print("""
What you should do now:
1. Run this notebook and inspect the first-wet-date map
2. Look for spatial patterns in flood propagation
3. Note any abrupt boundaries in flood timing (potential embankments)
4. Save/export your results for use in Part 2
5. Take screenshots of interesting patterns for your next D-Lab presentation

Key figures to show your supervisor:
  - First-wet-date map (your primary connectivity evidence)
  - Area time series plot (flood hydrograph shape)
  - Individual date comparison (before vs. during vs. after peak)
""")
