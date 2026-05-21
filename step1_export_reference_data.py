#!/usr/bin/env python3
"""
==============================================================================
STEP 1: PRE-DIGITIZATION REFERENCE DATA EXPORT
==============================================================================
Exports reference imagery and layers for the Tanguar-Matian-Shanir AOI
that you'll use in QGIS to manually digitize haor boundaries.

What this produces:
  - Sentinel-2 dry-season composite (Jan-Mar) — see the haor floor
  - Sentinel-2 wet-season composite (Jun-Aug) — see the haor maximum extent
  - Sentinel-1 dry vs wet difference — shows seasonal water
  - JRC water occurrence — historical frequency
  - FABDEM elevation hillshade — see the bowl shape
  - HAND classes — see drainage relationships

All exports go to Google Drive, then you download them to your QGIS folder.

Why each layer matters for digitizing:
  - S2 dry-season: shows permanent beels clearly (water = dark blue)
  - S2 wet-season: shows the full haor extent during monsoon
  - JRC occurrence: tells you the "natural" boundary based on 36 years
  - FABDEM hillshade: reveals the bowl shape — your haor outline should
                      follow contour lines around the bowl rim
  - HAND classes: HAND < 1m = haor floor, HAND 1-3m = transition zone
==============================================================================
"""

import sys
import os
venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["HOME"] = "/home/wasif"
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")

import ee

# --- GEE Init (same pattern as your export script) ---
try:
    ee.Initialize(project="79803231644")
    print("✓ GEE initialized")
except Exception:
    import os
    from google.oauth2.credentials import Credentials
    credentials = Credentials(
        token=None,
        refresh_token=os.environ["GEE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GEE_CLIENT_ID"],
        client_secret=os.environ["GEE_CLIENT_SECRET"],
    )
    ee.Initialize(credentials=credentials, project=os.environ.get("GEE_PROJECT", ""))
    print("✓ GEE initialized from environment credentials")


# ============================================================================
# AOI: TANGUAR-MATIAN-SHANIR HAOR CLUSTER
# ============================================================================
# Centered on the cluster, with buffer for context.
# Tanguar Haor center: ~91.05°E, 25.07°N
# Matian Haor:         ~91.12°E, 24.95°N
# Shanir Haor:         ~91.20°E, 25.00°N
#
# Bounding box covers all three with ~5km buffer
TANGUAR_BBOX = [90.85, 24.85, 91.50, 25.25]  # [W, S, E, N]

aoi = ee.Geometry.Rectangle(TANGUAR_BBOX)
print(f"AOI: {TANGUAR_BBOX}")
print(f"AOI area: {aoi.area().getInfo()/1e6:.1f} km²")

GDRIVE_FOLDER = "haor_digitization_tanguar"


# ============================================================================
# LAYER 1: SENTINEL-2 DRY-SEASON COMPOSITE (Jan-Mar 2025)
# ============================================================================
# Shows the permanent water bodies clearly. Cloud-free dry season.
# Use this to digitize BEEL boundaries (permanent water cores).
print("\n=== Layer 1: Sentinel-2 dry-season composite ===")

s2_dry = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2025-01-01", "2025-03-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
    .median()
    .clip(aoi))

# True-color RGB + NIR for water detection
s2_dry_rgb = s2_dry.select(["B4", "B3", "B2"]).divide(10000)  # Red, Green, Blue
s2_dry_nir = s2_dry.select(["B8", "B4", "B3"]).divide(10000)  # False color (NIR=red)
print(f"  Dry season images: {s2_dry.bandNames().getInfo()[:5]}...")


# ============================================================================
# LAYER 2: SENTINEL-2 WET-SEASON COMPOSITE (Jun-Aug 2024)
# ============================================================================
# Shows haor maximum seasonal extent. Note: monsoon has clouds so we use 2024
# (full season) and take cloud-masked composite.
print("\n=== Layer 2: Sentinel-2 wet-season composite ===")

def mask_s2_clouds(image):
    qa = image.select("QA60")
    cloud_bit = 1 << 10
    cirrus_bit = 1 << 11
    mask = qa.bitwiseAnd(cloud_bit).eq(0).And(qa.bitwiseAnd(cirrus_bit).eq(0))
    return image.updateMask(mask)

s2_wet = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterBounds(aoi)
    .filterDate("2024-06-01", "2024-08-31")
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
    .map(mask_s2_clouds)
    .median()
    .clip(aoi))

s2_wet_rgb = s2_wet.select(["B4", "B3", "B2"]).divide(10000)
s2_wet_nir = s2_wet.select(["B8", "B4", "B3"]).divide(10000)


# ============================================================================
# LAYER 3: SEASONAL WATER DIFFERENCE (S1 dry - S1 wet)
# ============================================================================
# Where backscatter dropped significantly between dry and wet seasons.
# Strong signal for seasonal haor flooding.
print("\n=== Layer 3: Sentinel-1 seasonal difference ===")

s1_dry = (ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(aoi)
    .filterDate("2025-01-01", "2025-03-31")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .select("VV").median().clip(aoi))

s1_wet = (ee.ImageCollection("COPERNICUS/S1_GRD")
    .filterBounds(aoi)
    .filterDate("2024-06-01", "2024-08-31")
    .filter(ee.Filter.eq("instrumentMode", "IW"))
    .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
    .select("VV").median().clip(aoi))

s1_seasonal_change = s1_dry.subtract(s1_wet).rename("seasonal_change_dB")


# ============================================================================
# LAYER 4: JRC WATER OCCURRENCE (clipped to AOI)
# ============================================================================
print("\n=== Layer 4: JRC water occurrence ===")
jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").clip(aoi)


# ============================================================================
# LAYER 5: FABDEM ELEVATION (for hillshade in QGIS)
# ============================================================================
print("\n=== Layer 5: FABDEM elevation ===")
fabdem = (ee.ImageCollection("projects/sat-io/open-datasets/FABDEM")
    .filterBounds(aoi).mosaic().clip(aoi))


# ============================================================================
# LAYER 6: MERIT HYDRO HAND
# ============================================================================
print("\n=== Layer 6: MERIT Hydro HAND ===")
merit = ee.Image("MERIT/Hydro/v1_0_1")
hand = merit.select("hnd").clip(aoi)


# ============================================================================
# EXPORT ALL LAYERS
# ============================================================================
print(f"\n=== Exporting to Google Drive folder: {GDRIVE_FOLDER} ===")

exports = {
    "tanguar_s2_dry_rgb": (s2_dry_rgb.float(), 10),
    "tanguar_s2_dry_nir": (s2_dry_nir.float(), 10),
    "tanguar_s2_wet_rgb": (s2_wet_rgb.float(), 10),
    "tanguar_s2_wet_nir": (s2_wet_nir.float(), 10),
    "tanguar_s1_seasonal_change": (s1_seasonal_change.float(), 10),
    "tanguar_jrc_occurrence": (jrc.float(), 30),
    "tanguar_fabdem": (fabdem.float(), 30),
    "tanguar_hand": (hand.float(), 90),
}

for name, (image, scale) in exports.items():
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=name,
        folder=GDRIVE_FOLDER,
        fileNamePrefix=name,
        region=aoi,
        scale=scale,
        crs="EPSG:4326",
        maxPixels=1e10,
    )
    task.start()
    print(f"  Started: {name} ({scale}m)")

print(f"\n✓ {len(exports)} exports started.")
print(f"  Wait for them to finish at code.earthengine.google.com → Tasks")
print(f"  Then download to /work/a06/wasif/haor_flood_analysis/digitization/")
