#!/usr/bin/env python3
"""
==============================================================================
HAOR FLOOD — MULTI-YEAR EXPORT (cluster only)  haor_export_multiyear.py
==============================================================================
Purpose: export ONLY the files the multi-year cluster analysis needs, for each
year you ask for, into ONE new Google Drive folder so nothing overlaps with the
single-year run. No CoCoAH, no barriers, no regional CSVs — those are not used
by the multi-year notebook, so they are deliberately left out to keep the
workflow clean.

For EACH year in YEARS this exports three things (all year-tagged):
  1. flood_stack_cluster_<year>.tif  — per-DATE flood stack for the 5-haor
     cluster, sub-scenes composited to one band per date (1=flood, 0=dry,
     nan=unobserved). Feeds the hydrographs AND the connectivity.
  2. first_wet_date_<year>.tif       — onset map (per-pixel earliest wet day).
  3. last_wet_date_<year>.tif        — offset map (per-pixel latest wet day).

Everything goes to ONE Drive folder: haor_flood_analysis_multiyear
Download that whole folder into a server folder of the same name:
  /work/a06/wasif/haor_flood_analysis_multiyear/

Baseline (dry-season Z-score reference) is rebuilt PER YEAR from that year's
Jan–Mar, so each year is self-contained. Monitor window is Apr 1 – Dec 31 of
the year, which (confirmed for 2025) reaches the full dry-down.

Usage:
    python haor_export_multiyear.py
    # edit YEARS below to control which years run.
==============================================================================
"""
import sys, os
venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")
os.environ["HOME"] = "/home/wasif"

import ee

# ============================ SETTINGS ============================
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]   # edit as needed
GDRIVE_FOLDER = "haor_flood_analysis_multiyear"       # ONE folder, year-tagged files
CLUSTER_BBOX  = [90.85, 24.85, 91.50, 25.25]          # 5-haor Tanguar cluster (W,S,E,N)
EXPORT_SCALE  = 100                                    # meters
Z_THRESHOLD   = -2.5
SLOPE_MAX     = 5
# =================================================================

# --- GEE init (env-var credentials; never hardcode) ---
try:
    ee.Initialize(project="79803231644")
    print("\u2713 GEE initialized")
except Exception:
    from google.oauth2.credentials import Credentials
    cred = Credentials(
        token=None,
        refresh_token=os.environ["GEE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GEE_CLIENT_ID"],
        client_secret=os.environ["GEE_CLIENT_SECRET"],
    )
    ee.Initialize(credentials=cred, project=os.environ.get("GEE_PROJECT", "79803231644"))
    print("\u2713 GEE initialized from environment credentials")

cluster = ee.Geometry.Rectangle(CLUSTER_BBOX)
slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003"))
slope_mask = slope.lt(SLOPE_MAX)


def s1(roi, start, end, pol="VV"):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
            .select(pol))


def export_one_year(year):
    """Build flood masks for one year over the cluster and export stack + maps."""
    print(f"\n=== {year} ===")
    baseline = s1(cluster, f"{year}-01-01", f"{year}-03-31").map(
        lambda i: i.updateMask(i.gt(-30)))
    b_mean = baseline.mean()
    b_std = baseline.reduce(ee.Reducer.stdDev())
    b_std = b_std.where(b_std.lte(0), 0.5)

    def flood_mask(image):
        m = image.updateMask(image.gt(-30))
        z = m.subtract(b_mean).divide(b_std)
        f = z.lt(Z_THRESHOLD).rename("flood_clean")
        f = f.updateMask(slope_mask)
        f = f.focal_mode(radius=60, units="meters").rename("flood_clean")
        return f.copyProperties(image, ["system:time_start"])

    monitor = s1(cluster, f"{year}-04-01", f"{year}-12-31").map(flood_mask)
    n = monitor.size().getInfo()
    print(f"  flood masks: {n}")
    if n == 0:
        print(f"  \u26a0 no S1 data for {year} — skipped")
        return

    # ---- onset / offset maps (days since Jan 1 of that year) ----
    def date_band(img):
        days = ee.Number(img.date().millis()).subtract(
            ee.Date(f"{year}-01-01").millis()).divide(86400000)
        return ee.Image.constant(days).float().rename("d").updateMask(
            img.select("flood_clean"))

    first_wet = monitor.map(date_band).min().rename("first_wet_date")
    last_wet  = monitor.map(date_band).max().rename("last_wet_date")

    # ---- per-DATE composited stack (one band per acquisition date) ----
    dated = monitor.map(lambda img: img.set(
        "date_str", ee.Date(img.get("system:time_start")).format("YYYYMMdd")))
    dates = dated.aggregate_array("date_str").distinct().sort()

    def comp_one(d):
        d = ee.String(d)
        day = dated.filter(ee.Filter.eq("date_str", d))
        return day.select("flood_clean").max().rename(ee.String("f").cat(d)).toFloat()

    stack = ee.ImageCollection(dates.map(comp_one)).toBands()

    # ---- exports (all year-tagged, one folder) ----
    jobs = {
        f"flood_stack_cluster_{year}": stack,
        f"first_wet_date_{year}": first_wet.clip(cluster).float(),
        f"last_wet_date_{year}": last_wet.clip(cluster).float(),
    }
    for name, img in jobs.items():
        ee.batch.Export.image.toDrive(
            image=img, description=name, folder=GDRIVE_FOLDER, fileNamePrefix=name,
            region=cluster, scale=EXPORT_SCALE, crs="EPSG:4326", maxPixels=1e10).start()
        print(f"  started: {name}")


for y in YEARS:
    export_one_year(y)

print(f"\n\u2713 All exports started -> Drive/{GDRIVE_FOLDER}")
print("  When done, download the WHOLE folder to:")
print("  /work/a06/wasif/haor_flood_analysis_multiyear/")
print("  Tip: check band names once with rasterio .descriptions (expect f<date> per band).")
