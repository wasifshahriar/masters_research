#!/usr/bin/env python3
"""
HAOR FLOOD - HYDROLOGICAL YEAR EXPORT  (haor_export_hydroyear.py)
Supersedes haor_export_multiyear.py

THREE METHOD CHANGES vs the previous version:
  1. ONE pooled baseline: dry season (Jan to Mar) of ALL years 2019 to 2025
     combined, giving one mean and one std per pixel, used as the Z-score
     reference for every year. Previously each year used its own baseline.
  2. Hydrological year window: 1 Apr of year Y to 31 Mar of year Y+1, so the
     recession completes inside the data instead of being cut at 31 Dec.
  3. The pooled mean doubles as the multi year permanent water map, so no
     per year baseline files are needed.

Exports to Drive folder haor_flood_analysis_hydroyear:
  baseline_mean_pooled.tif        1 file, the shared Z-score reference
  baseline_std_pooled.tif         1 file
  flood_stack_cluster_<Y>.tif     7 files, one band per date, 1=flood 0=dry nan=unseen
  first_wet_date_<Y>.tif          7 files, days since 1 Jan of year Y
  last_wet_date_<Y>.tif           7 files, same units
  = 23 files total
"""
import sys, os
venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")
os.environ["HOME"] = "/home/wasif"

import ee

# ============================ SETTINGS ============================
YEARS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
GDRIVE_FOLDER = "haor_flood_analysis_hydroyear"
CLUSTER_BBOX  = [90.85, 24.85, 91.50, 25.25]
EXPORT_SCALE  = 100
Z_THRESHOLD   = -2.5
SLOPE_MAX     = 5
# =================================================================

try:
    ee.Initialize(project="79803231644")
    print("\u2713 GEE initialized")
except Exception:
    from google.oauth2.credentials import Credentials
    cred = Credentials(token=None,
        refresh_token=os.environ["GEE_REFRESH_TOKEN"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.environ["GEE_CLIENT_ID"],
        client_secret=os.environ["GEE_CLIENT_SECRET"])
    ee.Initialize(credentials=cred, project=os.environ.get("GEE_PROJECT", "79803231644"))
    print("\u2713 GEE initialized from environment credentials")

cluster = ee.Geometry.Rectangle(CLUSTER_BBOX)
slope_mask = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).lt(SLOPE_MAX)

def s1(roi, start, end, pol="VV"):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
            .select(pol))

# ---------------- ONE POOLED BASELINE, ALL SEVEN DRY SEASONS ----------------
# filterDate spans 2019-01-01 to 2025-04-01, then calendarRange keeps only
# January, February and March, giving exactly seven dry seasons.
print("\n=== pooled baseline: Jan to Mar of all years ===")
baseline_all = (s1(cluster, f"{YEARS[0]}-01-01", f"{YEARS[-1]}-04-01")
                .filter(ee.Filter.calendarRange(1, 3, "month"))
                .map(lambda i: i.updateMask(i.gt(-30))))
n_base = baseline_all.size().getInfo()
print(f"  baseline images pooled across 7 dry seasons: {n_base}")

b_mean = baseline_all.mean().rename("VV")
b_std  = baseline_all.reduce(ee.Reducer.stdDev()).rename("VV")
b_std  = b_std.where(b_std.lte(0), 0.5)

# diagnostic: how much does pooling inflate the standard deviation?
one_year = (s1(cluster, f"{YEARS[-1]}-01-01", f"{YEARS[-1]}-04-01")
            .map(lambda i: i.updateMask(i.gt(-30))))
std_1y = one_year.reduce(ee.Reducer.stdDev()).rename("single")
stats = ee.Image.cat([b_std.rename("pooled"), std_1y]).reduceRegion(
    reducer=ee.Reducer.median(), geometry=cluster, scale=200, maxPixels=1e9).getInfo()
sp, ss = stats.get("pooled"), stats.get("single")
print(f"  median std, pooled 7 years : {sp:.3f} dB")
print(f"  median std, {YEARS[-1]} alone     : {ss:.3f} dB")
if sp and ss:
    print(f"  inflation factor           : {sp/ss:.2f}x")
    print(f"  a {3.0:.1f} dB drop scores z = {-3.0/sp:.2f} pooled vs {-3.0/ss:.2f} single year")

def flood_mask(image):
    """Z-score against the POOLED baseline, identical for every year."""
    m = image.updateMask(image.gt(-30))
    z = m.subtract(b_mean).divide(b_std)
    f = z.lt(Z_THRESHOLD).rename("flood_clean")
    f = f.updateMask(slope_mask)
    f = f.focal_mode(radius=60, units="meters").rename("flood_clean")
    return f.copyProperties(image, ["system:time_start"])

# ---------------- per year exports ----------------
def export_one_year(year):
    print(f"\n=== hydrological year {year} (1 Apr {year} to 31 Mar {year+1}) ===")
    monitor = s1(cluster, f"{year}-04-01", f"{year+1}-04-01").map(flood_mask)
    n = monitor.size().getInfo()
    print(f"  monitor images: {n}")
    if n == 0:
        print("  \u26a0 no S1 data, skipped"); return

    # days since 1 Jan of year Y, so January of Y+1 becomes ~370, not ~15
    def date_band(img):
        days = ee.Number(img.date().millis()).subtract(
            ee.Date(f"{year}-01-01").millis()).divide(86400000)
        return ee.Image.constant(days).float().rename("d").updateMask(
            img.select("flood_clean"))

    first_wet = monitor.map(date_band).min().rename("first_wet_date")
    last_wet  = monitor.map(date_band).max().rename("last_wet_date")

    dated = monitor.map(lambda img: img.set(
        "date_str", ee.Date(img.get("system:time_start")).format("YYYYMMdd")))
    dates = dated.aggregate_array("date_str").distinct().sort()

    def comp_one(d):
        d = ee.String(d)
        day = dated.filter(ee.Filter.eq("date_str", d))
        return day.select("flood_clean").max().rename(ee.String("f").cat(d)).toFloat()

    stack = ee.ImageCollection(dates.map(comp_one)).toBands()

    jobs = {
        f"flood_stack_cluster_{year}": stack,
        f"first_wet_date_{year}": first_wet.clip(cluster).float(),
        f"last_wet_date_{year}":  last_wet.clip(cluster).float(),
    }
    for name, img in jobs.items():
        ee.batch.Export.image.toDrive(
            image=img, description=name, folder=GDRIVE_FOLDER, fileNamePrefix=name,
            region=cluster, scale=EXPORT_SCALE, crs="EPSG:4326", maxPixels=1e10).start()
        print(f"  started: {name}")

# the shared baseline, exported once
for name, img in {"baseline_mean_pooled": b_mean.clip(cluster).float(),
                  "baseline_std_pooled":  b_std.clip(cluster).float()}.items():
    ee.batch.Export.image.toDrive(
        image=img, description=name, folder=GDRIVE_FOLDER, fileNamePrefix=name,
        region=cluster, scale=EXPORT_SCALE, crs="EPSG:4326", maxPixels=1e10).start()
    print(f"  started: {name}")

for y in YEARS:
    export_one_year(y)

print(f"\n\u2713 all exports started -> Drive/{GDRIVE_FOLDER}")
print("  expect 23 files: 2 baseline + 7 stacks + 7 onset + 7 offset")
print("  download the whole folder to /work/a06/wasif/haor_flood_analysis_hydroyear/")