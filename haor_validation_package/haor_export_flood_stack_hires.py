#!/usr/bin/env python3
"""
==============================================================================
HAOR FLOOD STACKS AT ANOTHER RESOLUTION OR BASELINE  (haor_export_flood_stack_hires.py)
==============================================================================
Why this exists (meeting, 1 October):
  1. Prof. Yamazaki asked how wide a typical embankment is compared with the
     pixel. The production stacks are 100 m. A submersible embankment is a few
     metres wide at the crest, so at 100 m it is invisible: water on both sides
     of a dry embankment reads as one continuous body. The connectivity result
     has to be shown to survive at a resolution closer to Sentinel-1's own.
  2. The professors assumed 10 m pixels, so "3 pixels" meant 30 m to them. The
     analysis is at 100 m (3 px = 300 m). Re-running at 30 m makes the
     resolution question answerable with data instead of argument.
  3. DeVries et al. (2020), whose detector the export follows, builds the
     baseline separately for each orbit direction (orbitProperties_pass). The
     production export pools ascending and descending passes into one baseline.
     BASELINE_MODE = "by_pass" reproduces the DeVries choice so the effect can be
     measured.

Everything else is identical to haor_export_hydroyear.py: same collection,
same -30 dB floor, same slope mask, same z < -2.5, same 60 m focal mode, same
same-day compositing, same hydrological-year window and day numbering.
Output is uint8 (1 flooded, 0 dry, 255 unobserved) to keep files small.

Recommended runs, one at a time (change the two settings, run, wait):
  A. SCALE = 30,  BASELINE_MODE = "pooled"   -> resolution test  (2023 to 2025)
  B. SCALE = 100, BASELINE_MODE = "by_pass"  -> baseline test    (all years)

Drive folder: haor_flood_<BASELINE_MODE>_<SCALE>m
Download to:  /work/a06/wasif/haor_flood_<BASELINE_MODE>_<SCALE>m/
Then run:     python haor_validation_labels.py --stack-dir <that folder> --res <SCALE>

Usage:  python haor_export_flood_stack_hires.py
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
SCALE         = 100                       # metres; 100 reproduces the production grid
BASELINE_MODE = "by_pass"                 # "pooled" (production) or "by_pass" (DeVries)
YEARS         = [2020, 2021, 2022, 2023, 2024, 2025]       # the SWOT years; use 2019 to 2025 for run B
POOL_YEARS    = [2019, 2025]             # baseline years, identical to production
EXPORT_BBOX   = [90.93, 24.91, 91.34, 25.22] if SCALE < 100 else [90.85, 24.85, 91.50, 25.25]
Z_THRESHOLD   = -2.5
SLOPE_MAX     = 5
GDRIVE_FOLDER = f"haor_flood_{BASELINE_MODE}_{SCALE}m"
# =================================================================

try:
    ee.Initialize(project="79803231644")
    print("✓ GEE initialized")
except Exception:
    from google.oauth2.credentials import Credentials
    cred = Credentials(token=None, refresh_token=os.environ["GEE_REFRESH_TOKEN"],
                       token_uri="https://oauth2.googleapis.com/token",
                       client_id=os.environ["GEE_CLIENT_ID"],
                       client_secret=os.environ["GEE_CLIENT_SECRET"])
    ee.Initialize(credentials=cred, project=os.environ.get("GEE_PROJECT", "79803231644"))
    print("✓ GEE initialized from environment credentials")

region = ee.Geometry.Rectangle(EXPORT_BBOX)
slope_mask = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).lt(SLOPE_MAX)


def s1(roi, start, end, pol="VV"):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
            .select(pol))


def baseline(direction=None):
    col = (s1(region, f"{POOL_YEARS[0]}-01-01", f"{POOL_YEARS[-1]}-04-01")
           .filter(ee.Filter.calendarRange(1, 3, "month")))
    if direction:
        col = col.filter(ee.Filter.eq("orbitProperties_pass", direction))
    col = col.map(lambda i: i.updateMask(i.gt(-30)))
    mean = col.mean().rename("VV")
    std = col.reduce(ee.Reducer.stdDev()).rename("VV")
    std = std.where(std.lte(0), 0.5)       # identical to production
    return mean, std, col.size()


if BASELINE_MODE == "pooled":
    b_mean, b_std, nb = baseline()
    print(f"pooled baseline images: {nb.getInfo()}")
else:
    mA, sA, nA = baseline("ASCENDING")
    mD, sD, nD = baseline("DESCENDING")
    print(f"baseline images: ascending {nA.getInfo()}, descending {nD.getInfo()}")


def flood_mask(image):
    m = image.updateMask(image.gt(-30))
    if BASELINE_MODE == "pooled":
        mean, std = b_mean, b_std
    else:
        asc = ee.String(image.get("orbitProperties_pass")).equals("ASCENDING")
        mean = ee.Image(ee.Algorithms.If(asc, mA, mD))
        std = ee.Image(ee.Algorithms.If(asc, sA, sD))
    z = m.subtract(mean).divide(std)
    f = z.lt(Z_THRESHOLD).rename("flood_clean")
    f = f.updateMask(slope_mask)
    f = f.focal_mode(radius=60, units="meters").rename("flood_clean")
    return f.copyProperties(image, ["system:time_start"])


def export_one_year(year):
    print(f"\n=== {year}: 1 Apr {year} to 31 Mar {year+1}, {SCALE} m, {BASELINE_MODE} ===")
    monitor = s1(region, f"{year}-04-01", f"{year+1}-04-01").map(flood_mask)
    n = monitor.size().getInfo()
    print(f"  monitor images: {n}")
    if n == 0:
        return
    dated = monitor.map(lambda img: img.set(
        "date_str", ee.Date(img.get("system:time_start")).format("YYYYMMdd")))
    dates = dated.aggregate_array("date_str").distinct().sort()

    def comp_one(d):
        d = ee.String(d)
        day = dated.filter(ee.Filter.eq("date_str", d))
        return day.select("flood_clean").max().unmask(255).rename(ee.String("f").cat(d)).toUint8()

    stack = ee.ImageCollection(dates.map(comp_one)).toBands().clip(region)
    name = f"flood_stack_cluster_{year}"
    ee.batch.Export.image.toDrive(image=stack, description=f"{name}_{BASELINE_MODE}_{SCALE}m",
        folder=GDRIVE_FOLDER, fileNamePrefix=name, region=region, scale=SCALE,
        crs="EPSG:4326", maxPixels=1e11).start()
    print(f"  started: {name}  ({dates.size().getInfo()} dates)")


for y in YEARS:
    export_one_year(y)
print(f"\n✓ exports started -> Drive/{GDRIVE_FOLDER}")
print(f"  download to /work/a06/wasif/{GDRIVE_FOLDER}/")
print(f"  then: python haor_validation_labels.py --stack-dir /work/a06/wasif/{GDRIVE_FOLDER} --res {SCALE}")
