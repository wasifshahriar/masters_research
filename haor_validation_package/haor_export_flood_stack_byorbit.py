#!/usr/bin/env python3
"""
==============================================================================
HAOR FLOOD STACKS WITH ONE DRY BASELINE PER RELATIVE ORBIT
(haor_export_flood_stack_byorbit.py)
==============================================================================
Round 2, step 1. Runs on the server, starts Google Earth Engine exports to
Google Drive (the same way you made haor_flood_pooled_30m), then exits. GEE
takes a few hours to finish them.

One run starts BOTH versions (7 years each, 14 tasks):
  Drive folder haor_flood_byorbit_100m        run A: -30 dB floor MASKED (as production)
  Drive folder haor_flood_byorbit_100m_clamp  run B: -30 dB floor CLAMPED to -30 dB

Why (validation round 1):
  * Two families of Sentinel-1 passes see very different amounts of water a
    few days apart. On clear Sentinel-2 dates one family misses about half of
    the open water in the haor cores.
  * Production compares every pass with ONE pooled dry-season baseline (all
    orbits together). Two orbits see the same field at different incidence
    angles, so the pooled mean sits between them and the pooled spread is
    inflated: water is missed on one geometry. Here each pass is compared with
    dry-season images from its OWN relative orbit (the DeVries et al. 2020
    rule, also used in the Ninh Binh scripts).
  * On the other family the calmest water falls below -30 dB. Production MASKS
    it (it becomes "unobserved" and is filled from earlier dates). Run B sets
    it to -30 dB instead, so it stays observed and counts as water.

Everything else is identical to production: same collection, slope mask,
z < -2.5, 60 m focal mode, same-day maximum, hydrological year (April to
March), 100 m grid and box. Output is uint8 (1 flooded, 0 dry, 255
unobserved), so haor_validation_labels.py reads it without changes.

Usage:  cd /work/a06/wasif/haor_validation_package
        python haor_export_flood_stack_byorbit.py
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
SCALE         = 100                      # metres, production grid
YEARS         = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
POOL_YEARS    = [2019, 2025]             # baseline years, identical to production
EXPORT_BBOX   = [90.85, 24.85, 91.50, 25.25]
Z_THRESHOLD   = -2.5
SLOPE_MAX     = 5
MIN_BASE_IMGS = 8                        # an orbit needs this many dry-season images for its own baseline
RUNS          = {"mask": "haor_flood_byorbit_100m",          # FLOOR_MODE -> Drive folder
                 "clamp": "haor_flood_byorbit_100m_clamp"}
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


def s1(start, end):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .select(["VV"]))


def floor(img, mode):
    """-30 dB floor: 'mask' = production (below -30 dB becomes unobserved), 'clamp' = set to -30 dB."""
    out = img.max(-30) if mode == "clamp" else img.updateMask(img.gt(-30))
    return ee.Image(out.copyProperties(img, img.propertyNames()))      # keep the orbit number


def mean_std(col):
    mean = col.mean().rename(["VV"])
    std = col.reduce(ee.Reducer.stdDev()).rename(["VV"])
    std = std.where(std.lte(0), 0.5)                 # identical to production
    return mean.addBands(std.rename(["VV_sd"]))


def baselines(mode):
    """Dry-season (Jan to Mar) baseline per relative orbit, plus the pooled one as fallback."""
    col = (s1(f"{POOL_YEARS[0]}-01-01", f"{POOL_YEARS[-1]}-04-01")
           .filter(ee.Filter.calendarRange(1, 3, "month"))
           .map(lambda i: floor(i, mode)))
    pooled = mean_std(col)
    counts = col.aggregate_histogram("relativeOrbitNumber_start").getInfo()
    per = {}
    for o, n in counts.items():
        o = int(float(o))
        if n >= MIN_BASE_IMGS:
            per[str(o)] = mean_std(col.filter(ee.Filter.eq("relativeOrbitNumber_start", o)))
        else:
            print(f"  orbit {o}: only {n} dry-season images, uses the pooled baseline")
    print(f"  [{mode}] dry-season images per orbit: { {int(float(k)): v for k, v in counts.items()} }")
    return ee.Dictionary(per), pooled


def export_run(mode, folder):
    BASE, pooled = baselines(mode)

    def flood_mask(image):
        orb = ee.Number(image.get("relativeOrbitNumber_start")).format("%d")
        b = ee.Image(ee.Algorithms.If(BASE.contains(orb), BASE.get(orb), pooled))
        m = floor(image, mode)
        z = m.select("VV").subtract(b.select("VV")).divide(b.select("VV_sd"))
        f = z.lt(Z_THRESHOLD).rename("flood_clean").updateMask(slope_mask)
        f = f.focal_mode(radius=60, units="meters").rename("flood_clean")
        return f.copyProperties(image, ["system:time_start", "relativeOrbitNumber_start"])

    for year in YEARS:
        monitor = s1(f"{year}-04-01", f"{year + 1}-04-01").map(flood_mask)
        dated = monitor.map(lambda img: img.set(
            "date_str", ee.Date(img.get("system:time_start")).format("YYYYMMdd")))
        dates = dated.aggregate_array("date_str").distinct().sort()

        def comp_one(d):
            d = ee.String(d)
            day = dated.filter(ee.Filter.eq("date_str", d))
            return day.select("flood_clean").max().unmask(255).rename(ee.String("f").cat(d)).toUint8()

        stack = ee.ImageCollection(dates.map(comp_one)).toBands().clip(region)
        name = f"flood_stack_cluster_{year}"
        ee.batch.Export.image.toDrive(image=stack, description=f"{name}_{folder}",
                                      folder=folder, fileNamePrefix=name, region=region, scale=SCALE,
                                      crs="EPSG:4326", maxPixels=1e11).start()
        print(f"  [{mode}] started {name}  ({dates.size().getInfo()} dates) -> Drive/{folder}")


for mode, folder in RUNS.items():
    print(f"\n=== run {mode}: one baseline per relative orbit, -30 dB floor {mode} ===")
    export_run(mode, folder)

print("\n✓ 14 exports started. Check https://code.earthengine.google.com/tasks")
print("When all are COMPLETED, download each Drive folder to the server:")
for folder in RUNS.values():
    print(f"  Drive/{folder}/  ->  /work/a06/wasif/{folder}/")
