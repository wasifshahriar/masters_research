#!/usr/bin/env python3
"""
==============================================================================
SENTINEL-1 PASS CATALOGUE FOR THE HAOR CLUSTER  (haor_s1_pass_catalog.py)
==============================================================================
Round 2, step 2. Runs on the server, talks to Google Earth Engine, about
10 to 20 minutes. No Google Drive step: the table is written straight to

  /work/a06/wasif/haor_flood_analysis_hydroyear/analysis/validation/next_checks/s1_pass_catalog.csv

One row per Sentinel-1 scene over the cluster, 1 April 2019 to 31 March 2026:
  date, time_utc     UTC date and time (the same date the flood stacks use)
  year, doy          analysis year (April to March) and days since 1 January
                     of that year, the same numbering as the stacks
  platform           A or B (Sentinel-1A or 1B)
  relative_orbit     the track; one track = one viewing geometry
  direction          ASCENDING or DESCENDING
  slice              slice number (one pass can have two slices)
  cover_frac         share of the cluster box inside the scene
  inc_mean           mean incidence angle over the cluster box (degrees)
  vv_perm_med        median VV (dB) over permanent water (JRC occurrence >= 90)
  vv_core_med        median VV (dB) over the haor-core box

Why: the round-1 notebook found two families of passes that see very different
amounts of water days apart. This table names each family by its real orbit,
and haor_next_checks.ipynb (cells N9 and N11) reads it from next_checks/.

Usage:  cd /work/a06/wasif/haor_validation_package
        python haor_s1_pass_catalog.py
==============================================================================
"""
import sys, os, csv, time, datetime

venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")
os.environ["HOME"] = "/home/wasif"

import ee

# ============================ SETTINGS ============================
CLUSTER_BBOX = [90.85, 24.85, 91.50, 25.25]    # production export box
CORE_BOX     = [91.00, 24.97, 91.25, 25.15]    # haor cores (same box as N10)
START, END   = "2019-04-01", "2026-04-01"      # hydrological years 2019 to 2025
OUT_DIR      = "/work/a06/wasif/haor_flood_analysis_hydroyear/analysis/validation/next_checks"
OUT_CSV      = os.path.join(OUT_DIR, "s1_pass_catalog.csv")
COLS = ["date", "time_utc", "year", "doy", "platform", "relative_orbit", "direction", "slice",
        "cover_frac", "inc_mean", "vv_perm_med", "vv_core_med"]
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

cluster = ee.Geometry.Rectangle(CLUSTER_BBOX)
core = ee.Geometry.Rectangle(CORE_BOX)
perm = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").gte(90)


def scenes(start, end):
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(cluster).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")))


def describe(img):
    t = ee.Date(img.get("system:time_start"))
    y = t.get("year"); m = t.get("month")
    ay = ee.Number(ee.Algorithms.If(m.gte(4), y, y.subtract(1)))           # analysis year, April to March
    doy = t.difference(ee.Date.fromYMD(ay, 1, 1), "day").floor()          # same numbering as the stacks
    cover = img.geometry().intersection(cluster, ee.ErrorMargin(100)).area(100).divide(cluster.area(100))
    vv = img.select("VV")
    inc = img.select("angle").reduceRegion(ee.Reducer.mean(), cluster, 500, maxPixels=1e9).get("angle")
    vperm = vv.updateMask(perm).reduceRegion(ee.Reducer.median(), cluster, 100, maxPixels=1e9).get("VV")
    vcore = vv.reduceRegion(ee.Reducer.median(), core, 100, maxPixels=1e9).get("VV")
    return ee.Feature(None, {
        "date": t.format("YYYY-MM-dd"), "time_utc": t.format("HH:mm:ss"),
        "year": ay, "doy": doy,
        "platform": img.get("platform_number"),
        "relative_orbit": img.get("relativeOrbitNumber_start"),
        "direction": img.get("orbitProperties_pass"),
        "slice": img.get("sliceNumber"),
        "cover_frac": cover, "inc_mean": inc, "vv_perm_med": vperm, "vv_core_med": vcore})


def fetch(start, end, tries=3):
    """Rows for one time window, computed on GEE and returned directly (no Drive)."""
    for k in range(tries):
        try:
            fc = ee.FeatureCollection(scenes(start, end).map(describe)).getInfo()
            return [f["properties"] for f in fc["features"]]
        except Exception as e:
            print(f"    {start} to {end}: try {k + 1} failed ({str(e)[:80]})")
            time.sleep(10)
    return None


def months(start, end):
    d = datetime.date.fromisoformat(start)
    stop = datetime.date.fromisoformat(end)
    while d < stop:
        n = datetime.date(d.year + (d.month == 12), d.month % 12 + 1, 1)
        yield d.isoformat(), min(n, stop).isoformat()
        d = n


os.makedirs(OUT_DIR, exist_ok=True)
rows = []
# three months per request; a window that fails three times is retried month by month
qs = list(months(START, END))
for i in range(0, len(qs), 3):
    a, b = qs[i][0], qs[min(i + 2, len(qs) - 1)][1]
    got = fetch(a, b)
    if got is None:
        got = []
        for ma, mb in qs[i:i + 3]:
            r = fetch(ma, mb)
            if r is None:
                print(f"  !! {ma} to {mb} failed; run the script again later")
                r = []
            got += r
    rows += got
    print(f"  {a} to {b}: {len(got)} scenes (total {len(rows)})")

rows.sort(key=lambda r: (str(r.get("date")), str(r.get("time_utc"))))
with open(OUT_CSV, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"\n✓ {len(rows)} scenes -> {OUT_CSV}")

# quick look: scenes per relative orbit and direction
by = {}
for r in rows:
    k = (r.get("relative_orbit"), r.get("direction"))
    by[k] = by.get(k, 0) + 1
for (o, d), n in sorted(by.items(), key=lambda x: (str(x[0][1]), x[0][0] or 0)):
    print(f"  orbit {o:>4} {str(d):10s} {n:4d} scenes")
print("Two or more orbits in the SAME direction would explain why the ascending/descending baseline did not help.")
