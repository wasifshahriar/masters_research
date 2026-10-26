#!/usr/bin/env python3
"""
==============================================================================
HAOR VALIDATION IMAGES  (haor_export_validation_images.py)
==============================================================================
Why this exists (meeting, 1 October, Prof. Yamazaki): "Check the actual image.
One day before merging, after merging: how exactly does the image look? Show
that to convince others." Until now every connectivity figure was drawn from
the 100 m flood flags. This script exports the REAL satellite images for the
handful of key dates the notebook picks, so the merge can be checked by eye
against the radar backscatter itself and, where the sky allows, an optical
image.

For each date in validation/key_dates.csv (written by haor_visual_validation.ipynb,
cell V1) it exports, clipped to the haor cluster:
  <label>_S1_<YYYYMMDD>.tif     Sentinel-1 VV and VH backscatter, dB x 100 (int16),
                                20 m, same-day scenes mosaicked
  <label>_FL20_<YYYYMMDD>.tif   the SAME flood flag as production (pooled baseline,
                                z < -2.5, slope mask, 60 m focal mode) but kept at
                                20 m: 1 flooded, 0 dry, 255 unobserved
  <label>_S2_<YYYYMMDD>.tif     Sentinel-2 surface reflectance, cloud masked with
                                SCL, median of scenes within +-WINDOW days,
                                B4 B3 B2 B8 B11 at 20 m (uint16). Only exported
                                when some clear pixels exist (monsoon is cloudy)
Once, for the embankment question:
  dry_S2_10m.tif                cloud-free dry season composite (Jan to Mar 2024),
                                B4 B3 B2 B8 at 10 m, to look for embankments and
                                measure their width against the 100 m grid
A metadata table s2_availability.csv is printed and saved locally.

Drive folder: haor_validation_images. Download it to
  /work/a06/wasif/haor_flood_analysis_hydroyear/analysis/validation/images/

Usage:  python haor_export_validation_images.py
        python haor_export_validation_images.py --dry-run    (lists, starts nothing)
==============================================================================
"""
import sys, os, argparse, datetime

venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")
os.environ["HOME"] = "/home/wasif"

import pandas as pd
import ee

# ============================ SETTINGS ============================
ANALYSIS_ROOT = "/work/a06/wasif/haor_flood_analysis_hydroyear/analysis"
KEY_DATES_CSV = os.path.join(ANALYSIS_ROOT, "validation", "key_dates.csv")
GDRIVE_FOLDER = "haor_validation_images"
EXPORT_BBOX   = [90.93, 24.91, 91.34, 25.22]     # W, S, E, N: the five haors plus margin
POOL_YEARS    = [2019, 2025]                     # baseline years, identical to the production export
Z_THRESHOLD   = -2.5
SLOPE_MAX     = 5
SCALE_S1      = 20
SCALE_S2      = 20
S2_WINDOW     = 7          # days either side of the radar date for the optical median
S2_WINDOW_MAX = 15         # widened once if nothing clear is found
S2_MIN_CLEAR  = 0.10       # export the optical image only if this share of the box is clear
# =================================================================

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

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

assert os.path.exists(KEY_DATES_CSV), (
    f"missing {KEY_DATES_CSV}. Run haor_visual_validation.ipynb cell V1 first.")
kd = pd.read_csv(KEY_DATES_CSV)
kd["date"] = pd.to_datetime(kd["date"]).dt.strftime("%Y-%m-%d")
print(f"{len(kd)} key dates, {kd['date'].nunique()} unique")


def s1(roi, start, end, pol="VV"):
    """Identical to haor_export_hydroyear.py."""
    return (ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(roi).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol))
            .select(pol))


# ---- the pooled baseline, built exactly as in haor_export_hydroyear.py ----
baseline_all = (s1(region, f"{POOL_YEARS[0]}-01-01", f"{POOL_YEARS[-1]}-04-01")
                .filter(ee.Filter.calendarRange(1, 3, "month"))
                .map(lambda i: i.updateMask(i.gt(-30))))
b_mean = baseline_all.mean().rename("VV")
b_std = baseline_all.reduce(ee.Reducer.stdDev()).rename("VV")
b_std = b_std.where(b_std.lte(0), 0.5)          # kept identical to production on purpose


def flood_mask(image):
    """Identical to haor_export_hydroyear.py flood_mask()."""
    m = image.updateMask(image.gt(-30))
    z = m.subtract(b_mean).divide(b_std)
    f = z.lt(Z_THRESHOLD).rename("flood_clean")
    f = f.updateMask(slope_mask)
    f = f.focal_mode(radius=60, units="meters").rename("flood_clean")
    return f.copyProperties(image, ["system:time_start"])


def s2_composite(date, window):
    d = ee.Date(date)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(d.advance(-window, "day"), d.advance(window + 1, "day")))
    def clean(img):
        scl = img.select("SCL")
        ok = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(1))
        return img.updateMask(ok)
    comp = col.map(clean).median().select(["B4", "B3", "B2", "B8", "B11"])
    clear = comp.select("B4").mask().reduceRegion(ee.Reducer.mean(), region, 200, maxPixels=1e9)
    return comp, col.size(), clear


def start(img, name, scale):
    if args.dry_run:
        print(f"  [dry-run] {name}")
        return
    ee.batch.Export.image.toDrive(image=img.clip(region), description=name[:100],
        folder=GDRIVE_FOLDER, fileNamePrefix=name, region=region, scale=scale,
        crs="EPSG:4326", maxPixels=1e10).start()
    print(f"  started: {name}")


meta = []
for (date, label), _ in kd.groupby(["date", "label"]):
    ymd = date.replace("-", "")
    nxt = (datetime.date.fromisoformat(date) + datetime.timedelta(days=1)).isoformat()
    print(f"\n{label}  {date}")
    vv = s1(region, date, nxt, "VV")
    n_vv = vv.size().getInfo()
    if n_vv == 0:
        print("  no Sentinel-1 scene on this date, skipped")
        meta.append(dict(label=label, date=date, s1_scenes=0)); continue
    vh = s1(region, date, nxt, "VH")
    vvm = vv.mosaic().updateMask(vv.mosaic().gt(-30))
    vhm = ee.Image(ee.Algorithms.If(vh.size().gt(0), vh.mosaic(), ee.Image.constant(-99).rename("VH")))
    s1img = (ee.Image.cat([vvm.rename("VV"), ee.Image(vhm).rename("VH")])
             .multiply(100).round().unmask(-32768).toInt16())
    start(s1img, f"{label}_S1_{ymd}", SCALE_S1)

    fl = vv.map(flood_mask).select("flood_clean").max()
    start(fl.unmask(255).toUint8(), f"{label}_FL20_{ymd}", SCALE_S1)

    comp, nsc, clear = s2_composite(date, S2_WINDOW)
    n, cf = nsc.getInfo(), (clear.getInfo() or {}).get("B4", 0) or 0
    win = S2_WINDOW
    if cf < S2_MIN_CLEAR:
        comp, nsc, clear = s2_composite(date, S2_WINDOW_MAX)
        n, cf = nsc.getInfo(), (clear.getInfo() or {}).get("B4", 0) or 0
        win = S2_WINDOW_MAX
    print(f"  Sentinel-1 scenes {n_vv} | Sentinel-2 scenes {n} within +-{win} d, clear share {cf:.2f}")
    if cf >= S2_MIN_CLEAR:
        start(comp.unmask(0).toUint16(), f"{label}_S2_{ymd}", SCALE_S2)
    meta.append(dict(label=label, date=date, s1_scenes=n_vv, s2_scenes=n,
                     s2_window_days=win, s2_clear_share=round(cf, 3),
                     s2_exported=cf >= S2_MIN_CLEAR))

# ---- once: a dry season optical composite at 10 m, for embankments ----
dry, ndry, cdry = s2_composite("2024-02-15", 45)
start(dry.select(["B4", "B3", "B2", "B8"]).unmask(0).toUint16(), "dry_S2_10m", 10)

out = pd.DataFrame(meta)
os.makedirs(os.path.dirname(KEY_DATES_CSV), exist_ok=True)
out.to_csv(os.path.join(os.path.dirname(KEY_DATES_CSV), "s2_availability.csv"), index=False)
print("\n" + out.to_string(index=False))
print(f"\n✓ exports started -> Drive/{GDRIVE_FOLDER}")
print("  download the folder to " + os.path.join(ANALYSIS_ROOT, "validation", "images") + "/")
