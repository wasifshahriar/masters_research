#!/usr/bin/env python3
"""
==============================================================================
ROUND 3, STEP 1: ONE FIXED SENTINEL-1 WATER DATASET (r3_1_export_water.py)
==============================================================================
Runs on the server, starts Google Earth Engine exports to Google Drive, then exits.

What it makes (one file per hydrological year, April to March, 2019 to 2025):
  Drive/haor_water20m/water20m_YYYY.tif         20 m open-water maps over the five haors
                                                 one band per radar date, 1 water, 0 dry, 255 no data
  Drive/haor_water20m/water20m_meta_YYYY.csv    one row per date: orbit, angle, the two thresholds,
                                                 water share inside the haors
  Drive/haor_water100m/flood_stack_cluster_YYYY.tif
                                                 the same water at 100 m on the SAME grid and band
                                                 names as your published stacks, so the main
                                                 notebook can later run on it unchanged

The method (why it fixes the 45 / 32 degree problem) is written at the top of r3_gee.py.

Usage:  cd /work/a06/wasif/haor_validation_package/round3
        python r3_1_export_water.py --test      # 1 to 2 minutes: checks two October 2024 dates
        python r3_1_export_water.py             # starts 21 exports (7 years x 3 files)
Then:   when all tasks in https://code.earthengine.google.com/tasks are COMPLETED, download
        Drive/haor_water20m   ->  /work/a06/wasif/haor_water20m/
        Drive/haor_water100m  ->  /work/a06/wasif/haor_water100m/
==============================================================================
"""
import sys
import r3_common as C
import r3_gee as G
ee = G.ee

G.init()
R20, R100 = G.rect(C.BOX20), G.rect(C.BOX100)
HAORS = G.haor_geometry()


def per_date(date_str):
    water, info = G.classify(date_str, R100, R20)     # thresholds from the haors + 1.5 km box (land and water)
    stats = water.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), "", True), HAORS, 40,
                               maxPixels=1e9, bestEffort=True)
    tag = ee.String("w").cat(ee.Date(date_str).format("YYYYMMdd"))
    band = water.unmask(255).clip(R100).toUint8().rename(tag)
    feat = ee.Feature(None, info).set("water_share_haors", stats.get("water_mean"),
                                      "pixels_haors", stats.get("water_count"))
    return ee.List([band, feat])


if "--test" in sys.argv:
    for d in ["2024-10-20", "2024-10-22"]:
        f = ee.Feature(per_date(d).get(1))
        print(" ", f.getInfo()["properties"])
    print("Expected: orbit 114 (angle ~45) and orbit 150 (angle ~33), water_share_haors close to each other\n"
          "(about 0.85 to 0.95 on both days; the published maps gave about 0.72 and 0.31).")
    sys.exit(0)

for y in C.YEARS:
    start, end = G.year_window(y)
    dates = G.s1_dates(start, end, R100)
    n = dates.size().getInfo()
    pairs = dates.map(per_date)
    bands = ee.ImageCollection(pairs.map(lambda p: ee.List(p).get(0)))
    feats = ee.FeatureCollection(pairs.map(lambda p: ee.List(p).get(1)))
    stack20 = bands.toBands().clip(R20)
    # 100 m: share of 20 m water pixels in each 100 m cell; water if half or more; no data if less than half observed
    def to100(img):
        img = ee.Image(img)
        w = img.updateMask(img.neq(255))
        frac = w.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(crs="EPSG:4326", scale=100)
        obs = img.neq(255).reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(crs="EPSG:4326", scale=100)
        tag = ee.String("f").cat(ee.String(img.bandNames().get(0)).slice(1))
        return frac.gte(0.5).where(obs.lt(0.5), 255).unmask(255).toUint8().rename(tag)
    stack100 = ee.ImageCollection(bands.toList(500).map(lambda i: ee.Image(i).reproject(crs="EPSG:4326", scale=20))).map(to100).toBands()
    ee.batch.Export.image.toDrive(image=stack20, description=f"water20m_{y}", folder="haor_water20m",
                                  fileNamePrefix=f"water20m_{y}", region=R20, scale=20, crs="EPSG:4326",
                                  maxPixels=1e11).start()
    ee.batch.Export.image.toDrive(image=stack100, description=f"water100m_{y}", folder="haor_water100m",
                                  fileNamePrefix=f"flood_stack_cluster_{y}", region=R100, scale=100,
                                  crs="EPSG:4326", maxPixels=1e11).start()
    ee.batch.Export.table.toDrive(collection=feats, description=f"water20m_meta_{y}", folder="haor_water20m",
                                  fileNamePrefix=f"water20m_meta_{y}", fileFormat="CSV",
                                  selectors=["date", "orbit", "direction", "angle", "t_vv", "t_vh",
                                             "water_share_haors", "pixels_haors"]).start()
    print(f"  {y}: {n} dates -> 3 exports started")

print("\n✓ 21 exports started. Check https://code.earthengine.google.com/tasks")
print("When all are COMPLETED, download:\n  Drive/haor_water20m  -> /work/a06/wasif/haor_water20m/\n"
      "  Drive/haor_water100m -> /work/a06/wasif/haor_water100m/\nThen run: python r3_2_check_s2.py")
