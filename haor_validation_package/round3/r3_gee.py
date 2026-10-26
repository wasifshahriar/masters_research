"""
Round 3 Google Earth Engine helpers (r3_gee.py). Imported by steps 1, 2 and 4.

THE FIX FOR THE BRIGHT-WATER PASSES (tested on your own S1 and S2 images, see the round-3 doc):
  * Use both radar channels. VH (cross-polarised) water stays very dark (about -30 dB) at both
    viewing angles, so it is far less affected by the 45 / 32 degree problem than VV.
  * Give every image its own threshold, found from that image's own histogram (Otsu's method),
    instead of one rule for all images. Calm water at 45 degrees and wind-roughened water at
    32 degrees then both fall below their image's threshold.
  * Keep the thresholds inside limits learned from Sentinel-2:
        VV between -20 and -14 dB,  VH between -26 and -22 dB
    so a dry-season image with very little water cannot push the threshold onto dry fields.
  * Smooth radar speckle with a 3 x 3 (about 60 m) median before thresholding.
  * Water = VV below its threshold OR VH below its threshold.
  * Map OPEN WATER (rivers, beels and flood water together), not a change from the dry season.
    Connection between haors runs through the rivers, so the rivers must be in the map.
  * The -30 dB floor is clamped, not masked: the calmest water counts as water.
"""
import os, sys, datetime
import r3_common as C

if C.VENV_SITE not in sys.path and os.path.isdir(C.VENV_SITE):
    sys.path.insert(0, C.VENV_SITE)
os.environ.setdefault("HOME", "/home/wasif")
import ee

T_VV = (-20.0, -14.0)
T_VH = (-26.0, -22.0)
SMOOTH_M = 30            # focal median radius in metres (square kernel, about 60 m wide)
Z_OLD = -2.5             # the published (old) detector, kept only for the before/after check


def init():
    try:
        ee.Initialize(project="79803231644")
    except Exception:
        from google.oauth2.credentials import Credentials
        cred = Credentials(token=None, refresh_token=os.environ["GEE_REFRESH_TOKEN"],
                           token_uri="https://oauth2.googleapis.com/token",
                           client_id=os.environ["GEE_CLIENT_ID"], client_secret=os.environ["GEE_CLIENT_SECRET"])
        ee.Initialize(credentials=cred, project=os.environ.get("GEE_PROJECT", "79803231644"))
    print("✓ GEE initialized")


def rect(box):
    return ee.Geometry.Rectangle(box, "EPSG:4326", False)


def haor_geometry():
    """The five haor outlines as one ee.Geometry (simplified to 20 m to keep requests small)."""
    h = C.load_haors().to_crs(C.UTM)
    u = h.geometry.union_all() if hasattr(h.geometry, "union_all") else h.geometry.unary_union
    import geopandas as gpd
    g = gpd.GeoSeries([u.simplify(20)], crs=C.UTM).to_crs("EPSG:4326").iloc[0]
    return ee.Geometry(g.__geo_interface__)


def slope_ok():
    return ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003")).lt(5)


def s1(start, end, region):
    return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region).filterDate(start, end)
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH")))


def s1_dates(start, end, region):
    col = s1(start, end, region)
    return col.aggregate_array("system:time_start").map(
        lambda t: ee.Date(t).format("YYYY-MM-dd")).distinct().sort()


def otsu(histogram):
    """Otsu threshold from an ee.Reducer.histogram() dictionary (server side)."""
    h = ee.Dictionary(histogram)
    counts = ee.Array(h.get("histogram"))
    means = ee.Array(h.get("bucketMeans"))
    size = means.length().get([0])
    total = counts.reduce(ee.Reducer.sum(), [0]).get([0])
    total_sum = means.multiply(counts).reduce(ee.Reducer.sum(), [0]).get([0])
    mean = total_sum.divide(total)

    def between(i):
        i = ee.Number(i)
        a_counts = counts.slice(0, 0, i)
        a_count = a_counts.reduce(ee.Reducer.sum(), [0]).get([0])
        a_mean = means.slice(0, 0, i).multiply(a_counts).reduce(ee.Reducer.sum(), [0]).get([0]).divide(a_count.max(1))
        b_count = total.subtract(a_count)
        b_mean = total_sum.subtract(a_count.multiply(a_mean)).divide(b_count.max(1))
        v = a_count.multiply(a_mean.subtract(mean).pow(2)).add(b_count.multiply(b_mean.subtract(mean).pow(2)))
        return ee.Number(ee.Algorithms.If(a_count.gt(0).And(b_count.gt(0)), v, 0))

    keys = ee.List.sequence(1, size).map(between)
    return means.sort(keys).get([-1])


def day_image(date_str, region):
    """All Sentinel-1 slices of one UTC day, mosaicked, kept at the native 10 m grid."""
    d = ee.Date(date_str)
    col = s1(d, d.advance(1, "day"), region)
    first = ee.Image(col.first())
    img = col.mosaic().setDefaultProjection(first.select("VV").projection())
    return img, first


def classify(date_str, region, hist_region):
    """Open water for one date: 1 water, 0 dry, masked = no data. Returns (image, info dict)."""
    img, first = day_image(date_str, region)
    vv = img.select("VV").max(-30).focal_median(SMOOTH_M, "square", "meters")
    vh = img.select("VH").max(-35).focal_median(SMOOTH_M, "square", "meters")
    red = ee.Reducer.histogram(255, 0.1)
    hv = vv.reduceRegion(red, hist_region, 40, maxPixels=1e9, bestEffort=True).get("VV")
    hh = vh.reduceRegion(red, hist_region, 40, maxPixels=1e9, bestEffort=True).get("VH")
    t_vv = ee.Number(ee.Algorithms.If(hv, otsu(hv), -17)).max(T_VV[0]).min(T_VV[1])
    t_vh = ee.Number(ee.Algorithms.If(hh, otsu(hh), -24)).max(T_VH[0]).min(T_VH[1])
    water = vv.lt(t_vv).Or(vh.lt(t_vh)).updateMask(slope_ok()).rename("water")
    angle = img.select("angle").reduceRegion(ee.Reducer.mean(), hist_region, 1000, bestEffort=True).get("angle")
    info = dict(date=date_str, orbit=first.get("relativeOrbitNumber_start"),
                direction=first.get("orbitProperties_pass"), angle=angle, t_vv=t_vv, t_vh=t_vh)
    return water, info


def old_flag(date_str, region):
    """The published detector (pooled dry baseline, z < -2.5, -30 dB masked), for the before/after check."""
    base = (s1("2019-01-01", "2025-04-01", region).filter(ee.Filter.calendarRange(1, 3, "month"))
            .select("VV").map(lambda i: i.updateMask(i.gt(-30))))
    mean = base.mean()
    sd = base.reduce(ee.Reducer.stdDev()); sd = sd.where(sd.lte(0), 0.5)
    img, _ = day_image(date_str, region)
    vv = img.select("VV").updateMask(img.select("VV").gt(-30))
    z = vv.subtract(mean).divide(sd)
    return z.lt(Z_OLD).updateMask(slope_ok()).focal_mode(60, "circle", "meters").rename("old")


def s2_near(date_str, region, before=2, after=2):
    """Least-cloudy Sentinel-2 mosaic from 2 days before to 2 days after (default), clouds and shadows removed (SCL).
    Bands: B4, B3, B2 (true colour) and ndwi = (B3 - B8) / (B3 + B8); water = ndwi > 0."""
    d = ee.Date(date_str)
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate(d.advance(-before, "day"), d.advance(after + 1, "day")))

    def prep(i):
        scl = i.select("SCL")
        ok = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
        nd = i.normalizedDifference(["B3", "B8"]).rename("ndwi")
        return i.select(["B4", "B3", "B2"]).toFloat().addBands(nd).updateMask(ok)

    mos = col.map(prep).sort("CLOUDY_PIXEL_PERCENTAGE", False).mosaic()
    dates = col.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY-MM-dd")).distinct()
    return mos, col.size(), dates


def clear_fraction(s2img, region, scale=40):
    return s2img.select("ndwi").mask().unmask(0).reduceRegion(
        ee.Reducer.mean(), region, scale, maxPixels=1e9, bestEffort=True).get("ndwi")


def download(img, box, scale, path):
    """Download a small clip as GeoTIFF (for the evidence cards). Skips if already cached."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    import requests
    img = ee.Image(img).unmask(-9999)          # masked pixels (cloud, outside image) -> -9999, read back as NaN
    url = img.getDownloadURL({"region": rect(box), "scale": scale, "crs": "EPSG:4326",
                              "format": "GEO_TIFF", "filePerBand": False})
    r = requests.get(url, timeout=300)
    r.raise_for_status()
    with open(path, "wb") as f:
        f.write(r.content)
    return path


def year_window(y):
    return f"{y}-04-01", f"{y + 1}-04-01"
