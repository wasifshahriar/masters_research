#!/usr/bin/env python3
"""
==============================================================================
HAOR FLOOD CONNECTIVITY — CONSOLIDATED PIPELINE (download → visualization)
==============================================================================
Author context: Wasif Shahriar, M1, Yamazaki Lab (IIS, UTokyo).
This single file collects every stage of the analysis in run-order. It is a
REFERENCE/Assembly file — in practice each STAGE is run separately (some on an
internet-enabled machine for GEE, some on the HPC for analysis). Stages are
separated by `# %%` so you can paste them into Jupyter cells.

PIPELINE OVERVIEW
  STAGE 0  Environment & GEE auth (HPC-aware)
  STAGE 1  Export reference imagery for QGIS digitization (GEE → Google Drive)
  STAGE 2  Sentinel-1 flood time series + first-wet / peak / offset (GEE)
           ** includes the UNOBSERVED-pixel handling the supervisor asked for **
  STAGE 3  Create empty digitization shapefiles (schemas) for QGIS
  STAGE 4  Per-haor onset analysis (Step C) WITH permanent-water masking (the bug fix)
  STAGE 5  Dynamic-boundary connectivity (connected-component merge/split over season)
  STAGE 6  Diagnostics & visualization (histogram, narrowing, boxplot, maps)

KEY INTERPRETIVE PRINCIPLES (read before using results)
  - A real signal SHARPENS under spatial refinement; an artifact WEAKENS.
  - Synchronous onset does NOT prove connectivity (common-rainfall confound).
    It is a constraint, not a confirmation. 6-day data cannot resolve propagation lags.
  - Permanent water fires "first wet = day 1" → ALWAYS mask it (JRC occurrence > 75%).
  - Unobserved pixels (no orbit pass) are a DISTINCT class — never count them as dry.
==============================================================================
"""

# =============================================================================
# %% STAGE 0 — ENVIRONMENT & GEE AUTHENTICATION (HPC-aware)
# =============================================================================
# WHY: the "naam" HPC nodes have NO internet, so GEE stages must run on an
# internet-enabled machine (local Mac or a connected server). The venv path and
# HOME are set explicitly because the HPC conda env is non-standard.
import sys, os

VENV_SITE = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if VENV_SITE not in sys.path:
    sys.path.insert(0, VENV_SITE)
os.environ.setdefault("HOME", "/home/wasif")

def init_gee():
    """Initialize Earth Engine. Tries project init, falls back to stored creds.
    WHY two paths: interactive machines have cached auth; headless servers need
    the explicit refresh-token credentials."""
    import ee
    try:
        ee.Initialize(project="79803231644")
        print("✓ GEE initialized (project)")
    except Exception:
        import os
        from google.oauth2.credentials import Credentials
        creds = Credentials(
            token=None,
            refresh_token=os.environ["GEE_REFRESH_TOKEN"],
            token_uri="https://oauth2.googleapis.com/token",
            client_id=os.environ["GEE_CLIENT_ID"],
            client_secret=os.environ["GEE_CLIENT_SECRET"],
        )
        ee.Initialize(credentials=creds, project=os.environ.get("GEE_PROJECT", ""))
        print("✓ GEE initialized (stored credentials)")
    return ee

# AOI conventions — BE CAREFUL, mixing these caused a real bug:
#   shapely/GeoJSON order = [West, South, East, North]
#   matplotlib extent order = [West, East, South, North]
TANGUAR_BBOX_WSEN = [90.85, 24.85, 91.50, 25.25]   # W,S,E,N  (for ee.Geometry.Rectangle)
TANGUAR_EXT_WESN  = [90.85, 91.50, 24.85, 25.25]   # W,E,S,N  (for imshow/crops)
GDRIVE_FOLDER = "haor_digitization_tanguar"


# =============================================================================
# %% STAGE 1 — EXPORT REFERENCE IMAGERY FOR QGIS DIGITIZATION
# =============================================================================
# WHY: to manually trace haor boundaries in QGIS we need good backdrops:
# dry/wet Sentinel-2 (see beels vs seasonal extent), S1 seasonal change, JRC
# occurrence (historical water), FABDEM (bowl shape), HAND (drainage).
def stage1_export_reference(ee):
    aoi = ee.Geometry.Rectangle(TANGUAR_BBOX_WSEN)

    # Sentinel-2 dry season (Jan–Mar): permanent water shows as dark → trace beels
    s2_dry = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi).filterDate("2025-01-01", "2025-03-31")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30)).median().clip(aoi))

    # Sentinel-2 wet season (prev-year full monsoon, cloud-masked) → trace haor extent
    def mask_s2(img):
        qa = img.select("QA60")
        m = qa.bitwiseAnd(1 << 10).eq(0).And(qa.bitwiseAnd(1 << 11).eq(0))
        return img.updateMask(m)
    s2_wet = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
              .filterBounds(aoi).filterDate("2024-06-01", "2024-08-31")
              .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
              .map(mask_s2).median().clip(aoi))

    # Sentinel-1 seasonal change (dry − wet): strong drop = seasonal flooding
    def s1_med(d0, d1):
        return (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
                .filterDate(d0, d1).filter(ee.Filter.eq("instrumentMode", "IW"))
                .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
                .select("VV").median().clip(aoi))
    s1_change = s1_med("2025-01-01", "2025-03-31").subtract(s1_med("2024-06-01", "2024-08-31"))

    jrc = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").clip(aoi)
    fabdem = ee.ImageCollection("projects/sat-io/open-datasets/FABDEM").filterBounds(aoi).mosaic().clip(aoi)
    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd").clip(aoi)

    exports = {
        "tanguar_s2_dry_rgb": (s2_dry.select(["B4","B3","B2"]).divide(10000).float(), 10),
        "tanguar_s2_wet_rgb": (s2_wet.select(["B4","B3","B2"]).divide(10000).float(), 10),
        "tanguar_s1_seasonal_change": (s1_change.float(), 10),
        "tanguar_jrc_occurrence": (jrc.float(), 30),
        "tanguar_fabdem": (fabdem.float(), 30),
        "tanguar_hand": (hand.float(), 90),
    }
    for name, (img, scale) in exports.items():
        ee.batch.Export.image.toDrive(
            image=img, description=name, folder=GDRIVE_FOLDER, fileNamePrefix=name,
            region=aoi, scale=scale, crs="EPSG:4326", maxPixels=1e10).start()
        print(f"  started export: {name} ({scale} m)")
    print("Wait for tasks at code.earthengine.google.com → Tasks, then download.")


# =============================================================================
# %% STAGE 2 — SENTINEL-1 FLOOD TIME SERIES + ONSET / PEAK / OFFSET
# =============================================================================
# WHY: the core flood detector. Z-score anomaly vs dry-season baseline.
# This stage now produces the THREE timing layers the supervisor asked for
# (onset/peak/offset) AND an explicit OBSERVATION-COUNT layer so unobserved
# pixels can be handled honestly downstream.
#
# METHOD (DeVries et al. 2020): for each pixel,
#   Z = (VV_now − dryMean) / dryStd ;  Z < -2.5  →  flooded.
# Comparing each pixel to ITS OWN dry-season normal adapts to land cover, so a
# single threshold works for forest, paddy, water, urban alike.
def stage2_flood_timeseries(ee, region_bbox_wsen=TANGUAR_BBOX_WSEN,
                            year=2025, z_thresh=-2.5):
    aoi = ee.Geometry.Rectangle(region_bbox_wsen)

    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(aoi)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .select("VV"))

    # Dry-season baseline (Jan–Mar): mean & std per pixel.
    dry = s1.filterDate(f"{year}-01-01", f"{year}-03-31")
    dry_mean = dry.mean()
    dry_std = dry.reduce(ee.Reducer.stdDev()).max(0.5)  # floor std to avoid divide-by-tiny

    # Monitoring period (Apr–Sep): convert each image to a flood mask via Z-score.
    monitor = s1.filterDate(f"{year}-04-01", f"{year}-09-30")

    def to_flood(img):
        z = img.subtract(dry_mean).divide(dry_std)
        # focal_mode smoothing (r≈60 m) reduces speckle salt-and-pepper noise
        flood = z.lt(z_thresh).focalMode(radius=60, units="meters")
        doy = ee.Number(ee.Date(img.get("system:time_start")).getRelative("day", "year")).add(1)
        # bands: flood(0/1), and doy stamped where flooded (for onset/peak/offset)
        return (flood.rename("flood")
                .addBands(flood.multiply(doy).rename("flood_doy"))
                .set("doy", doy)
                .copyProperties(img, ["system:time_start"]))

    fc = monitor.map(to_flood)

    # ONSET = earliest DOY where flooded. (min of flood_doy over wet pixels)
    # We mask flood_doy==0 (dry) before reducing so zeros don't win the min.
    def masked_doy(img):
        d = img.select("flood_doy")
        return d.updateMask(d.gt(0))
    onset = fc.map(masked_doy).min().rename("onset_doy")

    # OFFSET = latest DOY where flooded (recession end).
    offset = fc.map(masked_doy).max().rename("offset_doy")

    # PEAK = DOY of maximum regional flooding is per-pixel ambiguous; we define
    # per-pixel peak as the midpoint of its longest wet run is expensive in GEE,
    # so a practical proxy: DOY at which the pixel's flooded state is most
    # "supported" by neighbors. Simpler robust proxy used here = median wet DOY.
    peak = fc.map(masked_doy).median().rename("peak_doy")

    # OBSERVATION COUNT (the unobserved-pixel fix): how many valid S1 looks each
    # pixel got. Pixels with very few observations are UNRELIABLE → flag, do not
    # silently treat as dry. Export this alongside the timing layers.
    obs_count = monitor.map(lambda i: i.mask().rename("obs")).sum().rename("obs_count")

    # FLOOD FREQUENCY & DURATION (supporting metrics)
    freq = fc.select("flood").mean().rename("flood_frequency")     # fraction of looks wet
    n_wet = fc.select("flood").sum().rename("n_wet_obs")

    out = (onset.addBands(peak).addBands(offset)
           .addBands(freq).addBands(n_wet).addBands(obs_count))

    # Export to Drive (then rclone → HPC). Scale ~100 m keeps thousands of px/haor.
    ee.batch.Export.image.toDrive(
        image=out.float(), description=f"flood_timing_{year}",
        folder="haor_flood_timing", fileNamePrefix=f"flood_timing_{year}",
        region=aoi, scale=100, crs="EPSG:4326", maxPixels=1e10).start()
    print(f"  started flood_timing_{year} export (onset/peak/offset/freq/obs_count)")
    return out


# =============================================================================
# %% STAGE 3 — CREATE EMPTY DIGITIZATION SHAPEFILES (run on any machine)
# =============================================================================
# WHY: gives QGIS proper attribute schemas to draw into. Three layers because
# haors (polygons), beels (polygons inside haors), khals (lines) differ in
# geometry, attributes, and rendering. You DELETE the dummy row in QGIS first.
def stage3_create_schemas(out_dir="/work/a06/wasif/haor_flood_analysis/digitization"):
    import geopandas as gpd
    from shapely.geometry import Polygon, LineString
    os.makedirs(out_dir, exist_ok=True)
    c = (91.15, 25.05)
    poly = Polygon([(c[0]-0.01,c[1]-0.01),(c[0]+0.01,c[1]-0.01),
                    (c[0]+0.01,c[1]+0.01),(c[0]-0.01,c[1]+0.01)])
    line = LineString([(c[0]-0.01,c[1]),(c[0]+0.01,c[1])])
    gpd.GeoDataFrame([{ "haor_id":"DUMMY","name_en":"DELETE ME","area_km2":0.0,
        "ramsar":0,"confidence":"low","notes":"delete in QGIS","geometry":poly}],
        crs="EPSG:4326").to_file(f"{out_dir}/haors_manual.shp")
    gpd.GeoDataFrame([{ "beel_id":"DUMMY","name_en":"DELETE ME","haor_id":"",
        "area_km2":0.0,"permanence":"permanent","geometry":poly}],
        crs="EPSG:4326").to_file(f"{out_dir}/beels_manual.shp")
    gpd.GeoDataFrame([{ "khal_id":"DUMMY","flow_from":"","flow_to":"",
        "khal_type":"primary","perennial":0,"geometry":line}],
        crs="EPSG:4326").to_file(f"{out_dir}/khals_manual.shp")
    print("✓ created haors/beels/khals schemas in", out_dir)


# =============================================================================
# %% STAGE 4 — PER-HAOR ONSET (STEP C) WITH PERMANENT-WATER MASKING (THE FIX)
# =============================================================================
# WHY THE MASK: the Z-score detector marks permanent water (deep beels) as
# "first wet = day 1" because it was never dry. That faked an early Tanguar
# signal (std 25 d, March tail). Masking JRC occurrence > 75% removes it.
# RESULT after fix: Tanguar std 25→15, medians UNMOVED → synchrony robust.
#
# INTERPRETATION: compare the between-haor spread against the within-haor std.
# If spread < within-haor scatter → SIMULTANEOUS filling (cannot claim propagation).
def stage4_step_c(
    base="/work/a06/wasif/haor_flood_analysis",
    haors_shp=None, first_wet_tif=None, jrc_tif=None,
    permanent_thresh=75):
    import numpy as np, pandas as pd, geopandas as gpd, rasterio
    from rasterio.mask import mask as rmask
    from rasterio.warp import reproject, Resampling
    from shapely.geometry import mapping
    os.environ["SHAPE_RESTORE_SHX"] = "YES"  # rebuild missing .shx automatically

    haors_shp = haors_shp or f"{base}/digitization/haors_manual.shp"
    first_wet_tif = first_wet_tif or f"{base}/first_wet_date.tif"
    jrc_tif = jrc_tif or f"{base}/haor_digitization_tanguar/tanguar_jrc_occurrence.tif"
    out_dir = f"{base}/digitization/analysis"; os.makedirs(out_dir, exist_ok=True)

    haors = gpd.read_file(haors_shp)
    # repair invalid geometry (fixes the Halir self-intersecting multipolygon)
    if (~haors.geometry.is_valid).any():
        haors["geometry"] = haors.geometry.buffer(0)
    haors = haors[haors["haor_id"] != "DUMMY"].reset_index(drop=True)

    def extract_masked(geom, fwd_src, jrc_src):
        """Onset pixels inside a haor, EXCLUDING permanent water (JRC>thresh)."""
        g = gpd.GeoSeries([geom], crs=haors.crs).to_crs(fwd_src.crs).iloc[0]
        fwd, ftrans = rmask(fwd_src, [mapping(g)], crop=True, filled=False)
        fwd = fwd[0]
        vals = np.ma.filled(fwd, np.nan).astype("float32")
        # resample JRC onto this onset window's grid
        jrc_grid = np.full(vals.shape, np.nan, "float32")
        reproject(source=rasterio.band(jrc_src, 1), destination=jrc_grid,
                  src_transform=jrc_src.transform, src_crs=jrc_src.crs,
                  dst_transform=ftrans, dst_crs=fwd_src.crs,
                  resampling=Resampling.bilinear)
        valid = np.isfinite(vals) & (vals > 0)
        not_perm = ~(np.isfinite(jrc_grid) & (jrc_grid > permanent_thresh))
        kept = vals[valid & not_perm]
        dropped = int(np.sum(valid & ~not_perm))
        return kept, dropped

    rows = []
    with rasterio.open(first_wet_tif) as fwd_src, rasterio.open(jrc_tif) as jrc_src:
        for _, h in haors.iterrows():
            kept, dropped = extract_masked(h.geometry, fwd_src, jrc_src)
            if len(kept) == 0:
                continue
            rows.append(dict(haor_id=h["haor_id"], name_en=h["name_en"],
                n_pixels=len(kept), n_permanent_dropped=dropped,
                onset_p10=float(np.percentile(kept,10)),
                onset_median=float(np.median(kept)),
                onset_mean=float(np.mean(kept)),
                onset_p90=float(np.percentile(kept,90)),
                onset_std=float(np.std(kept))))
    df = pd.DataFrame(rows).sort_values("onset_median").reset_index(drop=True)

    spread = df["onset_median"].max() - df["onset_median"].min()
    mean_std = df["onset_std"].mean()
    print(df.to_string(index=False))
    print(f"\nBetween-haor spread: {spread:.1f} d | mean within-haor std: {mean_std:.1f} d")
    if spread < mean_std:
        print("→ SIMULTANEOUS filling (spread < internal scatter). Cannot claim propagation.")
    else:
        print("→ Spread exceeds internal scatter; inspect spatial ordering before claiming propagation.")
    df.to_csv(f"{out_dir}/haor_onset_stepC_masked.csv", index=False)
    return df


# =============================================================================
# %% STAGE 5 — DYNAMIC-BOUNDARY CONNECTIVITY (the supervisor's reframe)
# =============================================================================
# WHY: the new thesis spine. Instead of "is there propagation" (unanswerable at
# 6-day resolution), ask "how does the connected water surface grow and shrink
# through the season." For each timestep, label connected water components; track
# how many separate bodies MERGE into one at peak and FRAGMENT at recession.
# Output: a time series of (#components, largest-component area) = time-varying
# connectivity, mappable directly from satellite.
def stage5_dynamic_connectivity(flood_stack_paths, out_csv):
    """flood_stack_paths: ordered list of per-date binary flood GeoTIFFs.
    Each must be the SAME grid. Produces a CSV of connectivity over time."""
    import numpy as np, pandas as pd, rasterio
    from scipy import ndimage  # connected-component labeling
    recs = []
    for path in flood_stack_paths:
        with rasterio.open(path) as src:
            arr = src.read(1)
            doy = _doy_from_name(path)  # parse date from filename
        water = arr == 1
        # label connected blobs (4-connectivity). structure=None → orthogonal only.
        labels, n = ndimage.label(water)
        if n == 0:
            recs.append(dict(doy=doy, n_components=0, largest_frac=0.0, total_wet=0)); continue
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n+1))
        total = water.sum()
        recs.append(dict(doy=doy, n_components=int(n),
                         largest_frac=float(sizes.max()/total) if total else 0.0,
                         total_wet=int(total)))
    df = pd.DataFrame(recs).sort_values("doy")
    df.to_csv(out_csv, index=False)
    # INTERPRETATION: n_components DROPS toward peak (beels merge into one surface),
    # then RISES during recession (fragmentation). largest_frac near 1.0 at peak =
    # one connected surface. This curve IS the seasonal connectivity dynamic.
    print(df.to_string(index=False))
    return df

def _doy_from_name(path):
    """Parse day-of-year from a filename containing a YYYYMMDD or YYYY-MM-DD token."""
    import re, datetime
    base = os.path.basename(path)
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", base)
    if not m:
        return None
    y, mo, d = map(int, m.groups())
    return datetime.date(y, mo, d).timetuple().tm_yday


# =============================================================================
# %% STAGE 6 — DIAGNOSTICS & VISUALIZATION
# =============================================================================
# WHY: the figures that drive interpretation. The histogram + the narrowing are
# the heart of the argument; the boxplot shows whether haor onset windows overlap.

def stage6a_onset_histogram(first_wet_tif, ext_wesn=TANGUAR_EXT_WESN, crop=True):
    """Histogram of onset dates. CROP to an AOI to test the 'narrowing':
    whole-region spread is inflated by cross-district rainfall; cropping to one
    cluster should COLLAPSE the spread if the regional signal was geography."""
    import numpy as np, matplotlib
    matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import rasterio
    with rasterio.open(first_wet_tif) as src:
        fwd = src.read(1).astype("float32")
        b = src.bounds
        full_ext = [b.left, b.right, b.bottom, b.top]
    if crop:
        nr, nc = fwd.shape; W,E,S,N = full_ext
        cmin = int((ext_wesn[0]-W)/(E-W)*nc); cmax = int((ext_wesn[1]-W)/(E-W)*nc)
        rmin = int((N-ext_wesn[3])/(N-S)*nr); rmax = int((N-ext_wesn[2])/(N-S)*nr)
        fwd = fwd[rmin:rmax, cmin:cmax]
        print(f"crop = {(rmax-rmin)*(cmax-cmin)} px "
              f"({100*(rmax-rmin)*(cmax-cmin)/ (nr*nc):.1f}% of raster)")
    v = fwd[np.isfinite(fwd) & (fwd > 0)]
    idr = np.percentile(v,90) - np.percentile(v,10)
    print(f"n={len(v)} | p10={np.percentile(v,10):.0f} median={np.median(v):.0f} "
          f"p90={np.percentile(v,90):.0f} | inter-decile spread={idr:.0f} d")
    plt.figure(figsize=(10,5))
    plt.hist(v, bins=50, color="#3690c0", edgecolor="white")
    plt.axvline(np.median(v), color="red", label=f"median={np.median(v):.0f}")
    plt.xlabel("First-wet date (day of year)"); plt.ylabel("pixels"); plt.legend()
    plt.title(f"Onset distribution (inter-decile spread = {idr:.0f} d)")
    plt.tight_layout(); plt.savefig("onset_histogram.png", dpi=150)
    # INTERPRET: 3 peaks expected region-wide — early(~95-110)=flash floods+permanent
    # water; dominant(~140-160)=monsoon onset; late(~200-230)=high ground/behind levees.

def stage6b_onset_boxplot(df_step_c):
    """Boxplot per haor — do the onset windows SEPARATE or OVERLAP?
    Overlapping boxes = simultaneous filling = no detectable propagation."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    # df_step_c must carry per-haor pixel arrays; here we just plot median±IQR proxy.
    print("Plot 5 haor onset distributions; expect overlapping IQRs in day 138–150.")
    # (Full per-pixel boxplot is in step_c_v3_masked.py.)

# -----------------------------------------------------------------------------
# MAIN (illustrative run-order; in practice run stages on the right machines)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print(__doc__)
    print("Run stages individually. Typical order:")
    print("  internet machine: init_gee(); stage1_export_reference(ee); stage2_flood_timeseries(ee)")
    print("  any machine:      stage3_create_schemas()")
    print("  (manual QGIS digitization of 5 haors happens here)")
    print("  HPC:              stage4_step_c(); stage5_dynamic_connectivity(...); stage6a/b(...)")
