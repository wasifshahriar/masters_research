#!/usr/bin/env python3
"""
==============================================================================
SWOT PIXELS FOR EVERY PASS  (haor_swot_extract_pixels_all.py)
==============================================================================
Why this exists (meeting, 1 October, Prof. Yamazaki): "Use SWOT for different
days. When the haors are judged not merged, the water surface should look
step-wise across the boundary. On the day you say they are merged, the step
should disappear. That is validation." The existing files cannot answer this:
swot_perhaor_wse.csv keeps one median per haor per pass, which throws away
WHERE inside the haor each height was measured, and swot_extract_pixels.py
kept pixels for only four dates.

This script keeps EVERY SWOT raster pixel inside the cluster box, for EVERY
pass, with all its quality columns, so the notebook can draw a real map of
water height for each pass and measure the height on both sides of every
shared boundary.

Quality filtering is deliberately loose here (finite height, water_frac > 0)
and the notebook applies the strict map filter (water_frac >= 0.5,
wse_qual <= 1, wse_uncert <= 1 m). Keeping the columns means the filter can be
changed without re-reading thousands of NetCDF files.

OUTPUT  <ANALYSIS_ROOT>/validation/swot_pixels_all/
          px_<YYYYMMDDTHHMMSS>_<res>_<tile>.npz   one file per granule, float32
                                                  columns; np.load(f) gives them back
          index.csv                               one row per granule
          processed.txt                           resume ledger
        .npz rather than CSV: about ten times faster to write and read, needs
        only numpy, and the notebook loads it in one line.

Same filename parser, tile filter and corner test as swot_extract_wse.py.
Resumable: re-running skips files already in processed.txt.
Usage:  python haor_swot_extract_pixels_all.py
==============================================================================
"""
import sys, os, glob, datetime

venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")
os.environ["HOME"] = "/home/wasif"

import numpy as np
import pandas as pd
from netCDF4 import Dataset

# ============================ SETTINGS ============================
SWOT_DIR      = "/work/a06/wasif/swot_raster_data"
ANALYSIS_ROOT = "/work/a06/wasif/haor_flood_analysis_hydroyear/analysis"
OUT_DIR       = os.path.join(ANALYSIS_ROOT, "validation", "swot_pixels_all")
EXTRACT_BBOX  = [90.93, 24.91, 91.34, 25.22]      # W, S, E, N: the five haors plus margin
TILE_PREFIXES = ("UTM45R", "UTM46R")
PAD           = 0.05
CORE_VARS     = ["wse", "wse_uncert", "wse_qual", "water_frac"]
OPTIONAL_VARS = ["dark_frac", "layover_impact", "cross_track", "sig0", "water_area",
                 "geoid", "ice_clim_flag", "inc"]
# =================================================================
W, S, E, N = EXTRACT_BBOX
os.makedirs(OUT_DIR, exist_ok=True)
INDEX = os.path.join(OUT_DIR, "index.csv")
DONE  = os.path.join(OUT_DIR, "processed.txt")


def parse_name(path):
    """Same parser as swot_extract_wse.py."""
    p = os.path.basename(path).split("_")
    try:
        res, tile = p[4], p[5]
        t0 = next(t for t in p if t.startswith("20") and "T" in t)
        dt = datetime.datetime.strptime(t0[:15], "%Y%m%dT%H%M%S")
    except (IndexError, StopIteration, ValueError):
        return None
    return dict(path=path, res=res, tile=tile, dt=dt)


def corners_touch_cluster(nc):
    """Same cheap footprint test as swot_extract_wse.py."""
    try:
        lat, lon = nc["latitude"], nc["longitude"]
        r, c = lat.shape[0] - 1, lat.shape[1] - 1
        las = [float(np.ma.filled(lat[i, j], np.nan)) for i, j in ((0, 0), (0, c), (r, 0), (r, c))]
        los = [float(np.ma.filled(lon[i, j], np.nan)) for i, j in ((0, 0), (0, c), (r, 0), (r, c))]
    except Exception:
        return True
    las = [v for v in las if np.isfinite(v)]; los = [v for v in los if np.isfinite(v)]
    if not las or not los:
        return True
    return not (max(las) < S - PAD or min(las) > N + PAD or
                max(los) < W - PAD or min(los) > E + PAD)


def var(nc, name, fill):
    a = nc[name][:]
    return np.ma.filled(a.astype("float64") if np.ma.isMaskedArray(a) else np.asarray(a, "float64"), fill)


done = set(open(DONE).read().split()) if os.path.exists(DONE) else set()
cands = []
for f in glob.glob(os.path.join(SWOT_DIR, "SWOT_L2_HR_Raster_*.nc")):
    m = parse_name(f)
    if m and m["tile"].startswith(TILE_PREFIXES) and f not in done:
        cands.append(m)
cands.sort(key=lambda m: (m["dt"], 0 if m["res"] == "100m" else 1))
print(f"granules to process: {len(cands)} (already done: {len(done)})")

new_index = not os.path.exists(INDEX)
n_written = 0
with open(INDEX, "a") as idx, open(DONE, "a") as donef:
    if new_index:
        idx.write("dt,date,res,tile,n_pixels,n_strict,file\n")
    for k, m in enumerate(cands, 1):
        try:
            nc = Dataset(m["path"])
        except OSError:
            donef.write(m["path"] + "\n"); donef.flush(); continue
        try:
            if not corners_touch_cluster(nc):
                donef.write(m["path"] + "\n"); donef.flush(); continue
            lon = var(nc, "longitude", np.nan); lat = var(nc, "latitude", np.nan)
            cols = {}
            fills = {"wse": np.nan, "wse_uncert": 99.0, "wse_qual": 3.0, "water_frac": 0.0}
            for v in CORE_VARS:
                cols[v] = var(nc, v, fills[v]) if v in nc.variables else np.full(lon.shape, np.nan)
            for v in OPTIONAL_VARS:
                if v in nc.variables:
                    cols[v] = var(nc, v, np.nan)
        finally:
            nc.close()
        ok = ((lon >= W) & (lon <= E) & (lat >= S) & (lat <= N) &
              np.isfinite(cols["wse"]) & (cols["water_frac"] > 0))
        n = int(ok.sum())
        if n:
            arrs = {"lon": lon[ok], "lat": lat[ok], **{c: a[ok] for c, a in cols.items()}}
            arrs = {c: a.astype("float64" if c in ("lon", "lat") else "float32")
                    for c, a in arrs.items()}
            strict = int(((arrs["water_frac"] >= 0.5) & (arrs["wse_qual"] <= 1) &
                          (arrs["wse_uncert"] <= 1.0)).sum())
            fn = f"px_{m['dt']:%Y%m%dT%H%M%S}_{m['res']}_{m['tile']}.npz"
            np.savez_compressed(os.path.join(OUT_DIR, fn), **arrs)
            idx.write(f"{m['dt'].isoformat()},{m['dt']:%Y-%m-%d},{m['res']},{m['tile']},{n},{strict},{fn}\n")
            idx.flush(); n_written += 1
        donef.write(m["path"] + "\n"); donef.flush()
        if k % 25 == 0:
            print(f"  {k}/{len(cands)} granules | files written {n_written}")

ix = pd.read_csv(INDEX)
print(f"\n✓ {n_written} new pixel files. Index now holds {len(ix)} granules over "
      f"{ix['date'].nunique()} dates ({ix['date'].min()} to {ix['date'].max()}).")
print("Next: haor_swot_validation.ipynb")
