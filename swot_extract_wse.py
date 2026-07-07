#!/usr/bin/env python3
"""
==============================================================================
SWOT PER-HAOR WSE EXTRACTION  (swot_extract_wse.py)  — v2, repaired
==============================================================================
Heavy preprocessing moved OUT of the notebook. Scans SWOT L2_HR_Raster NetCDFs,
keeps files that can cover the 5-haor cluster, extracts per-haor median
water-surface elevation (WSE) with QC filters, writes ONE small CSV that the
notebook analyses in seconds.

v2 FIXES:
  - CSV schema guard + auto-repair: the old notebook wrote an 8-column
    swot_perhaor_wse.csv; v1 appended 9-column rows to it and the final
    read crashed. v2 salvages the 9-column rows (the complete dataset) and
    rewrites the file cleanly. Any future mismatch is caught at startup.
  - Corner test now samples a 3x3 grid of positions (scene corners are often
    masked, which made the old 4-corner test always inconclusive).
  - Resumable as before: processed paths in swot_processed_files.txt, so
    re-running after the crash skips all 1673 files and finishes in seconds.

OUTPUTS (both in ANALYSIS_ROOT):
  swot_perhaor_wse.csv        <- the notebook's SWOT cells load this
  swot_processed_files.txt    <- resume ledger; delete to force reprocessing
==============================================================================
"""
import os, glob, datetime, shutil
import numpy as np
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset
from matplotlib.path import Path as MplPath

# ============================ SETTINGS ============================
SWOT_DIR   = "/work/a06/wasif/swot_raster_data"
HAORS_SHP  = "/work/a06/wasif/haor_flood_analysis/digitization/haors_manual.shp"
ANALYSIS_ROOT = "/work/a06/wasif/haor_flood_analysis_multiyear/analysis"
OUT_CSV   = os.path.join(ANALYSIS_ROOT, "swot_perhaor_wse.csv")
DONE_FILE = os.path.join(ANALYSIS_ROOT, "swot_processed_files.txt")
CLUSTER_BBOX = [90.85, 24.85, 91.50, 25.25]      # W,S,E,N
TILE_PREFIXES = ("UTM45R", "UTM46R")             # band R = 24-32 N; zones 45/46
PAD = 0.05
MIN_PIXELS_PER_HAOR = 5
QC = dict(wse_qual_max=1, wse_uncert_max=3.0, water_frac_min=0.01,
          water_frac_max=1.0, wse_min=-50.0, wse_max=200.0)
HEADER = "year,doy,dt,res,tile,haor_id,wse_median,wse_iqr,n"
# =================================================================

os.makedirs(ANALYSIS_ROOT, exist_ok=True)
os.environ["SHAPE_RESTORE_SHX"] = "YES"
W, S, E, N = CLUSTER_BBOX

# ---------- CSV schema guard + repair ----------
def repair_csv(path):
    """Keep only valid 9-field data rows; rewrite with a clean header.
    Handles the mixed 8/9-column file produced by the v1 crash."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = [l.rstrip("\n") for l in f]
    good = [l for l in lines
            if l.count(",") == HEADER.count(",") and not l.startswith("year,")]
    if lines and lines[0] == HEADER and len(good) == len(lines) - 1:
        return  # already clean
    shutil.copy(path, path + ".bak")
    with open(path, "w") as f:
        f.write(HEADER + "\n")
        for l in good:
            f.write(l + "\n")
    print(f"repaired {os.path.basename(path)}: kept {len(good)} valid rows "
          f"(backup at {os.path.basename(path)}.bak)")

repair_csv(OUT_CSV)

haors = gpd.read_file(HAORS_SHP)
haors = haors[haors["haor_id"] != "DUMMY"].to_crs("EPSG:4326")

def _largest_poly(geom):
    return max(geom.geoms, key=lambda g: g.area) if geom.geom_type == "MultiPolygon" else geom

haor_paths = {h.haor_id: MplPath(np.asarray(_largest_poly(h.geometry).exterior.coords))
              for _, h in haors.iterrows()}
print(f"haors: {list(haor_paths)}")

def parse_name(path):
    p = os.path.basename(path).split("_")
    try:
        res, tile = p[4], p[5]
        t0 = next(t for t in p if t.startswith("20") and "T" in t)
        dt = datetime.datetime.strptime(t0[:15], "%Y%m%dT%H%M%S")
    except (IndexError, StopIteration, ValueError):
        return None
    return dict(path=path, res=res, tile=tile, dt=dt,
                year=dt.year, doy=dt.timetuple().tm_yday)

def footprint_touches_cluster(nc):
    """Sample a 3x3 grid of positions (corners are often masked in SWOT scenes)."""
    try:
        lat, lon = nc["latitude"], nc["longitude"]
        r, c = lat.shape[0] - 1, lat.shape[1] - 1
        ii = [0, r // 2, r]; jj = [0, c // 2, c]
        las, los = [], []
        for i in ii:
            for j in jj:
                la = float(np.ma.filled(lat[i, j], np.nan))
                lo = float(np.ma.filled(lon[i, j], np.nan))
                if np.isfinite(la) and np.isfinite(lo):
                    las.append(la); los.append(lo)
    except Exception:
        return True
    if not las:
        return True                                   # inconclusive -> full read
    return not (max(las) < S - PAD or min(las) > N + PAD or
                max(los) < W - PAD or min(los) > E + PAD)

def extract(nc, meta):
    lon = np.ma.filled(nc["longitude"][:], np.nan)
    lat = np.ma.filled(nc["latitude"][:], np.nan)
    wse = np.ma.filled(nc["wse"][:], np.nan)
    q   = np.ma.filled(nc["wse_qual"][:], 3) if "wse_qual" in nc.variables else np.zeros_like(wse)
    unc = np.ma.filled(nc["wse_uncert"][:], 99) if "wse_uncert" in nc.variables else np.zeros_like(wse)
    wf  = np.ma.filled(nc["water_frac"][:], 0) if "water_frac" in nc.variables else np.ones_like(wse)
    ok = ((lon >= W) & (lon <= E) & (lat >= S) & (lat <= N) & np.isfinite(wse) &
          (q <= QC["wse_qual_max"]) & (unc < QC["wse_uncert_max"]) &
          (wf > QC["water_frac_min"]) & (wf <= QC["water_frac_max"]) &
          (wse > QC["wse_min"]) & (wse < QC["wse_max"]))
    if not ok.any():
        return []
    pts = np.column_stack([lon[ok], lat[ok]]); vals = wse[ok]
    rows = []
    for hid, pth in haor_paths.items():
        inside = pth.contains_points(pts)
        if inside.sum() >= MIN_PIXELS_PER_HAOR:
            v = vals[inside]
            rows.append(dict(year=meta["year"], doy=meta["doy"], dt=meta["dt"].isoformat(),
                             res=meta["res"], tile=meta["tile"], haor_id=hid,
                             wse_median=float(np.median(v)),
                             wse_iqr=float(np.subtract(*np.percentile(v, [75, 25]))),
                             n=int(inside.sum())))
    return rows

# ---------- main loop (resumable) ----------
done = set()
if os.path.exists(DONE_FILE):
    done = set(open(DONE_FILE).read().split())
    print(f"resuming: {len(done)} files already processed")

cands = []
for f in glob.glob(os.path.join(SWOT_DIR, "SWOT_L2_HR_Raster_*.nc")):
    m = parse_name(f)
    if m and m["tile"].startswith(TILE_PREFIXES):
        cands.append(m)
cands.sort(key=lambda m: (m["dt"], 0 if m["res"] == "100m" else 1))
print(f"candidates after tile+band filter: {len(cands)}")

header_needed = not os.path.exists(OUT_CSV)
n_new, n_skip, n_empty = 0, 0, 0
with open(OUT_CSV, "a") as out, open(DONE_FILE, "a") as donef:
    if header_needed:
        out.write(HEADER + "\n")
    for k, m in enumerate(cands, 1):
        if m["path"] in done:
            continue
        try:
            nc = Dataset(m["path"])
        except OSError:
            donef.write(m["path"] + "\n"); donef.flush(); continue
        try:
            if not footprint_touches_cluster(nc):
                n_skip += 1
            else:
                rows = extract(nc, m)
                if rows:
                    for r in rows:
                        out.write(",".join(str(r[c]) for c in HEADER.split(",")) + "\n")
                    out.flush(); n_new += len(rows)
                else:
                    n_empty += 1
        finally:
            nc.close()
        donef.write(m["path"] + "\n"); donef.flush()
        if k % 50 == 0:
            print(f"  {k}/{len(cands)} | rows {n_new} | skipped {n_skip} | empty {n_empty}")

print(f"\n\u2713 extraction pass done. new rows: {n_new} | footprint-skipped: {n_skip} | no-valid-pixels: {n_empty}")

# ---------- final dedup (one row per acquisition+haor, 100m preferred) ----------
repair_csv(OUT_CSV)                                   # belt and suspenders
df = pd.read_csv(OUT_CSV)
before = len(df)
df["res_rank"] = (df["res"] != "100m").astype(int)
df = (df.sort_values(["dt", "haor_id", "res_rank", "n"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["dt", "haor_id"], keep="first")
        .drop(columns="res_rank"))
df.to_csv(OUT_CSV, index=False)
print(f"  dedup: {before} -> {len(df)} rows")
print(f"  output: {OUT_CSV}")
print(f"  dates per year: {df.groupby('year')['doy'].nunique().to_dict()}")