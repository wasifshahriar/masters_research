#!/usr/bin/env python3
"""
==============================================================================
SWOT PIXEL EXTRACTION FOR MAPPING  (swot_extract_pixels.py)
==============================================================================
Companion to swot_extract_wse.py. That script produced ONE MEDIAN water level
per haor per pass, which is all the statistics need. This script keeps the
INDIVIDUAL PIXELS for a small number of chosen dates, which is what a map needs.

Run this AFTER the notebook cell that writes swot_map_dates.csv, because that
file tells this script which dates to extract.

INPUT   <ANALYSIS_ROOT>/swot_map_dates.csv   written by the notebook
OUTPUT  <ANALYSIS_ROOT>/swot_pixels/swot_px_<label>_<YYYYMMDD>.csv
        one row per SWOT pixel: lon, lat, wse, wse_uncert, water_frac

Usage:  python swot_extract_pixels.py
==============================================================================
"""
import sys, os, glob, datetime
import numpy as np
import pandas as pd
from netCDF4 import Dataset

# ---- make the venv importable, same pattern as your other scripts ----
venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")
os.environ["HOME"] = "/home/wasif"

# ============================ SETTINGS ============================
SWOT_DIR      = "/work/a06/wasif/swot_raster_data"
ANALYSIS_ROOT = "/work/a06/wasif/haor_flood_analysis_hydroyear/analysis"
DATES_CSV     = os.path.join(ANALYSIS_ROOT, "swot_map_dates.csv")
OUT_DIR       = os.path.join(ANALYSIS_ROOT, "swot_pixels")

CLUSTER_BBOX  = [90.85, 24.85, 91.50, 25.25]      # W, S, E, N
TILE_PREFIXES = ("UTM45R", "UTM46R")               # band R covers 24 to 32 N
# QC, identical to swot_extract_wse.py so the map matches the statistics
QC = dict(wse_qual_max=1, wse_uncert_max=3.0, water_frac_min=0.01,
          water_frac_max=1.0, wse_min=-50.0, wse_max=200.0)
# =================================================================

W, S, E, N = CLUSTER_BBOX
os.makedirs(OUT_DIR, exist_ok=True)

assert os.path.exists(DATES_CSV), (
    f"missing {DATES_CSV}\n"
    "Run the notebook cell 'PART C1: choose the dates to map' first.")

want = pd.read_csv(DATES_CSV)
print(f"dates requested by the notebook: {len(want)}")
print(want.to_string(index=False))

# swot_date in the CSV is the calendar date of the SWOT pass, as YYYY-MM-DD
targets = {}
for _, r in want.iterrows():
    key = str(r["swot_date"]).replace("-", "")
    targets.setdefault(key, []).append(str(r["label"]))
print(f"\nunique SWOT dates to find: {sorted(targets)}")


def parse_name(path):
    """Filename carries resolution, tile and timestamp. Same parser as before."""
    p = os.path.basename(path).split("_")
    try:
        res, tile = p[4], p[5]
        t0 = next(t for t in p if t.startswith("20") and "T" in t)
        dt = datetime.datetime.strptime(t0[:15], "%Y%m%dT%H%M%S")
    except (IndexError, StopIteration, ValueError):
        return None
    return dict(path=path, res=res, tile=tile, dt=dt, ymd=dt.strftime("%Y%m%d"))


# ---- find candidate files: right tile band, and one of the wanted dates ----
cands = []
for f in glob.glob(os.path.join(SWOT_DIR, "SWOT_L2_HR_Raster_*.nc")):
    m = parse_name(f)
    if not m:
        continue
    if not m["tile"].startswith(TILE_PREFIXES):
        continue
    if m["ymd"] in targets:
        cands.append(m)
# prefer the finer 100 m product when both exist for the same pass
cands.sort(key=lambda m: (m["ymd"], 0 if m["res"] == "100m" else 1))
print(f"\nmatching files found: {len(cands)}")
for m in cands:
    print(f"  {m['ymd']}  {m['res']:5}  {m['tile']}  {os.path.basename(m['path'])[:60]}")

if not cands:
    print("\nNo files matched. Check that the dates in swot_map_dates.csv "
          "correspond to passes you actually downloaded.")
    sys.exit(1)

# ---- extract pixels, one output file per requested date ----
collected = {}
for m in cands:
    try:
        nc = Dataset(m["path"])
    except OSError:
        print(f"  cannot open {os.path.basename(m['path'])}, skipped")
        continue
    try:
        lon = np.ma.filled(nc["longitude"][:], np.nan)
        lat = np.ma.filled(nc["latitude"][:], np.nan)
        wse = np.ma.filled(nc["wse"][:], np.nan)
        q   = np.ma.filled(nc["wse_qual"][:], 3)   if "wse_qual"   in nc.variables else np.zeros_like(wse)
        unc = np.ma.filled(nc["wse_uncert"][:], 99) if "wse_uncert" in nc.variables else np.zeros_like(wse)
        wf  = np.ma.filled(nc["water_frac"][:], 0)  if "water_frac" in nc.variables else np.ones_like(wse)
    finally:
        nc.close()

    ok = ((lon >= W) & (lon <= E) & (lat >= S) & (lat <= N) & np.isfinite(wse) &
          (q <= QC["wse_qual_max"]) & (unc < QC["wse_uncert_max"]) &
          (wf > QC["water_frac_min"]) & (wf <= QC["water_frac_max"]) &
          (wse > QC["wse_min"]) & (wse < QC["wse_max"]))
    n = int(ok.sum())
    print(f"  {m['ymd']} {m['res']:5}: {n:6,} pixels pass QC inside the cluster")
    if n == 0:
        continue

    df = pd.DataFrame({"lon": lon[ok], "lat": lat[ok], "wse": wse[ok],
                       "wse_uncert": unc[ok], "water_frac": wf[ok],
                       "res": m["res"], "tile": m["tile"]})
    collected.setdefault(m["ymd"], []).append(df)

# ---- write one file per date, per label ----
print()
for ymd, frames in collected.items():
    df = pd.concat(frames, ignore_index=True)
    # if both 100m and 250m are present, keep only the finer product
    if (df["res"] == "100m").any():
        df = df[df["res"] == "100m"].copy()
    for label in targets[ymd]:
        out = os.path.join(OUT_DIR, f"swot_px_{label}_{ymd}.csv")
        df.to_csv(out, index=False)
        print(f"wrote {out}  ({len(df):,} pixels, "
              f"wse {df.wse.min():.2f} to {df.wse.max():.2f} m)")

missing = [d for d in targets if d not in collected]
if missing:
    print(f"\nno usable pixels for: {missing}")
    print("Pick a different date in the notebook and run this script again.")
else:
    print("\nAll requested dates extracted. Return to the notebook, Part C2.")
