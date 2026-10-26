"""
Round 3 shared settings and helpers (r3_common.py).
Every round-3 script imports this file, so paths are set in ONE place.
Nothing here talks to Google Earth Engine.
"""
import os, re, glob
import numpy as np
import pandas as pd

# ============================ PATHS (server) ============================
WORK      = "/work/a06/wasif"
VAL       = f"{WORK}/haor_flood_analysis_hydroyear/analysis/validation"
OUT       = f"{VAL}/round3"                         # everything round 3 writes goes here
HAORS_SHP = f"{WORK}/haor_flood_analysis/digitization/haors_manual.shp"
SWOT_DIR  = f"{VAL}/swot_pixels_all"                 # SWOT pixel files from the SWOT notebook
WATER20   = f"{WORK}/haor_water20m"                  # Drive folder haor_water20m, downloaded
WATER100  = f"{WORK}/haor_water100m"                 # Drive folder haor_water100m, downloaded
VENV_SITE = f"{WORK}/.venv/lib/python3.12/site-packages"
# Local test override: R3_ROOT=/some/folder python r3_3_crossings.py
if os.environ.get("R3_ROOT"):
    _r = os.environ["R3_ROOT"]
    OUT, WATER20, SWOT_DIR = f"{_r}/round3", f"{_r}/haor_water20m", f"{_r}/swot_pixels_all"
    HAORS_SHP = os.environ.get("R3_SHP", HAORS_SHP)

import sys as _sys
if os.path.isdir(VENV_SITE) and VENV_SITE not in _sys.path:
    _sys.path.insert(0, VENV_SITE)                   # same Python packages as your other scripts

# ============================ SETTINGS ============================
YEARS     = [2019, 2020, 2021, 2022, 2023, 2024, 2025]   # hydrological years, April to March
BOX20     = [90.96, 24.94, 91.29, 25.19]   # 20 m water stack: the five haors + about 1.5 km
BOX100    = [90.85, 24.85, 91.50, 25.25]   # same grid as the published 100 m stacks
WET_MONTHS = [6, 7, 8, 9, 10, 11]          # June to November = wet season for counting
TOL_M     = 300     # a boundary is "shared" where two outlines are within 300 m (the river between them)
STEP_M    = 50      # one transect every 50 m along each shared boundary
INSIDE_M  = 150     # each transect reaches 150 m into both haors (covers the river banks)
MAX_DRY_M = 20      # a dry run of up to 20 m (one radar pixel, speckle) does not block a crossing
MERGED_AT = 0.5     # merged = at least half of the shared boundary is open water
MIN_OPEN_LINES = 2  # an opening must be at least 2 neighbouring lines wide (100 m), so one noisy line does not count
UTM       = "EPSG:32646"
IDS   = ["TGR", "MTN", "GMA", "SHN", "HLR"]
NAME  = {"TGR": "Tanguar", "MTN": "Matian", "GMA": "Gumaria", "SHN": "Shanir", "HLR": "Halir"}
COLOR = {"GMA": "#E07856", "TGR": "#1C7293", "HLR": "#0F766E", "SHN": "#7C3AED", "MTN": "#0A2540"}


def ensure_dirs():
    for d in ("main", "data", "side", "side/swot_pass_maps", "cache"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)


def load_haors():
    import geopandas as gpd
    os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
    h = gpd.read_file(HAORS_SHP)
    h = h[h["haor_id"].isin(IDS)].copy()
    h["geometry"] = h.geometry.buffer(0)
    return h.set_index("haor_id")


def band_dates(descriptions):
    """'12_w20240620' or 'w20240620' -> Timestamp."""
    return [pd.Timestamp(re.search(r"(\d{8})", d).group(1)) for d in descriptions]


def water_files():
    return sorted(glob.glob(os.path.join(WATER20, "water20m_*.tif")))


def read_meta():
    fs = sorted(glob.glob(os.path.join(WATER20, "water20m_meta_*.csv")))
    if not fs:
        return pd.DataFrame(columns=["date", "orbit", "angle", "t_vv", "t_vh"])
    m = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    m["date"] = pd.to_datetime(m["date"])
    return m.drop_duplicates("date")


# ============================ TRANSECTS ============================
def build_transects(haors):
    """One short line every STEP_M metres along each shared boundary, from INSIDE_M inside
    haor A, across the river corridor, to INSIDE_M inside haor B (UTM metres)."""
    import geopandas as gpd
    from shapely.geometry import LineString, Point
    from shapely.ops import nearest_points
    hu = haors.to_crs(UTM)
    rows = []
    ids = [i for i in IDS if i in hu.index]
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            A, B = hu.loc[a, "geometry"], hu.loc[b, "geometry"]
            if A.distance(B) > TOL_M:
                continue
            near = A.boundary.intersection(B.buffer(TOL_M))
            parts = [near] if near.geom_type == "LineString" else [g for g in getattr(near, "geoms", []) if g.geom_type == "LineString"]
            k = 0
            for ln in sorted(parts, key=lambda g: -g.length):
                if ln.length < STEP_M:
                    continue
                for s in np.arange(STEP_M / 2, ln.length, STEP_M):
                    pa = ln.interpolate(s)
                    pb = nearest_points(B.boundary, pa)[0]
                    d = np.array([pb.x - pa.x, pb.y - pa.y]); gap = float(np.hypot(*d))
                    if gap < 1:                                   # outlines touch: use the normal
                        p0, p1 = ln.interpolate(max(s - 5, 0)), ln.interpolate(min(s + 5, ln.length))
                        t = np.array([p1.x - p0.x, p1.y - p0.y]); n = np.array([-t[1], t[0]]) / max(np.hypot(*t), 1e-9)
                        if A.contains(Point(pa.x + 5 * n[0], pa.y + 5 * n[1])):
                            n = -n
                        d = n
                    else:
                        d = d / gap
                    st = Point(pa.x - INSIDE_M * d[0], pa.y - INSIDE_M * d[1])
                    en = Point(pb.x + INSIDE_M * d[0], pb.y + INSIDE_M * d[1])
                    rows.append(dict(pair=f"{a}-{b}", tid=k, gap_m=round(gap, 1),
                                     mid_x=pa.x, mid_y=pa.y, geometry=LineString([st, en])))
                    k += 1
    return gpd.GeoDataFrame(rows, crs=UTM)


def transect_pixels(transects, transform, shape, spacing_m=10):
    """For each transect, the (row, col) of raster pixels sampled every spacing_m metres."""
    from pyproj import Transformer
    tf = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    inv = ~transform
    out = []
    for g in transects.geometry:
        n = int(g.length / spacing_m) + 1
        pts = [g.interpolate(f, normalized=True) for f in np.linspace(0, 1, n)]
        lon, lat = tf.transform([p.x for p in pts], [p.y for p in pts])
        cr = [inv * (x, y) for x, y in zip(lon, lat)]
        r = np.array([int(np.floor(v[1])) for v in cr]); c = np.array([int(np.floor(v[0])) for v in cr])
        ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
        out.append((r, c, ok))
    return out


def is_open(values, spacing_m=10):
    """values: water codes along a transect (1 water, 0 dry, 255 no data).
    Returns 1.0 open (water all the way, dry gaps of at most MAX_DRY_M), 0.0 blocked, nan unknown."""
    v = np.asarray(values)
    if v.size == 0 or (v == 255).mean() > 0.2:
        return np.nan
    dry = (v == 0).astype(int)
    if not dry.any():
        return 1.0
    edges = np.diff(np.r_[0, dry, 0])
    runs = np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)
    return 1.0 if runs.max() * spacing_m <= MAX_DRY_M else 0.0


def openings(open_flags):
    """Number of separate openings = runs of at least MIN_OPEN_LINES consecutive open lines
    along the boundary (lines are numbered in order along the boundary)."""
    f = (np.nan_to_num(np.asarray(open_flags, float)) > 0.5).astype(int)
    if not f.any():
        return 0
    e = np.diff(np.r_[0, f, 0])
    runs = np.flatnonzero(e == -1) - np.flatnonzero(e == 1)
    return int((runs >= MIN_OPEN_LINES).sum())


def state_of(open_frac, n_open):
    if not np.isfinite(open_frac):
        return "no data"
    if n_open == 0:
        return "separate"
    return "merged" if open_frac >= MERGED_AT else "exchange"
