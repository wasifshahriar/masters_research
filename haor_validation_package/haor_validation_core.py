#!/usr/bin/env python3
"""
==============================================================================
HAOR VALIDATION CORE  (haor_validation_core.py)
==============================================================================
One shared copy of the production connectivity logic, imported by every
validation script and notebook so that the maps, the sensitivity sweep and the
SWOT tests all look at exactly the algorithm that produced the published
numbers.

The per-date logic is analyze_year() cell 4 of haor_flood_hydroyear.ipynb,
line for line. The only change is that every parameter is passed in instead of
read from globals, so a sensitivity run can change one value at a time.
With DEFAULTS it reproduces the published connectivity_curve_<Y>.csv and
haor_connection_matrix_<Y>.csv exactly; haor_validation_labels.py checks this
on every run and prints any mismatch.

Nothing in this file writes to disk.
==============================================================================
"""
import re, datetime
import numpy as np
import pandas as pd
from scipy import ndimage

# The production values, cell 1 of haor_flood_hydroyear.ipynb
DEFAULTS = dict(
    MIN_PX=10,              # smallest body kept, pixels
    BRIDGE=1,               # binary opening iterations (removes links < 3 px wide)
    MIN_OBSERVED_FRAC=0.50, # raw coverage guard
    BUFFER_PX=8,            # study area = haor union dilated by this many pixels
    FILL_GAPS=True,         # persistence filling
    MAX_GAP_DAYS=12.0,      # filling limit
    CONNECTIVITY=4,         # 4 = scipy default cross; 8 = include diagonals
    MIN_OVERLAP=0,          # pixels of water a haor needs before it gets a label
)

# Pixel classes written by haor_validation_labels.py, one uint8 band per date
CODE_OUTSIDE   = 0   # outside the study area
CODE_DRY_OBS   = 1   # observed, not flooded
CODE_WET_OBS   = 2   # observed flooded, kept in a labelled body
CODE_WET_FILL  = 3   # unobserved today, filled as flooded, kept in a body
CODE_WET_DROP  = 4   # flooded (observed or filled) but removed by the opening or the speckle filter
CODE_DRY_FILL  = 5   # unobserved today, filled as not flooded
CODE_NODATA    = 6   # unobserved and not filled
CODE_SKIPPED   = 255 # whole date skipped by the coverage guard
CODE_NAMES = {CODE_OUTSIDE: "outside study area", CODE_DRY_OBS: "observed dry",
              CODE_WET_OBS: "observed flooded (kept)", CODE_WET_FILL: "filled flooded (kept)",
              CODE_WET_DROP: "flooded, removed by guards", CODE_DRY_FILL: "filled dry",
              CODE_NODATA: "unobserved", CODE_SKIPPED: "date skipped"}
LABEL_SKIPPED = 65535


def dayidx(desc, base_year):
    """Days since 1 Jan of the ANALYSIS year (copy of _dayidx in cell 2)."""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", desc or "")
    if not m:
        return np.nan
    y, mo, d = map(int, m.groups())
    return float((datetime.date(y, mo, d) - datetime.date(base_year, 1, 1)).days)


def doy_to_date(year, doy):
    return datetime.date(int(year), 1, 1) + datetime.timedelta(days=float(doy))


def read_stack(path, year):
    """Read a flood stack the way analyze_year does: nodata -> NaN, sorted by date.
    Also accepts uint8 stacks from haor_export_flood_stack_hires.py (255 = nodata).
    Returns bands (list of float32), days (array), meta dict."""
    import rasterio
    with rasterio.open(path) as src:
        nod = src.nodata
        days = np.array([dayidx(d, year) for d in src.descriptions], float)
        bands = []
        for i in range(src.count):
            b = src.read(i + 1).astype("float32")
            if nod is not None:
                b[b == nod] = np.nan
            if src.dtypes[i] == "uint8":
                b[b == 255] = np.nan
            bands.append(b)
        meta = dict(shape=(src.height, src.width), transform=src.transform, crs=src.crs,
                    descriptions=list(src.descriptions), res=src.res)
    if (~np.isfinite(days)).any():
        raise ValueError(f"{year}: {int((~np.isfinite(days)).sum())} band names did not parse to a date")
    order = np.argsort(days)
    return [bands[i] for i in order], days[order], meta


def haor_masks(haors, meta):
    """Burn each haor polygon onto the stack grid (same call as analyze_year)."""
    from rasterio.features import geometry_mask
    hv = haors.to_crs(meta["crs"])
    return {h.haor_id: geometry_mask([h.geometry], meta["shape"], meta["transform"], invert=True)
            for _, h in hv.iterrows()}


def persistence_fill(bands, days, max_gap):
    """Copy of cell 2. Carries each pixel's LAST TRUE observation forward for up to
    max_gap days. Filled values are never a source for later filling."""
    filled = [b.copy() for b in bands]
    last_v = np.full(bands[0].shape, np.nan, "float32")
    last_d = np.full(bands[0].shape, -9e9, "float32")
    n_fill = 0
    for i, b in enumerate(bands):
        obs = np.isfinite(b)
        can = (~obs) & np.isfinite(last_v) & ((days[i] - last_d) <= max_gap)
        filled[i][can] = last_v[can]
        n_fill += int(can.sum())
        last_v[obs] = b[obs]
        last_d[obs] = days[i]
    return filled, n_fill


def cluster_union(masks):
    c = None
    for m in masks.values():
        c = m.copy() if c is None else (c | m)
    return c


def label_one(b, masks, study, obs_raw_i, p, raw=None):
    """One date of the production connectivity step, parameterised.
    b      : band after filling (1 / 0 / NaN)
    raw    : the same band BEFORE filling, only needed for the pixel code raster
    Returns a dict. With p == DEFAULTS this equals analyze_year exactly."""
    out = dict(skipped=False)
    if obs_raw_i < p["MIN_OBSERVED_FRAC"]:
        out.update(skipped=True, n=-1, largest=-1.0, total=-1, all_connected=False,
                   comp={h: -1 for h in masks})
        return out
    water0 = np.isfinite(b) & (b == 1) & study
    water = ndimage.binary_opening(water0, iterations=p["BRIDGE"]) if p["BRIDGE"] else water0.copy()
    struct = ndimage.generate_binary_structure(2, 2) if p["CONNECTIVITY"] == 8 else None
    lab, n = ndimage.label(water, structure=struct)
    if n:
        sz = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        water &= np.isin(lab, np.where(sz >= p["MIN_PX"])[0] + 1)
        lab, n = ndimage.label(water, structure=struct)
    total = int(water.sum())
    largest = float(ndimage.sum(np.ones_like(lab), lab,
                    index=np.arange(1, n + 1)).max() / total) if (n and total) else 0.0
    comp = {}
    for hid, m in masks.items():
        v = lab[m & (lab > 0)]
        ok = v.size > 0 and v.size >= max(p["MIN_OVERLAP"], 1)
        comp[hid] = int(np.bincount(v).argmax()) if ok else 0
    nz = [x for x in comp.values() if x > 0]
    out.update(n=int(n), largest=largest, total=total, comp=comp, lab=lab,
               water0=water0, water=water,
               all_connected=(len(nz) == len(masks) and len(set(nz)) == 1))
    return out


def code_raster(b_filled, b_raw, study, res):
    """Pixel classes for one scored date (see CODE_* above)."""
    code = np.full(b_filled.shape, CODE_OUTSIDE, np.uint8)
    obs = np.isfinite(b_raw)
    fill = (~obs) & np.isfinite(b_filled)
    s = study
    code[s & obs & (b_raw == 0)] = CODE_DRY_OBS
    code[s & fill & (b_filled == 0)] = CODE_DRY_FILL
    code[s & ~np.isfinite(b_filled)] = CODE_NODATA
    wet = s & np.isfinite(b_filled) & (b_filled == 1)
    code[wet & ~res["water"]] = CODE_WET_DROP
    code[wet & res["water"] & obs] = CODE_WET_OBS
    code[wet & res["water"] & fill] = CODE_WET_FILL
    return code


def adjacency(haors, utm="EPSG:32646", tol_m=50.0):
    """Measured shared boundary length for every pair, in metres. A pair is
    adjacent when its boundaries run within tol_m of each other for > 0 m
    (digitised polygons rarely share vertices exactly)."""
    h = haors.to_crs(utm).set_index("haor_id")
    ids = list(h.index)
    rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = h.geometry[ids[i]], h.geometry[ids[j]]
            L = float(a.boundary.intersection(b.buffer(tol_m)).length)
            rows.append(dict(pair=f"{ids[i]}-{ids[j]}", a=ids[i], b=ids[j],
                             shared_m=round(L, 1), gap_m=round(float(a.distance(b)), 1),
                             adjacent=L > 0))
    return pd.DataFrame(rows)


def contact_zones(masks, adj_df, k):
    """For each adjacent pair, the strip within k pixels of BOTH haors: where a
    direct connection across the shared boundary has to pass."""
    zones = {}
    dil = {h: ndimage.binary_dilation(m, iterations=k) for h, m in masks.items()}
    for _, r in adj_df[adj_df.adjacent].iterrows():
        zones[r.pair] = dil[r.a] & dil[r.b]
    return zones


def stack_dates(path, year):
    """Band numbers of a stack in date order, without reading pixels."""
    import rasterio
    with rasterio.open(path) as src:
        days = np.array([dayidx(d, year) for d in src.descriptions], float)
        meta = dict(shape=(src.height, src.width), transform=src.transform, crs=src.crs,
                    descriptions=list(src.descriptions), res=src.res, count=src.count)
    if (~np.isfinite(days)).any():
        raise ValueError(f"{year}: {int((~np.isfinite(days)).sum())} band names did not parse to a date")
    order = np.argsort(days)
    return order + 1, days[order], meta


def iter_stack(path, year, masks, p):
    """Memory-light version of run_year_conn for large (fine resolution) stacks.
    Reads ONE band at a time in date order, applies persistence filling
    incrementally (same rule as persistence_fill: only true observations are a
    source) and yields (day, raw_band, filled_band, result) per date. Gives the
    same states as run_year_conn."""
    import rasterio
    bandnos, days, meta = stack_dates(path, year)
    cluster = cluster_union(masks)
    n_cluster = int(cluster.sum())
    study = ndimage.binary_dilation(cluster, iterations=p["BUFFER_PX"])
    do_fill = p["FILL_GAPS"] and p["MAX_GAP_DAYS"] > 0
    last_v = np.full(meta["shape"], np.nan, "float32")
    last_d = np.full(meta["shape"], -9e9, "float32")
    with rasterio.open(path) as src:
        nod = src.nodata
        for bn, day in zip(bandnos, days):
            b = src.read(int(bn)).astype("float32")
            if nod is not None:
                b[b == nod] = np.nan
            if src.dtypes[int(bn) - 1] == "uint8":
                b[b == 255] = np.nan
            obs = np.isfinite(b)
            obs_raw = float((obs & cluster).sum()) / n_cluster
            f = b.copy()
            if do_fill:
                can = (~obs) & np.isfinite(last_v) & ((day - last_d) <= p["MAX_GAP_DAYS"])
                f[can] = last_v[can]
                last_v[obs] = b[obs]
                last_d[obs] = day
            r = label_one(f, masks, study, obs_raw, p)
            r["doy"] = day
            r["obs_raw"] = obs_raw
            r["obs_eff"] = float((np.isfinite(f) & cluster).sum()) / n_cluster
            yield day, b, f, r, study, meta


def run_year_conn(bands, days, masks, p, raw_bands=None, keep_arrays=True):
    """Connectivity for a whole year with parameters p. Returns conn_df,
    matrix_df, list of per-date result dicts (labels kept unless
    keep_arrays=False), obs_raw, bands_used."""
    cluster = cluster_union(masks)
    n_cluster = int(cluster.sum())
    obs_raw = np.array([float((np.isfinite(b) & cluster).sum()) / n_cluster for b in bands])
    if p["FILL_GAPS"] and p["MAX_GAP_DAYS"] > 0:
        used, _ = persistence_fill(bands, days, p["MAX_GAP_DAYS"])
    else:
        used = bands
    obs_eff = np.array([float((np.isfinite(b) & cluster).sum()) / n_cluster for b in used])
    study = ndimage.binary_dilation(cluster, iterations=p["BUFFER_PX"])
    crecs, cmat, res = [], [], []
    for i, b in enumerate(used):
        r = label_one(b, masks, study, obs_raw[i], p)
        r["doy"] = days[i]
        if not keep_arrays:
            for k in ("lab", "water0", "water"):
                r.pop(k, None)
        res.append(r)
        crecs.append(dict(doy=days[i], n_components=r["n"], largest_frac=round(r["largest"], 3),
                          total_wet=r["total"], obs_raw=round(obs_raw[i], 3),
                          obs_eff=round(obs_eff[i], 3), all_connected=r["all_connected"]))
        cmat.append(dict(doy=days[i], **r["comp"]))
    return (pd.DataFrame(crecs).sort_values("doy"), pd.DataFrame(cmat).sort_values("doy"),
            res, obs_raw, used, study)


def episodes(conn):
    """Runs of consecutive merged dates among scored dates (same as Part A3)."""
    v = conn[conn["n_components"] >= 0].sort_values("doy")
    out, run = [], []
    for _, r in v.iterrows():
        if r["all_connected"]:
            run.append(r["doy"])
        elif run:
            out.append((run[0], run[-1], len(run))); run = []
    if run:
        out.append((run[0], run[-1], len(run)))
    return out


def best_threshold(frac_merged, frac_other, grid=None):
    """Part A2 threshold search (raw accuracy) plus balanced accuracy at that value."""
    a, b = np.asarray(frac_merged), np.asarray(frac_other)
    if len(a) < 3 or len(b) < 3:
        return dict(thr=np.nan, acc=np.nan, bal=np.nan)
    grid = np.linspace(0.05, 0.95, 181) if grid is None else grid
    acc = [((a >= t).sum() + (b < t).sum()) / (len(a) + len(b)) for t in grid]
    t = float(grid[int(np.argmax(acc))])
    bal = ((a >= t).mean() + (b < t).mean()) / 2
    return dict(thr=round(t, 3), acc=round(max(acc), 3), bal=round(float(bal), 3))
