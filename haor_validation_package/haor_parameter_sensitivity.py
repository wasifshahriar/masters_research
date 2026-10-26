#!/usr/bin/env python3
"""
==============================================================================
HAOR PARAMETER SENSITIVITY  (haor_parameter_sensitivity.py)
==============================================================================
Why this exists (meeting, 1 October, assistant professor): "You chose the 800 m
ring, not 1 km. How sensitive is the algorithm to your choices? Report it, even
in the supplement." In the meeting the answer was that 800 m against 500 m
"doesn't really make any difference". That has never been tested. This script
tests it, and every other connectivity parameter, one at a time.

Design: one-at-a-time sweep around the production values. Every run uses the
same flood stacks and the same code (haor_validation_core.py). For each run it
records what the thesis reports, so a reader can see which conclusions move:

  n_merged          merged dates inside the wet season (published window)
  ws_merge_frac     share of wet-season dates that are merged
  merge_first/last  the merge window
  episodes          number of merge episodes
  thr / bal         the A2 threshold and its balanced accuracy, using the
                    published cluster flooded fraction (unchanged by these
                    parameters, so only the labels move)
  min_frac          lowest flooded fraction at which any merge occurred
  pair_<A-B>        share of scored dates on which that pair is merged

Two rule variants are included as well:
  ADJACENT_ONLY  a merged date also requires the shared body to cross the
                 boundary of enough adjacent pairs to join all five through
                 shared borders (the idea raised in the meeting: only allow
                 connection where haors share a boundary)
  FILL off       MAX_GAP_DAYS = 0, which is also the A8 fill check

OUTPUT  validation/sensitivity/sensitivity_by_year.csv
        validation/sensitivity/sensitivity_pooled.csv
        validation/sensitivity/dates_<run>.csv   (per-date states, for plots)

Runtime: about 30 runs x 7 years. Expect 10 to 30 minutes on the server.
Usage:   python haor_parameter_sensitivity.py
==============================================================================
"""
import sys, os, glob, re, time

venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
import haor_validation_core as core

# ============================ SETTINGS ============================
BASE          = "/work/a06/wasif/haor_flood_analysis_hydroyear"
ANALYSIS_ROOT = os.path.join(BASE, "analysis")
HAORS_SHP     = "/work/a06/wasif/haor_flood_analysis/digitization/haors_manual.shp"
OUT           = os.path.join(ANALYSIS_ROOT, "validation", "sensitivity")
CONTACT_PX    = 3

SWEEP = {                              # production value is marked in the output
    "BUFFER_PX":         [0, 3, 5, 8, 10, 15],
    "BRIDGE":            [0, 1, 2],
    "MIN_PX":            [3, 10, 30, 100],
    "MIN_OBSERVED_FRAC": [0.30, 0.50, 0.70, 0.90],
    "MAX_GAP_DAYS":      [0, 6, 12, 18, 24],
    "CONNECTIVITY":      [4, 8],
    "MIN_OVERLAP":       [0, 5, 20],
}
# =================================================================
os.makedirs(OUT, exist_ok=True)
for f in glob.glob(os.path.join(OUT, "dates_*.csv")):      # fresh run, no stale appends
    os.remove(f)
os.environ["SHAPE_RESTORE_SHX"] = "YES"
haors = gpd.read_file(HAORS_SHP)
haors = haors[haors["haor_id"] != "DUMMY"].copy()
if (~haors.geometry.is_valid).any():
    haors["geometry"] = haors.geometry.buffer(0)
adj = core.adjacency(haors)
ADJ_PAIRS = list(adj.loc[adj.adjacent, "pair"])

summ = pd.read_csv(os.path.join(ANALYSIS_ROOT, "multiyear_summary.csv")).set_index("year")
YEARS = sorted(int(re.search(r"(\d{4})", os.path.basename(f)).group(1))
               for f in glob.glob(os.path.join(BASE, "flood_stack_cluster_*.tif")))

runs = [("baseline", "none", "-", dict(core.DEFAULTS))]
for par, vals in SWEEP.items():
    for v in vals:
        if v == core.DEFAULTS[par]:
            continue
        pp = dict(core.DEFAULTS); pp[par] = v
        if par == "MAX_GAP_DAYS" and v == 0:
            pp["FILL_GAPS"] = False
        runs.append((f"{par}={v}", par, v, pp))
runs.append(("ADJACENT_ONLY", "RULE", "adjacent_only", dict(core.DEFAULTS)))
print(f"{len(runs)} runs x {len(YEARS)} years")


def adjacent_all_connected(r, masks, zones):
    """True when all five share one label AND the shared body crosses enough
    shared boundaries to link all five through adjacent pairs."""
    if not r["all_connected"]:
        return False
    lab, c = r["lab"], r["comp"][next(iter(masks))]
    ids = list(masks)
    parent = {h: h for h in ids}
    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for pair, z in zones.items():
        a, b = pair.split("-")
        if ((lab == c) & z).any():
            parent[find(a)] = find(b)
    return len({find(h) for h in ids}) == 1


cache = {}
rows, t0 = [], time.time()
for year in YEARS:
    bands, days, meta = core.read_stack(os.path.join(BASE, f"flood_stack_cluster_{year}.tif"), year)
    masks = core.haor_masks(haors, meta)
    zones = core.contact_zones(masks, adj, CONTACT_PX)
    cur = pd.read_csv(os.path.join(ANALYSIS_ROOT, str(year), f"haor_hydrograph_curves_{year}.csv"))
    cur["cluster_frac"] = cur[list(masks)].mean(axis=1)
    frac = dict(zip(cur["doy"].round(3), cur["cluster_frac"]))
    ws0, ws1 = float(summ.loc[year, "ws_start"]), float(summ.loc[year, "ws_end"])
    for name, par, val, pp in runs:
        conn, mat, res, *_ = core.run_year_conn(bands, days, masks, pp)
        if par == "RULE":
            conn["all_connected"] = [adjacent_all_connected(r, masks, zones) if not r["skipped"] else False
                                     for r in res]
        conn["cluster_frac"] = conn["doy"].round(3).map(frac)
        v = conn[conn["n_components"] >= 0]
        ws = v[(v.doy >= ws0) & (v.doy <= ws1)]
        mg = ws[ws.all_connected]
        ep = core.episodes(conn)
        th = core.best_threshold(v.loc[v.all_connected, "cluster_frac"].dropna(),
                                 v.loc[~v.all_connected, "cluster_frac"].dropna())
        row = dict(run=name, param=par, value=val, year=year, n_scored=len(v),
                   n_ws=len(ws), n_merged=len(mg),
                   ws_merge_frac=round(len(mg) / len(ws), 3) if len(ws) else np.nan,
                   merge_first=mg.doy.min() if len(mg) else np.nan,
                   merge_last=mg.doy.max() if len(mg) else np.nan,
                   episodes=len(ep), thr=th["thr"], bal=th["bal"],
                   min_frac=round(float(v.loc[v.all_connected, "cluster_frac"].min()), 3)
                            if v.all_connected.any() else np.nan)
        m = mat.set_index("doy").loc[v.doy]
        for i, a in enumerate(masks):
            for b in list(masks)[i + 1:]:
                both = (m[a] > 0) & (m[b] > 0)
                row[f"pair_{a}-{b}"] = round(float(((m[a] == m[b]) & both).sum() / max(len(m), 1)), 3)
        rows.append(row)
        conn.assign(run=name, year=year).to_csv(
            os.path.join(OUT, f"dates_{name.replace('=', '_')}.csv"),
            mode="a", index=False, header=not os.path.exists(
                os.path.join(OUT, f"dates_{name.replace('=', '_')}.csv")))
    print(f"  {year} done, {time.time()-t0:.0f} s elapsed")

by_year = pd.DataFrame(rows)
by_year.to_csv(os.path.join(OUT, "sensitivity_by_year.csv"), index=False)

agg = (by_year.groupby(["run", "param", "value"], sort=False)
       .agg(n_merged=("n_merged", "sum"), n_ws=("n_ws", "sum"), episodes=("episodes", "sum"),
            thr_median=("thr", "median"), bal_median=("bal", "median"),
            min_frac=("min_frac", "min"),
            window_min=("merge_first", "min"), window_max=("merge_last", "max")).reset_index())
agg["ws_merge_frac"] = (agg.n_merged / agg.n_ws).round(3)
base = agg.loc[agg.run == "baseline"].iloc[0]
agg["n_merged_change_pct"] = (100 * (agg.n_merged - base.n_merged) / base.n_merged).round(1)
agg.to_csv(os.path.join(OUT, "sensitivity_pooled.csv"), index=False)
print("\n=== POOLED OVER YEARS ===")
print(agg[["run", "n_merged", "n_merged_change_pct", "ws_merge_frac", "episodes",
           "thr_median", "bal_median", "min_frac"]].to_string(index=False))
print(f"\nbaseline must reproduce the published total of merged dates: {int(base.n_merged)} "
      f"(published: {int(summ['n_merged_dates'].sum())})")
print(f"\n✓ wrote {OUT}")
