#!/usr/bin/env python3
"""
==============================================================================
HAOR VALIDATION: LABEL MAPS AND CONTACT ZONES  (haor_validation_labels.py)
==============================================================================
Why this exists (meeting, 1 October): the connectivity result was only ever
shown as tables and curves. Both professors asked to SEE, on real maps, what
the algorithm saw on a given date: which pixels were water, which were filled,
which were thrown away, and which water body each haor belonged to. The
notebook cannot draw that unless the per-date rasters exist. This script makes
them, once, on the server.

For every year it re-runs the production connectivity step (through
haor_validation_core.py, identical to analyze_year cell 4) and writes:

  validation/labels/labels_<Y>.tif     uint16, one band per date: body label of
                                       every kept water pixel, 0 = none,
                                       65535 = date skipped by the coverage guard
  validation/labels/codes_<Y>.tif      uint8, one band per date: what each pixel
                                       was (observed dry / observed flooded /
                                       filled / removed by the guards / unseen)
  validation/labels/contacts_<Y>.csv   one row per date per ADJACENT pair: did the
                                       pair merge, and did the shared body actually
                                       cross their common boundary, and how much of
                                       that crossing was made of FILLED pixels
  validation/labels/check_<Y>.txt      comparison against the published CSVs
  validation/adjacency.csv             measured shared boundary length per pair

The check matters most: if the recomputed states do not match the published
connectivity_curve_<Y>.csv exactly, every map would show something other than
what the results are based on. The script prints the mismatch count per year.

High resolution mode (after haor_export_flood_stack_hires.py):
  python haor_validation_labels.py --stack-dir /work/a06/wasif/haor_flood_pooled_30m --res 30
writes to validation/labels_hires_haor_flood_pooled_30m_physical/ with the guards scaled to the same
PHYSICAL size (10 px at 100 m -> same 0.1 km2; 800 m ring; 300 m bridge).
Add --bridge-mode pixel to keep one opening iteration instead.

Usage:  python haor_validation_labels.py            (all years, 100 m)
        python haor_validation_labels.py --years 2024 2025
==============================================================================
"""
import sys, os, glob, re, argparse, math

venv_site = "/work/a06/wasif/.venv/lib/python3.12/site-packages"
if venv_site not in sys.path:
    sys.path.insert(0, venv_site)
os.environ["PATH"] = "/work/a06/wasif/.venv/bin:" + os.environ.get("PATH", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # haor_validation_core.py

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import haor_validation_core as core

# ============================ SETTINGS ============================
BASE          = "/work/a06/wasif/haor_flood_analysis_hydroyear"
ANALYSIS_ROOT = os.path.join(BASE, "analysis")
HAORS_SHP     = "/work/a06/wasif/haor_flood_analysis/digitization/haors_manual.shp"
VAL_ROOT      = os.path.join(ANALYSIS_ROOT, "validation")
CONTACT_PX_M  = 300      # contact zone reaches this far into each haor of a pair
# =================================================================

ap = argparse.ArgumentParser()
ap.add_argument("--years", nargs="*", type=int, default=None)
ap.add_argument("--stack-dir", default=BASE)
ap.add_argument("--res", type=float, default=100.0, help="pixel size of the stacks in metres")
ap.add_argument("--bridge-mode", choices=["physical", "pixel"], default="physical")
args = ap.parse_args()

hires = abs(args.res - 100.0) > 1e-6 or os.path.abspath(args.stack_dir) != os.path.abspath(BASE)
p = dict(core.DEFAULTS)
if hires:
    f = 100.0 / args.res
    p["MIN_PX"] = int(round(core.DEFAULTS["MIN_PX"] * f * f))       # same area, 0.1 km2
    p["BUFFER_PX"] = int(round(800.0 / args.res))                     # same 800 m ring
    if args.bridge_mode == "physical":                                 # links < ~300 m removed
        p["BRIDGE"] = int(math.ceil((300.0 / args.res - 1) / 2))
    tag = f"labels_hires_{os.path.basename(os.path.normpath(args.stack_dir))}_{args.bridge_mode}"
else:
    tag = "labels"
OUT = os.path.join(VAL_ROOT, tag)
os.makedirs(OUT, exist_ok=True)
contact_k = max(1, int(round(CONTACT_PX_M / args.res)))
print(f"stacks : {args.stack_dir}")
print(f"output : {OUT}")
print(f"params : {p}   contact zone: {contact_k} px")

os.environ["SHAPE_RESTORE_SHX"] = "YES"
haors = gpd.read_file(HAORS_SHP)
haors = haors[haors["haor_id"] != "DUMMY"].copy()
if (~haors.geometry.is_valid).any():
    haors["geometry"] = haors.geometry.buffer(0)
adj = core.adjacency(haors)
adj.to_csv(os.path.join(VAL_ROOT, "adjacency.csv"), index=False)
print("\nmeasured adjacency (shared boundary within 50 m):")
print(adj.to_string(index=False))

stacks = sorted(glob.glob(os.path.join(args.stack_dir, "flood_stack_cluster_*.tif")))
years = sorted(int(re.search(r"(\d{4})", os.path.basename(s)).group(1)) for s in stacks)
if args.years:
    years = [y for y in years if y in args.years]
assert years, f"no flood_stack_cluster_<year>.tif in {args.stack_dir}"


def open_out(path, meta, count, dtype):
    prof = dict(driver="GTiff", height=meta["shape"][0], width=meta["shape"][1],
                count=count, dtype=dtype, crs=meta["crs"], transform=meta["transform"],
                compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256,
                BIGTIFF="IF_SAFER")
    return rasterio.open(path, "w", **prof)


for year in years:
    print(f"\n=== {year} ===")
    path = os.path.join(args.stack_dir, f"flood_stack_cluster_{year}.tif")
    _, _, meta0 = core.stack_dates(path, year)
    masks = core.haor_masks(haors, meta0)
    ids = list(masks)
    zones = core.contact_zones(masks, adj, contact_k)
    crecs, cmat, rows = [], [], []
    # one band at a time: read, fill, label, write. Memory stays at a few bands.
    with open_out(os.path.join(OUT, f"labels_{year}.tif"), meta0, meta0["count"], "uint16") as dl, \
         open_out(os.path.join(OUT, f"codes_{year}.tif"), meta0, meta0["count"], "uint8") as dc:
        for k, (day, raw, filled, r, study, meta) in enumerate(core.iter_stack(path, year, masks, p), 1):
            d = core.doy_to_date(year, day)
            desc = f"d{d:%Y%m%d}_doy{int(day)}"
            crecs.append(dict(doy=day, n_components=r["n"], largest_frac=round(r["largest"], 3),
                              total_wet=r["total"], obs_raw=round(r["obs_raw"], 3),
                              obs_eff=round(r["obs_eff"], 3), all_connected=r["all_connected"]))
            cmat.append(dict(doy=day, **r["comp"]))
            if r["skipped"]:
                dl.write(np.full(meta["shape"], core.LABEL_SKIPPED, np.uint16), k)
                dc.write(np.full(meta["shape"], core.CODE_SKIPPED, np.uint8), k)
            else:
                dl.write(r["lab"].astype(np.uint16), k)
                dc.write(core.code_raster(filled, raw, study, r), k)
                filled_now = (~np.isfinite(raw)) & np.isfinite(filled)
                for pair, z in zones.items():
                    a, b = pair.split("-")
                    ca, cb = r["comp"][a], r["comp"][b]
                    merged = ca > 0 and ca == cb
                    shared = (r["lab"] == ca) & z if merged else np.zeros_like(z)
                    rows.append(dict(year=year, doy=day, date=d.isoformat(), pair=pair,
                                     merged=merged, all_connected=r["all_connected"],
                                     zone_px=int(z.sum()),
                                     zone_obs_frac=round(float(np.isfinite(raw)[z].mean()), 3),
                                     zone_water_px=int(r["water"][z].sum()),
                                     crossing_px=int(shared.sum()),
                                     crossing_filled_px=int((shared & filled_now).sum()),
                                     link="direct" if (merged and shared.any()) else
                                          ("indirect" if merged else "none")))
            dl.set_band_description(k, desc); dc.set_band_description(k, desc)
    conn = pd.DataFrame(crecs).sort_values("doy")
    mat = pd.DataFrame(cmat).sort_values("doy")
    cdf = pd.DataFrame(rows)
    cdf.to_csv(os.path.join(OUT, f"contacts_{year}.csv"), index=False)
    conn.to_csv(os.path.join(OUT, f"connectivity_recomputed_{year}.csv"), index=False)
    mat.to_csv(os.path.join(OUT, f"matrix_recomputed_{year}.csv"), index=False)

    # ---- the check against the published files ----
    lines = []
    if not hires:
        pub_c = os.path.join(ANALYSIS_ROOT, str(year), f"connectivity_curve_{year}.csv")
        pub_m = os.path.join(ANALYSIS_ROOT, str(year), f"haor_connection_matrix_{year}.csv")
        if os.path.exists(pub_c) and os.path.exists(pub_m):
            pc = pd.read_csv(pub_c).set_index("doy"); rc = conn.set_index("doy")
            pm = pd.read_csv(pub_m).set_index("doy"); rm = mat.set_index("doy")
            common = pc.index.intersection(rc.index)
            d_state = int((pc.loc[common, "all_connected"].astype(bool) !=
                           rc.loc[common, "all_connected"].astype(bool)).sum())
            d_ncomp = int((pc.loc[common, "n_components"] != rc.loc[common, "n_components"]).sum())
            d_mat = int((pm.loc[common, ids].values != rm.loc[common, ids].values).any(axis=1).sum())
            lines += [f"dates published {len(pc)}, recomputed {len(rc)}, common {len(common)}",
                      f"all_connected mismatches : {d_state}",
                      f"n_components mismatches  : {d_ncomp}",
                      f"matrix row mismatches    : {d_mat}"]
            ok = d_state == 0 and d_ncomp == 0 and d_mat == 0 and len(common) == len(pc)
            lines.append("IDENTICAL to the published output" if ok else
                         "DIFFERENT from the published output: re-run cells 0 to 6 of "
                         "haor_flood_hydroyear.ipynb, or the stack on disk has changed")
        else:
            lines.append("published CSVs not found; nothing to compare")
    else:
        lines.append("high resolution run: compare against 100 m in the notebook (V10)")
    open(os.path.join(OUT, f"check_{year}.txt"), "w").write("\n".join(lines) + "\n")
    for l in lines:
        print("  " + l)
    nm = int(conn["all_connected"].sum())
    if len(cdf):
        mg = cdf[cdf.merged]
        direct = (mg.link == "direct").mean() if len(mg) else np.nan
        fillshare = (mg.crossing_filled_px.sum() / max(mg.crossing_px.sum(), 1)) if len(mg) else np.nan
        print(f"  merged dates {nm} | merged adjacent-pair observations {len(mg)} | "
              f"crossing their own boundary {100*direct:.0f}% | "
              f"crossing pixels that were FILLED {100*fillshare:.1f}%")

print(f"\n✓ done. Open haor_visual_validation.ipynb next.")
