#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SWOT L2_HR_Raster: RELAXED WSE mosaic + QC reason-map (per cycle, path)

UPDATED per your request:
- REMOVED the "Out of physical range" filter entirely:
    * No more wse_min / wse_max in QC_RELAXED
    * No more m_phys mask
    * No QC_LABEL 16
    * No QC_COLOR for 16
    * No "valid_relaxed &= ~m_phys"
    * No "qc[m_phys] = 16"

Everything else remains the same:
- WSE panel uses relaxed filters (wse_qual<=2, wse_uncert<10, water_area_qual<=2, water_frac>0.01)
- QC panel still explains missing/suspicious pixels using bitwise flags + proxy shoreline mixed pixels.

"""

import os
import glob
from copy import copy
from collections import defaultdict

# Use non-interactive backend (important on HPC / no-display)
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

import xarray as xr
import numpy as np

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


# ---------------------------
# CONFIG
# ---------------------------

DATA_DIR = "/work/a06/wasif/swot_raster_data"
RESULTS_DIR = "/work/a06/wasif/swot_results_wse/per_cycle_test6"
os.makedirs(RESULTS_DIR, exist_ok=True)

# AOI bbox: (west, south, east, north)
BBOX_BD = (89.8, 23.5, 92.8, 25.5)

# "250" or "100"
RES = "100"

START_DATE = "2022-07-20"
END_DATE   = "2025-10-31"

# Fixed WSE color scale
WSE_VMIN = 2.0
WSE_VMAX = 5.0


# ---------------------------
# RELAXED WSE VALIDITY (match your older behavior)
# ---------------------------

QC_RELAXED = dict(
    max_wse_qual=2,          # allow suspect+degraded
    max_wse_uncert=10.0,     # meters (your relaxed code)
    max_water_area_qual=2,   # allow suspect+degraded
    min_water_frac=0.01,     # allow tiny water fraction like before
)

# ONLY for labeling shoreline/mixed proxy in QC map.
SHORELINE_THR = 0.70  # 0<wf<0.70 = mixed/near-land proxy, wf>=0.70 = open-water-like


# ---------------------------
# BIT MASKS (SWOT Raster Product Description)
# ---------------------------

WSE_BITS = dict(
    # SUSPECT
    large_uncert_suspect=32,
    dark_water_suspect=64,
    bright_land=128,
    low_coherence_water_suspect=256,
    spec_ringing_prior_water_suspect=512,
    spec_ringing_prior_land_suspect=1024,
    few_pixels=4096,
    far_range_suspect=8192,
    near_range_suspect=16384,

    # DEGRADED
    classification_qual_degraded=262144,
    geolocation_qual_degraded=524288,
    dark_water_degraded=1048576,
    low_coherence_water_degraded=2097152,
    spec_ringing_prior_land_degraded=4194304,

    # BAD / NOT OBSERVED
    value_bad=16777216,
    outside_data_window=67108864,
    no_pixels=268435456,
    outside_scene_bounds=536870912,
    inner_swath=1073741824,
    missing_karin_data=2147483648,
)

# ---------------------------
# QC CATEGORY MAP (discrete colors)
# ---------------------------

QC_LABELS = {
    0:  "Clean water (passes relaxed + no major warnings)",
    1:  "Inner swath (nadir gap)",
    2:  "Outside scene bounds",
    3:  "Outside data window",
    4:  "Missing KaRIn data",
    5:  "No PIXC pixels",
    6:  "Not water / null",

    7:  "Near-land mixed pixel (0<water_frac<shoreline_thr) [proxy]",
    8:  "Dark water flag (suspect/degraded) or dark_frac high",
    9:  "Low coherence flag (suspect/degraded)",
    10: "Specular ringing flag",
    11: "Bright land contamination",
    12: "Near/Far range suspect",
    13: "Few pixels (wse_qual_bitwise few_pixels) or n_wse_pix small",
    14: "Degraded class/geo or large_uncert_suspect / high uncert",

    15: "Bad flags (value_bad or qual==3)",
}

QC_COLORS = [
    "#2ca02c",  # 0 clean water (green)
    "#ffffff",  # 1 inner swath (white)
    "#dddddd",  # 2 outside scene
    "#cccccc",  # 3 outside window
    "#bbbbbb",  # 4 missing karin
    "#aaaaaa",  # 5 no pixels
    "#000000",  # 6 not water/null (black)

    "#ff7f0e",  # 7 near-land mixed (orange)
    "#1f77b4",  # 8 dark water (blue)
    "#d62728",  # 9 low coherence (red)
    "#9467bd",  # 10 specular (purple)
    "#8c564b",  # 11 bright land (brown)
    "#e377c2",  # 12 range (pink)
    "#7f7f7f",  # 13 few pixels (gray)
    "#bcbd22",  # 14 degraded/high-uncert (olive)

    "#17becf",  # 15 bad flags (cyan)
]
QC_CMAP = ListedColormap(QC_COLORS, name="qcmap")
QC_NORM = BoundaryNorm(np.arange(-0.5, len(QC_COLORS)+0.5, 1), QC_CMAP.N)


# ---------------------------
# UTILITIES
# ---------------------------

def parse_ids_from_name(fn):
    import os.path as op
    name = op.basename(fn)
    parts = name.split("_")

    cycle = None
    path_num = None
    tile_id = None
    date_str = "unknown"

    if len(parts) > 10 and parts[10].isdigit():
        try:
            cycle = int(parts[10])
        except Exception:
            cycle = None

    if len(parts) > 11 and parts[11].isdigit():
        try:
            path_num = int(parts[11])
        except Exception:
            path_num = None

    if len(parts) > 12:
        tile_id = parts[12]

    date_token = None
    for p in parts:
        if p.startswith("20") and "T" in p:
            date_token = p
            break
    if date_token is not None and len(date_token) >= 8:
        d = date_token[0:8]
        date_str = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"

    return cycle, path_num, tile_id, date_str


def in_date_window(date_str, start=START_DATE, end=END_DATE):
    if date_str == "unknown":
        return True
    d  = date_str.replace("-", "")
    s0 = start.replace("-", "")
    s1 = end.replace("-", "")
    return (s0 <= d <= s1)


# ---------------------------
# FILE INDEX
# ---------------------------

def build_file_index():
    pattern = os.path.join(DATA_DIR, f"*{RES}m*.nc")
    all_nc_files = sorted(glob.glob(pattern))
    if not all_nc_files:
        raise FileNotFoundError(f"No NetCDF files with '{RES}m' in {DATA_DIR}")

    print(f"Found {len(all_nc_files)} SWOT Raster {RES} m files in total")

    index = []
    for fn in all_nc_files:
        try:
            ds = xr.open_dataset(fn)
        except Exception as e:
            print(f"  !! Could not open {os.path.basename(fn)}: {e}")
            continue

        if "longitude" in ds:
            lon = ds["longitude"].values
            lat = ds["latitude"].values
        elif "lon" in ds:
            lon = ds["lon"].values
            lat = ds["lat"].values
        else:
            print(f"  !! No longitude variable in {os.path.basename(fn)}; skipping.")
            ds.close()
            continue

        if isinstance(lon, np.ma.MaskedArray):
            lon = lon.filled(np.nan)
        if isinstance(lat, np.ma.MaskedArray):
            lat = lat.filled(np.nan)

        if not np.isfinite(lon).any() or not np.isfinite(lat).any():
            print(f"  !! All lon/lat NaN in {os.path.basename(fn)}; skipping.")
            ds.close()
            continue

        lon_min = float(np.nanmin(lon))
        lon_max = float(np.nanmax(lon))
        lat_min = float(np.nanmin(lat))
        lat_max = float(np.nanmax(lat))
        lon_mid = 0.5 * (lon_min + lon_max)

        cycle, path_num, tile_id, date_str = parse_ids_from_name(fn)

        index.append(dict(
            fn=fn,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_mid=lon_mid,
            cycle=cycle,
            path=path_num,
            tile=tile_id,
            date=date_str,
        ))
        ds.close()

    print("\nExample entries (first 5):")
    for e in index[:5]:
        print(
            f"  {os.path.basename(e['fn'])}  "
            f"cycle={e['cycle']}, path={e['path']}, tile={e['tile']}, "
            f"date={e['date']}, "
            f"lon[{e['lon_min']:.2f},{e['lon_max']:.2f}] "
            f"lat[{e['lat_min']:.2f},{e['lat_max']:.2f}]"
        )

    return index


def filter_index_by_aoi_and_time(index):
    lon_min_aoi, lat_min_aoi, lon_max_aoi, lat_max_aoi = BBOX_BD
    selected = []
    for e in index:
        if not in_date_window(e["date"]):
            continue
        if (
            e["lon_max"] < lon_min_aoi or e["lon_min"] > lon_max_aoi or
            e["lat_max"] < lat_min_aoi or e["lat_min"] > lat_max_aoi
        ):
            continue
        selected.append(e)

    print(f"\nFiles after AOI + time filter: {len(selected)}")
    if not selected:
        raise RuntimeError("No files left after AOI/time filtering.")
    return selected


def group_by_cycle_and_path(entries):
    by_key = defaultdict(list)
    for e in entries:
        cyc = e["cycle"]
        path = e["path"]
        if cyc is None or path is None:
            print(f"  !! Entry with missing cycle/path: {os.path.basename(e['fn'])}")
            continue
        by_key[(cyc, path)].append(e)

    groups_sorted = dict(sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])))
    print(f"\nNumber of (cycle, path) groups in this AOI/time window: {len(groups_sorted)}")
    for (cyc, path), es in list(groups_sorted.items())[:5]:
        dates = sorted({e["date"] for e in es})
        print(
            f"  cycle {cyc:03d}, path {path:03d}: "
            f"{len(es)} tiles, dates {dates[0]} ... {dates[-1]}"
        )
    return groups_sorted


# ---------------------------
# QC HELPERS
# ---------------------------

def _get_var(ds, name, fill_to_nan=True):
    if name not in ds:
        return None
    v = ds[name].values
    if isinstance(v, np.ma.MaskedArray):
        v = v.filled(np.nan)
    if fill_to_nan:
        try:
            fv = ds[name]._FillValue
            v = np.where(v == fv, np.nan, v)
        except Exception:
            pass
    return v


def _has_bit(arr, bit):
    a = np.array(arr)
    if np.issubdtype(a.dtype, np.floating):
        a = np.where(np.isnan(a), 0, a).astype(np.uint64)
    else:
        a = a.astype(np.uint64)
    return (a & np.uint64(bit)) != 0


def extract_wse_relaxed_and_qc(ds, prof=QC_RELAXED, shoreline_thr=SHORELINE_THR):
    if "wse" not in ds:
        return None, None

    wse = _get_var(ds, "wse", fill_to_nan=True)
    wse_qual = _get_var(ds, "wse_qual", fill_to_nan=False)
    wse_unc = _get_var(ds, "wse_uncert", fill_to_nan=True)

    wf = _get_var(ds, "water_frac", fill_to_nan=True)
    waq = _get_var(ds, "water_area_qual", fill_to_nan=False)

    wse_bw = _get_var(ds, "wse_qual_bitwise", fill_to_nan=False)

    n_wse = _get_var(ds, "n_wse_pix", fill_to_nan=True)
    dark_frac = _get_var(ds, "dark_frac", fill_to_nan=True)

    shape = wse.shape

    if wse_qual is None: wse_qual = np.zeros(shape, dtype=np.uint8)
    if wse_unc is None:  wse_unc = np.full(shape, np.nan)
    if wf is None:       wf = np.full(shape, np.nan)
    if waq is None:      waq = np.zeros(shape, dtype=np.uint8)
    if wse_bw is None:   wse_bw = np.zeros(shape, dtype=np.uint64)
    if n_wse is None:    n_wse = np.full(shape, np.nan)
    if dark_frac is None: dark_frac = np.full(shape, np.nan)

    is_wse_present = np.isfinite(wse)

    # NOT OBSERVED / hard-bad
    m_inner = _has_bit(wse_bw, WSE_BITS["inner_swath"])
    m_out_scene = _has_bit(wse_bw, WSE_BITS["outside_scene_bounds"])
    m_out_window = _has_bit(wse_bw, WSE_BITS["outside_data_window"])
    m_missing_karin = _has_bit(wse_bw, WSE_BITS["missing_karin_data"])
    m_no_pixels = _has_bit(wse_bw, WSE_BITS["no_pixels"])
    m_value_bad = _has_bit(wse_bw, WSE_BITS["value_bad"])

    # Water-ness
    m_not_water = ~np.isfinite(wf) | (wf <= 0.0)
    m_near_land_mixed = np.isfinite(wf) & (wf > 0.0) & (wf < shoreline_thr)

    # Warning flags (labeled only)
    m_darkbit = _has_bit(wse_bw, WSE_BITS["dark_water_suspect"]) | _has_bit(wse_bw, WSE_BITS["dark_water_degraded"])
    m_darkfrac = np.isfinite(dark_frac) & (dark_frac >= 0.20)

    m_lowcoh = _has_bit(wse_bw, WSE_BITS["low_coherence_water_suspect"]) | _has_bit(wse_bw, WSE_BITS["low_coherence_water_degraded"])
    m_spec = (_has_bit(wse_bw, WSE_BITS["spec_ringing_prior_water_suspect"]) |
              _has_bit(wse_bw, WSE_BITS["spec_ringing_prior_land_suspect"]) |
              _has_bit(wse_bw, WSE_BITS["spec_ringing_prior_land_degraded"]))
    m_bright_land = _has_bit(wse_bw, WSE_BITS["bright_land"])
    m_range = _has_bit(wse_bw, WSE_BITS["far_range_suspect"]) | _has_bit(wse_bw, WSE_BITS["near_range_suspect"])
    m_fewpix_bit = _has_bit(wse_bw, WSE_BITS["few_pixels"])

    m_class_geo_degraded = _has_bit(wse_bw, WSE_BITS["classification_qual_degraded"]) | _has_bit(wse_bw, WSE_BITS["geolocation_qual_degraded"])
    m_large_unc_sus = _has_bit(wse_bw, WSE_BITS["large_uncert_suspect"])

    # Numeric warnings
    m_unc_hi = np.isfinite(wse_unc) & (wse_unc >= 3.0)  # label only
    m_unc_fail = np.isfinite(wse_unc) & (wse_unc >= prof["max_wse_uncert"])
    m_npix_low = np.isfinite(n_wse) & (n_wse < 3)       # label only

    # Summary “bad”
    m_wsequal_bad = (wse_qual == 3)
    m_waqual_bad = (waq == 3)

    # -----------------------
    # RELAXED validity mask (NO physical range filter anymore)
    # -----------------------
    valid_relaxed = np.ones(shape, dtype=bool)
    valid_relaxed &= is_wse_present
    valid_relaxed &= (wse_qual <= prof["max_wse_qual"])
    valid_relaxed &= (~m_wsequal_bad)
    valid_relaxed &= (waq <= prof["max_water_area_qual"])
    valid_relaxed &= (~m_waqual_bad)
    valid_relaxed &= np.isfinite(wf) & (wf > prof["min_water_frac"]) & (wf <= 1.0)

    # If wse_uncert missing (NaN), don't kill it; apply only when finite
    valid_relaxed &= (~m_unc_fail) | (~np.isfinite(wse_unc))

    # Hard exclusions
    hard_exclude = (
        m_inner | m_out_scene | m_out_window | m_missing_karin | m_no_pixels |
        m_value_bad
    )
    valid_relaxed &= ~hard_exclude

    wse_plot = np.where(valid_relaxed, wse, np.nan).astype(np.float32)

    # -----------------------
    # QC category map (NO label 16 anymore)
    # -----------------------
    qc = np.full(shape, 6, dtype=np.uint8)  # default not water/null
    qc[~is_wse_present] = 6

    qc[m_inner] = 1
    qc[m_out_scene] = 2
    qc[m_out_window] = 3
    qc[m_missing_karin] = 4
    qc[m_no_pixels] = 5

    qc[m_value_bad | m_wsequal_bad | m_waqual_bad] = 15
    qc[m_not_water] = 6

    qc[m_near_land_mixed] = 7
    qc[m_darkbit | m_darkfrac] = 8
    qc[m_lowcoh] = 9
    qc[m_spec] = 10
    qc[m_bright_land] = 11
    qc[m_range] = 12

    qc[m_fewpix_bit | m_npix_low] = 13
    qc[m_class_geo_degraded | m_large_unc_sus | m_unc_hi] = 14

    major_warn = (m_darkbit | m_darkfrac | m_lowcoh | m_spec | m_bright_land |
                  m_range | m_fewpix_bit | m_class_geo_degraded | m_large_unc_sus | m_unc_hi)

    clean = valid_relaxed & np.isfinite(wf) & (wf >= shoreline_thr) & (~major_warn)
    qc[clean] = 0

    return wse_plot, qc


# ---------------------------
# PLOT ONE PATH (MOSAIC OF TILES) WITH QC MAP
# ---------------------------

def plot_path_wse(cycle, path, entries, vmin, vmax, out_dir):
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(16, 8))

    ax_wse = plt.subplot(1, 2, 1, projection=proj)
    ax_qc  = plt.subplot(1, 2, 2, projection=proj)

    lon_min, lat_min, lon_max, lat_max = BBOX_BD
    for ax in (ax_wse, ax_qc):
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "rivers_lake_centerlines", "10m",
                edgecolor="k", facecolor="none"
            ),
            linewidth=0.4,
            zorder=3
        )

    gl = ax_wse.gridlines(
        crs=proj, draw_labels=True, linewidth=0.5,
        color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MultipleLocator(1.0)
    gl.ylocator = mticker.MultipleLocator(0.5)

    base_cmap = mpl.cm.get_cmap("viridis")
    cmap_nan = copy(base_cmap)
    cmap_nan.set_bad((0, 0, 0, 0))

    any_plotted = False
    last_img = None
    counts_total = defaultdict(int)

    for e in entries:
        fn = e["fn"]
        tile_id = e["tile"]
        print(f"  cycle {cycle:03d}, path {path:03d}: tile {tile_id}, {os.path.basename(fn)}")

        try:
            ds = xr.open_dataset(fn)
        except Exception as exc:
            print(f"    !! Failed to open {os.path.basename(fn)}: {exc}")
            continue

        if "longitude" in ds:
            lon = ds["longitude"].values
            lat = ds["latitude"].values
        elif "lon" in ds:
            lon = ds["lon"].values
            lat = ds["lat"].values
        else:
            print(f"    !! No lon/lat in {os.path.basename(fn)}; skipping")
            ds.close()
            continue

        if isinstance(lon, np.ma.MaskedArray):
            lon = lon.filled(np.nan)
        if isinstance(lat, np.ma.MaskedArray):
            lat = lat.filled(np.nan)

        if not np.isfinite(lon).any() or not np.isfinite(lat).any():
            print("    !! All lon/lat NaN; skipping")
            ds.close()
            continue

        tile_lon_min = float(np.nanmin(lon))
        tile_lon_max = float(np.nanmax(lon))
        tile_lat_min = float(np.nanmin(lat))
        tile_lat_max = float(np.nanmax(lat))

        if (
            tile_lon_max < lon_min or tile_lon_min > lon_max or
            tile_lat_max < lat_min or tile_lat_min > lat_max
        ):
            ds.close()
            print("    -> tile outside AOI; skipping.")
            continue

        wse_plot, qc_plot = extract_wse_relaxed_and_qc(ds, prof=QC_RELAXED, shoreline_thr=SHORELINE_THR)
        ds.close()

        if wse_plot is None or qc_plot is None:
            continue

        lat_col0 = lat[:, 0]
        if lat_col0[0] > lat_col0[-1]:
            wse_plot = wse_plot[::-1, :]
            qc_plot  = qc_plot[::-1, :]

        uniq, cnts = np.unique(qc_plot, return_counts=True)
        for u, c in zip(uniq, cnts):
            counts_total[int(u)] += int(c)

        wse_masked = np.ma.masked_invalid(wse_plot)
        img = ax_wse.imshow(
            wse_masked,
            origin="lower",
            extent=[tile_lon_min, tile_lon_max, tile_lat_min, tile_lat_max],
            transform=proj,
            cmap=cmap_nan,
            vmin=vmin,
            vmax=vmax,
            zorder=4
        )
        last_img = img

        ax_qc.imshow(
            qc_plot,
            origin="lower",
            extent=[tile_lon_min, tile_lon_max, tile_lat_min, tile_lat_max],
            transform=proj,
            cmap=QC_CMAP,
            norm=QC_NORM,
            zorder=4
        )

        any_plotted = True

    if not any_plotted:
        print(f"  -> No valid tiles for cycle {cycle:03d}, path {path:03d}")
        plt.close(fig)
        return

    cbar = plt.colorbar(last_img, ax=ax_wse, orientation="vertical", shrink=0.75)
    cbar.set_label("WSE (m)")

    present = [k for k in sorted(counts_total.keys()) if (counts_total[k] > 0 and k in QC_LABELS)]
    handles = [mpatches.Patch(color=QC_COLORS[k], label=f"{k:02d}: {QC_LABELS[k]}") for k in present]
    ax_qc.legend(handles=handles, loc="lower left", fontsize=8, frameon=True)

    dates = sorted({e["date"] for e in entries if e["date"] != "unknown"})
    if dates:
        date_label = dates[0] if dates[0] == dates[-1] else f"{dates[0]} to {dates[-1]}"
    else:
        date_label = "unknown"

    ax_wse.set_title(
        f"RELAXED WSE (matches your old filters)\n"
        f"wse_qual<=2, wse_uncert<10, water_area_qual<=2, water_frac>0.01\n"
        f"cycle {cycle:03d}, path {path:03d} ({date_label}), Raster {RES} m",
        fontsize=10
    )
    ax_qc.set_title(
        f"QC category map (why missing / what type)\n"
        f"(includes near-land proxy via water_frac<{SHORELINE_THR})\n"
        f"cycle {cycle:03d}, path {path:03d} ({date_label}), Raster {RES} m",
        fontsize=10
    )

    plt.tight_layout()

    safe_date = date_label.replace(" ", "").replace("to", "_").replace("-", "")
    out_name = f"swot_wse_relaxed_qc_cycle{cycle:03d}_path{path:03d}_{safe_date}_{RES}m.png"
    out_path = os.path.join(out_dir, out_name)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print("  QC category counts (present):")
    for k in present:
        print(f"    {k:02d} {QC_LABELS[k]}: {counts_total[k]}")
    print(f"  Saved: {out_path}")


# ---------------------------
# MAIN
# ---------------------------

def main():
    index = build_file_index()
    selected = filter_index_by_aoi_and_time(index)
    groups = group_by_cycle_and_path(selected)

    for (cycle, path), entries in groups.items():
        print(f"\n=== Processing cycle {cycle:03d}, path {path:03d} ({len(entries)} tiles) ===")
        plot_path_wse(
            cycle=cycle,
            path=path,
            entries=entries,
            vmin=WSE_VMIN,
            vmax=WSE_VMAX,
            out_dir=RESULTS_DIR
        )


if __name__ == "__main__":
    main()