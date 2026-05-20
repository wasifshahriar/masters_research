#!/usr/bin/env python

import os
import glob
from copy import copy
from collections import defaultdict

# Use non-interactive backend (important on HPC / no-display)
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import (
    LONGITUDE_FORMATTER,
    LATITUDE_FORMATTER
)

# ---------------------------
# CONFIG
# ---------------------------

# Directory with all downloaded SWOT Raster granules
DATA_DIR = "/work/a06/wasif/swot_raster_data"

# Where to save output figures (one PNG per cycle)
RESULTS_DIR = "/work/a06/wasif/swot_results_wse/per_cycle_test"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Big AOI that covers west-side river + Haor
# (west, south, east, north)
BBOX_BD = (89.8, 23.5, 92.8, 25.5)

# Choose resolution to plot: "250" or "100"
RES = "100"

# Time window for cycles you want to look at
START_DATE = "2022-07-20"
END_DATE   = "2025-10-31"

# WSE colour scale (fixed for all cycles so maps are comparable)
WSE_VMIN = 0.0
WSE_VMAX = 20.0


# ---------------------------
# UTILITIES
# ---------------------------

def parse_cycle_and_date_from_name(fn):
    """
    Try to get cycle number and observation date from filename.

    Example name:
      SWOT_L2_HR_Raster_250m_UTM46Q_N_x_x_x_001_230_057F_20230729T101350_...

    Returns
      cycle (int or None), date_str ('YYYY-MM-DD' or 'unknown')
    """
    import os.path as op

    name = op.basename(fn)
    parts = name.split("_")

    cycle = None
    date_str = "unknown"

    # --- cycle: the '001' / '002' part ---
    if len(parts) > 10 and parts[10].isdigit():
        try:
            cycle = int(parts[10])
        except Exception:
            cycle = None

    # --- date: first token starting with '20' and containing 'T' ---
    date_token = None
    for p in parts:
        if p.startswith("20") and "T" in p:
            date_token = p
            break
    if date_token is not None and len(date_token) >= 8:
        d = date_token[0:8]  # 'YYYYMMDD'
        date_str = f"{d[0:4]}-{d[4:6]}-{d[6:8]}"

    return cycle, date_str


def in_date_window(date_str, start=START_DATE, end=END_DATE):
    """Check if YYYY-MM-DD string falls within [start,end]."""
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
    """
    Scan all SWOT raster files at given resolution and
    build a metadata list for each file:
      dict with keys:
        fn, lon_min, lon_max, lat_min, lat_max, lon_mid, cycle, date
    """
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

        # lon/lat arrays (for extent)
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

        # Skip files where lon/lat are all NaN
        if not np.isfinite(lon).any() or not np.isfinite(lat).any():
            print(f"  !! All lon/lat NaN in {os.path.basename(fn)}; skipping.")
            ds.close()
            continue

        lon_min = float(np.nanmin(lon))
        lon_max = float(np.nanmax(lon))
        lat_min = float(np.nanmin(lat))
        lat_max = float(np.nanmax(lat))
        lon_mid = 0.5 * (lon_min + lon_max)

        # cycle and date
        cycle, date_str = parse_cycle_and_date_from_name(fn)

        entry = dict(
            fn=fn,
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_mid=lon_mid,
            cycle=cycle,
            date=date_str,
        )
        index.append(entry)
        ds.close()

    print("\nExample entries (first 5):")
    for e in index[:5]:
        print(
            f"  {os.path.basename(e['fn'])}  "
            f"cycle={e['cycle']}, date={e['date']}, "
            f"lon[{e['lon_min']:.2f},{e['lon_max']:.2f}] "
            f"lat[{e['lat_min']:.2f},{e['lat_max']:.2f}]"
        )

    return index


def filter_index_by_aoi_and_time(index):
    """
    Keep entries that intersect BBOX_BD and lie in the specified date window.
    """
    lon_min_aoi, lat_min_aoi, lon_max_aoi, lat_max_aoi = BBOX_BD

    selected = []
    for e in index:
        # Time window
        if not in_date_window(e["date"]):
            continue

        # Spatial intersection with AOI
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


def group_by_cycle(entries):
    """
    Group entries by cycle number.
    Returns an Ordered dict: cycle -> list of entries (each entry is a dict).
    """
    by_cycle = defaultdict(list)
    for e in entries:
        cyc = e["cycle"]
        if cyc is None:
            print(f"  !! Entry with no cycle: {os.path.basename(e['fn'])}")
            continue
        by_cycle[cyc].append(e)

    # Sort by cycle number
    cycles_sorted = dict(sorted(by_cycle.items(), key=lambda kv: kv[0]))
    print(f"\nNumber of cycles in this AOI/time window: {len(cycles_sorted)}")
    for cyc, es in list(cycles_sorted.items())[:5]:
        dates = sorted({e["date"] for e in es})
        print(f"  cycle {cyc:03d}: {len(es)} files, dates {dates[0]} ... {dates[-1]}")
    return cycles_sorted


# ---------------------------
# QC HELPER FOR WSE
# ---------------------------

def extract_qc_wse(ds):
    """
    Apply QC to wse from one SWOT Raster granule and
    return 2D numpy array with NaNs where invalid.

    CURRENT QC (relaxed):
      - wse_qual <= 1          (0=good, 1=use with caution)
      - wse_uncert < 3.0 m     (allow more uncertain pixels)
      - water_area_qual <= 1   (good or acceptable water detection)
      - 0.01 < water_frac <= 1 (non-negligible water fraction)
      - -50 < WSE < 200 m      (broad physical range for this region)
    """
    if "wse" not in ds:
        return None

    da = ds["wse"]

    if "wse_qual" in ds:
        da = da.where(ds["wse_qual"] <= 1)

    if "wse_uncert" in ds:
        da = da.where(ds["wse_uncert"] < 3.0)

    if "water_area_qual" in ds:
        da = da.where(ds["water_area_qual"] <= 1)

    if "water_frac" in ds:
        wf = ds["water_frac"]
        da = da.where((wf > 0.01) & (wf <= 1.0))

    data = da.values
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(np.nan)

    # Physical range
    data = np.where((data > -50.0) & (data < 200.0), data, np.nan)

    return data


# ---------------------------
# PLOT ONE CYCLE
# ---------------------------

def plot_cycle_wse(cycle, entries, vmin, vmax, out_dir):
    """
    Mosaic WSE for all granules in one cycle over BBOX_BD
    on a plain white background with rivers/borders.
    """

    proj = ccrs.PlateCarree()  # data CRS (lon/lat)
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=proj)

    lon_min, lat_min, lon_max, lat_max = BBOX_BD
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Background: just white (default figure background)

    # Overlays: borders, coastlines, rivers
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, zorder=2)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "rivers_lake_centerlines",
            "10m",
            edgecolor="k",
            facecolor="none"
        ),
        linewidth=0.4,
        zorder=3
    )

    # Gridlines + labels
    gl = ax.gridlines(
        crs=proj,
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlocator = mticker.MultipleLocator(1.0)
    gl.ylocator = mticker.MultipleLocator(0.5)

    # NaN-aware colormap (NaNs transparent)
    base_cmap = mpl.cm.get_cmap("viridis")
    cmap_nan = copy(base_cmap)
    cmap_nan.set_bad((0, 0, 0, 0))

    any_plotted = False
    last_img = None

    for e in entries:
        fn = e["fn"]
        print(f"  cycle {cycle:03d}: plotting {os.path.basename(fn)}")
        try:
            ds = xr.open_dataset(fn)
        except Exception as exc:
            print(f"    !! Failed to open {os.path.basename(fn)}: {exc}")
            continue

        # lon/lat
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
            print(f"    !! All lon/lat NaN in {os.path.basename(fn)}; skipping")
            ds.close()
            continue

        tile_lon_min = float(np.nanmin(lon))
        tile_lon_max = float(np.nanmax(lon))
        tile_lat_min = float(np.nanmin(lat))
        tile_lat_max = float(np.nanmax(lat))

        # Skip tiles completely outside AOI (double check)
        if (
            tile_lon_max < lon_min or tile_lon_min > lon_max or
            tile_lat_max < lat_min or tile_lat_min > lat_max
        ):
            ds.close()
            continue

        data = extract_qc_wse(ds)
        ds.close()

        if data is None:
            print("    -> no wse variable; skipping")
            continue

        if not np.isfinite(data).any():
            print("    -> all WSE NaN after QC; skipping")
            continue

        # Make sure north is up
        lat_col0 = lat[:, 0]
        if lat_col0[0] > lat_col0[-1]:
            data_plot = data[::-1, :]
        else:
            data_plot = data

        data_plot_masked = np.ma.masked_invalid(data_plot)

        img = ax.imshow(
            data_plot_masked,
            origin="lower",
            extent=[tile_lon_min, tile_lon_max, tile_lat_min, tile_lat_max],
            transform=proj,
            cmap=cmap_nan,
            vmin=vmin,
            vmax=vmax,
            zorder=4   # above rivers/borders
        )

        last_img = img
        any_plotted = True

    if not any_plotted:
        print(f"  -> No valid WSE in cycle {cycle:03d}, skipping figure.")
        plt.close(fig)
        return

    cbar = plt.colorbar(last_img, ax=ax, orientation="vertical", shrink=0.75)
    cbar.set_label("WSE (m)")

    # Build a date label for this cycle
    dates = sorted({e["date"] for e in entries if e["date"] != "unknown"})
    if dates:
        if dates[0] == dates[-1]:
            date_label = dates[0]
        else:
            date_label = f"{dates[0]} to {dates[-1]}"
    else:
        date_label = "unknown dates"

    ax.set_title(
        f"SWOT water surface elevation (WSE)\n"
        f"cycle {cycle:03d} ({date_label}), Raster {RES} m",
        fontsize=12
    )
    plt.tight_layout()

    # Output filename
    safe_date = date_label.replace(" ", "").replace("to", "_").replace("-", "")
    out_name = f"swot_wse_cycle{cycle:03d}_{safe_date}_{RES}m.png"
    out_path = os.path.join(out_dir, out_name)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"  Saved cycle {cycle:03d} figure to: {out_path}")


# ---------------------------
# MAIN
# ---------------------------

def main():
    # 1) Build metadata index for all files
    index = build_file_index()

    # 2) Filter by AOI + time window
    selected = filter_index_by_aoi_and_time(index)

    # 3) Group by cycle
    cycles = group_by_cycle(selected)

    # 4) Loop over cycles and make one WSE map per cycle
    for cycle, entries in cycles.items():
        print(f"\n=== Processing cycle {cycle:03d} ({len(entries)} files) ===")
        plot_cycle_wse(
            cycle=cycle,
            entries=entries,
            vmin=WSE_VMIN,
            vmax=WSE_VMAX,
            out_dir=RESULTS_DIR
        )


if __name__ == "__main__":
    main()