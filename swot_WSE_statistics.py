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

# Where to save output figures
RESULTS_DIR = "/work/a06/wasif/swot_results_wse/mean_min_max"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Big AOI that covers west-side river + Haor
# (west, south, east, north)
BBOX_BD = (89.8, 23.5, 92.8, 25.5)

# Choose resolution to plot: "250" or "100"
RES = "250"   # change to "100" if you want the 100 m product


# ---------------------------
# FILE INDEX
# ---------------------------

def build_file_info():
    """
    Scan all SWOT raster files at given resolution and
    compute (filename, lon_min, lon_max, lon_mid) for each.
    """
    pattern = os.path.join(DATA_DIR, f"*{RES}m*.nc")
    all_nc_files = sorted(glob.glob(pattern))
    if not all_nc_files:
        raise FileNotFoundError(f"No NetCDF files with '{RES}m' in {DATA_DIR}")

    print(f"Found {len(all_nc_files)} SWOT Raster {RES} m files in total:")

    file_info = []  # (filename, lon_min, lon_max, lon_mid)
    for fn in all_nc_files:
        try:
            ds = xr.open_dataset(fn)
        except Exception as e:
            print(f"  !! Could not open {os.path.basename(fn)}: {e}")
            continue

        if "longitude" in ds:
            lon = ds["longitude"].values
        elif "lon" in ds:
            lon = ds["lon"].values
        else:
            print(f"  !! No longitude variable in {os.path.basename(fn)}; skipping.")
            ds.close()
            continue

        lon_min = float(np.nanmin(lon))
        lon_max = float(np.nanmax(lon))
        lon_mid = 0.5 * (lon_min + lon_max)

        file_info.append((fn, lon_min, lon_max, lon_mid))
        ds.close()

    print("\nLongitude coverage per file (first 20 shown):")
    for fn, mn, mx, mid in file_info[:20]:
        print(f"  {os.path.basename(fn)}  lon_min={mn:.3f}, lon_max={mx:.3f}, lon_mid={mid:.3f}")
    if len(file_info) > 20:
        print(f"  ... ({len(file_info) - 20} more files)")

    return file_info


def in_time_window(fn, start="2022-07-20", end="2025-10-31"):
    """
    Crude test using the YYYYMMDD in the filename, e.g. ..._20230730T101422_...

    >>> CHANGE start/end HERE to change the averaging period <<<
    """
    import re
    import os.path as op

    name = op.basename(fn)
    m = re.search(r"_20(\d{6})T", name)
    if not m:
        # If no date in filename, keep it (or you can choose False)
        return True
    datestr = "20" + m.group(1)  # "20230730"
    t0 = start.replace("-", "")
    t1 = end.replace("-", "")
    return (t0 <= datestr <= t1)


def select_nc_files(file_info):
    """
    Split files into west, middle, Haor bands by longitude,
    then apply time window filter, then concatenate.
    """
    west_edge = 89.5   # boundary between west & middle
    haor_edge = 91.0   # boundary between middle & Haor

    west_files = [
        fn for fn, mn, mx, mid in file_info
        if mid < west_edge
    ]

    middle_files = [
        fn for fn, mn, mx, mid in file_info
        if (mid >= west_edge) and (mid <= haor_edge)
    ]

    haor_files = [
        fn for fn, mn, mx, mid in file_info
        if mid > haor_edge
    ]

    # Apply time window filter
    west_files   = [f for f in west_files   if in_time_window(f)]
    middle_files = [f for f in middle_files if in_time_window(f)]
    haor_files   = [f for f in haor_files   if in_time_window(f)]

    nc_files = west_files + middle_files + haor_files

    if not nc_files:
        raise RuntimeError("No files selected for west + middle + Haor. Check lon bands or time window.")

    print("\nSelected files for WSE stats (first 20 shown):")
    for f in nc_files[:20]:
        print("  ", os.path.basename(f))
    if len(nc_files) > 20:
        print(f"  ... ({len(nc_files) - 20} more files)")

    return nc_files


# ---------------------------
# GROUPING BY TILE
# ---------------------------

def tile_key_from_filename(fn):
    """
    Extract a tile identifier from filename by stripping date/time.
    Example:
      SWOT_L2_HR_Raster_250m_UTM46Q_N_x_x_x_001_230_057F_20230729T101350_...nc
    -> SWOT_L2_HR_Raster_250m_UTM46Q_N_x_x_x_001_230_057F
    """
    import re
    import os.path as op

    name = op.basename(fn)
    # Remove from "_20YYYYMMDDT..." onwards
    key = re.sub(r"_20\d{6}T.*", "", name)
    return key


def group_files_by_tile(nc_files):
    """
    Group files that share the same tile key (same UTM tile, different dates).
    Returns: dict[tile_key] -> list of filenames
    """
    groups = defaultdict(list)
    for fn in nc_files:
        key = tile_key_from_filename(fn)
        groups[key].append(fn)
    print(f"\nNumber of tile-groups: {len(groups)}")
    return groups


# ---------------------------
# PER-TILE WSE STATS (MEAN/MIN/MAX)
# ---------------------------

def compute_tile_stats_wse(files):
    """
    For one tile (list of files with same grid), compute:
      - time-mean WSE
      - time-min WSE (per pixel)
      - time-max WSE (per pixel)
    after rigorous QC & physical filters.

    QC rules (if variables exist):
      - wse_qual == 0           (only 'good' WSE)
      - wse_uncert < 1.0 m      (uncertainty filter)
      - water_area_qual <= 1    (geometry & water detection good/suspect)
      - 0.01 < water_frac <= 1  (non-negligible water fraction)
      - WSE in [-50, 200] m (broad physical range)

    Returns:
      lon (2D), lat (2D),
      mean_wse (2D), min_wse (2D), max_wse (2D),
      tile_min_raw (float), tile_max_raw (float)
      OR (None, None, None, None, None, None, None) if all invalid
    """
    lon = lat = None
    sum_data = None
    count = None
    min_data = None
    max_data = None

    # track RAW min/max WSE in this tile (after QC, before any time-ops)
    tile_min_raw = np.inf
    tile_max_raw = -np.inf

    for fn in files:
        try:
            ds = xr.open_dataset(fn)
        except Exception as e:
            print(f"  !! Failed to open {os.path.basename(fn)}: {e}")
            continue

        # lon/lat
        if "longitude" in ds:
            lon_f = ds["longitude"].values
            lat_f = ds["latitude"].values
        elif "lon" in ds:
            lon_f = ds["lon"].values
            lat_f = ds["lat"].values
        else:
            print(f"  !! No longitude/latitude in {os.path.basename(fn)}; skipping this file")
            ds.close()
            continue

        if isinstance(lon_f, np.ma.MaskedArray):
            lon_f = lon_f.filled(np.nan)
        if isinstance(lat_f, np.ma.MaskedArray):
            lat_f = lat_f.filled(np.nan)

        # ---------- WSE variable ----------
        if "wse" not in ds:
            print(f"  !! wse not in {os.path.basename(fn)}; skipping this file")
            ds.close()
            continue

        da = ds["wse"]

        # --- WSE QC flags ---
        if "wse_qual" in ds:
            da = da.where(ds["wse_qual"] == 0)

        if "wse_uncert" in ds:
            da = da.where(ds["wse_uncert"] < 1.0)

        if "water_area_qual" in ds:
            da = da.where(ds["water_area_qual"] <= 1)

        if "water_frac" in ds:
            wf = ds["water_frac"]
            da = da.where((wf > 0.01) & (wf <= 1.0))

        data = da.values
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        # physical range
        data = np.where((data > -50.0) & (data < 200.0), data, np.nan)

        # update RAW min/max for this tile
        f_min = np.nanmin(data)
        f_max = np.nanmax(data)
        if np.isfinite(f_min) and f_min < tile_min_raw:
            tile_min_raw = f_min
        if np.isfinite(f_max) and f_max > tile_max_raw:
            tile_max_raw = f_max

        # Initialize accumulators using first good file
        if lon is None:
            lon = lon_f
            lat = lat_f
            sum_data = np.zeros_like(data, dtype=np.float64)
            count = np.zeros_like(data, dtype=np.int32)
            min_data = np.full_like(data, np.inf, dtype=np.float64)
            max_data = np.full_like(data, -np.inf, dtype=np.float64)
        else:
            if data.shape != sum_data.shape:
                print(f"  !! Shape mismatch in {os.path.basename(fn)}; skipping this file")
                ds.close()
                continue

        # Accumulate only finite values
        valid = np.isfinite(data)
        sum_data[valid] += data[valid]
        count[valid] += 1

        # Per-pixel time-min & time-max
        min_data[valid] = np.minimum(min_data[valid], data[valid])
        max_data[valid] = np.maximum(max_data[valid], data[valid])

        ds.close()

    if lon is None or sum_data is None:
        # All files failed / empty
        return None, None, None, None, None, None, None

    # Compute time-mean WSE
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_data = sum_data / count
    mean_data[count == 0] = np.nan

    # Where no valid obs, set min/max to NaN
    min_data[count == 0] = np.nan
    max_data[count == 0] = np.nan

    return lon, lat, mean_data, min_data, max_data, tile_min_raw, tile_max_raw


# ---------------------------
# PLOTTING (MEAN/MIN/MAX WSE MAPS)
# ---------------------------

def _init_wse_axis(title_suffix):
    """Create one Cartopy axis for WSE plotting over BBOX_BD."""
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=proj)

    # Background: borders, coastlines, rivers
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical",
            "rivers_lake_centerlines",
            "10m",
            edgecolor="k",
            facecolor="none"
        ),
        linewidth=0.4
    )

    lon_min, lat_min, lon_max, lat_max = BBOX_BD
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)

    # Gridlines + labels
    gl = ax.gridlines(
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

    ax.set_title(title_suffix, fontsize=13)

    return fig, ax


def plot_swot_raster_wse_stats(groups, cmap_name, vmin, vmax, out_dir):
    """
    For every tile-group, compute time-mean/min/max WSE with QC,
    then plot three mosaics (mean, time-min, time-max) over BBOX_BD.

    Also prints global statistics:
      - RAW WSE min/max (after QC, before time-ops)
      - MEAN field min/max
      - MIN  field min/max
      - MAX  field min/max
    """

    # NaN-aware colormap (NaNs transparent)
    base_cmap = mpl.cm.get_cmap(cmap_name)
    cmap_nan = copy(base_cmap)
    cmap_nan.set_bad((0, 0, 0, 0))

    # Three figures / axes: mean, min, max
    fig_mean, ax_mean = _init_wse_axis(
        "SWOT mean water surface elevation (WSE) (SWOT Raster {} m, time-mean)".format(RES)
    )
    fig_min, ax_min = _init_wse_axis(
        "SWOT minimum water surface elevation (WSE) (SWOT Raster {} m, time-min)".format(RES)
    )
    fig_max, ax_max = _init_wse_axis(
        "SWOT maximum water surface elevation (WSE) (SWOT Raster {} m, time-max)".format(RES)
    )

    lon_min, lat_min, lon_max, lat_max = BBOX_BD

    any_plotted_mean = False
    any_plotted_min  = False
    any_plotted_max  = False

    last_img_mean = None
    last_img_min  = None
    last_img_max  = None

    # Global stats
    global_raw_min  = np.inf
    global_raw_max  = -np.inf
    global_mean_min = np.inf
    global_mean_max = -np.inf
    global_min_min  = np.inf
    global_min_max  = -np.inf
    global_max_min  = np.inf
    global_max_max  = -np.inf

    for tile_key, files in groups.items():
        print(f"\nProcessing tile group: {tile_key}  ({len(files)} files)")
        (lon, lat,
         mean_wse, min_wse, max_wse,
         tile_min_raw, tile_max_raw) = compute_tile_stats_wse(files)

        if lon is None or mean_wse is None:
            print("  -> no valid WSE data in this tile after QC/stats")
            continue

        # Update global RAW min/max
        if np.isfinite(tile_min_raw) and tile_min_raw < global_raw_min:
            global_raw_min = tile_min_raw
        if np.isfinite(tile_max_raw) and tile_max_raw > global_raw_max:
            global_raw_max = tile_max_raw

        # Update global field stats
        tmin_mean = np.nanmin(mean_wse)
        tmax_mean = np.nanmax(mean_wse)
        if np.isfinite(tmin_mean) and tmin_mean < global_mean_min:
            global_mean_min = tmin_mean
        if np.isfinite(tmax_mean) and tmax_mean > global_mean_max:
            global_mean_max = tmax_mean

        tmin_min = np.nanmin(min_wse)
        tmax_min = np.nanmax(min_wse)
        if np.isfinite(tmin_min) and tmin_min < global_min_min:
            global_min_min = tmin_min
        if np.isfinite(tmax_min) and tmax_min > global_min_max:
            global_min_max = tmax_min

        tmin_max = np.nanmin(max_wse)
        tmax_max = np.nanmax(max_wse)
        if np.isfinite(tmin_max) and tmin_max < global_max_min:
            global_max_min = tmin_max
        if np.isfinite(tmax_max) and tmax_max > global_max_max:
            global_max_max = tmax_max

        # Tile extent
        tile_lon_min = float(np.nanmin(lon))
        tile_lon_max = float(np.nanmax(lon))
        tile_lat_min = float(np.nanmin(lat))
        tile_lat_max = float(np.nanmax(lat))

        # Skip tiles that don't intersect AOI
        if (tile_lon_max < lon_min or tile_lon_min > lon_max or
                tile_lat_max < lat_min or tile_lat_min > lat_max):
            print("  -> tile outside AOI; skipping")
            continue

        # Ensure north is up
        lat_col0 = lat[:, 0]
        if lat_col0[0] > lat_col0[-1]:
            mean_plot = mean_wse[::-1, :]
            min_plot  = min_wse[::-1, :]
            max_plot  = max_wse[::-1, :]
        else:
            mean_plot = mean_wse
            min_plot  = min_wse
            max_plot  = max_wse

        # Mask NaNs so cmap_nan "bad" color applies (transparent)
        mean_plot_masked = np.ma.masked_invalid(mean_plot)
        min_plot_masked  = np.ma.masked_invalid(min_plot)
        max_plot_masked  = np.ma.masked_invalid(max_plot)

        # Mean map
        img = ax_mean.imshow(
            mean_plot_masked,
            origin="lower",
            extent=[tile_lon_min, tile_lon_max, tile_lat_min, tile_lat_max],
            transform=ccrs.PlateCarree(),
            cmap=cmap_nan,
            vmin=vmin,
            vmax=vmax
        )
        last_img_mean = img
        any_plotted_mean = True

        # Min map
        img = ax_min.imshow(
            min_plot_masked,
            origin="lower",
            extent=[tile_lon_min, tile_lon_max, tile_lat_min, tile_lat_max],
            transform=ccrs.PlateCarree(),
            cmap=cmap_nan,
            vmin=vmin,
            vmax=vmax
        )
        last_img_min = img
        any_plotted_min = True

        # Max map
        img = ax_max.imshow(
            max_plot_masked,
            origin="lower",
            extent=[tile_lon_min, tile_lon_max, tile_lat_min, tile_lat_max],
            transform=ccrs.PlateCarree(),
            cmap=cmap_nan,
            vmin=vmin,
            vmax=vmax
        )
        last_img_max = img
        any_plotted_max = True

    # If nothing plotted, bail
    if not (any_plotted_mean or any_plotted_min or any_plotted_max):
        print("No valid WSE data found to plot.")
        plt.close(fig_mean)
        plt.close(fig_min)
        plt.close(fig_max)
        return

    # Colorbars
    if any_plotted_mean:
        cbar = plt.colorbar(last_img_mean, ax=ax_mean, orientation="vertical", shrink=0.75)
        cbar.set_label("Mean WSE (m)")
    if any_plotted_min:
        cbar = plt.colorbar(last_img_min, ax=ax_min, orientation="vertical", shrink=0.75)
        cbar.set_label("Minimum WSE (m)")
    if any_plotted_max:
        cbar = plt.colorbar(last_img_max, ax=ax_max, orientation="vertical", shrink=0.75)
        cbar.set_label("Maximum WSE (m)")

    plt.tight_layout()

    # Output filenames
    bbox_tag = (
        f"lon{lon_min:.1f}-{lon_max:.1f}_"
        f"lat{lat_min:.1f}-{lat_max:.1f}"
    ).replace(".", "p")

    if any_plotted_mean:
        out_mean = os.path.join(out_dir, f"swot_mean_wse_{RES}m_{bbox_tag}.png")
        fig_mean.savefig(out_mean, dpi=300)
        print(f"\nSaved mean WSE figure to: {out_mean}")
    plt.close(fig_mean)

    if any_plotted_min:
        out_min = os.path.join(out_dir, f"swot_min_wse_{RES}m_{bbox_tag}.png")
        fig_min.savefig(out_min, dpi=300)
        print(f"Saved minimum WSE figure to: {out_min}")
    plt.close(fig_min)

    if any_plotted_max:
        out_max = os.path.join(out_dir, f"swot_max_wse_{RES}m_{bbox_tag}.png")
        fig_max.savefig(out_max, dpi=300)
        print(f"Saved maximum WSE figure to: {out_max}")
    plt.close(fig_max)

    # --- PRINT GLOBAL STATS ---
    print("\n=== Global WSE statistics over selected time & region ===")
    print(f"RAW  WSE min (after QC, before time stats): {global_raw_min:.3f} m")
    print(f"RAW  WSE max (after QC, before time stats): {global_raw_max:.3f} m")
    print(f"MEAN field min WSE:                          {global_mean_min:.3f} m")
    print(f"MEAN field max WSE:                          {global_mean_max:.3f} m")
    print(f"MIN  field min WSE (time-min of WSE):        {global_min_min:.3f} m")
    print(f"MIN  field max WSE (time-min of WSE):        {global_min_max:.3f} m")
    print(f"MAX  field min WSE (time-max of WSE):        {global_max_min:.3f} m")
    print(f"MAX  field max WSE (time-max of WSE):        {global_max_max:.3f} m")


# ---------------------------
# MAIN
# ---------------------------

def main():
    # 1) Build metadata index once
    file_info = build_file_info()

    # 2) Select subset of files for west + middle + Haor, within time window
    nc_files = select_nc_files(file_info)

    # 3) Group by tile (same UTM tile, different dates)
    groups = group_files_by_tile(nc_files)

    # 4) Colour scaling for WSE (same range for mean/min/max)
    wse_vmin = -23.0
    wse_vmax = 200.0   # tweak as needed

    # 5) Plot mean, min and max WSE mosaics + print stats
    plot_swot_raster_wse_stats(
        groups=groups,
        cmap_name="viridis",
        vmin=wse_vmin,
        vmax=wse_vmax,
        out_dir=RESULTS_DIR
    )


if __name__ == "__main__":
    main()