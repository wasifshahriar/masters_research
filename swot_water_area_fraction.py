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
DATA_DIR = "/work/a06/wasif/swot_raster_tmp2"

# Where to save output figures
RESULTS_DIR = "/work/a06/wasif/swot_results"
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

    print("\nSelected files for averaging (first 20 shown):")
    for f in nc_files[:20]:
        print("  ", os.path.basename(f))
    if len(nc_files) > 20:
        print(f"  ... ({len(nc_files) - 20} more files)")

    return nc_files


# ---------------------------
# GROUPING BY TILE & MEAN
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


def compute_tile_mean(files, var_name):
    """
    For one tile (list of files with same grid), compute:
      mean over time for var_name, after QC and physical filters.

    Returns:
      lon (2D), lat (2D), mean_data (2D)   OR (None, None, None) if all invalid
    """
    lon = lat = None
    sum_data = None
    count = None

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

        # data variable
        if var_name not in ds:
            print(f"  !! {var_name} not in {os.path.basename(fn)}; skipping this file")
            ds.close()
            continue

        da = ds[var_name]

        # QC: water_area_qual <= 1
        if "water_area_qual" in ds and var_name in ["water_area", "water_frac", "wse"]:
            qual = ds["water_area_qual"]
            da = da.where(qual <= 1)

        data = da.values
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)

        # Physical filters
        if var_name == "water_area":
            data = np.where(data >= 0, data, np.nan)
        elif var_name == "water_frac":
            data = np.where((data >= 0) & (data <= 1), data, np.nan)

        # Initialize accumulators using first good file
        if lon is None:
            lon = lon_f
            lat = lat_f
            # safety: ensure shapes align
            sum_data = np.zeros_like(data, dtype=np.float64)
            count = np.zeros_like(data, dtype=np.int32)
        else:
            # If shapes differ, skip this file (unlikely for same tile)
            if data.shape != sum_data.shape:
                print(f"  !! Shape mismatch in {os.path.basename(fn)}; skipping this file")
                ds.close()
                continue

        # Accumulate only finite values
        valid = np.isfinite(data)
        sum_data[valid] += data[valid]
        count[valid] += 1

        ds.close()

    if lon is None or sum_data is None:
        # All files failed / empty
        return None, None, None

    # Compute mean
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_data = sum_data / count
    mean_data[count == 0] = np.nan

    return lon, lat, mean_data


# ---------------------------
# PLOTTING MEAN FIELD
# ---------------------------

def plot_swot_raster_mean(groups, var_name, title, cmap_name, vmin, vmax, out_dir):
    """
    For every tile-group, compute mean over time for var_name,
    then plot all mean tiles in one mosaic over BBOX_BD.

    Saves figure to out_dir.
    """

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

    # NaN-aware colormap (optional: make NaNs transparent instead of red)
    base_cmap = mpl.cm.get_cmap(cmap_name)
    cmap_nan = copy(base_cmap)
    # NaN transparent:
    cmap_nan.set_bad((0, 0, 0, 0))
    # or bright red to debug:
    # cmap_nan.set_bad("red")

    any_plotted = False
    last_img = None

    for tile_key, files in groups.items():
        print(f"\nProcessing tile group: {tile_key}  ({len(files)} files)")
        lon, lat, mean_data = compute_tile_mean(files, var_name)

        if lon is None or mean_data is None:
            print("  -> no valid data in this tile after QC/mean")
            continue

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

        # Make sure north is up
        lat_col0 = lat[:, 0]
        if lat_col0[0] > lat_col0[-1]:
            data_plot = mean_data[::-1, :]
        else:
            data_plot = mean_data

        # Mask NaNs so cmap_nan "bad" color applies
        data_plot_masked = np.ma.masked_invalid(data_plot)

        img = ax.imshow(
            data_plot_masked,
            origin="lower",
            extent=[tile_lon_min, tile_lon_max, tile_lat_min, tile_lat_max],
            transform=proj,
            cmap=cmap_nan,
            vmin=vmin,
            vmax=vmax
        )

        last_img = img
        any_plotted = True

    if not any_plotted:
        print(f"No valid data found to plot mean for {var_name}")
        plt.close(fig)
        return

    cbar = plt.colorbar(last_img, ax=ax, orientation="vertical", shrink=0.75)
    if var_name == "water_area":
        cbar.set_label("Mean water area (m²)")
    elif var_name == "water_frac":
        cbar.set_label("Mean water fraction (0–1)")
    else:
        cbar.set_label(f"Mean {var_name}")

    ax.set_title(f"{title} (SWOT Raster {RES} m, time-mean)", fontsize=13)
    plt.tight_layout()

    # Output filename
    bbox_tag = (
        f"lon{lon_min:.1f}-{lon_max:.1f}_"
        f"lat{lat_min:.1f}-{lat_max:.1f}"
    ).replace(".", "p")
    out_name = f"swot_mean_{var_name}_{RES}m_{bbox_tag}.png"
    out_path = os.path.join(out_dir, out_name)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    print(f"\nSaved mean figure to: {out_path}")


# ---------------------------
# MAIN
# ---------------------------

def main():
    # Build metadata index once
    file_info = build_file_info()

    # Select subset of files for west + middle + Haor, within time window
    nc_files = select_nc_files(file_info)

    # Group by tile (same UTM tile, different dates)
    groups = group_files_by_tile(nc_files)

    # Colour scaling
    if RES == "250":
        cell_size = 250.0
    else:
        cell_size = 100.0

    cell_area = cell_size ** 2

    area_vmin = 0.0
    area_vmax = cell_area          # ~62,500 for 250 m
    frac_vmin = 0.0
    frac_vmax = 1.0

    # Plot mean water_area
    plot_swot_raster_mean(
        groups=groups,
        var_name="water_area",
        title="SWOT mean water_area",
        cmap_name="YlGnBu",
        vmin=area_vmin,
        vmax=area_vmax,
        out_dir=RESULTS_DIR
    )

    # Plot mean water_frac
    plot_swot_raster_mean(
        groups=groups,
        var_name="water_frac",
        title="SWOT mean water_frac",
        cmap_name="Blues",
        vmin=frac_vmin,
        vmax=frac_vmax,
        out_dir=RESULTS_DIR
    )


if __name__ == "__main__":
    main()