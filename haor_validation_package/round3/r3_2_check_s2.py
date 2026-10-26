#!/usr/bin/env python3
"""
==============================================================================
ROUND 3, STEP 2: DOES THE FIXED RADAR WATER MATCH SENTINEL-2? (r3_2_check_s2.py)
==============================================================================
Runs on the server. Talks to Google Earth Engine (about 30 to 60 minutes) and reads the
downloaded step-1 files. Safe to stop and restart: dates already done are skipped.

For EVERY radar date 2019 to 2025 that has a Sentinel-2 image within 2 days before to 2 days
after with at least 60% of the haor area cloud free, it compares, inside the five haors:
    Sentinel-2 water (NDWI > 0, 10 m)  vs  new radar water (step 1)  vs  old published detector

Writes  .../validation/round3/
          main/M0_fix_20_22_Oct_2024.png   old and new radar maps on two dates two days apart
                                           (45 and 32 degree passes) next to Sentinel-2
          side/S_check_vs_sentinel2.csv    one row per checked date
          side/S_check_summary.csv         the same, summarised by viewing angle
          side/S_zigzag_2024.png           water share inside the haors on every 2024 pass,
                                           old vs new, coloured by viewing angle

Usage:  cd /work/a06/wasif/haor_validation_package/round3
        python r3_2_check_s2.py
==============================================================================
"""
import os, sys, glob
import numpy as np, pandas as pd
import r3_common as C
import r3_gee as G
ee = G.ee
C.ensure_dirs()
G.init()
R20, R100 = G.rect(C.BOX20), G.rect(C.BOX100)
HAORS = G.haor_geometry()
OUTCSV = os.path.join(C.OUT, "side", "S_check_vs_sentinel2.csv")
MIN_CLEAR = 0.6


def check(d):
    d = ee.String(d)
    s2, n, s2dates = G.s2_near(d, HAORS)
    empty = ee.Feature(None, {"date": d, "s2_images": 0})

    def full():
        water, info = G.classify(d, R100, R20)
        old = G.old_flag(d, R100).unmask(0)
        clear = s2.select("ndwi").mask().unmask(0).gt(0)
        s = s2.select("ndwi").gt(0).unmask(0)
        nw = water.unmask(0)
        img = ee.Image.cat([s.rename("s2"), nw.rename("new"), old.rename("old"),
                            nw.And(s).rename("new_s2"), old.And(s).rename("old_s2"),
                            nw.eq(s).rename("new_agree"), old.eq(s).rename("old_agree")]).updateMask(clear)
        m = img.reduceRegion(ee.Reducer.mean(), HAORS, 20, maxPixels=1e9, bestEffort=True)
        cf = clear.reduceRegion(ee.Reducer.mean(), HAORS, 40, maxPixels=1e9, bestEffort=True).get("ndwi")
        return ee.Feature(None, m).set(info).set("s2_images", n, "s2_dates", s2dates.join(" "), "clear_frac", cf)

    return ee.Feature(ee.Algorithms.If(n.gt(0), full(), empty))


# Every row is written with the SAME columns in the SAME order. Earth Engine returns each feature's
# properties in its own order and leaves out empty ones, so writing them as they come shifted values
# into the wrong columns (the cause of "dtype 'str' does not support operation 'median'").
COLS = ["date", "orbit", "direction", "angle", "t_vv", "t_vh", "s2_images", "s2_dates", "clear_frac",
        "s2", "new", "old", "new_s2", "old_s2", "new_agree", "old_agree"]
NUM = [c for c in COLS if c not in ("date", "direction", "s2_dates")]

# a table written by the old version has shifted columns: set it aside (it is not deleted) and start clean
if os.path.exists(OUTCSV):
    with open(OUTCSV) as fh:
        header = fh.readline().strip().split(",")
    if header != COLS:
        bad = OUTCSV.replace(".csv", "_broken_old_version.csv")
        os.replace(OUTCSV, bad)
        print(f"  old table had shifted columns; moved to {os.path.basename(bad)} and starting again")

done = set(pd.read_csv(OUTCSV, usecols=["date"]).date) if os.path.exists(OUTCSV) else set()
for y in C.YEARS:
    start, end = G.year_window(y)
    dates = [d for d in G.s1_dates(start, end, R100).getInfo() if d not in done]
    for i in range(0, len(dates), 12):
        chunk = dates[i:i + 12]
        fc = None
        for attempt in range(3):
            try:
                fc = ee.FeatureCollection(ee.List(chunk).map(check)).getInfo()
                break
            except Exception as e:
                print(f"    retry {attempt + 1}: {str(e)[:90]}")
        if fc is None:
            print(f"  !! {chunk[0]} to {chunk[-1]} failed, run again later"); continue
        rows = pd.DataFrame([f["properties"] for f in fc["features"]]).reindex(columns=COLS)
        rows.to_csv(OUTCSV, mode="a", header=not os.path.exists(OUTCSV), index=False)
        print(f"  {y}: {chunk[0]} to {chunk[-1]} done")

# ---------------------------------------------------------------- summary
R = pd.read_csv(OUTCSV)
for c in NUM:
    R[c] = pd.to_numeric(R[c], errors="coerce")          # anything that is not a number becomes NaN
R = R[(R["clear_frac"] >= MIN_CLEAR) & R["s2"].notna() & R["angle"].notna()].copy()
R["family"] = np.where(R.angle > 40, "45 deg (orbits 114, 77)", "32 deg (orbits 41, 150)")
R["f1_new"] = 2 * R.new_s2 / (R.new + R.s2)
R["f1_old"] = 2 * R.old_s2 / (R.old + R.s2)
S = R.groupby("family").agg(dates=("date", "size"), s2_water=("s2", "median"), new_water=("new", "median"),
                            old_water=("old", "median"), agree_new=("new_agree", "median"),
                            agree_old=("old_agree", "median"), f1_new=("f1_new", "median"),
                            f1_old=("f1_old", "median")).round(3)
S.to_csv(os.path.join(C.OUT, "side", "S_check_summary.csv"))
print(f"\nSentinel-2 check: {len(R)} dates with at least {int(100 * MIN_CLEAR)}% clear sky (medians):")
print(S.to_string())

# ---------------------------------------------------------------- M0 and the 2024 zigzag
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from rasterio.windows import from_bounds

haors = C.load_haors()
OLD = f"{C.WORK}/haor_flood_analysis_hydroyear/flood_stack_cluster_2024.tif"
NEW = os.path.join(C.WATER20, "water20m_2024.tif")
VIEW = [90.97, 24.95, 91.28, 25.18]


def band(path, date):
    with rasterio.open(path) as s:
        ds = C.band_dates(s.descriptions)
        i = ds.index(pd.Timestamp(date)) + 1
        w = from_bounds(*VIEW, transform=s.transform).round_offsets().round_lengths()
        a = s.read(i, window=w).astype(float); t = s.window_transform(w)
    return a, (t.c, t.c + t.a * a.shape[1], t.f + t.e * a.shape[0], t.f)


cmap = ListedColormap(["#F2EFE6", "#2B7BB9", "#C9C9C9"])
def show(ax, a, ext, title):
    v = np.where(~np.isfinite(a) | (a == 255), 2, a)
    ax.imshow(v, extent=ext, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    haors.boundary.plot(ax=ax, color="#333333", lw=0.8)
    ax.set_xlim(VIEW[0], VIEW[2]); ax.set_ylim(VIEW[1], VIEW[3]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_title(title, fontsize=13)

try:
    s2, n, s2d = G.s2_near("2024-10-21", G.rect(VIEW), before=3, after=3)
    p = G.download(s2, VIEW, 30, os.path.join(C.OUT, "cache", "s2_20241021.tif"))   # 30 m keeps the download under 32 MB
    with rasterio.open(p) as s:
        b = s.read().astype(float); t = s.transform
    b[b <= -9998] = np.nan
    ext2 = (t.c, t.c + t.a * b.shape[2], t.f + t.e * b.shape[1], t.f)
    rgb = np.dstack([b[0], b[1], b[2]])
    for k in range(3):
        x = rgb[..., k]; ok = np.isfinite(x) & (x > 0)
        lo, hi = np.percentile(x[ok], (2, 98)); rgb[..., k] = np.clip((x - lo) / (hi - lo), 0, 1)
    fig, axs = plt.subplots(2, 3, figsize=(17, 10.5))
    for ax, (path, d, t_) in zip(axs[0, :2], [(OLD, "2024-10-20", "OLD map, 20 Oct 2024, 45 deg pass"),
                                              (OLD, "2024-10-22", "OLD map, 22 Oct 2024, 32 deg pass")]):
        a, e = band(path, d); show(ax, a, e, t_)
    for ax, (path, d, t_) in zip(axs[1, :2], [(NEW, "2024-10-20", "NEW map, 20 Oct 2024, 45 deg pass"),
                                              (NEW, "2024-10-22", "NEW map, 22 Oct 2024, 32 deg pass")]):
        a, e = band(path, d); show(ax, a, e, t_)
    axs[0, 2].imshow(np.nan_to_num(rgb), extent=ext2); axs[0, 2].set_title("Sentinel-2 true colour, same week", fontsize=13)
    sw = np.where(np.isfinite(b[3]), (b[3] > 0).astype(float), 2)
    show(axs[1, 2], sw, ext2, "Sentinel-2 water (NDWI > 0)")
    haors.boundary.plot(ax=axs[0, 2], color="white", lw=0.8)
    for ax in axs[0, 2:]:
        ax.set_xlim(VIEW[0], VIEW[2]); ax.set_ylim(VIEW[1], VIEW[3]); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("The fix: two radar passes two days apart now agree with each other and with Sentinel-2\n"
                 "blue = water, beige = dry, grey = no data", fontsize=14)
    fig.savefig(os.path.join(C.OUT, "main", "M0_fix_20_22_Oct_2024.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved main/M0_fix_20_22_Oct_2024.png")
except Exception as e:
    print("M0 figure skipped:", str(e)[:200])

# zigzag: share of the haors that is water on every 2024 pass, old vs new
try:
    from rasterio.features import geometry_mask
    meta = C.read_meta()
    def shares(path):
        out = {}
        with rasterio.open(path) as s:
            m = ~geometry_mask(haors.geometry.values, out_shape=(s.height, s.width), transform=s.transform)
            for i, d in enumerate(C.band_dates(s.descriptions), start=1):
                a = s.read(i).astype(float); ok = m & np.isfinite(a) & (a != 255)
                out[d] = (a[ok] == 1).mean() if ok.sum() > 0.5 * m.sum() else np.nan
        return pd.Series(out)
    so, sn = shares(OLD), shares(NEW)
    Z = pd.DataFrame({"old": so, "new": sn}).dropna(how="all")
    Z = Z.join(meta.set_index("date")[["angle"]], how="left")
    fig, axs = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for ax, col, t_ in zip(axs, ["old", "new"], ["OLD published maps", "NEW fixed maps"]):
        ax.plot(Z.index, Z[col], "-", color="#C5CDD8", lw=1)
        for fam, c in [(Z.angle > 40, "#1F4E79"), (Z.angle <= 40, "#E07856")]:
            ax.plot(Z.index[fam], Z[col][fam], "o", color=c, ms=6)
        ax.set_ylim(0, 1); ax.grid(alpha=0.3); ax.set_ylabel("share of the haors\nthat is water"); ax.set_title(t_, loc="left")
    axs[0].plot([], [], "o", color="#1F4E79", label="45 deg passes"); axs[0].plot([], [], "o", color="#E07856", label="32 deg passes")
    axs[0].legend(frameon=False, loc="upper right")
    fig.savefig(os.path.join(C.OUT, "side", "S_zigzag_2024.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("saved side/S_zigzag_2024.png")
except Exception as e:
    print("zigzag figure skipped:", str(e)[:200])
print("\n✓ step 2 done. Next: python r3_3_crossings.py")