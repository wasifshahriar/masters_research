#!/usr/bin/env python3
"""
==============================================================================
ROUND 3, STEP 3: WHERE DOES WATER CROSS FROM ONE HAOR TO THE NEXT?
(r3_3_crossings.py)
==============================================================================
Runs on the server, no Google Earth Engine. About 10 to 30 minutes.

Reads   /work/a06/wasif/haor_water20m/water20m_YYYY.tif   (step 1, the fixed 20 m water maps)
Writes  .../validation/round3/
          main/M1_crossing_map.png        where water crosses each shared boundary, and how often
          main/M_pairs.csv                one row per neighbouring pair (the main table)
          data/transects.gpkg             the measuring lines (open in QGIS)
          data/crossings_by_date.csv      every line on every date: 1 open, 0 blocked
          data/pair_states.csv            every pair on every date: separate / exchange / merged
          data/evidence_candidates.csv    dates for step 4 to check with Sentinel-2 and SWOT
          side/S_pair_calendar_YYYY.png   the state of every pair through each year

How it works, in one paragraph:
  Along every boundary that two haors share (outlines within 300 m, the river between them), a line is drawn
  every 50 m. Each line starts 150 m inside haor A, crosses the river and its banks, and ends
  150 m inside haor B. On each radar date the line is OPEN when water runs along its whole
  length (a dry gap of one pixel, 20 m, is allowed as speckle), and BLOCKED when dry ground
  (a bank, a road, a field) cuts it. Then for each pair and date:
     separate = no open line
     exchange = at least one opening (2 or more neighbouring open lines, 100 m wide), but less than half open
     merged   = half or more of the lines open (the boundary is mostly under water)

Usage:  cd /work/a06/wasif/haor_validation_package/round3
        python r3_3_crossings.py
==============================================================================
"""
import os, sys, glob
import numpy as np, pandas as pd
import rasterio
import r3_common as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap

C.ensure_dirs()
haors = C.load_haors()
tr = C.build_transects(haors)
tr.to_file(os.path.join(C.OUT, "data", "transects.gpkg"), driver="GPKG")
print(f"transects: {len(tr)} lines on {tr.pair.nunique()} shared boundaries")
print(tr.groupby("pair").agg(lines=("tid", "size"), median_gap_m=("gap_m", "median")).to_string())

files = C.water_files()
if not files:
    sys.exit(f"No water20m_*.tif in {C.WATER20}. Run step 1 and download the Drive folder first.")
meta = C.read_meta()

# ---------------------------------------------------------------- 1. every line, every date
rows, wetfreq, wetn, ref = [], None, None, None
for f in files:
    with rasterio.open(f) as src:
        dates = C.band_dates(src.descriptions)
        pix = C.transect_pixels(tr, src.transform, (src.height, src.width))
        if ref is None:
            ref = (src.transform, src.height, src.width)
            wetfreq = np.zeros((src.height, src.width), np.float32); wetn = np.zeros_like(wetfreq)
        for b, d in enumerate(dates, start=1):
            a = src.read(b)
            if d.month in C.WET_MONTHS:
                ok = a != 255
                wetfreq += (a == 1); wetn += ok
            for i, (r, c, ok) in enumerate(pix):
                v = np.full(r.shape, 255, np.uint8); v[ok] = a[r[ok], c[ok]]
                rows.append((d, i, C.is_open(v)))
    print(f"  {os.path.basename(f)}: {len(dates)} dates")

X = pd.DataFrame(rows, columns=["date", "row", "open"])
X = X.merge(tr[["pair", "tid"]].reset_index().rename(columns={"index": "row"}), on="row").drop(columns="row")
X.to_csv(os.path.join(C.OUT, "data", "crossings_by_date.csv"), index=False)

# ---------------------------------------------------------------- 2. pair state per date
def pair_day(g):
    g = g.sort_values("tid")
    valid = g["open"].notna()
    n_valid = int(valid.sum()); n_open = int((g["open"] == 1).sum())
    frac = n_open / n_valid if n_valid >= 0.5 * len(g) else np.nan
    k = C.openings(g["open"].values)
    return pd.Series(dict(n_lines=len(g), n_valid=n_valid, n_open=n_open, open_frac=frac,
                          open_km=round(n_open * C.STEP_M / 1000, 2), n_openings=k, state=C.state_of(frac, k)))

P = X.groupby(["date", "pair"])[["tid", "open"]].apply(pair_day).reset_index()
if len(meta):
    P = P.merge(meta[["date", "orbit", "angle"]], on="date", how="left")
P["angle_family"] = np.where(P.get("angle", pd.Series(np.nan, index=P.index)) > 40, "45", np.where(P.get("angle", pd.Series(np.nan, index=P.index)) > 0, "32", "?"))
P.to_csv(os.path.join(C.OUT, "data", "pair_states.csv"), index=False)

# ---------------------------------------------------------------- 3. main table, one row per pair
P["wet"] = P.date.dt.month.isin(C.WET_MONTHS)
W = P[P.wet & (P.state != "no data")]
Xw = X[X.date.dt.month.isin(C.WET_MONTHS) & X["open"].notna()]
freq = Xw.groupby(["pair", "tid"])["open"].mean().rename("open_share").reset_index()
tr2 = tr.merge(freq, on=["pair", "tid"], how="left")
tr2.to_file(os.path.join(C.OUT, "data", "transects.gpkg"), driver="GPKG")

rows = []
for pair, g in W.groupby("pair"):
    t = tr2[tr2.pair == pair]
    r = dict(pair=pair, haors=f"{C.NAME[pair[:3]]} - {C.NAME[pair[4:]]}",
             boundary_km=round(len(t) * C.STEP_M / 1000, 2), lines=len(t), wet_dates=len(g),
             pct_dates_exchange_or_merged=round(100 * (g.state != "separate").mean(), 1),
             pct_dates_merged=round(100 * (g.state == "merged").mean(), 1),
             median_open_pct=round(100 * g.open_frac.median(), 1),
             lines_open_half_the_time=int((t.open_share >= 0.5).sum()))
    for fam in ["45", "32"]:
        gg = g[g.angle_family == fam]
        r[f"pct_exchange_or_merged_{fam}deg"] = round(100 * (gg.state != "separate").mean(), 1) if len(gg) else np.nan
    rows.append(r)
M = pd.DataFrame(rows).sort_values("pct_dates_exchange_or_merged", ascending=False)
M.to_csv(os.path.join(C.OUT, "main", "M_pairs.csv"), index=False)
print("\nMAIN TABLE (wet season, June to November, all years):")
print(M.to_string(index=False))

# ---------------------------------------------------------------- 4. M1: the crossing map
from shapely.geometry import box
hu = haors.to_crs(C.UTM)
T, H, Wd = ref
freqmap = np.where(wetn > 0, wetfreq / np.maximum(wetn, 1), np.nan)
ext = (T.c, T.c + T.a * Wd, T.f + T.e * H, T.f)
fig, ax = plt.subplots(figsize=(13, 10))
ax.imshow(freqmap, extent=ext, cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
tl = tr2.to_crs("EPSG:4326")
cm = plt.cm.RdYlGn
haors.boundary.plot(ax=ax, color=[C.COLOR[i] for i in haors.index], lw=1.4)
from pyproj import Transformer
_tf = Transformer.from_crs(C.UTM, "EPSG:4326", always_xy=True)
tl["lon"], tl["lat"] = _tf.transform(tr2.mid_x.values, tr2.mid_y.values)
ax.scatter(tl.lon, tl.lat, c=tl.open_share.fillna(-1), cmap=cm, vmin=0, vmax=1, s=14, zorder=5, edgecolors="none")
for h, g in haors.iterrows():
    p = g.geometry.representative_point()
    ax.text(p.x, p.y, C.NAME[h], ha="center", va="center", fontsize=15, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.25", fc=C.COLOR[h], ec="none", alpha=0.9))
for _, r in M.iterrows():
    t = tr2[tr2.pair == r.pair]
    from pyproj import Transformer
    lon, lat = Transformer.from_crs(C.UTM, "EPSG:4326", always_xy=True).transform(t.mid_x.median(), t.mid_y.median())
    ax.annotate(f"{r.pair}: {r.pct_dates_exchange_or_merged:.0f}%", (lon, lat), xytext=(16, 16), textcoords="offset points",
                fontsize=12, fontweight="bold", color="black", arrowprops=dict(arrowstyle="-", color="#333333", lw=0.8),
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#555555", alpha=0.9))
x0, y0, x1, y1 = haors.total_bounds
ax.set_xlim(x0 - 0.01, x1 + 0.01); ax.set_ylim(y0 - 0.01, y1 + 0.01)
ax.set_aspect(1 / np.cos(np.radians((y0 + y1) / 2))); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(0, 100)); sm.set_array([])
cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
cb.set_label("% of wet-season radar dates this line is open (water crosses)", fontsize=13)
ax.set_title("Where water crosses between haors, June to November 2019 to 2025\n"
             "dots = measuring lines across each shared boundary (one every 50 m); labels = % of dates the pair "
             "exchanges water\nbackground = how often each 20 m pixel is water (white = always)", fontsize=13)
fig.savefig(os.path.join(C.OUT, "main", "M1_crossing_map.png"), dpi=170, bbox_inches="tight")
plt.close(fig)
print("saved main/M1_crossing_map.png")

# M1b: one zoom panel per shared boundary, same colours
pairs_sorted = list(M.pair)
nc = 4; nr = int(np.ceil(len(pairs_sorted) / nc))
fig, axs = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.6 * nr))
for ax, pair in zip(axs.flat, pairs_sorted):
    t = tl[tl.pair == pair]
    bx0, by0, bx1, by1 = t.total_bounds; pad = 0.004
    ax.imshow(freqmap, extent=ext, cmap="Greys_r", vmin=0, vmax=1, interpolation="nearest")
    haors.boundary.plot(ax=ax, color=[C.COLOR[i] for i in haors.index], lw=1.2)
    ax.scatter(t.lon, t.lat, c=t.open_share.fillna(-1), cmap=cm, vmin=0, vmax=1, s=22, zorder=5, edgecolors="none")
    cx = (bx0 + bx1) / 2; cy = (by0 + by1) / 2; half = max(bx1 - bx0, (by1 - by0) / np.cos(np.radians(cy))) / 2 + pad
    ax.set_xlim(cx - half, cx + half); ax.set_ylim(cy - half * np.cos(np.radians(cy)), cy + half * np.cos(np.radians(cy)))
    ax.set_aspect(1 / np.cos(np.radians(cy))); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    r = M[M.pair == pair].iloc[0]
    ax.set_title(f"{r.haors}\nexchange or merged on {r.pct_dates_exchange_or_merged:.0f}% of wet dates", fontsize=11)
    km = 1.0; deg = km / (111.32 * np.cos(np.radians(cy)))
    x1b = cx + half * 0.9; y1b = cy - half * np.cos(np.radians(cy)) * 0.85
    ax.plot([x1b - deg, x1b], [y1b, y1b], color="red", lw=3); ax.text(x1b - deg / 2, y1b, "1 km", color="red", ha="center", va="bottom", fontsize=9, fontweight="bold")
for ax in list(axs.flat)[len(pairs_sorted):]:
    ax.axis("off")
fig.suptitle("Each shared boundary up close: green dots = water usually crosses here, red = usually blocked "
             "(background: white = water most of the wet season)", fontsize=13)
fig.savefig(os.path.join(C.OUT, "main", "M1b_crossings_by_pair.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved main/M1b_crossings_by_pair.png")

# ---------------------------------------------------------------- 5. side: state calendar per year
code = {"separate": 0, "exchange": 1, "merged": 2, "no data": 3}
cmap = ListedColormap(["#D9534F", "#F0AD4E", "#2E8B57", "#E0E0E0"])
pairs = list(M.pair)
for y in C.YEARS:
    g = P[(P.date >= f"{y}-04-01") & (P.date < f"{y + 1}-04-01")]
    if g.date.nunique() < 2:
        continue
    piv = g.pivot_table(index="pair", columns="date", values="state", aggfunc="first").reindex(pairs)
    arr = piv.apply(lambda s: s.map(code)).fillna(3).values
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest",
              extent=[mdates.date2num(piv.columns.min()), mdates.date2num(piv.columns.max()), len(pairs) - 0.5, -0.5])
    ax.set_yticks(range(len(pairs))); ax.set_yticklabels(pairs)
    ax.xaxis_date(); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_title(f"{y}: red separate, orange exchange (openings), green merged (half or more of the boundary open)")
    fig.savefig(os.path.join(C.OUT, "side", f"S_pair_calendar_{y}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

# ---------------------------------------------------------------- 6. candidate dates for step 4
cand = []
for pair, g in W.groupby("pair"):
    for st in ["exchange", "merged", "separate"]:
        gg = g[g.state == st].sort_values("open_frac", ascending=(st != "merged"))
        for _, r in gg.head(12).iterrows():
            cand.append(dict(pair=pair, date=r.date.date(), state=st, open_frac=r.open_frac, n_openings=r.n_openings))
pd.DataFrame(cand).to_csv(os.path.join(C.OUT, "data", "evidence_candidates.csv"), index=False)
print("saved data/evidence_candidates.csv")
print("\n✓ step 3 done. Next: python r3_4_evidence.py")
