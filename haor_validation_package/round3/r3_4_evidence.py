#!/usr/bin/env python3
"""
==============================================================================
ROUND 3, STEP 4: SHOW IT ON THE IMAGES (r3_4_evidence.py)
==============================================================================
Runs on the server. Downloads small image clips from Google Earth Engine (10 to 30 minutes)
and reads the SWOT pixel files. Run it after step 3.

For each shared boundary it picks radar dates where water crosses (and, if one exists, a date
where most of the boundary is under water) that have a cloud-free Sentinel-2 image within two
days, and makes one EVIDENCE CARD with four panels of the same place:
    A  the raw radar image (Sentinel-1 VV, 10 m)          dark = water
    B  the fixed radar water map + the measuring lines     green = water crosses, red = blocked
    C  Sentinel-2 true colour + the same lines, judged on Sentinel-2 water (NDWI > 0, 10 m)
    D  SWOT water surface elevation, nearest pass within 3 days (2023 onwards)
It also draws SWOT water-level maps of the whole cluster for every pass, and one sheet per year
(2024, 2025) with three passes: before the haors join, joined, after they part.

Writes  .../validation/round3/
          main/M2_evidence_<pair>_<date>.png     the evidence cards
          main/M3_swot_wse_<year>.png             SWOT water level maps, three passes
          data/evidence_summary.csv               radar vs Sentinel-2, line by line, per card
          side/swot_pass_maps/swot_<date>.png     every SWOT pass over the cluster
          side/S_events.csv                       the biggest flood date of each year

Usage:  cd /work/a06/wasif/haor_validation_package/round3
        python r3_4_evidence.py            # add --offline to use only files already in round3/cache
==============================================================================
"""
import os, sys, json, glob
import numpy as np, pandas as pd, geopandas as gpd
import rasterio
import r3_common as C
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pyproj import Transformer

OFFLINE = "--offline" in sys.argv
MIN_CLEAR = 0.7
CARDS_PER_STATE = 1                  # one 'exchange' and one 'merged' card per pair at most
C.ensure_dirs()
CACHE = os.path.join(C.OUT, "cache")
haors = C.load_haors()
tr = gpd.read_file(os.path.join(C.OUT, "data", "transects.gpkg"))
X = pd.read_csv(os.path.join(C.OUT, "data", "crossings_by_date.csv"), parse_dates=["date"])
P = pd.read_csv(os.path.join(C.OUT, "data", "pair_states.csv"), parse_dates=["date"])
cand = pd.read_csv(os.path.join(C.OUT, "data", "evidence_candidates.csv"), parse_dates=["date"])
meta = C.read_meta()
tf = Transformer.from_crs(C.UTM, "EPSG:4326", always_xy=True)
tr["lon"], tr["lat"] = tf.transform(tr.mid_x.values, tr.mid_y.values)
trl = tr.to_crs("EPSG:4326")

if not OFFLINE:
    import r3_gee as G
    G.init()


def window(pair, pad=0.004):
    x0, y0, x1, y1 = trl[trl.pair == pair].total_bounds
    return [round(x0 - pad, 5), round(y0 - pad, 5), round(x1 + pad, 5), round(y1 + pad, 5)]


def cached(kind, date, pair):
    return os.path.join(CACHE, f"{kind}_{pd.Timestamp(date):%Y%m%d}_{pair}.tif")


# ---------------------------------------------------------------- SWOT helpers
def swot_index():
    f = os.path.join(C.SWOT_DIR, "index.csv")
    if not os.path.exists(f):
        return pd.DataFrame(columns=["date", "file"])
    ix = pd.read_csv(f)
    ix = ix[ix.res == "100m"].copy(); ix["date"] = pd.to_datetime(ix["date"])
    return ix


SWOT_IX = swot_index()


def swot_pixels(date, box=None):
    rows = []
    for f in SWOT_IX[SWOT_IX.date == date].file.unique():
        p = os.path.join(C.SWOT_DIR, f)
        if not os.path.exists(p):
            continue
        d = np.load(p)
        df = pd.DataFrame({k: d[k] for k in ["lon", "lat", "wse", "wse_qual", "wse_uncert", "water_frac"]})
        df = df[(df.water_frac >= 0.5) & (df.wse_qual <= 1) & (df.wse_uncert <= 1.0) & np.isfinite(df.wse)]
        rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["lon", "lat", "wse"])
    df = pd.concat(rows)
    if box:
        df = df[(df.lon > box[0]) & (df.lon < box[2]) & (df.lat > box[1]) & (df.lat < box[3])]
    return df


def haor_medians(df, min_px=30):
    if df.empty:
        return {}
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    j = gpd.sjoin(pts, haors.reset_index()[["haor_id", "geometry"]], predicate="within")
    g = j.groupby("haor_id").wse.agg(["median", "count"])
    return {h: float(r["median"]) for h, r in g.iterrows() if r["count"] >= min_px}


# ---------------------------------------------------------------- image helpers
def read_clip(path):
    with rasterio.open(path) as s:
        a = s.read().astype(float); t = s.transform
    a[a <= -9998] = np.nan
    return a, t, (t.c, t.c + t.a * a.shape[2], t.f + t.e * a.shape[1], t.f)


def water20(date, box):
    y = date.year if date.month >= 4 else date.year - 1
    f = os.path.join(C.WATER20, f"water20m_{y}.tif")
    from rasterio.windows import from_bounds
    with rasterio.open(f) as s:
        i = C.band_dates(s.descriptions).index(pd.Timestamp(date)) + 1
        w = from_bounds(*box, transform=s.transform).round_offsets().round_lengths()
        a = s.read(i, window=w); t = s.window_transform(w)
    return a, t, (t.c, t.c + t.a * a.shape[1], t.f + t.e * a.shape[0], t.f)


def open_flags_on(water, transform, pair):
    t = tr[tr.pair == pair].sort_values("tid")
    pix = C.transect_pixels(t, transform, water.shape)
    out = []
    for r, c, ok in pix:
        v = np.full(r.shape, 255, np.uint8); v[ok] = water[r[ok], c[ok]]
        out.append(C.is_open(v))
    return t.tid.values, np.array(out)


def stretch(rgb):
    out = np.zeros_like(rgb)
    for k in range(3):
        x = rgb[..., k]; ok = np.isfinite(x) & (x > 0)
        if ok.sum() < 10:
            continue
        lo, hi = np.percentile(x[ok], (2, 98)); out[..., k] = np.clip((x - lo) / (hi - lo), 0, 1)
    return out


def frame(ax, box, title, km=1.0):
    haors.boundary.plot(ax=ax, color=[C.COLOR[i] for i in haors.index], lw=1.0)
    ax.set_xlim(box[0], box[2]); ax.set_ylim(box[1], box[3])
    ax.set_aspect(1 / np.cos(np.radians((box[1] + box[3]) / 2)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(title, fontsize=12)
    deg = km / (111.32 * np.cos(np.radians(box[1])))
    x1 = box[2] - (box[2] - box[0]) * 0.04; y1 = box[1] + (box[3] - box[1]) * 0.04
    ax.plot([x1 - deg, x1], [y1, y1], color="red", lw=3)
    ax.text(x1 - deg / 2, y1, f"{km:g} km", color="red", ha="center", va="bottom", fontsize=9, fontweight="bold")


def dots(ax, pair, flags):
    t = tr[tr.pair == pair].sort_values("tid")
    col = np.where(np.isnan(flags), "#9E9E9E", np.where(flags == 1, "#00C853", "#D50000"))
    ax.scatter(t.lon, t.lat, c=col, s=16, zorder=6, edgecolors="black", linewidths=0.2)


# ---------------------------------------------------------------- choose the dates
def s2_clear(date, pair, box):
    key = os.path.join(CACHE, "s2_clear.csv")
    tab = pd.read_csv(key, parse_dates=["date"]) if os.path.exists(key) else pd.DataFrame(columns=["date", "pair", "clear", "s2_dates"])
    hit = tab[(tab.date == date) & (tab.pair == pair)]
    if len(hit):
        return float(hit.clear.iloc[0]), str(hit.s2_dates.iloc[0])
    if OFFLINE:
        return (1.0, "cache") if os.path.exists(cached("s2", date, pair)) else (0.0, "")
    img, n, ds = G.s2_near(f"{date:%Y-%m-%d}", G.rect(box), before=2, after=2)
    info = G.ee.Dictionary({"n": n, "c": G.clear_fraction(img, G.rect(box)), "d": ds}).getInfo()
    clear = float(info["c"] or 0) if info["n"] else 0.0
    row = pd.DataFrame([dict(date=date, pair=pair, clear=clear, s2_dates=" ".join(info["d"] or []))])
    pd.concat([tab, row]).to_csv(key, index=False)
    return clear, row.s2_dates.iloc[0]


picks = []
for pair in tr.pair.unique():
    box = window(pair)
    for state in ["exchange", "merged"]:
        got = 0
        for _, r in cand[(cand.pair == pair) & (cand.state == state)].iterrows():
            if got >= CARDS_PER_STATE:
                break
            clear, sd = s2_clear(r.date, pair, box)
            if clear >= MIN_CLEAR:
                picks.append(dict(pair=pair, date=r.date, state=state, s2_dates=sd, clear=clear)); got += 1
print(f"evidence cards to make: {len(picks)}")

# ---------------------------------------------------------------- make the cards
summary = []
for pk in picks:
    pair, date = pk["pair"], pk["date"]; box = window(pair)
    try:
        if not OFFLINE:
            img, _ = G.day_image(f"{date:%Y-%m-%d}", G.rect(box))
            G.download(img.select("VV").toFloat(), box, 10, cached("s1", date, pair))
            s2, _, _ = G.s2_near(f"{date:%Y-%m-%d}", G.rect(box), before=2, after=2)
            G.download(s2.toFloat(), box, 10, cached("s2", date, pair))
        vv, tv, ev = read_clip(cached("s1", date, pair))
        s2, ts, es = read_clip(cached("s2", date, pair))
    except Exception as e:
        print(f"  skip {pair} {date:%Y-%m-%d}: {str(e)[:120]}"); continue
    w, tw, ew = water20(date, box)
    tid, f_radar = open_flags_on(w, tw, pair)
    s2w = np.where(np.isfinite(s2[3]), (s2[3] > 0).astype(np.uint8), 255).astype(np.uint8)
    _, f_s2 = open_flags_on(s2w, ts, pair)
    both = ~np.isnan(f_radar) & ~np.isnan(f_s2)
    agree = float((f_radar[both] == f_s2[both]).mean()) if both.any() else np.nan
    st = P[(P.pair == pair) & (P.date == date)].iloc[0]
    m = meta[meta.date == date]
    orb = f"orbit {int(m.orbit.iloc[0])}, {m.angle.iloc[0]:.0f} deg" if len(m) else ""
    # SWOT within 3 days
    near = SWOT_IX[(SWOT_IX.date - date).abs() <= pd.Timedelta(days=3)]
    sw_date = near.iloc[(near.date - date).abs().argsort()].date.iloc[0] if len(near) else None
    sw = swot_pixels(sw_date, box) if sw_date is not None else pd.DataFrame(columns=["lon", "lat", "wse"])
    med = haor_medians(sw, min_px=20)

    fig, axs = plt.subplots(2, 2, figsize=(15, 13))
    a, b = pair.split("-")
    axs[0, 0].imshow(vv[0], extent=ev, cmap="gray", vmin=-27, vmax=-3)
    frame(axs[0, 0], box, f"A. Radar image (Sentinel-1 VV), {date:%d %b %Y} ({orb})\ndark = water, bright = banks, trees, villages")
    axs[0, 1].imshow(np.where(w == 255, 2, w), extent=ew, cmap=ListedColormap(["#F2EFE6", "#2B7BB9", "#C9C9C9"]), vmin=0, vmax=2, interpolation="nearest")
    dots(axs[0, 1], pair, f_radar)
    frame(axs[0, 1], box, f"B. Radar water map: {st.state.upper()}, {st.n_openings} opening(s), "
                          f"{100 * st.open_frac:.0f}% of the boundary open\ngreen dot = water crosses here, red = blocked")
    axs[1, 0].imshow(stretch(np.dstack([s2[0], s2[1], s2[2]])), extent=es)
    dots(axs[1, 0], pair, f_s2)
    frame(axs[1, 0], box, f"C. Sentinel-2 true colour ({pk['s2_dates']})\ndots judged on Sentinel-2 water: "
                          f"{100 * np.nanmean(f_s2):.0f}% open; agrees with radar on {100 * agree:.0f}% of lines")
    ax = axs[1, 1]
    if len(sw):
        mid = np.median(sw.wse)
        sc = ax.scatter(sw.lon, sw.lat, c=sw.wse, s=5, marker="s", cmap="viridis", vmin=mid - 0.5, vmax=mid + 0.5, lw=0)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label("SWOT water level (m)")
        txt = "   ".join(f"{C.NAME[h]} {v:.2f} m" for h, v in med.items() if h in (a, b))
        frame(ax, box, f"D. SWOT water level, {sw_date:%d %b %Y}\nmedian in this window: {txt or 'too few pixels'}")
    else:
        frame(ax, box, "D. SWOT water level\nno SWOT pass within 3 days of this date")
    fig.suptitle(f"{C.NAME[a]} - {C.NAME[b]}, {date:%d %B %Y}", fontsize=16, fontweight="bold")
    out = os.path.join(C.OUT, "main", f"M2_evidence_{pair}_{date:%Y%m%d}.png")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    print("  saved", os.path.basename(out))
    summary.append(dict(pair=pair, date=date.date(), state=st.state, radar_open_pct=round(100 * st.open_frac, 1),
                        s2_open_pct=round(100 * np.nanmean(f_s2), 1), line_agreement_pct=round(100 * agree, 1),
                        s2_dates=pk["s2_dates"], swot_date=(sw_date.date() if sw_date is not None else ""),
                        swot_level_a=med.get(a, np.nan), swot_level_b=med.get(b, np.nan)))
pd.DataFrame(summary).to_csv(os.path.join(C.OUT, "data", "evidence_summary.csv"), index=False)

# ---------------------------------------------------------------- SWOT maps of the whole cluster
CL = [haors.total_bounds[0] - 0.01, haors.total_bounds[1] - 0.01, haors.total_bounds[2] + 0.01, haors.total_bounds[3] + 0.01]
spread = []
for d in sorted(SWOT_IX.date.unique()):
    sw = swot_pixels(d, CL)
    if len(sw) < 200:
        continue
    med = haor_medians(sw)
    spread.append(dict(date=d, n_haors=len(med), spread=(max(med.values()) - min(med.values())) if len(med) >= 3 else np.nan, **{h: med.get(h, np.nan) for h in C.IDS}))
    fig, ax = plt.subplots(figsize=(9, 7.5))
    lo, hi = np.percentile(sw.wse, [5, 95])
    sc = ax.scatter(sw.lon, sw.lat, c=sw.wse, s=3, marker="s", cmap="viridis", vmin=lo, vmax=hi, lw=0)
    plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label("SWOT water level (m)")
    for h, v in med.items():
        p = haors.loc[h, "geometry"].representative_point()
        ax.text(p.x, p.y, f"{C.NAME[h]}\n{v:.2f} m", ha="center", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C.COLOR[h], alpha=0.85))
    frame(ax, CL, f"SWOT water surface elevation, {pd.Timestamp(d):%d %b %Y}", km=5)
    fig.savefig(os.path.join(C.OUT, "side", "swot_pass_maps", f"swot_{pd.Timestamp(d):%Y%m%d}.png"), dpi=110, bbox_inches="tight")
    plt.close(fig)
SP = pd.DataFrame(spread)
SP.to_csv(os.path.join(C.OUT, "data", "swot_haor_levels.csv"), index=False)

for y in [2024, 2025]:
    s = SP[(SP.date >= f"{y}-04-01") & (SP.date < f"{y + 1}-04-01") & (SP.n_haors >= 3)]
    if len(s) < 3:
        continue
    pre = s[s.date < f"{y}-06-15"].sort_values("spread").tail(1)                 # most different before mid-June
    mid = s[(s.date.dt.month.isin([7, 8, 9]))].sort_values(["n_haors", "spread"], ascending=[False, True]).head(1)
    post = s[s.date >= f"{y}-12-01"].sort_values("spread").tail(1)              # most different after December 1
    chosen = [r for r in [pre, mid, post] if len(r)]
    if len(chosen) < 2:
        continue
    fig, axs = plt.subplots(1, len(chosen), figsize=(7 * len(chosen), 6.4))
    for ax, r in zip(np.atleast_1d(axs), chosen):
        d = r.date.iloc[0]; sw = swot_pixels(d, CL); med = haor_medians(sw)
        sc = ax.scatter(sw.lon, sw.lat, c=sw.wse, s=2.5, marker="s", cmap="viridis",
                        vmin=np.percentile(sw.wse, 5), vmax=np.percentile(sw.wse, 95), lw=0)
        plt.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label("water level (m)")
        for h, v in med.items():
            p = haors.loc[h, "geometry"].representative_point()
            ax.text(p.x, p.y, f"{v:.2f} m", ha="center", va="center", fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=C.COLOR[h], alpha=0.85))
        frame(ax, CL, f"{pd.Timestamp(d):%d %b %Y}: highest minus lowest haor {r.spread.iloc[0]:.2f} m", km=5)
    fig.suptitle(f"SWOT water level over the haors, {y}: before they join, joined, after they part", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(C.OUT, "main", f"M3_swot_wse_{y}.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved main/M3_swot_wse_{y}.png")

# ---------------------------------------------------------------- biggest flood date per year
if len(meta) and "water_share_haors" in meta:
    m2 = meta.copy(); m2["year"] = np.where(m2.date.dt.month >= 4, m2.date.dt.year, m2.date.dt.year - 1)
    ev = m2.sort_values("water_share_haors", ascending=False).groupby("year").head(3).sort_values(["year", "date"])
    ev[["year", "date", "orbit", "angle", "water_share_haors"]].to_csv(os.path.join(C.OUT, "side", "S_events.csv"), index=False)
    print("saved side/S_events.csv")
print("\n✓ step 4 done. Copy the whole round3 folder to your Mac.")
