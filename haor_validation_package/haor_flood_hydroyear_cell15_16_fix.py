# ==============================================================================
# REPLACEMENT CELLS 15 AND 16 FOR haor_flood_hydroyear.ipynb
# ==============================================================================
# Paste each block below (each block starts at a cell-marker line) over the cell with the
# same title. They use only names that cells 0 to 14 already define:
# results, haors, rec_M, FABDEM_TIF, JRC_TIF, ANALYSIS_ROOT, rmask, mapping,
# reproject, Resampling, spearmanr, gpd, np, pd, rasterio, os.
#
# Why cell 15 changes
#   Cell 14 says "report the hydrograph recession, the per-pixel offset median is
#   fragile". Cell 15 then ranks the haors with offset_M, the fragile measure.
#   The rim test must use the same ranking as the Kendall's W result it is meant
#   to explain: rec_M. With 5 haors Spearman's p is also coarse, so the exact
#   permutation p over all 120 orderings is printed next to it.
#
# Why cell 16 changes
#   The old corridor was buffer(300) of A intersected with buffer(300) of B. That
#   area reaches 300 m into BOTH haor floors, so its 5th percentile finds the
#   floor, the same failure that made rim v1 give rho = 0. A sill is the LOWEST
#   point of the CREST between two basins. The new cell walks along the shared
#   boundary every 30 m, takes the HIGHEST clean elevation across a 200 m
#   perpendicular profile at each station (the crest), and reports the 10th
#   percentile of those crest heights (the low gap water would top first).
#   Cell 16b then asks the question the sill exists for: do pairs with lower
#   sills merge earlier and stay merged longer?
# ==============================================================================

# %%
# ================= MECHANISM v3: OUTWARD rim elevation vs drain order =================
# v2 fixes kept (outward-only ring, JRC permanent water masked, zero pixels
# dropped). v3: ranks by the hydrograph recession (rec_M, cell 14), the measure
# the talk reports, and gives an exact permutation p for n = 5.
from itertools import permutations
UTM = "EPSG:32646"
with rasterio.open(FABDEM_TIF) as fsrc, rasterio.open(JRC_TIF) as jsrc:
    h_utm = haors.to_crs(UTM)
    rim_rows = []
    for _, h in h_utm.iterrows():
        ring = h.geometry.buffer(150).difference(h.geometry)     # OUTWARD ring only
        ring_r = gpd.GeoSeries([ring], crs=UTM).to_crs(fsrc.crs).iloc[0]
        try:
            dat, tr = rmask(fsrc, [mapping(ring_r)], crop=True, filled=False)
        except ValueError:
            print(f"  {h.haor_id}: ring outside FABDEM"); continue
        elev = np.ma.filled(dat[0], np.nan).astype("float32")
        jrc_g = np.full(elev.shape, np.nan, "float32")
        reproject(rasterio.band(jsrc, 1), jrc_g, src_transform=jsrc.transform,
                  src_crs=jsrc.crs, dst_transform=tr, dst_crs=fsrc.crs,
                  resampling=Resampling.bilinear)
        keep = np.isfinite(elev) & (np.abs(elev) > 0.05) & ~(np.isfinite(jrc_g) & (jrc_g > 75))
        vals = elev[keep]
        if vals.size < 100:
            print(f"  {h.haor_id}: too few clean rim pixels ({vals.size})"); continue
        rim_rows.append(dict(haor_id=h.haor_id, n_px=int(vals.size),
            rim_p2=float(np.percentile(vals, 2)), rim_p5=float(np.percentile(vals, 5)),
            rim_p10=float(np.percentile(vals, 10)), rim_median=float(np.median(vals))))
rim_df = pd.DataFrame(rim_rows).set_index("haor_id")
rim_df = rim_df.join(rec_M.rank(axis=0).mean(axis=1).rename("mean_drain_rank"))
rim_df.to_csv(os.path.join(ANALYSIS_ROOT, "rim_elevation_vs_drain_v3.csv"))
print(rim_df.sort_values("mean_drain_rank").to_string())

def exact_spearman_p(x, y):
    """Two-sided exact p: share of all orderings of y with |rho| >= observed."""
    x = pd.Series(x).rank().values; y = pd.Series(y).rank().values
    r0 = abs(np.corrcoef(x, y)[0, 1])
    rs = [abs(np.corrcoef(x, np.array(p))[0, 1]) for p in permutations(y)]
    return float(np.mean(np.array(rs) >= r0 - 1e-12))

print("\nSpearman, drain rank (hydrograph recession) against OUTWARD rim height:")
for col in ["rim_p2", "rim_p5", "rim_p10", "rim_median"]:
    rho, p_as = spearmanr(rim_df["mean_drain_rank"], rim_df[col])
    print(f"   {col:10}: rho = {rho:+.2f}   asymptotic p = {p_as:.3f}   "
          f"exact p = {exact_spearman_p(rim_df['mean_drain_rank'], rim_df[col]):.3f}")
print("""
With 5 haors the smallest possible two-sided exact p is 2/120 = 0.017, reached
only by a perfect ordering. Report the exact p. Four percentiles are shown so
the choice of p5 is visible; quote the one fixed in advance (p5) and say so.
FABDEM vertical error (about 1 m or more) is large next to the relief here; if
the values cluster, the DEM cannot resolve the mechanism.""")

# %%
# ================= MECHANISM 2 v2: inter-haor SILL = low point of the CREST =================
from shapely.geometry import LineString, MultiLineString, Point
STEP_M, HALF_W, MIN_STATIONS = 30.0, 100.0, 10

def boundary_lines(a_geom, b_geom):
    """Shared boundary of two haors (boundary of A within 50 m of B), as lines."""
    g = a_geom.boundary.intersection(b_geom.buffer(50))
    if g.is_empty:
        return []
    if isinstance(g, LineString):
        return [g]
    return [x for x in getattr(g, "geoms", []) if isinstance(x, LineString) and x.length > 0]

def crest_profile(line, fsrc, jsrc):
    """Crest height at a station every STEP_M along the line: highest clean
    elevation within +-HALF_W across the line."""
    crests = []
    n = max(int(line.length // STEP_M), 1)
    offs = np.arange(-HALF_W, HALF_W + 1e-6, STEP_M)
    for i in range(n + 1):
        d = min(i * STEP_M, line.length)
        p0 = line.interpolate(d); p1 = line.interpolate(min(d + 1.0, line.length))
        if p1.equals(p0):
            p1 = line.interpolate(max(d - 1.0, 0)); p0, p1 = p1, p0
        tx, ty = p1.x - p0.x, p1.y - p0.y
        L = np.hypot(tx, ty) or 1.0
        nx, ny = -ty / L, tx / L                          # unit normal
        pts = gpd.GeoSeries([Point(p0.x + o * nx, p0.y + o * ny) for o in offs], crs=UTM).to_crs(fsrc.crs)
        xy = list(zip(pts.x, pts.y))
        z = np.array([v[0] for v in fsrc.sample(xy)], "float64")
        jx = gpd.GeoSeries(pts.values, crs=fsrc.crs).to_crs(jsrc.crs)
        occ = np.array([v[0] for v in jsrc.sample(list(zip(jx.x, jx.y)))], "float64")
        nod = fsrc.nodata
        ok = np.isfinite(z) & (np.abs(z) > 0.05) & ~(occ > 75)
        if nod is not None:
            ok &= z != nod
        if ok.sum() >= 3:
            crests.append(float(z[ok].max()))
    return np.array(crests)

pairs = []
with rasterio.open(FABDEM_TIF) as fsrc, rasterio.open(JRC_TIF) as jsrc:
    for i in range(len(h_utm)):
        for j in range(i + 1, len(h_utm)):
            a, b = h_utm.iloc[i], h_utm.iloc[j]
            lines = boundary_lines(a.geometry, b.geometry)
            if not lines:
                continue                                   # not adjacent: no sill
            cr = np.concatenate([crest_profile(l, fsrc, jsrc) for l in lines])
            if cr.size < MIN_STATIONS:
                print(f"  {a.haor_id}-{b.haor_id}: only {cr.size} crest stations"); continue
            pairs.append(dict(pair=f"{a.haor_id}-{b.haor_id}",
                              boundary_m=round(sum(l.length for l in lines)),
                              n_stations=int(cr.size),
                              sill_p10=float(np.percentile(cr, 10)),
                              sill_p25=float(np.percentile(cr, 25)),
                              crest_median=float(np.median(cr))))
sill_df = pd.DataFrame(pairs).sort_values("sill_p10")
sill_df.to_csv(os.path.join(ANALYSIS_ROOT, "interhaor_sills_v2.csv"), index=False)
print("Inter-haor sills from the CREST along each shared boundary (lower = should merge first):")
print(sill_df.to_string(index=False))

# %%
# ================= 16b: do lower sills merge earlier and stay merged longer? =================
# Per adjacent pair and year, from the connection matrix: first date the pair is
# in one body, last date, and the share of scored dates merged.
mrows = []
for y in results:
    M = results[y]["matrix_df"]
    M = M[(M.drop(columns="doy") >= 0).all(axis=1)]
    for pr in sill_df.pair:
        a, b = pr.split("-")
        mg = (M[a] > 0) & (M[a] == M[b])
        mrows.append(dict(year=y, pair=pr,
                          first_merge=float(M.doy[mg].min()) if mg.any() else np.nan,
                          last_merge=float(M.doy[mg].max()) if mg.any() else np.nan,
                          merged_share=float(mg.mean())))
MT = pd.DataFrame(mrows)
MS = MT.groupby("pair")[["first_merge", "last_merge", "merged_share"]].mean().join(sill_df.set_index("pair"))
MS.to_csv(os.path.join(ANALYSIS_ROOT, "sill_vs_merge_timing.csv"))
print(MS.round(3).to_string())
print(f"\n{len(MS)} adjacent pairs. Expected signs if sills control merging:")
for col, sign in [("first_merge", "+"), ("last_merge", "-"), ("merged_share", "-")]:
    ok = MS[[col, "sill_p10"]].dropna()
    if len(ok) >= 4:
        rho, _ = spearmanr(ok["sill_p10"], ok[col])
        print(f"   sill_p10 vs {col:13}: rho = {rho:+.2f} (expected {sign})   exact p = "
              f"{exact_spearman_p(ok['sill_p10'], ok[col]):.3f}   n = {len(ok)}")
print("""
Seven pairs at most, so this is a direction check, not a proof. If the signs
are wrong or rho is near 0, FABDEM does not see the controlling sills (likely
for embankments narrower than 30 m) and the SWOT transition levels (S8) are
the better sill estimate.""")
