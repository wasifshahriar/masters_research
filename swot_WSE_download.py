#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Incremental download of SWOT L2 HR Raster granules (100m + 250m) over a bbox,
downloading only granules newer than the latest local file timestamp.

Key fixes:
- Use the HR Raster dataset short_name that is actually used operationally: SWOT_L2_HR_Raster_D
  (Resolution is in the granule name: ..._100m_... or ..._250m_...)
- Robust provider handling: try POCLOUD and PODAAC and choose whichever returns data
- Robust incremental filtering by parsing timestamps from GranuleUR / filename
"""

import os
import re
from datetime import datetime, timezone

import earthaccess

# ---------------------------
# USER CONFIG
# ---------------------------

DATA_DIR = "/work/a06/wasif/swot_raster_data"
os.makedirs(DATA_DIR, exist_ok=True)

BBOX = (89.8, 23.5, 92.8, 25.5)  # (lon_min, lat_min, lon_max, lat_max)

# Put your Earthdata Login here OR set them in the shell before running.
EARTHDATA_USERNAME = "wasifshahriar7"
EARTHDATA_PASSWORD = "chintaTATA!!1"

# Providers to try
PROVIDERS = ["POCLOUD", "PODAAC"]

# Use Version D HR Raster dataset short_name
SHORT_NAME = "SWOT_L2_HR_Raster_D"

# Search end date: today (UTC)
END_DATE_UTC = datetime.now(timezone.utc).date().isoformat()

# Parse timestamps from filenames like:
# ..._20250430T152151_20250430T152208_...
TS_RE = re.compile(r"_(20\d{6}T\d{6})_(20\d{6}T\d{6})_")

# ---------------------------
# TIME / NAME HELPERS
# ---------------------------

def parse_start_time_from_name(name: str):
    """Return start datetime (UTC) from granule name token _YYYYMMDDTHHMMSS_"""
    m = TS_RE.search(name)
    if not m:
        return None
    start_str = m.group(1)
    try:
        return datetime.strptime(start_str, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None

def latest_local_time_for_res(res_str: str):
    """
    res_str: "100m" or "250m"
    Looks for files containing f"_Raster_{res_str}_" and returns latest start time.
    """
    latest = None
    for fn in os.listdir(DATA_DIR):
        if f"_Raster_{res_str}_" not in fn:
            continue
        dt = parse_start_time_from_name(fn)
        if dt is None:
            continue
        if latest is None or dt > latest:
            latest = dt
    return latest

def granule_name_from_obj(g):
    """
    Get a useful granule identifier (usually GranuleUR) from earthaccess granule object.
    Handles both dict-like and object-like granules.
    """
    # dict-like
    try:
        umm = g.get("umm", {})
        if isinstance(umm, dict):
            return umm.get("GranuleUR")
    except Exception:
        pass

    # object-like
    try:
        umm = getattr(g, "umm", None)
        if isinstance(umm, dict):
            return umm.get("GranuleUR")
    except Exception:
        pass

    # last resort: string
    return None

# ---------------------------
# PROVIDER PICKING
# ---------------------------

def pick_provider_for_query(res_str, temporal_start, temporal_end):
    """
    Try providers and pick the one that returns the most granules.
    """
    best_provider = None
    best_count = -1

    # granule_name filter for resolution (your filenames contain _100m_ / _250m_)
    gname = f"*_{res_str}_*"

    for p in PROVIDERS + [None]:
        try:
            results = earthaccess.search_data(
                short_name=SHORT_NAME,
                bounding_box=BBOX,
                temporal=(temporal_start, temporal_end),
                granule_name=gname,
                provider=p,
                count=-1
            )
            n = len(list(results))
        except Exception as e:
            print(f"[DIAG] Provider test failed for provider={p}: {e}")
            n = 0

        print(f"[DIAG] Provider={p} -> granules={n} for res={res_str}")
        if n > best_count:
            best_count = n
            best_provider = p

    return best_provider, best_count

# ---------------------------
# DOWNLOAD LOGIC
# ---------------------------

def download_new_granules(res_str: str):
    latest_local = latest_local_time_for_res(res_str)
    if latest_local is None:
        # If nothing local, start from science orbit era; adjust if you want earlier
        latest_local = datetime(2022, 12, 1, tzinfo=timezone.utc)

    print(f"\n===== PROCESSING {res_str} =====")
    print(f"[INFO] Latest local start timestamp: {latest_local.isoformat()}")

    temporal_start = latest_local.date().isoformat()
    temporal_end = END_DATE_UTC
    print(f"[INFO] Search window (UTC dates): {temporal_start} -> {temporal_end}")
    print(f"[INFO] Granule name filter: '*_{res_str}_*'")
    print(f"[INFO] Dataset short_name: {SHORT_NAME}")

    # Pick provider that actually returns data
    provider, nprov = pick_provider_for_query(res_str, temporal_start, temporal_end)
    print(f"[INFO] Chosen provider={provider} (granules in window={nprov})")
    if nprov <= 0:
        print("[WARN] No granules found for ANY provider with this bbox/time/resolution.")
        print("       Possible causes: bbox too small/strict, no overpasses in window, or wrong short_name.")
        return

    # Fetch granules from chosen provider
    results = earthaccess.search_data(
        short_name=SHORT_NAME,
        bounding_box=BBOX,
        temporal=(temporal_start, temporal_end),
        granule_name=f"*_{res_str}_*",
        provider=provider,
        count=-1
    )
    granules = list(results)
    print(f"[DIAG] earthaccess returned granules: {len(granules)}")

    existing = set(os.listdir(DATA_DIR))

    to_dl = []
    n_no_name = 0
    n_no_time = 0

    for g in granules:
        gname = granule_name_from_obj(g)
        if not gname:
            n_no_name += 1
            continue

        # Skip if already present
        if gname in existing or (gname + ".nc") in existing:
            continue

        gtime = parse_start_time_from_name(gname)
        if gtime is None:
            n_no_time += 1
            # If no time parsed, keep it (better to download than miss)
            to_dl.append(g)
            continue

        if gtime > latest_local:
            to_dl.append(g)

    print(f"[DIAG] Granules missing GranuleUR: {n_no_name}")
    print(f"[DIAG] Granules with unparseable time token: {n_no_time}")
    print(f"[INFO] New granules to download: {len(to_dl)}")

    if not to_dl:
        print("[INFO] Nothing to download.")
        return

    earthaccess.download(to_dl, local_path=DATA_DIR)
    print(f"[INFO] Download complete for {res_str}")

def main():
    print("[INFO] Starting incremental SWOT HR Raster download")
    print(f"[INFO] Target directory: {DATA_DIR}")
    print(f"[INFO] BBOX: {BBOX}")
    print(f"[INFO] End date (UTC): {END_DATE_UTC}")

    # ---- Credentials: set before login (environment strategy reads these) ----
    if "EARTHDATA_USERNAME" not in os.environ:
        os.environ["EARTHDATA_USERNAME"] = EARTHDATA_USERNAME
    if "EARTHDATA_PASSWORD" not in os.environ:
        os.environ["EARTHDATA_PASSWORD"] = EARTHDATA_PASSWORD

    print("[DIAG] EARTHDATA_USERNAME set?", "EARTHDATA_USERNAME" in os.environ)
    print("[DIAG] EARTHDATA_PASSWORD set?", "EARTHDATA_PASSWORD" in os.environ)

    # Login (reads env vars)
    earthaccess.login(strategy="environment", persist=True)

    for res in ["100m", "250m"]:
        try:
            download_new_granules(res)
        except Exception as e:
            print(f"[ERROR] Failed for {res}: {e}")

    print("\n[DONE] Incremental download finished.")

if __name__ == "__main__":
    main()