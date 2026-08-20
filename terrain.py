"""Ground elevation for Thrusty — launch altitude and impact/glide floor.

Two layers, both from the AWS Terrarium open terrain tiles (see dem_build.py
for provenance):

  * COARSE (always available, offline): a bundled 0.05° global int16 grid
    (data/dem/terrain_0p05deg.npy), memory-mapped and bilinearly sampled.
  * HIGH-RES (opt-in, on demand): individual Terrarium tiles at a higher zoom
    fetched over the network and cached on disk, for precise pad elevation and
    mountainous terminal glide.  Falls back to the coarse grid when the network
    is unavailable or hi-res is not requested.

Public API:
    elevation(lat, lon, hi_res=None)         -> raw terrain height (m), signed
    ground_elevation(lat, lon, hi_res=None)  -> max(elevation, 0): land height
                                                 on land, sea surface (0) over
                                                 water — the trajectory floor.
    configure_terrain(source)                -> select the default source for
                                                 hi_res=None calls: 'terrarium'
                                                 (network tiles, coarse fallback)
                                                 or 'coarse' (offline grid) —
                                                 surfaced in the GUI under
                                                 Analysis ▸ Reference Data.
    have_coarse()                            -> bool, coarse grid present

All heights are metres above the WGS84 ellipsoid's geoid proxy (the tiles'
native vertical datum); good to the tens-of-metres a screening tool needs.
"""

import io
import math
import os
import threading
import urllib.request

import numpy as np

_DEM_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dem")
_COARSE_NPY = os.path.join(_DEM_DIR, "terrain_0p05deg.npy")
_TILE_URL   = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_HI_ZOOM    = 11                          # ~30–50 m/px near the equator
_CACHE_DIR  = os.path.join(
    os.path.expanduser("~"), ".gui_missile_flyout", "dem_tiles")

_lock = threading.Lock()
_coarse = None            # lazily memory-mapped (rows = lat +90→−90, cols = lon −180→+180)
_coarse_tried = False
_tile_cache = {}          # (z,x,y) -> decoded elevation ndarray (in-process)

# Selected source for default lookups (Analysis ▸ Reference Data ▸ Terrain):
#   'terrarium' — hi-res network tiles, coarse fallback on any failure
#   'coarse'    — bundled offline grid only
# The trajectory integrator always passes hi_res=False (offline, deterministic)
# regardless of this setting; the choice governs GUI-side pad sampling.
_SOURCE = "terrarium"


def configure_terrain(source):
    """Select the default elevation source: 'terrarium' or 'coarse'."""
    global _SOURCE
    if source not in ("terrarium", "coarse"):
        raise ValueError(f"unknown terrain source '{source}'")
    _SOURCE = source


def terrain_source():
    return _SOURCE


# ── coarse grid ─────────────────────────────────────────────────────────────

def _load_coarse():
    global _coarse, _coarse_tried
    if _coarse is not None or _coarse_tried:
        return _coarse
    with _lock:
        if _coarse is None and not _coarse_tried:
            _coarse_tried = True
            try:
                _coarse = np.load(_COARSE_NPY, mmap_mode="r")
            except Exception:
                _coarse = None
    return _coarse


def have_coarse():
    return _load_coarse() is not None


def _coarse_elevation(lat, lon):
    """Bilinear sample of the coarse 0.05° grid; 0.0 if the grid is absent."""
    g = _load_coarse()
    if g is None:
        return 0.0
    nlat, nlon = g.shape
    lon = ((float(lon) + 180.0) % 360.0) - 180.0
    lat = max(-90.0, min(90.0, float(lat)))
    # Fractional grid coordinates (cell centres): row 0 = +90−½cell, etc.
    fr = (90.0 - lat) / 180.0 * nlat - 0.5
    fc = (lon + 180.0) / 360.0 * nlon - 0.5
    r0 = int(math.floor(fr)); c0 = int(math.floor(fc))
    dr = fr - r0; dc = fc - c0
    r0c = min(max(r0, 0), nlat - 1); r1c = min(r0c + 1, nlat - 1)
    c0w = c0 % nlon; c1w = (c0 + 1) % nlon          # wrap longitude
    v00 = float(g[r0c, c0w]); v01 = float(g[r0c, c1w])
    v10 = float(g[r1c, c0w]); v11 = float(g[r1c, c1w])
    top = v00 * (1 - dc) + v01 * dc
    bot = v10 * (1 - dc) + v11 * dc
    return top * (1 - dr) + bot * dr


# ── high-res tiles (on demand) ──────────────────────────────────────────────

def _deg2tile(lat, lon, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    latr = math.radians(max(-85.05, min(85.05, lat)))
    y = (1.0 - math.asinh(math.tan(latr)) / math.pi) / 2.0 * n
    return x, y


def _decode_tile(rgb):
    a = rgb.astype(np.float64)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def _load_tile(z, x, y):
    key = (z, x, y)
    if key in _tile_cache:
        return _tile_cache[key]
    path = os.path.join(_CACHE_DIR, str(z), str(x), f"{y}.npy")
    if os.path.exists(path):
        try:
            arr = np.load(path)
            _tile_cache[key] = arr
            return arr
        except Exception:
            pass
    # Fetch from the network.
    from PIL import Image                       # local import: optional at runtime
    url = _TILE_URL.format(z=z, x=x, y=y)
    with urllib.request.urlopen(url, timeout=30) as r:
        rgb = np.asarray(Image.open(io.BytesIO(r.read())).convert("RGB"),
                         dtype=np.uint8)
    arr = _decode_tile(rgb).astype(np.float32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, arr)
    except Exception:
        pass
    _tile_cache[key] = arr
    return arr


def _hires_elevation(lat, lon):
    """Bilinear sample of the high-res Terrarium tile; None on any failure."""
    try:
        fx, fy = _deg2tile(lat, lon, _HI_ZOOM)
        tx, ty = int(fx), int(fy)
        arr = _load_tile(_HI_ZOOM, tx, ty)
        px = (fx - tx) * arr.shape[1] - 0.5
        py = (fy - ty) * arr.shape[0] - 0.5
        x0 = int(math.floor(px)); y0 = int(math.floor(py))
        dx = px - x0; dy = py - y0
        x0 = min(max(x0, 0), arr.shape[1] - 1); x1 = min(x0 + 1, arr.shape[1] - 1)
        y0 = min(max(y0, 0), arr.shape[0] - 1); y1 = min(y0 + 1, arr.shape[0] - 1)
        top = float(arr[y0, x0]) * (1 - dx) + float(arr[y0, x1]) * dx
        bot = float(arr[y1, x0]) * (1 - dx) + float(arr[y1, x1]) * dx
        return top * (1 - dy) + bot * dy
    except Exception:
        return None


# ── public API ──────────────────────────────────────────────────────────────

def elevation(lat, lon, hi_res=None):
    """Raw terrain elevation (m, signed) at (lat, lon).

    hi_res=True tries a network Terrarium tile first (cached), falling back to
    the bundled coarse grid on any failure so the call always returns a value.
    hi_res=None (default) follows the configured source (configure_terrain);
    hi_res=False forces the offline coarse grid.
    """
    if hi_res is None:
        hi_res = (_SOURCE == "terrarium")
    if hi_res:
        v = _hires_elevation(lat, lon)
        if v is not None:
            return v
    return _coarse_elevation(lat, lon)


def ground_elevation(lat, lon, hi_res=None):
    """Trajectory floor: land height on land, 0 (sea surface) over water."""
    return max(elevation(lat, lon, hi_res=hi_res), 0.0)
