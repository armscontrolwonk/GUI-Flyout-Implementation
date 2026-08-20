"""Bake a coarse global elevation grid for Thrusty from AWS Terrarium tiles.

Source: AWS "elevation-tiles-prod" open terrain tiles (Terrarium PNG encoding),
        s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png
        — a global, public-domain / open blend of SRTM, GMTED2010, ETOPO1,
        and national datasets.  Elevation (metres) = R*256 + G + B/256 − 32768.

Bake plan (reproducible):
  * Fetch the full z5 tile set (32×32 = 1024 tiles, each 256 px) → an
    8192×8192 native grid at ~0.044° resolution (Web-Mercator rows).
  * Reproject the Mercator rows to an equirectangular 0.05° lat grid and
    resample to a 7200 (lon) × 3600 (lat) int16 array covering the full
    globe, lon −180..+180, lat +90..−90 (row 0 = +90°).
  * Write data/dem/terrain_0p05deg.npy (int16 metres) + MANIFEST.md.

The runtime module (terrain.py) memory-maps that array for the coarse
lookup and fetches higher-zoom Terrarium tiles on demand for detail.

Run:  python3 dem_build.py          (needs network; ~89 MB download)
      python3 dem_build.py sites    (re-bake elev_m into launch_sites.json
                                     from hi-res tiles; needs network)
"""

import io
import math
import os
import sys
import time
import urllib.request

import numpy as np
from PIL import Image

_TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
_Z = 5                                   # 32×32 tiles → 8192×8192 native
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "dem")
_GRID_LON = 7200                         # 0.05° in longitude
_GRID_LAT = 3600                         # 0.05° in latitude


def _decode(rgb):
    """Terrarium RGB → elevation in metres."""
    a = rgb.astype(np.float64)
    return a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0


def _fetch_tile(z, x, y, retries=4):
    url = _TILE_URL.format(z=z, x=x, y=y)
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                img = Image.open(io.BytesIO(r.read())).convert("RGB")
            return np.asarray(img, dtype=np.uint8)
        except Exception as exc:                       # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 ** k)
    raise RuntimeError("unreachable")


def _mercator_native():
    """Assemble the full z5 tile set into an 8192×8192 native Mercator grid."""
    n = 2 ** _Z
    side = n * 256
    native = np.empty((side, side), dtype=np.float32)
    total = n * n
    done = 0
    for ty in range(n):
        for tx in range(n):
            rgb = _fetch_tile(_Z, tx, ty)
            native[ty * 256:(ty + 1) * 256, tx * 256:(tx + 1) * 256] = _decode(rgb)
            done += 1
            if done % 32 == 0 or done == total:
                print(f"  tiles {done}/{total}", flush=True)
    return native


def _mercator_row_to_lat(row, side):
    """Latitude (deg) at a native Mercator pixel row (row 0 = top = +85.05°)."""
    n = side  # pixels
    yfrac = (row + 0.5) / n
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * yfrac)))
    return math.degrees(lat_rad)


def _to_equirect(native):
    """Resample the Mercator native grid to a 0.05° equirectangular grid.

    Longitude is linear in both projections, so columns resample directly.
    Latitude rows are remapped through the inverse-Mercator relation; the
    poleward bands beyond Web-Mercator's ±85.05° cutoff are filled by holding
    the edge value (no launch/impact happens there in practice)."""
    side = native.shape[0]
    # Longitude: native 0..side ↔ lon −180..+180; target 0.05° columns.
    src_cols = (np.arange(_GRID_LON) + 0.5) / _GRID_LON * side
    src_cols = np.clip(src_cols.astype(np.int64), 0, side - 1)
    native_lonres = native[:, src_cols]                 # side × 7200

    # Latitude of each native row (descending from +85.05° to −85.05°).
    row_lat = np.array([_mercator_row_to_lat(r, side) for r in range(side)])
    # Target latitudes: +90 (row 0) → −90, centre of each 0.05° cell.
    tgt_lat = 90.0 - (np.arange(_GRID_LAT) + 0.5) * (180.0 / _GRID_LAT)

    out = np.empty((_GRID_LAT, _GRID_LON), dtype=np.int16)
    # row_lat is descending; np.interp needs ascending x → flip.
    row_lat_asc = row_lat[::-1]
    native_asc = native_lonres[::-1, :]
    for col in range(_GRID_LON):
        out[:, col] = np.round(
            np.interp(tgt_lat, row_lat_asc, native_asc[:, col])
        ).astype(np.int16)
    return out


def bake_sites():
    """Stamp each launch_sites.json entry with a hi-res pad elevation.

    One-time (re-runnable) network bake: elev_m rounded to the metre plus an
    elev_source provenance string.  The GUI prefers this baked value over the
    coarse grid whenever the lat/lon fields still match the site."""
    import datetime
    import json
    import terrain

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "launch_sites.json")
    with open(path) as fh:
        sites = json.load(fh)
    today = datetime.date.today().isoformat()
    for s in sites:
        lat, lon = float(s["lat"]), float(s["lon"])
        elev = terrain.elevation(lat, lon, hi_res=True)
        s["elev_m"] = round(float(elev))
        s["elev_source"] = f"AWS Terrarium z{terrain._HI_ZOOM} (retrieved {today})"
        print(f"  {s['name']:<28s} {s['elev_m']:>6d} m")
    with open(path, "w") as fh:
        json.dump(sites, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote {path} ({len(sites)} sites)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "sites":
        return bake_sites()
    os.makedirs(_OUT_DIR, exist_ok=True)
    print(f"Fetching z{_Z} Terrarium tiles (1024) …", flush=True)
    native = _mercator_native()
    print("Resampling to 0.05° equirectangular grid …", flush=True)
    grid = _to_equirect(native)
    out_path = os.path.join(_OUT_DIR, "terrain_0p05deg.npy")
    np.save(out_path, grid)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"Wrote {out_path}  ({size_mb:.1f} MB, {grid.shape} int16)")
    # Provenance manifest.
    man = os.path.join(_OUT_DIR, "MANIFEST.md")
    with open(man, "w") as fh:
        fh.write(
            "# Thrusty coarse DEM — provenance\n\n"
            "- **File**: terrain_0p05deg.npy — int16 metres, "
            f"{grid.shape[0]}×{grid.shape[1]} (lat×lon), 0.05° equirectangular,\n"
            "  row 0 = +90° lat, col 0 = −180° lon, cell centres.\n"
            "- **Source**: AWS `elevation-tiles-prod` Terrarium tiles, zoom 5\n"
            "  (`s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`),\n"
            "  a global blend of SRTM, GMTED2010, ETOPO1, and national DEMs.\n"
            "- **Encoding decoded**: elev_m = R*256 + G + B/256 − 32768.\n"
            "- **Resample**: z5 native Mercator (8192²) → inverse-Mercator to\n"
            "  0.05° equirectangular; poleward of ±85.05° holds the edge value.\n"
            "- **Reproducible**: `python3 dem_build.py`.\n"
            f"- **Baked**: elevation range {int(grid.min())}..{int(grid.max())} m.\n"
        )
    print(f"Wrote {man}")


if __name__ == "__main__":
    sys.exit(main())
