"""Terrain DEM: real launch elevation and ground-height termination.

TODO item 3(c).  Before this feature every trajectory began and ended on a
flat sea-level Earth: a launch from Xichang (1 857 m) started at 0 m and an
impact in the Tibetan plateau "flew" 4 000 m underground before the
integrator noticed.  These tests pin:

  1. terrain.py's bundled coarse grid returns sane elevations at known
     points (Denver, Everest, oceans), is continuous across the antimeridian,
     and floors to sea level over water (ground_elevation).
  2. The source selector (configure_terrain / MODEL_OPTIONS['terrain'])
     validates its choices and always yields a value — hi-res falls back to
     the coarse grid on any network failure.
  3. integrate_trajectory(terrain_dem=False) is byte-identical to the legacy
     flat-Earth dynamics (the Forden benchmark condition), while
     terrain_dem=True starts the state at the pad's real elevation,
     honours a baked launch_elev_m override, and terminates on terrain.
  4. launch_sites.json carries a baked elev_m + provenance for every site.
"""

import json
import os

import numpy as np
import pytest

import terrain
import booster_models as mm
from booster_models import get_booster, load_booster_library
from trajectory import integrate_trajectory

load_booster_library()

_CACHE = {}


def _fly(**kw):
    key = repr(sorted(kw.items()))
    if key not in _CACHE:
        p = get_booster("No-dong")
        # Denver-ish pad: high, flat, far from coasts — coarse-grid friendly.
        _CACHE[key] = integrate_trajectory(p, 39.74, -104.99, 90.0,
                                           max_time_s=3600.0, **kw)
    return _CACHE[key]


# ── 1. coarse grid sanity ───────────────────────────────────────────────────

def test_coarse_grid_present():
    assert terrain.have_coarse()


@pytest.mark.parametrize("lat,lon,lo,hi", [
    (39.74, -104.99, 1400.0, 1900.0),      # Denver ~1600 m
    (27.99,   86.93, 5000.0, 8900.0),      # Everest massif (0.05° cell mean)
    (28.46,  -80.53,  -50.0,   50.0),      # Cape Canaveral, near sea level
    (0.0,    -30.0, -6500.0, -2000.0),     # mid-Atlantic abyssal plain
    (45.82,    6.86, 1500.0, 4800.0),      # Mont Blanc massif
])
def test_coarse_known_elevations(lat, lon, lo, hi):
    e = terrain.elevation(lat, lon, hi_res=False)
    assert lo <= e <= hi, f"({lat},{lon}) -> {e} m outside [{lo},{hi}]"


def test_ground_elevation_floors_to_sea_level():
    # Open ocean: raw elevation is deep negative, trajectory floor is 0.
    assert terrain.elevation(0.0, -30.0, hi_res=False) < -1000.0
    assert terrain.ground_elevation(0.0, -30.0, hi_res=False) == 0.0
    # Land: floor equals the raw elevation.
    e = terrain.elevation(39.74, -104.99, hi_res=False)
    assert terrain.ground_elevation(39.74, -104.99, hi_res=False) == e


def test_coarse_antimeridian_continuity():
    """Bilinear longitude wrap: no seam crossing ±180°."""
    e_w = terrain.elevation(-17.8, 179.999, hi_res=False)
    e_e = terrain.elevation(-17.8, -179.999, hi_res=False)
    assert abs(e_w - e_e) < 50.0


def test_coarse_bilinear_continuity():
    """Adjacent samples 0.001° apart differ by far less than a cell."""
    e0 = terrain.elevation(39.740, -104.990, hi_res=False)
    e1 = terrain.elevation(39.741, -104.990, hi_res=False)
    assert abs(e0 - e1) < 30.0


def test_poles_do_not_crash():
    for lat in (90.0, -90.0):
        assert np.isfinite(terrain.elevation(lat, 0.0, hi_res=False))


# ── 2. source selection ─────────────────────────────────────────────────────

def test_configure_terrain_validates():
    with pytest.raises(ValueError):
        terrain.configure_terrain("srtm")


def test_source_round_trip():
    orig = terrain.terrain_source()
    try:
        terrain.configure_terrain("coarse")
        assert terrain.terrain_source() == "coarse"
        # In coarse mode the default lookup IS the offline grid.
        assert (terrain.elevation(39.74, -104.99)
                == terrain.elevation(39.74, -104.99, hi_res=False))
        terrain.configure_terrain("terrarium")
        assert terrain.terrain_source() == "terrarium"
    finally:
        terrain.configure_terrain(orig)


def test_model_options_entry():
    """The GUI's Analysis ▸ Reference Data menu builds itself from
    MODEL_OPTIONS; the terrain entry must be present and wired through."""
    spec = mm.MODEL_OPTIONS["terrain"]
    assert set(spec["choices"]) == {"terrarium", "coarse"}
    orig = mm.get_model_option("terrain")
    try:
        mm.set_model_option("terrain", "coarse")
        assert terrain.terrain_source() == "coarse"
    finally:
        mm.set_model_option("terrain", orig)


def test_hires_always_returns_a_value():
    """hi_res=True must degrade to the coarse grid on any failure — the call
    can never raise or return None (offline runs included)."""
    v = terrain.elevation(39.74, -104.99, hi_res=True)
    assert np.isfinite(v) and 1400.0 <= v <= 1900.0


# ── 3. trajectory integration ───────────────────────────────────────────────

def test_default_off_is_byte_identical():
    r_legacy = _fly()
    r_off = _fly(terrain_dem=False)
    assert np.array_equal(np.asarray(r_legacy['alt']), np.asarray(r_off['alt']))
    assert r_legacy['range_km'] == r_off['range_km']


def test_dem_starts_at_pad_elevation():
    r = _fly(terrain_dem=True)
    pad = terrain.ground_elevation(39.74, -104.99, hi_res=False)
    assert abs(np.asarray(r['alt'])[0] - pad) < 5.0


def test_launch_elev_override_wins():
    r = _fly(terrain_dem=True, launch_elev_m=1609.0)
    assert abs(np.asarray(r['alt'])[0] - 1609.0) < 5.0


def test_dem_terminates_on_terrain():
    """The trajectory stops at real ground height, not sea level.

    Output arrays end on the last dt_output sample before the terminal event
    (legacy behavior — the event state is not appended), so the final sample
    sits within one output step of the ground: at or above it, by less than
    the distance the RV falls in dt_output seconds."""
    r = _fly(terrain_dem=True)
    g = terrain.ground_elevation(r['impact_lat'], r['impact_lon'], hi_res=False)
    # An eastward No-dong shot from Denver lands in the US interior — the
    # ground there is real terrain, well above sea level.
    assert g > 100.0
    last_alt = float(np.asarray(r['alt'])[-1])
    assert last_alt >= g - 30.0
    assert last_alt - g < abs(r['impact_speed_ms']) * 1.5
    # The flat-Earth run keeps falling past the terrain to sea level.
    assert float(np.asarray(_fly()['alt'])[-1]) < g


def test_dem_elevated_pad_extends_range():
    """Starting 1.6 km up with less atmosphere to punch through must not
    shorten the shot; the gain stays small (sanity bound)."""
    r0, r1 = _fly(), _fly(terrain_dem=True)
    assert r1['range_km'] > r0['range_km']
    assert r1['range_km'] - r0['range_km'] < 50.0


def test_dem_ocean_launch_matches_flat_earth_start():
    p = get_booster("No-dong")
    r = integrate_trajectory(p, 40.0, 134.0, 90.0,        # Sea of Japan
                             terrain_dem=True, max_time_s=3600.0)
    assert abs(np.asarray(r['alt'])[0]) < 5.0


# ── 4. baked site elevations ────────────────────────────────────────────────

def _sites():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "launch_sites.json")
    with open(path) as fh:
        return json.load(fh)


def test_all_sites_have_baked_elevation():
    for s in _sites():
        assert isinstance(s.get("elev_m"), int), s["name"]
        assert "Terrarium" in s.get("elev_source", ""), s["name"]


def test_baked_elevations_plausible():
    """Baked values agree with the coarse grid to within its cell scale —
    catches lat/lon transposition or unit errors in the bake.  700 m bound:
    a 0.05° (~5.5 km) cell averages the surrounding relief, so a valley pad
    in steep terrain (Xichang: 1 857 m pad in a ~2 400 m-mean cell) can sit
    several hundred metres below the cell value — exactly why the hi-res
    bake exists."""
    for s in _sites():
        coarse = terrain.ground_elevation(s["lat"], s["lon"], hi_res=False)
        assert abs(s["elev_m"] - coarse) < 700.0, (
            f"{s['name']}: baked {s['elev_m']} m vs coarse {coarse:.0f} m")
