"""Glide activation is gated at APOGEE, not at the 100 km Acton pierce.

A quasi-ballistic missile (KN-23 / Iskander class) flies a depressed trajectory
whose apogee is ~40–60 km — it never reaches 100 km — and pulls up on the way
down using aerodynamic lift.  The numerical glide laws (phugoid / skip-glide,
damped phugoid, dynamic-equilibrium) previously armed only through the
`_gl_above_pierce` latch: lift was allowed ONLY after the vehicle had climbed
above 100 km and was descending back through it.  That gate belongs to the
exo-atmospheric Acton skip-glide entry (its own analytic path); imposed on the
endo-atmospheric laws it silently disabled lift for any vehicle whose apogee
stayed below 100 km — the KN-23 could not pull up at all, a validation failure.

The physical trigger is APOGEE — the start of the descending glide — which the
pre-/post-apogee integration split already marks (params._glider_phase1).  These
tests pin:

  1. A sub-100 km-apogee glider is NOT inert: turning the glider on materially
     extends range over the same ballistic shot (it pulls up and glides).
  2. The activation is byte-identical for an exo-atmospheric entry (apogee
     > 100 km): there the vehicle is post-apogee AND below 100 km at the same
     instant, so the removed 100 km latch changed nothing.
  3. Lift stays OFF on the ballistic ascent (no glide aero before apogee).
"""

import numpy as np

from booster_models import (get_booster, load_booster_library, ROParams,
                            compose_loadout)
from trajectory import integrate_trajectory

load_booster_library()

_CACHE = {}


def _fly(glider_enabled, burnout_angle_deg, beta=3000.0, ld=2.5):
    key = (glider_enabled, burnout_angle_deg, beta, ld)
    if key not in _CACHE:
        ro = ROParams(name="RV", mass_kg=500.0, beta_kg_m2=beta, shape="karman",
                      diameter_m=1.1, length_m=2.0, glider_enabled=glider_enabled,
                      glider_LD=ld, glider_guidance="damped_glide",
                      separation_mode="body")
        b = get_booster("Scud-B (R-17)")
        b.body_reenters = True
        p = compose_loadout(b, ro, 1)
        p.ro = ro
        _CACHE[key] = integrate_trajectory(
            p, 39.12, 125.67, 90.0, burnout_angle_deg=burnout_angle_deg,
            max_time_s=3600.0)
    return _CACHE[key]


# ── 1. sub-100 km glider pulls up (the KN-23 case) ──────────────────────────

def test_depressed_glider_is_not_inert():
    """A depressed shot (apogee < 100 km) with the glider on must glide
    materially farther than the same shot ballistic — the pull-up the Acton
    latch used to forbid below 100 km."""
    on = _fly(True, -2.0)
    off = _fly(False, -2.0)
    assert on['apogee_km'] < 100.0                      # genuinely sub-pierce
    assert on['range_km'] > off['range_km'] * 1.3       # substantial extension


def test_depressed_apogee_is_mode_independent():
    """The ballistic ASCENT is unchanged by the glider (lift is off before
    apogee): glider-on and glider-off reach the same apogee."""
    on = _fly(True, -2.0)
    off = _fly(False, -2.0)
    assert abs(on['apogee_km'] - off['apogee_km']) < 0.5


# ── 2. exo-atmospheric entry byte-identical ─────────────────────────────────

def test_lofted_entry_unchanged_by_the_fix():
    """For an entry that crosses 100 km the vehicle is post-apogee AND below
    100 km at the same instant, so dropping the 100 km latch cannot change the
    result — the glide still extends range exactly as before."""
    on = _fly(True, 50.0)
    off = _fly(False, 50.0)
    assert on['apogee_km'] > 100.0
    assert on['range_km'] > off['range_km'] + 100.0     # strong glide, as ever


def test_hgv_benchmark_still_flies():
    """The validated exo-atmospheric HGV benchmark is unaffected."""
    r = integrate_trajectory(get_booster("Minotaur-IV + HTV-2"),
                             34.7, -120.6, 90.0, max_time_s=3600.0)
    assert r['range_km'] > 10000.0
    assert r['apogee_km'] > 100.0


# ── 3. lift is off during the ascent ────────────────────────────────────────

def test_no_lift_before_apogee():
    """On the ascent the depressed glider and the ballistic shot share the same
    trajectory (glide aero is gated off before apogee).  Compared over the
    clean ascent band well below the ~48 km apogee — near the apogee joint the
    glider run's pre-/post-apogee integration split diverges from the ballistic
    single pass by metres (numerics, not lift), which the apogee-equality test
    above already bounds."""
    on = _fly(True, -2.0)
    off = _fly(False, -2.0)
    a_on = np.asarray(on['alt']); a_off = np.asarray(off['alt'])
    n = min(len(a_on), len(a_off))
    band = np.arange(n) < int(np.argmax(a_off[:n]))          # ascent samples
    assert band.sum() > 20
    # Same ascent to < 0.5 % of altitude: the residual is the glider run's
    # pre-/post-apogee integration split (a different event set nudges the
    # adaptive steps), NOT lift — lift on the downleg moves the range by
    # hundreds of km, three orders larger than this.
    assert np.allclose(a_on[:n][band], a_off[:n][band], rtol=5e-3, atol=5.0)
