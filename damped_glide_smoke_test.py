"""Smoke test for the damped_glide guidance mode (see DAMPED_GLIDE.md and
GLIDE_CAPTURE_DESIGN.md).

damped_glide is a PURE DYNAMIC EOM glide law: the nominal is the equilibrium-
glide trim L·cosσ = m·(g − V²/r), plus ζ altitude-rate damping of the residual
phugoid; lift is bounded by the aerodynamic ceiling and drag is coupled to the
actual commanded lift (no "free" lift).  Consequences this test pins:

  1. SERIALIZATION round-trips the ζ knob.
  2. NO FREE LIFT — a captured glide never exceeds the analytic equilibrium-glide
     range (effective L/D ≤ vehicle L/D).  [This is what the old free-lift bug
     violated, gliding ~20 % too far.]
  3. HONEST PLUNGE — with the physical polar aero, the C-HGB on a *lofted*
     ballistic boost cannot dynamically pull out of the thin-air entry and
     PLUNGES, at every ζ (capturability is entry-geometry dependent; a lofted
     ballistic insertion is not capturable — real boost-glide uses a shallow,
     depressed insertion).
  4. DAMPING IMPROVES CAPTURE — in the lumped constant_LD model (which lacks an
     aerodynamic lift ceiling, so it *can* pull out), range is non-decreasing in
     ζ and the glide is captured at high ζ.
  5. NO ZOOM-CLIMB — the equilibrium trim descends monotonically post-apogee.

Run:  python damped_glide_smoke_test.py        (needs numpy + scipy)
"""

import copy
import json

import numpy as np

from missile_models import get_missile, rv_from_dict, rv_to_dict
from trajectory import integrate_trajectory
from glide_regime import regime_from_result

_BOOSTER = "Minotaur-IV + HTV-2"
_CHGB = rv_from_dict(json.load(open("rv_library/C-HGB.rv.json")))
_CUTOFF = 170.0                          # sub-circular lofted entry (~5.6 km/s)


def _fly(mode, zeta=None, aero="polar", cutoff=_CUTOFF):
    p = get_missile(_BOOSTER)
    rv = copy.deepcopy(_CHGB)
    rv.glider_guidance = mode
    rv.glider_aero_model = aero
    if zeta is not None:
        rv.glider_damping_zeta = zeta
    p.rv = rv
    r = integrate_trajectory(p, 0.0, 0.0, 90.0, max_time_s=8000.0,
                             dt_output=2.0, cutoff_time_s=cutoff)
    return r, rv


def _range_km(r):
    return float(np.asarray(r['range']).ravel()[-1] / 1000.0)


def _post_apogee_reclimb_km(r):
    alt = np.asarray(r['alt']).ravel() / 1000.0
    post = alt[int(np.argmax(alt)):]
    return float(np.max(post - np.minimum.accumulate(post))) if len(post) else 0.0


def test_serialization_roundtrip():
    rv = copy.deepcopy(_CHGB)
    rv.glider_guidance = "damped_glide"
    rv.glider_damping_zeta = 0.55
    rv2 = rv_from_dict(rv_to_dict(rv))
    assert rv2.glider_guidance == "damped_glide"
    assert abs(rv2.glider_damping_zeta - 0.55) < 1e-9
    print("  ok  serialization round-trip")


def test_no_free_lift_bound():
    # A captured damped glide must not out-range a true equilibrium glide
    # (effective L/D ≤ vehicle L/D).  Use constant_LD, which captures here.
    r_eq, _ = _fly("equilibrium_glide", aero="constant_LD")
    eq_km = _range_km(r_eq)
    for z in (0.7, 1.0, 2.0):
        r_d, _ = _fly("damped_glide", zeta=z, aero="constant_LD")
        d_km = _range_km(r_d)
        assert d_km <= eq_km * 1.02, \
            f"ζ={z}: damped range {d_km:.0f} exceeds equilibrium {eq_km:.0f} (free lift!)"
    print(f"  ok  no free lift: damped ≤ equilibrium-glide range ({eq_km:.0f} km)")


def test_polar_lofted_plunges():
    # Physical polar aero: a lofted ballistic entry cannot be dynamically pulled
    # out of (thin-air lift ceiling) → plunge, at every ζ.
    for z in (0.0, 0.7, 2.0):
        r, rv = _fly("damped_glide", zeta=z, aero="polar")
        g = regime_from_result(r, rv=rv)
        assert g.verdict == "plunge", f"ζ={z} polar: expected plunge, got {g}"
    print("  ok  polar lofted entry plunges at every ζ (honest dynamic capture)")


def test_damping_improves_capture_constant_ld():
    # Lumped constant_LD (no aero lift ceiling) can pull out: range is
    # non-decreasing in ζ and captures at high ζ.
    rngs = []
    for z in (0.0, 0.7, 2.0):
        r, _ = _fly("damped_glide", zeta=z, aero="constant_LD")
        rngs.append(_range_km(r))
    assert rngs[0] <= rngs[1] + 1.0 <= rngs[2] + 2.0, f"range not non-decreasing in ζ: {rngs}"
    r_hi, rv_hi = _fly("damped_glide", zeta=2.0, aero="constant_LD")
    assert regime_from_result(r_hi, rv=rv_hi).verdict == "capture", "ζ=2 constant_LD should capture"
    print(f"  ok  damping improves capture (constant_LD range {rngs[0]:.0f}→{rngs[2]:.0f} km, ζ=0→2)")


def test_no_zoom_climb():
    for aero in ("polar", "constant_LD"):
        for z in (0.0, 0.7, 2.0):
            r, _ = _fly("damped_glide", zeta=z, aero=aero)
            rc = _post_apogee_reclimb_km(r)
            assert rc < 1.0, f"{aero} ζ={z}: post-apogee zoom-climb {rc:.1f} km"
    print("  ok  no post-apogee zoom-climb (equilibrium trim descends cleanly)")


def main():
    tests = [
        test_serialization_roundtrip,
        test_no_free_lift_bound,
        test_polar_lofted_plunges,
        test_damping_improves_capture_constant_ld,
        test_no_zoom_climb,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} damped_glide checks passed.")


if __name__ == "__main__":
    main()
