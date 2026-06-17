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


def _fly_aur_shallow(mode, zeta=None, aero="polar"):
    """AUR on a *depressed* (shallow) insertion — the adv-pitch program from a
    saved AUR guidance file (stage 1 → 27°, stage 2 → 0° flat, launch elev 80°,
    az 103°, cutoff 117 s).  Apogee ~110 km, high horizontal speed: a capturable
    insertion (contrast the lofted C-HGB above)."""
    p = copy.deepcopy(get_missile("AUR+HGB"))
    p.stage_turn_start_s, p.stage_turn_stop_s, p.stage_burnout_angle_deg = 1.0, 30.0, 27.0
    if getattr(p, "stage2", None) is not None:
        p.stage2.stage_turn_start_s = 54.0
        p.stage2.stage_turn_stop_s = 90.0
        p.stage2.stage_burnout_angle_deg = 0.0
    p.launch_elevation_deg = 80.0
    from missile_models import effective_rv
    rv = copy.deepcopy(effective_rv(p))
    rv.glider_guidance = mode
    rv.glider_aero_model = aero
    if zeta is not None:
        rv.glider_damping_zeta = zeta
    node = p
    while node is not None:
        if node.rv is not None:
            node.rv = rv
            break
        node = node.stage2
    r = integrate_trajectory(p, 28.458, -80.5286, 103.0, guidance="pitch_program",
                             burnout_angle_deg=25.0, cutoff_time_s=117.0,
                             gt_turn_start_s=5.0, launch_elevation_deg=80.0,
                             max_time_s=8000.0, dt_output=2.0)
    return r, rv


def test_shallow_insertion_captures():
    # Capturability is entry-geometry dependent: where the lofted C-HGB plunges,
    # a shallow depressed insertion is captured by the same polar EOM law, with
    # ζ damping — and still no free lift (range ≤ analytic equilibrium glide).
    r_eq, rv_eq = _fly_aur_shallow("equilibrium_glide")
    eq_km = _range_km(r_eq)
    r_d, rv_d = _fly_aur_shallow("damped_glide", zeta=0.7, aero="polar")
    g = regime_from_result(r_d, rv=rv_d)
    assert g.verdict == "capture", f"polar damped_glide should capture the shallow insertion, got {g}"
    assert _range_km(r_d) <= eq_km * 1.02, "captured glide exceeds equilibrium (free lift!)"
    print(f"  ok  shallow insertion captures (polar damped ζ=0.7 → {_range_km(r_d):.0f} km, "
          f"≤ equilibrium {eq_km:.0f} km)")


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
        test_shallow_insertion_captures,
        test_no_zoom_climb,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} damped_glide checks passed.")


if __name__ == "__main__":
    main()
