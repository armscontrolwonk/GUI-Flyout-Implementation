"""Smoke tests for the two dynamic glide modes (see DAMPED_GLIDE.md and
GLIDE_CAPTURE_DESIGN.md).  Both remove the old "free lift" (drag is coupled to
the actual commanded lift, lift bounded by the aerodynamic ceiling):

  damped_glide              — NOMINAL = max-L/D α* (skip) lift + ζ altitude-rate
                              damping.  Preserves the phugoid, so ζ genuinely
                              DAMPS the skips: ζ=0 ≡ skip_glide (undamped), larger
                              ζ → fewer/smaller decaying skips into equilibrium.
  dynamic_equilibrium_glide — NOMINAL = equilibrium trim m·(g − V²/r); captures
                              SMOOTHLY (no oscillation).  ζ is a tracking gain.

Both PLUNGE an uncapturable (lofted) entry; capturability is entry-geometry
dependent (read off by glide_regime.py).

Run:  python damped_glide_smoke_test.py        (needs numpy + scipy)
"""

import copy
import json

import numpy as np

from missile_models import get_missile, rv_from_dict, rv_to_dict, effective_rv
from trajectory import integrate_trajectory
from glide_regime import regime_from_result

_BOOSTER = "Minotaur-IV + HTV-2"
_CHGB = rv_from_dict(json.load(open("rv_library/C-HGB.rv.json")))
_CUTOFF = 170.0


def _fly(mode, zeta=None, aero="polar", cutoff=_CUTOFF):
    """Lofted sub-circular C-HGB entry (steep → uncapturable)."""
    p = get_missile(_BOOSTER)
    rv = copy.deepcopy(_CHGB)
    rv.glider_guidance, rv.glider_aero_model = mode, aero
    if zeta is not None:
        rv.glider_damping_zeta = zeta
    p.rv = rv
    r = integrate_trajectory(p, 0.0, 0.0, 90.0, max_time_s=8000.0,
                             dt_output=2.0, cutoff_time_s=cutoff)
    return r, rv


def _fly_aur_shallow(mode, zeta=None, aero="polar"):
    """AUR on a depressed (shallow) insertion — capturable (adv-pitch program:
    stage 1 → 27°, stage 2 → 0° flat, launch elev 80°, az 103°, cutoff 117 s)."""
    p = copy.deepcopy(get_missile("AUR+HGB"))
    p.stage_turn_start_s, p.stage_turn_stop_s, p.stage_burnout_angle_deg = 1.0, 30.0, 27.0
    if getattr(p, "stage2", None) is not None:
        p.stage2.stage_turn_start_s = 54.0
        p.stage2.stage_turn_stop_s = 90.0
        p.stage2.stage_burnout_angle_deg = 0.0
    p.launch_elevation_deg = 80.0
    rv = copy.deepcopy(effective_rv(p))
    rv.glider_guidance, rv.glider_aero_model = mode, aero
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


def _alt(r):
    return np.asarray(r["alt"]).ravel() / 1000.0


def _range_km(r):
    return float(np.asarray(r["range"]).ravel()[-1] / 1000.0)


def _reclimb_km(r):
    a = _alt(r)
    post = a[int(np.argmax(a)):]
    return float(np.max(post - np.minimum.accumulate(post))) if len(post) else 0.0


# --------------------------------------------------------------------------
def test_serialization_roundtrip():
    for mode in ("damped_glide", "dynamic_equilibrium_glide"):
        rv = copy.deepcopy(_CHGB)
        rv.glider_guidance, rv.glider_damping_zeta = mode, 0.55
        rv2 = rv_from_dict(rv_to_dict(rv))
        assert rv2.glider_guidance == mode
        assert abs(rv2.glider_damping_zeta - 0.55) < 1e-9
    print("  ok  serialization round-trip (both modes)")


def test_damped_nests_to_skip_glide():
    # damped_glide ζ=0 reduces exactly to skip_glide (α* lift, no feedback).
    a_s, _ = _fly_aur_shallow("skip_glide")
    a_d, _ = _fly_aur_shallow("damped_glide", zeta=0.0)
    s, d = _alt(a_s), _alt(a_d)
    n = min(len(s), len(d))
    dmax = float(np.max(np.abs(s[:n] - d[:n])))
    assert len(s) == len(d) and dmax < 1e-9, f"ζ=0 not identical to skip_glide (max|Δ|={dmax})"
    print("  ok  damped_glide ζ=0 ≡ skip_glide (max|Δalt|=%.1e km)" % dmax)


def test_damped_decaying_skips():
    # Increasing ζ damps the phugoid: the skip amplitude (max re-climb) falls
    # monotonically toward zero.
    climbs = [_reclimb_km(_fly_aur_shallow("damped_glide", zeta=z)[0])
              for z in (0.0, 0.4, 0.7, 2.0)]
    assert climbs[0] > climbs[1] > climbs[2] >= climbs[3], f"skips not decaying: {climbs}"
    assert climbs[0] > 20.0 and climbs[3] < 1.0, f"endpoints off: {climbs}"
    print("  ok  damped_glide decaying skips: re-climb %s km (ζ=0→2)" %
          " → ".join(f"{c:.0f}" for c in climbs))


def test_damped_no_free_lift():
    # A captured damped glide must not out-range a true equilibrium glide.
    eq = _range_km(_fly_aur_shallow("equilibrium_glide", aero="constant_LD")[0])
    for z in (0.7, 1.0, 2.0):
        d = _range_km(_fly_aur_shallow("damped_glide", zeta=z, aero="constant_LD")[0])
        assert d <= eq * 1.02, f"ζ={z}: damped {d:.0f} > equilibrium {eq:.0f} (free lift!)"
    print(f"  ok  damped_glide no free lift (≤ equilibrium {eq:.0f} km)")


def test_lofted_plunges_both_modes():
    for mode in ("damped_glide", "dynamic_equilibrium_glide"):
        for aero in ("polar", "constant_LD"):
            r, rv = _fly(mode, zeta=0.7, aero=aero)
            g = regime_from_result(r, rv=rv)
            assert g.verdict == "plunge", f"{mode}/{aero} lofted: expected plunge, got {g}"
    print("  ok  lofted entry plunges (both modes, both aero)")


def test_dyneq_captures_smoothly():
    # dynamic_equilibrium_glide captures the shallow insertion with NO skips
    # (smooth) and no free lift.
    eq = _range_km(_fly_aur_shallow("equilibrium_glide", aero="polar")[0])
    r, rv = _fly_aur_shallow("dynamic_equilibrium_glide", zeta=0.7, aero="polar")
    g = regime_from_result(r, rv=rv)
    assert g.verdict == "capture", f"dyn-eq should capture, got {g}"
    assert _reclimb_km(r) < 1.0, "dyn-eq should capture smoothly (no climb)"
    assert _range_km(r) <= eq * 1.02, "dyn-eq exceeds equilibrium (free lift!)"
    print(f"  ok  dynamic_equilibrium_glide smooth capture ({_range_km(r):.0f} km, no climb)")


def test_no_zoom_climb():
    # No post-apogee zoom-climb — check the *capturing* shallow insertion (where
    # the β-cap must prevent the m·g_eff zoom for constant_LD) and the lofted boost.
    for fly in (_fly, _fly_aur_shallow):
        for mode in ("damped_glide", "dynamic_equilibrium_glide"):
            for aero in ("polar", "constant_LD"):
                for z in (0.0, 0.7, 2.0):
                    r, _ = fly(mode, zeta=z, aero=aero)
                    rc = _reclimb_km(r)
                    # damped_glide legitimately skips at low ζ (decaying phugoid);
                    # only flag a *runaway* zoom (> 60 km) there.  dyn-eq must be
                    # monotonic (< 1 km).
                    lim = 60.0 if mode == "damped_glide" else 1.0
                    assert rc < lim, f"{fly.__name__} {mode}/{aero} ζ={z}: re-climb {rc:.1f} km"
    print("  ok  no runaway zoom-climb (damped skips bounded; dyn-eq monotonic)")


def main():
    tests = [
        test_serialization_roundtrip,
        test_damped_nests_to_skip_glide,
        test_damped_decaying_skips,
        test_damped_no_free_lift,
        test_lofted_plunges_both_modes,
        test_dyneq_captures_smoothly,
        test_no_zoom_climb,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} glide-mode checks passed.")


if __name__ == "__main__":
    main()
