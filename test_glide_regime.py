"""Tests for the glide-regime classifier (glide_regime.py).

Validates the diagnostic skip / capture / plunge classifier against trajectories
whose physical regime is known and *stable* (independent of the pending
damped_glide lift-law rebuild):

  - skip_glide on a lofted sub-circular boost  → PLUNGE (α* lift cannot pull
    out of the steep lofted entry; ~22 g, no glide).  See GLIDE_CAPTURE_DESIGN.md.
  - equilibrium_glide on the same boost        → CAPTURE (force-balance trim
    holds a sustained glide).
  - skip_glide on the near-orbital built-in    → SKIP (lofts back above the
    100 km interface; the classic undamped skip).

Also exercises synthetic trajectories so the core logic is covered without the
integrator.

Run:  python test_glide_regime.py        (integrator tests need numpy + scipy)
"""

import copy
import json

import numpy as np

from glide_regime import classify_glide_regime, regime_from_result


# --------------------------------------------------------------------------
# Synthetic-trajectory unit tests (no integrator) — pin the core logic.
# --------------------------------------------------------------------------
def _traj(alt_km, speed_ms, dt=2.0):
    t = np.arange(len(alt_km)) * dt
    return np.asarray(alt_km, float), np.asarray(speed_ms, float), t


def test_synthetic_capture():
    # Apogee then a long, *shallow* in-atmosphere glide (|γ| ≈ 1–2°): 55 km of
    # descent spread over ~150 steps × 2 s at multi-km/s speed.
    up = [85, 95, 100]
    glide = list(np.linspace(98, 43, 150))         # very slow shallow descent
    alt = up + glide
    spd = list(np.linspace(5000, 5100, len(up))) + list(np.linspace(5100, 2500, len(glide)))
    r = classify_glide_regime(*_traj(alt, spd))
    assert r.verdict == "capture", r
    print("  ok  synthetic capture →", r)


def test_synthetic_skip():
    # Two shallow dips into the atmosphere, each lofting back above the 100 km
    # interface (the classic undamped skip); gentle speed loss → low decel.
    dip = [120, 108, 95, 78, 70, 78, 95, 108]
    alt = dip + dip + [115, 100, 80, 55, 30]
    spd = list(np.linspace(7000, 6200, len(alt)))   # gentle: ~0.3 g
    r = classify_glide_regime(*_traj(alt, spd, dt=10.0))
    assert r.verdict == "skip", r
    assert r.n_reascents >= 1
    print("  ok  synthetic skip →", r)


def test_synthetic_inatmosphere_skip():
    # Phugoid skips that oscillate *below* the 100 km interface (never re-ascend
    # above it) are still skips — caught by the re-climb criterion, not re-ascent.
    dip = [95, 75, 55, 75, 92]
    alt = dip + dip + [88, 70, 50, 30]   # ~40 km re-climbs, all < 100 km
    spd = list(np.linspace(6000, 5200, len(alt)))   # gentle → low decel
    r = classify_glide_regime(*_traj(alt, spd, dt=10.0))
    assert r.verdict == "skip", r
    assert r.n_reascents == 0 and r.max_reclimb_km > 15.0, r
    print("  ok  synthetic in-atmosphere skip →", r)


def test_synthetic_plunge_decel():
    # Steep monotonic dive; large deceleration as it slams into dense air.
    alt = list(np.linspace(120, 5, 40))
    # speed roughly constant high, then a sharp drop near the bottom (high decel)
    spd = [7000] * 30 + list(np.linspace(7000, 1000, 10))
    r = classify_glide_regime(*_traj(alt, spd), g_limit_g=10.0)
    assert r.verdict == "plunge", r
    print("  ok  synthetic plunge (decel) →", r)


def test_synthetic_no_entry():
    # Stays well above the interface the whole time (e.g. vacuum lob).
    alt = list(np.linspace(300, 150, 20)) + list(np.linspace(150, 200, 20))
    spd = [7600] * 40
    r = classify_glide_regime(*_traj(alt, spd))
    assert r.verdict == "no_entry", r
    print("  ok  synthetic no_entry →", r)


# --------------------------------------------------------------------------
# Integrator-backed tests — real Thrusty trajectories, stable fixtures.
# --------------------------------------------------------------------------
def _fly(mode, cutoff, aero="polar", zeta=None, boost="Minotaur-IV + HTV-2"):
    from booster_models import (get_booster, ro_from_dict,
                                 apply_reentry_plan, load_reentry_plan)
    from trajectory import integrate_trajectory
    # Reentry-object files are hardware-only; apply the shipped reentry plan to
    # get the ready-to-fly object (glide mode, L/D, beta, separation).
    chgb = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    _rp = load_reentry_plan("C-HGB")
    if _rp is not None:
        chgb = apply_reentry_plan(chgb, _rp)
    # Pin the terminal-dive command these fixtures were written against
    # (dive below 30 km) so the regime verdicts don't track the shipped
    # plan's tunable dive altitude (now 0 = glide to impact by default).
    chgb.glider_terminal_dive = True
    chgb.glider_terminal_alt_km = 30.0
    p = get_booster(boost)
    ro = copy.deepcopy(chgb)
    # These fixtures were tuned when the Minotaur file carried a 1,000 kg
    # payload baked into its stage masses; booster files are stack-only now
    # and the object's mass is composed on, so pin the same boosted mass.
    ro.mass_kg = 1000.0
    ro.glider_guidance = mode
    ro.glider_aero_model = aero
    if zeta is not None:
        ro.glider_damping_zeta = zeta
    p.ro = ro
    r = integrate_trajectory(p, 0.0, 0.0, 90.0, max_time_s=6000.0,
                             dt_output=2.0, cutoff_time_s=cutoff)
    return r, ro


def test_skip_glide_lofted_is_plunge():
    r, ro = _fly("skip_glide", 170.0)
    g = regime_from_result(r, ro=ro)
    assert g.verdict == "plunge", g
    print("  ok  skip_glide (lofted sub-circular) →", g)


def test_equilibrium_glide_is_capture():
    r, ro = _fly("equilibrium_glide", 170.0)
    g = regime_from_result(r, ro=ro)
    assert g.verdict == "capture", g
    print("  ok  equilibrium_glide →", g)


def test_skip_glide_near_orbital_is_skip():
    # Full HTV-2 burn → near-orbital entry; the classic undamped skip.
    r, ro = _fly("skip_glide", None)
    g = regime_from_result(r, ro=ro)
    assert g.verdict == "skip", g
    print("  ok  skip_glide (near-orbital) →", g)


def main():
    tests = [
        test_synthetic_capture, test_synthetic_skip,
        test_synthetic_inatmosphere_skip,
        test_synthetic_plunge_decel, test_synthetic_no_entry,
        test_skip_glide_lofted_is_plunge, test_equilibrium_glide_is_capture,
        test_skip_glide_near_orbital_is_skip,
    ]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} glide_regime checks passed.")


if __name__ == "__main__":
    main()
