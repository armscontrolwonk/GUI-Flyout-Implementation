"""
Smoke test for the damped_glide guidance mode (see DAMPED_GLIDE.md).

Validated on the C-HGB glide body (rv_library/C-HGB.rv.json) — the repo's
SWERVe/AHW-descendant Common Hypersonic Glide Body (Gulan, Georgia Tech, 2024:
SWERVe is the publicly-available C-HGB predecessor) — flown on a sub-circular
(MRBM-class) boost so it actually glides IN the atmosphere rather than skipping
out of it.  The HTV-2-on-Minotaur built-in is deliberately NOT used here: it
re-enters near orbital speed (~7.6 km/s) and genuinely skips above 100 km, which
is skip-entry, not gliding.

Verifies:
  1. NESTING — damped_glide(ζ=0) reproduces skip_glide bit-exactly (safety).
  2. IN-ATMOSPHERE GLIDE — increasing ζ keeps the glide in the atmosphere
     instead of skipping out above 100 km (where there is no air to glide on),
     which is the whole point of the mode.

Run:  python damped_glide_smoke_test.py        (needs numpy + scipy)
"""

import copy
import json
import numpy as np
from missile_models import get_missile, rv_from_dict
from trajectory import integrate_trajectory

_BOOSTER = "Minotaur-IV + HTV-2"        # used only as a boost stack
_CHGB = rv_from_dict(json.load(open("rv_library/C-HGB.rv.json")))
_CUTOFF = 170.0                         # early cutoff → sub-circular entry (~5.6 km/s)


def _fly(mode, zeta=None, aero="polar", cutoff=_CUTOFF):
    p = get_missile(_BOOSTER)
    rv = copy.deepcopy(_CHGB)
    rv.glider_guidance = mode
    rv.glider_aero_model = aero
    if zeta is not None:
        rv.glider_damping_zeta = zeta
    p.rv = rv
    r = integrate_trajectory(p, 0.0, 0.0, 90.0, max_time_s=6000.0,
                             dt_output=2.0, cutoff_time_s=cutoff)
    alt = np.asarray(r['alt']).ravel() / 1000.0
    rng = np.asarray(r['range']).ravel() / 1000.0
    t = np.asarray(r['t']).ravel()
    return alt, rng, t


def _frac_above_100(alt, t):
    """Fraction of the post-apogee descent spent above 100 km (no atmosphere)."""
    iap = int(np.argmax(alt))
    g = alt[iap:]
    return float(np.mean(g > 100.0))


def test_serialization_roundtrip():
    from missile_models import rv_to_dict
    rv = copy.deepcopy(_CHGB)
    rv.glider_guidance = "damped_glide"
    rv.glider_damping_zeta = 0.55
    rv2 = rv_from_dict(rv_to_dict(rv))
    assert rv2.glider_guidance == "damped_glide"
    assert abs(rv2.glider_damping_zeta - 0.55) < 1e-9
    print("  ok  serialization round-trip")


def test_nesting_exact():
    for aero in ("constant_LD", "polar"):
        a_s, _, _ = _fly("skip_glide", aero=aero)
        a_d, _, _ = _fly("damped_glide", zeta=0.0, aero=aero)
        n = min(len(a_s), len(a_d))
        d = float(np.max(np.abs(a_s[:n] - a_d[:n]))) if n else 9e9
        assert len(a_s) == len(a_d) and d < 1e-9, \
            f"{aero}: ζ=0 not identical to skip_glide (max|Δ|={d})"
        print(f"  ok  nesting {aero}: ζ=0 ≡ skip_glide (max|Δalt|={d:.1e} km)")


def test_damping_keeps_glide_in_atmosphere():
    # Undamped skip throws the vehicle out of the atmosphere; damping holds it
    # in the glide corridor.  Require a large reduction in time-above-100-km.
    a0, r0, t0 = _fly("damped_glide", zeta=0.0)
    a1, r1, t1 = _fly("damped_glide", zeta=0.7)
    f0, f1 = _frac_above_100(a0, t0), _frac_above_100(a1, t1)
    assert f1 < 0.5 * f0, \
        f"damping did not pull the glide into the atmosphere ({f0:.0%} → {f1:.0%})"
    # and a vehicle that actually glides carries farther than one that skips out
    assert r1[-1] > 1.5 * r0[-1], f"range did not improve ({r0[-1]:.0f}→{r1[-1]:.0f})"
    print(f"  ok  in-atmosphere: above-100km {f0:.0%} → {f1:.0%}; "
          f"range {r0[-1]:.0f} → {r1[-1]:.0f} km (ζ=0 → 0.7)")


def test_damping_monotone():
    # More damping never spends more time out of the atmosphere.
    fr = [_frac_above_100(*_fly("damped_glide", zeta=z)[::2]) for z in (0.0, 0.7, 2.0)]
    assert fr[0] >= fr[1] >= fr[2] - 1e-9, f"non-monotone above-100km fractions: {fr}"
    print(f"  ok  monotone: above-100km fraction ζ=[0,0.7,2.0] = "
          f"[{fr[0]:.0%}, {fr[1]:.0%}, {fr[2]:.0%}]")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} damped_glide checks passed.")


if __name__ == "__main__":
    main()
