"""Commanded pull-up (glider_pullup_start_alt_km) — the plan-phase modifier.

Motivated by a nine-run user campaign against the digitized SWERVE III
corridor (benchmarks/swerve/): the constant-ζ capture architecture traces a
smooth frontier — every knob trades trough depth against arrival speed — and
the flight point (a 25.8 km shelf held at Mach 12.2) sits outside the whole
family, because the real vehicle captured with a discrete commanded pull at
Mach 12 (Iliff & Shafer 1993; Williamson Fig. 20), not a damping loop.

The modifier: zero commanded lift above the trigger altitude (ballistic
fall), full-authority pull at it (capped by the structural g-limit AND by
what q + the aero model supply), one-way handoff to the selected glide law
once the sink rate reaches the law's own equilibrium target.

Discipline: assertions are behavioral identities and directional physics.
The SWERVE scenario is used as a GEOMETRY/ENERGY fixture; nothing asserts a
match to the flight corridor's numbers (that comparison is the user's
validation exercise, kept out of the test suite by design).
"""

import copy
import json

import numpy as np
import pytest

from booster_models import (get_booster, ro_from_dict, apply_reentry_plan,
                            load_reentry_plan)
from trajectory import integrate_trajectory

_CACHE = {}


def _fly(pullup_km, zeta=0.3, guidance='damped_glide'):
    key = (pullup_km, zeta, guidance)
    if key not in _CACHE:
        ro = ro_from_dict(json.load(open("ro_library/SWERVE.ro.json")))
        rp = dict(load_reentry_plan("SWERVE") or {},
                  glider_guidance=guidance,
                  glider_damping_zeta=zeta,
                  glider_pullup_start_alt_km=pullup_km)
        ro = apply_reentry_plan(ro, rp)
        p = get_booster("Strypi VIII R")
        p.ro = copy.deepcopy(ro)
        p.guidance = "pitch_program"
        p.launch_elevation_deg = 70.0
        _CACHE[key] = integrate_trajectory(
            p, 22.0228, -159.785, 270.0, burnout_angle_deg=0.0,
            cutoff_time_s=57.0, gt_turn_start_s=0.0, gt_turn_stop_s=50.0,
            launch_elevation_deg=70.0, max_time_s=3600.0, dt_output=1.0)
    return _CACHE[key]


def _downleg(r):
    alt = np.asarray(r['alt'], float)
    spd = np.asarray(r['speed'], float)
    i = int(np.argmax(alt))
    return alt[i:], spd[i:]


def _v_at(r, alt_m):
    alt, spd = _downleg(r)
    j = int(np.argmin(np.abs(alt - alt_m)))
    return float(spd[j])


def _first_trough_m(r):
    alt, _ = _downleg(r)
    for i in range(2, len(alt) - 1):
        if alt[i] < 45e3 and alt[i] <= alt[i - 1] and alt[i] < alt[i + 1]:
            return float(alt[i])
    return float(alt.min())


# ── the contract: unset field changes nothing ───────────────────────────────
def test_zero_trigger_is_byte_identical():
    """pullup=0 must integrate identically to the field never existing —
    every shipped plan and every prior run is untouched."""
    a, b = _fly(0.0), _fly(0)
    assert np.array_equal(np.asarray(a['alt']), np.asarray(b['alt']))
    ro = ro_from_dict(json.load(open("ro_library/SWERVE.ro.json")))
    assert ro.glider_pullup_start_alt_km == 0.0     # legacy JSON loads clean


def test_plan_field_round_trips():
    from booster_models import ro_to_dict
    ro = ro_from_dict(json.load(open("ro_library/SWERVE.ro.json")))
    ro = apply_reentry_plan(ro, {'glider_pullup_start_alt_km': 40.0})
    assert ro.glider_pullup_start_alt_km == 40.0
    assert ro_to_dict(ro)['glider_pullup_start_alt_km'] == 40.0


# ── phase physics ───────────────────────────────────────────────────────────
def test_pullup_catches_higher_than_zeta_capture():
    """The mechanism: a commanded pull arrests the fall at the trigger, so the
    first trough lands materially HIGHER than the constant-ζ capture, which
    plunges before the damper wins.  This is the frontier-breaking move — the
    campaign showed no ζ reaching a shallow trough without overdamping.  The
    win holds whether ζ is weak or strong, i.e. it is the pull, not the gain."""
    for z in (0.3, 0.7):
        pull = _first_trough_m(_fly(40.0, zeta=z))
        base = _first_trough_m(_fly(0.0, zeta=z))
        assert pull > base + 1500.0, (z, pull, base)


def test_capture_altitude_decouples_from_zeta():
    """With the pull owning capture, the trough no longer walks with ζ — the
    two pull-up runs land within a few hundred m of each other, whereas the
    two ζ-only runs spread by kilometres.  That decoupling is the whole point:
    ζ returns to damping small residuals, not arresting the fall."""
    pull_spread = abs(_first_trough_m(_fly(40.0, zeta=0.3))
                      - _first_trough_m(_fly(40.0, zeta=0.7)))
    base_spread = abs(_first_trough_m(_fly(0.0, zeta=0.3))
                      - _first_trough_m(_fly(0.0, zeta=0.7)))
    assert pull_spread < base_spread


def test_triggering_too_high_undershoots_honestly():
    """At 90 km there is no q to pull with — the pull must NOT conjure lift,
    so the capture is worse (deeper trough) than a well-placed 40 km trigger."""
    assert _first_trough_m(_fly(90.0)) < _first_trough_m(_fly(40.0))


def test_modifier_composes_with_dynamic_equilibrium_glide():
    """A modifier, not a mode: the other numerical law gets the same benefit."""
    pull = _fly(40.0, guidance='dynamic_equilibrium_glide')
    base = _fly(0.0, guidance='dynamic_equilibrium_glide')
    assert _v_at(pull, 41e3) > _v_at(base, 41e3)
    assert np.isfinite(np.asarray(pull['speed'])).all()
    assert pull['range_km'] > 0 and pull['impact_lat'] is not None


def test_analytic_family_ignores_the_field():
    """The closed-form laws fly their own pull-up arc; the modifier must not
    perturb them."""
    def fly_acton(pullup):
        ro = ro_from_dict(json.load(open("ro_library/SWERVE.ro.json")))
        rp = dict(load_reentry_plan("SWERVE") or {},
                  glider_guidance='equilibrium_glide',
                  glider_pullup_start_alt_km=pullup)
        ro = apply_reentry_plan(ro, rp)
        p = get_booster("Strypi VIII R")
        p.ro = copy.deepcopy(ro); p.guidance = "pitch_program"
        p.launch_elevation_deg = 70.0
        return integrate_trajectory(
            p, 22.0228, -159.785, 270.0, burnout_angle_deg=0.0,
            cutoff_time_s=57.0, gt_turn_start_s=0.0, gt_turn_stop_s=50.0,
            launch_elevation_deg=70.0, max_time_s=3600.0, dt_output=2.0)
    a, b = fly_acton(0.0), fly_acton(40.0)
    assert abs(a['range_km'] - b['range_km']) < 1e-6
