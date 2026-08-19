"""Boost angle of attack and the q·α combined load (NASA SP-8099).

SP-8099 "Combining Ascent Loads" (1972) is the criteria monograph for how a
maneuvering booster's aerodynamic and steering loads combine during ascent:

  * §2.1.2.2 (p. 12): the standard preliminary-design load condition is a
    5°–10° angle of attack at the maximum-dynamic-pressure condition, bounded
    by the "hard-over engine" case.
  * p. 13: a sharp range-safety "dog-leg" maneuver produced a large angle of
    attack whose steering-plus-aero load became the DESIGN combined-load
    condition — the α a dogleg demands is the physically limiting quantity.

Before this feature, Thrusty's yaw program flew any commanded azimuth slew
instantly and for free: thrust was defined as the flight direction, so a
55→56 s 90° dogleg reported α = 0 and max-q never moved.  These tests pin:

  1. α(t), q(t), q·α(t) are reported for every run, with a "Max q·α"
     timeline milestone (the SP-8099 design metric made visible).
  2. An un-limited dogleg that demands α beyond the envelope is flagged in
     the timeline — flown as commanded, but never silently.
  3. With alpha_limit_deg set, the flown α rails at the limit while q is
     significant, the engagement is reported, and the maneuver stretches
     over the time it physically needs (the trajectory actually changes).
  4. alpha_limit_deg=None leaves the dynamics untouched (legacy behavior).
"""

import numpy as np

from booster_models import BOOSTER_DB, get_booster, load_booster_library
from trajectory import integrate_trajectory, _ALPHA_GATE_Q_PA

load_booster_library()

LAT, LON, AZ = 40.0, 128.0, 90.0
# A 1-second 90° dogleg commanded through the high-q regime — the impossible
# instantaneous turn this feature exists to catch.
DOGLEG = [(55.0, 56.0, 0.0)]

_CACHE = {}


def _fly(**kw):
    key = repr(sorted(kw.items()))
    if key not in _CACHE:
        p = get_booster("No-dong")
        _CACHE[key] = integrate_trajectory(p, LAT, LON, AZ,
                                           max_time_s=3600.0, **kw)
    return _CACHE[key]


# ── 1. α / q·α reporting on every run ───────────────────────────────────────

def test_alpha_arrays_present_and_shaped():
    r = _fly()
    t = np.asarray(r['t'])
    for k in ('alpha_deg', 'alpha_cmd_deg', 'q_pa', 'q_alpha_kpa_deg'):
        assert len(np.asarray(r[k])) == len(t), k


def test_alpha_finite_during_boost_nan_after():
    r = _fly()
    t = np.asarray(r['t'])
    a = np.asarray(r['alpha_deg'])
    # Finite through the heart of the burn (No-dong burns ~70 s) …
    mid = (t > 20.0) & (t < 60.0)
    assert np.all(np.isfinite(a[mid]))
    # … NaN in ballistic flight (well after burnout).
    assert np.all(~np.isfinite(a[t > 200.0]))


def test_straight_ascent_alpha_within_envelope_at_max_q():
    """A plain pitch program flies near-zero-to-small α through max-q —
    inside SP-8099's 5–10° preliminary-design envelope, so no warning."""
    r = _fly()
    q = np.asarray(r['q_pa'])
    a = np.asarray(r['alpha_deg'])
    # Boost max-q: restrict to powered flight (α finite) — the global q peak
    # is on the reentry leg, where boost α is undefined.
    boost = np.isfinite(a)
    i_maxq = int(np.flatnonzero(boost)[np.argmax(q[boost])])
    assert a[i_maxq] < 10.0
    assert not any('SP-8099 envelope' in m['event'] for m in r['milestones'])


def test_max_q_alpha_milestone_present():
    r = _fly()
    ms = [m for m in r['milestones'] if m['event'].startswith('Max q·α')]
    assert len(ms) == 1
    # Milestone sits at the argmax of the reported q·α trace.
    qa = np.asarray(r['q_alpha_kpa_deg'])
    qa = np.where(np.isfinite(qa), qa, -1.0)
    t_pk = float(np.asarray(r['t'])[int(np.argmax(qa))])
    assert abs(ms[0]['t_s'] - t_pk) < 2.0


# ── 2. un-limited dogleg: flown as commanded, loudly flagged ────────────────

def test_dogleg_spikes_alpha_and_flags_envelope():
    r = _fly(yaw_maneuvers=DOGLEG)
    t = np.asarray(r['t'])
    a = np.asarray(r['alpha_deg'])
    q = np.asarray(r['q_pa'])
    dl = (t >= 55.0) & (t <= 70.0)
    # The turn demands a large α at significant q — the SP-8099 p. 13 case.
    assert np.nanmax(a[dl]) > 25.0
    assert q[dl].max() > 1000.0
    warn = [m for m in r['milestones'] if 'SP-8099 envelope' in m['event']]
    assert len(warn) == 1
    assert r['alpha_limit_engaged'] is None      # no limit was set


# ── 3. α-limited dogleg: railed, reported, stretched ────────────────────────

def test_alpha_limit_clamps_flown_alpha():
    r = _fly(yaw_maneuvers=DOGLEG, alpha_limit_deg=10.0)
    a = np.asarray(r['alpha_deg'])
    q = np.asarray(r['q_pa'])
    gated = np.isfinite(a) & (q > _ALPHA_GATE_Q_PA)
    assert np.max(a[gated]) <= 10.0 + 0.2
    eng = r['alpha_limit_engaged']
    assert eng is not None
    assert eng['alpha_limit_deg'] == 10.0
    assert eng['alpha_cmd_max_deg'] > 25.0       # the raw demand was outsize
    assert 55.0 <= eng['t_start_s'] <= 60.0
    assert any(m['event'].startswith('α-limit engaged')
               for m in r['milestones'])


def test_alpha_limit_stretches_the_maneuver():
    """Clamped, the vehicle turns only as fast as bounded lateral force
    rotates the velocity vector — so the trajectory genuinely differs from
    the instantaneous free turn."""
    r_free = _fly(yaw_maneuvers=DOGLEG)
    r_lim = _fly(yaw_maneuvers=DOGLEG, alpha_limit_deg=10.0)
    assert abs(r_free['range_km'] - r_lim['range_km']) > 10.0
    # Commanded azimuth completes instantly; the FLOWN azimuth (derived from
    # the clamped thrust vector) is still far from the 0° target at yaw end.
    t = np.asarray(r_lim['t'])
    az = np.asarray(r_lim['az_cmd_deg'])
    i58 = int(np.searchsorted(t, 58.0))
    err = abs(((az[i58] - 0.0 + 180.0) % 360.0) - 180.0)
    assert err > 20.0


# ── 4. default off = legacy dynamics ────────────────────────────────────────

def test_no_limit_is_byte_identical_to_legacy_call():
    """alpha_limit_deg=None must not perturb the integration: the clamp is
    never evaluated, so the trajectory equals the same call without the
    argument."""
    r_a = _fly()
    p = get_booster("No-dong")
    r_b = integrate_trajectory(p, LAT, LON, AZ, max_time_s=3600.0,
                               alpha_limit_deg=None)
    assert r_a['range_km'] == r_b['range_km']
    assert np.array_equal(np.asarray(r_a['alt']), np.asarray(r_b['alt']))
