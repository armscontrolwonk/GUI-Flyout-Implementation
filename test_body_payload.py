"""A non-separating body's payload lives on the front end (A2 ownership).

A body (V-2 / Scud / KN-23) IS the booster's last stage, so its structural +
residual burnout mass is inherited from the stage.  What the front end OWNS is
an explicit ADDED payload — warhead / bus / guidance — entered on the reentry
object as ``payload_kg``.  compose_loadout adds it to the boosted stack and
keeps it fused through burnout, so the reentry mass is airframe_burnout +
payload.

The reentry object's own ``mass_kg`` is NEVER added for a body (that was the
574 → 137 km double-count); only ``payload_kg`` is.  Default 0 → every existing
file flies byte-identical.  These tests pin that contract, including
idempotency (composing twice adds the payload once).
"""

import pytest

from booster_models import (get_booster, load_booster_library, ROParams, BoosterParams,
                            effective_ro,
                            compose_loadout, effective_ro, ro_to_dict,
                            ro_from_dict, booster_to_dict, booster_from_dict)
from trajectory import integrate_trajectory

load_booster_library()


def _kn23(payload=0.0, mass_kg=2198.0, glider=False):
    p = get_booster("Scud-B (R-17)")
    p.body_reenters = True
    p.diameter_m = 1.1
    p.length_m = 6.7
    ro = ROParams(name="KN23", mass_kg=mass_kg, beta_kg_m2=3000.0, shape="karman",
                  diameter_m=1.1, length_m=6.7, separation_mode="body",
                  glider_enabled=glider, body_nose_length_m=2.0,
                  payload_kg=payload)
    return p, ro


def _compose(payload=0.0, mass_kg=2198.0):
    p, ro = _kn23(payload, mass_kg)
    c = compose_loadout(p, ro, 1)
    c.ro = ro
    return c


def _fly(p):
    return integrate_trajectory(p, 39.12, 125.67, 90.0, burnout_angle_deg=-2.0,
                                max_time_s=3600.0)


def test_zero_payload_is_byte_identical():
    """payload 0 (every existing file) adds nothing to the stack."""
    p, ro = _kn23(0.0)
    base = compose_loadout(p, ro, 1).mass_initial
    built = get_booster("Scud-B (R-17)")
    built.diameter_m = 1.1
    built.length_m = 6.7
    assert base == pytest.approx(built.mass_initial, abs=1e-9)


def test_mass_kg_is_never_added_for_a_body():
    """A huge RO mass_kg with payload 0 still changes nothing — mass_kg is the
    inherited airframe mass, not an addend (the double-count guard)."""
    unchanged = _compose(payload=0.0, mass_kg=99999.0).mass_initial
    plain = _compose(payload=0.0, mass_kg=500.0).mass_initial
    assert unchanged == pytest.approx(plain, abs=1e-9)


def test_payload_raises_both_boost_and_reentry_mass():
    """The payload rides the boosted stack AND stays fused for reentry."""
    base = _compose(0.0)
    heavy = _compose(500.0)
    assert heavy.mass_initial - base.mass_initial == pytest.approx(500.0, abs=1e-6)
    assert (effective_ro(heavy).mass_kg - effective_ro(base).mass_kg
            == pytest.approx(500.0, abs=1e-6))


def test_compose_twice_is_idempotent():
    """Composing an already-composed body adds the payload once, not twice."""
    p, ro = _kn23(500.0)
    once = compose_loadout(p, ro, 1)
    twice = compose_loadout(once, ro, 1)
    assert twice.mass_initial == pytest.approx(once.mass_initial, abs=1e-9)


def test_heavier_payload_flies_shorter():
    """The added payload propagates to the flyout: a heavier body lands short."""
    r0 = _fly(_compose(0.0))
    r5 = _fly(_compose(500.0))
    assert r5["range_km"] < r0["range_km"]


def test_payload_round_trips():
    _p, ro = _kn23(750.0)
    assert ro_from_dict(ro_to_dict(ro)).payload_kg == pytest.approx(750.0)


def test_separating_rv_ignores_payload_kg():
    """payload_kg is a body concept — a separating RV still composes on its
    mass_kg, unaffected by a stray payload_kg."""
    p = get_booster("Scud-B (R-17)")
    p.body_reenters = False              # a separating stack for this test
    ro = ROParams(name="RV", mass_kg=500.0, beta_kg_m2=8000.0, shape="cone",
                  diameter_m=0.6, length_m=1.8, separation_mode="separating_ro",
                  payload_kg=999.0)
    with_field = compose_loadout(p, ro, 1).mass_initial
    ro2 = ROParams(name="RV", mass_kg=500.0, beta_kg_m2=8000.0, shape="cone",
                   diameter_m=0.6, length_m=1.8, separation_mode="separating_ro",
                   payload_kg=0.0)
    without = compose_loadout(p, ro2, 1).mass_initial
    assert with_field == pytest.approx(without, abs=1e-9)


# ── booster body_reenters flag (the non-separating master switch) ───────────

def test_body_reenters_defaults_false_and_round_trips():
    """The booster's non-separating flag defaults off on the dataclass and on
    a separating shipped booster, and survives a to_dict/from_dict round-trip.
    (Scud-B ships body_reenters=True since the payload migration: the R-17
    does not separate its warhead.)"""
    assert BoosterParams(name="x", mass_initial=2.0, mass_propellant=1.0,
                         mass_final=1.0, diameter_m=1.0, length_m=1.0,
                         thrust_N=1.0, burn_time_s=1.0, isp_s=200.0).body_reenters is False
    assert get_booster("No-dong").body_reenters is False
    assert get_booster("Scud-B (R-17)").body_reenters is True
    p = get_booster("No-dong")
    p.body_reenters = True
    assert booster_from_dict(booster_to_dict(p)).body_reenters is True


def test_body_reenters_is_the_separation_switch():
    """The booster's body_reenters flag is the single source of the
    booster<->object link: it decides the run, and the object's own
    separation_mode is derived from it.  A booster marked body_reenters
    inherits the last stage's mass into the object even when the object was
    built as 'separating'; a booster NOT so marked separates even when the
    object was built as 'body'."""
    p0 = get_booster("Scud-B (R-17)")
    p0.body_reenters = False
    p0.diameter_m = 1.1
    p0.length_m = 6.7
    p1 = get_booster("Scud-B (R-17)")
    p1.diameter_m = 1.1
    p1.length_m = 6.7
    p1.body_reenters = True
    ro_body = ROParams(name="KN23", mass_kg=2198.0, beta_kg_m2=3000.0, shape="karman",
                       diameter_m=1.1, length_m=6.7, separation_mode="body",
                       body_nose_length_m=2.0)
    ro_sep = ROParams(name="KN23", mass_kg=2198.0, beta_kg_m2=3000.0, shape="karman",
                      diameter_m=1.1, length_m=6.7, separation_mode="separating_ro",
                      body_nose_length_m=2.0)
    # Derivation follows the booster, whatever the object says.
    assert effective_ro(_bind(p1, ro_sep)).separation_mode == "body"
    assert effective_ro(_bind(p0, ro_body)).separation_mode == "separating_ro"
    # Body-reentering booster: object mass is the stage's burnout mass.
    e1 = effective_ro(_bind(p1, ro_sep))
    assert e1.mass_kg == pytest.approx(p1.mass_initial - p1.mass_propellant)
    # Not marked: the object keeps its own mass (it separates).
    e0 = effective_ro(_bind(p0, ro_body))
    assert e0.mass_kg == pytest.approx(2198.0)
    # And the run differs: same object, the flag alone changes the trajectory.
    r0 = integrate_trajectory(_bind(p0, ro_body), 39.12, 125.67, 90.0,
                              burnout_angle_deg=-2.0, max_time_s=3600.0)
    r1 = integrate_trajectory(_bind(p1, ro_body), 39.12, 125.67, 90.0,
                              burnout_angle_deg=-2.0, max_time_s=3600.0)
    assert r0["range_km"] != pytest.approx(r1["range_km"], rel=1e-6)


def _bind(p, ro):
    c = compose_loadout(p, ro, 1)
    c.ro = ro
    return c
