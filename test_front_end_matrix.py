"""The derived-vs-entered matrix guard (FRONT_END_DESIGN.md Part II §9, P2-C).

The standing guarantee that a non-separating body's emergent quantities stay
emergent: mass / diameter / length are INHERITED from the last stage (the RO's
own values are ignored), and β / L/D are DERIVED from the whole airframe when
left at 0.  For a separating RV the same fields are the RV's OWN designed
inputs and ARE honored.  If a future change lets a typed body-RO field leak
into the flown result, one of these tests fails.
"""

import numpy as np

from booster_models import (get_booster, load_booster_library, ROParams,
                            compose_loadout, effective_ro)
from trajectory import integrate_trajectory

load_booster_library()


def _booster():
    p = get_booster("Scud-B (R-17)")
    p.diameter_m = 1.1
    p.length_m = 6.7
    return p


def _fly(ro):
    p = _booster()
    pc = compose_loadout(p, ro, 1)
    pc.ro = ro
    return integrate_trajectory(pc, 39.12, 125.67, 90.0,
                                burnout_angle_deg=-2.0, max_time_s=3600.0)


def _body(**over):
    base = dict(name="b", mass_kg=500.0, beta_kg_m2=0.0, shape="karman",
                diameter_m=1.1, length_m=2.0, separation_mode="body",
                glider_enabled=True, glider_LD=0.0, glider_guidance="damped_glide",
                body_nose_length_m=2.0)
    base.update(over)
    return ROParams(**base)


# ── inherited: RO mass / diameter / length are ignored for a body ───────────

def test_body_ignores_ro_length():
    """The airframe length is the stage's; the RO's own length_m must not
    change the flown trajectory."""
    r_a = _fly(_body(length_m=2.0))
    r_b = _fly(_body(length_m=99.0))
    assert r_a["range_km"] == r_b["range_km"]


def test_body_inherits_stage_geometry():
    """effective_ro reports the stage's diameter/length, not the RO's."""
    p = _booster()
    ro = _body(diameter_m=0.3, length_m=2.0)
    pc = compose_loadout(p, ro, 1); pc.ro = ro
    eff = effective_ro(pc)
    assert eff.diameter_m == p.diameter_m == 1.1        # stage, not the RO's 0.3
    assert eff.length_m == p.length_m == 6.7            # stage, not the RO's 2.0


# ── derived: β and L/D emerge from geometry when 0 ──────────────────────────

def test_body_beta_and_ld_are_derived_when_zero():
    r = _fly(_body(beta_kg_m2=0.0, glider_LD=0.0))
    assert r["derived_beta_kg_m2"] is not None and r["derived_beta_kg_m2"] > 0.0
    # A derived-L/D body glides materially farther than the same shot ballistic.
    r_ballistic = _fly(_body(glider_enabled=False))
    assert r["range_km"] > r_ballistic["range_km"] * 1.2


# ── separating RV: the same fields ARE its own honored inputs ────────────────

def test_separating_rv_honors_its_own_length_and_beta():
    def rv(length_m, beta):
        return ROParams(name="rv", mass_kg=500.0, beta_kg_m2=beta, shape="cone",
                        diameter_m=0.5, length_m=length_m,
                        separation_mode="separating_ro")
    # β is a real input: a very different β gives a very different flight.
    r_lo = _fly(rv(1.5, 2000.0))
    r_hi = _fly(rv(1.5, 20000.0))
    assert abs(r_lo["range_km"] - r_hi["range_km"]) > 20.0
    # and it never derives β.
    assert _fly(rv(1.5, 8000.0))["derived_beta_kg_m2"] is None
