"""A ballistic reentry generates no lift — enforced in the physics.

The reentry EOM's lift block has cases for the glide laws (damped_glide,
dynamic_equilibrium_glide, skip/equilibrium via the polar catch-all) but none
for ``glider_guidance == 'ballistic'``.  It used to stay quiet only because the
GUI turned ``glider_enabled`` off for a ballistic plan.  But the body setup
derives ``glider_LD > 0`` for ANY glider_enabled body, and the polar catch-all
then flew even a ballistic-guidance body at its max-L/D angle of attack — a
hidden skip-glide that added ~65 % range.  These tests pin that ballistic
guidance produces the drag·gravity·rotation trajectory regardless of the
``glider_enabled`` / ``glider_LD`` bookkeeping.
"""

import copy

import pytest

from booster_models import (get_booster, load_booster_library, ROParams,
                            compose_loadout)
from trajectory import integrate_trajectory

load_booster_library()


def _body(guidance, glider_enabled, aero='polar'):
    base = copy.deepcopy(get_booster("Scud-B (R-17)"))
    base.body_reenters = True
    _last = base
    while getattr(_last, 'stage2', None) is not None:
        _last = _last.stage2
    ro = ROParams(name="fe", mass_kg=max(float(_last.mass_final), 1.0),
                  beta_kg_m2=0.0, shape="cone",
                  diameter_m=float(_last.diameter_m),
                  length_m=float(_last.length_m), separation_mode='body',
                  glider_enabled=glider_enabled, glider_LD=0.0,
                  glider_guidance=guidance, glider_aero_model=aero)
    p = compose_loadout(base, ro, 1)
    p.ro = ro
    return p


def _range(p):
    return integrate_trajectory(p, 39.0, 125.0, 90.0, burnout_angle_deg=42.4,
                                max_time_s=3600.0)["range_km"]


@pytest.mark.parametrize("aero", ["polar", "constant_LD"])
def test_ballistic_guidance_never_lifts(aero):
    """A ballistic body flies the same range whether the glider flag is on or
    off — the physics refuses lift on ballistic guidance, so a plan that left
    glider_enabled=True cannot leak glide range."""
    off = _range(_body("ballistic", glider_enabled=False, aero=aero))
    on = _range(_body("ballistic", glider_enabled=True, aero=aero))
    assert on == pytest.approx(off, rel=0.02), (
        f"ballistic with glider_enabled=True ({on:.1f}) leaked lift vs "
        f"true ballistic ({off:.1f})")


def test_glide_still_lifts_much_further():
    """The guard is specific to ballistic — a damped-phugoid glider on the same
    body still lifts and ranges far beyond the ballistic case."""
    ballistic = _range(_body("ballistic", glider_enabled=True))
    glide = _range(_body("damped_glide", glider_enabled=True))
    assert glide > ballistic * 1.3
