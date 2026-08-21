"""Non-separating body CG / stability: the nose is subtractive, not stacked.

A body-mode vehicle (V-2 / Scud / KN-23) is one airframe; its nose is carved
from the last stage's length (FRONT_END_DESIGN.md), not added on top.  The CG /
static-margin machinery (grid_fin_sizing.estimate_cg, _stack_layout, feeding
trim_gate) used to take the nose length from effective_ro.length_m — which in
body mode is the *inherited stage length* — and ADD it to the stage length,
doubling the airframe (6.7 m body → 13.4 m) and floating the CG out past the
tail.  trim_gate then read CP ahead of CG and declared a perfectly good body
"unstable → tumbles → no glide", so a KN-23 flown with L/D left at the
auto-derive sentinel (0) never pulled up.

These tests pin the corrected geometry: the estimated airframe length equals the
stage length, the CG lands inside the body, and the auto-derived body glides.
"""

import numpy as np
import pytest

from booster_models import (get_booster, load_booster_library, ROParams,
                            compose_loadout)
import grid_fin_sizing as gfs
import trim_gate
import glider_ld
from trajectory import integrate_trajectory

load_booster_library()


def _kn23(glider_LD=0.0, glider_enabled=True):
    p = get_booster("Scud-B (R-17)")
    p.diameter_m = 1.1
    p.length_m = 6.7
    ro = ROParams(name="KN23", mass_kg=500.0, beta_kg_m2=3000.0, shape="karman",
                  diameter_m=1.1, length_m=2.0, separation_mode="body",
                  glider_enabled=glider_enabled, glider_LD=glider_LD,
                  glider_guidance="damped_glide", body_nose_length_m=2.0)
    p2 = compose_loadout(p, ro, 1)
    p2.ro = ro
    return p2


def test_estimated_length_is_the_airframe_not_double():
    """estimate_cg's total length is the airframe (6.7 m), not airframe+nose."""
    _x_cg, total = gfs.estimate_cg(_kn23())
    assert total == pytest.approx(6.7, abs=1e-6)


def test_cg_lands_inside_the_body():
    """The CG must sit within the airframe, not out past the tail."""
    x_cg, total = gfs.estimate_cg(_kn23())
    assert 0.0 < x_cg < total


def test_stack_layout_length_is_the_airframe():
    """_stack_layout agrees: the body isn't doubled."""
    _nd, _xcp, _sections, L_total = gfs._stack_layout(_kn23())
    assert L_total == pytest.approx(6.7, abs=1e-6)


def test_body_is_not_falsely_unstable_at_reference_mach():
    """At the glide reference Mach the KN-23-class body trims and glides — it is
    not spuriously flagged unstable by a CG floated out past the tail."""
    g = trim_gate.trim_gate(_kn23(), mach=glider_ld.GLIDE_MACH_REF)
    assert not g.get("error")
    assert g["static_margin_cal"] > 0.0
    assert g["LD_achievable"] > 1.0


def test_auto_derived_body_pulls_up():
    """L/D left at the sentinel 0 → the body derives its L/D from geometry and
    glides materially past the ballistic baseline (the reported bug)."""
    r_derive = integrate_trajectory(_kn23(glider_LD=0.0), 39.12, 125.67, 90.0,
                                    burnout_angle_deg=-2.0, max_time_s=3600.0)
    r_ball = integrate_trajectory(_kn23(glider_enabled=False), 39.12, 125.67,
                                  90.0, burnout_angle_deg=-2.0, max_time_s=3600.0)
    assert r_derive["reentry_trim"] is not None
    assert r_derive["reentry_trim"]["LD_achievable"] > 1.0
    assert r_derive["range_km"] > r_ball["range_km"] * 1.3


def test_separating_rv_length_is_unchanged():
    """A separating RV still caps the stack additively (its own length on top) —
    the fix is scoped to body mode only."""
    p = get_booster("Scud-B (R-17)")
    p.diameter_m = 1.1
    p.length_m = 6.7
    ro = ROParams(name="RV", mass_kg=500.0, beta_kg_m2=3000.0, shape="cone",
                  diameter_m=0.6, length_m=1.8, separation_mode="separating_ro")
    p2 = compose_loadout(p, ro, 1)
    p2.ro = ro
    _x_cg, total = gfs.estimate_cg(p2)
    assert total > 6.7          # RV length is added on top for a separating RV
