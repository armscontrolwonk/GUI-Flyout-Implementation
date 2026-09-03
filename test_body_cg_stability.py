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
    p.body_reenters = True
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
    p.body_reenters = False         # separating: the booster says so
    p.diameter_m = 1.1
    p.length_m = 6.7
    ro = ROParams(name="RV", mass_kg=500.0, beta_kg_m2=3000.0, shape="cone",
                  diameter_m=0.6, length_m=1.8, separation_mode="separating_ro")
    p2 = compose_loadout(p, ro, 1)
    p2.ro = ro
    _x_cg, total = gfs.estimate_cg(p2)
    assert total > 6.7          # RV length is added on top for a separating RV


# ── reentry CG override (warhead-forward) ───────────────────────────────────

def _kn23_cg(reentry_cg_m):
    p = get_booster("Scud-B (R-17)")
    p.body_reenters = True
    p.diameter_m = 1.1
    p.length_m = 6.7
    ro = ROParams(name="KN23", mass_kg=500.0, beta_kg_m2=0.0, shape="karman",
                  diameter_m=1.1, length_m=2.0, separation_mode="body",
                  glider_enabled=True, glider_LD=0.0, glider_guidance="damped_glide",
                  body_nose_length_m=2.0, reentry_cg_m=reentry_cg_m)
    p2 = compose_loadout(p, ro, 1)
    p2.ro = ro
    return p2


def _fly_cg(reentry_cg_m):
    return integrate_trajectory(_kn23_cg(reentry_cg_m), 39.12, 125.67, 90.0,
                                burnout_angle_deg=-2.0, max_time_s=3600.0)


def test_reentry_cg_override_moves_the_static_margin():
    """A forward CG raises the static margin; an aft CG drops it below zero
    (CP ahead of CG) — the override is honoured by the trim gate."""
    g_fwd = trim_gate.trim_gate(_kn23_cg(2.0), mach=glider_ld.GLIDE_MACH_REF,
                                x_cg_m=2.0)
    g_aft = trim_gate.trim_gate(_kn23_cg(6.0), mach=glider_ld.GLIDE_MACH_REF,
                                x_cg_m=6.0)
    assert g_fwd["static_margin_cal"] > g_aft["static_margin_cal"]
    assert g_fwd["static_margin_cal"] > 0.0            # forward -> stable
    assert g_aft["static_margin_cal"] < 0.0            # aft -> unstable


def test_reentry_cg_forward_glides_aft_tumbles():
    """The override flows through the run: a warhead-forward body glides
    (nose-first), an aft CG tumbles (bluff, slow impact)."""
    r_fwd = _fly_cg(2.0)
    r_aft = _fly_cg(6.0)
    assert r_fwd["range_km"] > r_aft["range_km"] * 2.0
    assert r_fwd["impact_speed_ms"] > 2.0 * r_aft["impact_speed_ms"]


def test_reentry_cg_auto_is_the_airframe_centroid():
    """0 = auto places the CG at the uniform-airframe centre (half the
    airframe length)."""
    x_cg, total = gfs.estimate_cg(_kn23_cg(0.0))
    assert abs(x_cg - 0.5 * total) < 1e-6


# ── declared warhead (payload_kg) is auto-placed forward ────────────────────

def _body_with_warhead(struct_kg, warhead_kg, nose_len, length=9.18, diam=1.10):
    """A long-nosed heavy-warhead body (KN-23A class): the airframe structure is
    the body mass, the warhead is a DECLARED forward payload."""
    p = get_booster("Scud-B (R-17)")
    p.body_reenters = True
    p.diameter_m = diam
    p.length_m = length
    ro = ROParams(name="body", mass_kg=struct_kg, payload_kg=warhead_kg,
                  beta_kg_m2=0.0, shape="tangent_ogive", diameter_m=diam,
                  length_m=length, separation_mode="body", glider_enabled=True,
                  glider_LD=0.0, glider_guidance="damped_glide",
                  body_nose_length_m=nose_len)
    p2 = compose_loadout(p, ro, 1)
    p2.ro = ro
    return p2


def test_declared_warhead_pulls_cg_forward_of_tube_centre():
    """A body with a declared warhead (ro.payload_kg) packs it in the nose, so
    Thrusty's auto CG sits AHEAD of the uniform-tube centroid — no reentry_cg_m
    override needed.  Legacy bodies (no payload) keep the tube centre."""
    p = _body_with_warhead(struct_kg=988.0, warhead_kg=2500.0, nose_len=4.44)
    x_cg, total = gfs.estimate_cg(p)
    assert x_cg < 0.5 * total - 1e-3            # forward of the tube centre
    assert 0.28 < x_cg / total < 0.42          # nose-heavy, still inside the body
    # no declared warhead -> unchanged tube centroid
    p0 = _body_with_warhead(struct_kg=3488.0, warhead_kg=0.0, nose_len=4.44)
    x0, t0 = gfs.estimate_cg(p0)
    assert abs(x0 - 0.5 * t0) < 1e-6


def test_fuelled_cg_is_aft_of_reentry_cg():
    """estimate_cg(fuelled=True) adds the aft motor propellant, so the liftoff CG
    sits AFT of the empty re-entry CG (the schematic draws both, labelled).  The
    trim gate keeps using the re-entry CG (default), so stability is unchanged."""
    p = _body_with_warhead(struct_kg=988.0, warhead_kg=2500.0, nose_len=4.44)
    x_re, _ = gfs.estimate_cg(p, fuelled=False)
    x_fu, _ = gfs.estimate_cg(p, fuelled=True)
    assert x_fu > x_re + 0.5                       # propellant pulls the CG aft
    # a stack (separating RV) has no distinct re-entry CG — the flag is a no-op
    sep = get_booster("Scud-B (R-17)")
    sep.body_reenters = False                    # a separating stack here
    from booster_models import ROParams as _RO
    sep.diameter_m = 1.1
    sep_ro = _RO(name="RV", mass_kg=500.0, beta_kg_m2=3000.0, shape="cone",
                 diameter_m=0.6, length_m=1.8, separation_mode="separating_ro")
    sep2 = compose_loadout(sep, sep_ro, 1); sep2.ro = sep_ro
    assert gfs.estimate_cg(sep2, fuelled=True)[0] == \
           pytest.approx(gfs.estimate_cg(sep2, fuelled=False)[0], abs=1e-9)


def test_heavy_warhead_body_glides_on_auto_cg():
    """The long-nosed KN-23A tumbles on the bare tube centroid but, with the
    warhead declared, Thrusty's auto CG makes it stable and it glides at best
    glide — the reported behaviour with no hand-set CG."""
    p = _body_with_warhead(struct_kg=988.0, warhead_kg=2500.0, nose_len=4.44)
    g = trim_gate.trim_gate(p, mach=glider_ld.GLIDE_MACH_REF)   # auto CG
    assert g["static_margin_cal"] > 1.0
    assert g["LD_achievable"] > 2.5
