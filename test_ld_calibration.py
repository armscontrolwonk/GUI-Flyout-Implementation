"""Whole-body L/D calibration anchors (glider_ld.whole_booster_LD).

The geometry L/D build-up (Jorgensen + Allen-Perkins cross-flow + NKP carryover)
over-predicts the lift-to-drag ratio of a slender BODY in the hypersonic regime,
and — worse — lets L/D_max *rise* with Mach (M3 -> M8) instead of plateauing.
That inflates non-separating body glide range.

The over-prediction is in the BODY term, not the lifting-surface term.  Two
free-flight anchors bracket it and localize it:

  * Low end — Fournier & Dupuis, AIAA 96-3399 (2016 reprint) and the CAN-4 data
    of Dupuis & Edwards, DREV TM-9525: a cone-cylinder-flare projectile
    (L/d = 5.84) measures L/D_max ~= 1.25 at M ~= 5 (Yates & Chapman AIAA
    96-3360 quote its coefficients: C_A = 0.646 - 0.11 dM + 2.26 sin^2 a,
    C_N = 7.0 sin a).

  * High end — Seiff & Wilkins, NASA TN D-341 (1961): a slender ogive-cylinder
    carrying three large highly-swept wings (chord = body length) — a purpose-
    built hypersonic glider — reaches L/D_max ~= 4-6.7 at M3-6.  Crucially they
    found the *linear* (wing) lift-curve slope accurate to ~10%, but the
    *nonlinear* body lift (a Newtonian 2*alpha^2 term) OVER-predicted: measured
    C_L = 0.064 vs an estimated 0.08 near best-glide alpha (~24% high).

  * Blunt floor — Intrieri, NASA TM X-569 (1961): a Mercury-type capsule at
    M5.5 tops out at L/D ~= 0.38.  (Not constructible in this slender-body
    model; quoted for scale only.)

So a plain slender body with no wings (our missile-body case, L/d ~13) should
sit nearer the CAN-4 end (~1.5-2.5), well below the winged glider — and its
L/D must not climb into the hypersonic regime.

These tests pin that.  Two are the CALIBRATION TARGETS, marked strict-xfail:
they fail against today's build-up and will XPASS (turning the suite red) once
whole_booster_LD is recalibrated, which is the signal to drop the markers and
lock the win in.  The other two are GUARDS that must stay green THROUGH the
calibration: the winged-glider anchor (wing lift is already right — don't break
it) and the lifting-surface ordering.
"""

import copy

import pytest

from booster_models import (get_booster, load_booster_library, ROParams,
                            compose_loadout)
import glider_ld as gld

load_booster_library()


def _body(l_over_d=13.4, fins=False, big_wings=False):
    """A non-separating body off a real stage, with the last stage's fineness
    and lifting surfaces overridden to a named reference shape."""
    base = copy.deepcopy(get_booster("Scud-B (R-17)"))
    base.body_reenters = True
    last = base
    while getattr(last, 'stage2', None) is not None:
        last = last.stage2
    last.length_m = l_over_d * last.diameter_m
    last.has_fins = fins or big_wings
    if fins:
        last.n_fins = 4; last.fin_span_m = 0.4
        last.fin_root_chord_m = 1.0; last.fin_tip_chord_m = 0.3
        last.fin_thickness_m = 0.03; last.fin_sweep_deg = 45.0
    if big_wings:            # Seiff-Wilkins-like: chord ~ body length, big span
        last.n_fins = 3; last.fin_span_m = 1.2
        last.fin_root_chord_m = last.length_m * 0.9; last.fin_tip_chord_m = 0.2
        last.fin_thickness_m = 0.03; last.fin_sweep_deg = 70.0
    ro = ROParams(name="fe", mass_kg=max(float(last.mass_final), 1.0),
                  beta_kg_m2=0.0, shape="cone",
                  diameter_m=float(last.diameter_m),
                  length_m=float(last.length_m), separation_mode='body',
                  glider_enabled=True, glider_LD=0.0)
    p = compose_loadout(base, ro, 1)
    p.ro = ro
    return p


def _ld(p, mach):
    return gld.whole_booster_LD(p, mach=mach)['ld_max']


# ── Calibration targets (strict-xfail until whole_booster_LD is recalibrated) ─

@pytest.mark.xfail(strict=True, reason="L/D calibration pending: hypersonic L/D "
                   "must plateau, not rise (Seiff-Wilkins TN D-341; NACA 1328). "
                   "Remove marker when whole_booster_LD is recalibrated.")
def test_slender_body_ld_does_not_rise_into_hypersonic():
    """A slender body's L/D_max should peak in the low-supersonic range and
    plateau or fall by the hypersonic regime — never end HIGHER at M8 than at
    M3.  Today it climbs (~2.45 -> ~3.07) because Cd0 keeps dropping while the
    linear body-lift slope is held Mach-flat."""
    p = _body(l_over_d=13.4, fins=False)
    assert _ld(p, 8.0) <= _ld(p, 3.0) + 0.05


@pytest.mark.xfail(strict=True, reason="L/D calibration pending: finless slender-"
                   "body L/D_max at M5 is over-predicted (Seiff-Wilkins nonlinear "
                   "body-lift ~24% high). Remove marker when recalibrated.")
def test_finless_slender_body_ld_within_physical_band():
    """A wingless slender missile body at M5 belongs above the CAN-4 flared
    projectile (~1.25) but well below a winged hypersonic glider (~4-6.7):
    physically ~1.5-2.5.  Today the build-up gives ~3.06."""
    ld5 = _ld(_body(l_over_d=13.4, fins=False), 5.0)
    assert 1.0 <= ld5 <= 2.5


# ── Guards: must stay green THROUGH the calibration ──────────────────────────

def test_winged_glider_anchor_preserved():
    """A body with large wings (Seiff-Wilkins class) is wing-lift dominated,
    which the build-up already gets right (~5.3 at M5, inside the measured
    4-6.7).  The calibration targets the BODY cross-flow term, so this anchor
    must NOT be dragged down with it."""
    ld5 = _ld(_body(l_over_d=13.4, big_wings=True), 5.0)
    assert 3.5 <= ld5 <= 7.0


def test_ld_increases_with_lifting_surface():
    """Adding lifting surface must raise L/D monotonically: winged > tail-finned
    > finless.  A guard that the recalibration reshapes the body term without
    inverting the model's response to fins/wings."""
    m = 5.0
    finless = _ld(_body(l_over_d=13.4, fins=False), m)
    finned = _ld(_body(l_over_d=13.4, fins=True), m)
    winged = _ld(_body(l_over_d=13.4, big_wings=True), m)
    assert winged > finned > finless
