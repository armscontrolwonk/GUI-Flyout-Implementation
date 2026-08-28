"""Whole-body L/D: ceiling stays DATCOM-validated, flown L/D is trim-limited.

Investigating "non-separating bodies over-range in phugoid glide", the first
hypothesis was that whole_booster_LD over-predicts L/D.  It does NOT, for the
shape it models: cross-checked against Digital DATCOM (USAF, AFFDL-TR-79-3032)
for the finless slender reference body, the build-up sits within ~10% and
CONSERVATIVE (under-predicts) at M2/3/5.  The low free-flight L/D quoted for
fin-stabilized bodies (~1) and the flared-projectile / winged-glider anchors are
DIFFERENT quantities or DIFFERENT shape classes:

  * ~1 for a fin-stabilized drag-driven body is the TRIMMED L/D at its low trim
    alpha (cg-set), not the L/D_max ceiling — trim_gate evaluates it.
  * CAN-4 (~1.25, DREV TM-9525 / Yates AIAA 96-3360) is a stubby cone-cylinder-
    FLARE projectile (L/d 5.84), a high-drag shape.
  * Seiff-Wilkins (~4-6.7, NASA TN D-341) is a WINGED hypersonic glider.

So the ceiling must NOT be de-rated (that breaks the DATCOM validation), and the
over-range realism lever is TRIM / cg — which the trim gate already applies.
These tests pin both: the ceiling stays glued to DATCOM, and small stabilizing
fins do not buy full best-glide L/D.
"""

import copy

import pytest

from booster_models import (get_booster, load_booster_library, ROParams,
                            compose_loadout)
import glider_ld as gld
import trim_gate as tg
from validation.datcom.compare_datcom import BODY as DATCOM_BODY, OUT, parse_datcom

load_booster_library()


def _body(l_over_d=13.4, fins=False, big_wings=False,
          fin_span=0.4, fin_root=1.0, cg_frac=None):
    base = copy.deepcopy(get_booster("Scud-B (R-17)"))
    base.body_reenters = True
    last = base
    while getattr(last, 'stage2', None) is not None:
        last = last.stage2
    last.length_m = l_over_d * last.diameter_m
    last.has_fins = fins or big_wings
    if fins:
        last.n_fins = 4; last.fin_span_m = fin_span
        last.fin_root_chord_m = fin_root; last.fin_tip_chord_m = fin_root * 0.3
        last.fin_thickness_m = 0.03; last.fin_sweep_deg = 45.0
    if big_wings:
        last.n_fins = 3; last.fin_span_m = 1.2
        last.fin_root_chord_m = last.length_m * 0.9; last.fin_tip_chord_m = 0.2
        last.fin_thickness_m = 0.03; last.fin_sweep_deg = 70.0
    ro = ROParams(name="fe", mass_kg=max(float(last.mass_final), 1.0),
                  beta_kg_m2=0.0, shape="cone",
                  diameter_m=float(last.diameter_m),
                  length_m=float(last.length_m), separation_mode='body',
                  glider_enabled=True, glider_LD=0.0,
                  reentry_cg_m=(cg_frac * last.length_m if cg_frac else 0.0))
    p = compose_loadout(base, ro, 1)
    p.ro = ro
    return p


def _ld(p, mach):
    return gld.whole_booster_LD(p, mach=mach)['ld_max']


# ── Ceiling: glued to Digital DATCOM, conservative — the anti-de-rate guard ───

@pytest.mark.parametrize("mach,d_ld", [(2.0, 2.23), (3.0, 2.71), (5.0, 3.51)])
def test_ld_ceiling_matches_datcom_and_stays_conservative(mach, d_ld):
    """whole_booster_LD's L/D_max for the finless slender reference body must
    track Digital DATCOM within 12% and stay conservative (<= DATCOM).  This is
    the guard that fails loudly if anyone (re-)introduces a crossflow de-rate:
    the low L/D of a real fin-stabilized body is a TRIM effect, not a lower
    ceiling."""
    got = gld.whole_booster_LD(DATCOM_BODY, mach=mach)['ld_max']
    assert got <= d_ld * 1.001                    # conservative, never above
    assert got >= d_ld * 0.88                      # within ~12%


def test_datcom_reference_output_parses():
    """The committed DATCOM reference still parses to the three Mach blocks the
    guard above pins (guards the parser + fixture, not the physics)."""
    blocks = parse_datcom(OUT)
    assert sorted(m for m, _ in blocks) == [2.0, 3.0, 5.0]


# ── Ceiling shape sanity ─────────────────────────────────────────────────────

def test_ld_plateaus_into_hypersonic():
    """L/D_max may rise through the transonic cross-flow-drag peak (~M3->M5) but
    must PLATEAU beyond it, not keep climbing to M12."""
    p = _body(l_over_d=13.4, fins=False)
    assert _ld(p, 12.0) <= _ld(p, 5.0) * 1.15


def test_winged_glider_anchor_in_band():
    """A big-winged body (Seiff-Wilkins class) sits inside the measured 4-6.7
    at M5 (~5.3)."""
    assert 3.5 <= _ld(_body(l_over_d=13.4, big_wings=True), 5.0) <= 7.0


def test_ld_increases_with_lifting_surface():
    """winged > tail-finned > finless (monotone response to lifting surface)."""
    m = 5.0
    finless = _ld(_body(l_over_d=13.4, fins=False), m)
    finned = _ld(_body(l_over_d=13.4, fins=True), m)
    winged = _ld(_body(l_over_d=13.4, big_wings=True), m)
    assert winged > finned > finless


# ── Operative cap: trim decides the FLOWN L/D (the realism lever) ─────────────

def test_small_fins_do_not_reach_best_glide_ld():
    """A slender body with only small stabilizing fins must NOT fly at its
    L/D_max ceiling: unstable (tumbles, L/D 0) or control-limited to a low trim
    alpha.  Full best-glide L/D is bought by control authority + cg, not
    geometry — this is where the ~1 free-flight number lives."""
    g = tg.trim_gate(_body(l_over_d=8.0, fins=True, fin_span=0.25,
                           fin_root=0.6, cg_frac=0.55), mach=5.0)
    assert g['LD_achievable'] < g['LD_max']


def test_control_rich_stable_body_reaches_best_glide():
    """The converse: a stable, control-rich body DOES trim to best glide, so the
    trim gate returns the full (DATCOM-validated) ceiling."""
    g = tg.trim_gate(_body(l_over_d=13.4, big_wings=True, cg_frac=0.55),
                     mach=5.0)
    assert g['LD_achievable'] == pytest.approx(g['LD_max'], rel=1e-6)
    assert g['LD_achievable'] > 3.0
