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
    base.body_reenters = True       # the booster owns the separation link
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


# ── The gate must actually gate: control authority is read, not assumed ───────

def _tiered(tier, **kw):
    """A body whose reentry object declares a control-surface tier."""
    p = _body(**kw)
    p.ro = copy.deepcopy(p.ro)
    p.ro.glider_control_surfaces = tier
    p.__dict__.pop('_ero_memo', None)
    return p


def test_fixed_surfaces_do_not_glide():
    """A stable body with NO commanded control surfaces trims at zero incidence
    and reenters ballistically, however good its aerodynamic ceiling is.

    This is the branch a fin-stabilized ballistic-missile body belongs in, and
    the one the gate could not express while it assumed a 25 deg all-moving
    control on every finned airframe (BODY_GLIDE_LD_PLAN.md 7.1)."""
    g = tg.trim_gate(_tiered('none', l_over_d=13.4, fins=True, cg_frac=0.55),
                     mach=5.0)
    assert g['LD_max'] > 2.0                 # the ceiling is untouched...
    assert g['LD_achievable'] == 0.0         # ...but nothing can command it
    assert g['alpha_trim_max_deg'] == 0.0
    assert g['delta_max_deg'] == 0.0


def test_control_tier_orders_the_achievable_glide():
    """More control authority reaches a higher trim alpha, so the achievable
    glide is monotone in the tier while the ceiling stays fixed."""
    kw = dict(l_over_d=13.4, fins=True, cg_frac=0.50)
    gs = {t: tg.trim_gate(_tiered(t, **kw), mach=5.0)
          for t in ('none', 'small', 'substantial')}
    assert all(abs(gs[t]['LD_max'] - gs['none']['LD_max']) < 1e-9 for t in gs)
    assert (gs['none']['alpha_trim_max_deg']
            < gs['small']['alpha_trim_max_deg']
            < gs['substantial']['alpha_trim_max_deg'])
    assert gs['none']['LD_achievable'] < gs['small']['LD_achievable']
    assert gs['small']['LD_achievable'] <= gs['substantial']['LD_achievable']


def test_trim_alpha_stays_physical():
    """The trim solve must never leave the build-up's own alpha sweep (1-59 deg).

    The linearised relation this replaced divided by (x_cp - x_cg), a small
    difference of large numbers, and returned 144 deg for a Scud-B body and over
    600 deg near neutral stability -- which the gate then read as 'control
    reaches best glide' and so handed back the unconstrained peak."""
    for cg in (0.40, 0.45, 0.50, 0.55, 0.60, 0.62):
        for tier in ('none', 'small', 'substantial', 'unknown'):
            g = tg.trim_gate(_tiered(tier, l_over_d=13.4, fins=True,
                                     cg_frac=cg), mach=5.0)
            if g.get('error'):
                continue
            assert 0.0 <= g['alpha_trim_max_deg'] <= 59.0, (
                f"cg={cg} tier={tier}: alpha_trim {g['alpha_trim_max_deg']}")
            assert g['LD_achievable'] <= g['LD_max'] + 1e-9


def test_unknown_control_is_flagged_as_assumed():
    """'unknown' is the shipped default on every reentry object, so a glide it
    produces must be reported as resting on an assumption."""
    g = tg.trim_gate(_tiered('unknown', l_over_d=13.4, fins=True, cg_frac=0.50),
                     mach=5.0)
    assert g['control_assumed'] is True
    assert g['control_tier'] == 'unknown'
    if g['LD_achievable'] > 0.0:
        assert 'ASSUMED' in g['verdict']
    d = tg.trim_gate(_tiered('substantial', l_over_d=13.4, fins=True,
                             cg_frac=0.50), mach=5.0)
    assert d['control_assumed'] is False


def test_deflection_respects_the_separation_limit():
    """Usable deflection is capped at the Kumar & Stollery separation limit
    (docs/cl_margin_references.md), the same 15 deg damping_estimate.py uses --
    not the uncited 25 deg this replaced."""
    assert tg._DELTA_MAX_BY_CONTROL['substantial'] == 15.0
    assert tg._DELTA_MAX_BY_CONTROL['none'] == 0.0
    import damping_estimate as de
    assert tg._DELTA_MAX_BY_CONTROL['substantial'] == de.DELTA_MAX_DEG
    # An explicit per-object deflection overrides the tier, still capped.
    p = _tiered('small', l_over_d=13.4, fins=True, cg_frac=0.50)
    p.ro.glider_flap_deflection_deg = 40.0
    assert tg.control_authority(p.ro)['delta_max_deg'] == 15.0


def test_cn_components_regroup_the_sweep_exactly():
    """The moment balance must use the SAME normal force as the L/D sweep: the
    three components are a regrouping of C_N, not a second model of it."""
    import math
    p = _body(l_over_d=13.4, fins=True)
    a = gld.whole_booster_LD(p, mach=5.0)
    A_p = a['body_planform_m2'] + a['fin_planform_m2']
    for i in range(1, 60):
        r = math.radians(i)
        sn = math.sin(r)
        expect = (a['c_na_pot'] * math.sin(2 * r) / 2.0
                  + gld._ETA * gld.crossflow_cd(5.0 * sn)
                  * (A_p / a['ref_area_m2']) * sn * sn)
        assert sum(gld.cn_components(a, r).values()) == pytest.approx(expect,
                                                                     abs=1e-12)
