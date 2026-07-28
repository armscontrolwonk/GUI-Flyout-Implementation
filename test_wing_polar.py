"""Wing-decoupled drag polar (trajectory._aero_polar / _polar_cd).

The pull tax the SWERVE campaign exposed: the polar's induced-drag factor k is
back-solved from (β, L/D), which pins the vehicle to the slender-BODY lift
behaviour, so a hard pull runs at L/D ≈ 0.94 (half the nominal).  A winged
vehicle carries lift more efficiently off the cruise point.  The decoupling
adds a geometry anchor — wing area + aspect ratio — that broadens the drag
bucket ONLY on the pull side (C_L > C_L*), leaving the cruise bucket (and
therefore glider_LD) exactly as-is.

Discipline: every assertion is an identity, a monotonicity, or a direction —
never a fit to the corridor.  wing_area = 0 must reproduce the old polar
byte-for-byte, since that guards every shipped vehicle.
"""

import dataclasses
import json

import numpy as np
import pytest

import math

from booster_models import ro_from_dict, wing_geometry
from trajectory import (_aero_polar, _polar_cd, _C_L_MAX_BODY,
                        WING_DEFAULT_AR)


def _swerve(**wing):
    ro = ro_from_dict(json.load(open("ro_library/SWERVE.ro.json")))
    return dataclasses.replace(ro, **wing)


# ── the identity that guards every shipped vehicle ──────────────────────────
def test_no_wings_reproduces_the_old_polar_exactly():
    p = _aero_polar(_swerve())
    A = np.pi * (0.476 / 2) ** 2
    C_D0 = 300.0 / (24733.0 * A)
    k = 1.0 / (4.0 * C_D0 * 1.8 ** 2)
    assert p.C_D0 == pytest.approx(C_D0)
    assert p.k == pytest.approx(k)
    assert p.C_L_max == pytest.approx(_C_L_MAX_BODY)     # universal 25° limit
    assert p.e_pull == 1.0                               # no bucket broadening


def test_polar_cd_is_the_plain_polar_when_e_pull_is_one():
    p = _aero_polar(_swerve())
    for cl in (0.0, 0.1, p.C_L_star, 0.5, p.C_L_max):
        assert _polar_cd(cl, p) == pytest.approx(p.C_D0 + p.k * cl * cl)


def test_wing_area_zero_is_identical_to_no_key():
    a = _aero_polar(_swerve())
    b = _aero_polar(_swerve(wing_area_m2=0.0, wing_aspect_ratio=5.0))
    assert (a.C_D0, a.k, a.C_L_max, a.e_pull) == (b.C_D0, b.k, b.C_L_max, b.e_pull)


# ── what wings change, and what they must NOT ───────────────────────────────
def test_wings_broaden_the_pull_bucket_only():
    """e_pull > 1 with wings, and it softens drag ONLY above C_L* — the cruise
    side (C_L ≤ C_L*) is byte-identical, so glider_LD is never double-credited."""
    body = _aero_polar(_swerve())
    wing = _aero_polar(_swerve(wing_area_m2=0.2, wing_aspect_ratio=4.0))
    assert wing.e_pull > 1.0
    # cruise side unchanged
    for cl in (0.05, 0.15, body.C_L_star):
        assert _polar_cd(cl, wing) == pytest.approx(_polar_cd(cl, body))
    # pull side softened
    for cl in (0.5, 0.7, 0.85):
        assert _polar_cd(cl, wing) < _polar_cd(cl, body)


def test_ceiling_is_not_raised_by_wings():
    """The pull ceiling is a max-AoA limit, not a wing property — raising it
    just lets the pull rail at ruinous induced drag (a verified regression)."""
    for S, AR in ((0.1, 2.0), (0.5, 6.0), (2.0, 10.0)):
        assert _aero_polar(_swerve(wing_area_m2=S,
                                   wing_aspect_ratio=AR)).C_L_max == \
            pytest.approx(_C_L_MAX_BODY)


def test_pull_efficiency_rises_monotonically_with_aspect_ratio():
    e = [_aero_polar(_swerve(wing_area_m2=0.2, wing_aspect_ratio=ar)).e_pull
         for ar in (1.0, 2.0, 4.0, 8.0, 16.0)]
    assert all(e[i] < e[i + 1] for i in range(len(e) - 1))


def test_pull_efficiency_rises_with_wing_area():
    e = [_aero_polar(_swerve(wing_area_m2=s, wing_aspect_ratio=4.0)).e_pull
         for s in (0.05, 0.1, 0.3, 0.6)]
    assert all(e[i] < e[i + 1] for i in range(len(e) - 1))


# ── the fail-safe: area given, span/AR absent ───────────────────────────────
def test_area_only_falls_back_to_a_stubby_default_ar():
    """Area without AR must credit a modest, conservative benefit (the stubby
    WING_DEFAULT_AR), never zero and never the optimistic high-AR value."""
    area_only = _aero_polar(_swerve(wing_area_m2=0.3))
    as_default = _aero_polar(_swerve(wing_area_m2=0.3,
                                     wing_aspect_ratio=WING_DEFAULT_AR))
    hi_ar = _aero_polar(_swerve(wing_area_m2=0.3, wing_aspect_ratio=10.0))
    assert area_only.e_pull == pytest.approx(as_default.e_pull)   # uses default
    assert 1.0 < area_only.e_pull < hi_ar.e_pull                  # modest, safe


# ── end-to-end direction: wings reduce the pull tax ─────────────────────────
def test_wings_help_the_pull_retain_energy_end_to_end():
    """A full run: with a commanded pull, adding wings must catch at a similar
    altitude while retaining MORE speed (the pull tax shrinks) — the direction
    the decoupling exists to produce.  Not pinned to any flight number."""
    import copy
    from booster_models import (get_booster, apply_reentry_plan,
                                load_reentry_plan)
    from trajectory import integrate_trajectory
    from atmosphere import speed_of_sound

    def fly(wing):
        ro = _swerve(**wing)
        rp = dict(load_reentry_plan("SWERVE") or {}, glider_guidance='damped_glide',
                  glider_aero_model='polar', glider_damping_zeta=0.3,
                  glider_pullup_start_alt_km=40.0)
        ro = apply_reentry_plan(copy.deepcopy(ro), rp)
        b = get_booster("Strypi VIII R"); b.ro = copy.deepcopy(ro)
        b.guidance = "pitch_program"; b.launch_elevation_deg = 70.0
        r = integrate_trajectory(b, 22.0228, -159.785, 270.0, burnout_angle_deg=0.0,
                                 cutoff_time_s=57.0, gt_turn_start_s=0.0,
                                 gt_turn_stop_s=50.0, launch_elevation_deg=70.0,
                                 max_time_s=3600.0, dt_output=1.0)
        a = np.asarray(r['alt']); s = np.asarray(r['speed']); i = int(np.argmax(a))
        kft = a / 304.8; m = (np.arange(len(a)) > i) & (np.abs(kft - 85) < 3)
        return s[m][0] / speed_of_sound(85 * 304.8) if m.any() else float('nan')

    m_body = fly({})
    m_wing = fly({'wing_area_m2': 0.2, 'wing_aspect_ratio': 8.0})
    assert m_wing > m_body + 0.3, (m_body, m_wing)


# ── wing_geometry(): the single source of truth for S and AR ────────────────
# The rule the user asked for: whenever a planform (root chord + exposed span)
# is present, DERIVE S and AR from it — never trust a hand-typed pair.  These
# pin the reference-wing convention (trapezoid exposed panels + rectangular
# carry-through through the body) as an identity, not a fit.

def test_planform_derives_reference_area_and_ar_unswept():
    """Unswept: c_t = c_r, so S = 2·c_r·s_e + c_r·D and b = 2·s_e + D."""
    ro = _swerve(diameter_m=0.8, wing_root_chord_m=2.0,
                 wing_span_exposed_m=1.0, wing_sweep_deg=0.0,
                 wing_area_m2=0.0, wing_aspect_ratio=0.0)
    S, AR, src = wing_geometry(ro)
    assert src == 'planform'
    assert S == pytest.approx(2.0 * 2.0 * 1.0 + 2.0 * 0.8)          # 5.6
    assert AR == pytest.approx((2.0 * 1.0 + 0.8) ** 2 / S)          # b²/S


def test_planform_sweep_shrinks_tip_chord_and_area():
    """LE sweep shrinks the tip chord (c_t = c_r − s_e·tanΛ), so S falls and,
    at fixed span, AR rises."""
    ro = _swerve(diameter_m=0.8, wing_root_chord_m=2.0,
                 wing_span_exposed_m=1.0, wing_sweep_deg=30.0,
                 wing_area_m2=0.0, wing_aspect_ratio=0.0)
    S, AR, src = wing_geometry(ro)
    c_t = 2.0 - 1.0 * math.tan(math.radians(30.0))
    S_exp = (2.0 + c_t) * 1.0 + 2.0 * 0.8
    b = 2.0 * 1.0 + 0.8
    assert src == 'planform'
    assert S == pytest.approx(S_exp)
    assert AR == pytest.approx(b * b / S_exp)


def test_direct_pair_used_only_when_no_planform():
    S, AR, src = wing_geometry(_swerve(wing_area_m2=3.0, wing_aspect_ratio=2.5))
    assert src == 'direct'
    assert S == pytest.approx(3.0) and AR == pytest.approx(2.5)


def test_no_wing_data_is_none():
    S, AR, src = wing_geometry(_swerve(wing_area_m2=0.0, wing_aspect_ratio=0.0))
    assert (S, AR, src) == (0.0, 0.0, None)


def test_planform_beats_a_stale_direct_pair():
    """A planform present ALONGSIDE a (drifted) direct S/AR must win — the
    whole point of deriving is that a hand-typed number can't desync."""
    ro = _swerve(diameter_m=0.8, wing_root_chord_m=2.0, wing_span_exposed_m=1.0,
                 wing_sweep_deg=0.0, wing_area_m2=99.0, wing_aspect_ratio=99.0)
    S, AR, src = wing_geometry(ro)
    assert src == 'planform'
    assert S == pytest.approx(5.6) and AR != pytest.approx(99.0)


def test_polar_consumes_the_derived_planform_area():
    """End-to-end: the drag polar's pull benefit must track the DERIVED area,
    not the ignored direct field.  A planform giving S≈0.2 must broaden the
    pull bucket like an explicit wing_area_m2=0.2 of the same AR would."""
    # Pick a small planform whose derived S is order 0.2 m² so we stay on the
    # same side as the existing direct-entry tests.
    ro_pf = _swerve(diameter_m=0.5, wing_root_chord_m=0.2,
                    wing_span_exposed_m=0.2, wing_sweep_deg=0.0,
                    wing_area_m2=0.0, wing_aspect_ratio=0.0)
    S, AR, src = wing_geometry(ro_pf)
    assert src == 'planform' and S > 0.0
    p_pf = _aero_polar(ro_pf)
    p_eq = _aero_polar(_swerve(diameter_m=0.5, wing_area_m2=S,
                               wing_aspect_ratio=AR))
    assert p_pf.e_pull == pytest.approx(p_eq.e_pull)
    assert p_pf.e_pull > 1.0
