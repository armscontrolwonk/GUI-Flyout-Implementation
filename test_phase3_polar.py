"""Phase 3 — polar upgrades for lifting forms (the ONE phase that touches
trajectory physics, gated so every axisymmetric vehicle is byte-identical).

Three upgrades under test, per PHASE2_LIFTING_BODY_PLAN §7 / TODO:
  1. shape-derived C_L,max (Newtonian pressure at the 25° AoA cap) replacing
     the universal Munk 0.873 for lifting forms with sufficient geometry;
  2. the offset polar C_D = C_D0 + k·[(C_L − C_L0)² − C_L0²] (Lobanovskii
     trinomial; Fetterman Fig. 6b measured) — anchored so C_D(0) = C_D0
     exactly (β keeps its zero-lift meaning) and (L/D)max stays EXACTLY
     glider_LD (k is back-solved on the offset parabola);
  3. trim_alpha_deg / trim_CL0 stored on the RO (sweep-native coefficients),
     round-tripping through json and xlsx.

Every test is an identity or a gated-off byte-identity — never a fit.
"""

import dataclasses
import json
import math

import numpy as np
import pytest

from booster_models import (ro_from_dict, ro_to_dict, _half_cone_coeffs,
                            _wedge_coeffs, wedge_planform_area,
                            NEWTON_K_SLENDER)
from trajectory import _aero_polar, _polar_cd, _Polar, _C_L_MAX_BODY


def _ro(**overrides):
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    return dataclasses.replace(ro, **overrides) if overrides else ro


# ── gate: axisymmetric vehicles byte-identical ──────────────────────────────
def test_axisymmetric_polar_is_the_pre_phase3_polar_exactly():
    ro = _ro(body_form="axisymmetric")
    p = _aero_polar(ro)
    A_ref = 0.25 * np.pi * ro.diameter_m ** 2
    C_D0 = ro.mass_kg / (ro.beta_kg_m2 * A_ref)
    LD = ro.glider_LD
    k = 1.0 / (4.0 * C_D0 * LD * LD) if LD > 0 else 0.5
    assert p.C_D0 == pytest.approx(max(min(C_D0, 1.0), 0.005), rel=1e-12)
    assert p.k == pytest.approx(max(min(k, 5.0), 0.05), rel=1e-12)
    assert p.C_L0 == 0.0
    assert p.C_L_max == max(_C_L_MAX_BODY, 1.05 * p.C_L_star)


def test_stale_trim_on_axisymmetric_is_inert():
    """A trim row can only skew a lifting form's polar: on a body of
    revolution it is ignored entirely (and the editor zeroes it on save)."""
    a = _aero_polar(_ro(body_form="axisymmetric"))
    b = _aero_polar(_ro(body_form="axisymmetric", trim_CL0=0.05,
                        trim_alpha_deg=12.0))
    assert a == b


def test_zero_trim_lifting_form_keeps_the_symmetric_polar():
    p = _aero_polar(_ro(body_form="half_cone", trim_CL0=0.0))
    assert p.C_L0 == 0.0
    # symmetric reduction of _polar_cd: C_D0 + k·C_L²
    for cl in (0.0, 0.1, 0.3):
        assert _polar_cd(cl, p) == pytest.approx(p.C_D0 + p.k * cl * cl,
                                                 rel=1e-12)


# ── the offset polar identities ─────────────────────────────────────────────
def test_offset_polar_preserves_beta_meaning_at_zero_lift():
    """C_D(0) = C_D0 EXACTLY — β stays the zero-lift ballistic coefficient;
    the offset never leaks into the β anchor."""
    p = _aero_polar(_ro(body_form="half_cone", trim_CL0=0.02))
    assert p.C_L0 == pytest.approx(0.02)          # base-referenced: factor 1
    assert _polar_cd(0.0, p) == pytest.approx(p.C_D0, rel=1e-12)


def test_offset_polar_minimum_drag_sits_at_CL0():
    p = _aero_polar(_ro(body_form="half_cone", trim_CL0=0.02))
    grid = np.linspace(0.0, p.C_L_star, 2001)
    cds = [_polar_cd(c, p) for c in grid]
    assert grid[int(np.argmin(cds))] == pytest.approx(p.C_L0, abs=2e-4)
    assert _polar_cd(p.C_L0, p) == pytest.approx(
        p.C_D0 - p.k * p.C_L0 ** 2, rel=1e-12)     # the analytic vertex


def test_offset_polar_LDmax_is_exactly_glider_LD():
    """The k back-solve identity on the offset parabola: the polar's best
    L/D equals the stored glider_LD — the estimator's L/D survives the
    offset instead of being silently inflated by it."""
    ro = _ro(body_form="half_cone", trim_CL0=0.02)
    assert ro.glider_LD > 0
    p = _aero_polar(ro)
    grid = np.linspace(1e-4, p.C_L_star * 1.001, 4001)
    ld = max(c / _polar_cd(c, p) for c in grid)
    assert ld == pytest.approx(ro.glider_LD, rel=2e-3)
    # and the optimum still sits at C_L* = √(C_D0/k), unchanged by the offset
    assert p.C_L_star == pytest.approx(math.sqrt(p.C_D0 / p.k), rel=1e-12)


def test_inconsistent_CL0_falls_back_to_symmetric():
    """A C_L0 beyond the discriminant (C_L0 > LD·C_D0/2) cannot coexist with
    the stated L/D — fall back to the symmetric polar rather than invent."""
    p = _aero_polar(_ro(body_form="half_cone", trim_CL0=5.0))
    ps = _aero_polar(_ro(body_form="half_cone", trim_CL0=0.0))
    assert p.C_L0 == 0.0 and p.k == ps.k


def test_wedge_offset_converts_from_planform_reference():
    """The wedge's trim_CL0 is planform-referenced (the estimator's stated
    A_ref); the polar converts by A_planform/A_polar — lift force invariant.
    Without the span there is no planform to state: offset off."""
    ro = _ro(body_form="wedge", diameter_m=0.30, length_m=3.0,
             body_span_m=1.4, trim_CL0=0.001)
    p = _aero_polar(ro)
    a_plan = wedge_planform_area(3.0, 1.4)
    a_polar = 0.25 * np.pi * 0.30 ** 2
    assert p.C_L0 == pytest.approx(0.001 * a_plan / a_polar, rel=1e-9)
    no_span = _aero_polar(dataclasses.replace(ro, body_span_m=0.0))
    assert no_span.C_L0 == 0.0


# ── shape-derived C_L,max ───────────────────────────────────────────────────
def test_wedge_ceiling_is_newtonian_at_the_aoa_cap():
    ro = _ro(body_form="wedge", diameter_m=0.30, length_m=3.0,
             body_span_m=1.4)
    p = _aero_polar(ro)
    s_plan = wedge_planform_area(3.0, 1.4)
    c = _wedge_coeffs(3.0, 0.30, 1.4, 25.0, NEWTON_K_SLENDER, 0.0, False,
                      10.0, s_plan)
    assert p.C_L_max == pytest.approx(c['C_L'] * s_plan / p.A_ref, rel=1e-9)
    # force-level: the flat-bottom wedge out-pulls the Munk body ceiling
    assert p.C_L_max * p.A_ref > _C_L_MAX_BODY * p.A_ref * 2.0


def test_half_cone_ceiling_matches_the_composer():
    ro = _ro(body_form="half_cone")
    p = _aero_polar(ro)
    th = math.degrees(math.atan2(ro.diameter_m / 2.0, ro.length_m))
    c = _half_cone_coeffs(th, 25.0, NEWTON_K_SLENDER, 0.0, False, 10.0)
    assert p.C_L_max == pytest.approx(
        max(max(c['C_L'], 1e-6), 1.05 * p.C_L_star), rel=1e-9)


def test_wedge_without_span_keeps_the_body_ceiling():
    """Derive-don't-invent: no span → no planform → no shape-derived lift;
    the flagged fallback is the old universal ceiling, not a guess."""
    p = _aero_polar(_ro(body_form="wedge", body_span_m=0.0))
    assert p.C_L_max == max(_C_L_MAX_BODY, 1.05 * p.C_L_star)


# ── storage round-trips ─────────────────────────────────────────────────────
def test_trim_fields_round_trip_json():
    ro = _ro(body_form="half_cone", trim_alpha_deg=7.5, trim_CL0=0.013)
    back = ro_from_dict(ro_to_dict(ro))
    assert back.trim_alpha_deg == pytest.approx(7.5)
    assert back.trim_CL0 == pytest.approx(0.013)
    # absent keys default to 0 (pre-upgrade files import unchanged)
    d = ro_to_dict(ro)
    d.pop('trim_alpha_deg'); d.pop('trim_CL0')
    old = ro_from_dict(d)
    assert old.trim_alpha_deg == 0.0 and old.trim_CL0 == 0.0


def test_trim_fields_round_trip_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    import ro_xlsx
    ro = _ro(body_form="half_cone", trim_alpha_deg=7.5, trim_CL0=0.013)
    path = tmp_path / "ro.xlsx"
    ro_xlsx.export_ro_xlsx(str(path), ro)
    back = ro_xlsx.import_ro_xlsx(str(path))
    assert back.trim_alpha_deg == pytest.approx(7.5)
    assert back.trim_CL0 == pytest.approx(0.013)
