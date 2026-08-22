"""Biconic front end is end-to-end: DRAWN ≡ FLOWN for the two-cone shape.

A declared biconic (fore cone + aft frustum) used to be modeled as a single
cone everywhere except the manual β-estimate dialog: the flown drag, the trim
gate's CP, the L/D build-up and the schematic all read only the shape string
('cone' for a biconic) and ignored the fore-cone length / break diameter.  So
Thrusty would draw one cone, fly one cone, but let the user type a biconic β —
declared ≠ modeled.

These tests pin the biconic as a first-class shape through all four consumers,
anchored by two exact reduction identities (a biconic with equal half-angles is
a single cone; with break_ratio → 1 the aft annulus vanishes).
"""

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import booster_models as mm
import glider_ld as gld
import grid_fin_sizing as gfs
import trim_gate as tg
import booster_schematic as bs
from booster_models import (get_booster, load_booster_library, ROParams,
                            compose_loadout)
from trajectory import integrate_trajectory

load_booster_library()


def _kn23(biconic, fore=1.0, break_d=0.5):
    p = get_booster("Scud-B (R-17)")
    p.diameter_m = 1.1
    p.length_m = 6.7
    ro = ROParams(name="KN23", mass_kg=2100.0, beta_kg_m2=0.0, shape="cone",
                  diameter_m=1.1, length_m=6.7, separation_mode="body",
                  glider_enabled=True, glider_LD=0.0,
                  glider_guidance="damped_glide", body_nose_length_m=2.0,
                  nose_radius_m=0.055, biconic=biconic,
                  fore_length_m=fore, break_diameter_m=break_d)
    p2 = compose_loadout(p, ro, 1)
    p2.ro = ro
    return p2


# ── reduction identities (pure geometry / aero) ─────────────────────────────

@pytest.mark.parametrize("f", [0.3, 0.5, 0.7])
def test_cp_reduces_to_single_cone_on_a_straight_break(f):
    """A break lying on a straight cone (break_ratio = fore_len/nose_len) is one
    cone: the two-segment Barrowman CP recovers the single-cone 2/3, for ANY
    break station."""
    L = 2.0
    Lf, La, br = f * L, (1 - f) * L, f       # straight cone: br = Lf/L
    frac = mm.biconic_nose_cp_fraction(br, Lf, La)
    assert frac == pytest.approx(2.0 / 3.0, abs=1e-9)


@pytest.mark.parametrize("M", [3.0, 5.0, 8.0, 12.0])
def test_cd0_reduces_to_single_cone_when_sharp_and_equal(M):
    """A sharp biconic with θ2 = θ1 is one cone: its Cd0 equals the single
    hypersonic cone plus the shared cylindrical-afterbody term."""
    geom = dict(theta1_deg=12.0, theta2_deg=12.0, break_ratio=0.5, eps=0.0,
                fore_len_m=1.0, aft_len_m=1.0, nose_len_m=2.0,
                base_diameter_m=1.1, break_diameter_m=0.55)
    ld_body = 6.7 / 1.1
    cbc = mm.cd0_biconic_body(geom, ld_body, M)
    ld_nose = 2.0 / 1.1
    cone = (mm.cd_cone_hypersonic(12.0, 0.0, mach=M)['total']
            + mm.CONE_CF_TURBULENT * 4.0 * (ld_body - ld_nose))
    assert cbc == pytest.approx(cone, abs=1e-12)


def test_break_ratio_moves_the_cp():
    """The break ratio is the CP lever at fixed lengths: a slender break (small
    ratio) throws area onto the aft frustum and moves the CP aft of the straight
    cone's 2/3; a fat break moves it forward — the two-cone stability a single
    fraction cannot show."""
    straight = mm.biconic_nose_cp_fraction(0.5, 1.0, 1.0)
    assert straight == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert mm.biconic_nose_cp_fraction(0.35, 1.0, 1.0) > straight   # slender → aft
    assert mm.biconic_nose_cp_fraction(0.65, 1.0, 1.0) < straight   # fat → forward


# ── the resolver activates only on valid biconic geometry ───────────────────

def test_geometry_none_without_the_biconic_flag():
    assert mm.biconic_nose_geometry(_kn23(False)) is None


def test_geometry_none_when_break_fields_unset():
    """Biconic flagged but fore-length / break-diameter missing → None, so the
    single-cone path stands (biconic activates only when fully specified)."""
    p = _kn23(True, fore=0.0, break_d=0.0)
    assert mm.biconic_nose_geometry(p) is None


def test_geometry_resolves_the_two_half_angles():
    g = mm.biconic_nose_geometry(_kn23(True))
    assert g is not None
    assert g['theta1_deg'] > 0 and g['theta2_deg'] > 0
    assert 0.0 < g['break_ratio'] < 1.0
    assert g['nose_len_m'] == pytest.approx(2.0)


# ── the four consumers all see two cones (drawn ≡ flown) ─────────────────────

def test_drag_cp_ld_all_move_with_biconic():
    """Toggling biconic changes the flown Cd0, the trim-gate CP/static margin,
    and the L/D — not just the picture."""
    p0, p1 = _kn23(False), _kn23(True)
    cd0 = gld.body_cd0(p0, 5.0), gld.body_cd0(p1, 5.0)
    g0 = tg.trim_gate(p0, mach=gld.GLIDE_MACH_REF)
    g1 = tg.trim_gate(p1, mach=gld.GLIDE_MACH_REF)
    ld0 = gld.whole_booster_LD(p0, mach=5.0)['ld_max']
    ld1 = gld.whole_booster_LD(p1, mach=5.0)['ld_max']
    assert cd0[0] != pytest.approx(cd0[1], abs=1e-4)
    assert g0['x_cp_m'] != pytest.approx(g1['x_cp_m'], abs=1e-3)
    assert ld0 != pytest.approx(ld1, abs=1e-3)


def test_schematic_reports_a_biconic_front_end():
    """The schematic draws (and records) the biconic, so the invariant check
    sees body_biconic — the declared shape, not a stand-in cone."""
    fig, ax = plt.subplots()
    info = bs.draw_booster(ax, _kn23(True))
    plt.close(fig)
    fe = info['front_end']
    assert fe['kind'] == 'body_biconic'
    assert fe['shape'] == 'biconic'
    assert fe['nose_length_m'] == pytest.approx(2.0)
    # single-cone control still draws a plain nose
    fig, ax = plt.subplots()
    info0 = bs.draw_booster(ax, _kn23(False))
    plt.close(fig)
    assert info0['front_end']['kind'] == 'body_nose'


def test_schematic_total_height_unchanged_by_biconic():
    """The nose is still carved subtractively — a biconic body is the same
    height as the airframe, not airframe + nose."""
    fig, ax = plt.subplots()
    info = bs.draw_booster(ax, _kn23(True))
    plt.close(fig)
    assert info['total_height_m'] == pytest.approx(6.7, abs=1e-6)


# ── it flies, and differently from the single cone ──────────────────────────

def test_biconic_body_flies_and_differs_from_single_cone():
    """A biconic body integrates a full trajectory and lands a different range
    than the single-cone equivalent (the two-cone drag/CP genuinely propagate
    to the flyout, not just the estimate)."""
    r_bic = integrate_trajectory(_kn23(True), 39.12, 125.67, 90.0,
                                 burnout_angle_deg=-2.0, max_time_s=3600.0)
    r_cone = integrate_trajectory(_kn23(False), 39.12, 125.67, 90.0,
                                  burnout_angle_deg=-2.0, max_time_s=3600.0)
    assert r_bic['range_km'] > 0.0
    assert r_bic['range_km'] != pytest.approx(r_cone['range_km'], rel=1e-3)
