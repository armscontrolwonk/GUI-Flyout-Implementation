"""Biconic (two-cone) hypersonic Cd build-up (booster_models.cd_biconic_hypersonic).

A biconic RV — a steep forward cone on a shallow aft frustum — cannot be
represented by the single-cone estimator without inventing an "equivalent"
half-angle.  This module adds the honest two-segment build-up.  Every assertion
here is a physics identity, a geometry identity, or the exact single-cone
reduction — nothing is pinned to a flight outcome.
"""

import math

import pytest

from booster_models import (cd_cone_hypersonic, cd_biconic_hypersonic,
                            cd_blunted_cone_newtonian, biconic_angles,
                            CONE_CF_TURBULENT)


# ── the regression anchor: reductions to the single cone ────────────────────
def test_break_at_base_reduces_to_the_single_forward_cone():
    """break_ratio = 1 collapses the aft annulus to nothing → exactly the
    cd_cone_hypersonic(theta1, eps) value, component by component."""
    for th1, eps, M in ((5.25, 0.07, 10.0), (15.0, 0.3, 8.0), (30.0, 0.5, 12.0)):
        single = cd_cone_hypersonic(th1, eps, mach=M)
        bic = cd_biconic_hypersonic(th1, 45.0, 1.0, eps, mach=M)   # theta2 irrelevant
        assert bic["pressure"] == pytest.approx(single["pressure"])
        assert bic["friction"] == pytest.approx(single["friction"])
        assert bic["base"] == pytest.approx(single["base"])
        assert bic["total"] == pytest.approx(single["total"])


def test_sharp_equal_angles_are_one_cone_for_any_break():
    """A sharp biconic with theta2 = theta1 is geometrically a single cone, so
    it must return the single-cone Cd for ANY break location."""
    for th, br, M in ((10.0, 0.3, 10.0), (20.0, 0.6, 9.0), (7.0, 0.8, 14.0)):
        single = cd_cone_hypersonic(th, 0.0, mach=M)
        bic = cd_biconic_hypersonic(th, th, br, 0.0, mach=M)
        assert bic["total"] == pytest.approx(single["total"])
        assert bic["pressure"] == pytest.approx(single["pressure"])
        assert bic["friction"] == pytest.approx(single["friction"])


# ── component identities ────────────────────────────────────────────────────
def test_total_is_the_sum_of_parts():
    c = cd_biconic_hypersonic(12.0, 6.0, 0.5, 0.1, mach=11.0)
    assert c["total"] == pytest.approx(c["pressure"] + c["friction"]
                                       + c["base"] + c["wing"])
    assert c["pressure"] == pytest.approx(c["pressure_fore"] + c["pressure_aft"])


def test_aft_pressure_is_newtonian_annulus():
    """The aft frustum's pressure is 2·sin²θ2 over the projected annulus
    (1 − break²), base-area referenced."""
    for th2, br in ((6.0, 0.4), (10.0, 0.55), (20.0, 0.7)):
        c = cd_biconic_hypersonic(15.0, th2, br, 0.0)
        want = 2.0 * math.sin(math.radians(th2)) ** 2 * (1.0 - br * br)
        assert c["pressure_aft"] == pytest.approx(want)


def test_friction_is_cf_times_both_wetted_ratios():
    for th1, th2, br, eps in ((15.0, 6.0, 0.5, 0.1), (25.0, 8.0, 0.6, 0.3)):
        c = cd_biconic_hypersonic(th1, th2, br, eps)
        r_t = eps * math.cos(math.radians(th1))
        wf = (br * br - r_t * r_t) / math.sin(math.radians(th1))
        wa = (1.0 - br * br) / math.sin(math.radians(th2))
        assert c["swet_fore"] == pytest.approx(wf)
        assert c["swet_aft"] == pytest.approx(wa)
        assert c["friction"] == pytest.approx(CONE_CF_TURBULENT * (wf + wa))


# ── the physics a single cone can't capture ─────────────────────────────────
def test_biconic_differs_from_the_naive_equivalent_single_cone():
    """A steep-fore / shallow-aft biconic is NOT the single cone of its overall
    base-to-nose angle: the shallow aft frustum sheds pressure drag the
    equivalent cone would over-count."""
    D, L, Lf, Dbrk = 0.58, 1.5, 0.5, 0.42        # steep fore, shallow aft
    th1, th2, br, eps = biconic_angles(D, L, Lf, Dbrk, nose_radius_m=0.02)
    assert th1 > th2                              # steeper fore than aft
    bic = cd_biconic_hypersonic(th1, th2, br, eps, mach=10.0)
    # the "equivalent" single cone at the overall nose-to-base angle
    th_eq = math.degrees(math.atan2(D / 2, L))
    single = cd_cone_hypersonic(th_eq, eps, mach=10.0)
    assert abs(bic["total"] - single["total"]) / single["total"] > 0.05


def test_steeper_fore_cone_raises_pressure_drag():
    """Monotonicity: at fixed break and aft angle, a steeper forward cone must
    raise the fore pressure contribution."""
    prev = None
    for th1 in (5.0, 10.0, 20.0, 30.0):
        c = cd_biconic_hypersonic(th1, 6.0, 0.5, 0.0)
        assert prev is None or c["pressure_fore"] > prev
        prev = c["pressure_fore"]


# ── geometry derivation ─────────────────────────────────────────────────────
def test_biconic_angles_from_stored_geometry():
    """The half-angles derive from (fore length, break ⌀) against the base."""
    th1, th2, br, eps = biconic_angles(0.58, 1.5, 0.5, 0.42, nose_radius_m=0.02)
    assert th1 == pytest.approx(math.degrees(math.atan2(0.21, 0.5)))
    assert th2 == pytest.approx(math.degrees(math.atan2(0.29 - 0.21, 1.0)))
    assert br == pytest.approx(0.42 / 0.58)
    assert eps == pytest.approx(0.02 / 0.29)


def test_biconic_angles_rejects_degenerate_geometry():
    assert biconic_angles(0.58, 1.5, 0.0, 0.42) is None      # no fore length
    assert biconic_angles(0.58, 1.5, 0.5, 0.0) is None       # no break
    assert biconic_angles(0.58, 1.5, 0.5, 0.58) is None      # break == base
    assert biconic_angles(0.58, 1.5, 1.5, 0.42) is None      # fore == total (no aft)


def test_slender_biconic_beta_lands_in_the_rv_decade():
    """End to end: a C-HGB-class biconic (base 0.58 m, 450 kg) must give a
    physically plausible slender-RV β, not the pure-Newtonian anomaly."""
    D, mass = 0.58, 450.0
    A = math.pi * (D / 2) ** 2
    th1, th2, br, eps = biconic_angles(D, 1.5, 0.5, 0.42, nose_radius_m=0.02)
    for M in (8.0, 10.0, 12.0):
        c = cd_biconic_hypersonic(th1, th2, br, eps, mach=M)
        beta = mass / (c["total"] * A)
        assert 5_000 < beta < 60_000, f"M{M}: beta {beta:,.0f}"
