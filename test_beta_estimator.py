"""Blunted-cone hypersonic Cd build-up (booster_models.cd_cone_hypersonic).

The fault this guards (the memo's "anomalous ballistic coefficient"): the
Estimate Object β dialog computed Cd from Newtonian PRESSURE drag alone,
2·sin²θ.  On a slender cone that term is nearly zero — at θ = 5.25° it is
0.017, while skin friction alone contributes ~0.013 and base drag ~0.014 —
so the estimator under-counted Cd by ~4× and emitted β ≈ 95,000 kg/m² for a
SWERVE-geometry cone, an order of magnitude above any physical slender-RV
value.  The memo localized the fault to the m/(Cd·A) denominator; the cause
was the MISSING VISCOUS/BASE TERMS, not a units slip (the area was correct).

Discipline note: every assertion here is a physics identity, a geometry
identity, or an order-of-magnitude sanity band.  Nothing is pinned to a
flight outcome — the estimator must NOT be tuned to reproduce any particular
vehicle's reconstructed β (that would launder a fit into a physics label).
"""

import math

import pytest

from booster_models import (cd_blunted_cone_newtonian, cd_cone_hypersonic,
                            CONE_CF_TURBULENT)


# ── component identities ────────────────────────────────────────────────────
def test_sharp_cone_pressure_is_exact_newtonian():
    for th in (5.0, 5.25, 10.0, 20.0, 40.0):
        want = 2.0 * math.sin(math.radians(th)) ** 2
        assert cd_blunted_cone_newtonian(th, 0.0) == pytest.approx(want)
        assert cd_cone_hypersonic(th, 0.0)["pressure"] == pytest.approx(want)


def test_friction_is_cf_times_exact_frustum_wetted_ratio():
    """S_wet/A_base = (1 − ε²cos²θ)/sinθ — pure geometry, no free constant
    beyond Cf itself."""
    for th, ep in ((5.25, 0.0), (5.25, 0.07), (15.0, 0.3), (30.0, 0.6)):
        c = cd_cone_hypersonic(th, ep)
        thr = math.radians(th)
        want = (1.0 - (ep * math.cos(thr)) ** 2) / math.sin(thr)
        assert c["swet_ratio"] == pytest.approx(want)
        assert c["friction"] == pytest.approx(CONE_CF_TURBULENT * want)


def test_base_term_is_the_vacuum_base_limit():
    """Cd_base = 2/(γM²): the wake cannot push on the base harder than
    removing ambient pressure entirely."""
    for M in (8.0, 10.0, 14.0):
        c = cd_cone_hypersonic(20.0, 0.2, mach=M)
        assert c["base"] == pytest.approx(2.0 / (1.4 * M * M))
    # below the hypersonic-form floor the Mach is clamped, as Hoerner's C_p•
    assert cd_cone_hypersonic(20.0, 0.2, mach=1.0)["base"] == \
        pytest.approx(2.0 / (1.4 * 9.0))


def test_total_is_the_sum_of_its_parts():
    c = cd_cone_hypersonic(12.0, 0.15, mach=11.0)
    assert c["total"] == pytest.approx(c["pressure"] + c["friction"] + c["base"])


# ── the fault itself ────────────────────────────────────────────────────────
def test_slender_cone_beta_leaves_the_anomalous_decade():
    """A 300 kg, 0.476 m, 5.25° cone (SWERVE-class geometry, used here as a
    geometry fixture, not a flight target).  Pressure-only gave β ≈ 95,000
    kg/m² — the memo's anomaly.  With friction and base drag carried, β must
    land in the physically plausible slender-RV decade.  The band is wide on
    purpose: it rejects the failure mode without pinning a preferred value."""
    A = math.pi * (0.476 / 2.0) ** 2
    bad = 300.0 / (cd_blunted_cone_newtonian(5.25, 0.07) * A)
    assert bad > 90_000                       # the bug, reproduced
    for M in (8.0, 10.0, 12.0, 14.0):
        c = cd_cone_hypersonic(5.25, 0.07, mach=M)
        beta = 300.0 / (c["total"] * A)
        assert 15_000 < beta < 60_000, f"M{M}: beta {beta:,.0f}"


def test_slender_cone_is_friction_dominated_not_pressure_dominated():
    """The physical signature the old estimator missed: at θ ≈ 5° the viscous
    + base terms EXCEED the Newtonian pressure term."""
    c = cd_cone_hypersonic(5.25, 0.07, mach=10.0)
    assert c["friction"] + c["base"] > c["pressure"]


# ── blunt RVs must be left essentially alone ────────────────────────────────
def test_blunt_cones_are_pressure_dominated_and_barely_move():
    """The fix must land only where the bug lives: for the table's blunt-RV
    range the added terms are a small perturbation, so every β the estimator
    ever produced for a ballistic RV stays essentially unchanged."""
    for th, ep, tol in ((20.0, 0.2, 0.10), (25.0, 0.5, 0.06), (40.0, 0.2, 0.03)):
        c = cd_cone_hypersonic(th, ep, mach=10.0)
        assert (c["total"] / c["pressure"] - 1.0) < tol, (th, ep)


def test_wing_area_adds_friction_and_lowers_beta():
    """Level-2 advisory half: a winged vehicle is draggier than the bare cone,
    so its estimated β drops.  The wing term is Cf·2·(S_w/A_base) (both faces),
    zero without wings — so a non-winged estimate is byte-identical."""
    bare = cd_cone_hypersonic(5.25, 0.07, mach=10.0)
    assert bare["wing"] == 0.0
    assert bare["total"] == pytest.approx(bare["pressure"] + bare["friction"]
                                          + bare["base"])
    winged = cd_cone_hypersonic(5.25, 0.07, mach=10.0, wing_area_ratio=1.5)
    assert winged["wing"] == pytest.approx(CONE_CF_TURBULENT * 2.0 * 1.5)
    assert winged["total"] > bare["total"]        # draggier → lower β
    # cone terms unchanged by the wing
    for key in ("pressure", "friction", "base"):
        assert winged[key] == pytest.approx(bare[key])


def test_monotonicity():
    """Total Cd falls with Mach (base term) and rises with half-angle."""
    m_prev = None
    for M in (8.0, 10.0, 12.0, 16.0):
        tot = cd_cone_hypersonic(10.0, 0.1, mach=M)["total"]
        assert m_prev is None or tot < m_prev
        m_prev = tot
    t_prev = None
    for th in (5.0, 10.0, 20.0, 30.0, 40.0):
        tot = cd_cone_hypersonic(th, 0.1, mach=10.0)["total"]
        assert t_prev is None or tot > t_prev
        t_prev = tot
