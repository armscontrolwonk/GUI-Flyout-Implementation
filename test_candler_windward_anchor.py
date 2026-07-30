"""Candler & Leyva (S&GS 30, 2022) CFD anchor for the windward-flank screen.

An independent, flight-validated check (their US.3D CFD is within ~100 K of
Shuttle thermocouples) on Thrusty's windward acreage model at exactly our use
case: a US hypersonic glider at its glide point.  Flight condition (their
Fig. 1 text): V = 6 km/s, ρ = 1.015e-3 kg/m³ (49.7 km), ε = 0.85, non-
catalytic radiative equilibrium, laminar; α = 14° = L/D_max (their Fig. 2).
Their Fig. 3 lower-surface (windward) centreline temperature plateaus at
≈1150-1200 K over x/l ≈ 0.2-0.9.

FINDING (documented, not fitted): applied to a FLAT-BOTTOM lifting body the
cone-flank model OVER-predicts — windward T_eq ≈ 1900 K vs the CFD ≈1175 K
(×1.65 in T, ×7 in flux).  The cause is structural: BODY_FLUX_FRACTION = 0.13
is a CONE tail/stagnation ratio (Lu/Shi & Zhang 2024 + STS-1), and a flat
lower surface runs ~7× cooler relative to its blunt-scale stagnation than a
cone flank does.  The bias is CONSERVATIVE (screening-safe) but over-flags
lifting bodies; body_form-aware windward heating is future work (see TODO).
0.13 is anchored for its validated CONE domain and is left unchanged here.

These tests pin the anchor NUMBERS so any future body_form-aware fix is a
deliberate, measured change against a recorded target — not silent drift.
"""

import math

import numpy as np
import pytest

import heating

# Candler flight point and the generic-HGV geometry, mapped the way the app
# maps it (trajectory.py: δ = atan((D/2)/L), R_body = D/2).
_RHO, _V, _ALT = 1.015e-3, 6000.0, 49700.0
_L_HGV, _D_HGV = 3.6, 0.9                       # generic HGV proportions
_DELTA = math.degrees(math.atan((_D_HGV / 2) / _L_HGV))    # ≈ 7.1°
_R_BODY = _D_HGV / 2                            # 0.45 m
_ALPHA_LDMAX = 14.0                            # their Fig. 2 L/D_max
_CFD_WINDWARD_K = 1175.0                        # Fig. 3 lower-surface plateau


def _windward(alpha_op):
    n = 5
    t = np.linspace(0, 10, n)
    return heating.windward_flank_flux(
        np.full(n, 0.0) + t, np.full(n, _RHO), np.full(n, _V),
        np.full(n, _ALT), np.linspace(0, 60, n),
        body_radius_m=_R_BODY, nose_radius_m=0.034,
        flank_half_angle_deg=_DELTA, alpha_op_deg=alpha_op,
        alpha_band_deg=(5.0, 20.0), emissivity=0.85, body_material="")


def test_flight_point_is_laminar_like_the_cfd():
    """Candler: Re_θ/M_e < 100 → laminar upper AND lower surface at this
    condition.  Our transition gate must agree (else the anchor compares the
    wrong boundary-layer state)."""
    assert _windward(_ALPHA_LDMAX)["transition_state"] == "laminar"


def test_cone_flank_model_over_predicts_flat_bottom_hgv():
    """The recorded over-prediction: windward T_eq is well above the CFD, in
    the conservative direction.  Pin it as a band so a future fix has a target
    and this can't silently drift."""
    T_op = _windward(_ALPHA_LDMAX)["T_eq_windward_K"]["op"]
    assert T_op > _CFD_WINDWARD_K * 1.4               # clearly over-predicts
    assert 1800.0 < T_op < 2050.0                     # current model ≈1934 K
    # in flux terms the gap is ~7× (the 4th-root compresses it in T)
    flux_ratio = (T_op / _CFD_WINDWARD_K) ** 4
    assert flux_ratio > 4.0


def test_over_prediction_is_not_an_R_body_artefact():
    """T ∝ R_body^(−1/8): even an implausibly large flat-surface curvature
    scale cannot close the gap — so the bias is the acreage FRACTION / cone-
    flank assumption, not the stagnation scale."""
    n = 5
    def T_at(Rb):
        r = heating.windward_flank_flux(
            np.linspace(0, 10, n), np.full(n, _RHO), np.full(n, _V),
            np.full(n, _ALT), np.linspace(0, 60, n),
            body_radius_m=Rb, nose_radius_m=0.034, flank_half_angle_deg=_DELTA,
            alpha_op_deg=_ALPHA_LDMAX, alpha_band_deg=(5.0, 20.0),
            emissivity=0.85, body_material="")
        return r["T_eq_windward_K"]["op"]
    assert T_at(2.0) > _CFD_WINDWARD_K * 1.3         # 4× larger radius: still high


def test_implied_flat_bottom_fraction_is_far_below_the_cone_value():
    """Diagnosis, pinned: the effective acreage fraction Candler implies for a
    flat lower surface is ≈0.018 — ~7× below the shipped cone value 0.13.
    (This is the target for a body_form-aware fix; it does NOT change 0.13,
    which stays anchored for the cone domain.)"""
    q_stag_body = float(heating._stag_flux(_RHO, _V, _R_BODY))
    A = heating.windward_amplification(_DELTA, _ALPHA_LDMAX)[0]
    q_cfd = 0.85 * heating.SIGMA * _CFD_WINDWARD_K ** 4
    frac_eff = q_cfd / (A * q_stag_body)
    assert frac_eff == pytest.approx(0.018, abs=0.006)
    assert heating.BODY_FLUX_FRACTION / frac_eff > 5.0
    # the shipped cone constant is untouched by this anchor
    assert heating.BODY_FLUX_FRACTION == 0.13
