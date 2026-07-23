"""Boundary-layer transition gate (heating.transition_factor) — verified
against Kuntz 1999 Table 1 (AIAA 99-3460), the IRV-2 CFD case.

The gate places transition IN THE TRAJECTORY (when), not on the body (where),
on the freestream Reynolds number based on nose radius, Re_Rn = ρ·V·R_n/μ.
Kuntz Table 1 tabulates the flow state per trajectory point; the criterion
must reproduce it: laminar high, transitional through the middle, fully
turbulent low.  The turbulent flux factor stays in the cited 3–5× band.
"""

import numpy as np

import heating

_RN = 0.01905     # m — IRV-2 nose radius

# Kuntz 1999 Table 1: (pt, altitude_m, V_m_s, T_K, rho, state) — state is the
# paper's "Trans. Location" column: 'LAM' laminar, 'trans' a finite on-body x
# (transition front present), 'TURB' fully turbulent.
_KUNTZ = [
    (15, 13962, 5631.8, 216.65, 2.2786e-1, 'LAM'),
    (16, 12748, 5519.6, 216.65, 2.7946e-1, 'trans'),
    (17, 11528, 5401.2, 216.65, 3.3743e-1, 'trans'),
    (18, 10309, 5277.1, 221.31, 3.9840e-1, 'trans'),
    (19,  7892, 5014.3, 236.86, 5.3196e-1, 'trans'),
    (20,  5536, 4736.5, 252.11, 6.9366e-1, 'TURB'),
]


def _state_from_Re(Re):
    if Re >= heating.RE_RN_FULLY_TURBULENT:
        return 'TURB'
    if Re >= heating.RE_RN_TRANSITION_ONSET:
        return 'trans'
    return 'LAM'


def test_gate_reproduces_kuntz_table1():
    """The Re_Rn criterion (with Kuntz's own ρ, T — a clean criterion check,
    no atmosphere-model difference) must match the tabulated flow state."""
    for pt, alt, V, T, rho, state in _KUNTZ:
        Re = rho * V * _RN / heating._sutherland_mu(T)
        got = _state_from_Re(Re)
        assert got == state, (
            f"pt{pt} ({alt/1e3:.1f} km): Re_Rn {Re:.2e} → {got}, Kuntz says {state}")


def test_onset_and_fully_bracket_the_data():
    # Onset sits between the last laminar and first transitional point.
    Re15 = 2.2786e-1 * 5631.8 * _RN / heating._sutherland_mu(216.65)
    Re16 = 2.7946e-1 * 5519.6 * _RN / heating._sutherland_mu(216.65)
    assert Re15 < heating.RE_RN_TRANSITION_ONSET <= Re16
    # Fully-turbulent sits between the last transitional and first turbulent.
    Re19 = 5.3196e-1 * 5014.3 * _RN / heating._sutherland_mu(236.86)
    Re20 = 6.9366e-1 * 4736.5 * _RN / heating._sutherland_mu(252.11)
    assert Re19 < heating.RE_RN_FULLY_TURBULENT <= Re20


def test_factor_is_laminar_1_and_clamped_to_band():
    # High altitude / thin air → laminar, factor 1.
    f, Re, st = heating.transition_factor(
        np.array([1e-4]), np.array([6000.0]), np.array([70e3]), _RN)
    assert st == 'laminar' and abs(float(np.max(f)) - 1.0) < 1e-9
    # Deep turbulent → factor approaches but never exceeds the 5× ceiling.
    f2, Re2, st2 = heating.transition_factor(
        np.array([1.0]), np.array([5000.0]), np.array([3e3]), _RN)
    assert st2 == 'turbulent'
    assert 3.0 <= float(np.max(f2)) <= 5.0 + 1e-9


def test_zero_nose_radius_is_laminar():
    f, Re, st = heating.transition_factor(
        np.array([0.5]), np.array([5000.0]), np.array([5e3]), 0.0)
    assert st == 'laminar' and float(np.max(f)) == 1.0


def test_thresholds_drive_the_gate():
    import thresholds
    thresholds.reset(); thresholds.apply()
    rho, V, alt = np.array([0.3]), np.array([5000.0]), np.array([12e3])
    base = heating.transition_factor(rho, V, alt, _RN)[2]
    # Push the onset far above this point's Re → forced laminar.
    thresholds.set_override("re_rn_transition_onset", 5.0e7)
    thresholds.set_override("re_rn_fully_turbulent", 6.0e7)
    thresholds.apply()
    forced = heating.transition_factor(rho, V, alt, _RN)[2]
    assert forced == 'laminar'
    thresholds.reset(); thresholds.apply()
