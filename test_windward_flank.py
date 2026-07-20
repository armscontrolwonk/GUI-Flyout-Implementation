"""Windward-flank heating screening estimate (heating.windward_flank_flux).

Pins the closed-form physics: α=0 reduction to the cited acreage flux, the
windward>leeward ordering (AGARD-R-754 / Tracy), the Tracy M=7.95 cone
amplification magnitudes, the stagnation-approach guard, and the band/stamp.
The amplification A(α)=sin(δ+α)/sin(δ) is an inference; these tests fix the
behaviour it must reproduce, not a threshold.
"""

import numpy as np
import heating


def _arc(n=201, h0=60e3, h1=8e3, v0=6000.0, v1=900.0):
    import atmosphere
    t = np.linspace(0, 200, n)
    alt = np.linspace(h0, h1, n)
    V = np.linspace(v0, v1, n)
    rho = np.array([atmosphere.atmosphere(h)[2] for h in alt])
    return t, rho, V, alt, t * 0.0


def test_amplification_alpha0_and_tracy():
    # α=0 reduction is exact: A(0)=1, A_lee(0)=1.
    w, l = heating.windward_amplification(8.0, 0.0)
    assert abs(w - 1.0) < 1e-9 and abs(l - 1.0) < 1e-9
    # Tracy M=7.95 cone (δ=8°): A(12°)≈2.46, A(24°)≈3.81; monotonic.
    a12 = heating.windward_amplification(8.0, 12.0)[0]
    a24 = heating.windward_amplification(8.0, 24.0)[0]
    assert abs(a12 - 2.458) < 0.01
    assert abs(a24 - 3.808) < 0.01
    assert a24 > a12 > 1.0


def test_windward_gt_leeward_ordering():
    # AGARD/Tracy: windward increase >> leeward, across the AoA band.
    for a in (3.0, 8.0, 15.0, 20.0):
        w, l = heating.windward_amplification(8.0, a)
        assert w > 1.0 > l >= 0.0
    # δ floor guards the δ→0 divergence.
    w_sharp, _ = heating.windward_amplification(1.0, 20.0)
    w_floor, _ = heating.windward_amplification(heating._WINDWARD_DELTA_FLOOR, 20.0)
    assert w_sharp == w_floor          # 1° floored to the guard value


def test_alpha0_flux_equals_acreage_fraction():
    # At α=0 the windward flux must equal BODY_FLUX_FRACTION · body-stagnation
    # flux — byte-consistent with the two-location acreage screen.
    t, rho, V, alt, rng = _arc()
    r = heating.windward_flank_flux(
        t, rho, V, alt, rng, body_radius_m=0.3, flank_half_angle_deg=8.0,
        alpha_band_deg=(0.0, 0.0), body_material="cc_hot_structure")
    q_stag_body = heating._stag_flux(rho, V, 0.3)
    expect_MW = heating.BODY_FLUX_FRACTION * float(np.max(q_stag_body)) / 1e6
    assert abs(r["q_windward_MW_m2"]["lo"] - expect_MW) < 1e-6
    assert abs(r["amplification"]["lo"] - 1.0) < 1e-9


def test_band_stamp_and_op_point():
    t, rho, V, alt, rng = _arc()
    r = heating.windward_flank_flux(
        t, rho, V, alt, rng, body_radius_m=0.3, flank_half_angle_deg=8.0,
        alpha_op_deg=12.0, body_material="cc_hot_structure", nose_radius_m=0.02)
    T = r["T_eq_windward_K"]
    # Band brackets the operating point.
    assert T["lo"] < T["op"] < T["hi"]
    assert "Thompson 1989" in r["thompson_band"]
    # Turbulent + fin-LE severity flags always present.
    joined = " ".join(r["warnings"])
    assert "turbulent" in joined and "Alviani" in joined
    # A cool C/C body across α 5–20° stays inside its limits here.
    assert "within the body limits" in r["verdict"]


def test_stagnation_approach_guard():
    # A very sharp nose + small δ + high-α band drives windward toward the nose
    # stagnation flux → the "outside screening" flag fires.
    t, rho, V, alt, rng = _arc()
    r = heating.windward_flank_flux(
        t, rho, V, alt, rng, body_radius_m=0.02, flank_half_angle_deg=5.0,
        alpha_band_deg=(5.0, 40.0), body_material="cc_hot_structure",
        nose_radius_m=0.30)   # nose blunter than body → low stagnation flux
    assert any("approaches the nose" in w for w in r["warnings"])


def test_verdict_gate_off_by_default():
    # The module ships with the windward overlay OFF (context-only).
    assert heating.WINDWARD_DRIVES_VERDICT is False
    # A hot body over its soak limit produces the exceedance verdict text and
    # margins, but the flag being off is what keeps the report status unchanged.
    t, rho, V, alt, rng = _arc()
    r = heating.windward_flank_flux(
        t, rho, V, alt, rng, body_radius_m=0.1, flank_half_angle_deg=12.0,
        alpha_band_deg=(5.0, 20.0), body_material="silica_tile")  # low limit
    wc = r["criteria"]["windward_surface"]
    assert wc["T_lo_K"] > wc["limit_continuous_K"]        # clearly exceeds
    assert "exceeds the body continuous limit" in r["verdict"]
