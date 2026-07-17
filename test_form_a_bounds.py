"""Form A ablator BOUNDING tests — Stardust and Hayabusa recovered capsules.

These are *bounds, not fits*.  Read benchmarks/form_a and BENCHMARKING.md §"Form A
anchors" before touching them, and read the two-sentence warning below.

WHY THE DIRECTION IS "predicted >= measured", NOT "predicted == measured":
Recovered-capsule post-flight analysis found equilibrium-style ablation chemistry
OVER-predicts recession (Hayabusa calc/measured ~= 3x, Suzuki JSR 10.2514/1.A32549;
Stardust 51-61% over at the near-stagnation core and 22-25% at mid-flank — firsthand
Kontinos & Stackpoole AIAA 2008-1197 Table 1, from Stackpoole et al. AIAA 2008-1202).
That chemistry conservatism is larger than the radiative-gas heating the
convective-only model omits above ~9 km/s, so the two biases net to OVER-prediction.
The capsules therefore validate the model only as a lower-bounding sanity check:
the screening recession chain must predict AT LEAST the measured recession.  Do NOT
"fix" the over-prediction by tuning H_eff upward — that is the failure mode the
Form A plan §2 exists to prevent.  The in-envelope TUNING anchor is Reentry-F.

Each case reconstructs the documented stagnation heat pulse (half-sine calibrated to
the documented peak flux + integrated load / design environment, at the real capsule
nose radius) and runs it through the SAME heating.heating_figure_of_merit code path
the GUI uses, then reads criteria.recession.recession_m.  The half-sine is a
transparent stand-in for a full entry-trajectory reconstruction (which is not in the
repo); it is calibrated to reproduce the cited integrated load, not asserted to be the
true trajectory.

Run:  PYTHONPATH=. python -m pytest test_form_a_bounds.py -q
"""

import numpy as np

import heating


_SG_K = 1.7415e-4   # Sutton-Graves SI constant (matches heating._SG_K)


def _half_sine_pulse(q_peak_W_m2, tau_s, V_entry_m_s, nose_radius_m, n=400):
    """Return (t, rho, V) whose Sutton-Graves stagnation flux is a half-sine of
    amplitude q_peak over duration tau at fixed entry velocity.  Inverting
    q = K*sqrt(rho/R_n)*V^3 for rho at constant V reproduces the target flux
    exactly through heating_figure_of_merit's own flux calc, so the test drives
    the real code, not a private reimplementation."""
    t = np.linspace(0.0, tau_s, n)
    q = q_peak_W_m2 * np.sin(np.pi * t / tau_s)
    V = np.full_like(t, V_entry_m_s)
    # rho = (q / (K * V^3))^2 * R_n
    rho = (q / (_SG_K * V ** 3)) ** 2 * nose_radius_m
    return t, rho, V


def _run_recession(material, q_peak_MW, tau_s, V_entry_km_s, nose_radius_m):
    q_peak = q_peak_MW * 1e6
    V = V_entry_km_s * 1e3
    t, rho, Varr = _half_sine_pulse(q_peak, tau_s, V, nose_radius_m)
    alt = np.linspace(80e3, 20e3, t.size)     # descending; unused by recession, kept realistic
    rng = np.linspace(0.0, 1e6, t.size)
    fom = heating.heating_figure_of_merit(
        t, rho, Varr, alt, rng,
        nose_radius_m=nose_radius_m, emissivity=0.90, material=material)
    rec = fom["criteria"].get("recession")
    assert rec is not None, f"no recession criterion for {material}: {fom.get('warnings')}"
    return fom, rec


# --- Anchored / documented environment values (see benchmarks/form_a CSVs) ----
# Stardust: q_peak 9.4 MW/m^2, integrated load Q = 276 MJ/m^2 (heating._BENCHMARKS,
#   'solid').  Half-sine of that peak reproduces Q at tau = Q*pi/(2*q_peak).
#   Conservative for a lower-bound test: Kontinos & Stackpoole AIAA 2008-1197 give
#   an expected upper-bound environment of ~12 MW/m^2 / ~360 MJ/m^2, so if anything
#   the true load was higher and predicted recession would only grow.
_STARDUST_Q_MJ = 276.0
_STARDUST_QPK_MW = 9.4
_STARDUST_TAU = _STARDUST_Q_MJ / (_STARDUST_QPK_MW) * np.pi / 2.0   # ~46 s
_STARDUST_RN = 0.2286        # 60 deg sphere-cone, 0.827 m max dia (AIAA 2008-1197)
# Near-stagnation measured recession: Core 1 = 5.7±0.3 mm, the measured maximum
# (no core exists at the geometric stagnation point — the SRC impacted off-center).
# FIRSTHAND: Kontinos & Stackpoole AIAA 2008-1197 Table 1 (from Stackpoole et al.
# AIAA 2008-1202).  Comparing our stagnation-point prediction against the
# near-stagnation measured max is fair for a bound: stagnation recession >= Core 1
# recession (the paper's own calc: 9.6 mm stagnation vs 8.6 mm Core 1).
_STARDUST_MEAS_MM = 5.7

# Hayabusa: design environment ~15 MW/m^2 peak for ~30 s (Suzuki JSR 10.2514/1.A32549,
#   plan-restated / secondhand); half-sine Q ~= 286 MJ/m^2.
_HAYABUSA_QPK_MW = 15.0
_HAYABUSA_TAU = 30.0
_HAYABUSA_RN = 0.20          # ~0.4 m base dia sphere-cone
_HAYABUSA_MEAS_MM = 0.3      # max measured recession (secondhand)


def test_stardust_bound():
    """PICA screening recession must bound Stardust's measured 5.7 mm (Core 1) from above."""
    fom, rec = _run_recession("pica", _STARDUST_QPK_MW, _STARDUST_TAU, 12.9, _STARDUST_RN)
    pred_mm = rec["recession_m"] * 1e3
    ratio = pred_mm / _STARDUST_MEAS_MM
    print(f"[stardust_bound] predicted {pred_mm:.2f} mm vs measured "
          f"{_STARDUST_MEAS_MM} mm -> ratio {ratio:.2f}x "
          f"(load {fom['integrated_load_MJ_m2']:.0f} MJ/m^2)")
    assert pred_mm >= _STARDUST_MEAS_MM, (
        f"BOUND VIOLATED: predicted {pred_mm:.2f} mm < measured {_STARDUST_MEAS_MM} mm. "
        f"This indicates a broken Q pipeline or bad H_eff, NOT a radiative shortfall "
        f"(see Form A plan §2 / test docstring). Halt and investigate.")
    assert ratio > 1.0


def test_hayabusa_bound():
    """Carbon-phenolic screening recession must bound Hayabusa's measured ~0.3 mm."""
    fom, rec = _run_recession("carbon_phenolic", _HAYABUSA_QPK_MW, _HAYABUSA_TAU, 12.2, _HAYABUSA_RN)
    pred_mm = rec["recession_m"] * 1e3
    ratio = pred_mm / _HAYABUSA_MEAS_MM
    print(f"[hayabusa_bound] predicted {pred_mm:.2f} mm vs measured "
          f"{_HAYABUSA_MEAS_MM} mm -> ratio {ratio:.2f}x "
          f"(load {fom['integrated_load_MJ_m2']:.0f} MJ/m^2)")
    assert pred_mm >= _HAYABUSA_MEAS_MM, (
        f"BOUND VIOLATED: predicted {pred_mm:.2f} mm < measured {_HAYABUSA_MEAS_MM} mm. "
        f"Halt and investigate (see test docstring).")
    assert ratio > 1.0


if __name__ == "__main__":
    test_stardust_bound()
    test_hayabusa_bound()
    print("Form A bound tests passed.")
