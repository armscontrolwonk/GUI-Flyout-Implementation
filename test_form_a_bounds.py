"""Form A ablator BOUNDING tests — Stardust and Hayabusa recovered capsules.

The ablator verdict no longer computes a recession point-estimate (it compares
load against a flight record; the survival check is the burn-through BOUND at
the most optimistic cited H_eff — see METHODS §13.6).  So these capsule tests
now validate two things, both still bounds, not fits:

1. THE CONSERVATIVE δ STILL BOUNDS.  The reported δ band's NOMINAL edge (the
   conservative-low H_eff, the same value the old point-estimate used) must
   still predict AT LEAST the measured recession — the screen may over-predict,
   never under-predict.  Direction is "predicted >= measured", NOT "==":
   equilibrium-style ablation chemistry OVER-predicts recession (Hayabusa
   calc/measured ~= 3x, Suzuki JSR 10.2514/1.A32549; Stardust 51-61% over near
   the stagnation core — firsthand Kontinos & Stackpoole AIAA 2008-1197 Table 1,
   from Stackpoole et al. AIAA 2008-1202); that conservatism exceeds the
   radiative-gas heating the convective-only model omits above ~9 km/s, so the
   biases net to over-prediction.  Do NOT "fix" it by raising H_eff.

2. THE TRIPWIRE MUST NOT FIRE for a survived capsule.  Both Stardust and
   Hayabusa were recovered intact, so the burn-through bound (shield consumed
   even at the optimistic H_eff) must be FALSE against a realistic shield depth.
   The complementary check: a genuinely under-shielded case DOES trip.

Each case reconstructs the documented stagnation heat pulse (half-sine
calibrated to the cited peak flux + integrated load, at the real capsule nose
radius) and runs it through the SAME heating.heating_figure_of_merit code path
the GUI uses.

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


def _run_recession(material, q_peak_MW, tau_s, V_entry_km_s, nose_radius_m,
                   depth_m=0.0):
    q_peak = q_peak_MW * 1e6
    V = V_entry_km_s * 1e3
    t, rho, Varr = _half_sine_pulse(q_peak, tau_s, V, nose_radius_m)
    alt = np.linspace(80e3, 20e3, t.size)     # descending; unused by recession, kept realistic
    rng = np.linspace(0.0, 1e6, t.size)
    fom = heating.heating_figure_of_merit(
        t, rho, Varr, alt, rng,
        nose_radius_m=nose_radius_m, emissivity=0.90, material=material,
        recession_depth_m=depth_m)
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

# Hayabusa: FIRSTHAND reconstructed environment (Suzuki et al., JSR 51(1) 2014,
#   DOI 10.2514/1.A32549, PDF in repo data/): peak convective 5.3 MW/m^2 at 70 s
#   (converged CFD/SCMA2; radiative ~1 MW/m^2 peak is additional but our model is
#   convective-only — omitting it only SHRINKS predicted recession, i.e. makes the
#   bound harder, so it is safe to leave out).  Sanity: T_eq at 5.3 MW/m^2 /
#   eps 0.9 is ~3190 K vs the paper's calculated ~3200 K peak surface temperature.
#   Pulse duration: LABELED ESTIMATE 60 s from the paper's heating window
#   (temperature rise starts ~40 s, peak ~70 s, radiative-cooling decay after;
#   CFD points span 55-80 s) — the bound also holds at tau 20-30 s.
_HAYABUSA_QPK_MW = 5.3
_HAYABUSA_TAU = 60.0
_HAYABUSA_RN = 0.20          # ~0.4 m base dia sphere-cone
_HAYABUSA_MEAS_MM = 0.3      # measured max recession, laser scan, error <10% (FIRSTHAND)


def test_stardust_bound():
    """PICA conservative δ must bound Stardust's measured 5.7 mm (Core 1) from
    above, AND the burn-through tripwire must NOT fire (the capsule survived)."""
    # Realistic PICA forebody thickness for Stardust ≈ 5.8 cm (recovered).
    fom, rec = _run_recession("pica", _STARDUST_QPK_MW, _STARDUST_TAU, 12.9,
                              _STARDUST_RN, depth_m=0.058)
    pred_mm = rec["delta_nominal_cm"] * 10.0    # conservative-low H_eff edge
    ratio = pred_mm / _STARDUST_MEAS_MM
    print(f"[stardust_bound] conservative δ {pred_mm:.2f} mm vs measured "
          f"{_STARDUST_MEAS_MM} mm -> ratio {ratio:.2f}x; "
          f"tripwire={rec['burnthrough_bound']} "
          f"(load {fom['integrated_load_MJ_m2']:.0f} MJ/m^2)")
    assert pred_mm >= _STARDUST_MEAS_MM, (
        f"BOUND VIOLATED: conservative δ {pred_mm:.2f} mm < measured "
        f"{_STARDUST_MEAS_MM} mm. Broken Q pipeline or bad H_eff (see docstring).")
    assert ratio > 1.0
    assert rec["burnthrough_bound"] is False, (
        "TRIPWIRE FIRED for the recovered Stardust capsule — the burn-through "
        "bound must not trip for a survived flight.")


def test_hayabusa_bound():
    """Carbon-phenolic conservative δ must bound Hayabusa's measured ~0.3 mm,
    AND the tripwire must NOT fire (the capsule survived)."""
    fom, rec = _run_recession("carbon_phenolic", _HAYABUSA_QPK_MW, _HAYABUSA_TAU,
                              12.2, _HAYABUSA_RN, depth_m=0.02)
    pred_mm = rec["delta_nominal_cm"] * 10.0
    ratio = pred_mm / _HAYABUSA_MEAS_MM
    print(f"[hayabusa_bound] conservative δ {pred_mm:.2f} mm vs measured "
          f"{_HAYABUSA_MEAS_MM} mm -> ratio {ratio:.2f}x; "
          f"tripwire={rec['burnthrough_bound']} "
          f"(load {fom['integrated_load_MJ_m2']:.0f} MJ/m^2)")
    assert pred_mm >= _HAYABUSA_MEAS_MM, (
        f"BOUND VIOLATED: conservative δ {pred_mm:.2f} mm < measured "
        f"{_HAYABUSA_MEAS_MM} mm. Halt and investigate (see docstring).")
    assert ratio > 1.0
    assert rec["burnthrough_bound"] is False, (
        "TRIPWIRE FIRED for the recovered Hayabusa capsule.")


def test_family_generic_carbon_carries_the_record():
    """Regression: the family-generic 'carbon_ablator' entry must carry the
    SAME graphite-family flight record and optimistic H_eff bound as
    'carbon_carbon' (records are family-level, METHODS §13.6).  When these
    were unset, the burn-through BOUND silently collapsed to the conservative
    nominal H_eff — the banned point-estimate — and a C-HGB-class carbon nose
    flying ~97% of the Reentry-F load read false-red."""
    m = heating.TPS_MATERIALS["carbon_ablator"]
    cc = heating.TPS_MATERIALS["carbon_carbon"]
    assert m["demonstrated_load_MJ_m2"] == cc["demonstrated_load_MJ_m2"] == 3870
    assert m["H_eff_bound_MJ_kg"] == cc["H_eff_bound_MJ_kg"] == 175
    assert "Reentry-F" in m["demonstrated_load_source"]

    # The C-HGB-class case: ~13.6 MW/m² peak, ablating load just under the
    # Reentry-F record, on a 2 cm generic-carbon tip.  Survives with the load
    # stated against the record; the tripwire must NOT fire.
    fom, rec = _run_recession("carbon_ablator", 13.6, 435.0, 5.2, 0.02)
    print(f"[chgb_regression] load {rec['load_MJ_m2']:.0f} MJ/m^2 = "
          f"{rec['load_fraction']:.0%} of record; "
          f"tripwire={rec['burnthrough_bound']}")
    assert rec["burnthrough_bound"] is False, (
        "TRIPWIRE FIRED at a Reentry-F-class load on a carbon nose — the "
        "bound must use the cited optimistic H_eff, not the nominal.")
    assert rec["load_fraction"] is not None and rec["load_fraction"] < 1.1
    import survivability_report as sr
    regime = sr._ablator_regime(rec)
    assert regime["status"] == "survive"


def test_sirca_h_eff_from_tpsx_curve():
    """SIRCA-14A H_eff wired from the TPSX id-41 pressure curve (22.5–40.5
    kPa): nominal 37.4 MJ/kg = the curve's high-pressure MINIMUM (still
    labeled optimistic for steep entries — RV stagnation pressures exceed the
    curve's 0.4 atm ceiling and Q* falls with pressure); bound 165 MJ/kg =
    top of the mid-pressure cluster, with the 2.1 GJ/kg recession→0 artifact
    point excluded (BENCHMARKING documents the exclusion).  The bound is no
    longer collapsed onto the nominal, so the SIRCA tripwire is a real bound
    again."""
    m = heating.TPS_MATERIALS["sirca"]
    assert m["H_eff_MJ_kg"] == 37.4
    assert m["H_eff_bound_MJ_kg"] == 165.0
    assert m["H_eff_bound_MJ_kg"] > m["H_eff_MJ_kg"]
    assert m["density_kg_m3"] == 224.0 and m["c_J_kgK"] == 1200.0


def test_tripwire_fires_when_undershielded():
    """The complement: a genuinely under-shielded ablator DOES trip the bound.
    Stardust's load through a 2 mm PICA skin is consumed even at the optimistic
    H_eff — burn-through bound must be True."""
    fom, rec = _run_recession("pica", _STARDUST_QPK_MW, _STARDUST_TAU, 12.9,
                              _STARDUST_RN, depth_m=0.002)
    assert rec["burnthrough_bound"] is True, (
        "Tripwire failed to fire for a 2 mm shield under Stardust's load.")
    assert fom.get("compromise") is not None


if __name__ == "__main__":
    test_stardust_bound()
    test_hayabusa_bound()
    test_tripwire_fires_when_undershielded()
    print("Form A bound tests passed.")
