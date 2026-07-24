"""Stagnation-point RADIATIVE gas heating — Tauber-Sutton correlations.

Two papers, both read from primary (PDFs in data/):

  * Tauber & Sutton, "Stagnation-Point Radiative Heating Relations for Earth
    and Mars Entries", JSR 28(1):40-42, 1991 — the Earth/air correlation
    wired into heating.radiative_flux().
  * Tauber, Palmer & Prabhu, "Stagnation Point Radiative Heating Relations
    for Venus Entry", NTRS 20120001655 — used here as (a) an exact,
    digitization-free verification that we implement this correlation FAMILY
    correctly (its Table 1 publishes both the CFD values and the fitted-
    equation values), and (b) an INDEPENDENT cross-check of the Pioneer Venus
    carbon-phenolic load record that heating.TPS_MATERIALS relies on.

The load-bearing properties pinned here:
  1. exactly ZERO below 9 km/s, so Thrusty's whole fleet (RVs ~7 km/s,
     HGVs ~5-6 km/s) is numerically untouched by adding radiative heating;
  2. the correlation is CONSERVATIVE-HIGH where Thrusty extrapolates it
     (small, high-altitude capsules) — verified against Stardust's
     flight-derived radiative fraction;
  3. the ablator load-vs-record ladder stays on the CONVECTIVE basis its
     records were derived on (a recovered capsule must not read "beyond the
     flight record").
"""

import numpy as np

import heating

_SG_K = 1.7415e-4          # Sutton-Graves SI constant (matches heating._SG_K)


# ── The correlation itself ──────────────────────────────────────────────────
def test_zero_below_the_correlation_floor():
    """Below 9 km/s the correlation is undefined and radiative heating is
    negligible — it must return exactly zero, so no shipped verdict moves."""
    rho = np.array([1e-4, 1e-3, 1e-2])
    for V_kms in (5.0, 6.0, 7.0, 8.0, 8.999):
        q, info = heating.radiative_flux(rho, np.full(3, V_kms * 1e3), 0.05)
        assert np.all(q == 0.0), f"{V_kms} km/s produced radiative flux"
        assert info["q_rad_peak_MW_m2"] == 0.0


def test_fe_table_endpoints_and_monotonicity():
    """f_E(V) is the paper's Table 1; flux must rise steeply and monotonically
    with speed at fixed density."""
    rho = np.array([2.0e-4])
    last = -1.0
    for V in (9.0e3, 1.0e4, 1.1e4, 1.2e4, 1.4e4, 1.6e4):
        q, _ = heating.radiative_flux(rho, np.array([V]), 1.0)
        assert q[0] > last
        last = q[0]
    # Table ceiling: above 16 km/s f_E saturates at its last tabulated value
    q16, _ = heating.radiative_flux(rho, np.array([1.6e4]), 1.0)
    q18, _ = heating.radiative_flux(rho, np.array([1.8e4]), 1.0)
    assert q18[0] == q16[0]


def test_nose_radius_exponent_caps():
    """The paper caps the exponent a: <=0.6 for 1<=R_n<=2 m, <=0.5 for
    2<R_n<=3 m, and <=1 always.  Flux grows with R_n but sub-linearly."""
    rho, V = np.array([3.0e-4]), np.array([1.1e4])
    q_small, _ = heating.radiative_flux(rho, V, 0.5)
    q_mid, _ = heating.radiative_flux(rho, V, 1.5)
    q_big, _ = heating.radiative_flux(rho, V, 3.0)
    assert q_small[0] < q_mid[0] < q_big[0]
    # sub-linear: doubling R_n must less than double the flux (a < 1)
    q_1, _ = heating.radiative_flux(rho, V, 1.0)
    q_2, _ = heating.radiative_flux(rho, V, 2.0)
    assert q_2[0] < 2.0 * q_1[0]


def test_validity_flag_catches_extrapolation():
    """Inside the stated envelope (V 10-16 km/s, rho 6.66e-5..6.31e-4,
    R_n 0.3-3 m) the result is flagged valid; outside it says so."""
    q, info = heating.radiative_flux(np.array([3.0e-4]), np.array([1.2e4]), 1.0)
    assert info["valid"] and "within" in info["note"]
    # Stardust-class nose radius is below the correlation's 0.3 m floor
    _, info2 = heating.radiative_flux(np.array([2.0e-4]), np.array([1.2e4]), 0.23)
    assert not info2["valid"] and "R_n" in info2["note"]


# ── Verification: the Venus paper's own published table ─────────────────────
# Tauber/Palmer/Prabhu Eq. (2), Pioneer Venus large probe (R_n = 0.363 m).
# Columns: t_s, V_m_s, rho_kg_m3, q_rad_CFD_W_cm2, q_rad_Eq2_W_cm2  (Table 1)
_PV_RN = 0.363
_PV_TABLE = [
    (7.0, 11551, 2.86e-4,  519,  388),
    (7.5, 11475, 6.46e-4,  936,  915),
    (8.0, 11310, 1.39e-3, 1672, 1770),
    (8.6, 10880, 3.26e-3, 2449, 2450),
    (8.8, 10651, 4.24e-3, 2302, 2288),
    (9.2, 10028, 6.92e-3, 1396, 1393),
    (9.5,  9408, 9.64e-3, 1117, 1252),
    (9.7,  8927, 1.18e-2,  990, 1058),
    (10.0, 8134, 1.55e-2,  793,  700),
    (10.4, 7015, 2.13e-2,  374,  318),
]


def _venus_q_rad(V, rho, r_n=_PV_RN):
    """Tauber/Palmer/Prabhu 2012 Eqs. (2a)/(2b), Venus, SI (W/m^2)."""
    if V >= 10028.0:
        return 8.497e-63 * V ** 18 * rho ** 1.2 * r_n ** 0.49
    return 2.195e-22 * V ** 7.9 * rho ** 1.2 * r_n ** 0.49


def test_venus_equations_reproduce_the_papers_own_table():
    """Exact verification of the correlation family: our coding of Eqs. 2a/2b
    must reproduce the paper's own published 'q_rad, Eq. 2' column.  No
    digitization anywhere — both inputs and outputs are tabulated."""
    for t_s, V, rho, _q_cfd, q_eq2 in _PV_TABLE:
        got = _venus_q_rad(V, rho) / 1e4          # W/m^2 -> W/cm^2
        assert abs(got - q_eq2) / q_eq2 < 0.01, (
            f"t={t_s}s: ours {got:.0f} vs paper {q_eq2} W/cm^2")


def test_pioneer_venus_radiative_load_corroborates_the_cp_record():
    """INDEPENDENT cross-check of the weakest ablator record.

    heating.TPS_MATERIALS['carbon_phenolic'] carries a demonstrated load of
    ~60 MJ/m^2 from the Pioneer Venus Large Probe (Cabrera & West 2026,
    figure-integrated +/-25%).  Integrating the RADIATIVE pulse that a
    different NASA team computed for the same probe (Table 1 above, CFD
    column) must land below — but within a factor of ~2 of — that total-load
    record, since Cabrera & West describe the pulse as radiation-heavy.
    A wild mismatch would mean one of the two reconstructions is wrong."""
    ts = np.array([r[0] for r in _PV_TABLE])
    q = np.array([r[3] for r in _PV_TABLE]) * 1e4        # W/cm^2 -> W/m^2
    Q_rad = float(np.trapezoid(q, ts)) if hasattr(np, "trapezoid") \
        else float(np.trapz(q, ts))                       # J/m^2
    Q_rad_MJ = Q_rad / 1e6
    record = heating.TPS_MATERIALS["carbon_phenolic"]["demonstrated_load_MJ_m2"]
    print(f"[pioneer_venus] radiative-only load over the tabulated window "
          f"= {Q_rad_MJ:.0f} MJ/m^2 vs the {record:.0f} MJ/m^2 total-load "
          f"record ({Q_rad_MJ / record:.0%})")
    # Radiative alone cannot exceed the total-load record...
    assert Q_rad_MJ < record
    # ...but for a radiation-heavy CO2 entry it should be a large fraction of
    # it — this is the corroboration.  (The window is truncated at both ends,
    # so the true radiative load is somewhat higher than the integral here.)
    assert Q_rad_MJ > 0.5 * record


# ── Conservatism: verified against Stardust flight reconstruction ───────────
def test_stardust_radiative_is_conservative_high():
    """Kontinos & Stackpoole (AIAA 2008-1197) put Stardust's stagnation
    radiative heating at ~9% of the peak rate.  Tauber-Sutton is an
    EQUILIBRIUM correlation built for blunt bodies (R_n >= 0.3 m); Stardust
    is small (0.23 m) and enters high, where the shock layer is chemically
    nonequilibrium and optically thin.  It must therefore OVER-predict — the
    direction that keeps a screening bound honest.  This test pins the
    direction and the rough magnitude, so a future retune cannot silently
    flip it to under-prediction."""
    q_conv_peak = 9.4e6                       # cited convective peak, W/m^2
    R_n, V = 0.20, 12.9e3                     # Stardust forebody
    # Invert Sutton-Graves for the density at the cited convective peak.
    rho = (q_conv_peak / (_SG_K * V ** 3)) ** 2 * R_n
    q_rad, info = heating.radiative_flux(np.array([rho]), np.array([V]), R_n)
    frac = float(q_rad[0]) / (q_conv_peak + float(q_rad[0]))
    print(f"[stardust_radiative] correlation {q_rad[0]/1e6:.2f} MW/m^2 = "
          f"{frac:.0%} of total vs the ~9% flight-derived fraction "
          f"({frac / 0.09:.1f}x) | {info['note']}")
    assert not info["valid"]                  # correctly flagged extrapolated
    assert frac > 0.09, "correlation must not UNDER-predict the flight value"
    assert frac < 0.60, "over-prediction beyond ~6x would be unusable"


# ── No-regression guards on the shipped fleet + the record ladder ───────────
def test_shipped_fleet_flux_unchanged_by_radiative():
    """A 7 km/s ballistic RV arc must produce byte-identical peak flux and
    load with the radiative term on or off."""
    t = np.linspace(0.0, 60.0, 200)
    V = np.full_like(t, 7.0e3)
    rho = 1e-4 + 3e-3 * np.sin(np.pi * t / 60.0) ** 2
    alt = np.linspace(70e3, 15e3, t.size)
    rng = np.linspace(0.0, 5e5, t.size)
    kw = dict(nose_radius_m=0.05, emissivity=0.9, material="carbon_carbon")
    on = heating.heating_figure_of_merit(t, rho, V, alt, rng,
                                         include_radiative=True, **kw)
    off = heating.heating_figure_of_merit(t, rho, V, alt, rng,
                                          include_radiative=False, **kw)
    assert on["q_peak_MW_m2"] == off["q_peak_MW_m2"]
    assert on["integrated_load_MJ_m2"] == off["integrated_load_MJ_m2"]
    assert on["q_rad_peak_MW_m2"] == 0.0


def test_record_ladder_stays_on_the_convective_basis():
    """The regression this guards: with radiative folded into the ablating
    load, the RECOVERED Stardust capsule reconstructed to 1.26x its OWN
    demonstrated record — a false 'beyond the flight record'.  The ladder
    must stay on the convective basis its records were derived on, so
    Stardust closes at ~1.0x while radiative is still computed."""
    import test_form_a_bounds as T
    t, rho, V = T._half_sine_pulse(T._STARDUST_QPK_MW * 1e6, T._STARDUST_TAU,
                                   12.9e3, T._STARDUST_RN)
    alt = np.linspace(80e3, 20e3, t.size)
    rng = np.linspace(0.0, 1e6, t.size)
    fom = heating.heating_figure_of_merit(
        t, rho, V, alt, rng, nose_radius_m=T._STARDUST_RN, emissivity=0.90,
        material="pica", recession_depth_m=0.058)
    rec = fom["criteria"]["recession"]
    assert fom["q_rad_peak_MW_m2"] > 0.0          # radiative IS computed
    assert 0.9 <= rec["load_fraction"] <= 1.1     # ...but the ladder is clean
    assert rec["burnthrough_bound"] is False
