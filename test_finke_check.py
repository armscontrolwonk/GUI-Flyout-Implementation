"""Verification case — Finke, IDA Paper P-2395 (Sept 1990, SDIO; DTIC
ADA231552, read from primary, in the project archive).

Finke built an independent screening chain with the same architecture as
Thrusty's heating module — trajectory → stagnation flux → "inertialess"
radiative-equilibrium temperature T_eq → per-location scaling → 1-D material
response — using a Detra-Kemp-Riddell-family laminar correlation (q ∝ √ρ
V^3.15, "in numerical agreement with Detra, Kemp, and Riddell as validated
in Perini, 1975") and the 1962 model atmosphere.  His Fig. 2 plots T_eq vs
altitude for a hypothetical high-β ICBM RV:

    R_n = 0.077 m, β ≈ 1500 lb/ft² (≈ 7,320 kg/m²), V_entry ≈ 23,150 ft/s
    (≈ 7.06 km/s), γ = −24.8° at 400 kft, range 10,020 km.

At that β the vehicle barely decelerates above ~30 km, so V ≈ 7.0 km/s is a
fair constant-velocity assumption for the two comparison altitudes.  The two
targets are DIGITIZED BY EYE from Fig. 2 (labeled as such): T_eq ≈ 4,200 K
near 37 km ("20 s before impact" tick) and ≈ 2,600 K near 60 km ("30 s").

Two tiers of comparison, deliberately separated:

1. EXACT (the load-bearing check): Finke prints his correlation in closed
   form, so S-G can be ratioed against it at identical (ρ, V) with NO
   figure-reading in the loop.  Both scale as √ρ, so the ratio depends only
   on velocity: S-G/DKR flux = 1.01 at 3 km/s falling to 0.89 at 7.2 km/s —
   S-G sits mildly on the LOW side of the DKR family at ICBM speeds (−11%
   flux = −3% T at 7 km/s), a documented family spread well inside the
   screening tier's stated uncertainty.
2. FIGURE (context, now tightened): the Fig.-2 T_eq-vs-altitude curve.  The
   targets were PIXEL-TRACED, not eyeballed — tick-calibrated axes, a
   nearest-run curve tracker, frame-line masking (benchmarks/verification/
   digitize_finke_fig2.py + the QC overlay), replacing the earlier ±3–4 km
   eyeball (which on a curve falling ~2%/km injected ±6–8% in T by itself).
   The residual now sits at a steady ~5% (Thrusty reads ~4–6% BELOW the
   trace across 37/60/80 km) — the same SIGN and MAGNITUDE the exact
   correlation ratio predicts (S-G ~3% low on T at 7 km/s), so what's left
   is real method spread plus the 1962-vs-modern atmosphere, not read noise.
   Digitizing collapsed the apparent 9% outlier at 60 km to a consistent
   ~5% across the curve.
"""

import numpy as np

import heating
from atmosphere import atmosphere

_RN = 0.077          # m — Finke's nose radius
_V = 7000.0          # m/s — near-constant above ~30 km at his β

# (altitude_km, Finke Fig. 2 T_eq in K — PIXEL-TRACED, ε = 1; median of ±0.5 km
# around each altitude via digitize_finke_fig2.py, tick-calibrated).
_FINKE_POINTS = [(37.0, 4324.0), (60.0, 3006.0), (80.0, 2076.0)]


def test_finke_correlation_exact():
    """S-G vs Finke's printed DKR-family correlation at identical (ρ, V) —
    digitization-free.  Both ∝ √ρ, so the flux ratio is a pure function of
    velocity; pin the band 0.85–1.05 across 3–7.2 km/s and the known
    low-side value ~0.89 at 7 km/s."""
    BTU = 11356.5                       # W/m² per Btu/ft²/s
    rho0, Rn_ft = 1.225, _RN / 0.3048
    rho = 1e-3                          # arbitrary — cancels in the ratio
    for V_kms in (3.0, 5.0, 6.0, 7.0, 7.2):
        V = V_kms * 1e3
        q_sg = 1.7415e-4 * np.sqrt(rho / _RN) * V ** 3
        q_fk = (865 / np.sqrt(Rn_ft)) * np.sqrt(rho / rho0) \
            * ((V / 0.3048) / 1e4) ** 3.15 * BTU
        r = q_sg / q_fk
        assert 0.85 <= r <= 1.05, (
            f"S-G/DKR flux ratio {r:.3f} at {V_kms} km/s outside the "
            f"documented family band")
    # the ICBM-speed value: S-G reads ~11% below DKR (mildly optimistic side)
    V = 7000.0
    r7 = (1.7415e-4 * np.sqrt(rho / _RN) * V ** 3) / (
        (865 / np.sqrt(Rn_ft)) * np.sqrt(rho / rho0)
        * ((V / 0.3048) / 1e4) ** 3.15 * BTU)
    assert 0.86 <= r7 <= 0.92


def test_finke_teq_curve():
    # Pixel-traced targets (not eyeballed), so the band is tightened to ±8%:
    # Thrusty reads a steady ~4-6% below the trace, matching the exact
    # correlation ratio's sign and size (S-G ~3% low on T at 7 km/s) plus the
    # 1962-vs-modern atmosphere.  A drift outside this band means the S-G
    # constant, the atmosphere, or the trace changed — not read noise.
    for h_km, T_finke in _FINKE_POINTS:
        rho = atmosphere(h_km * 1e3)[2]
        q = heating._stag_flux(np.array([rho]), np.array([_V]), _RN)[0]
        T = (q / heating.SIGMA) ** 0.25          # ε=1, matching "inertialess" T_eq
        ratio = T / T_finke
        assert 0.92 <= ratio <= 1.08, (
            f"T_eq at {h_km} km: Thrusty {T:.0f} K vs Finke trace "
            f"{T_finke:.0f} K (ratio {ratio:.3f}) — outside the ±8% band")


def test_finke_ablation_onset_convention_matches():
    # Finke takes the screening "ablation" temperature "arbitrarily as an even
    # 2000 K"; Thrusty's ablator onset (continuous_K) for the phenolic family
    # is the same 2000 K — an independent screening-tier convention match,
    # pinned so a catalog retune surfaces this corroboration for re-check.
    assert heating.TPS_MATERIALS["carbon_phenolic"]["continuous_K"] == 2000
