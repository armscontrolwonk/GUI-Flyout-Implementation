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

Passing within ±15% verifies Thrusty's Sutton-Graves + radiative-equilibrium
chain against an independent, SDIO-era implementation of the same method
family (S-G vs DKR exponent 3 vs 3.15, and atmosphere differences, account
for the residual).  Measured agreement at wiring time: 1.3% (37 km), 9%
(60 km).
"""

import numpy as np

import heating
from atmosphere import atmosphere

_RN = 0.077          # m — Finke's nose radius
_V = 7000.0          # m/s — near-constant above ~30 km at his β

# (altitude_km, Finke Fig. 2 T_eq in K — digitized by eye, ε = 1)
_FINKE_POINTS = [(37.0, 4200.0), (60.0, 2600.0)]


def test_finke_teq_curve():
    for h_km, T_finke in _FINKE_POINTS:
        rho = atmosphere(h_km * 1e3)[2]
        q = heating._stag_flux(np.array([rho]), np.array([_V]), _RN)[0]
        T = (q / heating.SIGMA) ** 0.25          # ε=1, matching "inertialess" T_eq
        ratio = T / T_finke
        assert 0.85 <= ratio <= 1.15, (
            f"T_eq at {h_km} km: Thrusty {T:.0f} K vs Finke ~{T_finke:.0f} K "
            f"(ratio {ratio:.2f}) — outside the ±15% verification band")


def test_finke_ablation_onset_convention_matches():
    # Finke takes the screening "ablation" temperature "arbitrarily as an even
    # 2000 K"; Thrusty's ablator onset (continuous_K) for the phenolic family
    # is the same 2000 K — an independent screening-tier convention match,
    # pinned so a catalog retune surfaces this corroboration for re-check.
    assert heating.TPS_MATERIALS["carbon_phenolic"]["continuous_K"] == 2000
