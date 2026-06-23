"""Phugoid damping-ratio (ζ) estimator for hypersonic glide RVs.

Design & rationale: docs/damping_estimate_spec.md.  Literature: docs/
cl_margin_references.md.  Prototype numbers: damping_topout.py, cbo_glide_check.py.

Estimates the achievable ζ *ceiling* (limited by the vehicle's lift authority)
with a modeling-uncertainty band.  This is an order-of-~factor-2 estimate keyed
off β, L/D, and a control-surface descriptor — a suggested starting value with
an honest band, NOT a measurement.  ζ below the band is a free design choice;
inside the band is the airframe authority limit; above it is unphysical.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

from atmosphere import atmosphere

# --- physical constants / modelling assumptions (see spec §3) --------------
G0  = 9.80665
RE  = 6.378137e6
CP_MAX          = 1.84    # modified-Newtonian stagnation Cp (γ=1.4, M→∞)
REAL_GAS_DERATE = 0.85    # control-surface effectiveness derate above ~M7
CL_TRIM         = 0.12    # assumed max-L/D trim C_L of a slender glide body
ALPHA_TRIM_DEG  = 8.0     # assumed max-L/D trim α (NASA M=6 biconic; TN D-840)
DELTA_DEFAULT_DEG = 12.0
DELTA_MAX_DEG   = 15.0    # separation-limited usable deflection (Kumar/Stollery)
CL_MAX_CONE     = 0.45    # cone/biconic aerodynamic C_L ceiling (TN D-840/D-4098)
RANGE_KNEE      = 0.5     # usable ΔC_L ≤ 0.5·C_L,trim (~1.5× C_L,opt, ~8% L/D)
ZETA_FLOOR      = 0.05    # passive atmospheric-drag damping floor
ZETA_CAP        = 1.2

# Representative S_flap/S_ref per qualitative tier (used when no explicit ratio)
_TIER_RATIO = {"small": 0.08, "substantial": 0.30}


@dataclass
class EstimateResult:
    zeta: float          # central estimate (achievable ζ ceiling)
    zeta_lo: float       # modeling-uncertainty band, low
    zeta_hi: float       # modeling-uncertainty band, high
    h_eq_km_lo: float    # equilibrium glide altitude across the V sweep
    h_eq_km_hi: float
    margin_M: float      # control-surface C_L margin used
    s_flap_ratio: float  # S_flap/S_ref used (0 for tier-fixed / none)
    mode: str            # "computed" | "tier" | "none" | "ballistic" | "unknown"
    notes: str

    def text(self) -> str:
        return f"ζ ≈ {self.zeta:.2f}  ({self.zeta_lo:.2f}–{self.zeta_hi:.2f})"


def _rho_to_alt(rho_target: float, lo: float = 15000.0, hi: float = 85000.0) -> float:
    """Invert the (monotone-decreasing) atmosphere density profile for altitude."""
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if atmosphere(mid)[2] > rho_target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _scale_height(h: float) -> float:
    rho = atmosphere(h)[2]
    rho2 = atmosphere(h + 100.0)[2]
    drho = (rho2 - rho) / 100.0
    if drho >= 0 or rho <= 0:
        return 7000.0
    return min(max(-rho / drho, 4000.0), 12000.0)


def _zeta_active(beta: float, ld: float, nmax: float, M: float,
                 V: float, h_override: Optional[float] = None):
    """Lift-authority-limited active damping ratio at one glide state.
    Returns (zeta_active, h_eq_m) or None if the state is non-physical."""
    g_eff = G0 - V * V / RE
    if g_eff <= 0:
        return None
    if h_override is not None:
        h_eq = h_override
    else:
        rho_eq = 2.0 * beta * g_eff / (V * V * ld)
        h_eq = _rho_to_alt(rho_eq)
    Hr = _scale_height(h_eq)
    omega = math.sqrt(g_eff / Hr)
    gamma = 2.0 * ld * Hr * G0 / (V * V)
    headroom = (M - 1.0) * g_eff                       # at equilibrium a_L,trim = g_eff
    headroom = min(headroom, max(0.0, nmax * G0 - g_eff))   # structural g cap
    denom = 2.0 * omega * V * gamma
    return (headroom / denom if denom > 0 else 0.0), h_eq


def estimate_damping(beta_kg_m2: float, glider_ld: float, nmax: float = 10.0,
                     control: str = "unknown", flap_area_ratio: float = 0.0,
                     flap_deflection_deg: float = 0.0,
                     v_glide: Optional[float] = None,
                     h_glide: Optional[float] = None) -> EstimateResult:
    """Estimate the glide vehicle's achievable damping ratio ζ (± band).

    beta_kg_m2, glider_ld, nmax : from the RV.
    control                     : 'none'|'small'|'substantial'|'unknown'.
    flap_area_ratio             : S_flap/S_ref; >0 overrides the tier (computed).
    flap_deflection_deg         : usable control deflection; 0 ⇒ 12° default.
    v_glide, h_glide            : flown mid-glide speed (m/s) / altitude (m);
                                  None ⇒ sweep V = 3–5 km/s, derive h_eq.
    """
    beta = float(beta_kg_m2); ld = float(glider_ld)
    if ld <= 0 or beta <= 0:
        return EstimateResult(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, "ballistic",
                              "non-lifting body (L/D = 0): ballistic, ζ not applicable")

    control = (control or "unknown").lower()
    ratio = float(flap_area_ratio or 0.0)
    from_tier = False
    if ratio <= 0.0:
        if control == "none":
            return EstimateResult(0.10, 0.0, 0.20, 31.0, 41.0, 1.0, 0.0, "none",
                                  "no active control authority — ~undamped (skip-glide); "
                                  "weak passive drag damping only")
        if control in _TIER_RATIO:
            ratio = _TIER_RATIO[control]; from_tier = True
        else:  # unknown
            return EstimateResult(0.35, 0.0, 0.75, 31.0, 41.0, 1.0, 0.0, "unknown",
                                  "specify control surfaces (none/small/substantial) "
                                  "or a control-surface area ratio for an estimate")

    # --- computed path ---------------------------------------------------
    delta = flap_deflection_deg if (flap_deflection_deg and flap_deflection_deg > 0) \
        else DELTA_DEFAULT_DEG
    delta = min(delta, DELTA_MAX_DEG)
    a = math.radians(ALPHA_TRIM_DEG); d = math.radians(delta)
    dcl_newton = ratio * CP_MAX * (math.sin(a + d) ** 2 - math.sin(a) ** 2) \
        * math.cos(a) * REAL_GAS_DERATE
    dcl = min(dcl_newton, RANGE_KNEE * CL_TRIM, max(0.0, CL_MAX_CONE - CL_TRIM))
    M = 1.0 + dcl / CL_TRIM

    if v_glide:
        states = [(float(v_glide), h_glide)]
    else:
        states = [(v, None) for v in (3000.0, 3500.0, 4000.0, 4500.0, 5000.0)]

    zs, hs = [], []
    for V, h in states:
        r = _zeta_active(beta, ld, nmax, M, V, h)
        if r is None:
            continue
        z, h_eq = r
        zs.append(ZETA_FLOOR + z); hs.append(h_eq)
    if not zs:
        return EstimateResult(0.35, 0.0, 0.75, 0.0, 0.0, M, ratio, "unknown",
                              "glide state out of range; could not compute")
    zs.sort()
    central = min(zs[len(zs) // 2], ZETA_CAP)
    factor = 1.9 if from_tier else 1.4          # wider band when ratio is a tier guess
    lo = max(0.0, min(zs) / factor)
    hi = min(ZETA_CAP, max(zs) * factor)
    src = f"tier '{control}' (S_flap/S_ref≈{ratio:.2f})" if from_tier \
        else f"S_flap/S_ref={ratio:.2f}"
    note = f"computed: {src}, δ={delta:.0f}°, C_L margin M={M:.2f}"
    if hi >= 0.7:
        note += "; ≥0.7 reaches the textbook well-damped value — free to choose up to 0.7"
    return EstimateResult(central, lo, hi, min(hs) / 1000.0, max(hs) / 1000.0,
                          M, ratio, "tier" if from_tier else "computed", note)


def estimate_for_rv(rv, v_glide=None, h_glide=None) -> EstimateResult:
    """Convenience wrapper taking an RVParams-like object."""
    return estimate_damping(
        beta_kg_m2=getattr(rv, "beta_kg_m2", 0.0),
        glider_ld=getattr(rv, "glider_LD", 0.0),
        nmax=getattr(rv, "glider_pullup_g_max", 10.0),
        control=getattr(rv, "glider_control_surfaces", "unknown"),
        flap_area_ratio=getattr(rv, "glider_flap_area_ratio", 0.0),
        flap_deflection_deg=getattr(rv, "glider_flap_deflection_deg", 0.0),
        v_glide=v_glide, h_glide=h_glide)


if __name__ == "__main__":
    # C-HGB-class quick demo
    for ctrl in ("none", "small", "substantial", "unknown"):
        r = estimate_damping(15000.0, 2.0, control=ctrl)
        print(f"{ctrl:>12}: {r.text():<18} {r.notes}")
    r = estimate_damping(15000.0, 2.0, control="substantial", flap_area_ratio=0.30)
    print(f"\nC-HGB substantial, S_flap/S_ref=0.30: {r.text()}  "
          f"(h_eq {r.h_eq_km_lo:.0f}-{r.h_eq_km_hi:.0f} km, M={r.margin_M:.2f})")
