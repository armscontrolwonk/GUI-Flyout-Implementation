"""
Standard atmosphere model (COESA 1976) matching MATLAB's atmoscoesa,
extended above 86 km using US Standard Atmosphere 1976 Tables I/II
reference points (geometric altitude).
Returns temperature (K), pressure (Pa), density (kg/m^3), speed of sound (m/s).
"""

import numpy as np

# Layer base altitudes (m), temperatures (K), lapse rates (K/m)
_LAYERS = [
    (0,      288.15, -0.0065),
    (11000,  216.65,  0.0),
    (20000,  216.65,  0.001),
    (32000,  228.65,  0.0028),
    (47000,  270.65,  0.0),
    (51000,  270.65, -0.0028),
    (71000,  214.65, -0.002),
    (86000,  186.87,  0.0),
]

# Upper-atmosphere reference points from US Std Atm 1976 Tables I/II
# (geometric altitude in metres; kinetic T in K; P in Pa; ρ in kg/m³).
# Above 86 km the mean molecular weight varies, so ρ and P are taken
# directly from the standard rather than derived from a fixed gas
# constant.  Between adjacent rows we interpolate ρ and P exponentially
# (constant scale height in each band) and T linearly.
_UPPER_REF = [
    # (h_m,        T_K,    P_Pa,        rho_kg_m3)
    (    86_000, 186.87,  3.7338e-1,   6.958e-6),
    (    91_000, 186.87,  1.5381e-1,   2.860e-6),
    (   100_000, 195.08,  3.2011e-2,   5.604e-7),
    (   110_000, 240.00,  7.1042e-3,   9.708e-8),
    (   120_000, 360.00,  2.5382e-3,   2.222e-8),
    (   150_000, 634.39,  4.5422e-4,   2.076e-9),
    (   200_000, 854.56,  8.4736e-5,   2.541e-10),
    (   300_000, 976.01,  8.7704e-6,   1.916e-11),
    (   500_000, 999.24,  3.0236e-7,   5.215e-13),
    ( 1_000_000, 1000.00, 7.5138e-9,   3.561e-15),
]

_G0 = 9.80665      # m/s^2
_R  = 287.05287    # J/(kg·K)
_GAMMA = 1.4
_P0 = 101325.0     # Pa

def _base_pressure(h_base, T_base, lapse, P_base):
    """Not used directly; pressures at layer bases are precomputed."""
    pass

# Precompute pressures at each layer base
_P_BASE = [_P0]
for i in range(1, len(_LAYERS)):
    h0, T0, L0 = _LAYERS[i-1]
    h1, T1, L1 = _LAYERS[i]
    dh = h1 - h0
    P0 = _P_BASE[-1]
    if abs(L0) < 1e-12:
        P1 = P0 * np.exp(-_G0 * dh / (_R * T0))
    else:
        P1 = P0 * (T0 / (T0 + L0 * dh)) ** (_G0 / (_R * L0))
    _P_BASE.append(P1)


def atmosphere(altitude_m):
    """
    COESA 1976 standard atmosphere, extended to 1000 km.

    For 0 ≤ h ≤ 86 km the canonical COESA 1976 layered formulation
    (lapse-rate piecewise model) is used.  For 86 km < h ≤ 1000 km
    values are piecewise-interpolated from US Standard Atmosphere
    1976 reference points (exponential interpolation for ρ and P,
    linear for T).  Altitudes are clamped to 0–1000 km; above 1000 km
    the returned density (~3.6e-15 kg/m³) is effectively zero for
    drag purposes.

    Parameters
    ----------
    altitude_m : float or array-like
        Geometric altitude in metres (clamped to 0–1 000 000 m).

    Returns
    -------
    T : temperature (K)
    P : pressure (Pa)
    rho : density (kg/m^3)
    a : speed of sound (m/s)
    """
    scalar = np.ndim(altitude_m) == 0
    h = np.atleast_1d(np.asarray(altitude_m, dtype=float))
    h = np.clip(h, 0.0, 1_000_000.0)

    T   = np.zeros_like(h)
    P   = np.zeros_like(h)
    rho = np.zeros_like(h)

    # Lower atmosphere (0–86 km): COESA 1976 layered model.
    for i in range(len(_LAYERS)):
        h_base, T_base, lapse = _LAYERS[i]
        # Restrict to this layer's valid altitude band only so that
        # T never goes negative (which would produce NaN in the power formula).
        h_ceil = _LAYERS[i + 1][0] if i + 1 < len(_LAYERS) else 86000.0
        mask = (h >= h_base) & (h <= h_ceil)
        dh = h[mask] - h_base
        T[mask] = T_base + lapse * dh
        P_base = _P_BASE[i]
        if abs(lapse) < 1e-12:
            P[mask] = P_base * np.exp(-_G0 * dh / (_R * T_base))
        else:
            P[mask] = P_base * (T_base / T[mask]) ** (_G0 / (_R * lapse))
        rho[mask] = P[mask] / (_R * T[mask])

    # Upper atmosphere (86 km < h ≤ 1000 km): piecewise interpolation
    # between US Std Atm 1976 reference points.
    for i in range(len(_UPPER_REF) - 1):
        h0, T0, P0, rho0 = _UPPER_REF[i]
        h1, T1, P1, rho1 = _UPPER_REF[i + 1]
        mask = (h > h0) & (h <= h1)
        if not np.any(mask):
            continue
        dh    = h[mask] - h0
        span  = h1 - h0
        T[mask]   = T0 + (T1 - T0) * (dh / span)
        H_rho     = span / np.log(rho0 / rho1)
        rho[mask] = rho0 * np.exp(-dh / H_rho)
        H_P       = span / np.log(P0 / P1)
        P[mask]   = P0 * np.exp(-dh / H_P)

    a = np.sqrt(_GAMMA * _R * T)

    if scalar:
        return float(T[0]), float(P[0]), float(rho[0]), float(a[0])
    return T, P, rho, a


def speed_of_sound(altitude_m):
    """Speed of sound (m/s) at given altitude."""
    _, _, _, a = atmosphere(altitude_m)
    return a


def dynamic_pressure(velocity_ms, altitude_m):
    """Aerodynamic dynamic pressure q = 0.5 * rho * v^2 (Pa)."""
    _, _, rho, _ = atmosphere(altitude_m)
    return 0.5 * rho * velocity_ms ** 2
