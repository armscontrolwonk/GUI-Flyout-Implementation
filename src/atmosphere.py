"""
Standard atmosphere model (COESA 1976) matching MATLAB's atmoscoesa.
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
    COESA 1976 standard atmosphere.

    Parameters
    ----------
    altitude_m : float or array-like
        Geometric altitude in metres (clamped to 0–86 km).

    Returns
    -------
    T : temperature (K)
    P : pressure (Pa)
    rho : density (kg/m^3)
    a : speed of sound (m/s)
    """
    scalar = np.ndim(altitude_m) == 0
    h = np.atleast_1d(np.asarray(altitude_m, dtype=float))
    h = np.clip(h, 0.0, 86000.0)

    T   = np.zeros_like(h)
    P   = np.zeros_like(h)

    for i in range(len(_LAYERS)):
        h_base, T_base, lapse = _LAYERS[i]
        mask = h >= h_base
        dh = h[mask] - h_base
        T[mask] = T_base + lapse * dh
        P_base = _P_BASE[i]
        if abs(lapse) < 1e-12:
            P[mask] = P_base * np.exp(-_G0 * dh / (_R * T_base))
        else:
            P[mask] = P_base * (T_base / T[mask]) ** (_G0 / (_R * lapse))

    rho = P / (_R * T)
    a   = np.sqrt(_GAMMA * _R * T)

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
