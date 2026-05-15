"""
Standard atmosphere.  Returns temperature (K), pressure (Pa),
density (kg/m^3), speed of sound (m/s).

Default model: NRLMSISE-00 at mean conditions (F10.7=150, Ap=4,
vernal equinox, noon UT, 0°N 0°E) via pymsis, precomputed into a
dense lookup table for fast per-call interpolation.  Falls back
automatically to US Standard Atmosphere 1976 (extended to 1000 km)
if pymsis is not installed.

To switch model or override solar / date parameters call
configure_atmosphere() before running trajectories.  All callers use
atmosphere(altitude_m) — the signature never changes.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_G0    = 9.80665      # m/s²
_R     = 287.05287    # J/(kg·K) — dry air
_GAMMA = 1.4
_P0    = 101325.0     # Pa

# ---------------------------------------------------------------------------
# US Standard Atmosphere 1976 — fallback and lower-atmosphere reference
# ---------------------------------------------------------------------------

# Layer base altitudes (m), temperatures (K), lapse rates (K/m)
_LAYERS = [
    (     0, 288.15, -0.0065),
    ( 11000, 216.65,  0.0   ),
    ( 20000, 216.65,  0.001 ),
    ( 32000, 228.65,  0.0028),
    ( 47000, 270.65,  0.0   ),
    ( 51000, 270.65, -0.0028),
    ( 71000, 214.65, -0.002 ),
    ( 86000, 186.87,  0.0   ),
]

# Upper-atmosphere reference points from US Std Atm 1976 Tables I/II
# (geometric altitude m; kinetic T K; P Pa; ρ kg/m³).
_UPPER_REF = [
    (    86_000, 186.87,  3.7338e-1,  6.958e-6 ),
    (    91_000, 186.87,  1.5381e-1,  2.860e-6 ),
    (   100_000, 195.08,  3.2011e-2,  5.604e-7 ),
    (   110_000, 240.00,  7.1042e-3,  9.708e-8 ),
    (   120_000, 360.00,  2.5382e-3,  2.222e-8 ),
    (   150_000, 634.39,  4.5422e-4,  2.076e-9 ),
    (   200_000, 854.56,  8.4736e-5,  2.541e-10),
    (   300_000, 976.01,  8.7704e-6,  1.916e-11),
    (   500_000, 999.24,  3.0236e-7,  5.215e-13),
    ( 1_000_000, 1000.00, 7.5138e-9,  3.561e-15),
]

# Precompute pressures at each COESA layer base
_P_BASE = [_P0]
for _i in range(1, len(_LAYERS)):
    _h0, _T0, _L0 = _LAYERS[_i - 1]
    _h1            = _LAYERS[_i][0]
    _dh            = _h1 - _h0
    _Pb            = _P_BASE[-1]
    if abs(_L0) < 1e-12:
        _P_BASE.append(_Pb * np.exp(-_G0 * _dh / (_R * _T0)))
    else:
        _P_BASE.append(_Pb * (_T0 / (_T0 + _L0 * _dh)) ** (_G0 / (_R * _L0)))


def _atmosphere_std1976(altitude_m):
    """US Std Atm 1976 extended to 1000 km.  Internal; used as fallback."""
    scalar = np.ndim(altitude_m) == 0
    h = np.atleast_1d(np.asarray(altitude_m, dtype=float))
    h = np.clip(h, 0.0, 1_000_000.0)

    T   = np.zeros_like(h)
    P   = np.zeros_like(h)
    rho = np.zeros_like(h)

    for i in range(len(_LAYERS)):
        h_base, T_base, lapse = _LAYERS[i]
        h_ceil = _LAYERS[i + 1][0] if i + 1 < len(_LAYERS) else 86000.0
        mask = (h >= h_base) & (h <= h_ceil)
        dh = h[mask] - h_base
        T[mask] = T_base + lapse * dh
        Pb = _P_BASE[i]
        if abs(lapse) < 1e-12:
            P[mask] = Pb * np.exp(-_G0 * dh / (_R * T_base))
        else:
            P[mask] = Pb * (T_base / T[mask]) ** (_G0 / (_R * lapse))
        rho[mask] = P[mask] / (_R * T[mask])

    for i in range(len(_UPPER_REF) - 1):
        h0, T0, P0, rho0 = _UPPER_REF[i]
        h1, T1, P1, rho1 = _UPPER_REF[i + 1]
        mask = (h > h0) & (h <= h1)
        if not np.any(mask):
            continue
        dh   = h[mask] - h0
        span = h1 - h0
        T[mask]   = T0 + (T1 - T0) * (dh / span)
        rho[mask] = rho0 * np.exp(-dh * np.log(rho0 / rho1) / span)
        P[mask]   = P0  * np.exp(-dh * np.log(P0  / P1 ) / span)

    a = np.sqrt(_GAMMA * _R * T)
    if scalar:
        return float(T[0]), float(P[0]), float(rho[0]), float(a[0])
    return T, P, rho, a


# ---------------------------------------------------------------------------
# NRLMSISE-00 via pymsis — precomputed lookup table
# ---------------------------------------------------------------------------

# Active configuration.  Call configure_atmosphere(**kwargs) to change.
_ATM_CONFIG = {
    'model':   'msis',   # 'msis' or 'std1976'
    'f107':    150.0,    # daily F10.7 solar flux index
    'f107a':   150.0,    # 81-day average F10.7
    'ap':      4.0,      # geomagnetic Ap index (quiet)
    'doy':     80,       # day of year (≈ vernal equinox)
    'ut_sec':  43200,    # UT seconds (noon)
    'lat_deg': 0.0,      # geodetic latitude
    'lon_deg': 0.0,      # geodetic longitude
}

# Precomputed lookup table built by _build_msis_table().
# Keys: 'h_m', 'T', 'log_rho', 'log_P'  (all 1-D numpy arrays, same length).
_ATM_TABLE  = None
_ATM_SOURCE = 'std1976'   # 'msis' or 'std1976' — reflects what's actually active


def _build_msis_table(cfg):
    """Call pymsis once for 0–1000 km at 500 m intervals; return table dict."""
    import pymsis
    from datetime import datetime, timedelta

    epoch   = datetime(2000, 1, 1)
    dt      = epoch + timedelta(days=cfg['doy'] - 1, seconds=cfg['ut_sec'])
    alts_km = np.arange(0.0, 1000.5, 0.5)        # 2001 points, 500 m spacing
    n       = len(alts_km)
    ap_row  = [float(cfg['ap'])] * 7              # pymsis wants 7-element ap

    out = pymsis.calculate(
        [dt] * n,
        [float(cfg['lon_deg'])] * n,
        [float(cfg['lat_deg'])] * n,
        alts_km,
        f107s  = [float(cfg['f107'])]  * n,
        f107as = [float(cfg['f107a'])] * n,
        aps    = [ap_row] * n,
    )

    rho = np.maximum(out[:, pymsis.Variable.MASS_DENSITY].astype(float), 1e-25)
    T   = np.maximum(out[:, pymsis.Variable.TEMPERATURE ].astype(float), 50.0)
    P   = np.maximum(rho * _R * T, 1e-25)   # approximate: fine below 100 km;
                                             # above that P is not used for drag

    return {
        'h_m':    alts_km * 1000.0,
        'T':      T,
        'log_rho': np.log(rho),
        'log_P':   np.log(P),
    }


def _init_atmosphere():
    global _ATM_TABLE, _ATM_SOURCE
    if _ATM_CONFIG['model'] != 'msis':
        _ATM_TABLE  = None
        _ATM_SOURCE = 'std1976'
        return
    try:
        _ATM_TABLE  = _build_msis_table(_ATM_CONFIG)
        _ATM_SOURCE = 'msis'
    except Exception:
        _ATM_TABLE  = None
        _ATM_SOURCE = 'std1976'


def configure_atmosphere(**kwargs):
    """
    Configure the atmosphere model and (re)build the lookup table.

    Parameters
    ----------
    model : str
        'msis' (default) or 'std1976'.
    f107, f107a : float
        Daily and 81-day-average F10.7 solar flux.  Default 150.
    ap : float
        Geomagnetic Ap index.  Default 4 (quiet).
    doy : int
        Day of year (1–365).  Default 80 (~vernal equinox).
    ut_sec : float
        Universal time in seconds.  Default 43200 (noon UT).
    lat_deg, lon_deg : float
        Geodetic latitude / longitude for the MSIS evaluation point.
        For trajectory use the launch-site coordinates are a good choice.

    Example — switch to US Std Atm 1976::

        configure_atmosphere(model='std1976')

    Example — real flight date with measured solar indices::

        configure_atmosphere(doy=213, ut_sec=50400,
                             f107=182.4, f107a=175.0, ap=12,
                             lat_deg=28.5, lon_deg=-80.6)
    """
    _ATM_CONFIG.update(kwargs)
    _init_atmosphere()


# Build table on module import (takes ~100 ms with pymsis; silent fallback
# to US Std Atm 1976 if pymsis is unavailable).
_init_atmosphere()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def atmosphere(altitude_m):
    """
    Standard atmosphere model.  Returns (T_K, P_Pa, rho_kg_m3, a_m_s).

    Uses NRLMSISE-00 at mean conditions if pymsis is installed, otherwise
    US Standard Atmosphere 1976 extended to 1000 km.  Both variants clamp
    to 0–1000 km.  Call configure_atmosphere() to change model or solar /
    date parameters.

    Parameters
    ----------
    altitude_m : float or array-like
        Geometric altitude in metres.
    """
    if _ATM_TABLE is not None:
        scalar = np.ndim(altitude_m) == 0
        h = np.atleast_1d(np.asarray(altitude_m, dtype=float))
        h = np.clip(h, 0.0, 1_000_000.0)

        h_tab = _ATM_TABLE['h_m']
        T   = np.interp(h, h_tab, _ATM_TABLE['T'])
        rho = np.exp(np.interp(h, h_tab, _ATM_TABLE['log_rho']))
        P   = np.exp(np.interp(h, h_tab, _ATM_TABLE['log_P']))
        a   = np.sqrt(_GAMMA * _R * T)

        if scalar:
            return float(T[0]), float(P[0]), float(rho[0]), float(a[0])
        return T, P, rho, a

    return _atmosphere_std1976(altitude_m)


def atmosphere_source():
    """Return 'msis' or 'std1976' to indicate which model is active."""
    return _ATM_SOURCE


def speed_of_sound(altitude_m):
    """Speed of sound (m/s) at given altitude."""
    _, _, _, a = atmosphere(altitude_m)
    return a


def dynamic_pressure(velocity_ms, altitude_m):
    """Aerodynamic dynamic pressure q = 0.5 * rho * v² (Pa)."""
    _, _, rho, _ = atmosphere(altitude_m)
    return 0.5 * rho * velocity_ms ** 2
