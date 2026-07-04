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

import math
from functools import lru_cache

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
# MIL-STD-210A non-standard atmospheres (hot / cold / polar / tropical)
# ---------------------------------------------------------------------------
# Temperature vs geometric altitude (0–100 kft), ported from PDAS ATMOS
# hotcold.f90 (MODULE MILSTD210A, Ralph Carmichael / public domain) and
# converted to SI (kft→m, °R→K).  Per MIL-STD-210A the PRESSURE equals the
# US Std 1976 value and density follows from the perfect-gas law (ρ = P/RT).
# The tables stop at 100 kft (30.48 km); above that the temperature offset
# from standard is tapered to zero by 32 km so the profile reverts smoothly
# to US Std 1976 aloft (weather variations do not persist into the upper
# atmosphere, and density there is negligible for trajectory work anyway).
_MIL_ALT_M = [0.0, 304.8, 609.6, 914.4, 1219.2, 1524.0, 1828.8, 2133.6, 2438.4, 2743.2, 3048.0, 3352.8, 3657.6, 3962.4, 4267.2, 4572.0, 4876.8, 5181.6, 5486.4, 5791.2, 6096.0, 6400.8, 6705.6, 7010.4, 7315.2, 7620.0, 7924.8, 8229.6, 8534.4, 8839.2, 9144.0, 9448.8, 9753.6, 10058.4, 10363.2, 10668.0, 10972.8, 11277.6, 11582.4, 11887.2, 12192.0, 12496.8, 12801.6, 13106.4, 13411.2, 13716.0, 14020.8, 14325.6, 14630.4, 14935.2, 15240.0, 15544.8, 15849.6, 16154.4, 16459.2, 16764.0, 17068.8, 17373.6, 17678.4, 18288.0, 18897.6, 19507.2, 20116.8, 20726.4, 21336.0, 21945.6, 22555.2, 23164.8, 23774.4, 24384.0, 24993.6, 25603.2, 26212.8, 26822.4, 27432.0, 28041.6, 28651.2, 29260.8, 29870.4, 30480.0]
_MIL_HOT_K = [312.611, 310.5, 308.389, 306.222, 304.056, 301.889, 299.722, 297.5, 295.278, 293.056, 290.889, 288.833, 286.722, 284.611, 282.5, 280.333, 278.167, 276.0, 273.778, 271.611, 269.556, 267.5, 265.389, 263.333, 261.222, 259.111, 257.0, 254.833, 252.667, 250.556, 248.556, 246.556, 244.556, 242.5, 240.5, 238.667, 236.833, 235.0, 233.111, 231.222, 230.5, 230.778, 231.0, 231.222, 231.444, 231.722, 232.0, 232.222, 232.5, 232.778, 233.056, 233.222, 233.333, 233.444, 233.556, 233.667, 233.722, 233.833, 233.944, 234.167, 234.389, 234.611, 234.778, 235.333, 236.111, 236.889, 237.667, 238.444, 239.222, 240.0, 240.889, 241.722, 242.611, 243.556, 244.389, 245.222, 246.056, 247.0, 247.944, 248.944]
_MIL_COLD_K = [222.056, 229.556, 237.056, 244.667, 247.056, 247.056, 247.056, 247.056, 247.056, 247.056, 247.056, 246.611, 244.778, 242.944, 241.111, 239.222, 237.389, 235.5, 233.611, 231.667, 229.778, 227.833, 225.833, 223.889, 221.889, 219.889, 217.889, 215.889, 213.833, 211.722, 209.667, 208.167, 208.167, 208.167, 208.167, 208.167, 208.167, 208.167, 208.167, 208.167, 208.167, 208.167, 208.167, 206.389, 203.556, 200.611, 197.667, 194.667, 191.667, 189.167, 187.111, 185.944, 185.944, 185.944, 185.944, 185.944, 185.944, 185.944, 185.944, 185.944, 187.556, 190.944, 194.056, 196.889, 199.556, 202.0, 203.0, 202.722, 202.444, 202.111, 201.722, 201.278, 200.833, 200.444, 200.0, 199.556, 199.111, 198.667, 198.167, 197.667]
_MIL_POLAR_K = [246.667, 248.333, 250.056, 251.722, 251.944, 251.667, 251.333, 251.056, 250.722, 250.444, 250.0, 248.444, 246.833, 245.278, 243.722, 242.167, 240.556, 239.0, 237.444, 235.833, 234.278, 232.667, 231.111, 229.5, 227.944, 226.333, 224.722, 223.167, 221.556, 219.944, 218.333, 218.056, 217.889, 217.778, 217.611, 217.444, 217.333, 217.167, 217.056, 216.889, 216.722, 216.611, 216.444, 216.333, 216.167, 216.0, 215.889, 215.722, 215.611, 215.444, 215.278, 215.167, 215.0, 214.889, 214.722, 214.556, 214.444, 214.278, 214.167, 213.889, 213.556, 213.278, 213.0, 212.722, 212.444, 212.167, 211.889, 211.611, 211.278, 211.0, 210.722, 210.444, 210.167, 210.167, 210.167, 210.167, 210.167, 210.167, 210.167, 210.167]
_MIL_TROP_K = [305.278, 303.111, 300.944, 298.778, 296.667, 294.5, 292.333, 290.167, 288.0, 285.889, 283.722, 281.556, 279.389, 277.278, 275.111, 272.944, 270.833, 268.667, 266.5, 264.333, 262.222, 260.056, 257.889, 255.778, 253.611, 251.5, 249.333, 247.167, 245.389, 242.889, 240.778, 238.611, 236.444, 234.333, 232.167, 230.056, 227.889, 225.778, 223.667, 221.556, 219.5, 217.444, 215.389, 213.389, 211.389, 209.389, 207.444, 205.5, 203.611, 201.667, 199.778, 197.944, 196.056, 194.222, 193.667, 194.833, 196.056, 197.278, 198.444, 200.944, 203.389, 205.944, 208.5, 211.056, 213.444, 214.889, 216.333, 217.833, 219.278, 220.778, 222.278, 223.778, 225.278, 226.778, 228.278, 229.778, 231.278, 232.722, 234.222, 235.722]
_MIL_DAYS    = {'hot': _MIL_HOT_K, 'cold': _MIL_COLD_K,
                'polar': _MIL_POLAR_K, 'tropical': _MIL_TROP_K}
_MIL_TOP_M       = _MIL_ALT_M[-1]   # 30.48 km — table top (100 kft)
_MIL_TAPER_TOP_M = 32000.0          # ΔT tapered to zero by here


def _atmosphere_nonstd(altitude_m, day):
    """MIL-STD-210A hot/cold/polar/tropical atmosphere (geometric altitude, m).

    T is taken from the MIL-210 table, pressure equals US Std 1976, and
    ρ = P/(R·T).  The temperature offset from standard is tapered to zero
    between 30.48 and 32 km so the profile reverts to US Std 1976 above the
    table top.  Returns (T_K, P_Pa, rho_kg_m3, a_m_s).
    """
    scalar = np.ndim(altitude_m) == 0
    h = np.atleast_1d(np.asarray(altitude_m, dtype=float))
    T_std, P, _rho_std, _a = _atmosphere_std1976(h)
    T_std = np.atleast_1d(T_std); P = np.atleast_1d(P)
    T_mil = np.interp(np.clip(h, 0.0, _MIL_TOP_M), _MIL_ALT_M, _MIL_DAYS[day])
    taper = np.clip((_MIL_TAPER_TOP_M - h) / (_MIL_TAPER_TOP_M - _MIL_TOP_M),
                    0.0, 1.0)
    T   = T_std + (T_mil - T_std) * taper
    rho = P / (_R * T)
    a   = np.sqrt(_GAMMA * _R * T)
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
    # Invalidate the scalar cache — it may hold values from the previous model.
    _cs = globals().get('_atmosphere_scalar')
    if _cs is not None:
        _cs.cache_clear()
    model = _ATM_CONFIG['model']
    if model != 'msis':
        # 'std1976' or a MIL-STD-210A day ('hot'/'cold'/'polar'/'tropical')
        _ATM_TABLE  = None
        _ATM_SOURCE = model if model in _MIL_DAYS else 'std1976'
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
        'msis' (default), 'std1976', or a MIL-STD-210A non-standard day
        'hot' / 'cold' / 'polar' / 'tropical' (US Std 1976 pressure with the
        MIL-210 temperature profile below 30.5 km).
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
    # Scalar fast path — the per-integration-step case.  Route through a cached
    # helper (exact-altitude key → identical result), which also skips the
    # array-wrapping overhead: within one EOM step drag and thrust query the
    # SAME altitude, so the second call is a cache hit, and repeated altitudes
    # across a run are served without recomputation.  ~40% of trajectory runtime
    # was here.
    if np.ndim(altitude_m) == 0:
        h = float(altitude_m)
        if h < 0.0:
            h = 0.0
        elif h > 1_000_000.0:
            h = 1_000_000.0
        return _atmosphere_scalar(h)

    if _ATM_TABLE is not None:
        h = np.clip(np.asarray(altitude_m, dtype=float), 0.0, 1_000_000.0)
        h_tab = _ATM_TABLE['h_m']
        T   = np.interp(h, h_tab, _ATM_TABLE['T'])
        rho = np.exp(np.interp(h, h_tab, _ATM_TABLE['log_rho']))
        P   = np.exp(np.interp(h, h_tab, _ATM_TABLE['log_P']))
        a   = np.sqrt(_GAMMA * _R * T)
        return T, P, rho, a

    if _ATM_SOURCE in _MIL_DAYS:
        return _atmosphere_nonstd(altitude_m, _ATM_SOURCE)
    return _atmosphere_std1976(altitude_m)


@lru_cache(maxsize=1 << 17)
def _atmosphere_scalar(h):
    """Cached single-altitude atmosphere lookup.  `h` is a pre-clamped float.

    Cleared by _init_atmosphere() whenever the model/table is reconfigured, so
    the cache never serves values from a stale model.
    """
    if _ATM_TABLE is not None:
        h_tab = _ATM_TABLE['h_m']
        T   = float(np.interp(h, h_tab, _ATM_TABLE['T']))
        rho = math.exp(float(np.interp(h, h_tab, _ATM_TABLE['log_rho'])))
        P   = math.exp(float(np.interp(h, h_tab, _ATM_TABLE['log_P'])))
        a   = math.sqrt(_GAMMA * _R * T)
        return T, P, rho, a
    if _ATM_SOURCE in _MIL_DAYS:
        return _atmosphere_nonstd(h, _ATM_SOURCE)
    return _atmosphere_std1976(h)


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
