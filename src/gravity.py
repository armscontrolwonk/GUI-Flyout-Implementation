"""
Gravity model.  Matches the Forden tool's gravity.m:
  - WGS-84 oblate-spheroid gravity (J2 term included)
  - Returns acceleration vector in ECEF Cartesian frame (m/s^2)
"""

import numpy as np

# WGS-84 constants
GM   = 3.986004418e14   # m^3/s^2
RE   = 6378137.0        # m  (equatorial radius)
J2   = 1.08262668e-3    # second zonal harmonic
OMEGA_EARTH = 7.2921150e-5  # rad/s  (Earth rotation rate)


def gravity_ecef(pos_ecef):
    """
    Gravitational acceleration in ECEF frame (m/s^2), J2 model.

    Parameters
    ----------
    pos_ecef : array-like, shape (3,)
        Position vector [x, y, z] in metres (ECEF).

    Returns
    -------
    g_ecef : ndarray, shape (3,)
        Gravitational acceleration [gx, gy, gz] m/s^2.
    """
    x, y, z = pos_ecef
    r2 = x*x + y*y + z*z
    r  = np.sqrt(r2)
    r5 = r2 * r2 * r

    sin2_lat = (z / r) ** 2
    j2_factor = 1.5 * J2 * (RE / r) ** 2

    # Radial component (inward)
    g_r = -GM / r2 * (1 - j2_factor * (5 * sin2_lat - 1))

    # Polar axis component
    g_z_extra = -GM / r2 * j2_factor * (-2) * (z / r)

    # Full vector
    g_x = g_r * (x / r) + (-GM * 3 * J2 * RE**2 / (2 * r5)) * x * (1 - 5 * z**2 / r2)
    g_y = g_r * (y / r) + (-GM * 3 * J2 * RE**2 / (2 * r5)) * y * (1 - 5 * z**2 / r2)
    g_z = g_r * (z / r) + (-GM * 3 * J2 * RE**2 / (2 * r5)) * z * (3 - 5 * z**2 / r2)

    return np.array([g_x, g_y, g_z])


def gravity_magnitude(altitude_m, latitude_rad=0.0):
    """
    Scalar gravity at given altitude and geodetic latitude (approximate).
    Used for simple 1-D checks.
    """
    r = RE + altitude_m
    g0 = GM / RE**2
    # Somigliana formula approximation
    g_surface = 9.7803253359 * (1 + 0.00193185265241 * np.sin(latitude_rad)**2) \
                / np.sqrt(1 - 0.00669437999014 * np.sin(latitude_rad)**2)
    return g_surface * (RE / r) ** 2
