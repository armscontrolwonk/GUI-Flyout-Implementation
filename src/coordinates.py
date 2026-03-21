"""
Coordinate conversions matching Forden's geo2cart.m / cart2geo.m / localCoords.m.
WGS-84 ellipsoid.
"""

import numpy as np

# WGS-84
RE  = 6378137.0          # equatorial radius (m)
F   = 1.0 / 298.257223563
RP  = RE * (1 - F)       # polar radius (m)
E2  = 2*F - F**2         # first eccentricity squared
OMEGA_EARTH = 7.2921150e-5  # rad/s


def geodetic_to_ecef(lat_rad, lon_rad, alt_m):
    """
    Convert geodetic (lat, lon, alt) to ECEF Cartesian (x, y, z) metres.

    Parameters
    ----------
    lat_rad : geodetic latitude (rad)
    lon_rad : longitude (rad)
    alt_m   : altitude above WGS-84 ellipsoid (m)

    Returns
    -------
    pos : ndarray (3,)  [x, y, z] metres ECEF
    """
    sin_lat = np.sin(lat_rad)
    cos_lat = np.cos(lat_rad)
    N = RE / np.sqrt(1 - E2 * sin_lat**2)   # prime vertical radius

    x = (N + alt_m) * cos_lat * np.cos(lon_rad)
    y = (N + alt_m) * cos_lat * np.sin(lon_rad)
    z = (N * (1 - E2) + alt_m) * sin_lat
    return np.array([x, y, z])


def ecef_to_geodetic(pos_ecef):
    """
    Convert ECEF Cartesian to geodetic (lat, lon, alt).
    Uses Bowring's iterative method.

    Returns
    -------
    lat_rad, lon_rad, alt_m
    """
    x, y, z = pos_ecef
    lon = np.arctan2(y, x)
    p   = np.sqrt(x**2 + y**2)

    # Initial estimate
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(10):
        sin_lat = np.sin(lat)
        N = RE / np.sqrt(1 - E2 * sin_lat**2)
        lat_new = np.arctan2(z + E2 * N * sin_lat, p)
        if abs(lat_new - lat) < 1e-12:
            lat = lat_new
            break
        lat = lat_new

    sin_lat = np.sin(lat)
    N = RE / np.sqrt(1 - E2 * sin_lat**2)
    if abs(np.cos(lat)) > 1e-10:
        alt = p / np.cos(lat) - N
    else:
        alt = abs(z) / abs(sin_lat) - N * (1 - E2)

    return lat, lon, alt


def ecef_to_local_enu(pos_ecef, origin_ecef, origin_lat, origin_lon):
    """
    Convert ECEF position to local East-North-Up (ENU) frame
    centred at origin.

    Parameters
    ----------
    pos_ecef    : ndarray (3,)  target position in ECEF
    origin_ecef : ndarray (3,)  origin of local frame in ECEF
    origin_lat  : geodetic latitude of origin (rad)
    origin_lon  : longitude of origin (rad)

    Returns
    -------
    enu : ndarray (3,)  [east, north, up] in metres
    """
    dx = pos_ecef - origin_ecef
    sin_lat = np.sin(origin_lat)
    cos_lat = np.cos(origin_lat)
    sin_lon = np.sin(origin_lon)
    cos_lon = np.cos(origin_lon)

    # Rotation matrix ECEF -> ENU
    R = np.array([
        [-sin_lon,           cos_lon,          0       ],
        [-sin_lat*cos_lon,  -sin_lat*sin_lon,   cos_lat ],
        [ cos_lat*cos_lon,   cos_lat*sin_lon,   sin_lat ],
    ])
    return R @ dx


def coriolis_acceleration(vel_ecef):
    """
    Coriolis acceleration: a_cor = -2 * omega x v  (m/s^2)
    omega is Earth rotation vector = [0, 0, OMEGA_EARTH] in ECEF.
    """
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    return -2.0 * np.cross(omega, vel_ecef)


def centrifugal_acceleration(pos_ecef):
    """
    Centrifugal acceleration: a_cf = -omega x (omega x r)  (m/s^2)
    """
    omega = np.array([0.0, 0.0, OMEGA_EARTH])
    return -np.cross(omega, np.cross(omega, pos_ecef))


def range_between(lat1, lon1, lat2, lon2, radius=6371000.0):
    """
    Great-circle range between two geodetic points (m).
    """
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * radius * np.arcsin(np.sqrt(a))
