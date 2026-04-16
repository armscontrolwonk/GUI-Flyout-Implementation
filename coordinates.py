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


def range_between(lat1, lon1, lat2, lon2, radius=None):
    """
    Geodesic distance between two geodetic points on the WGS-84 ellipsoid (m).
    Uses the Vincenty inverse formula (accurate to ~0.5 mm).
    Falls back to spherical haversine for near-antipodal points.

    Parameters
    ----------
    lat1, lon1 : geodetic latitude/longitude of point 1 (rad)
    lat2, lon2 : geodetic latitude/longitude of point 2 (rad)
    radius     : ignored — kept for backward compatibility
    """
    if lat1 == lat2 and lon1 == lon2:
        return 0.0

    U1 = np.arctan((1.0 - F) * np.tan(lat1))
    U2 = np.arctan((1.0 - F) * np.tan(lat2))
    sin_U1, cos_U1 = np.sin(U1), np.cos(U1)
    sin_U2, cos_U2 = np.sin(U2), np.cos(U2)

    L = lon2 - lon1
    lam = L
    cos_2sigma_m = 0.0
    sin_alpha = 0.0
    cos2_alpha = 1.0
    sin_sigma = 0.0
    cos_sigma = 0.0
    sigma = 0.0
    for _ in range(200):
        sin_lam, cos_lam = np.sin(lam), np.cos(lam)
        sin_sigma = np.sqrt((cos_U2 * sin_lam) ** 2 +
                            (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam) ** 2)
        if sin_sigma == 0.0:
            return 0.0
        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
        sigma = np.arctan2(sin_sigma, cos_sigma)
        sin_alpha = cos_U1 * cos_U2 * sin_lam / sin_sigma
        cos2_alpha = 1.0 - sin_alpha ** 2
        cos_2sigma_m = (0.0 if cos2_alpha == 0.0
                        else cos_sigma - 2.0 * sin_U1 * sin_U2 / cos2_alpha)
        C = F / 16.0 * cos2_alpha * (4.0 + F * (4.0 - 3.0 * cos2_alpha))
        lam_new = L + (1.0 - C) * F * sin_alpha * (
            sigma + C * sin_sigma * (
                cos_2sigma_m + C * cos_sigma * (-1.0 + 2.0 * cos_2sigma_m ** 2)
            )
        )
        if abs(lam_new - lam) < 1e-12:
            lam = lam_new
            break
        lam = lam_new
    else:
        # Near-antipodal: Vincenty did not converge; fall back to haversine.
        dlat, dlon = lat2 - lat1, lon2 - lon1
        h = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return 2.0 * RE * np.arcsin(np.sqrt(h))

    u2 = cos2_alpha * (RE ** 2 - RP ** 2) / RP ** 2
    A_v = 1.0 + u2 / 16384.0 * (4096.0 + u2 * (-768.0 + u2 * (320.0 - 175.0 * u2)))
    B_v = u2 / 1024.0 * (256.0 + u2 * (-128.0 + u2 * (74.0 - 47.0 * u2)))
    d_sig = B_v * sin_sigma * (
        cos_2sigma_m + B_v / 4.0 * (
            cos_sigma * (-1.0 + 2.0 * cos_2sigma_m ** 2)
            - B_v / 6.0 * cos_2sigma_m
            * (-3.0 + 4.0 * sin_sigma ** 2)
            * (-3.0 + 4.0 * cos_2sigma_m ** 2)
        )
    )
    return RP * A_v * (sigma - d_sig)
