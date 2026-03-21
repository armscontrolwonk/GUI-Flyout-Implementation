"""
3-DOF trajectory integrator matching Forden's missileFull3D.m /
integrateTrajectory.m.

State vector: [x, y, z, vx, vy, vz]  (ECEF, metres and m/s)

Physics included:
  - Gravity (J2 spheroid)
  - Aerodynamic drag
  - Thrust (powered phase, fixed-pitch guidance)
  - Coriolis acceleration
  - Centrifugal acceleration
"""

import numpy as np
from scipy.integrate import solve_ivp

from gravity import gravity_ecef
from atmosphere import atmosphere
from coordinates import (
    geodetic_to_ecef, ecef_to_geodetic,
    coriolis_acceleration, centrifugal_acceleration,
    range_between, OMEGA_EARTH,
)
from missile_models import (
    MissileParams, missile_mass, drag_force_vector, thrust_force,
)


def _eom(t, state, params, cutoff_time, thrust_dir_fixed):
    """
    Equations of motion in ECEF frame.

    state = [x, y, z, vx, vy, vz]
    thrust_dir_fixed: unit vector (ECEF) giving fixed thrust direction during burn
    Returns d(state)/dt.
    """
    pos = state[:3]
    vel = state[3:]

    lat, lon, alt = ecef_to_geodetic(pos)
    alt = max(alt, 0.0)

    # --- Gravity ---
    g = gravity_ecef(pos)

    # --- Drag ---
    f_drag = drag_force_vector(params, vel, alt)

    # --- Thrust (fixed pitch: thrust in launch direction during powered phase) ---
    if t <= cutoff_time:
        thrust_dir = thrust_dir_fixed
    else:
        thrust_dir = np.zeros(3)

    f_thrust = thrust_force(params, t, alt, thrust_dir)

    # --- Mass ---
    m = missile_mass(params, t)

    # --- Non-inertial frame corrections ---
    a_coriolis    = coriolis_acceleration(vel)
    a_centrifugal = centrifugal_acceleration(pos)

    # --- Total acceleration ---
    accel = g + (f_drag + f_thrust) / m + a_coriolis + a_centrifugal

    return np.concatenate([vel, accel])


def _hit_ground(t, state, params, cutoff_time, thrust_dir_fixed):
    """Event: missile hits the ground (altitude = 0)."""
    pos = state[:3]
    _, _, alt = ecef_to_geodetic(pos)
    return alt
_hit_ground.terminal  = True
_hit_ground.direction = -1


def aim_missile(params: MissileParams,
                launch_lat_deg: float,
                launch_lon_deg: float,
                launch_azimuth_deg: float,
                target_range_km: float,
                launch_elevation_deg: float = None) -> float:
    """
    Find the engine cutoff time (seconds) that produces the desired range.
    Uses bisection over [5 s, burn_time_s] at the given launch elevation.

    Returns cutoff_time_s.
    """
    from scipy.optimize import brentq

    def range_error(cutoff):
        r = integrate_trajectory(params, launch_lat_deg, launch_lon_deg,
                                 launch_azimuth_deg,
                                 launch_elevation_deg=launch_elevation_deg,
                                 cutoff_time_s=cutoff)
        return r['range_km'] - target_range_km

    total_burn = params.burn_time_s
    if params.stage2 is not None:
        total_burn += params.stage2.burn_time_s
    lo, hi = 5.0, total_burn
    try:
        cutoff = brentq(range_error, lo, hi, xtol=1.0, maxiter=50)
    except ValueError:
        cutoff = total_burn
    return cutoff


def integrate_trajectory(params: MissileParams,
                         launch_lat_deg: float,
                         launch_lon_deg: float,
                         launch_azimuth_deg: float,
                         launch_elevation_deg: float = None,
                         cutoff_time_s: float = None,
                         dt_output: float = 1.0,
                         max_time_s: float = 3600.0):
    """
    Integrate a missile trajectory from launch to impact.

    Parameters
    ----------
    params              : MissileParams
    launch_lat_deg      : geodetic launch latitude (degrees)
    launch_lon_deg      : launch longitude (degrees)
    launch_azimuth_deg  : launch azimuth measured clockwise from North (degrees)
    launch_elevation_deg: initial pitch angle above horizontal (degrees)
    cutoff_time_s       : engine cutoff time (s); defaults to burn_time_s
    dt_output           : output time step (s)
    max_time_s          : maximum flight time (s)

    Returns
    -------
    result : dict with keys
        't'         : time array (s)
        'lat'       : geodetic latitude array (deg)
        'lon'       : longitude array (deg)
        'alt'       : altitude array (m)
        'speed'     : speed array (m/s)
        'range'     : downrange distance from launch (m)
        'pos_ecef'  : (N,3) ECEF positions (m)
        'vel_ecef'  : (N,3) ECEF velocities (m/s)
        'impact_lat': impact latitude (deg)
        'impact_lon': impact longitude (deg)
        'range_km'  : total range (km)
        'apogee_km' : maximum altitude (km)
    """
    total_burn = params.burn_time_s
    if params.stage2 is not None:
        total_burn += params.stage2.burn_time_s
    if cutoff_time_s is None:
        cutoff_time_s = total_burn

    # Auto-select elevation: 55° or enough to lift off (T/W > 1/sin(el))
    if launch_elevation_deg is None:
        tw = params.thrust_N / (params.mass_initial * 9.81)
        min_el = np.degrees(np.arcsin(min(1.0, 1.0 / tw))) if tw > 1.0 else 85.0
        launch_elevation_deg = max(55.0, min_el + 7.0)

    lat0 = np.radians(launch_lat_deg)
    lon0 = np.radians(launch_lon_deg)
    az   = np.radians(launch_azimuth_deg)
    el   = np.radians(launch_elevation_deg)

    # Initial ECEF position (on surface)
    pos0 = geodetic_to_ecef(lat0, lon0, 0.0)

    # Initial velocity: small upward nudge in local frame -> ECEF
    # Local ENU unit vectors
    sin_lat, cos_lat = np.sin(lat0), np.cos(lat0)
    sin_lon, cos_lon = np.sin(lon0), np.cos(lon0)

    e_east  = np.array([-sin_lon,  cos_lon,  0.0])
    e_north = np.array([-sin_lat*cos_lon, -sin_lat*sin_lon, cos_lat])
    e_up    = np.array([ cos_lat*cos_lon,  cos_lat*sin_lon, sin_lat])

    # Launch direction unit vector in ECEF (fixed thrust direction during burn)
    thrust_dir = (np.cos(el) * np.sin(az) * e_east +
                  np.cos(el) * np.cos(az) * e_north +
                  np.sin(el) * e_up)

    # Initial velocity: small nudge in launch direction
    v0 = 10.0 * thrust_dir

    state0 = np.concatenate([pos0, v0])

    t_span = (0.0, max_time_s)
    t_eval = np.arange(0.0, max_time_s, dt_output)

    sol = solve_ivp(
        fun=_eom,
        t_span=t_span,
        y0=state0,
        method='RK45',
        t_eval=t_eval,
        events=_hit_ground,
        args=(params, cutoff_time_s, thrust_dir),
        rtol=1e-8,
        atol=1e-6,
        dense_output=False,
        max_step=5.0,
    )

    t_arr    = sol.t
    pos_arr  = sol.y[:3].T   # (N, 3)
    vel_arr  = sol.y[3:].T   # (N, 3)

    lats, lons, alts = [], [], []
    for p in pos_arr:
        la, lo, al = ecef_to_geodetic(p)
        lats.append(np.degrees(la))
        lons.append(np.degrees(lo))
        alts.append(al)

    lats = np.array(lats)
    lons = np.array(lons)
    alts = np.array(alts)
    speeds = np.linalg.norm(vel_arr, axis=1)

    ranges = np.array([
        range_between(lat0, lon0, np.radians(la), np.radians(lo))
        for la, lo in zip(lats, lons)
    ])

    impact_lat = lats[-1]
    impact_lon = lons[-1]
    total_range_km = ranges[-1] / 1000.0
    apogee_km = np.max(alts) / 1000.0

    return {
        't':          t_arr,
        'lat':        lats,
        'lon':        lons,
        'alt':        alts,
        'speed':      speeds,
        'range':      ranges,
        'pos_ecef':   pos_arr,
        'vel_ecef':   vel_arr,
        'impact_lat': impact_lat,
        'impact_lon': impact_lon,
        'range_km':   total_range_km,
        'apogee_km':  apogee_km,
    }


def find_range(params: MissileParams,
               launch_lat_deg: float,
               launch_lon_deg: float,
               launch_azimuth_deg: float,
               cutoff_time_s: float = None,
               launch_elevation_deg: float = None) -> float:
    """
    Return the range (km) for a given cutoff time (fixed-pitch trajectory).
    """
    result = integrate_trajectory(
        params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
        launch_elevation_deg=launch_elevation_deg,
        cutoff_time_s=cutoff_time_s,
    )
    return result['range_km']


def maximize_range(params: MissileParams,
                   launch_lat_deg: float,
                   launch_lon_deg: float,
                   launch_azimuth_deg: float = 0.0) -> dict:
    """
    Find the maximum range by scanning elevation angles (full burn).

    Returns dict with 'max_range_km', 'optimal_cutoff_s', and full trajectory.
    """
    total_burn = params.burn_time_s
    if params.stage2 is not None:
        total_burn += params.stage2.burn_time_s

    best_range = -1.0
    best_el    = 45.0

    for el in range(30, 76, 5):
        try:
            r = integrate_trajectory(
                params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
                launch_elevation_deg=float(el),
                cutoff_time_s=total_burn)
            if r['range_km'] > best_range:
                best_range = r['range_km']
                best_el    = float(el)
        except Exception:
            pass

    traj = integrate_trajectory(
        params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
        launch_elevation_deg=best_el,
        cutoff_time_s=total_burn,
    )
    traj['max_range_km']     = traj['range_km']
    traj['optimal_cutoff_s'] = total_burn
    return traj
