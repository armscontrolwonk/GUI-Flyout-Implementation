"""
3-DOF trajectory integrator matching Forden's missileFull3D.m /
integrateTrajectory.m.

State vector: [x, y, z, vx, vy, vz]  (ECEF, metres and m/s)

Physics included:
  - Gravity (J2 spheroid — more accurate than Forden's point-mass)
  - Aerodynamic drag  (Forden Eq. 3)
  - Thrust (powered phase, loft-angle pitch-over guidance)
  - Coriolis acceleration
  - Centrifugal acceleration

Guidance law — Forden Eq. 8
----------------------------
The missile launches vertically (elevation 90° from horizontal) and pitches
over at a constant rate loft_angle_rate_deg_s until the elevation reaches
loft_angle_deg, where it holds for the remainder of powered flight:

    el(t) = max(loft_angle_deg, 90° - loft_angle_rate_deg_s * t)

Azimuth is constant (set at launch).  The ENU frame is re-evaluated at the
missile's current geodetic position each step so that "local vertical" tracks
the missile as it moves downrange.

Validation against Forden Table 3 (maximum ranges, azimuth 40° East of N):
  Missile         Our model   Forden    Notes
  Scud-B          ~288 km     288 km    matches with correct params + guidance
  Al Hussein      ~693 km     693 km    matches with correct params + guidance
  No-dong         ~973 km     973 km    matches with correct params + guidance
  Taepodong-I    ~2349 km    2349 km    2-stage, matches with correct params
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
    active_stage, active_stage_and_t, total_burn_time,
)


# ---------------------------------------------------------------------------
# Flight-event helpers
# ---------------------------------------------------------------------------

def _stage_event_times(params: MissileParams):
    """
    Walk the stage linked list and return a list of
    (event_label, mission_elapsed_time_s) pairs for:
      - ignition of each stage
      - burnout of each stage
    Does not include apogee, fairing jettison, or impact (those need
    the integrated trajectory arrays).
    """
    events = []
    t = 0.0
    node = params
    stage = 1
    while node is not None:
        if stage == 1:
            events.append(("Ignition", t))
        elif node.coast_time_s > 0 or stage > 1:
            # Only emit a separate ignition event when there was a coast gap
            # (instant staging shares its time with the previous burnout)
            if t > 0 and (stage == 1 or
                          (params if stage == 2 else None) is not None):
                pass  # handled below
        t_burnout = t + node.burn_time_s
        events.append((f"Stage {stage} burnout", t_burnout))
        if node.stage2 is not None:
            coast = node.coast_time_s
            t = t_burnout + coast
            if coast > 0:
                events.append((f"Stage {stage + 1} ignition", t))
        node = node.stage2
        stage += 1
    return events


def _interp_milestone(t_event, t_arr, alt_arr, range_arr, speed_arr,
                      accel_arr, mass_arr):
    """
    Interpolate all channel arrays at t_event.  Clamps to the array bounds
    so events that fall after cutoff (engine off) still return valid values.
    """
    t_event = float(np.clip(t_event, t_arr[0], t_arr[-1]))
    return {
        't_s':       t_event,
        'alt_km':    float(np.interp(t_event, t_arr, alt_arr  / 1000.0)),
        'range_km':  float(np.interp(t_event, t_arr, range_arr / 1000.0)),
        'speed_kms': float(np.interp(t_event, t_arr, speed_arr / 1000.0)),
        'accel_ms2': float(np.interp(t_event, t_arr, accel_arr)),
        'mass_t':    float(np.interp(t_event, t_arr, mass_arr  / 1000.0)),
    }


# ---------------------------------------------------------------------------
# Guidance helpers
# ---------------------------------------------------------------------------

def _enu_frame(lat_rad: float, lon_rad: float):
    """Return (e_east, e_north, e_up) unit vectors in ECEF at given geodetic pos."""
    sin_lat, cos_lat = np.sin(lat_rad), np.cos(lat_rad)
    sin_lon, cos_lon = np.sin(lon_rad), np.cos(lon_rad)
    e_east  = np.array([-sin_lon,           cos_lon,           0.0    ])
    e_north = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
    e_up    = np.array([ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat])
    return e_east, e_north, e_up


def _gravity_turn_thrust_dir(vel_ecef, lat_rad, lon_rad, azimuth_rad,
                             kick_angle_deg, kick_rate_deg_s, t):
    """
    Two-phase gravity-turn guidance (IRBM/ICBM).

    Phase 1 — kick (0 ≤ t < t_kick):
        Pitch from 90° above horizontal down to kick_angle_deg at kick_rate_deg_s.
        t_kick = (90 - kick_angle_deg) / kick_rate_deg_s

    Phase 2 — gravity turn (t ≥ t_kick):
        Thrust locked to the velocity vector direction.  Earth's gravity and
        curvature naturally pitch the vehicle to the Wheelon-optimal burnout
        angle: ε* = (180° − φ°) / 4  (Wheelon 1959, Eq. 17).

    kick_angle_deg : elevation above horizontal at end of kick (e.g. 85° = 5°
                     from vertical).  Should be close to 90° so gravity turn
                     starts near-vertical.
    kick_rate_deg_s: rate of initial kick (°/s).
    """
    t_kick = (90.0 - kick_angle_deg) / max(kick_rate_deg_s, 0.1)

    if t < t_kick:
        el_deg = 90.0 - kick_rate_deg_s * t
    else:
        speed = np.linalg.norm(vel_ecef)
        if speed > 1.0:
            return vel_ecef / speed
        # Speed not yet established — hold kick angle as fallback
        el_deg = kick_angle_deg

    el_rad = np.radians(el_deg)
    e_east, e_north, e_up = _enu_frame(lat_rad, lon_rad)
    thrust = (np.cos(el_rad) * np.sin(azimuth_rad) * e_east +
              np.cos(el_rad) * np.cos(azimuth_rad) * e_north +
              np.sin(el_rad) * e_up)
    norm = np.linalg.norm(thrust)
    return thrust / norm if norm > 1e-12 else e_up


def _loft_thrust_dir(lat_rad, lon_rad, azimuth_rad,
                     loft_angle_deg, loft_angle_rate_deg_s, t):
    """
    Unit thrust vector (ECEF) under Forden's loft-angle guidance (Eq. 8).

    el(t) = max(loft_angle_deg, 90° − loft_angle_rate_deg_s * t)

    t is mission elapsed time (seconds since launch).  The curve runs
    continuously through all stages and coast phases; during coast no thrust
    is applied so the instantaneous attitude has no effect on the trajectory.
    """
    el_deg = max(loft_angle_deg, 90.0 - loft_angle_rate_deg_s * t)
    el_rad = np.radians(el_deg)
    e_east, e_north, e_up = _enu_frame(lat_rad, lon_rad)
    thrust = (np.cos(el_rad) * np.sin(azimuth_rad) * e_east +
              np.cos(el_rad) * np.cos(azimuth_rad) * e_north +
              np.sin(el_rad) * e_up)
    norm = np.linalg.norm(thrust)
    return thrust / norm if norm > 1e-12 else e_up


# ---------------------------------------------------------------------------
# Equations of motion
# ---------------------------------------------------------------------------

def _eom(t, state, params, cutoff_time, azimuth_rad):
    """
    Equations of motion in ECEF frame (Forden Eq. 5/6).

    state = [x, y, z, vx, vy, vz]
    Returns d(state)/dt.

    Guidance uses a single continuous pitch-over driven by mission elapsed
    time t and the top-level params.loft_angle_deg / loft_angle_rate_deg_s.
    The curve runs through all stages and coast phases; thrust_force() returns
    zero during coast so attitude during coast has no effect on the trajectory.
    """
    pos = state[:3]
    vel = state[3:]

    lat, lon, alt = ecef_to_geodetic(pos)
    alt = max(alt, 0.0)

    # --- Gravity ---
    g = gravity_ecef(pos)

    # Active stage (needed for drag and mass; guidance uses top-level params)
    astage, _ = active_stage_and_t(params, t)

    # --- Drag ---
    # After final-stage burnout, if an RV ballistic coefficient is supplied,
    # use β-based drag for the separating warhead: F = q * m_rv / β.
    # rv_mass_kg (single RV mass) is used when specified; otherwise fall back
    # to payload_kg (old behaviour, treats whole payload as one object).
    if (params.rv_beta_kg_m2 > 0 and params.payload_kg > 0
            and t > total_burn_time(params)):
        speed = np.linalg.norm(vel)
        if speed > 1e-6:
            _, _, rho, _ = atmosphere(alt)
            q = 0.5 * rho * speed ** 2
            rv_mass = params.rv_mass_kg if params.rv_mass_kg > 0 else params.payload_kg
            drag_mag = q * rv_mass / params.rv_beta_kg_m2
            f_drag = -drag_mag * (vel / speed)
        else:
            f_drag = np.zeros(3)
    else:
        f_drag = drag_force_vector(astage, vel, alt)

    # --- Thrust with mode-selected guidance ---
    if t <= cutoff_time:
        if params.guidance == "gravity_turn":
            thrust_dir = _gravity_turn_thrust_dir(
                vel, lat, lon, azimuth_rad,
                params.loft_angle_deg,
                params.loft_angle_rate_deg_s,
                t)
        else:  # "loft" (Forden)
            thrust_dir = _loft_thrust_dir(lat, lon, azimuth_rad,
                                          params.loft_angle_deg,
                                          params.loft_angle_rate_deg_s,
                                          t)
        f_thrust = thrust_force(params, t, alt, thrust_dir)
    else:
        f_thrust = np.zeros(3)

    # --- Mass ---
    m = missile_mass(params, t, alt)

    # --- Non-inertial frame corrections ---
    a_coriolis    = coriolis_acceleration(vel)
    a_centrifugal = centrifugal_acceleration(pos)

    # --- Total acceleration (Forden Eq. 6) ---
    accel = g + (f_drag + f_thrust) / m + a_coriolis + a_centrifugal

    return np.concatenate([vel, accel])


def _hit_ground(t, state, params, cutoff_time, azimuth_rad):
    """Event: missile hits the ground (altitude = 0)."""
    _, _, alt = ecef_to_geodetic(state[:3])
    return alt

_hit_ground.terminal  = True
_hit_ground.direction = -1


# ---------------------------------------------------------------------------
# Public integration interface
# ---------------------------------------------------------------------------

def integrate_trajectory(params: MissileParams,
                         launch_lat_deg: float,
                         launch_lon_deg: float,
                         launch_azimuth_deg: float,
                         loft_angle_deg: float = None,
                         loft_angle_rate_deg_s: float = None,
                         cutoff_time_s: float = None,
                         dt_output: float = 1.0,
                         max_time_s: float = 3600.0):
    """
    Integrate a missile trajectory from launch to impact.

    Parameters
    ----------
    params                : MissileParams
    launch_lat_deg        : geodetic launch latitude (degrees)
    launch_lon_deg        : launch longitude (degrees)
    launch_azimuth_deg    : launch azimuth clockwise from North (degrees)
    loft_angle_deg        : final elevation above horizontal (°); defaults to
                            params.loft_angle_deg
    loft_angle_rate_deg_s : pitch-over rate (°/s); defaults to
                            params.loft_angle_rate_deg_s
    cutoff_time_s         : engine cutoff time (s); defaults to full burn
    dt_output             : output time step (s)
    max_time_s            : maximum flight time (s)

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
    import copy
    # Apply stage-1 overrides non-destructively so the caller's object is
    # unchanged and stages 2+ keep their own stored guidance values.
    if loft_angle_deg is not None or loft_angle_rate_deg_s is not None:
        params = copy.copy(params)
        if loft_angle_deg is not None:
            params.loft_angle_deg = loft_angle_deg
        if loft_angle_rate_deg_s is not None:
            params.loft_angle_rate_deg_s = loft_angle_rate_deg_s

    total_burn = total_burn_time(params)
    if cutoff_time_s is None:
        cutoff_time_s = total_burn

    lat0 = np.radians(launch_lat_deg)
    lon0 = np.radians(launch_lon_deg)
    az   = np.radians(launch_azimuth_deg)

    # Initial position on surface; initial velocity: small upward nudge
    pos0 = geodetic_to_ecef(lat0, lon0, 0.0)
    _, _, e_up = _enu_frame(lat0, lon0)
    v0 = 10.0 * e_up      # 10 m/s upward so integrator starts above ground

    state0 = np.concatenate([pos0, v0])

    t_span = (0.0, max_time_s)
    t_eval = np.arange(0.0, max_time_s, dt_output)

    eom_args = (params, cutoff_time_s, az)

    sol = solve_ivp(
        fun=_eom,
        t_span=t_span,
        y0=state0,
        method='RK45',
        t_eval=t_eval,
        events=_hit_ground,
        args=eom_args,
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

    lats   = np.array(lats)
    lons   = np.array(lons)
    alts   = np.array(alts)
    speeds = np.linalg.norm(vel_arr, axis=1)

    ranges = np.array([
        range_between(lat0, lon0, np.radians(la), np.radians(lo))
        for la, lo in zip(lats, lons)
    ])

    apo_idx = int(np.argmax(alts))

    # --- Mass array -------------------------------------------------------
    masses = np.array([missile_mass(params, t_arr[i], alts[i])
                       for i in range(len(t_arr))])

    # --- Acceleration array (central finite-difference on speed) ----------
    accels = np.empty_like(speeds)
    accels[1:-1] = (speeds[2:] - speeds[:-2]) / (t_arr[2:] - t_arr[:-2])
    accels[0]    = accels[1]
    accels[-1]   = accels[-2]

    # --- Flight-event milestones ------------------------------------------
    milestones = []

    # Stage ignition / burnout events from the stage list
    for label, t_ev in _stage_event_times(params):
        if t_ev > t_arr[-1]:
            break          # vehicle hit ground before this event
        row = _interp_milestone(t_ev, t_arr, alts, ranges, speeds, accels, masses)
        row['event'] = label
        milestones.append(row)

    # Fairing (shroud) jettison — first crossing of the threshold altitude
    if params.shroud_mass_kg > 0:
        thr_m = params.shroud_jettison_alt_km * 1000.0
        cross = np.where(np.diff(np.sign(alts - thr_m)) > 0)[0]
        if len(cross):
            idx = cross[0]
            # Linear interpolation to find exact crossing time
            frac = (thr_m - alts[idx]) / (alts[idx + 1] - alts[idx])
            t_fairing = t_arr[idx] + frac * (t_arr[idx + 1] - t_arr[idx])
            row = _interp_milestone(t_fairing, t_arr, alts, ranges,
                                    speeds, accels, masses)
            row['event'] = "Fairing jettison"
            # Insert in chronological order
            inserted = False
            for i, m in enumerate(milestones):
                if m['t_s'] > t_fairing:
                    milestones.insert(i, row)
                    inserted = True
                    break
            if not inserted:
                milestones.append(row)

    # Apogee
    apo_row = _interp_milestone(t_arr[apo_idx], t_arr, alts, ranges,
                                speeds, accels, masses)
    apo_row['event'] = "Apogee"
    # Insert chronologically
    inserted = False
    for i, m in enumerate(milestones):
        if m['t_s'] > t_arr[apo_idx]:
            milestones.insert(i, apo_row)
            inserted = True
            break
    if not inserted:
        milestones.append(apo_row)

    # Impact
    imp_row = _interp_milestone(t_arr[-1], t_arr, alts, ranges,
                                speeds, accels, masses)
    imp_row['event'] = "Impact"
    milestones.append(imp_row)

    return {
        't':                  t_arr,
        'lat':                lats,
        'lon':                lons,
        'alt':                alts,
        'speed':              speeds,
        'accel':              accels,
        'mass':               masses,
        'range':              ranges,
        'pos_ecef':           pos_arr,
        'vel_ecef':           vel_arr,
        'impact_lat':         lats[-1],
        'impact_lon':         lons[-1],
        'range_km':           ranges[-1] / 1000.0,
        'apogee_km':          np.max(alts) / 1000.0,
        'apogee_lat_deg':     lats[apo_idx],
        'apogee_lon_deg':     lons[apo_idx],
        'time_of_flight_s':   t_arr[-1],
        'impact_speed_ms':    speeds[-1],
        'milestones':         milestones,
    }


def aim_missile(params: MissileParams,
                launch_lat_deg: float,
                launch_lon_deg: float,
                launch_azimuth_deg: float,
                target_range_km: float,
                loft_angle_deg: float = None,
                loft_angle_rate_deg_s: float = None) -> float:
    """
    Find the engine cutoff time (seconds) that produces the desired range,
    using the missile's loft-angle guidance parameters.

    Returns cutoff_time_s.
    """
    from scipy.optimize import brentq

    la  = loft_angle_deg          if loft_angle_deg          is not None else params.loft_angle_deg
    lar = loft_angle_rate_deg_s   if loft_angle_rate_deg_s   is not None else params.loft_angle_rate_deg_s

    def range_error(cutoff):
        r = integrate_trajectory(params, launch_lat_deg, launch_lon_deg,
                                 launch_azimuth_deg,
                                 loft_angle_deg=la,
                                 loft_angle_rate_deg_s=lar,
                                 cutoff_time_s=cutoff)
        return r['range_km'] - target_range_km

    total_burn = total_burn_time(params)
    lo, hi = 5.0, total_burn
    try:
        cutoff = brentq(range_error, lo, hi, xtol=1.0, maxiter=50)
    except ValueError:
        cutoff = total_burn
    return cutoff


def find_range(params: MissileParams,
               launch_lat_deg: float,
               launch_lon_deg: float,
               launch_azimuth_deg: float,
               loft_angle_deg: float = None,
               loft_angle_rate_deg_s: float = None,
               cutoff_time_s: float = None) -> float:
    """Return the range (km) for the given loft parameters and cutoff time."""
    result = integrate_trajectory(
        params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
        loft_angle_deg=loft_angle_deg,
        loft_angle_rate_deg_s=loft_angle_rate_deg_s,
        cutoff_time_s=cutoff_time_s,
    )
    return result['range_km']


def maximize_range(params: MissileParams,
                   launch_lat_deg: float,
                   launch_lon_deg: float,
                   launch_azimuth_deg: float = 0.0,
                   loft_angle_deg: float = None,
                   loft_angle_rate_deg_s: float = None) -> dict:
    """
    Find the maximum range by optimising loft_angle and loft_angle_rate.

    If loft_angle_deg and loft_angle_rate_deg_s are both provided, the
    trajectory is run with those fixed values (no optimisation).  Otherwise
    a two-stage coarse/fine grid search is performed over the two parameters.

    Returns the full trajectory dict plus:
        'max_range_km'            : achieved maximum range (km)
        'optimal_loft_angle_deg'  : best loft angle (°)
        'optimal_loft_rate_deg_s' : best loft angle rate (°/s)
    """
    total_burn = total_burn_time(params)

    # If both guidance params are supplied, just run and return
    if loft_angle_deg is not None and loft_angle_rate_deg_s is not None:
        traj = integrate_trajectory(
            params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
            loft_angle_deg=loft_angle_deg,
            loft_angle_rate_deg_s=loft_angle_rate_deg_s,
            cutoff_time_s=total_burn,
        )
        traj['max_range_km']            = traj['range_km']
        traj['optimal_loft_angle_deg']  = loft_angle_deg
        traj['optimal_loft_rate_deg_s'] = loft_angle_rate_deg_s
        return traj

    def _run(la, lar):
        try:
            r = integrate_trajectory(
                params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
                loft_angle_deg=la, loft_angle_rate_deg_s=lar,
                cutoff_time_s=total_burn)
            return r['range_km']
        except Exception:
            return -1.0

    best_range = -1.0
    best_la  = params.loft_angle_deg
    best_lar = params.loft_angle_rate_deg_s

    if params.guidance == "gravity_turn":
        # Sweep kick angle (near-vertical, 75°–89° above horizontal).
        # Kick rate has little effect once gravity turn is engaged; fix at 5°/s.
        kick_rate = 5.0
        for kick_a in range(75, 90, 2):
            rng = _run(float(kick_a), kick_rate)
            if rng > best_range:
                best_range = rng
                best_la  = float(kick_a)
                best_lar = kick_rate
        # Fine search ±2° around best kick angle
        for dka in (-2.0, -1.0, 0.0, 1.0, 2.0):
            ka = best_la + dka
            if ka < 70.0 or ka > 89.5:
                continue
            rng = _run(ka, kick_rate)
            if rng > best_range:
                best_range = rng
                best_la = ka
    else:
        # Forden loft: coarse 10° steps in loft_angle, 0.5 °/s steps in rate
        for la in range(20, 75, 10):
            for lar_x2 in range(1, 7):   # 0.5 … 3.0 °/s
                lar = lar_x2 * 0.5
                rng = _run(float(la), lar)
                if rng > best_range:
                    best_range = rng
                    best_la  = float(la)
                    best_lar = lar
        # Fine search: ±5° and ±0.25 °/s around coarse best
        for dla in (-5.0, -2.5, 0.0, 2.5, 5.0):
            for dlar in (-0.25, 0.0, 0.25):
                la  = best_la  + dla
                lar = best_lar + dlar
                if la < 5.0 or la > 85.0 or lar < 0.1:
                    continue
                rng = _run(la, lar)
                if rng > best_range:
                    best_range = rng
                    best_la  = la
                    best_lar = lar

    traj = integrate_trajectory(
        params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
        loft_angle_deg=best_la,
        loft_angle_rate_deg_s=best_lar,
        cutoff_time_s=total_burn,
    )
    traj['max_range_km']            = traj['range_km']
    traj['optimal_loft_angle_deg']  = best_la
    traj['optimal_loft_rate_deg_s'] = best_lar
    return traj
