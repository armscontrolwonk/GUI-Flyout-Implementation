"""
3-DOF trajectory integrator matching Forden's missileFull3D.m /
integrateTrajectory.m.

Reference frame
---------------
The state vector [x, y, z, vx, vy, vz] is expressed in ECEF
(Earth-Centered, Earth-Fixed) — a frame that rotates with Earth at
OMEGA_EARTH = 7.2921150e-5 rad/s.  Consequently:

  * Velocities (and the "ground speed" output) are *relative to Earth's
    surface*, not relative to inertial space.
  * A missile sitting on the launch pad has ECEF velocity ≈ 0, which is
    the correct initial condition for this frame — no explicit Earth-
    rotation term needs to be added to v0.
  * Earth's rotation is fully accounted for during flight through the
    Coriolis (-2 ω × v) and centrifugal (-ω × (ω × r)) pseudo-forces
    that appear in the ECEF equations of motion.

Inertial speed
--------------
For applications that require inertial (ECI-frame) speed — re-entry
heating, radar cross-section, or energy calculations — the inertial
velocity vector is obtained by adding back the Earth-rotation contribution:

    v_inertial = v_ecef + ω × r

where ω = [0, 0, OMEGA_EARTH] and r is the ECEF position vector.
At a launch latitude of 33°N this adds ≈ 390 m/s eastward at the pad
and grows to several hundred m/s of correction at apogee.  Both ground
speed and inertial speed are included in the output arrays and the Flight
Timeline milestones.

Physics included
----------------
  - Gravity (J2 spheroid — more accurate than Forden's point-mass)
  - Aerodynamic drag  (Forden Eq. 3)
  - Thrust (powered phase, loft-angle pitch-over guidance)
  - Coriolis acceleration  (-2 ω × v)
  - Centrifugal acceleration  (-ω × (ω × r))

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
    (event_label, mission_elapsed_time_s) pairs for every stage
    ignition and burnout.

    Stage 1 fires at t=0 ("Ignition").  Each subsequent stage emits both
    a burnout for the previous stage and an ignition for itself.  When
    coast_time_s == 0 both events share the same timestamp; when there is
    a coast gap they are separated by that gap.
    """
    events = []
    t = 0.0
    node = params
    stage = 1
    while node is not None:
        label = "Ignition" if stage == 1 else f"Stage {stage} ignition"
        events.append((label, t))
        t_burnout = t + node.burn_time_s
        events.append((f"Stage {stage} burnout", t_burnout))
        if node.stage2 is not None:
            t = t_burnout + node.coast_time_s
        node = node.stage2
        stage += 1
    return events


def _interp_milestone(t_event, t_arr, alt_arr, range_arr, speed_arr,
                      inertial_speed_arr, accel_arr, mass_arr):
    """
    Interpolate all channel arrays at t_event.  Clamps to the array bounds
    so events that fall after cutoff (engine off) still return valid values.

    speed_arr          — ECEF-frame (ground) speed, m/s
    inertial_speed_arr — ECI-frame (inertial) speed, m/s;
                         = ||v_ecef + ω × r||, used for re-entry heating etc.
    """
    t_event = float(np.clip(t_event, t_arr[0], t_arr[-1]))
    return {
        't_s':              t_event,
        'alt_km':           float(np.interp(t_event, t_arr, alt_arr  / 1000.0)),
        'range_km':         float(np.interp(t_event, t_arr, range_arr / 1000.0)),
        'speed_kms':        float(np.interp(t_event, t_arr, speed_arr / 1000.0)),
        'inertial_speed_kms': float(np.interp(t_event, t_arr,
                                              inertial_speed_arr / 1000.0)),
        'accel_ms2':        float(np.interp(t_event, t_arr, accel_arr)),
        'mass_t':           float(np.interp(t_event, t_arr, mass_arr  / 1000.0)),
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


_GRAVITY_TURN_VERTICAL_S = 5.0   # seconds of straight-up flight before pitch program


def _gravity_turn_thrust_dir(lat_rad, lon_rad, azimuth_rad,
                             burnout_angle_deg, total_burn_s, t):
    """
    Linear pitch program to Wheelon-optimal burnout angle (Levanger/Wright).

    Phase 1 (0 – 5 s): hold vertical (90° elevation).
    Phase 2 (5 s – total_burn_s): constant pitch rate from 90° down to
        burnout_angle_deg, reaching that angle exactly at engine cutoff.

    burnout_angle_deg : desired elevation at burnout (°).  Set to the
        Wheelon-optimal ε* = (π/2 − range/4R) converted to degrees, or
        whatever value the user supplies.
    total_burn_s      : total powered-flight duration (seconds), used to
        set the constant pitch rate.  Passed as cutoff_time from _eom.
    """
    if t < _GRAVITY_TURN_VERTICAL_S:
        el_deg = 90.0
    else:
        pitch_duration = max(total_burn_s - _GRAVITY_TURN_VERTICAL_S, 1.0)
        frac = min(1.0, (t - _GRAVITY_TURN_VERTICAL_S) / pitch_duration)
        el_deg = 90.0 - frac * (90.0 - burnout_angle_deg)

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
                lat, lon, azimuth_rad,
                params.loft_angle_deg,
                cutoff_time,
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
                         guidance: str = None,
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
    # Apply session-level overrides non-destructively.  guidance, loft_angle_deg
    # and loft_angle_rate_deg_s are flight parameters (like launch site) that
    # the caller may override independently of the stored missile definition.
    if guidance is not None or loft_angle_deg is not None or loft_angle_rate_deg_s is not None:
        params = copy.copy(params)
        if guidance is not None:
            params.guidance = guidance
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

    # --- Inertial (ECI-frame) speed ---------------------------------------
    # v_inertial = v_ecef + ω × r   (ω = Earth rotation vector)
    # This is needed for re-entry heating, radar, and energy calculations.
    omega_vec = np.array([0.0, 0.0, OMEGA_EARTH])
    inertial_vel_arr = vel_arr + np.cross(omega_vec, pos_arr)
    inertial_speeds  = np.linalg.norm(inertial_vel_arr, axis=1)

    # --- Acceleration array (central finite-difference on ground speed) ---
    accels = np.empty_like(speeds)
    accels[1:-1] = (speeds[2:] - speeds[:-2]) / (t_arr[2:] - t_arr[:-2])
    accels[0]    = accels[1]
    accels[-1]   = accels[-2]

    # --- Flight-event milestones ------------------------------------------
    milestones = []

    def _insert_chrono(row):
        """Insert a milestone dict in ascending t_s order."""
        for i, m in enumerate(milestones):
            if m['t_s'] > row['t_s']:
                milestones.insert(i, row)
                return
        milestones.append(row)

    def _alt_crossing(threshold_m, ascending):
        """
        Return the interpolated time of the first altitude crossing of
        threshold_m in the requested direction (ascending=True → going up,
        False → going down).  Returns None if not found.
        """
        delta = alts - threshold_m
        sign_changes = np.diff(np.sign(delta))
        direction = 1 if ascending else -1
        indices = np.where(sign_changes * direction > 0)[0]
        if not len(indices):
            return None
        idx = indices[0]
        frac = (threshold_m - alts[idx]) / (alts[idx + 1] - alts[idx])
        return float(t_arr[idx] + frac * (t_arr[idx + 1] - t_arr[idx]))

    def _milestone(t_ev):
        return _interp_milestone(t_ev, t_arr, alts, ranges, speeds,
                                 inertial_speeds, accels, masses)

    # Stage ignition / burnout events from the stage list
    for label, t_ev in _stage_event_times(params):
        if t_ev > t_arr[-1]:
            break          # vehicle hit ground before this event
        row = _milestone(t_ev)
        row['event'] = label
        milestones.append(row)

    # Fairing (shroud) jettison — first upward crossing of jettison altitude
    if params.shroud_mass_kg > 0:
        t_ev = _alt_crossing(params.shroud_jettison_alt_km * 1000.0,
                             ascending=True)
        if t_ev is not None:
            row = _milestone(t_ev)
            row['event'] = "Fairing jettison"
            _insert_chrono(row)

    # Apogee
    apo_row = _milestone(t_arr[apo_idx])
    apo_row['event'] = "Apogee"
    _insert_chrono(apo_row)

    # Re-entry interface — first downward crossing of 100 km (after apogee)
    REENTRY_ALT_M = 100_000.0
    if np.max(alts) > REENTRY_ALT_M:
        t_ev = _alt_crossing(REENTRY_ALT_M, ascending=False)
        if t_ev is not None and t_ev > t_arr[apo_idx]:
            row = _milestone(t_ev)
            row['event'] = "Re-entry (100 km)"
            _insert_chrono(row)

    # Impact
    imp_row = _milestone(t_arr[-1])
    imp_row['event'] = "Impact"
    milestones.append(imp_row)

    return {
        't':                  t_arr,
        'lat':                lats,
        'lon':                lons,
        'alt':                alts,
        'speed':              speeds,          # ECEF-frame (ground speed), m/s
        'inertial_speed':     inertial_speeds, # ECI-frame (inertial speed), m/s
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
                guidance: str = None,
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
                                 guidance=guidance,
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
                   guidance: str = None,
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

    # If both loft params are supplied, just run and return (no grid search)
    if loft_angle_deg is not None and loft_angle_rate_deg_s is not None:
        traj = integrate_trajectory(
            params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
            guidance=guidance,
            loft_angle_deg=loft_angle_deg,
            loft_angle_rate_deg_s=loft_angle_rate_deg_s,
            cutoff_time_s=total_burn,
        )
        traj['max_range_km']            = traj['range_km']
        traj['optimal_loft_angle_deg']  = loft_angle_deg
        traj['optimal_loft_rate_deg_s'] = loft_angle_rate_deg_s
        return traj

    effective_guidance = guidance if guidance is not None else params.guidance

    def _run(la, lar):
        try:
            r = integrate_trajectory(
                params, launch_lat_deg, launch_lon_deg, launch_azimuth_deg,
                guidance=guidance,
                loft_angle_deg=la, loft_angle_rate_deg_s=lar,
                cutoff_time_s=total_burn)
            if r['time_of_flight_s'] >= 3599.0:
                return -1.0   # orbital / runaway — not a valid ballistic solution
            return r['range_km']
        except Exception:
            return -1.0

    best_range = -1.0
    best_la  = params.loft_angle_deg
    best_lar = params.loft_angle_rate_deg_s

    if effective_guidance == "gravity_turn":
        # Scan burnout angles 5°–70° (Wheelon linear pitch program).
        # loft_angle_rate_deg_s is unused for gravity_turn; pass 1.0 as placeholder.
        for burnout_a in range(5, 71, 5):
            rng = _run(float(burnout_a), 1.0)
            if rng > best_range:
                best_range = rng
                best_la    = float(burnout_a)
        # Fine search ±4° around best burnout angle
        for d in (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0):
            ba = best_la + d
            if ba < 1.0 or ba > 80.0:
                continue
            rng = _run(ba, 1.0)
            if rng > best_range:
                best_range = rng
                best_la    = ba
        best_lar = 1.0   # placeholder — not used by linear pitch program
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
        guidance=guidance,
        loft_angle_deg=best_la,
        loft_angle_rate_deg_s=best_lar,
        cutoff_time_s=total_burn,
    )
    traj['max_range_km']            = traj['range_km']
    traj['optimal_loft_angle_deg']  = best_la
    traj['optimal_loft_rate_deg_s'] = best_lar
    return traj
