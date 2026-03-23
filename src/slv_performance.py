"""
Space launch vehicle performance estimation using the Schilling/Townsend method.

Reference: John Schilling, "Launch Vehicle Performance Estimation", 3 December 2009.
           https://silverbirdastronautics.com/LVperform.html

The method answers whether a launch vehicle can deliver a stated payload to a
specified circular parking orbit, using an algebraic delta-V budget with an
empirical gravity/drag/steering-loss correction derived from ascent time.

Accuracy: ~260 m/s RMS error in total mission delta-V; typically <10 % error
in payload capacity.  No trajectory integration is required — the calculation
is purely algebraic and completes in milliseconds.

Schilling equations used
------------------------
(3)  ΔV_req  = V_circ + ΔV_pen − V_rot
(4)  T_3s    = 3 [1 − exp(−0.333 ΔV_avail / g Isp)] g Isp / A₀
(5)  ΔV_pen  = K₃ + K₄ T_mix
         K₃ = 429.9 + 1.602 Hₚ + 1.224×10⁻³ Hₚ²
         K₄ = 2.328 − 9.687×10⁻⁴ Hₚ      (Hₚ = parking orbit altitude, km)
(6)  T_mix   = 0.405 T_actual + 0.595 T_3s
"""

import numpy as np
from missile_models import MissileParams, total_burn_time
from coordinates import OMEGA_EARTH

_G0      = 9.80665          # standard gravity, m/s²
_MU      = 3.986004418e14   # Earth gravitational parameter, m³/s²
_R_EARTH = 6371000.0        # mean Earth radius, m


# ---------------------------------------------------------------------------
# Per-stage and full-stack delta-V (Tsiolkovsky rocket equation, vacuum)
# ---------------------------------------------------------------------------

def stage_delta_v(stage: MissileParams) -> float:
    """
    Vacuum delta-V (m/s) for one stage via the Tsiolkovsky rocket equation.

    The burnout mass is mass_initial − mass_propellant, which equals the total
    stack mass (upper stages + payload + dry hardware) remaining after this
    stage's propellant is exhausted.  For intermediate stages this is larger
    than mass_final (the jettisoned dry shell), so mass_final must NOT be
    used here.
    """
    m_burnout = stage.mass_initial - stage.mass_propellant
    if m_burnout <= 0.0 or stage.mass_initial <= m_burnout:
        return 0.0
    return _G0 * stage.isp_s * np.log(stage.mass_initial / m_burnout)


def total_delta_v(params: MissileParams) -> float:
    """Total vacuum delta-V (m/s) summed across all stages."""
    dv, s = 0.0, params
    while s is not None:
        dv += stage_delta_v(s)
        s = s.stage2
    return dv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _circular_speed(alt_km: float) -> float:
    """Circular orbital speed (m/s) at altitude alt_km."""
    return np.sqrt(_MU / (_R_EARTH + alt_km * 1_000.0))


def _earth_rotation_benefit(launch_lat_deg: float, launch_az_deg: float) -> float:
    """
    Velocity benefit (m/s) from Earth's rotation projected along the launch
    azimuth.  Positive (prograde) for eastward launches, negative (retrograde)
    for westward launches.
    """
    lat_rad = np.radians(launch_lat_deg)
    az_rad  = np.radians(launch_az_deg)
    v_east  = OMEGA_EARTH * _R_EARTH * np.cos(lat_rad)
    return v_east * np.sin(az_rad)


def _penalty_dv(alt_km: float, t_mix: float) -> float:
    """Schilling eq. 5 — gravity + drag + steering loss estimate (m/s)."""
    Hp = alt_km
    K3 = 429.9 + 1.602 * Hp + 1.224e-3 * Hp ** 2
    K4 = 2.328 - 9.687e-4 * Hp
    return K3 + K4 * t_mix


def _t3stage(dv_total: float, isp_s: float, a0_ms2: float) -> float:
    """
    Schilling eq. 4 — equivalent ascent time (s) of a hypothetical uniform
    three-stage vehicle with the same total ΔV, Isp, and initial acceleration.
    Used to correct for modern dissimilar-staging designs.
    """
    if a0_ms2 <= 0.0 or isp_s <= 0.0:
        return 0.0
    exponent = -0.333 * dv_total / (_G0 * isp_s)
    return 3.0 * (1.0 - np.exp(exponent)) * _G0 * isp_s / a0_ms2


def _dv_for_extra_payload(params: MissileParams, extra_kg: float) -> float:
    """
    Total vacuum ΔV (m/s) when the payload is changed by *extra_kg* kg
    relative to params.payload_kg.

    A positive extra_kg means more payload; negative means less.  The extra
    mass propagates through every stage's mass_initial (all stages carry it)
    and also through the last stage's mass_final (payload rides to the end).
    Intermediate-stage mass_final values are unchanged because those stages
    jettison only their own dry hardware.
    """
    stages: list[MissileParams] = []
    s = params
    while s is not None:
        stages.append(s)
        s = s.stage2

    dv = 0.0
    for s in stages:
        m0        = s.mass_initial + extra_kg
        m_burnout = m0 - s.mass_propellant   # same formula for every stage
        if m_burnout > 0.0 and m0 > m_burnout:
            dv += _G0 * s.isp_s * np.log(m0 / m_burnout)
    return dv


def _compute_t_mix(params: MissileParams, dv_avail: float,
                   t_actual: float, a0: float) -> tuple[float, float]:
    """Return (T_3s, T_mix) for the given vehicle state."""
    t3s   = _t3stage(dv_avail, params.isp_s, a0)
    t_mix = 0.405 * t_actual + 0.595 * t3s
    return t3s, t_mix


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def schilling_performance(params: MissileParams,
                           target_alt_km: float,
                           launch_lat_deg: float = 0.0,
                           launch_az_deg: float = 90.0) -> dict:
    """
    Estimate whether an SLV can deliver its payload to a circular parking
    orbit using the Schilling/Townsend algebraic method.

    Parameters
    ----------
    params         : MissileParams  — full SLV stage stack; params.payload_kg
                     is treated as the *claimed* payload for margin reporting.
    target_alt_km  : target circular orbit altitude (km)
    launch_lat_deg : launch-site geodetic latitude (degrees)
    launch_az_deg  : launch azimuth clockwise from North (degrees);
                     90° = due east (maximum rotation benefit)

    Returns
    -------
    dict with keys
        dv_available_ms   : total vacuum ΔV from rocket equation (m/s)
        dv_required_ms    : required ΔV to reach orbit (m/s)
        dv_margin_ms      : surplus (+) or deficit (−) ΔV (m/s)
        can_reach_orbit   : bool
        v_circular_ms     : circular orbit speed at target altitude (m/s)
        dv_penalty_ms     : estimated gravity + drag + steering loss (m/s)
        v_rotation_ms     : Earth-rotation velocity benefit (m/s)
        t_actual_s        : total vehicle burn time (s)
        t_3stage_s        : Schilling equivalent 3-stage ascent time (s)
        t_mix_s           : blended ascent time used in penalty formula (s)
        a0_ms2            : initial thrust/weight acceleration (m/s²)
        max_payload_kg    : maximum payload deliverable to this orbit (kg)
        payload_margin_kg : max_payload − claimed payload (kg); None if no
                            claimed payload is set
    """
    t_actual = total_burn_time(params)
    a0       = params.thrust_N / params.mass_initial   # m/s²
    v_circ   = _circular_speed(target_alt_km)
    v_rot    = _earth_rotation_benefit(launch_lat_deg, launch_az_deg)

    # Nominal delta-V budget — iterate T_mix / ΔV_pen to convergence
    dv_avail = total_delta_v(params)
    t3s, t_mix = _compute_t_mix(params, dv_avail, t_actual, a0)
    for _ in range(4):
        dv_pen = _penalty_dv(target_alt_km, t_mix)
        dv_req = v_circ + dv_pen - v_rot
        t3s, t_mix = _compute_t_mix(params, dv_avail, t_actual, a0)

    dv_pen    = _penalty_dv(target_alt_km, t_mix)
    dv_req    = v_circ + dv_pen - v_rot
    dv_margin = dv_avail - dv_req

    # Maximum payload — binary search on extra_kg until ΔV_avail == ΔV_req.
    # extra_kg is measured from params.payload_kg.
    extra_lo = -params.payload_kg   # zero payload lower bound
    extra_hi =  200_000.0           # 200 t — always infeasible upper bound

    # Check whether zero payload can reach orbit at all
    dv_zero   = _dv_for_extra_payload(params, extra_lo)
    t3s_z, t_mix_z = _compute_t_mix(params, dv_zero, t_actual, a0)
    dv_req_z  = v_circ + _penalty_dv(target_alt_km, t_mix_z) - v_rot

    if dv_zero < dv_req_z:
        max_payload_kg = 0.0
    else:
        for _ in range(50):   # 2^-50 ≈ 1e-15 relative error — converges in ~40
            extra_mid = 0.5 * (extra_lo + extra_hi)
            dv_mid    = _dv_for_extra_payload(params, extra_mid)
            t3s_m, t_mix_m = _compute_t_mix(params, dv_mid, t_actual, a0)
            dv_req_m  = v_circ + _penalty_dv(target_alt_km, t_mix_m) - v_rot
            if dv_mid >= dv_req_m:
                extra_lo = extra_mid
            else:
                extra_hi = extra_mid
        max_payload_kg = max(0.0, params.payload_kg + 0.5 * (extra_lo + extra_hi))

    payload_margin = (max_payload_kg - params.payload_kg
                      if params.payload_kg > 0 else None)

    return {
        'dv_available_ms':  dv_avail,
        'dv_required_ms':   dv_req,
        'dv_margin_ms':     dv_margin,
        'can_reach_orbit':  dv_margin >= 0.0,
        'v_circular_ms':    v_circ,
        'dv_penalty_ms':    dv_pen,
        'v_rotation_ms':    v_rot,
        't_actual_s':       t_actual,
        't_3stage_s':       t3s,
        't_mix_s':          t_mix,
        'a0_ms2':           a0,
        'max_payload_kg':   max_payload_kg,
        'payload_margin_kg': payload_margin,
    }
