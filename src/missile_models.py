"""
Missile parameter models matching Forden's:
  missileMass.m, missileRadius.m, thrust.m, thrustAngle.m,
  dragForce.m, Drag.m, calcCm_delta.m, aeroQ.m, calcMissileParameters.m

Built-in missile definitions follow Forden (2007) Table 1 parameters for the
four packaged models: Scud-B, Al Hussein, No-dong, and Taepodong-I.
Loft angle / loft angle rate for Scud-B taken from Figure 3 of the same paper.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from atmosphere import atmosphere, dynamic_pressure

_G0 = 9.80665   # standard gravity (m/s²)


def _thrust_from_isp(isp_s: float, propellant_kg: float, burn_s: float) -> float:
    """Vacuum thrust (N) derived from Isp, propellant mass, and burn time."""
    return isp_s * _G0 * propellant_kg / burn_s


@dataclass
class MissileParams:
    """All parameters needed to simulate one missile type."""
    name: str

    # Mass (kg)
    mass_initial: float       # launch mass (structure + propellant + payload)
    mass_propellant: float    # propellant mass
    mass_final: float         # burnout mass (structure + payload)

    # Geometry
    diameter_m: float         # body diameter (m)
    length_m: float           # body length (m)

    # Propulsion
    thrust_N: float           # vacuum thrust (N)
    burn_time_s: float        # powered flight duration (s)
    isp_s: float              # specific impulse (s)

    # Guidance — Forden pitch-over / gravity-turn model (Eq. 8)
    # Elevation from horizontal: starts at 90° (vertical), pitches over at
    # loft_angle_rate_deg_s until it reaches loft_angle_deg, then holds.
    loft_angle_deg: float = 45.0        # final elevation above horizontal (°)
    loft_angle_rate_deg_s: float = 2.0  # pitch-over rate (°/s)

    # Aerodynamics — Cd vs Mach lookup table
    mach_table: list = field(default_factory=list)
    cd_table:   list = field(default_factory=list)

    # Staging (optional second stage)
    stage2: Optional['MissileParams'] = None


# ---------------------------------------------------------------------------
# Shared Cd vs Mach table — Forden Figure 1 piecewise-linear approximation.
# All packaged missiles use this same curve (Forden note 6).
# ---------------------------------------------------------------------------
_FORDEN_MACH = [0.0, 0.85, 1.0,  1.2,  2.0,  4.5]
_FORDEN_CD   = [0.2, 0.20, 0.27, 0.27, 0.20, 0.20]


# ---------------------------------------------------------------------------
# Built-in missile database
# Parameters from Forden (2007) Table 1.  Thrust = Isp * g0 * m_dot.
# mass_initial  = Fueled Weight + Payload
# mass_propellant = Fueled Weight − Dry Weight
# mass_final    = Dry Weight + Payload
# ---------------------------------------------------------------------------

def _scud_b():
    # Forden Table 1: Dry=1198, Fueled=4897, Isp=230, Burn=75, Dia=0.84, Payload=1000
    prop = 4897 - 1198   # = 3699 kg
    return MissileParams(
        name="Scud-B (R-17)",
        mass_initial=4897 + 1000,   # 5897 kg
        mass_propellant=prop,        # 3699 kg
        mass_final=1198 + 1000,      # 2198 kg
        diameter_m=0.84,
        length_m=11.25,
        thrust_N=round(_thrust_from_isp(230, prop, 75)),   # ≈ 111 200 N
        burn_time_s=75.0,
        isp_s=230.0,
        # Loft angle / rate from Forden Figure 3 (SCUD-B example trajectory)
        loft_angle_deg=47.6563,
        loft_angle_rate_deg_s=1.348,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
    )


def _al_hussein():
    # Forden Table 1: Dry=1334, Fueled=6073, Isp=230, Burn=90, Dia=0.84, Payload=191
    prop = 6073 - 1334   # = 4739 kg
    return MissileParams(
        name="Al Hussein",
        mass_initial=6073 + 191,    # 6264 kg
        mass_propellant=prop,        # 4739 kg
        mass_final=1334 + 191,       # 1525 kg
        diameter_m=0.84,
        length_m=12.0,
        thrust_N=round(_thrust_from_isp(230, prop, 90)),   # ≈ 118 900 N
        burn_time_s=90.0,
        isp_s=230.0,
        loft_angle_deg=45.0,
        loft_angle_rate_deg_s=1.0,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
    )


def _nodong():
    # Forden Table 1: Dry=3900, Fueled=19900, Isp=240, Burn=70, Dia=0.88, Payload=1000
    prop = 19900 - 3900  # = 16 000 kg
    return MissileParams(
        name="No-dong",
        mass_initial=19900 + 1000,  # 20 900 kg
        mass_propellant=prop,        # 16 000 kg
        mass_final=3900 + 1000,      # 4 900 kg
        diameter_m=0.88,
        length_m=15.6,
        thrust_N=round(_thrust_from_isp(240, prop, 70)),   # ≈ 537 600 N
        burn_time_s=70.0,
        isp_s=240.0,
        loft_angle_deg=45.0,
        loft_angle_rate_deg_s=1.5,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
    )


def _taepodong_i():
    # Forden Table 1: Stage 1 = Nodong body, Stage 2 = Scud-B body, Payload = 454 kg
    #
    # Stage 2 (ignites after stage-1 separation):
    #   Fueled=4897, Dry=1198, Isp=230, Burn=75, Dia=0.84
    prop2 = 4897 - 1198  # = 3699 kg
    stage2 = MissileParams(
        name="Taepodong-I Stage 2",
        mass_initial=4897 + 454,    # 5 351 kg  (stage-2 wet + payload)
        mass_propellant=prop2,       # 3 699 kg
        mass_final=1198 + 454,       # 1 652 kg  (stage-2 dry + payload)
        diameter_m=0.84,
        length_m=11.25,
        thrust_N=round(_thrust_from_isp(230, prop2, 75)),   # ≈ 111 200 N
        burn_time_s=75.0,
        isp_s=230.0,
        loft_angle_deg=45.0,
        loft_angle_rate_deg_s=1.0,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
    )

    # Stage 1 (launched with stage-2 and payload riding on top):
    #   Fueled=19900, Dry=3800, Isp=240, Burn=70, Dia=0.88
    prop1 = 19900 - 3800  # = 16 100 kg
    return MissileParams(
        name="Taepodong-I",
        # Total stack at launch = stage-1 propellant+structure + stage-2 wet + payload
        mass_initial=19900 + 4897 + 454,   # 25 251 kg
        mass_propellant=prop1,              # 16 100 kg  (stage-1 only)
        mass_final=3800,                    # stage-1 dry (jettisoned at separation)
        diameter_m=0.88,
        length_m=25.5,
        thrust_N=round(_thrust_from_isp(240, prop1, 70)),   # ≈ 540 900 N
        burn_time_s=70.0,
        isp_s=240.0,
        loft_angle_deg=45.0,
        loft_angle_rate_deg_s=1.0,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
        stage2=stage2,
    )


def _shahab3():
    # Iranian Shahab-3, No-dong derivative; not in Forden (2007) but added
    # for completeness.  Parameters from open-source estimates.
    mach = [0.0, 0.85, 1.0, 1.2, 2.0, 4.5]
    cd   = [0.20, 0.20, 0.27, 0.27, 0.20, 0.20]
    prop = 11200
    return MissileParams(
        name="Shahab-3",
        mass_initial=16000.0,
        mass_propellant=prop,
        mass_final=4800.0,
        diameter_m=1.32,
        length_m=16.5,
        thrust_N=round(_thrust_from_isp(230, prop, 97)),
        burn_time_s=97.0,
        isp_s=230.0,
        loft_angle_deg=45.0,
        loft_angle_rate_deg_s=1.5,
        mach_table=mach,
        cd_table=cd,
    )


def _generic_icbm():
    mach = [0.0, 0.85, 1.0, 1.2, 2.0, 4.5, 8.0]
    cd   = [0.18, 0.18, 0.25, 0.25, 0.18, 0.18, 0.14]
    prop2 = 16000
    stage2 = MissileParams(
        name="Generic ICBM Stage 2",
        mass_initial=20000.0,
        mass_propellant=prop2,
        mass_final=4000.0,
        diameter_m=1.5,
        length_m=8.0,
        thrust_N=round(_thrust_from_isp(290, prop2, 120)),
        burn_time_s=120.0,
        isp_s=290.0,
        loft_angle_deg=30.0,
        loft_angle_rate_deg_s=0.5,
        mach_table=mach,
        cd_table=cd,
    )
    prop1 = 55000
    return MissileParams(
        name="Generic ICBM",
        mass_initial=80000.0,
        mass_propellant=prop1,
        mass_final=25000.0,
        diameter_m=2.0,
        length_m=20.0,
        thrust_N=round(_thrust_from_isp(280, prop1, 150)),
        burn_time_s=150.0,
        isp_s=280.0,
        loft_angle_deg=30.0,
        loft_angle_rate_deg_s=0.5,
        mach_table=mach,
        cd_table=cd,
        stage2=stage2,
    )


MISSILE_DB = {
    "Scud-B":        _scud_b,
    "Al Hussein":    _al_hussein,
    "No-dong":       _nodong,
    "Taepodong-I":   _taepodong_i,
    "Shahab-3":      _shahab3,
    "Generic ICBM":  _generic_icbm,
}


def get_missile(name: str) -> MissileParams:
    if name not in MISSILE_DB:
        raise ValueError(f"Unknown missile '{name}'. Available: {list(MISSILE_DB)}")
    return MISSILE_DB[name]()


# ---------------------------------------------------------------------------
# Physics helper functions
# ---------------------------------------------------------------------------

def missile_mass(params: MissileParams, t: float) -> float:
    """Current mass (kg) at time t seconds after launch.  Handles 2-stage."""
    if t <= 0:
        return params.mass_initial
    if t < params.burn_time_s:
        mdot = params.mass_propellant / params.burn_time_s
        return params.mass_initial - mdot * t
    # Stage 1 burnout
    if params.stage2 is None:
        return params.mass_final
    # Stage separation: jettison stage-1 structure, ignite stage 2
    p2 = params.stage2
    t2 = t - params.burn_time_s
    if t2 >= p2.burn_time_s:
        return p2.mass_final
    mdot2 = p2.mass_propellant / p2.burn_time_s
    return p2.mass_initial - mdot2 * t2


def missile_area(params: MissileParams) -> float:
    """Reference cross-sectional area (m^2)."""
    return np.pi * (params.diameter_m / 2) ** 2


def drag_coefficient(params: MissileParams, mach: float) -> float:
    """Cd interpolated from Mach table."""
    return float(np.interp(mach, params.mach_table, params.cd_table))


def drag_force_vector(params: MissileParams, vel_ecef, altitude_m) -> np.ndarray:
    """
    Aerodynamic drag force vector (N) opposing velocity.

    Parameters
    ----------
    params     : MissileParams
    vel_ecef   : velocity vector in ECEF (m/s), shape (3,)
    altitude_m : scalar altitude (m)

    Returns
    -------
    F_drag : ndarray (3,) in Newtons (opposing velocity direction)
    """
    speed = np.linalg.norm(vel_ecef)
    if speed < 1e-6:
        return np.zeros(3)
    _, _, rho, a_sound = atmosphere(altitude_m)
    mach = speed / a_sound
    cd   = drag_coefficient(params, mach)
    area = missile_area(params)
    q    = 0.5 * rho * speed**2
    drag_mag = cd * q * area
    return -drag_mag * (vel_ecef / speed)


def thrust_force(params: MissileParams, t: float, altitude_m: float,
                 thrust_dir: np.ndarray) -> np.ndarray:
    """
    Thrust force vector (N).

    Parameters
    ----------
    params     : MissileParams
    t          : time since launch (s)
    altitude_m : current altitude for ambient pressure correction
    thrust_dir : unit vector in direction of thrust (ECEF)

    Returns
    -------
    F_thrust : ndarray (3,) Newtons
    """
    if t < 0:
        return np.zeros(3)
    # Choose active stage
    if t <= params.burn_time_s:
        active = params
        t_active = t
    elif params.stage2 is not None:
        active = params.stage2
        t_active = t - params.burn_time_s
    else:
        return np.zeros(3)
    if t_active > active.burn_time_s:
        return np.zeros(3)
    # Vacuum thrust corrected for ambient back-pressure (~2% at sea level)
    _, P_amb, _, _ = atmosphere(altitude_m)
    correction = 1.0 - 0.02 * (P_amb / 101325.0)
    thrust_mag = active.thrust_N * correction
    return thrust_mag * thrust_dir
