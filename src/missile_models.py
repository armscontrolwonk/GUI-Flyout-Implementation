"""
Missile parameter models matching Forden's:
  missileMass.m, missileRadius.m, thrust.m, thrustAngle.m,
  dragForce.m, Drag.m, calcCm_delta.m, aeroQ.m, calcMissileParameters.m

Built-in missile definitions follow Forden's published parameters for
Scud-B (R-17), No-dong, Shahab-3, and a generic ICBM.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from atmosphere import atmosphere, dynamic_pressure


@dataclass
class MissileParams:
    """All parameters needed to simulate one missile type."""
    name: str

    # Mass (kg)
    mass_initial: float       # launch mass
    mass_propellant: float    # propellant mass
    mass_final: float         # burnout mass (= mass_initial - mass_propellant)

    # Geometry
    diameter_m: float         # body diameter (m)
    length_m: float           # body length (m)

    # Propulsion
    thrust_N: float           # vacuum thrust (N)
    burn_time_s: float        # powered flight duration (s)
    isp_s: float              # specific impulse (s)

    # Aerodynamics — Cd vs Mach lookup table
    mach_table: list = field(default_factory=list)
    cd_table:   list = field(default_factory=list)

    # Staging (optional second stage)
    stage2: Optional['MissileParams'] = None


# ---------------------------------------------------------------------------
# Built-in missile database
# Values from Forden (2001) "Measures and Provocations" and open literature
# ---------------------------------------------------------------------------

def _scud_b():
    mach = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    cd   = [0.20, 0.20, 0.22, 0.35, 0.40, 0.38, 0.32, 0.25, 0.18]
    return MissileParams(
        name="Scud-B (R-17)",
        mass_initial=5900.0,
        mass_propellant=3700.0,
        mass_final=2200.0,
        diameter_m=0.88,
        length_m=11.25,
        thrust_N=132000.0,
        burn_time_s=65.0,
        isp_s=230.0,
        mach_table=mach,
        cd_table=cd,
    )

def _nodong():
    mach = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    cd   = [0.20, 0.20, 0.22, 0.38, 0.42, 0.38, 0.30, 0.23, 0.16]
    return MissileParams(
        name="No-dong",
        mass_initial=16000.0,
        mass_propellant=11000.0,
        mass_final=5000.0,
        diameter_m=1.32,
        length_m=16.0,
        thrust_N=270000.0,
        burn_time_s=95.0,
        isp_s=228.0,
        mach_table=mach,
        cd_table=cd,
    )

def _shahab3():
    mach = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0]
    cd   = [0.20, 0.20, 0.22, 0.38, 0.42, 0.38, 0.30, 0.23, 0.16]
    return MissileParams(
        name="Shahab-3",
        mass_initial=16000.0,
        mass_propellant=11200.0,
        mass_final=4800.0,
        diameter_m=1.32,
        length_m=16.5,
        thrust_N=275000.0,
        burn_time_s=97.0,
        isp_s=230.0,
        mach_table=mach,
        cd_table=cd,
    )

def _generic_icbm():
    mach = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
    cd   = [0.18, 0.18, 0.20, 0.32, 0.36, 0.32, 0.26, 0.20, 0.14, 0.10]
    stage2 = MissileParams(
        name="Generic ICBM Stage 2",
        mass_initial=20000.0,
        mass_propellant=16000.0,
        mass_final=4000.0,
        diameter_m=1.5,
        length_m=8.0,
        thrust_N=300000.0,
        burn_time_s=120.0,
        isp_s=290.0,
        mach_table=mach,
        cd_table=cd,
    )
    return MissileParams(
        name="Generic ICBM",
        mass_initial=80000.0,
        mass_propellant=55000.0,
        mass_final=25000.0,
        diameter_m=2.0,
        length_m=20.0,
        thrust_N=800000.0,
        burn_time_s=150.0,
        isp_s=280.0,
        mach_table=mach,
        cd_table=cd,
        stage2=stage2,
    )


MISSILE_DB = {
    "Scud-B":       _scud_b,
    "No-dong":      _nodong,
    "Shahab-3":     _shahab3,
    "Generic ICBM": _generic_icbm,
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
