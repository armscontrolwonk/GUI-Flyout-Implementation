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

    # Nozzle exit area (m²).  When > 0, thrust at altitude is computed as
    # T(h) = T_vac − P_amb(h) × Ae  (proper pressure-thrust correction).
    # When 0, a legacy 2 % sea-level back-pressure approximation is used.
    nozzle_exit_area_m2: float = 0.0

    # Guidance mode
    #   "loft"         — Forden pitch-over (SRBM/MRBM): pitch to loft_angle_deg
    #                    at loft_angle_rate_deg_s then hold.  Floor preserved.
    #   "gravity_turn" — kick off vertical to loft_angle_deg at
    #                    loft_angle_rate_deg_s, then lock thrust to velocity
    #                    vector (IRBM/ICBM).  loft_angle_deg here is the kick
    #                    elevation (°, above horizontal; e.g. 85° = 5° from
    #                    vertical) and loft_angle_rate_deg_s is the kick rate.
    guidance: str = "loft"

    loft_angle_deg: float = 45.0        # Forden: final elev (°); GT: kick elev (°)
    loft_angle_rate_deg_s: float = 2.0  # Forden: pitch rate (°/s); GT: kick rate

    # Aerodynamics — Cd vs Mach lookup table
    mach_table: list = field(default_factory=list)
    cd_table:   list = field(default_factory=list)

    # Staging (optional next stage)
    stage2: Optional['MissileParams'] = None

    # Coast time (s) between this stage's burnout and the next stage's ignition.
    # Ignored (irrelevant) for the last / only stage.
    coast_time_s: float = 0.0

    # Payload (kg) — total mass carried to burnout (bus + all RVs).
    # Stored on the top-level stage only so _prefill can round-trip correctly.
    payload_kg: float = 0.0

    # RV ballistic coefficient β = m / (Cd·A) in kg/m².
    # When > 0, the post-burnout coast phase uses β-based drag for the
    # separating RV instead of the final-stage body aerodynamics.
    # Stored on the top-level node only (same convention as payload_kg).
    rv_beta_kg_m2: float = 0.0

    # Payload decomposition: bus (post-boost vehicle) + N reentry vehicles.
    # bus_mass_kg + num_rvs * rv_mass_kg should equal payload_kg.
    # When rv_mass_kg == 0 the decomposition is not specified; terminal drag
    # falls back to payload_kg as a proxy for RV mass.
    bus_mass_kg: float = 0.0
    num_rvs:     int   = 1
    rv_mass_kg:  float = 0.0   # mass of one RV

    # Shroud / payload fairing jettisoned during ascent.
    # shroud_mass_kg is included in mass_initial at launch and subtracted once
    # the missile crosses shroud_jettison_alt_km.  0 = no shroud.
    shroud_mass_kg:        float = 0.0
    shroud_jettison_alt_km: float = 80.0


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


def _taepodong_ii():
    # Forden (2007) discussion / Table 1 extension.
    # 3-stage missile: No-dong first stage, Scud-B second stage,
    # small solid-fuel third stage.  Payload ≈ 500 kg.
    #
    # Stage 3 (solid-fuel upper stage):
    prop3 = 1400  # fueled 1600 - dry 200
    stage3 = MissileParams(
        name="Taepodong-II Stage 3",
        mass_initial=1600 + 500,    # 2 100 kg (wet + payload)
        mass_propellant=prop3,       # 1 400 kg
        mass_final=200 + 500,        # 700 kg (dry + payload)
        diameter_m=0.60,
        length_m=4.0,
        thrust_N=round(_thrust_from_isp(275, prop3, 50)),
        burn_time_s=50.0,
        isp_s=275.0,
        loft_angle_deg=25.0,
        loft_angle_rate_deg_s=0.5,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
    )

    # Stage 2 (Scud-B derived):
    prop2 = 4897 - 1198  # = 3 699 kg
    stage2 = MissileParams(
        name="Taepodong-II Stage 2",
        mass_initial=4897 + stage3.mass_initial,   # 6 997 kg
        mass_propellant=prop2,                      # 3 699 kg
        mass_final=1198,                            # stage-2 dry only (jettisoned)
        diameter_m=0.84,
        length_m=11.25,
        thrust_N=round(_thrust_from_isp(230, prop2, 75)),
        burn_time_s=75.0,
        isp_s=230.0,
        loft_angle_deg=35.0,
        loft_angle_rate_deg_s=0.6,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
        stage2=stage3,
    )

    # Stage 1 (No-dong derived):
    prop1 = 19900 - 3800  # = 16 100 kg
    return MissileParams(
        name="Taepodong-II",
        mass_initial=19900 + stage2.mass_initial,   # 26 797 kg
        mass_propellant=prop1,                       # 16 100 kg
        mass_final=3800,                             # stage-1 dry (jettisoned)
        diameter_m=0.88,
        length_m=32.0,
        thrust_N=round(_thrust_from_isp(240, prop1, 70)),
        burn_time_s=70.0,
        isp_s=240.0,
        loft_angle_deg=40.0,
        loft_angle_rate_deg_s=0.8,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
        stage2=stage2,
    )


def _zoljanah():
    # Iranian Zoljanah (Zuljanah) space launch vehicle / ballistic missile.
    # 2-stage liquid-propellant vehicle with 100 kg RV payload.
    #
    # Mass model: total launch mass = 48 000 kg, payload = 100 kg.
    # Both stages share the remaining 47 900 kg equally → 23 950 kg wet each.
    # Dry mass fraction = 12 % per stage (improved structural efficiency):
    #   Dry  = 0.12 × 23 950 =  2 874 kg
    #   Prop = 0.88 × 23 950 = 21 076 kg
    #
    # Propulsion (open-source estimates):
    #   Stage 1:  ISP=255 s, Burn=71 s, Dia=1.5 m, Length=10.3 m
    #   Stage 2:  ISP=265 s, Burn=71 s, Dia=1.5 m, Length=9.9 m
    #   Payload:  100 kg RV (separates at stage-2 burnout)

    payload = 100    # kg  RV
    dry     = 2874   # kg  per stage  (12 % of 23 950 kg wet)
    prop    = 21076  # kg  per stage  (88 % of 23 950 kg wet)

    # Burn time: same engine as original (thrust unchanged), so burn time
    # scales with propellant mass: 71 s × (21 076 / 20 500) ≈ 73.0 s
    burn    = round(71.0 * prop / 20500, 1)   # 73.0 s

    isp1    = 255.0                # Stage 1 ISP (s) — near sea level
    isp2    = round(isp1 * 1.15)  # Stage 2 ISP +15 % vacuum bonus → 293 s
    # Thrust scales with ISP at the same mass flow rate, so stage 2 thrust
    # is also 15 % higher than stage 1 for no extra cost.

    # ── Stage 2 (liquid, last stage — carries RV payload) ────────────────────
    stage2 = MissileParams(
        name="Zoljanah (IRBM) Stage 2",
        mass_initial=prop + dry + payload,   # 24 050 kg
        mass_propellant=prop,                # 21 076 kg
        mass_final=dry,                      #  2 874 kg (dry only; RV separates)
        diameter_m=1.5,
        length_m=9.9,
        thrust_N=round(_thrust_from_isp(isp2, prop, burn)),
        burn_time_s=burn,
        isp_s=isp2,
        loft_angle_deg=25.0,
        loft_angle_rate_deg_s=0.5,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
    )

    # ── Stage 1 (liquid) ─────────────────────────────────────────────────────
    # RV ballistic coefficient estimate: cone-shaped RV, dia≈0.6 m, Cd≈0.25
    #   β = m / (Cd·A) = 100 / (0.25 · π·0.09) ≈ 1 415 kg/m²
    rv_beta = 1400.0   # kg/m²  (round estimate)

    p = MissileParams(
        name="Zoljanah (IRBM)",
        mass_initial=prop + dry + stage2.mass_initial,   # 48 000 kg
        mass_propellant=prop,                            # 21 076 kg
        mass_final=dry,                                  #  2 874 kg (jettisoned)
        diameter_m=1.5,
        length_m=10.3,
        thrust_N=round(_thrust_from_isp(isp1, prop, burn)),
        burn_time_s=burn,
        isp_s=isp1,
        loft_angle_deg=35.0,
        loft_angle_rate_deg_s=0.8,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
        stage2=stage2,
        payload_kg=float(payload),
        rv_beta_kg_m2=rv_beta,
    )
    return p


def _zoljanah_slv():
    # Iranian Zoljanah (Zuljanah) Space Launch Vehicle — 3-stage configuration.
    #
    # Mass budget (total launch mass = 51 935 kg):
    #   Stages 1 & 2: solid propellant
    #     prop = 20 517 kg, dry = 3 600 kg, fueled = 24 117 kg each
    #   Stage 3: liquid (NTO/UDMH, Vacuum R-27 verniers)
    #     prop = 3 050 kg, dry = 387 kg, fueled = 3 437 kg, burn = 300 s
    #   Satellite payload:  220 kg
    #   Payload fairing:     44 kg  (jettisoned at stage-3 ignition — carried in
    #                                stage-2 mass_initial; absent from stage-3 chain)
    #   Total at launch:   51 935 kg
    #     = 24 117 (S1) + 24 117+44 (S2+shroud) + 3 437 (S3) + 220 (sat)
    #
    # Propulsion:
    #   Stage 1 (solid): ISP=255.837 s, thrust≈725 kN, burn=71 s, nozzle=0.90 m²
    #                    post-burnout coast = 12 s
    #   Stage 2 (solid): ISP=265 s,     thrust≈751 kN, burn=71 s, nozzle=1.77 m²
    #                    post-burnout coast = 30 s
    #   Stage 3 (liq):   ISP=293 s,     thrust≈ 29 kN, burn=300 s, nozzle=0.14 m²
    #                    (2 × 0.3 m verniers; mass flow ≈ 10.2 kg/s)
    #
    # Reference orbit: 493 × 490 km, azimuth 140°, lat 35.238°
    # Guidance angles are first-estimate; tune against observed trajectory data.

    satellite =  220   # kg — payload to orbit
    shroud    =   44   # kg — fairing, jettisoned at stage-3 ignition
                       #      (built into stage-2 mass_initial; not in stage-3)

    # ── Stage 3 (liquid, R-27 verniers, NTO/UDMH) ───────────────────────────
    prop3  = 3_050
    dry3   =   387
    isp3   = 293.0
    burn3  = 300.0   # s  (mass flow = 3050/300 ≈ 10.2 kg/s)

    stage3 = MissileParams(
        name="Zoljanah (SLV) Stage 3",
        mass_initial=dry3 + prop3 + satellite,    # 3 657 kg
        mass_propellant=prop3,                     # 3 050 kg
        mass_final=dry3 + satellite,               #   607 kg  (dry + sat)
        diameter_m=1.25,
        length_m=3.3,
        thrust_N=round(_thrust_from_isp(isp3, prop3, burn3)),   # ≈ 29 200 N
        burn_time_s=burn3,
        isp_s=isp3,
        nozzle_exit_area_m2=0.14,    # 2 × 0.3 m diameter verniers
        guidance="gravity_turn",
        loft_angle_deg=15.0,
        loft_angle_rate_deg_s=0.5,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
    )

    # ── Stage 2 (solid) ──────────────────────────────────────────────────────
    # shroud (44 kg) included here; jettisoned implicitly when stage 3 ignites
    # and stage-2 chain hands off to stage3.mass_initial (which excludes shroud).
    dry2  = 3_600
    prop2 = 20_517
    isp2  = 265.0

    stage2 = MissileParams(
        name="Zoljanah (SLV) Stage 2",
        mass_initial=dry2 + prop2 + stage3.mass_initial + shroud,   # 27 818 kg
        mass_propellant=prop2,                                        # 20 517 kg
        mass_final=dry2,                                              #  3 600 kg  (jettisoned)
        diameter_m=1.5,
        length_m=8.7,    # stage body only (+ 1.0 m interstage = 9.7 m total)
        thrust_N=round(_thrust_from_isp(isp2, prop2, 71.0)),   # ≈ 751 000 N
        burn_time_s=71.0,
        isp_s=isp2,
        nozzle_exit_area_m2=1.77,
        guidance="gravity_turn",
        loft_angle_deg=45.0,
        loft_angle_rate_deg_s=0.5,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
        stage2=stage3,
        coast_time_s=30.0,
    )

    # ── Stage 1 (solid) ──────────────────────────────────────────────────────
    dry1  = 3_600
    prop1 = 20_517
    isp1  = 255.837

    return MissileParams(
        name="Zoljanah (SLV)",
        mass_initial=dry1 + prop1 + stage2.mass_initial,   # 51 935 kg
        mass_propellant=prop1,                               # 20 517 kg
        mass_final=dry1,                                     #  3 600 kg  (jettisoned)
        diameter_m=1.5,
        length_m=10.3,   # stage body only (+ 0.75 m interstage = 11.05 m total)
        thrust_N=round(_thrust_from_isp(isp1, prop1, 71.0)),   # ≈ 725 000 N
        burn_time_s=71.0,
        isp_s=isp1,
        nozzle_exit_area_m2=0.90,
        guidance="gravity_turn",
        loft_angle_deg=85.0,
        loft_angle_rate_deg_s=0.5,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
        stage2=stage2,
        coast_time_s=12.0,
        payload_kg=float(satellite),
    )


MISSILE_DB = {
    # Non-Forden missiles (user-editable)
    "Zoljanah (SLV)":         _zoljanah_slv,
    "Zoljanah (IRBM)":        _zoljanah,
    "Shahab-3":               _shahab3,
    "Generic ICBM":           _generic_icbm,
    # Forden (2007) Table 1 reference missiles — read-only validation targets
    "Scud-B (Forden)":        _scud_b,
    "Al Hussein (Forden)":    _al_hussein,
    "No-dong (Forden)":       _nodong,
    "Taepodong-I (Forden)":   _taepodong_i,
    "Taepodong-II (Forden)":  _taepodong_ii,
}


def get_missile(name: str) -> MissileParams:
    if name not in MISSILE_DB:
        raise ValueError(f"Unknown missile '{name}'. Available: {list(MISSILE_DB)}")
    return MISSILE_DB[name]()


def missile_to_dict(p: MissileParams) -> dict:
    """Serialise a MissileParams to a JSON-compatible dict."""
    d = {
        'name':                  p.name,
        'mass_initial':          p.mass_initial,
        'mass_propellant':       p.mass_propellant,
        'diameter_m':            p.diameter_m,
        'length_m':              p.length_m,
        'burn_time_s':           p.burn_time_s,
        'coast_time_s':          p.coast_time_s,
        'isp_s':                 p.isp_s,
        'guidance':               p.guidance,
        'loft_angle_deg':        p.loft_angle_deg,
        'loft_angle_rate_deg_s': p.loft_angle_rate_deg_s,
        'mach_table':            list(p.mach_table),
        'cd_table':              list(p.cd_table),
        'payload_kg':            p.payload_kg,
        'rv_beta_kg_m2':         p.rv_beta_kg_m2,
        'bus_mass_kg':           p.bus_mass_kg,
        'num_rvs':               p.num_rvs,
        'rv_mass_kg':            p.rv_mass_kg,
        'shroud_mass_kg':        p.shroud_mass_kg,
        'shroud_jettison_alt_km': p.shroud_jettison_alt_km,
        'nozzle_exit_area_m2':   p.nozzle_exit_area_m2,
    }
    if p.stage2 is not None:
        d['stage2'] = missile_to_dict(p.stage2)
    return d


def missile_from_dict(d: dict) -> MissileParams:
    """Reconstruct a MissileParams from a dict produced by missile_to_dict."""
    prop  = float(d['mass_propellant'])
    burn  = float(d['burn_time_s'])
    isp   = float(d['isp_s'])
    m0    = float(d['mass_initial'])
    stage2 = missile_from_dict(d['stage2']) if d.get('stage2') else None
    return MissileParams(
        name=d['name'],
        mass_initial=m0,
        mass_propellant=prop,
        mass_final=m0 - prop,
        diameter_m=float(d['diameter_m']),
        length_m=float(d['length_m']),
        thrust_N=round(_thrust_from_isp(isp, prop, burn)),
        burn_time_s=burn,
        coast_time_s=float(d.get('coast_time_s', 0.0)),
        isp_s=isp,
        guidance=d.get('guidance', 'loft'),
        loft_angle_deg=float(d.get('loft_angle_deg', 45.0)),
        loft_angle_rate_deg_s=float(d.get('loft_angle_rate_deg_s', 2.0)),
        mach_table=list(d.get('mach_table', _FORDEN_MACH)),
        cd_table=list(d.get('cd_table', _FORDEN_CD)),
        stage2=stage2,
        payload_kg=float(d.get('payload_kg', 0.0)),
        rv_beta_kg_m2=float(d.get('rv_beta_kg_m2', 0.0)),
        bus_mass_kg=float(d.get('bus_mass_kg', 0.0)),
        num_rvs=int(d.get('num_rvs', 1)),
        rv_mass_kg=float(d.get('rv_mass_kg', 0.0)),
        shroud_mass_kg=float(d.get('shroud_mass_kg', 0.0)),
        shroud_jettison_alt_km=float(d.get('shroud_jettison_alt_km', 80.0)),
        nozzle_exit_area_m2=float(d.get('nozzle_exit_area_m2', 0.0)),
    )


# ---------------------------------------------------------------------------
# Physics helper functions
# ---------------------------------------------------------------------------

def total_burn_time(params: MissileParams) -> float:
    """Total time from launch to end of last stage's burn (burn + coast phases)."""
    t, s = 0.0, params
    while s is not None:
        t += s.burn_time_s
        if s.stage2 is not None:
            t += s.coast_time_s   # inter-stage coast before next ignition
        s  = s.stage2
    return t


def active_stage(params: MissileParams, t: float) -> MissileParams:
    """Return the MissileParams for the stage (or vehicle) active at time t.

    During powered flight this is the burning stage.  During a coast phase
    it is the next (upper) stage — stage N has been jettisoned and the
    remaining vehicle has stage N+1's geometry.  After all stages have fired
    it is the last stage (used for drag during the ballistic coast/re-entry).
    """
    t_rem, s = t, params
    while s.stage2 is not None:
        if t_rem < s.burn_time_s:
            return s
        t_rem -= s.burn_time_s
        if t_rem < s.coast_time_s:
            return s.stage2   # coasting: stage s jettisoned, next is the vehicle
        t_rem -= s.coast_time_s
        s      = s.stage2
    return s   # last stage (or only stage)


def active_stage_and_t(params: MissileParams, t: float):
    """Return (active_stage, t_since_ignition) for time t.

    t_since_ignition is the time elapsed since the returned stage ignited,
    used to evaluate per-stage pitch-over guidance.  During a coast phase
    the next stage is returned with t_since_ignition = 0.
    """
    t_rem, s = t, params
    while s.stage2 is not None:
        if t_rem < s.burn_time_s:
            return s, t_rem
        t_rem -= s.burn_time_s
        if t_rem < s.coast_time_s:
            return s.stage2, 0.0   # coasting; next stage not yet ignited
        t_rem -= s.coast_time_s
        s      = s.stage2
    return s, t_rem   # last stage


def missile_mass(params: MissileParams, t: float, alt_m: float = 0.0) -> float:
    """Current mass (kg) at time t seconds after launch.  Handles N stages.

    alt_m is the current altitude in metres; it is used to determine whether
    a shroud has been jettisoned (when alt_m / 1000 >= shroud_jettison_alt_km).
    """
    if t <= 0:
        return params.mass_initial
    t_rem, s = t, params
    while s is not None:
        if t_rem < s.burn_time_s:
            mdot = s.mass_propellant / s.burn_time_s
            mass = s.mass_initial - mdot * t_rem
            # Subtract shroud once jettison altitude is crossed (during powered flight)
            if (params.shroud_mass_kg > 0
                    and alt_m / 1000.0 >= params.shroud_jettison_alt_km):
                mass -= params.shroud_mass_kg
            return mass
        t_rem -= s.burn_time_s
        if s.stage2 is None:
            # RV separates at last-stage burnout; coast on payload mass if known.
            # (Shroud is assumed already jettisoned before burnout.)
            return params.payload_kg if params.payload_kg > 0 else s.mass_final
        if t_rem < s.coast_time_s:
            return s.stage2.mass_initial   # stage s jettisoned, next stage is vehicle
        t_rem -= s.coast_time_s
        s = s.stage2
    return params.mass_final  # shouldn't reach here


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
    Thrust force vector (N).  Handles N stages.

    Parameters
    ----------
    params     : MissileParams (stage-1 node of the linked list)
    t          : time since launch (s)
    altitude_m : current altitude for ambient pressure correction
    thrust_dir : unit vector in direction of thrust (ECEF)

    Returns
    -------
    F_thrust : ndarray (3,) Newtons
    """
    if t < 0:
        return np.zeros(3)
    t_rem, s = t, params
    while s is not None:
        if t_rem <= s.burn_time_s:
            _, P_amb, _, _ = atmosphere(altitude_m)
            if s.nozzle_exit_area_m2 > 0:
                # Physics-based correction: T(h) = T_vac − P_amb × Ae
                thrust_mag = max(0.0, s.thrust_N - P_amb * s.nozzle_exit_area_m2)
            else:
                # Legacy approximation: ~2 % back-pressure penalty at sea level
                thrust_mag = s.thrust_N * (1.0 - 0.02 * (P_amb / 101325.0))
            return thrust_mag * thrust_dir
        t_rem -= s.burn_time_s
        if s.stage2 is None:
            return np.zeros(3)
        if t_rem <= s.coast_time_s:
            return np.zeros(3)   # coasting between stages
        t_rem -= s.coast_time_s
        s      = s.stage2
    return np.zeros(3)  # all stages burned out
