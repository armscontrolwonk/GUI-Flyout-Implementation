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
    launch_elevation_deg: float = 90.0  # elevation at liftoff (°); 90 = vertical

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
    bus_mass_kg:   float = 0.0
    num_rvs:       int   = 1
    rv_mass_kg:    float = 0.0   # mass of one RV

    # When True the RV/payload separates from the last-stage body at burnout.
    # The empty stage body then follows a tumbling-cylinder debris arc.
    # When False (default) the stage body stays attached to the warhead
    # (e.g. Scud-B) and no separate stage debris is computed.
    rv_separates:  bool  = False

    # When True this stage uses a solid rocket motor that cannot be shut off.
    # Orbital insertion guidance runs the engine to natural burnout and reports
    # the resulting orbit rather than commanding a cutoff at the target energy.
    solid_motor:   bool  = False

    # Solid-motor grain profile (Shafer 1959).
    # grain_type: canonical key from _GRAIN_CURVES, or "" for liquid / constant.
    # thrust_peak_N: peak vacuum thrust (N); 0 = derive from thrust_N.
    # thrust_profile: bespoke [(t_frac, F_frac), ...] list; overrides grain_type.
    grain_type:     str  = ""
    thrust_peak_N:  float = 0.0
    thrust_profile: list = field(default_factory=list)

    # ── Per-stage advanced pitch program (optional) ──────────────────────────
    # When set on a stage, these override the top-level turn_start / turn_stop /
    # burnout_angle for that stage's burn interval.  None = use global values.
    # Stored on each stage object in the chain independently so stages can have
    # different pitch schedules (e.g. Stage 1 aggressive pitch-over, Stage 3
    # horizontal burn for orbital insertion).
    stage_turn_start_s:      Optional[float] = None
    stage_turn_stop_s:       Optional[float] = None
    stage_burnout_angle_deg: Optional[float] = None

    # Per-stage yaw (dogleg) overrides.  Same priority as pitch overrides:
    # if stage_yaw_final_az_deg is not None the stage performs a linear
    # azimuth schedule from current az to final_az over [yaw_start, yaw_stop].
    stage_yaw_start_s:     Optional[float] = None
    stage_yaw_stop_s:      Optional[float] = None
    stage_yaw_final_az_deg: Optional[float] = None

    # Shroud jettisoned during ascent.
    # shroud_mass_kg is included in mass_initial at launch and subtracted once
    # the missile crosses shroud_jettison_alt_km.  0 = no shroud.
    shroud_mass_kg:         float = 0.0
    shroud_jettison_alt_km: float = 80.0
    # Physical dimensions of the shroud — used for drag (pre-jettison reference
    # area uses shroud_diameter_m when > 0) and debris tumbling-cylinder β.
    shroud_length_m:        float = 0.0
    shroud_diameter_m:      float = 0.0   # outer diameter of shroud/fairing (m)

    # Nose-shape aerodynamics (Chin/NACA decomposed model).
    # "" = no shape set → drag_force_vector falls back to mach_table/cd_table.
    # L/D is computed internally as nose_length_m / diameter_m (or shroud_diameter_m).
    # 0 = not specified → L/D defaults to 3.0 inside _cd_nose_shape.
    nose_shape:             str   = ""
    nose_length_m:          float = 0.0    # physical nose-cone length (m)
    shroud_nose_shape:      str   = ""
    shroud_nose_length_m:   float = 0.0    # physical shroud nose-cone length (m)

    # Payload diameter (m).  When > 0, used as the frontal reference diameter
    # for aerodynamic drag after shroud jettison (or throughout flight when no
    # shroud is fitted).  Falls back to the stage body diameter_m when 0.
    payload_diameter_m:     float = 0.0

    # RV geometry — stored for round-tripping the β calculator dialog.
    # These do not directly affect the trajectory equations of motion;
    # they are used only to pre-populate the "Estimate β" popup.
    rv_shape:      str   = ""
    rv_diameter_m: float = 0.0
    rv_length_m:   float = 0.0

    # Post-boost vehicle (PBV) geometry — mass is already carried in bus_mass_kg.
    pbv_diameter_m: float = 0.0
    pbv_length_m:   float = 0.0


# ---------------------------------------------------------------------------
# Shared Cd vs Mach table — Forden Figure 1 piecewise-linear approximation.
# All packaged missiles use this same curve (Forden note 6).
# ---------------------------------------------------------------------------
_FORDEN_MACH = [0.0, 0.85, 1.0,  1.2,  2.0,  4.5]
_FORDEN_CD   = [0.2, 0.20, 0.27, 0.27, 0.20, 0.20]

# ---------------------------------------------------------------------------
# Decomposed drag model  (Chin 1961; NACA TN 4201; Crowell 1996)
# Cd_total = Cd_wave_nose + Cd_friction + Cd_base
# ---------------------------------------------------------------------------

NOSE_SHAPES = ["cone", "tangent_ogive", "von_karman",
               "lv_haack", "parabola", "blunt_cylinder"]

NOSE_SHAPE_LABELS = {
    "cone":           "Cone",
    "tangent_ogive":  "Tangent Ogive",
    "von_karman":     "Von Kármán (LD-Haack)",
    "lv_haack":       "LV-Haack (Sears-Haack)",
    "parabola":       "Parabola",
    "blunt_cylinder": "Blunt Cylinder",
}

# Backwards-compatibility aliases for configurations saved with old shape names.
_SHAPE_ALIAS = {
    "conical":    "cone",
    "parabolic":  "parabola",
    "sears_haack":"lv_haack",
    "v2":         "tangent_ogive",
    "elliptical": "cone",
}

# Tabulated wave drag (Cd_wave) at reference l/d_nose = 3.0.
# Source: NACA TN 4201 comparison data (models 56-63, l/d_nose=3, M=0.8-2.0)
# calibrated against Chin (1961) cone formula to isolate wave component.
# Scaled to actual ld via (ld_ref/ld)^2 from slender-body theory.
_WAVE_MACH   = [0.0,  0.6,  0.8,  0.9,  1.0,  1.1,  1.2,  1.5,  2.0,  3.0,  4.0,  5.0]
_WAVE_VK     = [0.000, 0.000, 0.000, 0.010, 0.030, 0.050, 0.060, 0.069, 0.067, 0.058, 0.052, 0.047]
_WAVE_LVH    = [0.000, 0.000, 0.010, 0.030, 0.070, 0.082, 0.085, 0.084, 0.077, 0.068, 0.061, 0.055]
_WAVE_PARA   = [0.000, 0.000, 0.010, 0.040, 0.090, 0.100, 0.100, 0.094, 0.087, 0.077, 0.069, 0.062]
_WAVE_LD_REF = 3.0

# Base pressure coefficient (Cpb < 0) vs Mach, power-off — Chin Fig. 3-15.
_BASE_MACH = [0.0,    0.8,   1.0,   1.2,   1.5,   2.0,   2.5,   3.0,   4.0,   5.0]
_BASE_CPB  = [0.000, -0.13, -0.20, -0.18, -0.14, -0.10, -0.08, -0.06, -0.05, -0.04]


# ---------------------------------------------------------------------------
# Solid-rocket-motor grain profiles  (Shafer 1959, Ch.16, Space Technology)
# Normalised (t/burn_time, F/F_peak) piecewise-linear curves.
# ---------------------------------------------------------------------------
_GRAIN_CURVES = {
    # Progressive: growing internal port — thrust rises through burn.
    "tubular":          [(0.0, 0.700), (0.25, 0.775), (0.50, 0.850),
                         (0.75, 0.925), (1.0, 1.000)],
    # Neutral: rod + annular tube areas cancel — nearly flat.
    "rod_tube":         [(0.0, 1.000), (0.50, 1.000), (1.0, 0.970)],
    # Regressive: large initial web area decreases with burnback.
    "double_anchor":    [(0.0, 1.000), (0.25, 0.875), (0.50, 0.750),
                         (0.75, 0.625), (1.0, 0.500)],
    # Neutral: star port maintains near-constant burning perimeter.
    "star":             [(0.0, 0.950), (0.10, 1.000), (0.40, 1.000),
                         (0.70, 0.980), (1.0, 0.950)],
    # Two-phase boost-sustain: high initial thrust then step down.
    "multi_fin":        [(0.0, 1.000), (0.35, 1.000), (0.40, 0.450), (1.0, 0.430)],
    # Two-phase: high-energy outer propellant then lower-energy core.
    "dual_composition": [(0.0, 1.000), (0.30, 1.000), (0.33, 0.300), (1.0, 0.280)],
}

GRAIN_LABELS = {
    "tubular":          "Tubular (progressive)",
    "rod_tube":         "Rod and tube (neutral)",
    "double_anchor":    "Double anchor (regressive)",
    "star":             "Star (neutral)",
    "multi_fin":        "Multi-fin (two-phase)",
    "dual_composition": "Dual composition (two-phase)",
}

# Realistic fill-factor (F_avg/F_peak) ranges per grain type — for UI warnings.
_GRAIN_FILL_RANGE = {
    "tubular":          (0.70, 0.95),
    "rod_tube":         (0.90, 1.00),
    "double_anchor":    (0.60, 0.85),
    "star":             (0.85, 1.00),
    "multi_fin":        (0.50, 0.75),
    "dual_composition": (0.35, 0.60),
}


def grain_fill_factor(grain_type: str) -> float:
    """F_avg/F_peak (fill factor) computed by trapezoidal integration of the grain curve."""
    curve = _GRAIN_CURVES.get(grain_type)
    if curve is None:
        return 1.0
    total = 0.0
    for i in range(len(curve) - 1):
        t0, f0 = curve[i]; t1, f1 = curve[i + 1]
        total += 0.5 * (f0 + f1) * (t1 - t0)
    return total


def _instantaneous_thrust_frac(grain_type: str, t_frac: float,
                                thrust_profile=None) -> float:
    """Return F(t)/F_peak at normalised time t/burn_time."""
    if thrust_profile:
        ts = [p[0] for p in thrust_profile]
        fs = [p[1] for p in thrust_profile]
        return _lin_interp(t_frac, ts, fs)
    curve = _GRAIN_CURVES.get(grain_type)
    if curve is None:
        return 1.0
    ts = [p[0] for p in curve]
    fs = [p[1] for p in curve]
    return _lin_interp(t_frac, ts, fs)


def _lin_interp(x, xs, ys):
    """Piecewise-linear interpolation, clamped at endpoints."""
    if x <= xs[0]:  return ys[0]
    if x >= xs[-1]: return ys[-1]
    for i in range(len(xs) - 1):
        if xs[i] <= x <= xs[i + 1]:
            t = (x - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + t * (ys[i + 1] - ys[i])
    return ys[-1]


def _chin_pressure_coeff(sigma_deg: float, mach: float) -> float:
    """Chin (1961) Eq. 3-4: pressure coefficient for a cone of half-angle σ°."""
    if mach < 1e-6:
        return 0.0
    return (0.083 + 0.096 / mach**2) * (sigma_deg / 10.0) ** 1.69


def _cd_wave_cone(ld: float, mach: float) -> float:
    """Cone wave drag — Chin (1961) Eq. 3-4/3-6.  Linear ramp M=0.8→1.0."""
    import math
    sigma_deg = math.degrees(math.atan(1.0 / (2.0 * max(0.5, ld))))
    if mach >= 1.0:
        return _chin_pressure_coeff(sigma_deg, mach)
    if mach <= 0.8:
        return 0.0
    return _chin_pressure_coeff(sigma_deg, 1.0) * (mach - 0.8) / 0.2


def _cd_wave_ogive(ld: float, mach: float) -> float:
    """Tangent-ogive wave drag — Chin (1961) Eq. 3-9 (Miles formula)."""
    import math
    ld = max(0.5, ld)
    sigma_deg = math.degrees(math.atan(1.0 / (2.0 * ld)))
    if mach >= 1.0:
        P      = _chin_pressure_coeff(sigma_deg, mach)
        num    = 2.0 * (196.0 * ld**2 - 16.0)
        denom  = 28.0 * (mach + 18.0) * ld**2
        factor = max(0.0, 1.0 - num / denom)
        return P * factor
    if mach <= 0.8:
        return 0.0
    return _cd_wave_ogive(ld, 1.0) * (mach - 0.8) / 0.2


def _cd_wave_table(table_y, ld: float, mach: float) -> float:
    """Wave drag from NACA TN 4201 table at reference ld=3, scaled via (3/ld)²."""
    cd3 = _lin_interp(mach, _WAVE_MACH, table_y)
    return cd3 * (_WAVE_LD_REF / max(0.5, ld)) ** 2


def _nose_profile(shape: str, ld: float, n: int = 200):
    """
    Normalised (x, r) profile for a nose cone: x ∈ [0,1] (tip→base), r ∈ [0,1].
    r is radius / R_body, x is axial / L_nose.  Crowell (1996) geometry.
    """
    import math
    xs = np.linspace(0.0, 1.0, n + 1)

    if shape == 'cone':
        rs = xs.copy()

    elif shape == 'tangent_ogive':
        lod = 2.0 * max(0.5, ld)              # L/R
        rho = (lod**2 + 1.0) / 2.0            # radius of curvature / R
        rs  = np.sqrt(np.maximum(0.0, rho**2 - (lod * (1.0 - xs))**2)) - (rho - 1.0)

    elif shape == 'von_karman':
        theta = np.arccos(np.clip(1.0 - 2.0 * xs, -1.0, 1.0))
        rs    = np.sqrt(np.maximum(0.0, theta - np.sin(2.0 * theta) / 2.0)) / math.sqrt(math.pi)

    elif shape == 'lv_haack':
        theta = np.arccos(np.clip(1.0 - 2.0 * xs, -1.0, 1.0))
        rs    = np.sqrt(np.maximum(0.0,
                    theta - np.sin(2.0 * theta) / 2.0 + np.sin(theta)**3 / 3.0
                )) / math.sqrt(math.pi)

    elif shape == 'parabola':
        rs = 2.0 * xs - xs**2   # K'=1 tangent parabola (Crowell Eq. 7)

    else:
        rs = xs.copy()

    rs[0] = 0.0
    return xs, rs


def _s_wet_ratio(shape: str, ld: float) -> float:
    """
    Nose wetted area / reference area (A_ref = π R²).
    Numerical integration of 2π r ds.  (Crowell 1996 §5)
    ld = nose_length / body_diameter.
    """
    xs, rs   = _nose_profile(shape, ld)
    k        = 1.0 / (2.0 * max(0.5, ld))      # R/L
    drs      = np.diff(rs) / np.diff(xs)
    rs_mid   = 0.5 * (rs[:-1] + rs[1:])
    integrand = rs_mid * np.sqrt(1.0 + (k * drs)**2)
    return 4.0 * ld * float(np.sum(integrand * np.diff(xs)))  # = 2(L/R)·∫


def _mu_air(T_K: float) -> float:
    """Dynamic viscosity of air (Pa·s) — Sutherland's law."""
    T_ref, mu_ref, S = 273.15, 1.716e-5, 110.4
    return mu_ref * (T_K / T_ref) ** 1.5 * (T_ref + S) / (T_K + S)


def _cf_schoenherr(re_l: float) -> float:
    """Turbulent Cf — Schoenherr (Chin Eq. 4-2): √Cf·log₁₀(Cf·Re)=0.242."""
    import math
    cf = max(1e-8, 0.074 / re_l ** 0.2)   # Prandtl–Schlichting initial guess
    for _ in range(30):
        sq = math.sqrt(cf)
        f  = sq * math.log10(cf * re_l) - 0.242
        df = (math.log10(cf * re_l) / (2.0 * sq)
              + sq / (cf * math.log(10.0)))
        if abs(df) < 1e-15:
            break
        cf = max(1e-8, cf - f / df)
    return cf


def _cd_friction(re_l: float, mach: float, s_wet_ratio: float) -> float:
    """
    Friction drag coefficient.
      Blasius laminar Cf (Chin Eq. 4-1) + Schoenherr turbulent Cf (Chin Eq. 4-2)
      Mixed BL at Re_transition = 5×10^5 (Chin Eq. 4-3)
      Frankl-Voishel compressibility correction (Chin Eq. 4-6)
      +10 % roughness allowance (Chin §4-2)
    s_wet_ratio : S_wet / A_ref
    """
    import math
    if re_l < 1.0 or s_wet_ratio <= 0.0:
        return 0.0
    re_tr  = 5.0e5
    cf_lam  = 1.328 / math.sqrt(re_l)   # Blasius (Chin Eq. 4-1)
    cf_turb = _cf_schoenherr(re_l)       # Schoenherr (Chin Eq. 4-2)
    s_lam   = min(1.0, re_tr / re_l)
    cf_mix  = cf_lam * s_lam + cf_turb * (1.0 - s_lam)   # Chin Eq. 4-3
    fv      = (1.0 + 0.2 * mach**2) ** (-0.467)           # Frankl-Voishel (Chin Eq. 4-6)
    return cf_mix * fv * 1.10 * s_wet_ratio                # +10% roughness


def _cd_base(mach: float, base_area_ratio: float = 1.0) -> float:
    """Base pressure drag — Chin Fig. 3-15, power-off."""
    cpb = _lin_interp(mach, _BASE_MACH, _BASE_CPB)
    return -cpb * base_area_ratio   # Cpb < 0 → Cd_base > 0


def _cd_nose_shape(nose_shape: str, ld: float, mach: float,
                   re_l: float = 5e6, ld_body: float = None) -> float:
    """
    Total zero-lift drag coefficient (Cd_wave + Cd_friction + Cd_base).
    Source: Chin (1961) *Missile Configuration Design*; NACA TN 4201; Crowell (1996).

    nose_shape : key from NOSE_SHAPES
    ld         : nose fineness ratio = nose_length / body_diameter (clamped 0.5–10)
    mach       : free-stream Mach number
    re_l       : Reynolds number based on body length (default 5×10^6)
    ld_body    : full-body fineness ratio = body_length / body_diameter;
                 drives cylinder friction term.  None → 2×ld estimate.
    """
    nose_shape = _SHAPE_ALIAS.get(nose_shape, nose_shape)
    ld   = max(0.5, min(float(ld), 10.0))
    mach = max(0.0, float(mach))

    # ── Blunt Cylinder ────────────────────────────────────────────────────────
    if nose_shape == 'blunt_cylinder':
        if mach <= 0.8:   return 0.9
        if mach <= 1.5:   return 0.9 + (mach - 0.8) / 0.7 * 1.3
        return 2.2

    # ── Forden fallback (legacy) ──────────────────────────────────────────────
    if nose_shape in ('forden', '', None):
        return _lin_interp(mach, _FORDEN_MACH, _FORDEN_CD)

    # ── Wave drag (nose shape-specific) ──────────────────────────────────────
    if nose_shape == 'cone':
        cd_wave = _cd_wave_cone(ld, mach)
    elif nose_shape == 'tangent_ogive':
        cd_wave = _cd_wave_ogive(ld, mach)
    elif nose_shape == 'von_karman':
        cd_wave = _cd_wave_table(_WAVE_VK, ld, mach)
    elif nose_shape == 'lv_haack':
        cd_wave = _cd_wave_table(_WAVE_LVH, ld, mach)
    elif nose_shape == 'parabola':
        cd_wave = _cd_wave_table(_WAVE_PARA, ld, mach)
    else:
        cd_wave = _cd_wave_cone(ld, mach)

    # ── Friction drag (nose wetted area + cylindrical body section) ───────────
    nose_swet = _s_wet_ratio(nose_shape, ld)
    if ld_body is None:
        ld_body = max(ld * 2.0, ld + 2.0)
    # cylinder S_wet/A_ref = π D L_cyl/(π R²) = 4 L_cyl/D = 4(ld_body − ld_nose)
    cyl_swet = 4.0 * max(0.0, ld_body - ld)
    cd_fric  = _cd_friction(re_l, mach, nose_swet + cyl_swet)

    # ── Base drag ─────────────────────────────────────────────────────────────
    cd_base = _cd_base(mach)

    return cd_wave + cd_fric + cd_base


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
        mass_final=1198 + 1000,      # 2198 kg  (dry + warhead; body stays attached)
        diameter_m=0.84,
        length_m=11.25,
        thrust_N=round(_thrust_from_isp(230, prop, 75)),   # ≈ 111 200 N
        burn_time_s=75.0,
        isp_s=230.0,
        payload_kg=1000.0,
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
        mass_final=1334 + 191,       # 1525 kg  (dry + warhead)
        diameter_m=0.84,
        length_m=12.0,
        thrust_N=round(_thrust_from_isp(230, prop, 90)),   # ≈ 118 900 N
        burn_time_s=90.0,
        isp_s=230.0,
        payload_kg=191.0,
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
        mass_final=3900 + 1000,      # 4 900 kg  (dry + warhead)
        diameter_m=0.88,
        length_m=15.6,
        thrust_N=round(_thrust_from_isp(240, prop, 70)),   # ≈ 537 600 N
        burn_time_s=70.0,
        isp_s=240.0,
        payload_kg=1000.0,
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
        payload_kg=454.0,
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
        payload_kg=500.0,
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
        rv_separates=True,   # RV separates from stage-2 body at burnout
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
    #   Payload fairing:     44 kg  (shroud_mass_kg; jettisoned at 80 km)
    #   Total at launch:   51 935 kg
    #     = 24 117+44 (S1+shroud) + 24 117 (S2) + 3 437 (S3) + 220 (sat)
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
    shroud    =   44   # kg — fairing, jettisoned at 80 km via shroud_mass_kg

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
        guidance="loft",
        loft_angle_deg=25.0,
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
        mass_initial=dry2 + prop2 + stage3.mass_initial,   # 27 774 kg
        mass_propellant=prop2,                                        # 20 517 kg
        mass_final=dry2,                                              #  3 600 kg  (jettisoned)
        diameter_m=1.5,
        length_m=8.7,    # stage body only (+ 1.0 m interstage = 9.7 m total)
        thrust_N=round(_thrust_from_isp(isp2, prop2, 71.0)),   # ≈ 751 000 N
        burn_time_s=71.0,
        isp_s=isp2,
        nozzle_exit_area_m2=1.77,
        guidance="loft",
        loft_angle_deg=35.0,
        loft_angle_rate_deg_s=0.6,
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
        mass_initial=dry1 + prop1 + stage2.mass_initial + shroud,   # 51 935 kg
        mass_propellant=prop1,                               # 20 517 kg
        mass_final=dry1,                                     #  3 600 kg  (jettisoned)
        diameter_m=1.5,
        length_m=10.3,   # stage body only (+ 0.75 m interstage = 11.05 m total)
        thrust_N=round(_thrust_from_isp(isp1, prop1, 71.0)),   # ≈ 725 000 N
        burn_time_s=71.0,
        isp_s=isp1,
        nozzle_exit_area_m2=0.90,
        guidance="loft",
        loft_angle_deg=45.0,
        loft_angle_rate_deg_s=1.0,
        mach_table=_FORDEN_MACH,
        cd_table=_FORDEN_CD,
        stage2=stage2,
        coast_time_s=12.0,
        payload_kg=float(satellite),
        shroud_mass_kg=float(shroud),
        shroud_jettison_alt_km=80.0,
    )


def _aur():
    # AUR — two-stage ballistic missile, depressed trajectory.
    #
    # Physical parameters from open-source data:
    #   Body diameter: 34.5 in = 0.8763 m
    #   Total length:  5.0 (S1) + 2.6 (S2) + 2.6 (payload) = 10.2 m
    #
    # Stage 2 (upper stage — ignites after stage-1 separation):
    #   Propellant: 1 842 kg  Dry: 186 kg  Fueled: 2 028 kg
    #   Thrust: 80 kN  Isp: 280 s  Burn: 63 s  Nozzle exit area: 0.30 m²
    #   mass_final = stage-2 dry only (jettisoned at payload separation)
    stage2 = MissileParams(
        name="AUR Stage 2",
        mass_initial=2028 + 450,      # 2 478 kg (stage-2 wet + payload)
        mass_propellant=1842,
        mass_final=186,               # stage-2 dry shell, jettisoned at separation
        diameter_m=0.8763,
        length_m=2.6,
        thrust_N=80_000,              # 80 kN as stated
        burn_time_s=63.0,
        isp_s=280.0,
        nozzle_exit_area_m2=0.30,
        guidance="loft",
        # Depressed-trajectory — starting values; tune experimentally.
        loft_angle_deg=25.0,
        loft_angle_rate_deg_s=3.0,
        coast_time_s=0.0,
        # Empty tables → legacy 2 % sea-level back-pressure approximation used.
        mach_table=[],
        cd_table=[],
    )

    # Stage 1 (booster):
    #   Propellant: 4 509 kg  Dry: 454 kg  Fueled: 4 963 kg
    #   Thrust: 230 kN  Isp: 280 s  Burn: 54 s  Nozzle exit area: 0.30 m²
    #   mass_final = stage-1 dry only (jettisoned after burnout, no coast)
    return MissileParams(
        name="AUR",
        mass_initial=4963 + stage2.mass_initial,   # 7 441 kg total at launch
        mass_propellant=4509,
        mass_final=454,                            # stage-1 dry, jettisoned
        diameter_m=0.8763,                         # 34.5 in converted to metres
        length_m=10.2,                             # 5.0 + 2.6 + 2.6
        thrust_N=230_000,                          # 230 kN as stated
        burn_time_s=54.0,
        isp_s=280.0,
        nozzle_exit_area_m2=0.30,
        guidance="loft",
        # Depressed-trajectory — starting values; tune experimentally.
        loft_angle_deg=25.0,
        loft_angle_rate_deg_s=3.0,
        coast_time_s=0.0,
        payload_kg=450.0,
        rv_beta_kg_m2=1500.0,
        rv_separates=True,
        stage2=stage2,
        # Empty tables → legacy 2 % sea-level back-pressure approximation used.
        mach_table=[],
        cd_table=[],
    )


MISSILE_DB = {
    # Populated at runtime from custom_missiles.json via _load_custom_missiles()
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
        'mass_final':            p.mass_final,
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
        'rv_separates':          p.rv_separates,
        'bus_mass_kg':           p.bus_mass_kg,
        'num_rvs':               p.num_rvs,
        'rv_mass_kg':            p.rv_mass_kg,
        'shroud_mass_kg':         p.shroud_mass_kg,
        'shroud_jettison_alt_km': p.shroud_jettison_alt_km,
        'shroud_length_m':        p.shroud_length_m,
        'shroud_diameter_m':      p.shroud_diameter_m,
        'nozzle_exit_area_m2':    p.nozzle_exit_area_m2,
        'solid_motor':            p.solid_motor,
        'grain_type':             p.grain_type,
        'thrust_peak_N':          p.thrust_peak_N,
        'thrust_profile':         list(p.thrust_profile),
        'nose_shape':             p.nose_shape,
        'nose_length_m':          p.nose_length_m,
        'shroud_nose_shape':      p.shroud_nose_shape,
        'shroud_nose_length_m':   p.shroud_nose_length_m,
        'payload_diameter_m':     p.payload_diameter_m,
        'rv_shape':               p.rv_shape,
        'rv_diameter_m':          p.rv_diameter_m,
        'rv_length_m':            p.rv_length_m,
        'pbv_diameter_m':         p.pbv_diameter_m,
        'pbv_length_m':           p.pbv_length_m,
    }
    # Per-stage pitch overrides — only written when set (keeps dicts compact)
    if p.stage_turn_start_s is not None:
        d['stage_turn_start_s'] = p.stage_turn_start_s
    if p.stage_turn_stop_s is not None:
        d['stage_turn_stop_s'] = p.stage_turn_stop_s
    if p.stage_burnout_angle_deg is not None:
        d['stage_burnout_angle_deg'] = p.stage_burnout_angle_deg
    if p.stage_yaw_start_s is not None:
        d['stage_yaw_start_s'] = p.stage_yaw_start_s
    if p.stage_yaw_stop_s is not None:
        d['stage_yaw_stop_s'] = p.stage_yaw_stop_s
    if p.stage_yaw_final_az_deg is not None:
        d['stage_yaw_final_az_deg'] = p.stage_yaw_final_az_deg
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
        mass_final=float(d['mass_final']) if 'mass_final' in d else m0 - prop,
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
        rv_separates=bool(d.get('rv_separates', False)),
        bus_mass_kg=float(d.get('bus_mass_kg', 0.0)),
        num_rvs=int(d.get('num_rvs', 1)),
        rv_mass_kg=float(d.get('rv_mass_kg', 0.0)),
        shroud_mass_kg=float(d.get('shroud_mass_kg', 0.0)),
        shroud_jettison_alt_km=float(d.get('shroud_jettison_alt_km', 80.0)),
        shroud_length_m=float(d.get('shroud_length_m', 0.0)),
        shroud_diameter_m=float(d.get('shroud_diameter_m', 0.0)),
        nozzle_exit_area_m2=float(d.get('nozzle_exit_area_m2', 0.0)),
        solid_motor=bool(d.get('solid_motor', False)),
        grain_type=d.get('grain_type', ''),
        thrust_peak_N=float(d.get('thrust_peak_N', 0.0)),
        thrust_profile=list(d.get('thrust_profile', [])),
        nose_shape=d.get('nose_shape', ''),
        nose_length_m=float(d.get('nose_length_m',
                            float(d.get('nose_ld_ratio', 0.0)) * float(d['diameter_m']))),
        shroud_nose_shape=d.get('shroud_nose_shape', ''),
        shroud_nose_length_m=float(d.get('shroud_nose_length_m',
                            float(d.get('shroud_nose_ld_ratio', 0.0)) * float(d['diameter_m']))),
        payload_diameter_m=float(d.get('payload_diameter_m', 0.0)),
        rv_shape=d.get('rv_shape', ''),
        rv_diameter_m=float(d.get('rv_diameter_m', 0.0)),
        rv_length_m=float(d.get('rv_length_m', 0.0)),
        pbv_diameter_m=float(d.get('pbv_diameter_m', 0.0)),
        pbv_length_m=float(d.get('pbv_length_m', 0.0)),
        stage_turn_start_s=(float(d['stage_turn_start_s'])
                            if d.get('stage_turn_start_s') is not None else None),
        stage_turn_stop_s=(float(d['stage_turn_stop_s'])
                           if d.get('stage_turn_stop_s') is not None else None),
        stage_burnout_angle_deg=(float(d['stage_burnout_angle_deg'])
                                 if d.get('stage_burnout_angle_deg') is not None else None),
        stage_yaw_start_s=(float(d['stage_yaw_start_s'])
                           if d.get('stage_yaw_start_s') is not None else None),
        stage_yaw_stop_s=(float(d['stage_yaw_stop_s'])
                          if d.get('stage_yaw_stop_s') is not None else None),
        stage_yaw_final_az_deg=(float(d['stage_yaw_final_az_deg'])
                                if d.get('stage_yaw_final_az_deg') is not None else None),
    )


# ---------------------------------------------------------------------------
# Physics helper functions
# ---------------------------------------------------------------------------

def tumbling_cylinder_beta(mass_kg: float, diameter_m: float, length_m: float,
                           cd: float = 1.0) -> float:
    """
    Ballistic coefficient β (kg/m²) for a tumbling cylinder.

    The effective reference area is the mean of the end-on and broadside
    projected areas, which approximates the time-averaged area for a cylinder
    tumbling in the pitch plane:

        A_eff = (π d² / 4  +  d · L) / 2
        β     = m / (Cd · A_eff)

    Default Cd = 1.0 is representative of bluff-body turbulent flow.
    Returns 0 if length or diameter is zero.
    """
    A_end  = np.pi * diameter_m ** 2 / 4.0
    A_side = diameter_m * length_m
    A_eff  = (A_end + A_side) / 2.0
    if A_eff <= 0:
        return 0.0
    return mass_kg / (cd * A_eff)


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


def missile_area(params: MissileParams, altitude_m: float = None,
                 top_params: 'MissileParams' = None) -> float:
    """Reference cross-sectional area (m^2).

    When top_params carries a shroud and altitude_m is below the jettison
    altitude, returns the shroud frontal area instead of the body area.
    """
    if (top_params is not None
            and top_params.shroud_diameter_m > 0
            and altitude_m is not None
            and altitude_m < top_params.shroud_jettison_alt_km * 1000.0):
        d = top_params.shroud_diameter_m
    elif top_params is not None and top_params.rv_separates and top_params.rv_diameter_m > 0:
        d = top_params.rv_diameter_m
    elif top_params is not None and top_params.payload_diameter_m > 0:
        d = top_params.payload_diameter_m
    else:
        d = params.diameter_m
    return np.pi * (d / 2) ** 2


def drag_coefficient(params: MissileParams, mach: float) -> float:
    """Cd interpolated from Mach table."""
    return float(np.interp(mach, params.mach_table, params.cd_table))


def drag_force_vector(params: MissileParams, vel_ecef, altitude_m,
                      top_params: 'MissileParams' = None) -> np.ndarray:
    """
    Aerodynamic drag force vector (N) opposing velocity.

    Parameters
    ----------
    params     : MissileParams (current stage)
    vel_ecef   : velocity vector in ECEF (m/s), shape (3,)
    altitude_m : scalar altitude (m)
    top_params : top-level MissileParams (for shroud diameter lookup); optional

    Returns
    -------
    F_drag : ndarray (3,) in Newtons (opposing velocity direction)
    """
    speed = np.linalg.norm(vel_ecef)
    if speed < 1e-6:
        return np.zeros(3)
    T, _, rho, a_sound = atmosphere(altitude_m)
    mach  = speed / a_sound
    mu    = _mu_air(T)
    L_ref = params.length_m if params.length_m > 0.0 else 1.0
    re_l  = rho * speed * L_ref / mu if mu > 0.0 else 5e6

    # Choose Cd source: decomposed nose-shape model or Forden mach_table.
    # Shroud nose shape takes priority while shroud is still attached.
    _shroud_on = (top_params is not None
                  and top_params.shroud_diameter_m > 0
                  and altitude_m < top_params.shroud_jettison_alt_km * 1000.0)
    if _shroud_on and top_params.shroud_nose_shape not in ('', 'forden'):
        _sd = (top_params.shroud_diameter_m if top_params.shroud_diameter_m > 0
               else params.diameter_m)
        _ld = (top_params.shroud_nose_length_m / _sd
               if top_params.shroud_nose_length_m > 0 and _sd > 0 else 3.0)
        _ld_body = (top_params.shroud_length_m / _sd
                    if top_params.shroud_length_m > 0 and _sd > 0 else None)
        cd = _cd_nose_shape(top_params.shroud_nose_shape, _ld, mach,
                            re_l=re_l, ld_body=_ld_body)
    elif top_params is not None and (
            (top_params.rv_separates
             and top_params.rv_shape not in ('', 'forden')
             and top_params.rv_diameter_m > 0)
            or (not top_params.rv_separates
                and top_params.nose_shape not in ('', 'forden'))):
        # After shroud jettison: use RV geometry when rv_separates, else payload nose.
        if top_params.rv_separates and top_params.rv_shape not in ('', 'forden'):
            _shape  = top_params.rv_shape
            _diam   = top_params.rv_diameter_m
            _length = top_params.rv_length_m
        else:
            _shape  = top_params.nose_shape
            _diam   = (top_params.payload_diameter_m if top_params.payload_diameter_m > 0
                       else params.diameter_m)
            _length = top_params.nose_length_m
        _ld = (_length / _diam if _length > 0 and _diam > 0 else 3.0)
        _ld_body = (params.length_m / _diam if params.length_m > 0 and _diam > 0 else None)
        cd = _cd_nose_shape(_shape, _ld, mach, re_l=re_l, ld_body=_ld_body)
    else:
        cd = drag_coefficient(params, mach)

    area = missile_area(params, altitude_m=altitude_m, top_params=top_params)
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
            # Instantaneous vacuum thrust: grain modulation or constant.
            if s.grain_type or s.thrust_profile:
                T_peak = s.thrust_peak_N if s.thrust_peak_N > 0.0 else s.thrust_N
                t_frac = t_rem / s.burn_time_s if s.burn_time_s > 0.0 else 0.0
                frac   = _instantaneous_thrust_frac(
                    s.grain_type, t_frac,
                    s.thrust_profile if s.thrust_profile else None)
                T_vac = T_peak * frac
            else:
                T_vac = s.thrust_N
            if s.nozzle_exit_area_m2 > 0:
                thrust_mag = max(0.0, T_vac - P_amb * s.nozzle_exit_area_m2)
            else:
                thrust_mag = T_vac * (1.0 - 0.02 * (P_amb / 101325.0))
            return thrust_mag * thrust_dir
        t_rem -= s.burn_time_s
        if s.stage2 is None:
            return np.zeros(3)
        if t_rem <= s.coast_time_s:
            return np.zeros(3)   # coasting between stages
        t_rem -= s.coast_time_s
        s      = s.stage2
    return np.zeros(3)  # all stages burned out
