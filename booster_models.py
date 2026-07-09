"""
Booster parameter models matching Forden's:
  missileMass.m, missileRadius.m, thrust.m, thrustAngle.m,
  dragForce.m, Drag.m, calcCm_delta.m, aeroQ.m, calcMissileParameters.m

Built-in booster definitions follow Forden (2007) Table 1 parameters for the
four packaged models: Scud-B, Al Hussein, No-dong, and Taepodong-I.
Loft angle / loft angle rate for Scud-B taken from Figure 3 of the same paper.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from atmosphere import (atmosphere, dynamic_pressure,
                        configure_atmosphere, atmosphere_source)

_G0 = 9.80665   # standard gravity (m/s²)


def _thrust_from_isp(isp_s: float, propellant_kg: float, burn_s: float) -> float:
    """Vacuum thrust (N) derived from Isp, propellant mass, and burn time."""
    return isp_s * _G0 * propellant_kg / burn_s


@dataclass
class BoosterParams:
    """All parameters needed to simulate one booster type."""
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
    #   "loft"         — Forden pitch-over (SRBM/MRBM): pitch to burnout_angle_deg
    #                    at loft_angle_rate_deg_s then hold.  Floor preserved.
    #   "pitch_program" — kick off vertical to burnout_angle_deg at
    #                    loft_angle_rate_deg_s, then lock thrust to velocity
    #                    vector (IRBM/ICBM).  burnout_angle_deg here is the kick
    #                    elevation (°, above horizontal; e.g. 85° = 5° from
    #                    vertical) and loft_angle_rate_deg_s is the kick rate.
    guidance: str = "pitch_program"

    burnout_angle_deg: float = 45.0        # Forden: final elev (°); GT: kick elev (°)
    loft_angle_rate_deg_s: float = 2.0  # Forden: pitch rate (°/s); GT: kick rate
    launch_elevation_deg: float = 90.0  # elevation at liftoff (°); 90 = vertical

    # Aerodynamics — Cd vs Mach lookup table
    mach_table: list = field(default_factory=list)
    cd_table:   list = field(default_factory=list)

    # Staging (optional next stage)
    stage2: Optional['BoosterParams'] = None

    # Coast time (s) between this stage's burnout and the next stage's ignition.
    # Ignored (irrelevant) for the last / only stage.
    coast_time_s: float = 0.0

    # Payload (kg) — total mass carried to burnout (bus + all RVs).
    # Stored on the top-level stage only so _prefill can round-trip correctly.
    payload_kg: float = 0.0

    # Payload decomposition (throw-weight): bus (post-boost vehicle) + N reentry
    # objects.  bus_mass_kg + num_ros * ro_mass_kg should equal payload_kg.
    # These are booster-level payload bookkeeping — NOT reentry-object hardware
    # (beta, L/D, TPS, glide law all live on params.ro).
    ro_mass_kg:    float = 0.0   # per-object mass, for the throw-weight breakdown
    bus_mass_kg:   float = 0.0
    num_ros:       int   = 1

    # When True the RV/payload separates from the last-stage body at burnout.
    # The empty stage body then follows a tumbling-cylinder debris arc.
    # When False (default) the stage body stays attached to the warhead
    # (e.g. Scud-B) and no separate stage debris is computed.
    ro_separates:  bool  = False

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

    # Commanded engine cutoff for this stage: burn DURATION (s from this stage's
    # ignition) after which thrust is cut.  None = burn to completion.  Liquid
    # engines only — ignored on solid motors (they cannot be shut down).  The
    # stage still occupies its full burn_time_s slot; the tail after cutoff is a
    # dead coast with the unburned propellant riding as mass until jettison.
    stage_cutoff_s:          Optional[float] = None

    # Per-stage yaw (dogleg) overrides.  Same priority as pitch overrides:
    # if stage_yaw_final_az_deg is not None the stage performs a linear
    # azimuth schedule from current az to final_az over [yaw_start, yaw_stop].
    stage_yaw_start_s:     Optional[float] = None
    stage_yaw_stop_s:      Optional[float] = None
    stage_yaw_final_az_deg: Optional[float] = None

    # Shroud jettisoned during ascent.
    # shroud_mass_kg is included in mass_initial at launch and subtracted once
    # the booster crosses shroud_jettison_alt_km.  0 = no shroud.
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

    # Aerospike (drag-reduction probe protruding from the forebody).
    # When aerospike_LD > 0, the forebody wave drag is computed from an
    # "effective body" cone whose fineness ratio is derived from the
    # dividing-streamline geometry measured by Ahmed & Qin (2011) Fig. 5:
    #     L/D_eff = 1.0 + (2/3)·spike_LD + 2.0·spike_dD
    # The model takes whichever wave-drag is lower (actual nose vs. effective
    # body), so a spike never makes a slender body worse.
    # spike_LD = spike length / body diameter (typical 1.0–3.0)
    # spike_dD = aerodisk diameter / body diameter (0 = pointed tip; 0.05–0.4)
    aerospike_LD:           float = 0.0
    aerospike_dD:           float = 0.0

    # Fins — trapezoidal planar fins attached to the last stage body.
    # When has_fins is True the fin lift slope and drag are computed and added
    # to the body-only values (used by the L/D estimator and the synthesised
    # no-sep ROParams beta).  Fin aerodynamics follow the DATCOM/Barrowman
    # model (friction: Mandell et al. 1973; lift: DATCOM supersonic formula).
    # All dimensions in metres; sweep is leading-edge sweep in degrees.
    has_fins:             bool  = False
    n_fins:               int   = 4
    fin_span_m:           float = 0.0   # exposed semi-span from body surface
    fin_root_chord_m:     float = 0.0
    fin_tip_chord_m:      float = 0.0
    fin_thickness_m:      float = 0.0   # max thickness
    fin_sweep_deg:        float = 0.0   # leading-edge sweep angle

    # Grid (lattice) fins — a box-frame lattice of thin cells, NOT a planar
    # airfoil.  When has_grid_fins is True the grid-fin drag (and lift slope
    # for the estimator) is computed by _cd_gridfins/_cl_alpha_gridfins
    # (Washington & Miller; Kantrowitz start limit) and added to the body
    # values.  Attach to the stage on which the fins are mounted (they apply
    # only while that stage is the active stage, e.g. a finned first stage).
    has_grid_fins:        bool  = False
    n_grid_fins:          int   = 0
    grid_fin_width_m:         float = 0.0   # frame width (azimuthal span)
    grid_fin_height_m:        float = 0.0   # frame height (radial)
    grid_fin_chord_m:         float = 0.0   # streamwise lattice depth
    grid_fin_web_thickness_m: float = 0.0   # cell wall (web) thickness
    grid_fin_cell_pitch_m:    float = 0.0   # cell centre-to-centre spacing
    grid_fin_solidity:        float = 0.0   # σ = blocked frontal fraction; if >0
    #   used directly (approximation for web+pitch), else derived from them.
    grid_fin_edge_factor:     float = 1.0   # 1.0 blunt webs; ~0.6-0.85 shaped
    # Deployment schedule: list of [deploy_time_s, n_fins] batches (absolute
    # mission time).  n fins become aerodynamically active at deploy_time_s;
    # the deployed count (capped at n_grid_fins) scales the grid-fin drag.
    # Empty -> all n_grid_fins active from t=0.  E.g. STARS deploys 4 fins at
    # tower-clear and 4 more 60 s later: [[t_clear, 4], [t_clear+60, 4]].
    grid_fin_deploy_schedule: list = field(default_factory=list)

    # Payload diameter (m).  When > 0, used as the frontal reference diameter
    # for aerodynamic drag after shroud jettison (or throughout flight when no
    # shroud is fitted).  Falls back to the stage body diameter_m when 0.
    payload_diameter_m:     float = 0.0

    # Reentry-object hardware (beta, mass, geometry, L/D, glide law, TPS) lives
    # ONLY on params.ro (ROParams) — the booster no longer carries any reentry
    # fields.  Old JSON that stored them inline is migrated to a synthesised
    # params.ro at load time (see booster_from_dict).

    # Post-boost vehicle (PBV) geometry — mass is already carried in bus_mass_kg.
    pbv_diameter_m: float = 0.0
    pbv_length_m:   float = 0.0

    # Provenance — where these numbers came from and how firm they are
    # (mirrors ROParams.source/notes).
    source: str = ""
    notes:  str = ""

    # Strap-on boosters — fire from t=0 in parallel with stage 1; separate at
    # booster_burn_time_s.  These fields are only meaningful on the top-level
    # (stage-1) node; upper-stage nodes ignore them.
    n_boosters:             int   = 0
    booster_thrust_n:       float = 0.0    # vacuum thrust per booster (N)
    booster_burn_time_s:    float = 0.0    # burn duration (s)
    booster_inert_kg:       float = 0.0    # inert (empty) mass per booster (kg)
    booster_prop_kg:        float = 0.0    # propellant mass per booster (kg)
    booster_isp_s:          float = 0.0    # specific impulse (s)
    booster_nozzle_area_m2: float = 0.0    # nozzle exit area for P correction (m²)
    booster_diam_m:         float = 0.0    # outer diameter per booster (m)
    booster_length_m:       float = 0.0    # length per booster (0 → 2×diameter)
    booster_cd:             float = 0.20   # zero-lift Cd (0.20 = tangent ogive)
    # Seconds after T=0 (strap-on ignition / liftoff) before stage-1 core ignites.
    # 0 = all ignite together (Soyuz).  >0 = sequential (LVM3, Titan IIIC).
    booster_core_delay_s:   float = 0.0
    # Time (s after T=0) the spent boosters are physically jettisoned.  Many
    # strap-ons burn out and are then carried as dead (thrustless) mass for a
    # short coast before separation — during that coast they still add inert
    # mass and parasitic drag.  0 = jettison coincides with burnout (legacy).
    booster_jettison_s:     float = 0.0

    # ── RV reference (new architecture) ─────────────────────────────────────
    # When set, all RV flight properties (β, shape, glider params) are read
    # from this object rather than from the deprecated inline fields below.
    # Populated by the booster editor when "RV separates" is checked.
    ro: Optional['ROParams'] = None


# ---------------------------------------------------------------------------
# ROParams — independently loadable reentry-vehicle / glide-body definition.
# All fields that were previously scattered across BoosterParams as rv_* and
# glider_* inline fields now live here.  The inline fields on BoosterParams
# are kept for backward-compatible reading of old JSON files but are no
# longer written by the editor or read by the integrator when params.ro is set.
# ---------------------------------------------------------------------------

@dataclass
class ROParams:
    """Reentry vehicle or hypersonic glide body — independently loadable."""
    name:       str
    mass_kg:    float   # single-RV mass (kg)
    beta_kg_m2: float   # ballistic coefficient β = m/(Cd·A) (kg/m²)

    # Geometry — for Cd model on boost phase and for β-calculator round-trip
    shape:      str   = ""    # key from NOSE_SHAPES; "" → Forden Cd fallback
    diameter_m: float = 0.0
    length_m:   float = 0.0
    # Nose-tip (stagnation) radius of curvature (m), used for Sutton-Graves
    # stagnation heating q̇ ∝ 1/√R_N.  0.0 = AUTO: derive a screening default
    # from the nose shape + base diameter (see nose_tip_radius / the
    # effective_nose_radius_m() accessor).  A positive value is authoritative
    # and overrides the derived default.
    nose_radius_m: float = 0.0

    # Separation mode — does the terminal vehicle separate from the booster
    # body, or IS the booster body the terminal vehicle?
    #   "separating_ro" — distinct payload, mass/beta/diameter independent
    #                     (ICBM with MIRV, Scud-style separated warhead, …)
    #   "body"          — vehicle IS the booster body (Hwasong-11 / Iskander,
    #                     Pershing II MaRV, single-stage maneuvering body).
    #                     mass/beta/diameter are inherited from the booster's
    #                     last stage (mass_final, beta_kg_m2, diameter_m)
    #                     by effective_ro() at runtime.
    separation_mode: str = "separating_ro"

    # Glider / HGV properties
    #
    # glider_guidance — one of:
    #   "equilibrium_glide":       Tracy 2020.  Analytical Acton pull-up
    #                              (Eq. 11) applied as a one-shot arc
    #                              between the 100 km pierce point and
    #                              h_eq; the schema's t₂ and t₃ coincide
    #                              (no separate direct-re-entry phase).
    #                              Uses one β (= β_L for glide).
    #   "equilibrium_glide_acton": Acton 2015 three-phase model.  After
    #                              piercing at 100 km the vehicle enters a
    #                              direct-re-entry segment with β_S drag
    #                              and L/D = 0 until alt = h_3 (Acton
    #                              Eq. 8); at t_3 the analytical pull-up
    #                              fires.  Requires β_S =
    #                              glider_beta_entry_kg_m2 > 0.
    #   "skip_glide":              no analytical pull-up; the vehicle re-
    #                              enters with whatever γ it had and the
    #                              natural EOM produces a phugoid.
    #   "skip_to_equilibrium":     starts as skip_glide; after N upward
    #                              crossings of the equilibrium curve the
    #                              guidance switches one-way to
    #                              equilibrium_glide.  N is set by
    #                              glider_skip_count (default 1).
    #   "damped_glide":            skip_glide plus continuous altitude-rate
    #                              lift feedback (Lu 2013 / Yu & Chen 2011)
    #                              that damps the phugoid to a target ratio
    #                              ζ = glider_damping_zeta.  ζ=0
    #                              ≡ skip_glide; large ζ → equilibrium_glide.
    #                              See DAMPED_GLIDE.md.
    glider_enabled:         bool  = False
    glider_LD:              float = 0.0
    glider_guidance:        str   = "equilibrium_glide"
    glider_pullup_g_max:    float = 10.0
    glider_terminal_dive:   bool  = False
    glider_terminal_alt_km: float = 30.0
    # Bank-turn schedule: list of (t_start_s, t_end_s, bank_deg) tuples in
    # mission-elapsed seconds.  Positive bank = right turn; negative = left.
    # Up to 3 entries.  When non-empty, equilibrium-glide modes fall back
    # to numerical integration during the glide phase because the analytical
    # equilibrium-glide formula cannot represent banked maneuvers.
    glider_bank_schedule:   list  = field(default_factory=list)
    # Aerodynamic model used in the numerical EOM during glide:
    #   "polar"       — slender-body drag polar (Munk 1924, Ashley & Landahl
    #                   §6-7, §9-8): C_L = 2α with C_L referenced to the
    #                   base area, C_D = C_D0 + k·C_L².  C_D0 is derived
    #                   from the user's β (treated as zero-lift) and base
    #                   diameter; k is back-solved from the user's
    #                   glider_LD so (L/D)_max matches input exactly.
    #                   Vehicle trims for the lift required to balance the
    #                   centripetal deficit (m·(g − V²/r) / cos σ); off-
    #                   trim banking and pull-up incur the correct induced-
    #                   drag penalty.  DEFAULT — the realistic model.
    #   "constant_LD" — idealized fixed-L/D upper bound: lift = drag · L/D
    #                   with L/D from glider_LD, β-derived drag.  Implicitly
    #                   assumes the vehicle always flies at max-L/D AoA and
    #                   never pays induced drag off-design, so it over-ranges
    #                   relative to the polar.  Kept for cross-checking the
    #                   closed-form Sänger/Tracy/Acton range solutions, which
    #                   assume constant L/D.  At its trim point (C_L = C_L*)
    #                   the polar reproduces this model exactly.
    glider_aero_model:      str   = "polar"
    # Target-based dive trigger.  When glider_dive_target_radius_km > 0 the
    # vehicle starts the terminal dive (bank = π) as soon as its great-circle
    # distance to the target (lat/lon) drops below the radius — in addition
    # to the altitude trigger, whichever fires first.  Disabled when radius
    # = 0.  Detected in the EOM each step; max_step is tightened to ~2 s
    # while the trigger is armed so the granularity is ~6–8 km at HGV
    # speeds.  If used in equilibrium-glide modes, the analytical closed-form
    # is bypassed in favour of the numerical EOM (the analytical glide can't
    # see the target).
    glider_dive_target_lat_deg:    float = 0.0
    glider_dive_target_lon_deg:    float = 0.0
    glider_dive_target_radius_km:  float = 0.0     # 0 = disabled
    # Acton 2015 Phase-3 (direct re-entry) ballistic coefficient β_S.
    # During Phase 3 the glider holds a high-AoA orientation: flat lower
    # surface to airflow, large drag, L/D = 0.  Acton's HTV-2 fit gives
    # β_S ≈ 7 kg/m² (Table 3, p. 206).  Used only by the
    # "equilibrium_glide_acton" guidance mode; ignored otherwise.  Set 0
    # to disable Phase 3 (effectively reverts to Tracy when paired with
    # Acton mode).
    glider_beta_entry_kg_m2: float = 0.0
    # Number of phugoid upward crossings of the equilibrium curve before the
    # one-way handoff to equilibrium glide.  Only used by skip_to_equilibrium.
    glider_skip_count:      int   = 1

    # Target damping ratio ζ for the "damped_glide" guidance mode.  The vehicle
    # flies at the max-L/D trim angle α* plus a flight-path-angle feedback term
    #     α = α* + k_γ·(γ* − γ)                      (Yu & Chen 2011, Eq. 19)
    # whose gain k_γ is computed each step for this ζ from the phugoid natural
    # frequency obtained by linearising the equilibrium-glide EOM from first
    # principles (lift ∝ ρ ∝ e^(−h/H_ρ) ⇒ d a_L/dh = −g_eff/H_ρ), giving the
    # control law of Lu, Forbes & Baldwin AIAA 2013-4648 Eq. 33.  The phugoid
    # frequency is corroborated empirically by Liu et al. 2025 (0.021–0.037
    # rad/s).  See DAMPED_GLIDE_MEMO.md §2.
    #     ω_p² = g_eff/H_ρ ,  k_γ = ζ·C_L*·V/√(g_eff·H_ρ)
    # with g_eff = g − V²/r and H_ρ the local density scale height.  ζ = 0
    # recovers undamped skip_glide exactly (k_γ = 0); ζ ≈ 0.7 gives a couple of
    # decaying oscillations into equilibrium glide; large ζ → equilibrium_glide.
    # Only used by the "damped_glide" mode.
    glider_damping_zeta:    float = 0.7

    # Control-surface descriptor used only by the damping-ratio estimator
    # (docs/damping_estimate_spec.md): how much lifting/control-surface area the
    # vehicle carries, which bounds the achievable ζ.  "unknown" → the estimator
    # returns its widest band; "none"/"small"/"substantial" select a tier.
    # glider_flap_area_ratio (S_flap/S_ref) and glider_flap_deflection_deg, when
    # > 0, override the tier with an explicit Newtonian-flap computation.  These
    # do not affect the flown trajectory — only the estimate.
    glider_control_surfaces:   str   = "unknown"
    glider_flap_area_ratio:    float = 0.0      # 0 ⇒ use the tier default
    glider_flap_deflection_deg: float = 0.0     # 0 ⇒ use the 12° default

    # Surface emissivity used for radiative-equilibrium temperature at the
    # stagnation point: T_eq = (q̇ / (σ·ε))^(1/4).  0.85 matches the value
    # Anderson, "Hypersonic and High-Temperature Gas Dynamics," 2nd ed.,
    # AIAA, 2006, Section 18.8 (p. 781), uses in a worked HERMES reentry
    # example, citing Hirschel, "Basics of Aerothermodynamics," Springer.
    #
    # Verified RCC ε(T) values from Williams & Curry, "Thermal Protection
    # Materials: Thermophysical Property Data," NASA RP-1289, Dec. 1992
    # (Table for RCC, attributed to Space Shuttle Program Thermodynamic
    # Design Data Book SD73-SH-0226, Rockwell International, 1981):
    #     ε = 0.78 at   0°F   (256 K)
    #     ε = 0.87 at 1000°F  (811 K)
    #     ε = 0.90 at 1500°F (1089 K)  ← peak
    #     ε = 0.89 at 2000°F (1367 K)
    #     ε = 0.83 at 2500°F (1644 K)
    #     ε = 0.75 at 2800°F (1811 K)  ← max tabulated
    # Recent arc-jet measurements (Ohlhorst et al., NASA NTRS 20070031768,
    # 2007) report 0.88–0.91 at 2700–3000°F, suggesting the design data
    # are conservative.  For peak-heating T_eq in the 2500–3800 K range
    # RCC is above its working temperature anyway (surface ablates, no
    # longer at equilibrium); the constant-ε model is a lower bound there.
    emissivity:             float = 0.85
    # TPS material class for the heating survivability figure of merit (see
    # heating.py TPS_MATERIALS): '' / 'aluminum' / 'titanium' / 'steel' /
    # 'silica_tile' / 'rcc' / 'uhtc' / 'carbon_ablator'.  '' → physical heating
    # numbers only, no pass/fail verdict.
    tps_material:           str   = ""
    # Per-location TPS materials (HEATING_MODEL_CROSSCHECK.md §10.1 / §11 Phase 1).
    # Both selectable for every RV.  When blank, they fall back to tps_material for
    # BOTH locations (via nose_material()/body_material()), so existing RVs are
    # unchanged.  body_tps_thickness_m is the designed body-layer thickness (or, for
    # a bare hot structure, the skin/wall thickness feeding the transient heat-sink).
    # structure_material / structure_limit_K carry the bondline verdict; for a
    # hot-structure body (heating.is_hot_structure) the bondline collapses onto the
    # body material's own limit.  NOT yet consumed by the FOM — wired in Phase 2.
    nose_tps_material:      str   = ""
    body_tps_material:      str   = ""
    body_tps_thickness_m:   float = 0.0
    structure_material:     str   = ""
    structure_limit_K:      float = 0.0
    # Bespoke (user-defined) material properties, keyed by location.  When a
    # location's material is the sentinel 'custom_nose' / 'custom_body', the
    # matching dict here holds a catalog-shaped entry (label, group, is_ablator,
    # peak_K/continuous_K/melt_K, density_kg_m3, H_eff_MJ_kg) that
    # heating.register_custom_material() injects into TPS_MATERIALS before the FOM
    # runs.  Empty/None for the usual catalog-key case.
    nose_tps_custom:        Optional[dict] = None
    body_tps_custom:        Optional[dict] = None
    # Provenance: where this vehicle's numbers came from and how firm they are.
    # `source` is a short citation; `notes` is free-form (e.g. "mass 300 kg is a
    # trajectory-fit value, no primary source").  Round-tripped by
    # ro_to_dict/ro_from_dict so the justification travels with the vehicle and
    # is never silently dropped on a GUI/library save.
    source:                 str   = ""
    notes:                  str   = ""

    def nose_material(self) -> str:
        """Nose TPS material key, falling back to the single tps_material (Phase-1
        back-compat: a lone tps_material governs both nose and body)."""
        return self.nose_tps_material or self.tps_material

    def body_material(self) -> str:
        """Body/acreage TPS material key, falling back to the single tps_material."""
        return self.body_tps_material or self.tps_material

    def effective_nose_radius_m(self) -> float:
        """Stagnation radius (m) for Sutton-Graves heating: the explicit
        nose_radius_m when set (>0), otherwise the shape/diameter screening
        default from nose_tip_radius()."""
        if self.nose_radius_m and self.nose_radius_m > 0.0:
            return float(self.nose_radius_m)
        return nose_tip_radius(self.shape, self.diameter_m)


def _norm_sep_mode(v) -> str:
    """Normalise a separation_mode value to the current two-token vocabulary
    ('separating_ro' | 'body').  Legacy aliases from older saved files:
    'separating_rv' -> 'separating_ro' (pre-Phase-2 rename) and
    'non_separating' -> 'body' (the run path only branches on 'body', so the
    old token silently missed the body-mode mass/geometry inheritance)."""
    s = str(v or 'separating_ro')
    return {'separating_rv': 'separating_ro',
            'non_separating': 'body'}.get(s, s)


def ro_to_dict(ro: ROParams, include_reentry_plan: bool = True) -> dict:
    """Serialise an ROParams to a JSON-compatible dict.

    With ``include_reentry_plan=False`` the reentry-plan fields (everything in
    ``_REENTRY_PLAN_KEYS`` -- glide mode, turns, dives, separation) are omitted,
    yielding a hardware-only reentry object; that is the form ro_library stores,
    with the plan travelling separately in a ``.reentryplan.json`` file.  The
    ``glider_LD`` capability stays: it is what the airframe *can* do.  The
    default (``True``) keeps the full serialisation for internal round-trips.
    """
    d = {
        'name':                  ro.name,
        'mass_kg':               ro.mass_kg,
        'beta_kg_m2':            ro.beta_kg_m2,
        'shape':                 ro.shape,
        'diameter_m':            ro.diameter_m,
        'length_m':              ro.length_m,
        'nose_radius_m':         ro.nose_radius_m,
        'glider_enabled':        ro.glider_enabled,
        'glider_LD':             ro.glider_LD,
        'glider_guidance':       ro.glider_guidance,
        'glider_pullup_g_max':   ro.glider_pullup_g_max,
        'glider_terminal_dive':  ro.glider_terminal_dive,
        'glider_terminal_alt_km':ro.glider_terminal_alt_km,
        'glider_bank_schedule':  ro.glider_bank_schedule,
        'glider_aero_model':     ro.glider_aero_model,
        'glider_dive_target_lat_deg':   ro.glider_dive_target_lat_deg,
        'glider_dive_target_lon_deg':   ro.glider_dive_target_lon_deg,
        'glider_dive_target_radius_km': ro.glider_dive_target_radius_km,
        'glider_beta_entry_kg_m2': ro.glider_beta_entry_kg_m2,
        'glider_skip_count':     ro.glider_skip_count,
        'glider_damping_zeta':   ro.glider_damping_zeta,
        'glider_control_surfaces':   ro.glider_control_surfaces,
        'glider_flap_area_ratio':    ro.glider_flap_area_ratio,
        'glider_flap_deflection_deg':ro.glider_flap_deflection_deg,
        'separation_mode':       ro.separation_mode,
        'emissivity':            ro.emissivity,
        'tps_material':          ro.tps_material,
        'nose_tps_material':     ro.nose_tps_material,
        'body_tps_material':     ro.body_tps_material,
        'body_tps_thickness_m':  ro.body_tps_thickness_m,
        'structure_material':    ro.structure_material,
        'structure_limit_K':     ro.structure_limit_K,
        'nose_tps_custom':       ro.nose_tps_custom,
        'body_tps_custom':       ro.body_tps_custom,
        'source':                ro.source,
        'notes':                 ro.notes,
    }
    if not include_reentry_plan:
        for _k in _REENTRY_PLAN_KEYS:
            d.pop(_k, None)
    return d


def ro_from_dict(d: dict) -> ROParams:
    # Legacy mode aliases:
    #   "constant_bank"   → "skip_glide"   (old bank-angle knob is gone)
    #   "azimuth_command" → "skip_glide"   (proportional heading hold removed
    #                                       — saved boosters fall back to a
    #                                       wings-level skip-glide)
    _g = str(d.get('glider_guidance', 'equilibrium_glide'))
    if _g in ('constant_bank', 'azimuth_command'):
        _g = 'skip_glide'
    return ROParams(
        name=str(d.get('name', 'RV')),
        mass_kg=float(d['mass_kg']),
        beta_kg_m2=float(d['beta_kg_m2']),
        shape=str(d.get('shape', '')),
        diameter_m=float(d.get('diameter_m', 0.0)),
        length_m=float(d.get('length_m', 0.0)),
        nose_radius_m=float(d.get('nose_radius_m', 0.0)),   # 0 = auto (shape)
        glider_enabled=bool(d.get('glider_enabled', False)),
        glider_LD=float(d.get('glider_LD', 0.0)),
        glider_guidance=_g,
        glider_pullup_g_max=float(d.get('glider_pullup_g_max', 10.0)),
        glider_terminal_dive=bool(d.get('glider_terminal_dive', False)),
        glider_terminal_alt_km=float(d.get('glider_terminal_alt_km', 30.0)),
        glider_bank_schedule=[tuple(b) for b in d.get('glider_bank_schedule', [])],
        glider_aero_model=str(d.get('glider_aero_model', 'polar')),
        glider_dive_target_lat_deg=float(d.get('glider_dive_target_lat_deg', 0.0)),
        glider_dive_target_lon_deg=float(d.get('glider_dive_target_lon_deg', 0.0)),
        glider_dive_target_radius_km=float(d.get('glider_dive_target_radius_km', 0.0)),
        glider_beta_entry_kg_m2=float(d.get('glider_beta_entry_kg_m2', 0.0)),
        glider_skip_count=int(d.get('glider_skip_count', 1)),
        glider_damping_zeta=float(d.get('glider_damping_zeta', 0.7)),
        glider_control_surfaces=str(d.get('glider_control_surfaces', 'unknown')),
        glider_flap_area_ratio=float(d.get('glider_flap_area_ratio', 0.0)),
        glider_flap_deflection_deg=float(d.get('glider_flap_deflection_deg', 0.0)),
        separation_mode=_norm_sep_mode(d.get('separation_mode', 'separating_ro')),
        emissivity=float(d.get('emissivity', 0.85)),
        tps_material=str(d.get('tps_material', '')),
        nose_tps_material=str(d.get('nose_tps_material', '')),
        body_tps_material=str(d.get('body_tps_material', '')),
        body_tps_thickness_m=float(d.get('body_tps_thickness_m', 0.0)),
        structure_material=str(d.get('structure_material', '')),
        structure_limit_K=float(d.get('structure_limit_K', 0.0)),
        nose_tps_custom=(d.get('nose_tps_custom') or None),
        body_tps_custom=(d.get('body_tps_custom') or None),
        source=str(d.get('source', '')),
        notes=str(d.get('notes', '')),
    )


def effective_ro(params: 'BoosterParams') -> Optional[ROParams]:
    """Return the active reentry object (ROParams), or None if none is set.

    The reentry object is params.ro; the booster carries no reentry hardware.

    When params.ro has separation_mode == "body" the reentering body IS the
    booster's own last stage (Hwasong-11 / Iskander, Pershing II MaRV class, or an
    SSTO where that stage is the whole vehicle) — not a separating object.
    In that case mass_kg / diameter_m / length_m are
    inherited from the booster's last-stage burnout state (mass_final,
    beta_kg_m2, diameter_m) instead of being independent payload fields.
    The user only has to set the maneuvering properties (L/D, g-limit,
    βₛ) — the body's mass and shape come from the booster params.
    """
    if params.ro is not None:
        ro = params.ro
        if getattr(ro, 'separation_mode', 'separating_ro') == 'body':
            # The vehicle IS the booster body — inherit mass and geometry
            # from the last stage's burnout state.  β remains user-specified
            # on the RV itself because there's no clean way to derive a
            # single scalar β from a Mach-dependent body Cd table.
            _last = params
            while _last.stage2 is not None:
                _last = _last.stage2
            import dataclasses as _dc
            _body_mass = (_last.mass_initial - _last.mass_propellant
                          if _last.mass_propellant > 0 else _last.mass_final)
            return _dc.replace(ro,
                mass_kg=float(_body_mass) if _body_mass > 0 else ro.mass_kg,
                diameter_m=(float(_last.diameter_m)
                            if _last.diameter_m > 0 else ro.diameter_m),
                length_m=(float(_last.length_m)
                          if _last.length_m > 0 else ro.length_m))
        return ro
    # No reentry object configured.  (Old JSON that stored reentry fields inline
    # is migrated to a synthesised params.ro in booster_from_dict, so by the time
    # we get here params.ro is the single source of truth.)
    return None


# ---------------------------------------------------------------------------
# Shared Cd vs Mach table — Forden Figure 1 piecewise-linear approximation.
# All packaged boosters use this same curve (Forden note 6).
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

# Base-drag coefficient (referenced to base area) vs Mach, power-off.
# Two selectable empirical sources (see MODEL_OPTIONS below):
#   'datcom' — Booster DATCOM 2014, Fig 4.2.3.1-60 (verbatim DATA D4360) for the
#              supersonic table; the subsonic (M<1) portion is retained from
#              Chin because DATCOM's subsonic body base drag is a shape-dependent
#              correlation, not a Mach-only table.
#   'chin'   — Chin (1961) Fig 3-15 base-pressure coefficient (CD_base = -Cpb).
_BASE_MACH_CHIN = [0.0,   0.8,  1.0,  1.2,  1.5,  2.0,  2.5,  3.0,  4.0,  5.0]
_BASE_CDB_CHIN  = [0.000, 0.13, 0.20, 0.18, 0.14, 0.10, 0.08, 0.06, 0.05, 0.04]
_BASE_MACH_DATCOM = [0.0,  0.8,  0.9,   1.0,   1.125, 1.25, 1.5,  2.0,  2.5,
                     3.0,  3.5,  4.0,   4.5,   5.0,   5.5,  6.0]
_BASE_CDB_DATCOM  = [0.000, 0.13, 0.15, 0.178, 0.215, 0.20, 0.178, 0.144, 0.118,
                     0.097, 0.080, 0.068, 0.057, 0.049, 0.042, 0.037]


# ── Swappable reference-data / model sources ────────────────────────────────
# Lets the user pick the empirical source behind a model term, surfaced in the
# GUI under Analysis ▸ Reference Data.  Future toggles (atmosphere, etc.) are
# added as new entries with the same shape; the menu builds itself from this.
MODEL_OPTIONS = {
    "base_drag": {
        "label":   "Base drag",
        "choices": ("datcom", "chin"),
        "labels":  {"datcom": "DATCOM 2014 (Fig 4.2.3.1-60)",
                    "chin":    "Chin 1961 (Fig 3-15)"},
        "default": "datcom",
    },
    "friction": {
        "label":   "Skin friction",
        "choices": ("chin", "sommer_short"),
        "labels":  {"chin":         "Chin 1961 (mixed BL, Frankl-Voishel)",
                    "sommer_short": "Sommer-Short (reference temperature)"},
        "default": "chin",
    },
    "atmosphere": {
        "label":   "Atmosphere",
        "choices": ("msis", "std1976", "hot", "cold", "polar", "tropical"),
        "labels":  {"msis":     "NRLMSISE-00 (mean)",
                    "std1976":  "US Std 1976",
                    "hot":      "MIL-STD-210A hot day",
                    "cold":     "MIL-STD-210A cold day",
                    "polar":    "MIL-STD-210A polar day",
                    "tropical": "MIL-STD-210A tropical day"},
        "default": "msis",
        # Atmosphere model lives in atmosphere.py; reconfigure it on change.
        "apply":   lambda v: configure_atmosphere(model=v),
    },
}
_MODEL_SELECTION = {k: v["default"] for k, v in MODEL_OPTIONS.items()}
# Reflect the atmosphere model actually active (msis may have fallen back to
# std1976 if pymsis is unavailable) so the menu shows the true state.
_MODEL_SELECTION["atmosphere"] = atmosphere_source()


def get_model_option(key: str) -> str:
    """Currently selected source for model option *key*."""
    return _MODEL_SELECTION.get(key, MODEL_OPTIONS[key]["default"])


def set_model_option(key: str, value: str) -> None:
    """Select source *value* for model option *key* (validated)."""
    if key not in MODEL_OPTIONS:
        raise KeyError(f"unknown model option '{key}'")
    if value not in MODEL_OPTIONS[key]["choices"]:
        raise ValueError(f"'{value}' not a choice for '{key}' "
                         f"({MODEL_OPTIONS[key]['choices']})")
    _MODEL_SELECTION[key] = value
    _apply = MODEL_OPTIONS[key].get("apply")
    if _apply is not None:
        _apply(value)


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


# Shape bluntness multipliers for the screening nose-tip-radius default,
# relative to a sharp cone (=1.0), ordered by how rounded each profile runs
# near the tip (cone sharpest → Haack series bluntest; a blunt cylinder has a
# near-hemispherical cap).  These are deliberately MODEST, transparent factors,
# NOT a geometric tip curvature: the idealised nose profiles are all
# geometrically sharp at the very tip (R→0), and real nose-tip bluntness is a
# design choice the outer shape does not fix — every reentry object in ro_library/ is a
# "cone" yet spans 1–5 cm tips.  An explicit nose_radius_m therefore overrides
# this default (see ROParams.effective_nose_radius_m).
_NOSE_BLUNTNESS = {
    'cone':           1.0,
    'tangent_ogive':  1.25,
    'parabola':       1.25,
    'von_karman':     1.5,
    'lv_haack':       1.75,
    'blunt_cylinder': 3.0,
}


def nose_tip_radius(shape: str, diameter_m: float) -> float:
    """Screening default for the Sutton-Graves stagnation radius R_n (m) when an
    RV does not set nose_radius_m explicitly.

    R_n ≈ 0.10·R_body for a sharp cone — mid the 0.1–0.2 R_n/R_base range
    typical of blunted re-entry bodies — scaled up for blunter profiles via
    _NOSE_BLUNTNESS, clamped to [5 mm, R_body].  This is a transparent
    bluntness heuristic, not a geometric derivation; an explicit nose_radius_m
    is authoritative and overrides it.
    """
    R_body = 0.5 * max(0.0, float(diameter_m or 0.0))
    if R_body <= 0.0:
        return 0.05                       # legacy 5 cm fallback (no diameter)
    f = _NOSE_BLUNTNESS.get(_SHAPE_ALIAS.get(shape, shape), 1.0)
    return float(min(max(0.10 * R_body * f, 0.005), R_body))


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


def _cf_sommer_short(re_l: float, mach: float, temp_k: float = 250.0) -> float:
    """Mean all-turbulent flat-plate Cf — Sommer & Short reference-temperature
    method (PDAS TURBSF; Sivells-Payne incompressible base).  Validated to
    within ±3% of the published table at M≤3; better than Frankl-Voishel in the
    hypersonic regime where wall temperature matters.  temp_k defaults to a
    representative 250 K (the result is weakly T-dependent below ~M3)."""
    import math
    if re_l < 10.0:
        return 0.0
    SUTH = 110.4
    xx   = math.log10(re_l) - 1.5
    cfi  = 0.088 / (xx * xx)
    z    = 1.0 + 0.115 * mach * mach
    rstar = re_l / (((temp_k + SUTH) / (z * temp_k + SUTH)) * z ** 2.5)
    xx   = xx / (math.log10(rstar) - 1.5)
    return xx * xx * (cfi / z)


def _cd_friction(re_l: float, mach: float, s_wet_ratio: float) -> float:
    """
    Friction drag coefficient.  Source selectable via MODEL_OPTIONS['friction']:
      'chin' (default) — Blasius laminar (Chin Eq. 4-1) + Schoenherr turbulent
                         (Eq. 4-2), mixed BL at Re_tr=5×10^5 (Eq. 4-3),
                         Frankl-Voishel compressibility (Eq. 4-6).
      'sommer_short'   — Sommer-Short all-turbulent reference-temperature Cf
                         (better in the hypersonic regime).
    Both add a +10% roughness allowance (Chin §4-2).
    s_wet_ratio : S_wet / A_ref
    """
    import math
    if re_l < 1.0 or s_wet_ratio <= 0.0:
        return 0.0
    if get_model_option("friction") == "sommer_short":
        return _cf_sommer_short(re_l, mach) * 1.10 * s_wet_ratio
    re_tr  = 5.0e5
    cf_lam  = 1.328 / math.sqrt(re_l)   # Blasius (Chin Eq. 4-1)
    cf_turb = _cf_schoenherr(re_l)       # Schoenherr (Chin Eq. 4-2)
    s_lam   = min(1.0, re_tr / re_l)
    cf_mix  = cf_lam * s_lam + cf_turb * (1.0 - s_lam)   # Chin Eq. 4-3
    fv      = (1.0 + 0.2 * mach**2) ** (-0.467)           # Frankl-Voishel (Chin Eq. 4-6)
    return cf_mix * fv * 1.10 * s_wet_ratio                # +10% roughness


def _cd_base(mach: float, base_area_ratio: float = 1.0) -> float:
    """Base drag coefficient (ref. base area), power-off.

    Source selectable via MODEL_OPTIONS['base_drag']:
      'datcom' (default) — Booster DATCOM 2014 Fig 4.2.3.1-60
      'chin'             — Chin (1961) Fig 3-15
    """
    if get_model_option("base_drag") == "chin":
        cdb = _lin_interp(mach, _BASE_MACH_CHIN, _BASE_CDB_CHIN)
    else:
        cdb = _lin_interp(mach, _BASE_MACH_DATCOM, _BASE_CDB_DATCOM)
    return cdb * base_area_ratio


def _aerospike_effective_LD(spike_LD: float, spike_dD: float) -> float:
    """
    Effective-body fineness ratio for an aerospiked forebody.
    Derived from Ahmed & Qin (2011) Fig. 5 dividing-streamline angles for
    sharp pointed spikes on hemisphere-cylinder models:
        spike L/D = 1.5 → θ ≈ 14°  → L/D_eff ≈ 2.0
        spike L/D = 2.0 → θ ≈ 12.5° → L/D_eff ≈ 2.3
        spike L/D = 2.5 → θ ≈ 11°  → L/D_eff ≈ 2.6
    Linear fit:  L/D_eff = 1.0 + (2/3)·spike_LD
    Aerodisk contribution from §3.2.1 / Fig. 6:  + 2.0·spike_dD
    """
    if spike_LD <= 0.0:
        return 0.0
    return 1.0 + (2.0 / 3.0) * spike_LD + 2.0 * spike_dD


# ---------------------------------------------------------------------------
# Fin aerodynamics (lift + drag)
# ---------------------------------------------------------------------------

def _cl_alpha_fins(n_fins: int, span_m: float, c_root_m: float,
                   c_tip_m: float, body_diam_m: float,
                   mach: float, sweep_deg: float = 0.0) -> float:
    """
    Total fin normal-force (lift-curve) slope, /radian, referenced to body base
    area A_ref = π(d/2)².  Barrowman 1967 thesis **Eq 3-12** (subsonic, N fins),
    with the standard β = √(M²−1) supersonic extension:

        A_f  = s·(c_root + c_tip)/2        # one exposed fin planform
        AR   = (2s)² / A_f                 # reflected aspect ratio (span b = 2s)
        β    = √|M²−1|                     # Prandtl-Glauert (sub) / supersonic
        Γ_c  = mid-chord sweep; from LE sweep Λ:
                 tan Γ_c = tan Λ + (c_tip − c_root)/(2s)

        (C_Nα)_T = N·π·AR·(A_f/A_ref) / [ 2 + √(4 + (β·AR/cos Γ_c)²) ]

    The N·π numerator (Eq 3-12) is the correct cruciform result: for N=4, two
    fins lie in the pitch plane, giving 2× the single-fin 2π form (Eq 3-6).
    Body-fin interference uses Barrowman's simplified slender-body factor
    (Eq. K_T(B), the r/(s+r) form), identical to 1 + d/(2s+d):

        K_T(B) = 1 + r/(s+r) = 1 + d/(2s+d)

    NOTE — REGIME: this is Barrowman's small-AoA, linear, fin-stabilized
    slender-vehicle theory.  It is for BOOSTER fins (ascent stability / static
    margin), NOT for a high-AoA hypersonic gliding RV, whose L/D is a
    lifting-body (Newtonian) property and is supplied as ro.glider_LD.

    Falls back gracefully when span or chords are not yet entered.
    """
    import math
    if n_fins < 1 or span_m <= 0 or c_root_m <= 0 or body_diam_m <= 0:
        return 0.0
    c_tip_m = max(c_tip_m, 0.0)
    a_f = span_m * (c_root_m + c_tip_m) / 2.0       # one-fin exposed planform
    if a_f <= 0:
        return 0.0
    ar = (2.0 * span_m) ** 2 / a_f                  # reflected aspect ratio
    beta = math.sqrt(abs(mach * mach - 1.0))        # 0 at M=1 (finite denom)
    tan_gc = math.tan(math.radians(sweep_deg)) + (c_tip_m - c_root_m) / (2.0 * span_m)
    cos_gc = 1.0 / math.sqrt(1.0 + tan_gc * tan_gc)
    a_ref = math.pi * (body_diam_m / 2.0) ** 2
    denom = 2.0 + math.sqrt(4.0 + (beta * ar / cos_gc) ** 2)
    cn_alpha = n_fins * math.pi * ar * (a_f / a_ref) / denom
    k_tb = 1.0 + body_diam_m / (2.0 * span_m + body_diam_m)
    return float(cn_alpha * k_tb)


def _cd_fins(n_fins: int, span_m: float, c_root_m: float, c_tip_m: float,
             thickness_m: float, body_diam_m: float,
             mach: float, re_l: float = 5e6,
             sweep_deg: float = 0.0) -> float:
    """
    Total fin drag coefficient increment referenced to body base area A_ref.

    Two components (both at zero angle of attack):

    1. Friction drag — Mandell, Caporaso & Bengen (1973) / FerencDV model:
         CF = 1.328/√Re_c        (laminar, Re_c < 5×10⁵)
              0.074/Re_c⁰·²      (turbulent, Re_c ≥ 5×10⁵)
         mean chord  c_m  = (c_root + c_tip)/2
         AFP  = planform including body-overlap: A_exp + 0.5·c_root·d
         Cd_fric = 2·n·CF·(1 + 2·t/c_m)·AFP / A_ref   (both sides)

    2. Zero-lift wave drag at supersonic speeds (thin airfoil, Ackeret):
         Cd_wave = 4·n·(t/c_m)² / β  · A_exp / A_ref   (leading & trailing)

    Total = Cd_fric + Cd_wave.  Returns 0 when fins are not configured.
    """
    import math
    if n_fins < 1 or span_m <= 0 or c_root_m <= 0 or body_diam_m <= 0:
        return 0.0
    c_tip_m    = max(c_tip_m, 0.0)
    c_m        = 0.5 * (c_root_m + c_tip_m)          # mean chord
    a_exp      = span_m * c_m                          # exposed planform
    a_fp       = a_exp + 0.5 * c_root_m * body_diam_m # full planform (+ overlap)
    a_ref      = math.pi * (body_diam_m / 2.0) ** 2
    if a_ref <= 0 or c_m <= 0:
        return 0.0

    # Skin-friction coefficient at mean-chord Reynolds number
    mu_air = 1.789e-5                      # dynamic viscosity (kg/m·s, sea-level)
    speed_of_sound_ref = 340.0             # m/s reference (exact value doesn't
    # matter — re_l is passed in from the caller which has the real speed)
    # Use chord-based Re approximated from body Re_l and chord/length ratio.
    # Guard against body length zero.
    re_c = re_l * (c_m / max(c_m, 1.0))   # simplifies to re_l when body≈chord
    # A better estimate: use Re_l directly (conservative; fin is shorter).
    re_c = max(re_l * (c_m / max(c_m + span_m, 1e-3)), 1e3)

    if re_c < 5e5:
        cf = 1.328 / math.sqrt(re_c)
    else:
        cf = 0.074 / (re_c ** 0.2)

    tc = thickness_m / c_m if (thickness_m > 0 and c_m > 0) else 0.0
    cd_fric = (2.0 * n_fins * cf * (1.0 + 2.0 * tc) * a_fp) / a_ref

    # Supersonic wave drag (Ackeret thin-airfoil)
    beta = math.sqrt(max(mach ** 2 - 1.0, 0.0))
    if beta > 0.05 and tc > 0:
        cd_wave = (4.0 * n_fins * tc ** 2 / beta) * (a_exp / a_ref)
    else:
        cd_wave = 0.0

    return float(cd_fric + cd_wave)


# ---------------------------------------------------------------------------
# Grid (lattice) fins.  CALIBRATED to Washington & Miller, "Grid Fins - A New
# Concept for Booster Stability and Control," AIAA 93-0035 (the S1 fine-mesh
# configuration, their Fig. 2 geometry and Fig. 14 drag data).  Corroborated by
# three further papers (all read):
#   * Miller & Washington, "An Experimental Investigation of Grid Fin Drag
#     Reduction Techniques," AIAA 94-1914 — fin-only axial force for six
#     frame/web variants.  Confirms the transonic peak (CD rises 0.5→0.9,
#     decreases above 0.9) and quantifies that frame SHAPING cuts drag ~20-45%
#     (subsonic) / ~8-27% (supersonic; half-diamond best) vs the blunt baseline
#     F1, and that thick webs add ~13-19%.  This motivates the edge-shape factor
#     below; the model's default (blunt) matches W&M S1 ≈ Miller F1.
#   * DeSpirito & Sahu, ARL-RP-19 / AIAA 2001-0257 — total-booster Cx ≈ 0.43
#     (M2) → 0.45 (M3), roughly flat: corroborates the flat supersonic baseline.
#   * Abate, Duckerschein & Hathaway, AIAA 2000-0937 — free-flight GTCM;
#     total Cx flat below M≈0.77 then a steep transonic rise to a peak ~M1.05,
#     independently confirming the choke ONSET Mach (~0.77 ≈ _GRIDFIN_M_SUB).
# A grid fin is a
# box-frame lattice of thin cells, NOT a planar airfoil, so the flat-plate/
# Ackeret _cd_fins model does not apply.  W&M's measured axial-force (drag)
# coefficient, referenced to body cross-section, for the S1 fin (96.8% open,
# blunt-edged webs) is roughly FLAT at ~0.040 outside transonic with a modest
# transonic BUMP to ~0.065 (≈1.5×) peaking near M≈0.95 — NOT a large spike.
# The model therefore is:
#
#   drag = friction (on the wetted web area, chord Reynolds)
#        + blunt edge/profile drag (× web frontal blockage area)
#        + a transonic bump over [M_sub, M_rec] peaking at M_peak.
#
# The three flow regimes W&M describe (Fig. 6/7) — cells CHOKE below M=1, flow
# spills around the fin, the shock attaches then passes undisturbed, restoring
# supersonic behaviour by M≈1.6 — set the bump's onset/peak/recovery Mach
# anchors.  W&M (and Miller, Ref. 3) used 1-D isentropic relations for these;
# the Kantrowitz helper below reproduces that class of analysis, but note its
# GEOMETRIC contraction (1/porosity) under-predicts the choke for thin-web
# fins because boundary-layer blockage in the small cells is the co-cause W&M
# identify — so the bump Mach anchors are taken from the S1 data, not from
# geometric Kantrowitz alone.
#
# CALIBRATION CAVEATS: anchored to a single blunt-edged config (S1).  Sharp
# edges cut supersonic drag (W&M note this explicitly); the bucket shifts with
# cell size / Reynolds number; extrapolation to other geometries is uncertain.
# Constants are exposed for tuning.
#
# Supersonic corroboration (qualitative only): DeSpirito & Sahu, "Viscous CFD
# Calculations of Grid Fin Booster Aerodynamics in the Supersonic Flow Regime,"
# ARL-RP-19 / AIAA 2001-0257, measure a TOTAL-booster axial force Cx ≈ 0.43 at
# M2 and ≈ 0.45 at M3 (roughly flat / slightly rising) on a grid-finned TCAAM.
# That flat supersonic trend is consistent with this model's flat baseline and
# rules out a decaying-with-Mach form, but DeSpirito does NOT isolate the fin
# increment nor specify the cell web/pitch, so it is corroboration, not a
# quantitative fin-drag validation.
_GRIDFIN_CD_EDGE = 0.50   # blunt LE+TE / profile drag (× web blockage area)
_GRIDFIN_BUMP    = 0.55   # transonic peak as a fraction above the baseline
_GRIDFIN_M_SUB   = 0.75   # drag-rise onset Mach (W&M S1; Abate free-flight 0.77)
_GRIDFIN_M_PEAK  = 0.97   # Mach of the transonic drag peak (W&M S1; Miller ~0.9)
_GRIDFIN_M_REC   = 1.60   # Mach of recovery to supersonic behaviour (W&M S1)
# Edge-shape factor on the pressure (edge + transonic-bump) drag, NOT friction.
# 1.0 = blunt rectangular webs (W&M S1, Miller F1 baseline — the conservative
# default).  Miller 94-1914 Tables 2/3: shaping the frame cross-section to a
# single wedge / half-diamond cuts drag ~20-45% subsonic, ~8-27% supersonic, so
# a sharp/shaped fin is ~0.6-0.85.  Per-vehicle via grid_fin_edge_factor.
_GRIDFIN_EDGE_BLUNT = 1.0


def _isentropic_area_ratio(mach: float, gamma: float = 1.4) -> float:
    """A/A* for quasi-1D isentropic flow (Mach ≠ 0)."""
    import math
    if mach <= 0:
        return float('inf')
    return ((1.0 / mach)
            * ((2.0 / (gamma + 1.0)) * (1.0 + 0.5 * (gamma - 1.0) * mach * mach))
            ** ((gamma + 1.0) / (2.0 * (gamma - 1.0))))


def _kantrowitz_contraction_ratio(mach: float, gamma: float = 1.4) -> float:
    """
    Maximum self-starting internal contraction ratio CR = A_capture/A_throat
    at flight Mach `mach` (Kantrowitz & Donaldson, NACA 1945).  A normal shock
    at the face decelerates the flow to subsonic Mach M_y; the throat must then
    pass that subsonic flow at M=1, so CR = (A/A*)|_{M_y}.
    """
    import math
    if mach <= 1.0:
        return 1.0
    my2 = ((1.0 + 0.5 * (gamma - 1.0) * mach * mach)
           / (gamma * mach * mach - 0.5 * (gamma - 1.0)))
    return _isentropic_area_ratio(math.sqrt(my2), gamma)


def _gridfin_start_mach(contraction_ratio: float, gamma: float = 1.4) -> float:
    """
    Mach at which a lattice of contraction ratio CR (=1/porosity) self-starts,
    by inverting the Kantrowitz limit (monotone in M).  Bisection on [1.01, 8].
    """
    cr = max(float(contraction_ratio), 1.0 + 1e-6)
    lo, hi = 1.01, 8.0
    if _kantrowitz_contraction_ratio(hi, gamma) < cr:
        return hi
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _kantrowitz_contraction_ratio(mid, gamma) < cr:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Grid-fin SOLIDITY (σ)
# ---------------------------------------------------------------------------
# Solidity σ is the fraction of the grid fin's frontal frame area that is
# blocked by the lattice webs (σ = 1 − porosity φ).  It is the single physical
# quantity that drives grid-fin drag, and it stands in for TWO geometric
# details that are very hard to obtain from open sources — the web (wall)
# thickness t and the cell pitch (centre-to-centre spacing) p.
#
# If you DO know t and p, compute σ exactly for a square lattice from:
#
#       σ = 1 − ((p − t) / p)²        (≈ 2·t/p for thin webs)
#
# If you DON'T, estimate σ from imagery (how "filled in" the lattice looks).
# Empirical range for REAL fins (see METHODS "Empirical σ range"): the aero
# region is σ ≈ 0.04–0.12.  Large booster / launch-vehicle / SLBM fins (STARS,
# Topol, Falcon 9, R-77 boosters) sit open, σ ≈ 0.04–0.06; the smaller air-to-
# air (AA-12) class is σ ≈ 0.09–0.12.  σ ≳ 0.15 is atypical for an aero surface
# (only structural fin-roots / deliberately thick CFD cases — they choke
# transonically).  Booster default ≈ 0.05.  Supplying σ avoids guessing t and p
# separately; grid_fin_solidity() below converts t,p → σ when you have them.

# Representative cells-across-frame used to estimate the (secondary) friction
# wetted area when σ is given but the cell pitch is not.
_GRIDFIN_DEFAULT_CELLS = 10.0


def grid_fin_solidity(web_thickness_m: float, cell_pitch_m: float) -> float:
    """
    Grid-fin solidity σ (blocked frontal fraction) from web thickness and cell
    pitch, assuming a square lattice:

        σ = 1 − ((p − t) / p)²

    Returns σ clamped to (0, 1).  Use this to convert a known web/pitch pair
    into the σ that the drag model consumes; if web/pitch are unknown, estimate
    σ directly from imagery instead (see the section header above).
    """
    p = float(cell_pitch_m)
    t = float(web_thickness_m)
    if p <= 0.0:
        return 0.0
    t = min(max(t, 0.0), p)                       # web cannot exceed the pitch
    sigma = 1.0 - ((p - t) / p) ** 2
    return min(max(sigma, 1e-3), 0.999)


def _gridfin_geometry(width_m: float, height_m: float, chord_m: float,
                      web_thickness_m: float, cell_pitch_m: float,
                      solidity: float = 0.0):
    """
    Derived lattice geometry for one grid fin.  Returns
    (A_frame, porosity φ, A_block, web_wetted_area).  Square cells assumed.

    Blockage source: if `solidity` (σ) > 0 it is used directly (the
    approximation path — σ stands in for web thickness + cell pitch); otherwise
    σ is derived from web_thickness_m and cell_pitch_m via grid_fin_solidity().
    The friction wetted area needs an absolute cell pitch; when only σ is given
    (no pitch), a representative pitch (frame / _GRIDFIN_DEFAULT_CELLS) is used,
    which affects only the secondary friction term.
    """
    a_frame = max(width_m * height_m, 0.0)
    if solidity and solidity > 0.0:
        phi = 1.0 - min(max(float(solidity), 1e-3), 0.999)
        pitch = (cell_pitch_m if cell_pitch_m > 0.0
                 else max(min(width_m, height_m), 1e-3) / _GRIDFIN_DEFAULT_CELLS)
    else:
        pitch = max(cell_pitch_m, web_thickness_m + 1e-4)
        phi = ((pitch - web_thickness_m) / pitch) ** 2     # open-area fraction
    phi = min(max(phi, 0.02), 0.98)
    a_block = (1.0 - phi) * a_frame
    # Total web length in the frontal plane (horizontal + vertical members).
    # For a w×h frame on pitch p this is ≈ 2·A_frame/p for many cells.
    l_web = 2.0 * a_frame / pitch
    a_wet = 2.0 * l_web * max(chord_m, 0.0)            # both faces of each web
    return a_frame, phi, a_block, a_wet


def grid_fins_deployed(n_total: int, deploy_schedule, t_s) -> int:
    """
    Number of grid fins aerodynamically active at mission time t_s.

    deploy_schedule is a list of [deploy_time_s, n_fins] batches; n_fins become
    active once t_s >= deploy_time_s.  An empty/None schedule (or t_s None) means
    all n_total fins are active (the steady-state / no-timing case).  The result
    is capped at n_total.
    """
    if n_total <= 0:
        return 0
    if not deploy_schedule or t_s is None:
        return n_total
    deployed = 0
    for entry in deploy_schedule:
        try:
            t_dep, count = float(entry[0]), int(entry[1])
        except (TypeError, IndexError, ValueError):
            continue
        if t_s >= t_dep:
            deployed += count
    return max(0, min(deployed, n_total))


def _cd_gridfins(n_fins: int, width_m: float, height_m: float, chord_m: float,
                 web_thickness_m: float, cell_pitch_m: float,
                 body_diam_m: float, mach: float, re_chord: float = 5e6,
                 edge_factor: float = _GRIDFIN_EDGE_BLUNT,
                 solidity: float = 0.0) -> float:
    """
    Total grid-fin axial-force (drag) coefficient increment referenced to the
    body base area A_ref = π(d/2)².  Calibrated to Washington & Miller S1
    (AIAA 93-0035, Fig. 14): a roughly flat baseline (web friction + blunt-edge
    drag) plus a modest transonic bump.  See the module header for references
    and calibration caveats.  Returns 0 when not configured.

    Blockage source: if `solidity` (σ) > 0 it is used directly (σ stands in for
    web thickness + cell pitch — see grid_fin_solidity()); otherwise σ is
    derived from web_thickness_m and cell_pitch_m.

    edge_factor scales the pressure (edge + transonic-bump) drag, not friction:
    1.0 = blunt webs (W&M S1 / Miller F1 baseline); ~0.6-0.85 for shaped/sharp
    frames (Miller 94-1914 Tables 2/3).

    Validated against W&M S1 (4 fins, frame 2.14×3.243 in, web 0.006 in, pitch
    0.371 in, chord 0.384 in, 5.0 in body): reproduces ~0.042 subsonic,
    ~0.065 transonic peak, ~0.038 supersonic.
    """
    import math
    if (n_fins < 1 or width_m <= 0 or height_m <= 0 or chord_m <= 0
            or body_diam_m <= 0):
        return 0.0
    a_ref = math.pi * (body_diam_m / 2.0) ** 2
    if a_ref <= 0:
        return 0.0
    a_frame, phi, a_block, a_wet = _gridfin_geometry(
        width_m, height_m, chord_m, web_thickness_m, cell_pitch_m,
        solidity=solidity)

    # Baseline drag (roughly Mach-flat outside transonic):
    #   (a) skin friction on the wetted web area (flat-plate, chord Reynolds)
    #   (b) blunt leading/trailing-edge + profile drag on the web blockage area,
    #       scaled by the edge-shape factor (sharpening cuts pressure drag).
    re_c = max(re_chord, 1e3)
    cf = 1.328 / math.sqrt(re_c) if re_c < 5e5 else 0.074 / (re_c ** 0.2)
    cd_fric = n_fins * cf * a_wet / a_ref
    cd_edge = n_fins * _GRIDFIN_CD_EDGE * a_block / a_ref * max(edge_factor, 0.0)
    base = cd_fric + cd_edge

    # Transonic bump (choke → spillage → shock-attachment), peaking at M_peak
    # and recovering by M_rec.  Smooth half-cosines, zero outside [M_sub, M_rec].
    bump = 0.0
    if _GRIDFIN_M_SUB <= mach <= _GRIDFIN_M_REC:
        if mach <= _GRIDFIN_M_PEAK:
            x = (mach - _GRIDFIN_M_SUB) / (_GRIDFIN_M_PEAK - _GRIDFIN_M_SUB)
            shape = 0.5 * (1.0 - math.cos(math.pi * x))          # 0 → 1
        else:
            x = (mach - _GRIDFIN_M_PEAK) / (_GRIDFIN_M_REC - _GRIDFIN_M_PEAK)
            shape = 0.5 * (1.0 + math.cos(math.pi * x))          # 1 → 0
        bump = _GRIDFIN_BUMP * base * shape

    return float(base + bump)


def _cl_alpha_gridfins(n_fins: int, width_m: float, height_m: float,
                       chord_m: float, web_thickness_m: float,
                       cell_pitch_m: float, body_diam_m: float,
                       mach: float, solidity: float = 0.0) -> float:
    """
    Grid-fin normal-force slope (/rad) referenced to body base area, used by
    the L/D estimator (the trajectory integrator flies a pitch program and does
    not use fin lift).  Reduced-order cascade model: the lattice members act as
    low-aspect-ratio lifting surfaces; the slope scales with the lifting
    planform (≈ web-wetted area) and a per-Mach 2-D lift slope, with the
    transonic "bucket" (W&M Fig. 6: C_NF,α drops through M_sub..M_rec) folded
    in.  Approximate — flagged as such; the trajectory does not use it.
    """
    import math
    if (n_fins < 1 or width_m <= 0 or height_m <= 0 or chord_m <= 0
            or body_diam_m <= 0):
        return 0.0
    a_ref = math.pi * (body_diam_m / 2.0) ** 2
    if a_ref <= 0:
        return 0.0
    a_frame, phi, a_block, a_wet = _gridfin_geometry(
        width_m, height_m, chord_m, web_thickness_m, cell_pitch_m,
        solidity=solidity)
    # 2-D lift slope per member surface (Prandtl-Glauert / Ackeret)
    if mach < 0.95:
        cla_2d = 2.0 * math.pi / math.sqrt(max(1.0 - mach * mach, 0.04))
    else:
        cla_2d = 4.0 / math.sqrt(max(mach * mach - 1.0, 0.04))
    # Lifting planform ≈ half the wetted web area (the load-bearing members)
    a_lift = 0.5 * a_wet
    # Transonic "bucket": effectiveness drops through the choked band
    # [M_sub, M_rec] and recovers to full once the cells start (W&M Fig. 6).
    eff = 1.0
    if _GRIDFIN_M_SUB <= mach < _GRIDFIN_M_REC:
        eff = 0.5                                    # degraded while choked
    return float(n_fins * cla_2d * eff * a_lift / a_ref)


def _cd_nose_shape(nose_shape: str, ld: float, mach: float,
                   re_l: float = 5e6, ld_body: float = None,
                   aerospike_LD: float = 0.0,
                   aerospike_dD: float = 0.0) -> float:
    """
    Total zero-lift drag coefficient (Cd_wave + Cd_friction + Cd_base).
    Source: Chin (1961) *Booster Configuration Design*; NACA TN 4201; Crowell (1996).

    nose_shape   : key from NOSE_SHAPES
    ld           : nose fineness ratio = nose_length / body_diameter (clamped 0.5–10)
    mach         : free-stream Mach number
    re_l         : Reynolds number based on body length (default 5×10^6)
    ld_body      : full-body fineness ratio = body_length / body_diameter;
                   drives cylinder friction term.  None → 2×ld estimate.
    aerospike_LD : spike length / body diameter (0 = no aerospike)
    aerospike_dD : aerodisk diameter / body diameter (0 = pointed tip)
                   When aerospike_LD > 0, wave drag is replaced by the
                   minimum of (actual nose, effective-body cone) — see
                   _aerospike_effective_LD docstring.  Active only above
                   Mach 0.8 since a spike requires a bow shock to replace.
    """
    nose_shape = _SHAPE_ALIAS.get(nose_shape, nose_shape)
    ld   = max(0.5, min(float(ld), 10.0))
    mach = max(0.0, float(mach))

    # ── Blunt Cylinder ────────────────────────────────────────────────────────
    if nose_shape == 'blunt_cylinder':
        if mach <= 0.8:   cd_blunt = 0.9
        elif mach <= 1.5: cd_blunt = 0.9 + (mach - 0.8) / 0.7 * 1.3
        else:             cd_blunt = 2.2
        if aerospike_LD > 0.0 and mach > 0.8:
            # Spike replaces the blunt shock with an effective slender cone.
            ld_eff = _aerospike_effective_LD(aerospike_LD, aerospike_dD)
            # Friction (~0.05) + base (~0.06) typical for hemisphere-cylinder.
            return _cd_wave_cone(ld_eff, mach) + 0.11
        return cd_blunt

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

    # ── Aerospike: replace wave drag with effective-body cone if smaller ──────
    if aerospike_LD > 0.0 and mach > 0.8:
        ld_eff = _aerospike_effective_LD(aerospike_LD, aerospike_dD)
        cd_wave = min(cd_wave, _cd_wave_cone(ld_eff, mach))

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
# ---------------------------------------------------------------------------
# Built-in booster database (data files, not code)
# ---------------------------------------------------------------------------

# Boosters are DATA, not code: BOOSTER_DB is populated from
# booster_library/*.booster.json by load_booster_library() at the end of this
# module.  Additional boosters are loaded at runtime from custom_boosters.json
# and overlay any same-name entries.
BOOSTER_DB: dict = {}


def get_booster(name: str, plan: str = None) -> BoosterParams:
    """Return the named booster with its flight plan applied.

    A booster can fly many flight plans: the booster-named file is the
    "(default)" plan; named variants (see list_flight_plans) reference the
    booster and are applied ON TOP of the default, so a variant need only
    carry the fields it changes.  ``plan`` selects a variant explicitly;
    when None, the ACTIVE_FLIGHT_PLANS selection (set by the GUI) is used,
    falling back to the default.  Headless callers that never touch either
    get the default plan, exactly as before.
    """
    if name not in BOOSTER_DB:
        raise ValueError(f"Unknown booster '{name}'. Available: {list(BOOSTER_DB)}")
    p = BOOSTER_DB[name]()
    fp = load_flight_plan(name, extra_dirs=USER_FLIGHT_PLAN_DIRS)
    if fp is not None:
        p = apply_flight_plan(p, fp)
    if plan is None:
        plan = ACTIVE_FLIGHT_PLANS.get(name)
    if plan and plan != DEFAULT_PLAN_LABEL:
        vp = load_flight_plan(name, extra_dirs=USER_FLIGHT_PLAN_DIRS, plan=plan)
        if vp is not None:
            p = apply_flight_plan(p, vp)
    return p


def _migrate_guidance(g: str) -> str:
    """Migrate legacy guidance keys.

    The old "gravity_turn" key actually meant a user-directed pitch program
    (thrust pointed at a fixed elevation in ENU).  It was renamed to
    "pitch_program" when a true gravity-turn mode (thrust aligned with the
    velocity vector, internally "true_gravity_turn") was added.  Old JSON
    files use the legacy name and are silently migrated here.
    """
    if g == 'gravity_turn':
        return 'pitch_program'
    return g


def _convert_loft_to_gravity_turn(p: BoosterParams) -> None:
    """In-place conversion of a 'loft' booster to equivalent gravity_turn pitch overrides.

    The Forden formula el(t)=max(la, 90−rate·t) is mathematically identical to
    a linear gravity_turn with turn_start=0 and turn_stop=(90−la)/rate.
    """
    la   = p.burnout_angle_deg
    rate = p.loft_angle_rate_deg_s
    p.guidance = 'pitch_program'
    if p.stage_turn_start_s is None:
        p.stage_turn_start_s = 0.0
    if p.stage_turn_stop_s is None and rate > 0:
        p.stage_turn_stop_s = round((90.0 - la) / rate, 1)
    if p.stage_burnout_angle_deg is None:
        p.stage_burnout_angle_deg = la
    # Inner stages: set guidance and hold at the top-level target angle.
    s = p.stage2
    while s is not None:
        if s.guidance == 'loft':
            s.guidance = 'pitch_program'
        if s.stage_burnout_angle_deg is None:
            s.stage_burnout_angle_deg = la
        s = s.stage2


def booster_to_dict(p: BoosterParams, include_flight_plan: bool = True) -> dict:
    """Serialise a BoosterParams to a JSON-compatible dict.

    With ``include_flight_plan=False`` the flight-plan fields (guidance and the
    per-stage schedule -- everything in ``_FLIGHT_PLAN_TOP_KEYS`` and
    ``_FLIGHT_PLAN_STAGE_KEYS``) are omitted at every nesting level, yielding a
    hardware-only booster.  That is the form the booster library stores; the
    flight plan travels separately in a ``.flightplan.json`` file.  The default
    (``True``) keeps the full serialisation for internal round-trips.
    """
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
        'burnout_angle_deg':        p.burnout_angle_deg,
        'loft_angle_rate_deg_s': p.loft_angle_rate_deg_s,
        'mach_table':            list(p.mach_table),
        'cd_table':              list(p.cd_table),
        'payload_kg':            p.payload_kg,
        'ro_separates':          p.ro_separates,
        'bus_mass_kg':           p.bus_mass_kg,
        'num_ros':               p.num_ros,
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
        'aerospike_LD':           p.aerospike_LD,
        'aerospike_dD':           p.aerospike_dD,
        'has_fins':               p.has_fins,
        'n_fins':                 p.n_fins,
        'fin_span_m':             p.fin_span_m,
        'fin_root_chord_m':       p.fin_root_chord_m,
        'fin_tip_chord_m':        p.fin_tip_chord_m,
        'fin_thickness_m':        p.fin_thickness_m,
        'fin_sweep_deg':          p.fin_sweep_deg,
        'has_grid_fins':          p.has_grid_fins,
        'n_grid_fins':            p.n_grid_fins,
        'grid_fin_width_m':         p.grid_fin_width_m,
        'grid_fin_height_m':        p.grid_fin_height_m,
        'grid_fin_chord_m':         p.grid_fin_chord_m,
        'grid_fin_web_thickness_m': p.grid_fin_web_thickness_m,
        'grid_fin_cell_pitch_m':    p.grid_fin_cell_pitch_m,
        'grid_fin_solidity':        p.grid_fin_solidity,
        'grid_fin_edge_factor':     p.grid_fin_edge_factor,
        'grid_fin_deploy_schedule': list(p.grid_fin_deploy_schedule or []),
        'payload_diameter_m':     p.payload_diameter_m,
        'pbv_diameter_m':         p.pbv_diameter_m,
        'pbv_length_m':           p.pbv_length_m,
        'source':                 p.source,
        'notes':                  p.notes,
        'n_boosters':             p.n_boosters,
        'booster_thrust_n':       p.booster_thrust_n,
        'booster_burn_time_s':    p.booster_burn_time_s,
        'booster_inert_kg':       p.booster_inert_kg,
        'booster_prop_kg':        p.booster_prop_kg,
        'booster_isp_s':          p.booster_isp_s,
        'booster_nozzle_area_m2': p.booster_nozzle_area_m2,
        'booster_diam_m':         p.booster_diam_m,
        'booster_length_m':       p.booster_length_m,
        'booster_cd':             p.booster_cd,
        'booster_core_delay_s':   p.booster_core_delay_s,
        'booster_jettison_s':     p.booster_jettison_s,
    }
    # Reentry object: written as the embedded 'ro' dict when present.  The
    # booster carries no reentry hardware, so there is nothing else to write.
    if p.ro is not None:
        d['ro'] = ro_to_dict(p.ro)
    # Per-stage pitch overrides — only written when set (keeps dicts compact)
    if p.stage_turn_start_s is not None:
        d['stage_turn_start_s'] = p.stage_turn_start_s
    if p.stage_turn_stop_s is not None:
        d['stage_turn_stop_s'] = p.stage_turn_stop_s
    if p.stage_burnout_angle_deg is not None:
        d['stage_burnout_angle_deg'] = p.stage_burnout_angle_deg
    if p.stage_cutoff_s is not None:
        d['stage_cutoff_s'] = p.stage_cutoff_s
    if p.stage_yaw_start_s is not None:
        d['stage_yaw_start_s'] = p.stage_yaw_start_s
    if p.stage_yaw_stop_s is not None:
        d['stage_yaw_stop_s'] = p.stage_yaw_stop_s
    if p.stage_yaw_final_az_deg is not None:
        d['stage_yaw_final_az_deg'] = p.stage_yaw_final_az_deg
    if p.stage2 is not None:
        d['stage2'] = booster_to_dict(p.stage2, include_flight_plan)
    if not include_flight_plan:
        for _k in (*_FLIGHT_PLAN_TOP_KEYS, *_FLIGHT_PLAN_STAGE_KEYS):
            d.pop(_k, None)
    return d


def booster_from_dict(d: dict) -> BoosterParams:
    """Reconstruct a BoosterParams from a dict produced by booster_to_dict."""
    prop  = float(d['mass_propellant'])
    burn  = float(d['burn_time_s'])
    isp   = float(d['isp_s'])
    m0    = float(d['mass_initial'])
    stage2 = booster_from_dict(d['stage2']) if d.get('stage2') else None
    _p = BoosterParams(
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
        guidance=_migrate_guidance(d.get('guidance', 'pitch_program')),
        burnout_angle_deg=float(d.get('burnout_angle_deg', d.get('loft_angle_deg', 45.0))),
        loft_angle_rate_deg_s=float(d.get('loft_angle_rate_deg_s', 2.0)),
        mach_table=list(d.get('mach_table', _FORDEN_MACH)),
        cd_table=list(d.get('cd_table', _FORDEN_CD)),
        stage2=stage2,
        payload_kg=float(d.get('payload_kg', 0.0)),
        ro_separates=bool(d.get('ro_separates', d.get('rv_separates', False))),
        bus_mass_kg=float(d.get('bus_mass_kg', 0.0)),
        num_ros=int(d.get('num_ros', d.get('num_rvs', 1))),
        ro_mass_kg=float(d.get('ro_mass_kg', d.get('rv_mass_kg', 0.0))),
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
        aerospike_LD=float(d.get('aerospike_LD', 0.0)),
        aerospike_dD=float(d.get('aerospike_dD', 0.0)),
        has_fins=bool(d.get('has_fins', False)),
        n_fins=int(d.get('n_fins', 4)),
        fin_span_m=float(d.get('fin_span_m', 0.0)),
        fin_root_chord_m=float(d.get('fin_root_chord_m', 0.0)),
        fin_tip_chord_m=float(d.get('fin_tip_chord_m', 0.0)),
        fin_thickness_m=float(d.get('fin_thickness_m', 0.0)),
        fin_sweep_deg=float(d.get('fin_sweep_deg', 0.0)),
        has_grid_fins=bool(d.get('has_grid_fins', False)),
        n_grid_fins=int(d.get('n_grid_fins', 0)),
        grid_fin_width_m=float(d.get('grid_fin_width_m', 0.0)),
        grid_fin_height_m=float(d.get('grid_fin_height_m', 0.0)),
        grid_fin_chord_m=float(d.get('grid_fin_chord_m', 0.0)),
        grid_fin_web_thickness_m=float(d.get('grid_fin_web_thickness_m', 0.0)),
        grid_fin_cell_pitch_m=float(d.get('grid_fin_cell_pitch_m', 0.0)),
        grid_fin_solidity=float(d.get('grid_fin_solidity', 0.0)),
        grid_fin_edge_factor=float(d.get('grid_fin_edge_factor', 1.0)),
        grid_fin_deploy_schedule=list(d.get('grid_fin_deploy_schedule', []) or []),
        payload_diameter_m=float(d.get('payload_diameter_m', 0.0)),
        pbv_diameter_m=float(d.get('pbv_diameter_m', 0.0)),
        pbv_length_m=float(d.get('pbv_length_m', 0.0)),
        source=str(d.get('source', '')),
        notes=str(d.get('notes', '')),
        n_boosters=int(d.get('n_boosters', 0)),
        booster_thrust_n=float(d.get('booster_thrust_n', 0.0)),
        booster_burn_time_s=float(d.get('booster_burn_time_s', 0.0)),
        booster_inert_kg=float(d.get('booster_inert_kg', 0.0)),
        booster_prop_kg=float(d.get('booster_prop_kg', 0.0)),
        booster_isp_s=float(d.get('booster_isp_s', 0.0)),
        booster_nozzle_area_m2=float(d.get('booster_nozzle_area_m2', 0.0)),
        booster_diam_m=float(d.get('booster_diam_m', 0.0)),
        booster_length_m=float(d.get('booster_length_m', 0.0)),
        booster_cd=float(d.get('booster_cd', 0.20)),
        booster_core_delay_s=float(d.get('booster_core_delay_s', 0.0)),
        booster_jettison_s=float(d.get('booster_jettison_s', 0.0)),
        stage_turn_start_s=(float(d['stage_turn_start_s'])
                            if d.get('stage_turn_start_s') is not None else None),
        stage_turn_stop_s=(float(d['stage_turn_stop_s'])
                           if d.get('stage_turn_stop_s') is not None else None),
        stage_burnout_angle_deg=(float(d['stage_burnout_angle_deg'])
                                 if d.get('stage_burnout_angle_deg') is not None else None),
        stage_cutoff_s=(float(d['stage_cutoff_s'])
                        if d.get('stage_cutoff_s') is not None else None),
        stage_yaw_start_s=(float(d['stage_yaw_start_s'])
                           if d.get('stage_yaw_start_s') is not None else None),
        stage_yaw_stop_s=(float(d['stage_yaw_stop_s'])
                          if d.get('stage_yaw_stop_s') is not None else None),
        stage_yaw_final_az_deg=(float(d['stage_yaw_final_az_deg'])
                                if d.get('stage_yaw_final_az_deg') is not None else None),
    )
    # Load RV object when present (new format); legacy inline fields stay on
    # _p for effective_ro() to find when _p.ro is None (old format).
    if 'ro' in d or 'rv' in d:          # 'rv' = legacy embedded-object key
        _p.ro = ro_from_dict(d.get('ro') or d['rv'])
    else:
        # Migrate genuinely-old JSON that stored reentry hardware inline on the
        # booster (no embedded 'ro'/'rv' key) into a synthesised ROParams, so the
        # booster itself carries no reentry fields.  Gated on an inline β.
        _rb = float(d.get('ro_beta_kg_m2', d.get('rv_beta_kg_m2', 0.0)) or 0.0)
        if _rb > 0:
            _g = str(d.get('glider_guidance', 'equilibrium_glide'))
            if _g == 'constant_bank':
                _g = 'skip_glide'
            _p.ro = ROParams(
                name='(migrated)',
                mass_kg=float(d.get('ro_mass_kg', d.get('rv_mass_kg', 0.0))
                              or d.get('payload_kg', 0.0)),
                beta_kg_m2=_rb,
                shape=str(d.get('ro_shape', d.get('rv_shape', ''))),
                diameter_m=float(d.get('ro_diameter_m', d.get('rv_diameter_m', 0.0))),
                length_m=float(d.get('ro_length_m', d.get('rv_length_m', 0.0))),
                glider_enabled=bool(d.get('glider_enabled', False)),
                glider_LD=float(d.get('glider_LD', 0.0)),
                glider_guidance=_g,
                glider_pullup_g_max=float(d.get('glider_pullup_g_max', 10.0)),
                glider_terminal_dive=bool(d.get('glider_terminal_dive', False)),
                glider_terminal_alt_km=float(d.get('glider_terminal_alt_km', 30.0)),
                separation_mode=('separating_ro'
                                 if bool(d.get('ro_separates', d.get('rv_separates', False)))
                                 else 'body'),
            )
    # Backwards compatibility: old saved boosters with guidance="loft" are
    # auto-converted to gravity_turn with equivalent per-stage pitch overrides.
    if d.get('guidance', '') == 'loft':
        _convert_loft_to_gravity_turn(_p)
    return _p  # type: ignore[return-value]


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


def _eff_burn(s: BoosterParams) -> float:
    """Effective powered-burn duration for stage `s` (s).

    Equals burn_time_s unless a commanded engine cutoff (stage_cutoff_s) shuts
    the stage down early.  Cutoff applies to liquid engines only — a solid
    motor burns to completion regardless.  The value is clamped to
    [0, burn_time_s]; the stage's slot in the timeline is unchanged (only the
    powered fraction shrinks), so this never moves staging or guidance timing.
    """
    c = s.stage_cutoff_s
    if c is None or s.solid_motor or s.burn_time_s <= 0:
        return s.burn_time_s
    return max(0.0, min(float(c), s.burn_time_s))


def total_burn_time(params: BoosterParams) -> float:
    """Total time from launch (T=0) to end of last stage's burn.

    Includes booster_core_delay_s when strap-ons ignite before the core.
    """
    t, s = params.booster_core_delay_s, params
    while s is not None:
        t += s.burn_time_s
        if s.stage2 is not None:
            t += s.coast_time_s   # inter-stage coast before next ignition
        s  = s.stage2
    return t


def active_stage(params: BoosterParams, t: float) -> BoosterParams:
    """Return the BoosterParams for the stage (or vehicle) active at time t.

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


def active_stage_and_t(params: BoosterParams, t: float):
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


def booster_separation_time(params: BoosterParams) -> float:
    """Time (s after T=0) the spent strap-on boosters physically leave.

    booster_jettison_s if set later than burnout, else burnout.  A jettison
    time at or before burnout is treated as "separate at burnout" — boosters
    cannot leave while still thrusting.
    """
    t_b = params.booster_burn_time_s
    return params.booster_jettison_s if params.booster_jettison_s > t_b else t_b


def _booster_mass_addend(params: BoosterParams, t: float) -> float:
    """Mass (kg) contributed by attached strap-on boosters; 0 after separation.

    Between burnout and jettison (when booster_jettison_s > burn time) the
    spent boosters ride along as dead inert mass.
    """
    n, t_b = params.n_boosters, params.booster_burn_time_s
    if n <= 0 or t_b <= 0 or t > booster_separation_time(params):
        return 0.0
    t = max(0.0, t)
    if t >= t_b:                       # burned out, not yet jettisoned
        return n * params.booster_inert_kg
    return (n * (params.booster_prop_kg + params.booster_inert_kg)
            - n * params.booster_prop_kg / t_b * t)


# Free-molecular heating flux at which a payload shroud/fairing is jettisoned.
# 1135 W/m^2 = 0.1 BTU/ft^2-s, the standard launch-vehicle fairing-jettison
# thermal criterion (ULA Atlas V / Delta IV and SpaceX Falcon user's guides;
# collated in Isakowitz, International Reference Guide to Space Launch Systems).
# q_dot = 1/2 rho V^3 is the free-molecular convective flux (accommodation ~1).
SHROUD_Q_FAIRING = 1135.0   # W/m^2


def _shroud_jettisoned(params: 'BoosterParams', alt_m: float) -> bool:
    """Has the payload shroud been jettisoned by this point in the flight?

    Two modes, selected by shroud_jettison_alt_km:
      * > 0  — explicit altitude override: jettison at that altitude (altitude
               is monotonic during boost, so this is self-latching).
      * <= 0 — heating default ("exoatmospheric"): jettison once the
               free-molecular flux 1/2 rho V^3 falls below SHROUD_Q_FAIRING
               after max-q.  The EOM maintains the latch
               params._shroud_latch = [armed, jettisoned, t_jettison]; this
               reads the jettisoned flag.  Before integration (no latch set),
               the shroud is treated as still attached.
    """
    if params.shroud_jettison_alt_km > 0:
        return alt_m / 1000.0 >= params.shroud_jettison_alt_km
    lat = getattr(params, '_shroud_latch', None)
    return bool(lat[1]) if lat is not None else False


def _stage_chain_mass(params: BoosterParams, t: float, alt_m: float = 0.0) -> float:
    """Mass of the stage chain only (excludes strap-on boosters)."""
    if t <= 0:
        return params.mass_initial
    t_rem, s = t, params
    while s is not None:
        if t_rem < s.burn_time_s:
            mdot = s.mass_propellant / s.burn_time_s
            # Propellant is consumed only up to the commanded cutoff; after that
            # the unburned propellant rides on as dead mass until jettison.
            mass = s.mass_initial - mdot * min(t_rem, _eff_burn(s))
            if params.shroud_mass_kg > 0 and _shroud_jettisoned(params, alt_m):
                mass -= params.shroud_mass_kg
            return mass
        t_rem -= s.burn_time_s
        if s.stage2 is None:
            return params.payload_kg if params.payload_kg > 0 else s.mass_final
        if t_rem < s.coast_time_s:
            return s.stage2.mass_initial
        t_rem -= s.coast_time_s
        s = s.stage2
    return params.mass_final


def booster_mass(params: BoosterParams, t: float, alt_m: float = 0.0) -> float:
    """Current mass (kg) at time t seconds after launch.  Handles N stages and
    strap-on boosters.

    alt_m is the current altitude in metres; used for shroud-jettison accounting.
    When booster_core_delay_s > 0 the stage chain hasn't started burning until
    t >= delay, so we shift the time seen by _stage_chain_mass.
    """
    t_chain = t - params.booster_core_delay_s
    return (_stage_chain_mass(params, max(0.0, t_chain), alt_m)
            + _booster_mass_addend(params, max(0.0, t)))


def _boost_front_geometry(top_params: 'BoosterParams', params: BoosterParams,
                          altitude_m: float = None):
    """Front-end geometry exposed to the airstream during powered flight.

    Returns (nose_shape, diameter_m, nose_length_m, body_length_m, is_shroud).

    During boost the drag is set by whatever caps the front of the stack:
    the payload shroud/fairing while it is still attached, and the RV/payload
    nose once the shroud is gone.  The SAME body supplies both the reference
    area and the nose-shape Cd, so the two can never disagree (previously the
    area read the inline `ro_diameter_m` while the Cd read the ROParams
    diameter, which could differ).

    The front end sets the nose SHAPE (Cd); the reference DIAMETER is the
    widest body actually flying — normally the stage body, since an RV/payload
    rides atop a booster that is as wide or wider.  Using max(nose, stage)
    keeps the area correct during lower-stage burn (the fat booster, not the
    little RV, is what plows through the dense air) and also covers the rare
    case of an RV flared wider than its final stage.  When the diameters are
    equal — the usual nose-caps-body case — the choice is moot.

    The "warhead does not separate" case is handled by effective_ro(): a
    body-mode RV (separation_mode='body') inherits the body's own diameter, so
    the front end is the full body width rather than a narrower notional RV.
    The RV nose is the front end whether or not it later separates —
    separation happens at/after burnout, so it does not affect boost geometry.

    Deliberately omitted: slim-forebody "shielding" (a slender RV/payload on a
    wider body creating a shock the base rides in, the way our aerospike model
    reduces a blunt body's wave drag).  It is real physics, but for a launch
    where the RV is shrouded through the dense atmosphere and only exposed at
    high altitude (low q), the effect was measured at ~0.01% of burnout speed
    for the Minotaur-IV + HTV-2 case — two-to-three orders of magnitude below
    ordinary propulsion (~3%) and glide-model (~3%) error.  Treating the
    widest diameter as an unshielded nose is therefore close enough; the
    shielding term would only matter for an unshrouded slim body exposed in
    dense air, a different vehicle class.
    """
    if (top_params is not None
            and top_params.shroud_diameter_m > 0
            and altitude_m is not None
            and not _shroud_jettisoned(top_params, altitude_m)):
        return (top_params.shroud_nose_shape, top_params.shroud_diameter_m,
                top_params.shroud_nose_length_m, top_params.shroud_length_m, True)
    ro = effective_ro(top_params) if top_params is not None else None
    if ro is not None and ro.diameter_m > 0:
        return (ro.shape, max(ro.diameter_m, params.diameter_m),
                ro.length_m, params.length_m, False)
    if top_params is not None and top_params.payload_diameter_m > 0:
        return (top_params.nose_shape,
                max(top_params.payload_diameter_m, params.diameter_m),
                top_params.nose_length_m, params.length_m, False)
    return (params.nose_shape, params.diameter_m,
            params.nose_length_m, params.length_m, False)


def booster_area(params: BoosterParams, altitude_m: float = None,
                 top_params: 'BoosterParams' = None) -> float:
    """Reference cross-sectional area (m^2) of the exposed front end.

    Tracks the same front-end body as the nose-shape drag model: the shroud
    while attached, otherwise the RV/payload nose (see _boost_front_geometry).
    """
    _, d, _, _, _ = _boost_front_geometry(top_params, params, altitude_m)
    if d <= 0:
        d = params.diameter_m
    return np.pi * (d / 2) ** 2


def drag_coefficient(params: BoosterParams, mach: float) -> float:
    """Cd interpolated from Mach table; falls back to Forden table if empty."""
    if params.mach_table:
        return float(np.interp(mach, params.mach_table, params.cd_table))
    return float(_lin_interp(mach, _FORDEN_MACH, _FORDEN_CD))


def _ro_nose_shape(p: BoosterParams) -> str:
    """RV nose shape: from params.ro when available, else deprecated inline field."""
    ro = effective_ro(p)
    return ro.shape if ro is not None else getattr(p, 'ro_shape', '')


def _ro_diameter(p: BoosterParams) -> float:
    ro = effective_ro(p)
    return ro.diameter_m if ro is not None else getattr(p, 'ro_diameter_m', 0.0)


def _ro_length(p: BoosterParams) -> float:
    ro = effective_ro(p)
    return ro.length_m if ro is not None else getattr(p, 'ro_length_m', 0.0)


def drag_force_vector(params: BoosterParams, vel_ecef, altitude_m,
                      top_params: 'BoosterParams' = None,
                      t_s: float = None) -> np.ndarray:
    """
    Aerodynamic drag force vector (N) opposing velocity.

    Parameters
    ----------
    params     : BoosterParams (current stage)
    vel_ecef   : velocity vector in ECEF (m/s), shape (3,)
    altitude_m : scalar altitude (m)
    top_params : top-level BoosterParams (for shroud diameter lookup); optional
    t_s        : mission time (s); used for the grid-fin deployment schedule.
                 If None, all grid fins are treated as deployed.

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
    # The front-end body (shroud while attached, else RV/payload nose) sets
    # both the nose shape and the reference diameter — the same selector used
    # by booster_area() — so Cd and reference area always agree.
    _shape, _diam, _nose_len, _body_len, _is_shroud = _boost_front_geometry(
        top_params, params, altitude_m)
    if top_params is not None and _shape not in ('', 'forden') and _diam > 0:
        _ld = (_nose_len / _diam if _nose_len > 0 and _diam > 0 else 3.0)
        _ld_body = (_body_len / _diam if _body_len > 0 and _diam > 0 else None)
        if _is_shroud:
            cd = _cd_nose_shape(_shape, _ld, mach, re_l=re_l, ld_body=_ld_body,
                                aerospike_LD=top_params.aerospike_LD,
                                aerospike_dD=top_params.aerospike_dD)
        else:
            # Aerospike is attached to the shroud, so it stops working once
            # the shroud is jettisoned — no aerospike effect on this branch.
            cd = _cd_nose_shape(_shape, _ld, mach, re_l=re_l, ld_body=_ld_body)
    else:
        cd = drag_coefficient(params, mach)

    area = booster_area(params, altitude_m=altitude_m, top_params=top_params)
    q    = 0.5 * rho * speed**2
    drag_mag = cd * q * area

    # Grid (lattice) fins on the active stage — add their drag increment.
    # _cd_gridfins is referenced to the body base area π(d/2)², so multiply by
    # that same reference area (not the front-end `area`, which may be the
    # shroud/RV frontal area).
    if getattr(params, 'has_grid_fins', False) and params.n_grid_fins > 0:
        n_dep = grid_fins_deployed(
            params.n_grid_fins,
            getattr(params, 'grid_fin_deploy_schedule', None), t_s)
        if n_dep > 0:
            re_c = rho * speed * params.grid_fin_chord_m / mu if mu > 0.0 else 5e6
            cd_gf = _cd_gridfins(
                n_dep, params.grid_fin_width_m,
                params.grid_fin_height_m, params.grid_fin_chord_m,
                params.grid_fin_web_thickness_m, params.grid_fin_cell_pitch_m,
                params.diameter_m, mach, re_chord=re_c,
                edge_factor=getattr(params, 'grid_fin_edge_factor', 1.0),
                solidity=getattr(params, 'grid_fin_solidity', 0.0))
            a_base = np.pi * (params.diameter_m / 2.0) ** 2
            drag_mag += cd_gf * q * a_base

    # Planar fins on the active stage — add their drag increment.  Booster fins
    # (e.g. a finned first stage) plow through the dense lower atmosphere during
    # ascent and their drag matters for range; they jettison with their stage,
    # so gating on the active stage's has_fins is correct.  No lift is added:
    # an ascending vehicle flies at ~0 deg AoA, so the fins' normal force is a
    # stability effect (static margin), not a trajectory force.  _cd_fins is
    # referenced to the body base area, like the grid-fin term above.
    if getattr(params, 'has_fins', False) and params.n_fins > 0 \
            and params.fin_span_m > 0 and params.fin_root_chord_m > 0:
        cd_pf = _cd_fins(
            params.n_fins, params.fin_span_m, params.fin_root_chord_m,
            params.fin_tip_chord_m, params.fin_thickness_m, params.diameter_m,
            mach, re_l=re_l, sweep_deg=params.fin_sweep_deg)
        a_base = np.pi * (params.diameter_m / 2.0) ** 2
        drag_mag += cd_pf * q * a_base

    return -drag_mag * (vel_ecef / speed)


def _stage_chain_thrust(params: BoosterParams, t: float, altitude_m: float,
                        thrust_dir: np.ndarray) -> np.ndarray:
    """Thrust from the stage chain only (excludes strap-on boosters)."""
    if t < 0:
        return np.zeros(3)
    t_rem, s = t, params
    while s is not None:
        if t_rem <= s.burn_time_s:
            if t_rem > _eff_burn(s):
                return np.zeros(3)   # engine cut early — dead-coasting the tail
            _, P_amb, _, _ = atmosphere(altitude_m)
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
            return np.zeros(3)
        t_rem -= s.coast_time_s
        s      = s.stage2
    return np.zeros(3)


def thrust_force(params: BoosterParams, t: float, altitude_m: float,
                 thrust_dir: np.ndarray) -> np.ndarray:
    """
    Thrust force vector (N).  Handles N stages and strap-on boosters.

    Parameters
    ----------
    params     : BoosterParams (stage-1 node of the linked list)
    t          : time since launch (s)
    altitude_m : current altitude for ambient pressure correction
    thrust_dir : unit vector in direction of thrust (ECEF)

    Returns
    -------
    F_thrust : ndarray (3,) Newtons
    """
    # Stage chain only fires after booster_core_delay_s has elapsed.
    t_chain = t - params.booster_core_delay_s
    f = (_stage_chain_thrust(params, t_chain, altitude_m, thrust_dir)
         if t_chain >= 0 else np.zeros(3))

    # Add strap-on booster thrust while boosters are burning
    n, t_b = params.n_boosters, params.booster_burn_time_s
    if n > 0 and t_b > 0 and 0.0 <= t <= t_b:
        _, P_amb, _, _ = atmosphere(altitude_m)
        T_vac = params.booster_thrust_n
        if params.booster_nozzle_area_m2 > 0:
            T_mag = max(0.0, T_vac - P_amb * params.booster_nozzle_area_m2)
        else:
            T_mag = T_vac * (1.0 - 0.02 * (P_amb / 101325.0))
        f = f + n * T_mag * thrust_dir

    return f


def booster_drag_vector(top_params: BoosterParams, vel_ecef: np.ndarray,
                        altitude_m: float) -> np.ndarray:
    """
    Aerodynamic drag force (N) from the strap-on booster pack.

    Returns a zero vector when n_boosters == 0, booster_diam_m == 0, or
    speed is negligible.  Callers must gate by time: only invoke while
    t <= top_params.booster_burn_time_s.
    """
    n = top_params.n_boosters
    d = top_params.booster_diam_m
    if n <= 0 or d <= 0:
        return np.zeros(3)
    speed = np.linalg.norm(vel_ecef)
    if speed < 1e-6:
        return np.zeros(3)
    _, _, rho, _ = atmosphere(altitude_m)
    q        = 0.5 * rho * speed ** 2
    A_total  = n * np.pi * (d / 2.0) ** 2
    drag_mag = top_params.booster_cd * q * A_total
    return -drag_mag * (vel_ecef / speed)


# ---------------------------------------------------------------------------
# Booster library — shipped boosters live as data files (booster_library/
# *.booster.json), the same pattern as the reentry-object library.  The files
# are loaded here and overlaid onto BOOSTER_DB; the builder functions above
# remain as a fallback if a file is missing or fails to load.  Writers emit
# the new file form; the migration keeps builders authoritative until the
# file path is proven equivalent.
# ---------------------------------------------------------------------------
import glob as _glob
import json as _json
from pathlib import Path as _Path

_BUNDLED_BOOSTER_LIB = _Path(__file__).resolve().parent / "booster_library"


def load_booster_library(extra_dirs=()) -> int:
    """Overlay booster_library/*.booster.json onto BOOSTER_DB.

    Bundled files load first; any extra dirs (e.g. the user's writable
    library) override same-name entries.  Returns the number of files loaded.
    Never raises: a bad file is logged and skipped so the app still starts.
    """
    dirs = [_BUNDLED_BOOSTER_LIB, *[_Path(d) for d in extra_dirs]]
    n = 0
    for d in dirs:
        if not d.exists():
            continue
        for fp in sorted(d.glob("*.booster.json")):
            try:
                p = booster_from_dict(_json.loads(fp.read_text()))
                key = p.name or fp.name.replace(".booster.json", "")
                BOOSTER_DB[key] = (lambda _p=p: _p)
                n += 1
            except Exception as exc:   # pragma: no cover
                print(f"Warning: could not load booster '{fp.name}': {exc}")
    return n


load_booster_library()


# ---------------------------------------------------------------------------
# Flight plans — the "how it's flown" half of a booster,
# kept separate from the "what it is" hardware.  A flight plan carries the
# guidance mode, burnout/launch angles, and the per-stage pitch / yaw / coast /
# cutoff schedule.  Shipped as flight_plans/*.flightplan.json, applied onto a
# booster at run time.  Booster files stay hardware-only.
# ---------------------------------------------------------------------------
_FLIGHT_PLAN_TOP_KEYS = ('guidance', 'burnout_angle_deg', 'loft_angle_rate_deg_s',
                      'launch_elevation_deg',
                      # Subsystem-deployment timing read from the root booster:
                      # when the payload shroud is jettisoned (altitude, or <=0
                      # for the heating-flux default) and when spent strap-on
                      # boosters separate.  Both are flight decisions, not
                      # hardware, so they live in the flight plan.
                      'shroud_jettison_alt_km', 'booster_jettison_s')
_FLIGHT_PLAN_STAGE_KEYS = ('stage_turn_start_s', 'stage_turn_stop_s',
                        'stage_burnout_angle_deg', 'coast_time_s', 'stage_cutoff_s',
                        'stage_yaw_start_s', 'stage_yaw_stop_s', 'stage_yaw_final_az_deg',
                        # Grid-fin deployment schedule is read per active stage
                        # (drag_force_vector receives the current stage), so it is
                        # a per-stage flight-plan field.
                        'grid_fin_deploy_schedule')


def extract_flight_plan(p: BoosterParams) -> dict:
    """Pull the flight plan (guidance) out of a booster into a plain dict."""
    stages = []
    s = p
    while s is not None:
        stages.append({k: getattr(s, k) for k in _FLIGHT_PLAN_STAGE_KEYS})
        s = s.stage2
    fp = {k: getattr(p, k) for k in _FLIGHT_PLAN_TOP_KEYS}
    fp['stages'] = stages
    return fp


def apply_flight_plan(p: BoosterParams, fp: dict) -> BoosterParams:
    """Return a copy of booster `p` with flight plan `fp` stamped onto it.

    Top-level guidance fields and the per-stage schedule are set from `fp`;
    hardware is untouched.  Missing keys leave the booster's own value in place.
    """
    import copy as _copy
    q = _copy.deepcopy(p)
    for k in _FLIGHT_PLAN_TOP_KEYS:
        if k in fp:
            setattr(q, k, fp[k])
    node = q
    for st in fp.get('stages', []):
        if node is None:
            break
        for k in _FLIGHT_PLAN_STAGE_KEYS:
            if k in st:
                setattr(node, k, st[k])
        node = node.stage2
    return q


_BUNDLED_FLIGHT_PLANS = _Path(__file__).resolve().parent / "flight_plans"

# Extra directories (highest precedence last) searched for user flight plans,
# e.g. the GUI's writable ~/Documents/Thrusty/flight_plans.  A user file is
# merged *over* the bundled plan per key, so a user booster save can override
# just the fields it owns (subsystem-deployment timing) without discarding the
# shipped guidance.  Empty by default so headless/library use only sees the
# bundled plans.
USER_FLIGHT_PLAN_DIRS: list = []


def _merge_flight_plans(base: dict, over: dict) -> dict:
    """Overlay flight plan ``over`` onto ``base``, per key and per stage."""
    out = {k: v for k, v in base.items() if k != 'stages'}
    for k, v in over.items():
        if k != 'stages':
            out[k] = v
    stages = [dict(s) for s in base.get('stages', [])]
    for i, ost in enumerate(over.get('stages', [])):
        if i < len(stages):
            stages[i].update(ost)
        else:
            stages.append(dict(ost))
    if stages or 'stages' in base or 'stages' in over:
        out['stages'] = stages
    return out


# Sentinel label for the booster-named default plan (undeletable; the file
# named after the booster itself).
DEFAULT_PLAN_LABEL = "(default)"

# Active named flight plan per booster, set by the GUI (booster name -> plan
# name).  Consulted by get_booster when no explicit plan is passed; headless
# callers that never populate it always get the default plan.
ACTIVE_FLIGHT_PLANS: dict = {}


def load_flight_plan(name: str, extra_dirs=(), plan: str = None):
    """Load a booster's flight plan; None if none found.

    With ``plan=None``, loads the (default) plan: bundled directory first, then
    each of ``extra_dirs`` in order, merging each file found over the
    accumulated plan (later dirs win) — a user file need only carry the fields
    it overrides.  With a ``plan`` name, loads that named variant the same way.
    """
    from pathlib import Path as _P
    fname = flight_plan_filename(name, plan)
    result = None
    for d in [_BUNDLED_FLIGHT_PLANS, *[_P(x) for x in extra_dirs]]:
        fp = d / fname
        if fp.exists():
            try:
                data = _json.loads(fp.read_text())
            except Exception:
                continue
            result = data if result is None else _merge_flight_plans(result, data)
    return result


def flight_plan_filename(name: str, plan: str = None) -> str:
    """Canonical flight-plan filename: booster-named for the default plan,
    ``<booster>__<plan>.flightplan.json`` for a named variant."""
    if plan and plan != DEFAULT_PLAN_LABEL:
        return f"{_re_safe(name)}__{_re_safe(plan)}.flightplan.json"
    return f"{_re_safe(name)}.flightplan.json"


def list_flight_plans(name: str, extra_dirs=()) -> list:
    """Names of the flight plans available for booster ``name``.

    Always starts with DEFAULT_PLAN_LABEL; named variants follow in sorted
    order, discovered in the bundled dir and ``extra_dirs`` by the
    ``<booster>__<plan>.flightplan.json`` naming convention.
    """
    from pathlib import Path as _P
    prefix = f"{_re_safe(name)}__"
    suffix = ".flightplan.json"
    plans = set()
    for d in [_BUNDLED_FLIGHT_PLANS, *[_P(x) for x in extra_dirs]]:
        if not d.exists():
            continue
        for fp in d.glob(f"{prefix}*{suffix}"):
            token = fp.name[len(prefix):-len(suffix)]
            if not token:
                continue
            try:
                data = _json.loads(fp.read_text())
                label = str(data.get('name', '')) or token
            except Exception:
                label = token
            plans.add(label)
    return [DEFAULT_PLAN_LABEL, *sorted(plans)]


def save_flight_plan(name: str, fp: dict, out_dir, plan: str = None) -> str:
    """Write flight plan ``fp`` for booster ``name`` into ``out_dir``.

    ``plan`` names a variant (stamped into the file as ``name``/``booster``
    metadata so the file is self-describing); None writes the default plan.
    Returns the path.  ``fp`` is typically from :func:`extract_flight_plan`.
    """
    from pathlib import Path as _P
    d = _P(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if plan and plan != DEFAULT_PLAN_LABEL:
        fp = {**fp, 'name': plan, 'booster': name}
    path = d / flight_plan_filename(name, plan)
    path.write_text(_json.dumps(fp, indent=2) + "\n")
    return str(path)


def _re_safe(s: str, maxlen: int = 60) -> str:
    import re as _re
    s = _re.sub(r'\s+', '_', (s or '').strip())
    s = _re.sub(r'[^\w\-]', '-', s)
    return s[:maxlen] or 'booster'


# ---------------------------------------------------------------------------
# Reentry plans — the "how it's flown" half of a reentry object, the down-leg
# analogue of the flight plan.  A reentry object owns *what it is* (shape, beta,
# mass, TPS, and its max L/D capability); the reentry plan owns *how it flies the
# reentry* — glide mode, commanded L/D (capped at capability), turns, pull-ups,
# dives, and whether it separates.  Shipped as reentry_plans/*.reentryplan.json
# and applied onto a reentry object at run time; reentry-object files stay
# hardware-only.  The schema is deliberately open (schedules are list-valued) so
# cross-range S-turns, multi-phase pull-ups, and waypoints can be added as new
# keys without a format change; apply_reentry_plan ignores keys it does not
# know.  Reserved for later (no format change needed when they land):
#   'deployment' — post-boost bus/PBV dispense: a separation *time* (today
#                  separation is pinned to last-stage burnout) and, eventually,
#                  per-object aimpoints so the bus can deploy multiple reentry
#                  objects rather than carry one as dead mass.
_REENTRY_PLAN_KEYS = (
    'glider_enabled', 'glider_guidance', 'glider_pullup_g_max',
    'glider_terminal_dive', 'glider_terminal_alt_km', 'glider_bank_schedule',
    'glider_dive_target_lat_deg', 'glider_dive_target_lon_deg',
    'glider_dive_target_radius_km', 'glider_beta_entry_kg_m2',
    'glider_skip_count', 'glider_damping_zeta', 'glider_flap_deflection_deg',
    'separation_mode',
)


def extract_reentry_plan(ro: ROParams) -> dict:
    """Pull the reentry plan out of a reentry object into a plain dict.

    ``commanded_LD`` defaults to the vehicle's full L/D capability, so applying
    the extracted plan is a no-op; lowering it later flies the vehicle worse.
    """
    rp = {k: getattr(ro, k) for k in _REENTRY_PLAN_KEYS}
    rp['commanded_LD'] = ro.glider_LD
    return rp


def apply_reentry_plan(ro: ROParams, rp: dict) -> ROParams:
    """Return a copy of reentry object ``ro`` with reentry plan ``rp`` applied.

    Plan fields are stamped onto the copy; hardware (shape/beta/mass/TPS and the
    ``glider_LD`` capability) is untouched.  The commanded L/D is clamped to the
    vehicle's capability so a plan can only fly it worse, never beyond its
    aerodynamic limit.  Missing keys leave the object's own value in place.
    """
    import copy as _copy
    q = _copy.deepcopy(ro)
    for k in _REENTRY_PLAN_KEYS:
        if k in rp:
            setattr(q, k, rp[k])
    # Plan files bypass ro_from_dict, so legacy separation tokens are
    # normalised here too ('non_separating' -> 'body', etc.).
    q.separation_mode = _norm_sep_mode(q.separation_mode)
    cmd = rp.get('commanded_LD')
    if cmd is not None:
        q.glider_LD = min(float(cmd), ro.glider_LD)  # fly it worse, never better
    return q


_BUNDLED_REENTRY_PLANS = _Path(__file__).resolve().parent / "reentry_plans"
# Extra dirs (highest precedence last) for user reentry plans; merged over the
# bundled plan per key, mirroring USER_FLIGHT_PLAN_DIRS.
USER_REENTRY_PLAN_DIRS: list = []


def reentry_plan_filename(name: str) -> str:
    """Canonical reentry-plan filename for a reentry-object name."""
    return f"{_re_safe(name)}.reentryplan.json"


def load_reentry_plan(name: str, extra_dirs=()):
    """Load a reentry object's plan by name; None if none found.

    Bundled dir first, then each of ``extra_dirs`` merged over it (later wins),
    so a user plan need only carry the fields it overrides.
    """
    from pathlib import Path as _P
    safe = _re_safe(name)
    result = None
    for d in [_BUNDLED_REENTRY_PLANS, *[_P(x) for x in extra_dirs]]:
        fp = d / f"{safe}.reentryplan.json"
        if fp.exists():
            try:
                data = _json.loads(fp.read_text())
            except Exception:
                continue
            result = data if result is None else {**result, **data}
    return result


def save_reentry_plan(name: str, rp: dict, out_dir) -> str:
    """Write reentry plan ``rp`` for ``name`` into ``out_dir``; return the path."""
    from pathlib import Path as _P
    d = _P(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / reentry_plan_filename(name)
    path.write_text(_json.dumps(rp, indent=2) + "\n")
    return str(path)
