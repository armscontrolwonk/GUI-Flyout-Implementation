# Thrusty — Ballistic Missile / SLV Trajectory Simulator

Thrusty is a 3-DOF trajectory simulator and analysis tool for ballistic missiles
and space launch vehicles, implemented as a Python/Tkinter desktop application.
It is modelled after Geoffrey Forden's open-source MATLAB tool
(*Simulating the Operation of Ballistic Missiles*, Science & Global Security, 2007)
and validated against Forden's Table 3 maximum-range figures for the Scud-B,
Al Hussein, No-dong, and Taepodong-I.

---

## Source files

| File | Lines | Purpose |
|---|---|---|
| `thrusty.py` | ~4 600 | GUI application — all Tkinter widgets, dialogs, plotting, export |
| `trajectory.py` | ~1 750 | 3-DOF integrator, guidance laws, range optimiser, orbital planner |
| `missile_models.py` | ~1 060 | `MissileParams` dataclass, drag, thrust, mass, staging logic |
| `coordinates.py` | ~190 | WGS-84 coordinate conversions, Vincenty geodesic, Coriolis/centrifugal |
| `atmosphere.py` | ~97 | COESA 1976 standard atmosphere (0–86 km), dynamic pressure |
| `gravity.py` | ~62 | WGS-84 J2 gravity vector in ECEF |
| `slv_performance.py` | ~287 | Algebraic SLV payload-to-orbit estimation (Schilling/Townsend) |

---

## Quick start

```
pip install -r requirements.txt   # numpy, scipy, matplotlib, folium
python thrusty.py
```

User data is stored in `~/.gui_missile_flyout/`:

| File | Contents |
|---|---|
| `custom_missiles.json` | User-defined missile definitions |
| `custom_sites.json` | User-defined launch sites |
| `trajectory_profiles.json` | Per-missile guidance settings (loft angle, turn schedule, etc.) |

---

## User interface

The window is split into a scrollable **left control panel** and a **right
tabbed notebook**.

### Left control panel

- **Missile Type** — select from built-in (Forden) or user-defined missiles;
  New / Edit… / Delete buttons open `MissileDialog`.
- **Display Units** — km / nmi / miles for all plots and timeline distances.
- **Launch Site** — pick from a built-in list or define custom sites (lat/lon);
  azimuth is set manually (°, clockwise from North).
- **Guidance** — three modes (see below), with loft angle, pitch rate, turn
  start/stop, and optional advanced per-stage pitch and yaw programs.
- **Engine Cutoff** — optional early cutoff time (s); blank = full burn.
- **Target / Range** — optional target lat/lon or slant range for the
  *Aim at Target* function.
- **Re-entry Query Altitude** — altitude at which re-entry speed and angle are
  reported in the Flight Timeline.
- Action buttons: **Run**, **Maximize Range**, **Aim at Target**,
  **Parametric Sweep**, **Plan Orbit**.

### Right tabs

| Tab | Contents |
|---|---|
| **Plots** | Altitude-vs-range, altitude-vs-time, and speed-vs-time curves on a Matplotlib canvas |
| **Flight Timeline** | Tabular milestone events (ignition, burnout, apogee, shroud jettison, re-entry, impact) with lat/lon/alt/speed/range |
| **Missile Parameters** | Read-only summary of the active missile's mass, geometry, propulsion, and payload |
| **SLV Performance** | Algebraic payload-to-orbit analysis (circular or elliptical orbit) |

### Dialogs

- **MissileDialog** — define a missile with up to three stages plus payload/shroud/RV.
  Each stage has: fueled mass, dry mass, diameter, length, thrust (with Suggest
  estimator), Isp, nozzle exit area (with Estimate tool), burn time (computed),
  coast time, and a solid-motor flag.
  The Front End section covers payload mass, RV β (with Calculate… dialog using
  Newtonian hypersonic model), payload nose shape/length, and shroud parameters.
- **Parametric Sweep** — vary any one guidance parameter over a range and plot
  impact range vs. the swept variable.
- **β Calculator** — estimates RV ballistic coefficient from cone geometry
  (half-angle, nose bluntness ratio) using a bilinear interpolation of the
  Newtonian hypersonic Cd chart (Ref (4) Ch. 5).
- **Thrust Estimator** — back-calculates engine thrust from observed rocket
  acceleration: `T = m · √(a_h² + (a_v + g)²)`.

---

## Missile model (`MissileParams`)

A missile is a linked chain of `MissileParams` nodes (`stage2` pointer for
upper stages).  Key fields on the top-level node:

**Propulsion (per stage)**
- `mass_initial`, `mass_propellant`, `mass_final` (kg)
- `thrust_N` (vacuum, N), `isp_s` (s), `burn_time_s` (s)
- `nozzle_exit_area_m2` — enables proper ambient-pressure thrust correction
  `T(h) = T_vac − P_amb(h) · Ae`; zero falls back to a 2 % sea-level
  back-pressure approximation
- `coast_time_s` — inter-stage coast interval (s)
- `solid_motor` — if true the engine cannot be shut off early

**Geometry (per stage)**
- `diameter_m`, `length_m`
- `nose_shape` — one of `forden`, `v2`, `elliptical`, `conical`, `parabolic`,
  `tangent_ogive`, `sears_haack` (controls the FerencDV Cd model)
- `nose_length_m` — used to compute fineness ratio L/D for the nose model

**Shroud (top-level)**
- `shroud_mass_kg`, `shroud_jettison_alt_km` (default 80 km)
- `shroud_diameter_m`, `shroud_length_m`, `shroud_nose_shape`,
  `shroud_nose_length_m` — aerodynamics before jettison

**Payload / RV (top-level)**
- `payload_kg` — total payload mass carried to burnout
- `rv_beta_kg_m2` — RV ballistic coefficient β = m/(Cd·A) kg/m²; activates
  β-based drag for the post-burnout arc when > 0
- `rv_mass_kg`, `num_rvs`, `bus_mass_kg` — payload decomposition
- `rv_separates` — if true, the empty last-stage body follows a separate
  tumbling-cylinder debris arc

**Guidance (per stage, with global defaults)**
- `guidance`: `loft`, `gravity_turn`, or `orbital_insertion`
- `loft_angle_deg`, `loft_angle_rate_deg_s`
- Per-stage pitch/yaw overrides: `stage_turn_start_s`, `stage_turn_stop_s`,
  `stage_burnout_angle_deg`, `stage_yaw_*`

---

## Physics

### Reference frame

The state vector `[x, y, z, vx, vy, vz]` is in **ECEF** (Earth-Centred
Earth-Fixed), which rotates with the Earth.  Earth's rotation is fully
accounted for through Coriolis and centrifugal pseudo-forces; no explicit
rotation term is needed in the initial conditions.

Inertial (ECI-frame) speed is recovered when needed as
`v_eci = v_ecef + ω × r`, where `ω = [0, 0, Ω_Earth]`.

### Equations of motion (`_eom`, `trajectory.py:412`)

At each integration step:

```
ẍ = g_ecef(r)  +  a_drag  +  a_thrust  +  a_coriolis  +  a_centrifugal
```

- **Gravity**: WGS-84 J2 oblate-spheroid model (`gravity_ecef`, `gravity.py`).
- **Coriolis**: `−2 ω × v` (`coriolis_acceleration`, `coordinates.py`).
- **Centrifugal**: `−ω × (ω × r)` (`centrifugal_acceleration`, `coordinates.py`).
- **Integration**: `scipy.integrate.solve_ivp` with RK45 and event detection
  for ground impact, apogee, and milestone altitudes.

### Atmosphere

COESA 1976 standard atmosphere (`atmosphere.py`), seven layers from 0–86 km,
exact layer lapse rates and pressure integrals.  Clamped to 86 km for the
standard model.

For drag above 86 km and up to 120 km an exponential interpolation of a
tabulated NRLMSISE-00 density profile is used (solar flux F10.7 = 150,
conservative low-activity estimate).  **Above 120 km drag is zeroed** because
the atmosphere model becomes unreliable.

### Drag

Three regimes depending on flight phase:

| Phase | Drag model |
|---|---|
| **Boost** (motor burning, shroud attached) | Cd × A from Forden Mach-table, reference area uses shroud diameter |
| **Boost** (motor burning, shroud jettisoned) | Cd × A, reference area uses body diameter; nose-shape FerencDV model if not `forden` |
| **Coast / re-entry** (`rv_beta > 0`) | β ballistic coefficient: `F_drag = q · m_rv / β` |
| **Coast / re-entry** (`rv_beta = 0`) | Falls back to final-stage Mach-table Cd × A |

The shroud-jettison event fires on the first upward crossing of
`shroud_jettison_alt_km`.  At that point shroud mass is subtracted, the
reference area switches from shroud to body diameter, and the nose-shape
model switches from `shroud_nose_shape` to `nose_shape`.

The Forden Mach table (Figure 1, piecewise linear):
`Mach = [0.0, 0.85, 1.0, 1.2, 2.0, 4.5]`,
`Cd   = [0.20, 0.20, 0.27, 0.27, 0.20, 0.20]`.

### Guidance laws

**Forden Loft** (`loft`) — Forden Eq. 8.  The missile launches vertically
and pitches over at `loft_angle_rate_deg_s` until it reaches `loft_angle_deg`,
then holds.  Azimuth is fixed at launch.

**Gravity Turn** (`gravity_turn`) — The missile kicks off vertical at
`loft_angle_rate_deg_s` until reaching `loft_angle_deg` (the kick elevation),
then locks thrust to the velocity vector.  Appropriate for IRBM/ICBM.

**Orbital Insertion** (`orbital_insertion`) — The pitch program follows the
velocity vector after the kick (same as gravity turn) but engine cutoff is
commanded when the state vector reaches the target orbital energy, not at a
fixed time.  Solid stages burn to natural burnout.

All three modes support optional per-stage advanced pitch and yaw programs that
override the global schedule for a specific stage.

---

## Key algorithms

### Vincenty geodesic (`range_between`, `coordinates.py:122`)

Replaces Forden's spherical haversine with the Vincenty inverse formula on the
WGS-84 ellipsoid (~0.5 mm accuracy).  Falls back to haversine for near-antipodal
pairs where Vincenty does not converge.

### Wheelon optimal burnout angle (`_wheelon_gamma_opt`, `trajectory.py:1523`)

For a given burnout speed `v_bo` and altitude, the optimal elevation angle
that maximises range on a spherical Earth is:

```
Q       = v_bo² / (g_bo · r_bo)
γ_opt   = ½ · arccos( Q / (2 − Q) )
```

Used by `maximize_range` to narrow the coarse grid search to ±10° around
`γ_opt`, reducing evaluations by ~67%.

### Tsiolkovsky stack ΔV (`_tsiolkovsky_dv`, `trajectory.py:1512`)

Sums ideal vacuum ΔV across all stages: `Σ Isp_i · g₀ · ln(m0_i / mf_i)`.
Used to estimate burnout speed before the range-maximisation search.

### Range maximisation (`maximize_range`, `trajectory.py:1535`)

Two-phase parallel search:

1. **Coarse grid** — evaluate candidate (loft angle, pitch rate) pairs on a
   thread pool; search window is ±10° of the Wheelon optimum.
2. **Fine optimisation** — `scipy.optimize.minimize_scalar` (Brent) polishes
   the best coarse result for loft mode, or optimises independently for
   gravity-turn mode.

### Aim at target (`aim_missile`, `trajectory.py:1295`)

Binary search on engine cutoff time to minimise range error to the target
geodetic point (Vincenty distance).

### SLV algebraic estimator (`slv_performance.py`)

Schilling/Townsend method.  No integration required.  Computes the required
ΔV for a circular or elliptical orbit (vis-viva at perigee), applies an
empirical gravity/drag/steering-loss penalty derived from ascent time, and
solves for the maximum deliverable payload.  Accuracy ~260 m/s RMS in total
mission ΔV; ~10% payload error.

### Newtonian β calculator (`_cd_blunted_cone_newtonian`, `thrusty.py:114`)

Hypersonic Cd for a blunted cone.  For a sharp cone (ε = r_N/r_b = 0) the
exact Newtonian result is `Cd = 2·sin²θ`.  For blunted cones, bilinear
interpolation on a 4×6 table taken from the Ref (4) Ch. 5 chart
(θ = 10°–40°, ε = 0–1.0) is used, with the bluntness excess scaled onto
the exact Newtonian value for out-of-range angles.

---

## Outputs

| Output | How to produce |
|---|---|
| **Altitude / speed plots** | Runs automatically; displayed in the Plots tab |
| **Flight Timeline** | Tabular milestones in the Flight Timeline tab |
| **Missile Parameters** | Summary in the Missile Parameters tab |
| **Folium map** | File → Export Folium Map; produces an interactive HTML map with the ground track, milestone markers, debris arcs, and leader-line labels |
| **KML** | File → Export KML; opens in Google Earth |
| **Trajectory CSV** | File → Save Trajectory; time-series state vector |
| **Timeline CSV** | File → Export Timeline |
| **Missile JSON** | File → Export Missile Definition |

---

## Built-in missiles

The four Forden (2007) Table 1 reference missiles are included read-only:

| Missile | Class | Stages | Max range |
|---|---|---|---|
| Scud-B | SRBM | 1 | ~288 km |
| Al Hussein | SRBM | 1 | ~693 km |
| No-dong | MRBM | 1 | ~973 km |
| Taepodong-I | IRBM | 2 | ~2 349 km |

---

## Dependencies

```
numpy  >= 1.24
scipy  >= 1.10
matplotlib >= 3.7
folium >= 0.14
```

Standard library only otherwise (tkinter, json, pathlib, threading,
concurrent.futures, math).
