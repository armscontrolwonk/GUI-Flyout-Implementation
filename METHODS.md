# Thrusty — Methods

This document is the technical reference for Thrusty's models and algorithms.
It complements the in-repo [`README.md`](README.md) (overview, source-file
guide, quick-start). Each section gives the governing equation(s), a brief
description of the implementation, and citations to primary sources where one
exists. Code is referenced by file and symbol name (e.g. `trajectory.py:
integrate_trajectory`) rather than line number, so the references stay valid
as the source evolves.

Conventions:

- All lengths are metres, masses kilograms, times seconds, angles radians
  internally (degrees on the GUI surface).
- `g₀ = 9.80665 m/s²` is *standard gravity* (BIPM-defined nominal
  acceleration of free fall at Earth's surface), used as the
  definitional constant in propulsion expressions such as
  `T = I_sp · g₀ · ṁ`. This is distinct from the local gravitational
  acceleration `g(r)` from the J₂ field (Section 3), which varies
  with position.
- The Earth gravitational parameter `μ = GM = 3.986004418×10¹⁴ m³/s²` and
  equatorial radius `R_E = 6 378 137 m` are WGS-84 values.

> **Draft status note.** Specific equation, section, and figure numbers
> for Chin (1961), *Missile Configuration Design* appearing in Section
> 8 are taken from source-code docstrings; they have not been
> independently re-verified against the PDF in the current editing
> environment. Citations to Tracy & Wright 2020, Acton 2015, Schilling
> 2009, Sutton & Graves 1971, Forden 2007, Anderson 2006, and the NACA
> reports are verified to publication level. Readers preparing the
> document for publication should spot-check the Chin equation numbers
> before quoting.

---

## Table of contents

1. [Overview and modelling approach](#1-overview-and-modelling-approach)
2. [Coordinate systems and reference frames](#2-coordinate-systems-and-reference-frames)
3. [Gravity model](#3-gravity-model)
4. [Atmosphere](#4-atmosphere)
5. [Equations of motion](#5-equations-of-motion)
6. [Mass and staging](#6-mass-and-staging)
7. [Propulsion](#7-propulsion)
8. [Aerodynamics](#8-aerodynamics)
9. [Guidance laws](#9-guidance-laws)
10. [Range optimisation and targeting](#10-range-optimisation-and-targeting)
11. [SLV performance estimation](#11-slv-performance-estimation)
12. [Hypersonic glide vehicles](#12-hypersonic-glide-vehicles)
13. [Stagnation heating](#13-stagnation-heating)
14. [Outputs, events, and milestones](#14-outputs-events-and-milestones)
15. [Validation and built-in missile definitions](#15-validation-and-built-in-missile-definitions)
16. [References](#16-references)

---

## 1. Overview and modelling approach

Thrusty is a three-degree-of-freedom (3-DOF) trajectory simulator. The state
vector is the missile centre of mass position and velocity in an Earth-Centred
Earth-Fixed (ECEF) frame:

```
y(t) = [x, y, z, ẋ, ẏ, ż]ᵀ
```

The single ODE integrated by `integrate_trajectory` (`trajectory.py`) is

```
ẏ = [v,  g_ECEF(r) + a_drag + a_thrust + a_Coriolis + a_centrifugal]
```

with each term derived in the sections below.

The 3-DOF formulation captures translational dynamics only — attitude is not a
state variable but is *prescribed* by the guidance law (Section 9). This is
the standard approximation for preliminary missile and launch-vehicle analysis,
where pitch-program scheduling rather than full rigid-body dynamics determines
the trajectory shape.

Numerical integration uses `scipy.integrate.solve_ivp` with the explicit
adaptive Runge–Kutta 4(5) method (RK45) and event detection for ground impact,
apogee, milestone-altitude crossings, and orbital-energy cutoff. The original
Forden tool ([Forden 2007](#16-references)) used forward Euler; Thrusty's
adaptive RK45 is strictly more accurate for the same time-step budget, at the
cost of variable step size.

The design intent is *preliminary aerothermal and trajectory analysis* —
the kind of figure-of-merit work used in arms-control verification, threat
modelling, and reentry-vehicle scoping. It is **not** a 6-DOF flight dynamics
simulator and does not replace tools such as POST, OTIS, or ASTOS for vehicle
design.

### 1.1 Design principles

Several recurring choices give the code a particular character worth
flagging up front:

- **Mean-conditions defaults, extension hooks for accuracy.** Every
  modelling choice that has a reasonable "mean" or "conservative"
  value uses that value as the default, but exposes a configuration
  hook for users who need higher fidelity for a specific scenario.
  The atmosphere model (Section 4.1) uses NRLMSISE-00 at mean solar
  conditions by default, with `configure_atmosphere()` accepting real
  date / solar / geomagnetic inputs for cases tied to specific
  flights. Per-stage guidance overrides (Section 9.4) let advanced
  users replicate documented pitch programs without disturbing the
  simple case. The `glider_guidance` dropdown (Section 12) exposes
  four glide laws of increasing physical fidelity. The RV library
  (`rv_library/*.json`) provides a clean extension point for adding
  user-defined reentry vehicles, and the JSON missile-save format
  (`missile_to_dict` / `missile_from_dict`) gives users a way to
  share, version, and refine their own missile definitions. The
  pattern is consistent: keep the out-of-box case simple and
  reproducible, but never bake in an assumption that prevents a more
  careful user from improving it.

- **Closed-form first, table second, integration third.** Where a
  closed-form expression captures the physics (Chin's cone wave-drag
  formula, Tsiolkovsky's ΔV, vis-viva at perigee, Wheelon's optimal
  angle), it is preferred for both transparency and speed. Tabulated
  data (NACA TN 4201 wave drag, Chin Fig. 3-15 base pressure) is used
  where no closed form is available. Full numerical integration of
  the EOM is used only when the integrated trajectory itself is what
  is wanted.

- **Honest physics with flagged simplifications.** Where a
  simplification is made deliberately (power-off base drag during
  boost, no booster–core base-drag interaction, no rotational
  dynamics, no angle-of-attack-dependent Cd), this document flags it
  explicitly rather than presenting the formula as if it were
  rigorous. The intent is that a careful reader can see exactly what
  the tool is and is not modelling, and can extend it where needed.

---

## 2. Coordinate systems and reference frames

### 2.1 Frames in use

| Frame | Use |
|---|---|
| **ECEF** | Integration state vector; gravity, drag, thrust forces resolved here |
| **Geodetic (lat, lon, alt)** | All user-visible coordinates; launch sites; targets |
| **ENU (East–North–Up)** | Local frame for resolving the guidance thrust direction |

The ECEF frame co-rotates with the Earth at angular rate `ω = 7.2921150×10⁻⁵ rad/s`
about the polar axis. Choosing ECEF as the integration frame keeps launch
sites stationary, simplifies range calculations, and lets the user specify
azimuth as a local heading rather than an inertial direction. The cost is
that ECEF is non-inertial, so two fictitious accelerations — Coriolis
(velocity-dependent) and centrifugal (position-dependent) — must be
carried explicitly in the equations of motion (Section 5) to recover
Newton's second law.

### 2.2 Geodetic ↔ ECEF conversions

`coordinates.py:geodetic_to_ecef` and `coordinates.py:ecef_to_geodetic`
implement the WGS-84 ellipsoidal conversions. The forward conversion is
closed-form:

```
N(φ)  = R_E / √(1 − e² sin²φ)
x     = (N + h) cos φ cos λ
y     = (N + h) cos φ sin λ
z     = (N(1 − e²) + h) sin φ
```

where `φ` is geodetic latitude, `λ` longitude, `h` altitude above the
ellipsoid, `e² = 2f − f² = 6.694…×10⁻³` is the first eccentricity squared,
and `f = 1/298.257223563` is the WGS-84 flattening.

The inverse conversion uses Bowring's iterative method (10 fixed-point
iterations, converging to machine precision in 3–4 in practice).

### 2.3 Surface distance — Vincenty geodesic

`coordinates.py:range_between` implements the Vincenty inverse geodesic
formula on the WGS-84 ellipsoid. Vincenty achieves sub-millimetre accuracy
over arbitrary distances and replaces the spherical haversine formula used
by the original Forden tool, which has ~0.5 % error at long ranges. For
near-antipodal pairs where Vincenty's iteration does not converge (an
inherent limitation of the inverse formula), the code falls back to
haversine.

### 2.4 ENU local frame

For guidance computations the ECEF velocity is resolved into a local
East–North–Up (ENU) basis attached to the missile's current latitude and
longitude (`_enu_frame`, `trajectory.py`). The unit vectors are

```
ê_E = [−sin λ,            cos λ,            0       ]
ê_N = [−sin φ cos λ,     −sin φ sin λ,      cos φ   ]
ê_U = [ cos φ cos λ,      cos φ sin λ,      sin φ   ]
```

The thrust direction is computed in ENU according to the guidance law
(Section 9) and then transformed back to ECEF for the EOM integration.

---

## 3. Gravity model

Thrusty uses a WGS-84 gravity field truncated to the second zonal
harmonic `J₂` — that is, a spherical-harmonic expansion of the
geopotential keeping only the spherical and oblateness terms. The
acceleration in ECEF is computed component-wise
(`gravity.py:gravity_ecef`):

```
g_x = −(GM·x/r³) · [1 − (3J₂R_E²/2r²) · (5(z/r)² − 1)]
      − (GM · 3J₂R_E² / 2r⁵) · x · (1 − 5z²/r²)

g_y = −(GM·y/r³) · [1 − (3J₂R_E²/2r²) · (5(z/r)² − 1)]
      − (GM · 3J₂R_E² / 2r⁵) · y · (1 − 5z²/r²)

g_z = −(GM·z/r³) · [1 − (3J₂R_E²/2r²) · (5(z/r)² − 1)]
      − (GM · 3J₂R_E² / 2r⁵) · z · (3 − 5z²/r²)
```

The first term in each component is the spherical (point-mass at
Earth's centre) gravity with a J₂-dependent latitudinal correction; the
second term is the extra zonal contribution that gives the field a
component pointing toward the equatorial plane at non-equatorial
latitudes. Note that the z-component has a `(3 − 5z²/r²)` factor versus
`(1 − 5z²/r²)` for the x and y components — that asymmetry is what
makes the J₂ field oblate-symmetric rather than spherically symmetric.

The constants are the WGS-84 values (`gravity.py–13`):

| Symbol | Value | Description |
|---|---|---|
| `GM` | 3.986 004 418×10¹⁴ m³/s² | Earth gravitational parameter |
| `R_E` | 6 378 137 m | Equatorial radius |
| `J₂` | 1.082 626 68×10⁻³ | Second zonal harmonic |

The J₂ correction is roughly 0.1 % of the leading spherical term but is
significant at ICBM and SLV altitudes. A circular orbit at 400 km
experiences argument-of-perigee drift under J₂ that depends strongly on
inclination — from about +16°/day at the equator to zero at the
critical inclination of 63.4° to about −4°/day at polar orbits. For
the sub-orbital trajectories Thrusty is primarily used for, the J₂
correction shows up as a few-kilometre range bias relative to a
spherical Earth, larger for high-apogee lofted profiles.

Higher-order zonal harmonics (`J₃`, `J₄`, …) and the tesseral harmonics
(longitude-dependent terms) are not modelled. These matter for
orbit-tracking precision but not for the engineering questions Thrusty
answers.

---

## 4. Atmosphere

`atmosphere.py` provides a single function `atmosphere(altitude_m) → (T, P, ρ, a)`
returning temperature (K), pressure (Pa), density (kg/m³), and speed of
sound (m/s). Two models are selectable:

### 4.1 NRLMSISE-00 (default)

The default model is NRLMSISE-00 ([Picone, Hedin, Drob & Aikin
2002](#16-references)) accessed through the `pymsis` package. NRLMSISE-00
returns species-resolved number densities, total mass density, and
temperature for the upper atmosphere; for trajectory work the mass density
is what matters.

The model is queried once at module import for 0–1000 km at 500 m spacing,
producing a pre-computed lookup table with interpolation on `log ρ` and
`log P` to preserve the exponential character of upper-atmosphere variation.
Per-trajectory calls use this lookup at ~10 μs per scalar evaluation
(measured; vectorised batch calls amortise to far less per point), which
keeps the EOM evaluation cheap.

Default conditions (set in `_ATM_CONFIG`):

| Parameter | Default | Meaning |
|---|---|---|
| `f107` | 150 | Daily F10.7 solar flux index |
| `f107a` | 150 | 81-day average F10.7 |
| `ap` | 4 | Geomagnetic Ap index (quiet) |
| `doy` | 80 | Day of year (~vernal equinox) |
| `ut_sec` | 43 200 | Universal time (noon UT) |
| `lat_deg`, `lon_deg` | 0, 0 | Geographic location for the MSIS evaluation |

The user can override any of these via `configure_atmosphere(**kwargs)`,
which rebuilds the lookup table in ~100 ms. This is the path to model a real
launch date and site with measured solar indices.

#### Design rationale

**Implementation summary.** The NRLMSISE-00 table covers 0–1000 km at
500 m intervals — `alts_km = np.arange(0.0, 1000.5, 0.5)` — built at
import time from a single `pymsis.calculate()` call
(`atmosphere.py`). The COESA 1976 code (`_atmosphere_std1976`,
`atmosphere.py`) remains available both as an automatic fallback if
`pymsis` is not installed and as an explicit user choice via
`configure_atmosphere(model='std1976')`. The `atmosphere(altitude_m)`
function signature is identical in both modes, so no caller in
`trajectory.py`, `missile_models.py`, or `thrusty.py` needs to know
which model is active.

The lower atmosphere (0–86 km) is well-mixed and well-measured; US Std
Atm 1976 and NRLMSISE-00 agree to within ~10 % across most of that band
because both are anchored to the same tropospheric/stratospheric data.
The two models diverge meaningfully above ~200 km, where MSIS's
solar-cycle and latitudinal corrections start to matter. Measured
density differences at the mean-conditions defaults (computed
end-to-end against the actual implementation):

| Altitude | ρ_MSIS (kg/m³) | ρ_StdAtm (kg/m³) | % difference |
|---:|---:|---:|---:|
| 0 km | 1.16 | 1.23 | 5.4 % |
| 30 km | 1.78×10⁻² | 1.80×10⁻² | 1.1 % |
| 70 km | 8.32×10⁻⁵ | 7.42×10⁻⁵ | 12 % |
| 86 km | 6.22×10⁻⁶ | 5.64×10⁻⁶ | 10 % |
| 100 km | 6.21×10⁻⁷ | 5.60×10⁻⁷ | 11 % |
| 120 km | 2.02×10⁻⁸ | 2.22×10⁻⁸ | 9 % |
| 200 km | 2.88×10⁻¹⁰ | 2.54×10⁻¹⁰ | 13 % |
| 300 km | 3.15×10⁻¹¹ | 1.92×10⁻¹¹ | 64 % |
| 500 km | 1.31×10⁻¹² | 5.22×10⁻¹³ | 151 % |

Below 200 km the two models agree to 10–15 %, which is small compared
to typical parameter uncertainties on a missile's drag coefficient.
Above 200 km the divergence grows rapidly, but the absolute density is
small enough that drag is essentially negligible for the trajectory
regimes Thrusty addresses.

NRLMSISE-00 also captures variability that US Std Atm 1976 cannot —
the solar cycle (F10.7), geomagnetic activity (Ap), and diurnal,
seasonal, and latitudinal variation. Measured solar-cycle density
ratios (F10.7 = 70 vs F10.7 = 250 at the mean-conditions defaults
otherwise):

| Altitude | ρ(F10.7=250) / ρ(F10.7=70) |
|---:|---:|
| 100 km | 1.1× |
| 150 km | 1.4× |
| 200 km | 2.4× |
| 300 km | 6.6× |
| 500 km | 33× |

The amplification is geometric with altitude. For HGV glide trajectories
dwelling in the 60–100 km band the solar-cycle effect is small
(< 1.5×); for orbital-decay studies or for trajectories with sustained
flight above 200 km, MSIS at the actual mission conditions is the right
choice.

The "mean conditions" defaults are not arbitrary — each was chosen
deliberately to produce a *close-to-mean atmosphere*, so that switching
the default model from US Std Atm 1976 to MSIS would not silently
shift results:

| Parameter | Default | Why this value approximates "mean" |
|---|---|---|
| `f107` | 150 | Long-term mean of the 11-year solar cycle |
| `f107a` | 150 | Same — 81-day average set to the long-term mean |
| `ap` | 4 | Quiet geomagnetic state (near the historical median) |
| `doy` | 80 | Vernal equinox — geometrically between summer and winter solstices |
| `ut_sec` | 43 200 | Noon UT — a clean reference time |
| `lat`, `lon` | 0, 0 | Equator / prime meridian — a clean reference location |

Measured against a 5×4×5 grid sweep of (doy, UT, latitude) at the same
solar/Ap state, the default profile is within 5–10 % of the geometric
mean below 86 km (where ballistic-missile drag matters) and 11–22 %
above 100 km. The upper-altitude divergence comes from the noon-UT /
equator geometry sitting on the diurnal bulge peak rather than
averaging it out — a known feature of the choice, traded for the
simplicity of a fixed reference point.

The result is that scenarios run under the new MSIS default give
effectively the same answers as the prior US-Std-Atm default across
the drag-relevant atmospheric regime (within 10–15 % across 50 km –
200 km). Most users will never need to touch `configure_atmosphere()`
— it is there for the ~1-in-100 case of real flight data tied to a
specific date and time, or for solar-cycle sensitivity studies.

This is one example of the broader design pattern named in Section 1.1:
the default is a sensible mean-conditions reference, the configuration
hook is there for users who need to model a specific scenario more
accurately. A user analysing a real test launch with known date and
solar indices can drop those into `configure_atmosphere()` before
running the trajectory and recover the actual atmosphere the missile
flew through.

#### Expected impact on ballistic trajectories

For typical ballistic-missile and HGV scenarios the
atmosphere-model choice moves the answer by a small but non-zero
amount. The rough magnitudes recorded during development (engineering
estimates for an AUR/C-HGB-class case at β ≈ 15 000, apogee ≈ 200 km,
under the full range of solar variability):

| Quantity | Expected swing (solar min ↔ max) |
|---|---|
| Apogee | ±0.1 to ±1 km |
| Total ballistic range (3 000+ km flight) | ±1 to ±5 km |
| HGV glide range | ±1 – 3 % |

These are order-of-magnitude estimates, not measured values from a
calibrated test suite; the HGV glide-range swing is the largest of the
three because the glider dwells in the 60–100 km band where solar
variability accumulates over hundreds of seconds of flight time. For
ordinary ballistic-missile range/apogee studies the difference is
dwarfed by parameter uncertainties on mass, Isp, and loft angle.

### 4.2 US Standard Atmosphere 1976 (fallback)

If `pymsis` is unavailable, or if `configure_atmosphere(model='std1976')` is
called explicitly, the model falls back to the US Standard Atmosphere 1976
extended to 1000 km. The COESA layer base altitudes, temperatures, and lapse
rates are tabulated in `_LAYERS`:

| Layer base (m) | T (K) | Lapse rate (K/m) |
|---:|---:|---:|
| 0 | 288.15 | −0.0065 |
| 11 000 | 216.65 | 0 |
| 20 000 | 216.65 | +0.001 |
| 32 000 | 228.65 | +0.0028 |
| 47 000 | 270.65 | 0 |
| 51 000 | 270.65 | −0.0028 |
| 71 000 | 214.65 | −0.002 |
| 86 000 | 186.87 | 0 |

Pressure within each layer follows the analytic integration of hydrostatic
balance under constant lapse rate:

- Non-isothermal layer (`L ≠ 0`): `P = P_b · (T_b / T)^(g₀ / (R · L))`
- Isothermal layer (`L = 0`): `P = P_b · exp(−g₀ · Δh / (R · T_b))`

with `R = 287.052 87 J/(kg·K)` for dry air. Density is recovered from the
ideal gas law `ρ = P/(R T)`.

Above 86 km the fallback uses an exponential interpolation between the
US Std Atm 1976 Table I/II reference points at 86, 91, 100, 110, 120, 150,
200, 300, 500, and 1000 km.

### 4.3 Dynamic pressure and Mach number

Two convenience functions on top of the atmosphere model:

```
q       = ½ ρ V²                     (dynamic pressure, Pa)
M       = V / a                       (Mach number)
```

where `a = √(γ R T)` is the speed of sound and `γ = 1.4`.

### 4.4 Other atmosphere models

The atmosphere module is structured around a fixed signature
(`atmosphere(altitude_m) → (T, P, ρ, a)`) and a configuration dictionary.
Two test scripts in the repo, `mars_smoke_test.py` and `mars_smoke_test2.py`,
exercise the configuration interface with Martian parameters, but the Mars
atmosphere is not a documented capability of the production code path.
Treat the Mars files as exploratory tests rather than a supported feature.

---

## 5. Equations of motion

The full equation of motion in ECEF is

```
r̈ = g_ECEF(r) + a_thrust + a_drag − 2 ω × ṙ − ω × (ω × r)
```

(`_eom`, `trajectory.py`). The right-hand side gives acceleration
in m/s² directly; each of the five terms is an acceleration, not a
force. Below they are listed in the order they appear in the code.

### 5.1 Gravity

`g_ECEF(r)` from Section 3.

### 5.2 Thrust acceleration

```
a_thrust = T(h) · û_thrust(t) / m(t)
```

The thrust magnitude `T(h)` is altitude-corrected (Section 7.2); the unit
vector `û_thrust` is set by the guidance law (Section 9); and `m(t)` is the
instantaneous mass (Section 6).

### 5.3 Drag acceleration

```
a_drag = −½ ρ V² · C_D · A_ref / m · v̂
```

with `V = |ṙ|`, `v̂ = ṙ / V`, `C_D` and `A_ref` from Section 8. For
post-burnout flight with a ballistic-coefficient parameterisation
`β = m / (C_D · A_ref)` the drag magnitude reduces to `q · m / β`.

### 5.4 Coriolis

```
a_Coriolis = −2 ω × ṙ
```

with `ω = [0, 0, Ω_Earth]ᵀ` in the ECEF frame. For a 4 km/s velocity
this is up to `2 · Ω · |v| ≈ 0.58 m/s²` — small instantaneously but
cumulatively significant over long flight times. The Coriolis term
deflects long-range trajectories laterally — the well-known equatorward
or poleward drift depending on hemisphere and launch direction — and
contributes to the cross-range bias that targeting algorithms must
account for.

### 5.5 Centrifugal

```
a_centrifugal = −ω × (ω × r) = Ω² · [x, y, 0]ᵀ
```

At the equator at sea level this is 0.034 m/s², equivalent to ~0.35 % of
local gravity. It is included for self-consistency with the rotating frame
formulation.

### 5.6 Integration

`scipy.integrate.solve_ivp` with `method='RK45'`. Tolerances vary by path:
the full-fidelity ballistic integration uses relative tolerance `1e-8` and
absolute tolerance `1e-6`; the faster range-search and glide integrations
use looser tolerances (as coarse as `~1e-5`/`1e-2`), and the apogee
pre-finder coarser still. Adaptive step sizing concentrates evaluations
during the boost phase (where mass and thrust change rapidly) and stretches
them during coast.

Event functions used (passed to `solve_ivp(events=...)`):

| Event | Purpose |
|---|---|
| `_hit_ground` | Trajectory termination at altitude ≤ 0 |
| `_apogee_event` (sign change of `ṙ·r̂`) | Apogee detection |
| Milestone-altitude crossings | 100 km re-entry, shroud jettison, user-defined queries |
| `_glider_pierce_atmosphere` | HGV pull-up / equilibrium-glide handoff |

Orbital-energy engine cutoff is **not** a `solve_ivp` event — it is applied
inline inside `_eom` (the thrust is zeroed once the specific orbital energy
`ε = ½|v_ECI|² − μ/r` reaches the target `ε_target = −μ/(2 r_target)`, for a
liquid final stage only). Detected events become rows in the Flight Timeline
output (Section 14).

---

## 6. Mass and staging

A missile is represented by a linked-chain `MissileParams` dataclass
(`missile_models.py`). Each stage carries its own propulsive and
geometric parameters; the `.stage2` attribute points to the next stage.
Top-level fields (payload, shroud, RV) apply to the whole vehicle.

### 6.1 Mass schedule

For a stage with initial mass `m₀`, propellant mass `m_p`, and burn time `t_b`,
the instantaneous mass during burn is linear in time:

```
m(τ) = m₀ − (m_p / t_b) · τ              0 ≤ τ ≤ t_b
```

where `τ` is local stage burn time. The burnout mass is `m_f = m₀ − m_p`.

The full-vehicle mass `missile_mass(params, t)` (`missile_models.py`)
walks the stage chain to determine which stage is active and adds the mass
of all subsequent stages plus payload. After a stage burns out the spent
casing is jettisoned and its dry mass is removed from the chain.

### 6.2 Coast and inter-stage events

Each stage has an optional `coast_time_s` — a quiescent interval between
that stage's burnout and the next stage's ignition. During coast the engine
produces no thrust and drag plus gravity govern the trajectory. The
Flight Timeline records each stage's ignition, burnout, coast end, and any
shroud-jettison events.

### 6.3 Shroud jettison

The payload shroud (or fairing) is treated as a top-level mass that
contributes to drag area until it is jettisoned at the user-defined
altitude `shroud_jettison_alt_km` (default 80 km). The jettison event
fires on the first *upward* crossing of that altitude; after firing, the
shroud's mass is subtracted from the missile total and the reference
geometry for drag switches from the shroud envelope to the payload or
re-entry vehicle (Section 8.2).

### 6.4 Reentry vehicle separation

If `rv_separates = True` on the top-level node, the RV separates from the
final stage body at burnout. Post-burnout drag is then computed from the
RV's geometry and ballistic coefficient rather than from the spent final
stage. This matters for ICBMs and SLVs where the spent upper stage has
very different drag characteristics from the warhead or payload it deployed.

### 6.5 Strap-on boosters

A vehicle may carry up to nine strap-on boosters that fire in parallel
with stage 1 from t = 0, then separate at a configurable burn time.
These appear in operational launch vehicles (e.g. Delta-IV Heavy,
Ariane 5, H-IIA, ISRO PSLV) and in some ballistic-missile derivatives.

Booster parameters live on the *top-level* `MissileParams` node as 11
fields (defaults zero, so existing missile definitions are unaffected):

| Field | Meaning |
|---|---|
| `n_boosters` | Number of identical boosters (0–9) |
| `booster_thrust_n` | Vacuum thrust per booster (N) |
| `booster_burn_time_s` | Burn duration (s); separation occurs at this time |
| `booster_inert_kg` | Empty (dry) mass per booster (kg) |
| `booster_prop_kg` | Propellant mass per booster (kg) |
| `booster_isp_s` | Specific impulse (s) |
| `booster_nozzle_area_m2` | Nozzle exit area per booster (m²) |
| `booster_diam_m` | Outer diameter per booster (m) |
| `booster_length_m` | Length per booster (0 → defaults to 2 × diameter) |
| `booster_cd` | Zero-lift drag coefficient per booster (default 0.20 ≈ tangent ogive) |
| `booster_core_delay_s` | Seconds after T = 0 before stage-1 core ignites |

**Mass schedule.** Total vehicle mass at time `t` during the booster
burn is:

```
m(t) = m_core(t) + n_boosters · (booster_inert + booster_prop · (1 − t/t_booster))
```

At separation (`t > booster_burn_time_s`) the booster contribution to
vehicle mass drops to zero — the casings (now with empty propellant
tanks) detach from the vehicle (`_booster_mass_addend`,
`missile_models.py`). The spent casings then follow tumbling-cylinder
debris arcs separately (Section 14.3).

**Thrust addition.** During the booster burn window `t ∈ [0, t_booster]`,
the booster contribution adds to the stage-1 thrust:

```
T_total(t) = T_core(t) + n_boosters · T_booster(h)
```

with each `T_booster(h)` getting its own ambient-pressure correction
(Section 7.2). The core may be delayed up to `booster_core_delay_s`
after liftoff to model vehicles whose core engine ignites only after
the boosters lift the stack off the pad.

**Drag addition.** Booster aerodynamic drag is computed as a parallel
cluster of cylinders independent of the core (`booster_drag_vector`,
`missile_models.py`):

```
D_booster = n_boosters · C_D,booster · q · π · (d_booster / 2)²
```

opposing the velocity vector. The booster `C_D` is a single user-set
value covering wave + friction + base. The default `0.20` is appropriate
for a tangent-ogive booster (consistent with Section 8.2.2); the GUI
tooltip flags `0.10 – 1.00` as the realistic range.

**Simplification flagged.** The cluster of boosters around the rear of
the core would, in reality, change the core's base-pressure
distribution (the boosters partially fill the wake region where suction
develops). Thrusty does not model this interaction — both the core
base drag (Section 8.4) and the booster drag are computed
independently. The error is small relative to the boosters' direct
contribution at supersonic Mach but could matter at transonic flight
of a heavily-boostered vehicle.

**Separation event.** "Booster separation" is inserted chronologically
into the flight timeline at `t = booster_burn_time_s`. Spent booster
casings then follow tumbling-cylinder ballistic arcs (Section 14.3)
until impact, recorded as separate debris-impact rows in the timeline.

### 6.6 Stage dry-mass estimation (independent cross-check)

Thrusty integrates from each stage's **burnout mass**, supplied directly by the
user (Section 6.1). The dry-mass estimator (`mass_estimator.py`; full tables and
sources in [`MASS_ESTIMATOR.md`](MASS_ESTIMATOR.md)) is a separate design-time
tool that helps you judge whether a stated dry mass is plausible, using several
popular mass-estimating relationships (MERs). It never feeds the trajectory — it
is a sanity check on the number the trajectory trusts.

There is no single authoritative dry-mass equation: the published MERs are a
*range* fit to different datasets and eras, and they disagree — two reasonable
ones can differ by tens of percent on the same stage. The tool runs several side
by side and reports the spread, so their agreement (or disagreement) is itself
the signal for how trustworthy a stated dry mass is.

These relations fall into two complementary families:

| Family | Inputs needed | Best for |
|---|---|---|
| **Component-level** (Wilhite-school MERs) | geometry, propellant split, thrust | large LV stages; itemised "where is the mass" breakdown |
| **Aggregate** (whole-stage relations) | propellant mass (+ thrust) | quick sanity bound; small stages; solids |

The two families are themselves the cross-check: in the small / tactical-scale
regime the component build-up's fixed terms (engine `+59 kg`, avionics, wiring)
dominate and **over-predict**, so the aggregate ε estimate is the one to trust
there — an explicit "use family B to catch family A's failure mode." Solids and
liquids are split on physics grounds: a solid case is a *pressure-loaded* vessel
(chamber pressure × grain volume), whereas a liquid's tanks are *near-unloaded*,
sized by propellant volume.

#### 6.6.1 Liquid stages

The component path (`estimate_liquid_stage`) sums itemised MERs from D. L. Akin's
*Mass Estimating Relations* (UMD ENAE 791) — the SI compilation descending from
the Heineman / MacConochie–Klich / Glatt (WAATS) lineage, the same school as
Wilhite's SSDL relations, and independently corroborated against Rohrschneider /
SSDL and Gaspar (2014). Tanks, cryogenic insulation, pump-fed engines, thrust
structure, gimbals, fairing, avionics, and wiring each have their own relation;
only the **tanks** are scaled by `tank_material` (`material_tank_factor`:
Al 1.00, Al-Li 0.74, composite 0.45, steel 1.60) — engines, thrust structure and
avionics are treated as material-independent. Avionics is **one guidance package
per vehicle**, carried on the upper stage only and sized on vehicle gross
liftoff mass, not the stage it rides on.

> *Engine-MER decision.* Akin's lecture table lists 373 kg/engine, but his own
> formula gives ≈ 641 kg at the worked example — which matches the independent
> Zandbergen (2015) pump-fed-engine fit to ~5 %. The implementation follows the
> **formula** and treats the 373 kg slide value as an arithmetic slip;
> `test_mass_estimator.py` pins it against Zandbergen.

A **physics-based tank option** (`tank_structural_mass`, the GT-STRESS method of
Hutchinson & Olds) can replace the empirical tank MER: it sizes each tank as a
thin shell under the worse of a max-axial (burnout) and a max-q-α (liftoff, full
tank + lateral-g) load case, taking thickness as the max over ultimate, yield,
buckling, and minimum-gauge, times a `TANK_CORRELATION = 1.50` factor. Here
**material choice is physics, not a multiplier** (ρ, σ, E, gauge from
`MATERIALS`). Axial load comes from thrust, so no trajectory run is needed;
refining the loads from a flown trajectory is a flagged follow-on.

The aggregate path deliberately distinguishes *predictions* from *tautologies* —
a core philosophy of the tool:

- **Engine-mass-ratio** (`engine_mass_ratio_inert`, Shu et al. 2020) is the real
  predictive aggregate for any liquid: inert = `M_engine·(1 + 1/κ_E)` with
  `M_engine` predicted from thrust and `κ_E` (`_KAPPA_E_DEFAULT`: lower 0.25,
  upper 0.12) varying over a much narrower band than ε. This fills the
  non-hydrolox gap that an assumed ε could not.
- **Pietrobon** hydrolox stage-mass power law (`pietrobon_stage_mass`) is shown
  only for LOX/LH₂.
- An **assumed structural coefficient** (`inert_from_structural_coefficient`) is
  **opt-in only and explicitly not a prediction** — it merely restates an ε you
  supply, in kilograms, "carrying no information beyond the assumption itself,"
  and was deliberately demoted from a default estimate to an opt-in reporting
  lens.

#### 6.6.2 Solid stages

The headline solid estimate (`solid_stage_inert`) uses the whole-stage
regressions of Zandbergen (2026), fit to flown stages, because open-literature
*component* MERs for solids are sparse — nozzle and internal-insulation MERs are
**intentionally not invented**. A best-in-class cross-check (`source="lewis"`)
regresses the Northrop Grumman Propulsion Products Catalog (29 Orion / Castor /
GEM motors); being mature flight motors it runs ~10 % lighter — the lower edge of
the inert band. When length and diameter are both supplied a **slenderness (L/D)
correction** is added, with a *positive* exponent: a more slender motor carries
*more* inert per unit propellant, because case wall and insulation scale with
surface area, not enclosed volume. The L/D term expects the **motor** length
*including the nozzle* — trim only non-motor structure (interstages, skirts),
never the nozzle.

#### 6.6.3 The divergence report

When a stated dry mass is supplied, `divergence_report` leads with the
**structural coefficient** `ε = dry / (dry + propellant)` it implies (and
`λ = dry/propellant`), then lists each method's *estimated* ε beside the
percentage divergence and a verdict: **consistent** (≤ 15 %), **marginal**
(≤ 35 %), otherwise **optimistic** / **conservative**. Reporting in ε lets a
design be judged "in the units you think in," and — because tank material moves
the estimated ε — directly answers *is this dry mass plausible for this
material?* A large negative divergence means the stated structure is lighter than
any flown analogue (Peacekeeper-derived motors read "optimistic").

A hard physical bound backstops the report: the **Goldyn et al. (2025)
feasibility ceiling** `ε_max = 1/exp(Δv/(g₀·I_sp))` (`structural_index_ceiling`).
Above it the rocket equation drives propellant mass negative, so any stated or
estimated ε breaching the ceiling is flagged as impossible. This is the same
estimate-and-flag posture used throughout Thrusty (Section 1.1): give a
defensible default, surface the assumptions, and flag what is physically out of
bounds rather than silently accepting it.

> **Provenance note.** The estimator postdates the development-chat transcripts
> committed to this repo (`chat_transcript*.txt` end before `mass_estimator.py`
> was created), so its rationale is not in those logs; the authoritative record
> is the module's own git history and [`MASS_ESTIMATOR.md`](MASS_ESTIMATOR.md),
> on which this section is based.

---

## 7. Propulsion

### 7.1 Thrust from `I_sp`

The mass-flow rate, burn time, and vacuum thrust are linked by the
standard definition of specific impulse:

```
T_vac = I_sp · g₀ · ṁ                     (definition of I_sp, rearranged)
ṁ     = m_p / t_b                          (constant-mass-flow burn)
```

so any two of `T_vac`, `I_sp`, `t_b` determine the third given `m_p`.
(This is not the Tsiolkovsky rocket equation — that is the integrated
`Δv = I_sp · g₀ · ln(m₀ / m_f)` form used for stack-ΔV estimation in
Section 10.1.) `missile_models.py:_thrust_from_isp` is used by the GUI's
"Suggest" thrust estimator (Section 14 of the user guide) and by the
built-in missile definitions that specify `I_sp` and burn time rather
than thrust directly.

### 7.2 Ambient-pressure correction

Vacuum thrust overstates the thrust at lower altitudes because ambient
back-pressure works against the exhaust. The corrected thrust is

```
T(h) = T_vac − P_amb(h) · A_e
```

where `A_e` is the nozzle exit area and `P_amb(h)` comes from the
atmosphere model (Section 4). The nozzle-area parameter `nozzle_exit_area_m2`
on each stage enables this correction; if it is zero, a default 2 %
sea-level back-pressure approximation is used instead. The "Estimate"
button next to the nozzle-area field in the GUI uses an expansion-ratio
heuristic to pre-fill plausible values.

### 7.3 Solid motor grain profiles

For a solid motor the bulk thrust integral `∫T dt = I_sp g₀ m_p` is
constrained by total impulse, but the instantaneous thrust profile depends
on the grain geometry. Thrusty supports six grain types via a
`grain_type` field and an associated `thrust_peak_N` parameter (the peak
vacuum thrust). The `thrust_N` field continues to hold the *average*
vacuum thrust, and the ratio `thrust_N / thrust_peak_N` defines the
fill factor implied by the grain shape.

The grain types, after Shafer (1959), Ch. 16 in *Space Technology*. The
fill factor ranges are stored in the code as `_GRAIN_FILL_RANGE` and used
by the GUI to warn if a user-entered combination is unphysical for the
chosen grain:

| Grain type | Burn character | Realistic fill-factor range |
|---|---|---|
| Tubular | Progressive | 0.70 – 0.95 |
| Rod and tube | Neutral | 0.90 – 1.00 |
| Double anchor | Regressive | 0.60 – 0.85 |
| Star | Neutral | 0.85 – 1.00 |
| Multi-fin | Two-phase (boost-sustain) | 0.50 – 0.75 |
| Dual composition | Two-phase | 0.35 – 0.60 |

Each grain type also has a normalised piecewise-linear thrust-time
profile `_GRAIN_CURVES[grain]` giving `F/F_peak` at fractional burn
time `τ/t_b`. The instantaneous thrust fraction
`_instantaneous_thrust_frac(grain, τ_frac)` returns the thrust at
fractional burn time `τ/t_b ∈ [0, 1]` for the chosen grain by
linear interpolation on the appropriate curve. A user-supplied `thrust_profile` list of `(t_frac, F_frac)` pairs
overrides the built-in shape — this is the path for loading a CSV burn
curve from test data.

Note: for liquid stages the GUI does not expose grain selection because
liquid engines run at essentially constant thrust; the `grain_type = "liquid"`
sentinel disables the grain machinery and produces a flat thrust history.

> *Implementation rationale for the choice of this particular Shafer grain
> set is not recorded in available development history. The set covers
> the basic progressive / neutral / regressive characters plus two
> two-phase patterns, which span the range of behaviours seen in
> operational tactical and strategic solid motors.*

### 7.4 Solid motor cutoff

Liquid stages can be commanded to cut off early (the user-visible "Engine
Cutoff" field or, in orbital-insertion mode, the energy-cutoff event). Solid
stages cannot be shut off and burn to natural propellant exhaustion. The
`solid_motor` boolean on each stage gates this behaviour: the
orbital-insertion energy cutoff (applied inline in `_eom`) only fires for a
non-solid final stage, and the early-cutoff handling in the range/aim
solvers likewise applies to liquid stages — a `solid_motor = True` stage
always burns to natural propellant exhaustion.

This matters operationally because it changes the optimisation surface for
range maximisation (Section 10): a liquid-motor SCUD can trade burn time
for loft angle, but a solid-motor ICBM stage cannot.

### 7.5 Strap-on booster thrust

When `n_boosters > 0` on the top-level node (Section 6.5), the booster
contribution adds to the stage-1 thrust during the booster burn window:

```
T_total(t) = T_core(t) + n_boosters · T_booster(h)              for t ∈ [0, t_booster]
```

Each booster's vacuum thrust receives its own ambient-pressure correction
(7.2). After separation (`t > booster_burn_time_s`) only the core stage
contributes thrust. The core may optionally be delayed by
`booster_core_delay_s` to model liftoff on boosters alone before the
main engine ignites.

---

## 8. Aerodynamics

The boost-phase drag is computed by the **component build-up method**:
the total drag coefficient is assembled from independently-sourced wave,
friction, and base contributions, with optional boattail and aerospike
corrections, each evaluated against its own primary reference. This is
the USAF DATCOM-style preliminary-design approach.

```
C_D,total = C_D,wave(shape, M, l/d)
          + C_D,friction(M, Re, S_wet/S_ref)        × 1.10  (roughness)
          + C_D,base(M, S_base/S_ref)
          [+ C_D,boattail(M)]                       (if boattailed)
          [+ aerospike correction]                  (if spike present)
```

(`_cd_nose_shape`, `missile_models.py`). Each component has a
distinct, citable source: nose wave drag from
[Chin (1961)](#16-references) and the [NACA TN 4201](#16-references)
comparison report; friction from Chin's combination of Blasius
(laminar) and Schoenherr (turbulent) with the Frankl-Voishel
compressibility correction; base drag from Chin Fig. 3-15. This
replaces a single empirical `C_D(M)` lookup table used in earlier
versions and in the original Forden tool — that table is retained as a
fallback when no shape is specified (see Section 8.7).

For post-burnout flight the boost-phase build-up is replaced by a
ballistic-coefficient parameterisation (Section 8.8).

### 8.1 Nose shapes supported

The implemented nose-shape library is documented in
[`missile_models.py`](missile_models.py) under `NOSE_SHAPES` and
`NOSE_SHAPE_LABELS`:

| Key | Display name | Mathematical form | Wave-drag character |
|---|---|---|---|
| `cone` | Cone | Straight-line generator | Closed-form (Chin Eq. 3-4) |
| `tangent_ogive` | Tangent Ogive | Circular-arc generator tangent to body | Closed-form (Chin Eq. 3-9, Miles) |
| `von_karman` | Von Kármán (LD-Haack) | LD-Haack series, C = 0 | NACA RM tables |
| `lv_haack` | LV-Haack (Sears-Haack) | LD-Haack series, C = 1/3 | NACA RM tables |
| `parabola` | Parabola | y ∝ x − K·x², K = 1 | NACA RM tables |
| `blunt_cylinder` | Blunt Cylinder | Flat-faced | Sharp transonic rise, ~2.2 supersonic |

The five recommended shapes (cone through parabola) cover the range of
operational and developmental missile geometries; the blunt-cylinder option
is provided for testing and for unaerodynamic payloads. Elliptical and V-2
shapes available in some related tools are not implemented because no
properly-sourced wave-drag data was found for them with the same evidentiary
standard as the NACA reports.

### 8.2 Nose wave drag

#### 8.2.1 Cone — closed-form (Chin Eq. 3-4 / 3-6)

For a sharp cone with semi-vertex angle σ in *degrees*, valid for
attached flow (`M ≳ 1.2` for typical missile half-angles, `σ ≲ 50°`)
(`_chin_pressure_coeff`, `missile_models.py`):

```
Δp/q          = (0.083 + 0.096 / M²) · (σ° / 10)^1.69
C_D,wave,cone = Δp/q
```

The half-angle is derived from the user-specified fineness ratio
`l/d = nose_length / body_diameter` as
`σ = arctan(1 / (2 · l/d))`, then converted to degrees for the formula.

The code applies a transonic linear ramp to avoid the formula's
behaviour near the shock-attachment limit
(`_cd_wave_cone`, `missile_models.py`):

```
M ≤ 0.8:    C_D,wave,cone = 0
0.8 < M < 1.0:  C_D,wave,cone = (Δp/q at M=1.0) · (M − 0.8) / 0.2
M ≥ 1.0:    Use the formula directly
```

This smooths the subsonic-to-supersonic transition: below M = 0.8 the
nose wave drag is zero (the friction-dominated subsonic regime, handled
by Section 8.3), and the linear ramp avoids a discontinuous jump at the
critical Mach.

The 5 % accuracy quoted by Chin is adequate for the preliminary-design
role this simulator fills.

#### 8.2.2 Tangent ogive — closed-form (Chin Eq. 3-9, Miles formula)

For a tangent-ogive nose with fineness ratio `l/d`, Chin's adaptation of the
Miles slender-body formula is implemented as
(`_cd_wave_ogive`, `missile_models.py`):

```
σ              = arctan(1 / (2 · l/d))                  [radians; converted to ° for P]
P              = (0.083 + 0.096/M²) · (σ° / 10)^1.69    [cone reference]
factor         = max(0, 1 − 2·[196·(l/d)² − 16] / [28·(M + 18)·(l/d)²])
C_D,wave,ogive = P · factor
```

The reference pressure coefficient P is the Chin cone formula
(Section 8.2.1) evaluated at the tangent-ogive's equivalent half-angle;
the Miles factor reduces P toward zero as the ogive becomes more slender
(higher l/d) or as Mach increases. The `max(0, ...)` clip protects
against unphysical negative values at very high l/d where the formula
predicts more drag reduction than is physical.

The tangent ogive is the most common operational nose shape for ballistic
missiles and military rockets, and Chin reports satisfactory agreement
with correlated wind-tunnel data for preliminary design.

#### 8.2.3 Power-series shapes — NACA TN 4201 (with fineness scaling)

For the Von Kármán (LD-Haack, C = 0), LV-Haack / Sears-Haack (C = 1/3),
and parabola (K' = 1) shapes Chin provides no closed-form formula; the
wave-drag coefficient is obtained from **NACA TN 4201**, which compares
these shapes side-by-side at a reference fineness ratio
`l/d_nose = 3` (`_cd_wave_table`, `missile_models.py`).

For other fineness ratios the table is scaled by

```
C_D,wave(l/d) = C_D,wave(l/d_ref) · (l/d_ref / l/d)²
```

with `l/d_ref = 3`. This is the slender-body 1/f² scaling — exact for
cones and standard NACA practice for inter-fineness comparison of
power-series shapes.

**Note on how the lookup table is populated.** TN 4201's comparison figures
plot *total body drag* for a complete nose + cylindrical afterbody + 3.2°
boattail configuration, not nose wave drag alone. The wave-drag-only
component is isolated by **calibrating against Chin's closed-form cone
formula** (Section 8.2.1): the cone result is treated as the known reference,
and the residual offset between TN 4201's total-drag curves and the
cone-formula prediction at matching geometry gives the friction-plus-base
contribution. That contribution is then subtracted from each shape's total
drag to yield the wave-drag-only table. The source comment in the code
reads: *"Source: NACA TN 4201 comparison data (models 56-63, l/d_nose=3,
M=0.8-2.0) calibrated against Chin (1961) cone formula to isolate wave
component."*

The implemented table (at `l/d_nose = 3`):

| M | Von Kármán | LV-Haack | Parabola |
|---|---:|---:|---:|
| 0.0 – 0.8 | 0.000 | 0.000 – 0.010 | 0.000 – 0.010 |
| 0.9 | 0.010 | 0.030 | 0.040 |
| 1.0 | 0.030 | 0.070 | 0.090 |
| 1.1 | 0.050 | 0.082 | 0.100 |
| 1.2 | 0.060 | 0.085 | 0.100 |
| 1.5 | 0.069 | 0.084 | 0.094 |
| 2.0 | 0.067 | 0.077 | 0.087 |
| 3.0 | 0.058 | 0.068 | 0.077 |
| 4.0 | 0.052 | 0.061 | 0.069 |
| 5.0 | 0.047 | 0.055 | 0.062 |

Above M = 2.0 the table is extrapolated smoothly (no TN 4201 data above
M = 2); the trend from 2 → 5 reflects the expected slow decline of nose
wave drag at high supersonic Mach. Von Kármán is meaningfully (~30 – 50 %)
better than LV-Haack and parabola at the transonic peak — a real
performance difference, not a relabel.

#### 8.2.4 Blunt cylinder

For the `blunt_cylinder` option the wave-drag is a simple Mach-dependent
piecewise model:

```
M ≤ 0.8:        C_D ≈ 0.9
0.8 < M ≤ 1.5:  C_D ≈ 0.9 + (M − 0.8) / 0.7 · 1.3      (linear rise)
M > 1.5:        C_D ≈ 2.2
```

This is provided for blunt-faced payloads (e.g. unaerodynamic test
articles); it is not a recommended shape for any operational design.

#### 8.2.5 Blunting allowance

For sharp shapes (cone, tangent ogive) Chin §3-10 establishes that the
nose tip may be blunted to a radius ratio `r_N / r_b ≤ 0.2` with no
significant drag penalty. NACA TN 4201 Figure 7 (parabolic body at
fineness ratio 8.91) extends this finding — the curves at
`r_N / r_b = 0, 0.187, 0.274` are nearly indistinguishable over
M = 0.6 – 1.5, and significant drag rise only appears above
`r_N / r_b ≈ 0.4`. The empirical threshold is therefore *looser* than
Chin's conservative rule.

The code accordingly does not apply a separate bluntness correction
in the wave-drag term — bluntness only enters via the stagnation-heating
calculation (Section 13) and the Newtonian β calculator
(Section 8.8). For nose tips outside the safe regime
(`r_N / r_b > 0.4`), the user should switch to the `blunt_cylinder`
shape.

### 8.3 Skin friction

The skin-friction coefficient is computed as the Reynolds-number-weighted
sum of laminar and turbulent contributions, modified by compressibility
and a roughness allowance (`_cd_friction`, `missile_models.py`):

```
C_D,friction = (S_wet / S_ref) · C_f,mixed · C_compress(M) · 1.10
s_lam        = min(1, Re_tr / Re_L)
C_f,mixed    = s_lam · C_f,lam  +  (1 − s_lam) · C_f,turb
```

with transition Reynolds number `Re_tr = 5×10⁵` (Chin Eq. 4-3). The
`min(1, ...)` clamp ensures that when `Re_L ≤ Re_tr` the boundary layer
is treated as fully laminar (`s_lam = 1`), and for `Re_L → ∞`
`s_lam → 0` so the flow becomes fully turbulent. The length-based
Reynolds number `Re_L = ρ V L / μ` uses the active stage's body length
and the dynamic viscosity from Sutherland's law (8.3.4).

#### 8.3.1 Laminar Cf — Blasius (Chin Eq. 4-1)

```
C_f,lam = 1.328 / √Re_L
```

The Blasius flat-plate result. For typical missile boost-phase conditions
`Re_L` exceeds 10⁶ within seconds of liftoff, so the laminar contribution
is small but not negligible at the nose-body junction.

#### 8.3.2 Turbulent Cf — Schoenherr (Chin Eq. 4-2)

The incompressible turbulent flat-plate coefficient `C_f,turb` satisfies
the implicit Schoenherr equation:

```
√C_f,turb · log₁₀(C_f,turb · Re_L) = 0.242
```

Solved by Newton iteration from a Prandtl–Schlichting initial guess
(`C_f ≈ 0.074 · Re_L^(−0.2)`); convergence is 3–4 iterations to machine
precision in practice.

#### 8.3.3 Compressibility correction — Frankl-Voishel (Chin Eq. 4-6)

```
C_compress(M) = (1 + 0.2 M²)^(−0.467)
```

The compressibility factor reduces `C_f` modestly at low supersonic
speeds (about 17 % at M = 2) and more strongly at higher Mach. The
formula is fit to adiabatic-wall flat-plate data over 0 ≲ M ≲ 5.

#### 8.3.4 Viscosity — Sutherland's law

```
μ(T) = μ_ref · (T / T_ref)^1.5 · (T_ref + S) / (T + S)
```

with `μ_ref = 1.716×10⁻⁵ Pa·s`, `T_ref = 273.15 K`, `S = 110.4 K`
(`_mu_air`, `missile_models.py`). Sutherland's law provides
viscosity over the temperature range encountered in atmospheric flight
to within 1 % accuracy. Combined with `ρ` and `T` from the atmosphere
model (Section 4) and the missile's instantaneous speed and length,
this gives the Reynolds number used by the friction calculation.

#### 8.3.5 Mixed laminar/turbulent flow

The mixed Cf above assumes the boundary layer is laminar from the nose
tip to the point where local Reynolds equals `Re_tr = 5×10⁵`, then
turbulent beyond. For `Re_L < Re_tr` the boundary layer is entirely
laminar and `C_f,mixed = C_f,lam`; for `Re_L >> Re_tr` it is essentially
fully turbulent. For most operational missile boost trajectories the
turbulent regime dominates within the first few seconds.

#### 8.3.6 Wetted area — Crowell (1996)

The friction integral requires the wetted area `S_wet` of each body
component, taken as the area-weighted sum of nose wetted area and
cylindrical-body wetted area. The nose contribution is computed by
numerical integration of the appropriate generator profile
(`_s_wet_ratio`, `missile_models.py`, after [Crowell 1996](#16-references)):

| Component | Treatment |
|---|---|
| Cylindrical body | `S = π · L_body · D` — exact |
| Cone | `S = π R √(R² + L²)` — slant area, closed form |
| Tangent ogive | Closed form from Crowell |
| Parabola (K' = 1) | Closed form from Crowell Eq. 7 |
| Von Kármán / LV-Haack | Numerical 1-D quadrature of frustum panels |

#### 8.3.7 Surface roughness

Chin recommends adding 10 % to the smooth-wall friction coefficient
for typical operational missile surface finishes (Chin §4-2). The code
applies this as a flat multiplicative factor `× 1.10` on the friction
term, which is then summed with the wave-drag and base-drag terms.

#### 8.3.8 Alternatives considered

- **Reference-temperature method (Eckert 1955)**: Compute a reference
  temperature `T* = 0.5(T_wall + T_static) + 0.22(T_stag − T_static)`,
  evaluate properties at `T*`, then apply a Prandtl–Schlichting
  flat-plate formula. More physically grounded for hypersonic flow but
  requires wall temperature, which a 3-DOF trajectory tool does not
  track. The Frankl-Voishel approach is the textbook-standard simpler
  choice for preliminary work and matches NACA-era validation data
  cited in Chin Chapter 4.
- **Van Driest II compressibility correction**: higher fidelity than
  Frankl-Voishel but also requires wall conditions. Same reason for
  deferring.

### 8.4 Base drag

The base-drag coefficient is taken from Chin's digitised Fig. 3-15 (power-off
base pressure coefficient):

```
C_D,base = −C_pb · (S_base / S_ref)
```

with `C_pb` (the base pressure coefficient, negative — i.e. suction)
from `_BASE_CPB` (`missile_models.py`):

| M | C_pb |
|---|---:|
| 0.0 | 0.000 |
| 0.8 | −0.13 |
| 1.0 | −0.20 |
| 1.2 | −0.18 |
| 1.5 | −0.14 |
| 2.0 | −0.10 |
| 2.5 | −0.08 |
| 3.0 | −0.06 |
| 4.0 | −0.05 |
| 5.0 | −0.04 |

with linear interpolation between table entries. Note the inversion:
`C_pb` is negative (suction at the base), so `−C_pb` gives the positive
drag coefficient.

A useful closed-form approximation consistent with Chin's table and
USAF DATCOM:

```
M < 1:        C_D,base ≈ 0.12
M ≥ 1:        C_D,base ≈ 0.12 / M
```

agreeing with the tabulated values to ~10 % over M = 1–3.

**Power-on behaviour — not modelled.** Chin's Fig. 3-15 data is
explicitly power-off: the engine is not firing, so the entire base area
sees atmospheric back-pressure, producing the suction (negative `C_pb`)
that drives the drag. When the engine is firing, the exhaust plume
pressurises the region behind the nozzle and reduces the base suction;
power-on base drag is therefore *lower* in magnitude than power-off.
Thrusty does **not** make this distinction — `_cd_base(mach)` is called
unconditionally in every boost-phase drag evaluation
(`missile_models.py`), so powered flight sees the same `C_pb(M)` as
coast. The result is a small but consistent over-estimate of drag
(under-estimate of range) during boost. Modelling the power-on
correction properly would require nozzle exit conditions and plume CFD,
which is outside the scope of a preliminary 3-DOF tool. For the
operational class of vehicle this tool targets — solid and
storable-liquid ballistic missiles and SLV boosters — the power-off
treatment is conservative and consistent with Forden's original
approach.

The same simplification applies in the presence of strap-on boosters
(Section 6.5): the booster cluster around the rear of the core would,
in reality, partially fill the wake region where base suction develops,
reducing the core's `C_D,base`. This interaction is not modelled.

### 8.5 Fin drag and lift

For finned vehicles (e.g. SCUD-B and other early ballistic missiles)
two contributions are added:

**Fin lift slope** (`_cl_alpha_fins`, `missile_models.py`) — the fin
normal-force-curve slope from **Barrowman's 1967 thesis, Eq 3-12** (the
canonical slender-finned-vehicle method; the thesis itself is in
`data/`), referenced to the body base area `A_ref = π(d/2)²`:

```
A_f  = s·(c_root + c_tip)/2                 # one exposed fin planform
AR   = (2s)² / A_f                          # reflected aspect ratio (span 2s)
β    = √|M² − 1|                            # Prandtl-Glauert (sub) / supersonic
Γ_c  = mid-chord sweep;  tan Γ_c = tan Λ_LE + (c_tip − c_root)/(2s)

C_Nα = N·π·AR·(A_f/A_ref) / [ 2 + √(4 + (β·AR / cos Γ_c)²) ]   ·  K_T(B)
```

The `N·π` numerator is Barrowman's cruciform result — for N=4, two fins lie in
the pitch plane, giving 2× the single-fin `2π` form (Eq 3-6). Body–fin
interference uses Barrowman's simplified slender-body factor
`K_T(B) = 1 + r/(s+r) = 1 + d/(2s+d)`.

**Regime — this is BOOSTER aerodynamics, not a glider model.** Eq 3-12 is
small-angle-of-attack, linear, fin-stabilised slender-vehicle theory (valid to
~7° AoA, subsonic through supersonic). It is used for a **booster's static
margin** (Section 8.9). It is deliberately **not** applied to a gliding RV: a
glide vehicle is a high-AoA hypersonic *lifting body* whose L/D is a Newtonian
property, supplied directly as `rv.glider_LD` and used by the glide trajectory
(Section 12). Earlier code mis-applied a (buggy) fin lift slope to the glider
L/D estimate; that has been removed.

**Fin drag** (`_cd_fins`, `missile_models.py`) — flat-plate skin friction
on the fin wetted area plus a thickness/wave term:

```
C_D,fric = (2 N · A_fp / A_ref) · C_f · (1 + 2 t/c̄)      # both faces, Mandell 1973
C_D,wave = (4 N · A_exp / A_ref) · (t/c̄)² / β            # Ackeret, supersonic
```

with `N` the fin count, `A_exp` the exposed planform, `A_fp = A_exp + ½ c_root·d`
the body-overlap planform, `t/c̄` the thickness-to-mean-chord ratio, and
`β = √(M²−1)`. This is a **subset of Barrowman's §4** fin-drag decomposition
(`C_D = C_Df + C_DL + C_DB + C_Dw`: friction + leading-edge pressure +
trailing-edge/base + wave); the leading-edge and base components are not yet
modelled, so fin drag is mildly under-counted.

For strategic and theatre-range missiles flying mostly through the upper
atmosphere, fin drag is a second-order effect and is sometimes set to
zero by leaving the fin parameters at their default zero values. For
short-range tactical missiles or atmospheric flight the fin term matters
and should be enabled.

**Planar fin drag is applied in the trajectory** (`drag_force_vector`) while
the finned stage is the active stage — a finned first stage plows through the
dense lower atmosphere during ascent, so its fin drag affects range. It is
referenced to the body base area and added to the body drag exactly as the
grid-fin term is. **No lift is added**: an ascending vehicle flies at ≈0° angle
of attack, so the fins' normal force is a stability effect (static margin), not
a trajectory force — boosters are thus treated as drag + stability, distinct
from RVs/gliders where lift (L/D) is the governing aerodynamics. (Example: the
Strypi VIII R's four large swept Castor fins cost it ~18% of range; the effect
scales with (t/c)² in the wave-drag term, so it is sensitive to the fin
thickness when that is estimated.)

**Grid (lattice) fins** (`_cd_gridfins`, `_cl_alpha_gridfins`,
`missile_models.py`) — a grid fin is a box-frame lattice of thin cells, not
a planar airfoil, so the flat-plate/Ackeret model above does not apply. The
drag model is **calibrated to Washington & Miller, AIAA 93-0035** (the S1
fine-mesh configuration: their Fig. 2 geometry and Fig. 14 drag data). The
measured axial-force coefficient (referenced to body cross-section) is
roughly **flat at ~0.040 outside transonic with a modest bump to ~0.065
(≈1.5×) peaking near M ≈ 0.95** — not a large spike. The model is:

```
C_D,gridfin = C_friction(wetted web area, chord Re)
            + C_edge   (blunt LE+TE/profile drag × web blockage area)
            + transonic bump over [M_sub, M_rec], peaking at M_peak
```

with the web blockage area `(1 − φ)·A_frame`, porosity `φ = ((p − t)/p)²` for
cell pitch `p` and web thickness `t`.

**Solidity (σ) — the practical input.** The drag is driven by the blocked
frontal fraction, the *solidity* `σ = 1 − φ = 1 − ((p − t)/p)²` (≈ 2·t/p for
thin webs). σ is the single quantity that stands in for the two lattice
details that are hardest to obtain from open sources — web (wall) thickness `t`
and cell pitch `p` (centre-to-centre). See `docs/grid_fin_solidity_diagram.png`
for a labelled frontal view of `t`, `p`, and the open window `(p − t)`.

This blockage parameterisation is independently corroborated by **Dikbaş 2015**
(METU M.S. thesis, *Design of a Grid Fin… for Transonic Flight*, in `data/`),
whose grid-fin design driver is the **web-to-cell ratio `t/w`** — i.e. exactly
`σ/2` for thin webs. His transonic design study (D = 400 mm, c = 0.1 D, fixed
1 mm web, cell width swept) spans `t/w = 0.0025–0.030`, i.e. **σ ≈ 0.005–0.06 —
all open**, and the drag-optimal direction is toward *larger* cells (lower σ),
reinforcing that transonic-efficient fins are open. The thesis also validates
its CFD against Washington-Miller (our calibration anchor), and its "unit grid
fin" (single-cell) result — that grid-fin loads scale per-cell × cell count —
is the same per-cell basis as the σ·(frame area) scaling used here. It adds no
new fielded fin and no closed-form correlation to adopt; it corroborates the
approach rather than changing it. Two ways to supply σ:

- If `t` and `p` are known, `grid_fin_solidity(t, p)` converts them to σ via
  the equation above (or just enter `grid_fin_web_thickness_m`/
  `grid_fin_cell_pitch_m` and let the model derive φ).
- If they are not, set `grid_fin_solidity` directly — estimate it from imagery.
  The reference classes below are **anchored to real published fins** (σ computed
  from their cited geometry), and the key finding is that measured grid fins are
  **very open** — far more open than a naïve "looks blocked" guess:
  - **open · σ ≈ 0.04–0.06** — two independent real fins land here:
    (i) the standard US-tested fin: 0.371 in cell pitch (centre-to-centre,
    Miller-Washington 1994 AIAA-94-1914 Fig. 1; same lattice in
    Kretzschmar-Burkhalter 1998 G12–G16), 0.008 in web → σ = 1−(0.363/0.371)² =
    **0.043** (the most-tested grid fin in the open literature); and
    (ii) a **real Russian patented design** — RU 2 686 593 C1 (Komarov et al.,
    2019), worked example: 36 mm cell pitch, 1 mm wall → σ = 1−(35/36)² =
    **0.055**.
  - **typical · σ ≈ 0.09–0.12** — the **AA-12-class reference fin** used across
    ~20 studies: Debiasi 2020 (J. Spacecraft & Rockets, DOI 1.C035626) Fig. 1
    baseline, 0.1109 D pitch, 0.007 D wall → σ = 1−(0.1039/0.1109)² = **0.122**
    (measured wind-tunnel model). The same family with the thinner 0.005 D web —
    Abate 2000's GTCM fin and the DRDC/DeSpirito fin re-used by Despeyroux 2015
    (D = 30 mm) — gives σ ≈ 0.087–0.09, so several independent groups corroborate
    this anchor.
  - **dense · σ ≈ 0.2–0.3** — occurs on real hardware, but **localised to the
    structurally-loaded root** of a large fin, not the aero span. The **SpaceX
    Falcon 9** titanium grid fin (≈1.2 m, ~8 cells of **14.5 cm** pitch across)
    is the clean example: its lattice walls are only ~2–3 mm at the aerodynamic
    *tip* → σ = 1−(142/145)² ≈ **0.04** (open, same ballpark as the MICOM fin),
    but thicken markedly toward the *root* where they carry the fin's bending
    load — a ~15–25 mm root web gives σ ≈ **0.20–0.32**. So a single flight fin
    spans the whole open→dense range. (The only fully *aero* example of a dense
    lattice is Chen 2000's thick-web DREV CFD case, σ = 0.225 — the same
    AA-12-class fin with ~3× the standard web — and the literature notes such
    thick lattices choke and add transonic drag, which is why nobody flies a
    uniformly dense fin.) Falcon 9 cell/web figures are from observation/public
    imagery; SpaceX has not published official dimensions.

  **σ is not a single number for a real fin — it varies span-wise** (open
  aero-tip → dense structural-root, as above). The single σ this model takes
  should represent the **aerodynamically active** region (the open/typical tip
  and mid-span); the dense root is a structural feature contributing little
  lift or projected drag area.

  **Empirical σ range (all real fins gathered).** Pooling every fin for which a
  σ can be computed from cited/observed geometry (web-to-cell `t/w ≈ σ/2`):

  | Fin | type | σ | t/w |
  |---|---|---|---|
  | US MICOM (W&M / Kretzschmar) baseline | measured | 0.043 | 0.022 |
  | US MICOM thick-web variant | measured | 0.064 | 0.032 |
  | Russian patent (Komarov RU 2686593) | patent design | 0.055 | 0.028 |
  | AA-12-class, 0.005 D web (Abate/DeSpirito) | measured | 0.088 | 0.045 |
  | AA-12-class, 0.007 D web (Debiasi) | measured | 0.122 | 0.063 |
  | SpaceX Falcon 9 — aero tip | observed | 0.041 | 0.021 |
  | Dikbaş transonic design sweep | CFD design | 0.005–0.06 | 0.003–0.03 |
  | — Falcon 9 structural root (~20 mm web) | observed | ≈0.26 | 0.14 |
  | — Chen DREV thick web | CFD parametric | 0.225 | 0.12 |

  So the **aerodynamically-active σ of real grid fins is ≈ 0.04–0.12**
  (`t/w ≈ 0.02–0.06`). Large **booster / launch-vehicle / SLBM** fins — the class
  relevant to most vehicles modelled here (STARS-class boosters, Topol, Falcon 9,
  Russian SLBMs) — sit at the **open end, σ ≈ 0.04–0.06**; the denser ≈0.09–0.12
  values are the smaller **air-to-air (AA-12) class**. Only structural roots and
  deliberately thick CFD cases exceed ≈0.15 (and those choke transonically). A
  sensible default for a booster grid fin is therefore **σ ≈ 0.05**; treat σ ≳
  0.15 as atypical for an aero surface.

  See `docs/grid_fin_solidity_classes.png` for the three classes side by side.
  (Earlier drafts used invented bands of 0.10–0.30; those were not anchored to
  data and read too dense — corrected here against the cited geometry. The
  Russian patent independently confirms the construction the model assumes:
  square lattice cells at 45° with wedge-sharpened edges.)

  *Origin source.* The foundational monograph — **S. M. Belotserkovsky (ed.),
  "Решетчатые крылья" (Reshetchatye Krylya / Lattice Wings), Mashinostroenie,
  Moscow, 1985** (the original of the 1987 FTD translation) — is in `data/`. It
  is a theory/structures/materials work (vortex-lattice aerodynamics, structural
  mechanics, manufacturing), **not** a catalogue of fielded-missile fin
  dimensions, so it yields no new σ data point. Worth recording, though: its
  primary *dimensionless* geometric parameter (§1.1) is the **relative pitch
  t̄ = t/b** (cell pitch ÷ chord), and it shows lift depends on size and t̄,
  independent of cell-shape (frame "рамное" vs honeycomb "сотовое"). So the
  origin literature parameterises by t/b — neither the blockage *solidity* used
  here (a Western/derived convention) nor the chord/height ratio that an earlier
  draft of this note wrongly invented. σ remains the practical input for the
  drag model; t/b is the classical lift parameter.

  *Provenance:* these are mostly **research / generic test fins**, not fielded
  hardware — the open anchor is the US Army Missile Command (MICOM) grid-fin
  R&D fin (plus a Russian *patent*, which is a design rather than a known
  fielded missile), and the typical anchor is the AA-12-class reference fin
  (also the body of Abate's explicitly "Generic" Tail Control Missile). The
  closest tie to a fielded weapon is that AA-12-class geometry, which several
  papers state resembles the **Russian R-77 / AA-12 "Adder"**. Grid fins *are*
  widely fielded — the Sharma-Kumar 2019 review (INCAS Bull. 11(1)) lists, among
  others, the Soviet/Russian **SS-20, SS-21, SS-25 (Topol)** ballistic missiles,
  **MOAB**, **Soyuz** (launch-escape/braking), and **R-77/AA-12** — but it
  publishes no dimensioned cell geometry for them, so their σ cannot be computed
  from these sources. A second Russian patent — **RU 2 532 287 C1** (Leonov
  et al., **NPO Mashinostroyeniya**, 2014) — is a real missile-bureau design
  using **folding lattice stabilizers on the boost stage of a submarine-launched
  missile** (a direct analogue of the timed-deploy booster grid fins modelled
  here, §grid-fin deployment), but it likewise gives no cell dimensions. (That review also asserts the US AMRAAM uses grid fins;
  that appears to be an error — AMRAAM uses planar tail fins.) So the bands
  reflect grid-fin *technology*
  as tested in the open literature; the consistent picture across US (MICOM),
  Russian (patent), and AA-12-class sources is that real fins are **open to
  typical (σ ≈ 0.04–0.12)**, with higher σ only from deliberately thick webs.
  The review independently corroborates this: it notes thick fin panels *degrade*
  performance and that thinning the webs *reduces* drag, i.e. designs favour thin
  (low-σ) lattices.
  When σ
  is given without a pitch, a representative cell count (`_GRIDFIN_DEFAULT_CELLS`
  = 10) sets the wetted area for the *secondary* friction term only.

σ sets the blockage exactly but cannot recover the absolute mesh scale that
fixes friction, so the σ-only path agrees with the full web/pitch path to
~7% for blockage-dominated (thicker-web) fins like STARS, and less well for
unusually fine thin-web meshes (e.g. W&M S1, which is friction-dominated — use
its known `t`/`p` there). The bump represents the three flow
regimes W&M describe (Fig. 6/7): the cells **choke below M = 1**, flow spills
around the fin, the shock attaches and then passes undisturbed, restoring
supersonic behaviour by **M ≈ 1.6**. Those onset/peak/recovery Mach anchors
(`_GRIDFIN_M_SUB`/`M_PEAK`/`M_REC` = 0.75/0.97/1.60) are taken from the S1
data. A `_kantrowitz_contraction_ratio`/`_gridfin_start_mach` helper computes
the self-starting Mach for contraction ratio `CR = 1/φ` from the standard
normal-shock area relation (the class of 1-D isentropic analysis W&M used);
note its *geometric* contraction under-predicts the choke for thin-web fins
because boundary-layer blockage in the small cells is a co-cause, so the bump
anchors come from data, not geometric Kantrowitz alone.

An **edge-shape factor** (`grid_fin_edge_factor`, default 1.0 = blunt
rectangular webs) scales the pressure (edge + transonic-bump) drag but not the
friction. Miller & Washington (AIAA 94-1914, Tables 2/3) measured that shaping
the frame cross-section (single wedge, half-diamond) cuts grid-fin drag ~20–45%
subsonic and ~8–27% supersonic versus the blunt baseline, so a sharp/shaped fin
is ~0.6–0.85; the default matches the blunt W&M S1 / Miller F1 case.

**Validation:** run on W&M's exact S1 geometry the model reproduces ~0.042
(subsonic), ~0.065 (transonic peak) and ~0.038 (supersonic) to within ±13%
across M = 0.5–3.5. Three further papers (all read) corroborate the structure:

- **Miller & Washington, AIAA 94-1914** (fin-only axial force, six frame/web
  variants) confirms the transonic peak (CD rises 0.5→0.9, decreases above 0.9)
  and quantifies the edge-shape and web-thickness sensitivities above.
- **DeSpirito & Sahu, ARL-RP-19 / AIAA 2001-0257** (viscous CFD + DREV tunnel)
  gives a *total*-missile Cx ≈ 0.43 (M2) → 0.45 (M3), roughly flat — supporting
  the flat supersonic baseline and ruling out a decaying form.
- **DeSpirito et al., AIAA 2000-0391** (viscous CFD vs DERA tunnel, M 2.5)
  computes the missile body-alone, with planar fins, and with grid fins, which
  **isolates the grid-fin axial increment**: ΔCx ≈ 0.47 − 0.19 = **0.28** (4 L2
  fins, ref body area) — ~1.75× the equivalent planar-fin increment. This is the
  one fin-isolated supersonic check available: the model reproduces 0.28 at
  σ ≈ 0.33 (0.21–0.24 at σ = 0.25–0.30), and the flat M1.5→3 shape matches. Since
  the L2 fin's web/pitch (hence σ) is unpublished it is a consistency check, not
  a pinned match — but it validates the **solidity scaling** across ~10× from the
  σ ≈ 0.032 W&M S1 calibration point (and hints at a possible mild
  under-prediction of supersonic fin drag at high solidity).
- **Abate, Duckerschein & Hathaway, AIAA 2000-0937** (free-flight GTCM) finds
  total Cx flat below **M ≈ 0.77** then a steep transonic rise, independently
  confirming the choke-onset Mach (`_GRIDFIN_M_SUB` = 0.75).
- **Chen, Khalid, Xu & Lesage, AIAA 2000-0987** (Euler CFD, M 1.5/2.0,
  parametric in panel thickness and edge shape) corroborates three behaviours
  directly: thicker panels raise axial force (the solidity scaling), a
  knife-edge face "naturally produces less axial force than a blunt block"
  (the edge-factor sign), and axial force is higher at M1.5 than M2.0 with
  thick-web choking persisting to higher Mach (the transonic peak + choke).
- **Brooks & Burkhalter, J. Aircraft 1989** is the foundational subsonic
  vortex-lattice analysis underlying the Kretzschmar & Burkhalter method;
  incompressible and lift-focused, it confirms drag rises with added
  slats/blockage and the grid-fins-as-drag-brakes concept (no supersonic drag
  data).
- **Munawar, ICAS 2010** (standalone-fin RANS, M 0.5–2.5) qualitatively
  confirms the rise in grid-fin effectiveness from low to high supersonic as
  the shock is swallowed through the lattice (the recovery regime). Its drag is
  referenced to an unstated/fin-based area, has no cell spacing (no σ), and no
  Cd–Mach curve at α=0, so it is not quantitatively comparable.
- **Theerthamalai & Nagarathinam, J. Spacecraft & Rockets 43(4), 2006** is a
  shock-expansion analytical method for supersonic grid fins, validated against
  experiment for normal force, pitching moment, and axial force. Two points
  bear on this model: (a) supersonic fin axial force gently *reduces* with Mach
  (this model holds a flat supersonic baseline, matching its flat W&M S1
  calibration anchor; the decline is mild and within scatter, so it is left
  flat); (b) even this dedicated method under-predicts axial force by **<10%**
  vs experiment — i.e. ~10% is the achievable accuracy for supersonic grid-fin
  axial force, which puts this reduced-order model's ~13% (DeSpirito-2000
  check) in context rather than indicating a defect. High grid density at low
  supersonic is the hardest regime (web shock interaction), as caveated.
- **Kretzschmar & Burkhalter, "Aerodynamic Prediction Methodology for Grid
  Fins"** (NATO RTO-MP-5, 1998) is an independent analytical method (vortex
  lattice + Evvard's theory) whose grid-fin axial force decomposes the same way
  as this model — skin friction (wetted area, Cf(Re)) + pressure drag on the
  fin **frontal area** + an interference term — and is α-independent. Its
  flow-regime table (subsonic M<0.8; **choked 0.8<M<1.0**; bow shock 1.0–1.4;
  shock swallowed **1.4–1.9**) brackets our Mach anchors, and it models the
  transonic choke with the **same isentropic throat-area relation** used here,
  explicitly attributing the sub-M=1 choke to cell-wall *and boundary-layer*
  blockage — the co-cause noted above. K&B treats fin **span, cell density, and
  chord** as *independent* geometric parameters and varies each one directly
  (G12/G13 span, G12/G14 cell density, G15/G16 chord). It does **not** define a
  chord-to-height (`C/h`) normalizing ratio or state that chord scales with
  height — that framing, and any `C/h ≈ 0.07–0.35` range, were this author's
  inference and are **not** a sourced K&B design rule. Chord is an independent
  parameter; the model takes it as a direct input.
- **Washington & Miller, "Experimental Investigations of Grid Fin Aerodynamics:
  A Synopsis of Nine Wind Tunnel and Three Flight Tests"** (AGARD/RTO, 1998) is
  the umbrella dataset (26 configs; the Miller 94-1914 drag fins are G6–G11 and
  the Kretzschmar fins are G12–G16). Its Figure 8 marks the flow regimes as
  **choked flow at M = 0.75, leading-edge shock attachment at M = 1.35, and no
  shock reflection (supersonic recovery) at M = 1.60** — i.e. `_GRIDFIN_M_SUB`
  and `_GRIDFIN_M_REC` match W&M's own markers exactly. It confirms the five
  defining parameters (span, chord, height, cell spacing, web thickness) that
  the model takes as direct inputs (the G12–G16 fins are the Kretzschmar set).

**Limitation — fin sweep not modelled.** W&M (Synopsis, §3.7) found that
sweeping a grid fin forward or aft by ±30° **amplifies its axial force by a
factor of 2–3** while leaving normal force essentially unchanged (grid fins can
be used as deployable drag brakes). This model assumes **unswept** fins; a
swept installation would have substantially higher drag than computed.

The DeSpirito and Abate data are total-missile (not fin-isolated) and Abate's
fins are blunt sub-scale, so they are qualitative corroboration, not
quantitative fin-drag checks. **Caveats:** calibrated to a single blunt-edged
configuration; the bucket shifts with cell size/Reynolds number; extrapolation
to other geometries is uncertain. STARS-1 uses `grid_fin_edge_factor` = 1.0
(blunt — conservative) because its fin edge shape is not documented. Drag is referenced to body base area and added in
`drag_force_vector` only while the finned stage is active (first-stage fins
jettison at staging). The STARS-1 booster (AHW Flight-1 carrier) carries eight
first-stage grid fins via this model (dimensions are engineering estimates);
they cost it ~200 km of range from ascent drag.

### 8.6 Aerospike correction

An aerospike is a forward-projecting spike (sometimes terminated in a
small aerodisk) that creates a slender bow shock to replace the strong
detached shock of a blunt body, reducing wave drag at supersonic Mach
(`_aerospike_effective_LD`, `missile_models.py`).

The implementation replaces the actual nose's wave drag with the
*minimum* of (actual nose drag) and (effective-body cone drag), where the
effective body is a cone whose *fineness ratio* `L/D_eff` is set from the
spike geometry by a linear fit to the Ahmed & Qin (2011) dividing-streamline
angles for sharp spikes on hemisphere-cylinder models:

```
spike L/D = L_spike / D_body
spike d/D = D_aerodisk / D_body
L/D_eff   = 1.0 + (2/3)·spike_LD + 2.0·spike_dD      (Ahmed & Qin 2011 fit)
```

`L/D_eff` is then passed to the cone wave-drag routine `_cd_wave_cone`
(Section 8.2.1); the half-angle conversion `arctan(1/(2·L/D))` happens
*inside* that routine, not in the spike model itself.

For a hemispherical aerodisk tip the spike behaves as a blunt protrusion
that detaches the bow shock at the spike rather than at the body, and
the effective body is the cone running from the aerodisk to the body
shoulder. The correction is active only above M = 0.8 (where a bow
shock exists to replace).

This term is provided for cases where a missile is known to use an
aerospike (some SLBM and modern SLV designs); the GUI fields are zero
by default, leaving the standard nose drag intact.

### 8.7 Reference geometry and shroud transitions

The reference area `S_ref` used to non-dimensionalise drag depends on
flight phase:

| Phase | Reference geometry | Nose-shape source |
|---|---|---|
| Boost with shroud attached | Shroud envelope | `shroud_nose_shape`, `shroud_diameter_m`, `shroud_length_m` |
| Boost after shroud jettison, RV stays with stage | Payload geometry | `nose_shape`, `diameter_m`, `length_m` from active stage |
| Boost after shroud jettison, RV separates (`rv_separates = True`) | RV geometry | `rv_shape`, `rv_diameter_m`, `rv_length_m` |
| Coast / re-entry with `rv_beta > 0` | β-based — no `S_ref` needed | n/a (see 8.8) |
| Coast / re-entry with `rv_beta = 0` | Final stage Mach-table fallback | Final stage |

The shroud-jettison event (Section 6.3) handles the first transition. The
`rv_separates` flag handles the second. The fall-through cases use
sensible defaults; if a missile has neither a configured RV β nor a
specified RV geometry the original Forden Mach-table model applies.

### 8.8 Ballistic-coefficient parameterisation (coast and re-entry)

After final-stage burnout, drag on a reentry vehicle is conventionally
parameterised by its ballistic coefficient:

```
β = m / (C_D · A)              [kg/m²]
F_drag = q · m / β             [N]
```

This is a closed-form way to write drag without separately specifying
`C_D` and `A` — a small `β` indicates high drag (light, high-Cd), a large
`β` indicates low drag (heavy or aerodynamically clean). Operational
warheads typically run `β` ~ 10⁴ – 10⁵ kg/m²; light reentry test
articles can be ~ 10³.

The GUI provides a Newtonian β calculator (Section 14 of the user guide)
that estimates β for a cone-shaped RV from its half-angle and bluntness
ratio using a 4×6 interpolation table of Newtonian C_D values
(`_cd_blunted_cone_newtonian`, `thrusty.py`). For a sharp cone the
exact Newtonian result is `C_D = 2 sin²θ`. For blunted cones (half-angle
10°–40°, bluntness ratio `ε = r_N/r_b` from 0 to 1) the table is taken
from a published Newtonian-hypersonic Cd chart cited in the source code
as "Ref (4) Ch. 5". The full primary reference for "Ref (4)" is not
defined in the repository; given the surrounding source chain (Section 8.2)
it is plausibly Chin (1961) Ch. 5, but this should be verified before
citing the table as Chin in publication-grade work.

For hypersonic glide vehicles the constant-β model is augmented by a
lift term and (optionally) a polar-drag model — see Section 12.

### 8.9 Static margin and grid-fin sizing

The static margin tells whether a vehicle's fins are appropriately sized for
stable, controllable flight. It is computed (`grid_fin_sizing.py`) the Barrowman
way — the centre of pressure is the normal-force-weighted average of the
component contributions (thesis Eq 3-107):

```
x_CP = Σ_i (C_Nα,i · x_i) / Σ_i C_Nα,i
SM   = (x_CP − x_CG) / D            [calibers]
```

A margin of **~0.5–2 calibers** is the conventional "appropriate" band; below 0
is unstable, well above ~2 is over-finned. Two uses: **sanity-check** an OSINT
fin estimate, or **invert** for the fin area a vehicle of a given diameter
*should* carry (fins scale as area ∝ D² to hold the margin in calibers).

Component normal forces (all referenced to body base area `A_ref = π(d/2)²`):

- **Nose / body** (Barrowman Eq 3-65/3-66): `C_Nα = 2·A_base/A_r` (= 2 for a
  nose capping the body), with CP at a shape-dependent fraction of nose length
  (cone ⅔·L_N, tangent ogive ≈0.466·L_N, Eq 3-89/3-90).
- **Fins**: Barrowman Eq 3-12 (Section 8.5) for planar fins; the dedicated
  grid-fin slope `_cl_alpha_gridfins` for grid fins.

CG is estimated from the stage mass stack (mass-weighted longitudinal centroid
at liftoff/full — the most-aft CG, i.e. the *minimum*-margin case), overridable.
This is the **booster** stability tool; it does not apply to a gliding RV.

**Diameter transitions** are included (`body_normal_force` via `_stack_layout`):
Barrowman's body term is `ΔC_Nα = (2/A_r)·ΔA` at *every* cross-sectional-area
change, not just the nose. A multistage stack with a narrow payload/upper stage
stepping up to a wide lower stage has a forward-facing shoulder that adds a
stabilising (CP-aft) normal force. The net body C_Nα telescopes to `2·A_base/A_r`
(= 2 when the reference is the base), but its *distribution* — hence the body CP
— shifts aft when the transitions are modelled. For STARS-1 the narrow HGB→wide
first-stage shoulder moves the body CP from 1.33 m (nose-only) to 2.82 m,
raising the static margin from ~1.43 to ~1.59 cal. (Remaining limitation: a
separate payload section, when the nose is not the RV, is not yet inserted as
its own diameter step — only the nose and stage-to-stage transitions.)

### 8.10 No-separation glider: derived L/D and the trim/control gate

When the RV does **not** separate, the gliding/maneuvering vehicle is the whole
airframe and its L/D is an emergent geometric property (not a designed input,
as for a separating RV). Two modules handle this.

**Whole-missile L/D (`glider_ld.py`)** — the semi-empirical body+fin build-up at
angle of attack (the theoretical core of Missile DATCOM), assembled from primary
sources in `data/`: body normal force from **Jorgensen (NASA TR R-474)** Eq 2.12
— slender-body potential + **Allen-Perkins (NACA 1048)** viscous crossflow;
wing-body interference from **Pitts-Nielsen-Kaattari (NACA 1307)**, whose
slender-body factors satisfy the identity `K_W(B)+K_B(W) = (1+r/s)²`; combined
with Jorgensen Eq 5.3's `sin(2α)/(2α)` high-AoA correction. Referenced to body
base area:

```
C_Nα_pot = 2·(A_b/A_r) + (1+r/s)²·(C_Lα)_W·(S_W/A_r)
C_N(α)   = C_Nα_pot·sin(2α)/2 + η·C_dn(M_n)·(A_p/A_r)·sin²α
C_A(α)   = C_A0·cos²α ;  C_L = C_N cosα − C_A sinα ;  C_D = C_N sinα + C_A cosα
```
where `A_p` is the body's true side-projected **planform** area (the area the
Allen-Perkins crossflow term acts on): nose + cylindrical afterbody,
`A_p = fill·L_nose·d + (L−L_nose)·d`, with a shape fill factor (cone 0.5, tangent
ogive ≈0.67 by exact integration). The two crossflow factors are **sourced, not
assumed**: `η = 1` for supersonic/hypersonic free-stream Mach per **Jorgensen
(NASA TN D-7228, 1973)** — the analytic statement of this exact build-up
(Eq. 1) — and the crossflow drag coefficient `C_dn` is a function of the
crossflow Mach `M_n = M·sinα`, read from **Gowen & Perkins (NACA TN 2960, 1953)**
Fig. 7: ~1.2 at low M_n, a transonic peak ~2.1 at M_n=1, decaying to ~1.34 at
M_n=2.9. L/D is maximised over α. It is a preliminary-design estimate (a slender
body is a poor lifting shape, so L/D is modest, ~2–3).

**Validation against Digital DATCOM (USAF, public-domain, PDAS).** The build-up
was cross-checked against Digital DATCOM (AFFDL-TR-79-3032) for the finless
slender reference body (D=0.5 m, L=4 m, 1.5 m tangent-ogive nose) at M2/3/5,
α=0–20°. Zero-lift drag agrees within ~10% (C_A0: glider_ld 0.245/0.184/0.121 vs
DATCOM 0.272/0.189/0.109), and the best-glide AoA matches closely (16/14/12° vs
16/14/10°). The cross-check drove two sourced corrections:

1. The original `A_p = ½Ld` (a cone-only triangle) underestimated the planform
   of a body with a long cylinder, driving L/D ~20–30% low (worsening with
   Mach); replaced by the true nose+afterbody planform.
2. The original constant `C_dn = 1.2` under-predicted crossflow lift at high
   Mach — at a M5 best-glide AoA the crossflow Mach `M_n = M·sinα ≈ 1`, where
   the cylinder `C_dn ≈ 2.1`, not 1.2 — which is why the gap grew with Mach;
   replaced by `C_dn(M_n)` from Gowen-Perkins TN 2960 Fig. 7.

Together these bring L/D to within **−5%/−9%/−10%** (M2/3/5) of DATCOM, with the
residual now roughly **flat** in Mach instead of growing. `glider_ld` remains
slightly conservative (under-predicts L/D — the safe direction for range); the
~10% residual is consistent with the slender-body potential slope vs DATCOM's
fuller body-lift method. The input deck, reference output, and comparison script
are in `validation/datcom/`.

**Trim/control gate (`trim_gate.py`)** — a derived L/D is only *achievable* if
the airframe can trim and hold that AoA. Using the linearised pitching moment
about the CG:
```
C_mα = −SM·C_Nα,total ;   α_trim,max = (C_Nδ/C_Nα,total)·(x_fin−x_CG)/(x_CP−x_CG)·δ_max
C_Nδ = control_eff·C_Nα,fin     (control_eff = N-K-P k_W(B)/K_W(B): ~1 all-moving, ~0.85 typ., ~0.5 flap)
```
Outcomes: **SM ≤ 0 → unstable → tumbles → ballistic** (L/D≈0); SM > 0 with
`α_trim,max ≥ α_LDmax` → control reaches best glide (full L/D); otherwise
**control-limited** → achievable L/D is the curve value at `α_trim,max` (the
over-stable / weak-control case). The gate uses the static margin of §8.9 (body
incl. transitions + fins) and the `glider_ld` L/D curve, with a mass-stack CG
and aft fin station (both overridable). It is a preliminary gate, not a 6-DOF
trim solution.

**Wiring.** The GUI L/D estimator calls `whole_missile_LD` directly. In the
trajectory, a no-separation body glider left at the sentinel `glider_LD = 0` has
its value auto-derived once at integration setup (`derive_glider_LD`, at
`GLIDE_MACH_REF = 5`); a separating RV, or any body with an explicit
`glider_LD > 0`, is left untouched. The derive runs at setup, not per step (it is
outside the EOM hot loop).

**L/D during a pull-up maneuver.** The geometry-derived `L/D_max` above is the
*peak* lift-to-drag, available only at the best-glide angle of attack. A
non-separating warhead that pulls up does not fly there. A pull-up commands a
load factor `n` (capped at `glider_pullup_g_max`), so the lift it must generate
is `L = n·m·g`, i.e. a lift coefficient

```
C_L = n·m·g / (q·A_ref) ,     q = ½ρV²
```

The *effective* lift-to-drag at that commanded `C_L` follows from the drag polar
(§12.2.2, `_aero_polar`), whose two coefficients are back-solved from the
vehicle's ballistic coefficient β and its `L/D_max`:

```
C_D0 = m / (β·A_ref) ,    k = 1 / (4·C_D0·(L/D_max)²)
L/D(C_L) = C_L / (C_D0 + k·C_L²)
```

This peaks at `C_L* = √(C_D0/k)`, where it recovers exactly the input
`L/D_max = 1/(2·√(C_D0·k))` (and `C_D* = 2·C_D0`). Pulling harder than `C_L*`
— any load factor above the equilibrium-glide value — drives `C_L` up the
induced-drag branch `k·C_L²`, so the *instantaneous* L/D drops below `L/D_max`:
a steep pull-up trades glide efficiency for turn rate. The command is bounded —
`C_L ≤ C_L_lim ≈ 2·(25°·π/180) ≈ 0.87` (the slender-body `C_L ≈ 2α` at α_max)
and `n ≤ glider_pullup_g_max` — so the worst-case induced-drag penalty is
capped. For a **non-separating** body the `L/D_max` in these formulas is the
geometry-derived value from this section; for a separating RV it is the designed
input. The pull-up arc itself and the guidance modes that drive it are §12.

---

## 9. Guidance laws

In a 3-DOF formulation the missile's attitude is not a state variable —
it is *prescribed* by the guidance law as a unit thrust-direction vector
at each time step. Thrusty implements three guidance backend modes,
exposed in the GUI through a four-item "Ascent Mode" dropdown:

| GUI label | Backend mode | Notes |
|---|---|---|
| Simple pitch profile | `pitch_program` (basic) | Linear pitch to a single burnout angle |
| Advanced pitch profile | `pitch_program` (per-stage) | Per-stage pitch overrides exposed |
| Gravity turn | `true_gravity_turn` | Velocity-aligned thrust (Wright 2020) |
| Orbital insertion | `orbital_insertion` | Two-phase boost + horizontal final-stage burn |

Yaw (dogleg) maneuvers can be layered on top of any mode, either as
mission-level segments or per-stage overrides.

### 9.1 Pitch program (Forden / Levinger / Wright convention)

`_gravity_turn_thrust_dir` (`trajectory.py`) implements the linear
pitch program first introduced by Forden (2007) and refined by
Levinger / Wright. The elevation angle `θ(t)` above local horizontal is:

```
Phase 1   (0 ≤ t ≤ t_start):              θ(t) = θ_start
Phase 2   (t_start < t < t_stop):         θ(t) = θ_start − (t − t_start)/(t_stop − t_start) · (θ_start − θ_burnout)
Phase 3   (t ≥ t_stop):                   θ(t) = θ_burnout
```

`θ_start` is typically 90° (vertical) for the first stage; for upper stages
it inherits the previous stage's burnout angle. `θ_burnout` is the
user-specified burnout elevation, normally chosen to match the Wheelon
optimal angle (Section 10.2) for maximum range.

The commanded thrust unit vector in ENU is

```
û_thrust = cos θ · sin ψ · ê_E + cos θ · cos ψ · ê_N + sin θ · ê_U
```

where `ψ` is the commanded azimuth from the yaw program (Section 9.5).
The ENU vector is then transformed back to ECEF for the EOM.

### 9.2 True gravity turn (Wright 2020 convention)

`_true_gravity_turn_thrust_dir` (`trajectory.py`) implements a
velocity-aligned thrust direction with an optional tilt angle `η` (degrees
below velocity vector). At each step:

```
v̂           = velocity / |velocity|
n̂_perp      = (r̂ − (r̂ · v̂) v̂) / |·|              ("above velocity" unit vector)
û_thrust    = cos η · v̂  −  sin η · n̂_perp
```

`η > 0` tilts thrust *below* the velocity vector, which actively lowers
the flight-path angle γ — matching Wright's `etad` convention.

A near-vertical singularity (`r̂ ‖ v̂`) is handled by falling back to a
trajectory-plane basis defined by the launch azimuth. During the initial
liftoff window (`t < 4 s` or `|v| < 50 m/s`) thrust is held along
local-up to avoid the `v̂` singularity at zero velocity.

By default `η = 0` throughout the burn, giving a pure (velocity-aligned)
gravity turn. The per-stage `stage_burnout_angle_deg` field is reused as
`η` for the true gravity turn mode — non-zero values produce a controlled
pitch-over below pure velocity tracking (Section 9.5).

### 9.3 Orbital insertion (two-phase)

`_orbital_insertion_thrust_dir` (`trajectory.py`) implements a
two-phase program designed for space-launch vehicles:

```
Phase 1  (all stages before the final, t < t_final_ignition):
            Linear pitch from 90° at t = 0 down to boost_angle_deg
            over the window [t_start, t_stop], then hold boost_angle_deg
            until final-stage ignition.

Phase 2  (final stage, t ≥ t_final_ignition):
            Hold 0° (horizontal) so the final stage adds horizontal
            velocity at the boost-arc apogee until orbital energy
            cutoff fires.
```

Engine cutoff for the final stage is *not* time-based — it is commanded
by an event function that monitors the specific orbital energy

```
ε(t) = ½ |v_ECI|² − μ/r
```

against the target circular-orbit energy `ε_target = −μ/(2 r_target)`.
When `ε(t) = ε_target` the engine cuts off. This makes the trajectory
self-correcting: the burn ends when the orbit is closed, regardless of
small thrust or mass-flow errors. Solid stages cannot be commanded off,
so they burn to natural propellant exhaustion; the orbital-energy
condition only applies to liquid final stages.

The companion planner `plan_orbital_insertion` (`trajectory.py`)
searches automatically for the `boost_angle_deg` that places the perigee
closest to the requested orbit altitude, returning success/perigee/apogee
in a single call.

### 9.4 Per-stage pitch overrides

When "Advanced pitch profile" is selected, each stage exposes three
override fields (`stage_turn_start_s`, `stage_turn_stop_s`,
`stage_burnout_angle_deg`) that take priority over the missile-level
schedule. This is how the built-in missiles replicate published
boost-phase pitch programs — for example, a three-stage ICBM whose
documented profile has different turn-start/stop times in stages 1, 2,
and 3 sets each stage's overrides independently.

Per-stage overrides apply to all three guidance modes:

| Mode | Per-stage meaning |
|---|---|
| `pitch_program` | Three-phase linear pitch with stage-specific start/stop/end angles |
| `true_gravity_turn` | `stage_burnout_angle_deg` is reused as η (degrees below velocity); start/stop bracket the active η window |
| `orbital_insertion` | Override the boost-angle schedule for that stage |

Unset overrides default to the mission-level schedule. The flag for
"this stage uses overrides" is the presence of all three fields; if any
is `None` the global schedule applies.

### 9.5 Yaw program (dogleg maneuvers)

`_yaw_program` (`trajectory.py`) implements a multi-segment azimuth
schedule. Each segment is a tuple `(start_s, stop_s, final_az_deg)`
giving start and end times (mission-elapsed) and the commanded final
azimuth. Between segments the commanded azimuth interpolates linearly
from the previous segment's ending value:

```
For t in segment k with previous azimuth ψ_{k-1} and target ψ_k:
   ψ(t) = ψ_{k-1} + (t − start_k)/(stop_k − start_k) · (ψ_k − ψ_{k-1})
```

This is the model for dogleg maneuvers used by ICBMs and SLVs to adjust
the post-staging ground track without changing the launch azimuth.
Per-stage yaw fields (`stage_yaw_start_s`, `stage_yaw_stop_s`,
`stage_yaw_final_az_deg`) take priority over the mission-level list when
present. The commanded `ψ(t)` feeds the pitch-program's azimuth slot
(Section 9.1) at every EOM step; pitch (elevation) and yaw (azimuth)
combine in the ENU resolution via the standard spherical-coordinate
basis projection, with elevation θ and azimuth ψ as independent
parameters.

---

## 10. Range optimisation and targeting

### 10.1 Tsiolkovsky stack ΔV (pre-estimate)

The total vacuum ΔV available from the stage chain is

```
ΔV_total = Σ_stages  I_sp · g₀ · ln( m_initial / m_burnout )
```

(`_tsiolkovsky_dv`, `trajectory.py`). This is used as a fast estimate
of achievable burnout speed before the range-maximisation search — it
costs no integration calls and provides a tight bound for narrowing
the search window. The empirical relation

```
v_burnout ≈ 0.82 · ΔV_total − 300 m/s
```

calibrates the ideal ΔV against typical gravity-drag losses for
ballistic boost profiles. For SLVs the SLV-specific Schilling penalty
(Section 11) is used instead.

### 10.2 Wheelon optimal burnout angle

For a given burnout speed `v_bo` at altitude `h_bo`, the burnout
elevation angle that maximises ballistic range on a spherical Earth is
(Wheelon):

```
Q      = v_bo² / (g_bo · r_bo)        r_bo = R_E + h_bo,  g_bo = GM/r_bo²
γ_opt  = ½ · arccos( Q / (2 − Q) )
```

(`_wheelon_gamma_opt`, `trajectory.py`). The dimensionless ratio
`Q` is bounded above by 1 (orbital velocity); for sub-orbital
trajectories Q < 1 and γ_opt is well-defined and lies between 0° (Q = 1,
flat orbit) and 45° (Q → 0, throw the rock from a tower).

Used as the centre of a coarse-grid search window
`γ ∈ [γ_opt − 10°, γ_opt + 10°]`. This roughly two-thirds reduction in
search volume relative to a 5°–80° unbounded scan brings the
range-maximisation cost down to a level where a parallel grid search
finishes in seconds for typical missiles.

### 10.3 Range maximisation algorithm

`maximize_range` (`trajectory.py`) is a two-phase search over
(burnout angle, turn-stop time):

1. **Coarse parallel grid.** A grid of candidate `(γ, t_stop)` pairs
   is evaluated on a thread pool (up to 8 workers, capped to avoid
   hyperthreading thrash). The angle window is bounded by Wheelon
   (Section 10.2). The turn-stop window covers the powered-flight
   duration, with a 3600 s outer cap to bail out on degenerate cases
   that never impact.
2. **Bounded polish.** The best coarse candidate is refined by a single
   `scipy.optimize.minimize_scalar(method='bounded')` pass (bounded Brent
   over an interval) on the burnout angle, with the turn-stop fixed at the
   coarse-grid optimum.

The result dictionary returns the maximum range plus the optimal
`(burnout_angle, turn_stop)` and the full trajectory at the optimum.
A `cancel_event` parameter allows GUI cancellation between coarse-grid
evaluations.

### 10.4 Aim at target

`aim_missile` (`trajectory.py`) finds the engine cutoff time that
produces a specified range. With burnout angle held fixed, range is a
monotonic function of cutoff time (more burn ↔ more range), so Brent's
method on the scalar `cutoff_time` is well-behaved:

```
cutoff_time = brentq( range(cutoff) − target_range, lo=5 s, hi=t_burn_total )
```

`xtol = 1 s`, `maxiter = 50`. If the target range is outside the feasible
envelope (either because the missile lacks the energy or because even
zero-burn ballistic flight overshoots), the function returns `t_burn_total`
and the caller observes the resulting overshoot.

For a target *point* (lat/lon) rather than a target range, the caller
first computes the great-circle distance using Vincenty (Section 2.3),
then aims for that range.

### 10.5 Find range

`find_range` (`trajectory.py`) is a trivial wrapper around
`integrate_trajectory` that returns only the range; used by GUI calls
that don't need the full trajectory dictionary back.

---

## 11. SLV performance estimation

For space-launch vehicles the question is *can this rocket put a given
payload into a given orbit*, not *how far does it fly ballistically*.
This is answered by the Schilling method, an algebraic ΔV budget that does
not require trajectory integration (`slv_performance.py`).

### 11.1 The Schilling method

The available ΔV from a stack of stages is the Tsiolkovsky sum
(Section 10.1). The required ΔV to reach a target orbit is

```
ΔV_req  = V_inj  +  ΔV_pen  −  V_rot
V_inj   = √( μ · (2/r_p − 1/a) )                vis-viva at perigee
a       = (r_p + r_a) / 2                       semi-major axis
V_rot   = R_E · Ω · cos(lat) · sin(azimuth)     Earth-rotation assist
```

`V_inj` is the inertial-frame injection speed at perigee from the
vis-viva equation; for a circular orbit this collapses to
`√(μ/r)`, the circular orbital speed. For an elliptical transfer the
formula gives the perigee speed of the target orbit. `V_rot` is the
eastward ground-frame velocity at the launch site, which is "free" — a
launch toward the east adds Earth's rotation to the inertial speed at no
propellant cost. With azimuth measured clockwise from north, `sin(azimuth)`
picks out the eastward component (maximal due-east, zero for a polar
launch).

`ΔV_pen` is the empirical Schilling penalty that bundles gravity
losses, drag losses, and steering losses into a single function of
ascent time and perigee altitude. The formulae are
([Schilling 2009](#16-references)):

```
T_3s    = 3 · [1 − exp(−0.333 · ΔV_avail / (g₀ · I_sp_avg))] · g₀ · I_sp_avg / A₀
T_mix   = 0.405 · T_actual  +  0.595 · T_3s
ΔV_pen  = K_3 + K_4 · T_mix
K_3     = 429.9 + 1.602 · H_p + 1.224×10⁻³ · H_p²       (H_p in km)
K_4     = 2.328 − 9.687×10⁻⁴ · H_p                       (H_p in km)
```

`T_3s` is a notional 3-stage ascent time, `T_actual` is the real burn
time of the stack, and `T_mix` is the empirical blend Schilling found
to fit the calibration data best. `H_p` is the perigee altitude in km.

The vehicle has *positive ΔV margin* (i.e. can reach the target
orbit with the stated payload) iff `ΔV_avail > ΔV_req`.

### 11.2 Maximum payload — binary search

Payload enters the budget on only one side: it lowers every stage's
burnout mass and therefore `ΔV_avail` (through the Tsiolkovsky sum). The
ascent-time term `T_actual` in the penalty is held fixed at the stack's
nominal burn time, so `ΔV_req` does **not** move with payload — no outer
fixed-point iteration is needed.

The implementation is therefore a single binary search on payload between
0 and a fixed upper bound of 200 000 kg, bisecting on the sign of the
margin `ΔV_avail(payload) − ΔV_req` for 50 iterations, which converges to
sub-kilogram precision.

### 11.3 Accuracy and use cases

Schilling reports ~260 m/s RMS error in total mission ΔV calibrated
against historical launches, equivalent to ~10 % error in payload
capacity. This is sufficient for the questions the method is designed
to answer:

- *Is the claimed orbit feasible with the claimed payload?*
- *What is the maximum payload for this orbit and azimuth?*
- *How sensitive is payload to perigee altitude / azimuth / launch latitude?*

The algebraic approach is much faster than trajectory integration
(milliseconds vs. seconds) and guidance-law-agnostic — you don't need
to know the actual pitch program. For questions that need trajectory
shape (peak altitude, burnout velocity vector, range, debris arcs of
spent stages) the 3-DOF integrator (Sections 5–10) is the right tool.

The natural division of labour for a complete SLV analysis:

| Question | Tool |
|---|---|
| Can the full SLV reach the claimed orbit with the claimed payload? | Schilling (Section 11) |
| What does the stripped-down booster do as a ballistic missile? | 3-DOF simulation (Sections 5–10) |

---

## 12. Hypersonic glide vehicles

A boost-glide weapon is a re-entry vehicle (RV) that, after separating from
its booster on a depressed or lofted ballistic trajectory, re-enters the
upper atmosphere, executes a pull-up maneuver, and then *glides* at
hypersonic speed for thousands of kilometres before terminal dive on the
target. Compared to a ballistic RV, the glider trades range for lower
flight altitude, in-flight cross-range manoeuvring, and a flight profile
that bypasses exo-atmospheric defences.

The HGV machinery in Thrusty has six interlocking pieces, each documented
in its own subsection below: a state-machine latch that decides when the
vehicle is in "glide mode"; two aerodynamic models (constant L/D and a
drag polar); five guidance modes spanning a spectrum of phugoid
suppression; a bank-to-turn cross-range model; an inverted terminal dive;
and a JSON-based RV library for shipping or extending vehicle
definitions.

The trajectory machinery and physical conventions follow
[Tracy & Wright 2020](#16-references) and
[Acton 2015](#16-references); the five guidance modes implement (i) the
Tracy & Wright equilibrium-glide ansatz, (ii) the Acton three-phase
analytic pull-up, (iii) a pure phugoid skip-glide, (iv) a Thrusty-original
hybrid that lets the vehicle skip phugoidally for a user-set number of
cycles before settling into equilibrium glide, and (v) a damped-phugoid
glide ([Lu 2013](#16-references)) that fills the realistic physical middle
of the spectrum with a single damping knob ζ. The four **core** models —
ballistic, `skip_glide` (undamped phugoid, ζ = 0), `damped_glide` (a
guided pull-up plus a few decaying skips, ζ ≈ 0.7), and
`equilibrium_glide_acton` (Acton non-oscillatory capture, ζ → ∞) — form
one physical spectrum in how strongly the re-entry phugoid is damped;
`equilibrium_glide` (Tracy) and `skip_to_equilibrium` are retained as
**legacy** comparison modes. See [`DAMPED_GLIDE.md`](DAMPED_GLIDE.md) and
[`DAMPED_GLIDE_MEMO.md`](DAMPED_GLIDE_MEMO.md) for the full damped-glide
derivation.

### 12.1 The pierce-altitude latch and glide-mode state machine

Glide-mode aero forces are *not* active throughout the flight; they
activate only when a specific re-entry condition is met. The latch is the
`_gl_above_pierce` flag and the `_glider_active` gate, both evaluated each
step inside the EOM (`trajectory.py`); `effective_rv()` (`missile_models.py`)
separately selects *which* RV is the terminal vehicle (glider, separating
warhead, or maneuvering body):

```
glide_mode = ( RV has separated
            ∧ glider_enabled
            ∧ glider_LD > 0
            ∧ has-crossed-up-then-down-through ACTON_PIERCE_ALT_M )
```

The pierce altitude is

```
ACTON_PIERCE_ALT_M = 100_000.0          # trajectory.py
```

— the conventional 100 km Kármán-line value used by Acton 2015 (p. 204) as
the start of his Phase 3 (direct re-entry). The latch (`_gl_above_pierce`
at `trajectory.py`) requires the vehicle to have first crossed *up*
through 100 km, then *back down* through 100 km on descent. Until both
crossings have occurred, the active "RV" exposed to the EOM is the
non-glider RV (high-β, zero-lift), so a depressed-trajectory ballistic
flight that never exceeds 100 km is never mistreated as an HGV re-entry.

This is a deliberate design choice. Many real boost-glide concepts (e.g.
the AHW endo-atmospheric profile noted in Acton 2015 p. 194) launch on a
highly depressed trajectory that never crosses 100 km, and Acton's
analytic model genuinely does not apply to those trajectories — they need
a different (direct-injection) treatment. Forcing the user to model these
explicitly rather than auto-classifying any descending RV as a glider
prevents silent misuse.

### 12.2 Aerodynamic models

Two models are available for the glide phase, set on each RV by
`glider_aero_model ∈ {'constant_LD', 'polar'}`:

#### 12.2.1 Constant L/D (default)

Lift and drag are linked by a fixed ratio:

```
L = D · (L/D)
```

The drag itself is computed from the standard ballistic-coefficient form
`D = q·m / β`, with `β = m / (C_D · A)` stored as `beta_kg_m2` on the RV.
This is the model Tracy & Wright 2020 use throughout their analysis with
`L/D = 2.6` and `β = 13 000 kg/m²` for an HTV-2-class glider.

Default for new RV definitions and the convention used by all the shipped
RV-library entries unless overridden.

#### 12.2.2 Drag polar

An optional higher-fidelity model uses an explicit drag polar

```
C_D(C_L) = C_D0 + k · C_L²
```

with zero-lift drag coefficient `C_D0`, induced-drag factor `k`, and
reference area `A_ref` taken from the RV's `glider_CD0`, `glider_k_polar`,
and a derived reference area (`_aero_polar`, `trajectory.py`). The
drag polar lets the EOM see induced-drag rise as the guidance commands
larger C_L — important for steep pull-ups or aggressive cross-range
turns, where the constant-L/D approximation under-estimates drag.

In polar mode the guidance trims to a particular `C_L` rather than a
particular L/D:

- *Equilibrium-glide modes*: solve for the `C_L` that satisfies
  `L · cos σ = m·(g − v²/r)` (Tracy Eq. 7 with a bank-angle factor), then
  compute `C_D` and the corresponding drag from the polar.
- *Phugoid modes*: fly at the max-L/D angle of attack, where
  `C_L* = √(C_D0 / k)` and `C_D* = 2·C_D0`. Lift then scales linearly with
  q.

A `C_L` cap of `2 · (25°·π/180) ≈ 0.873` is applied in both modes
(`_C_L_lim` at `trajectory.py`), representing the slender-body
small-angle relation `C_L ≈ 2α` evaluated at α_max = 25°. At 25° the
linearization is starting to lose validity — the exact Newtonian value
`sin 2α` gives 0.766 — so this is best read as a conservative upper bound
on the trim solution rather than a precise aerodynamic limit.

### 12.3 Guidance modes

The `glider_guidance` field on each RV selects one of five guidance laws,
exposed by the GUI dropdown labelled "Glider guidance":

| GUI label | `glider_guidance` value | Origin | Phugoid damping |
|---|---|---|---|
| Phugoid / skip-glide | `skip_glide` | Sänger / classical skip-glide | ζ = 0 (undamped) |
| Damped-phugoid glide | `damped_glide` | Lu 2013 (Thrusty default) | ζ ≈ 0.7 |
| Equilibrium glide (Acton) | `equilibrium_glide_acton` | Acton 2015 | ζ → ∞ (limit) |
| Equilibrium glide (Tracy) | `equilibrium_glide` | Tracy & Wright 2020 | steady (legacy) |
| Skip → equilibrium (auto-handoff) | `skip_to_equilibrium` | Lewis (Thrusty-original) | N skips then steady (legacy) |

The modes form a spectrum of *how aggressively the guidance suppresses
phugoid amplitude*, ordered above by damping ratio ζ: from "let it ride at
max-L/D α*" (`skip_glide`, undamped phugoid, ζ = 0), through the tunable
middle (`damped_glide`, a guided pull-up plus a few decaying skips,
ζ ≈ 0.7), to "fully suppress it" (Acton and Tracy, analytic capture with
mild or no residual phugoid, ζ → ∞). `skip_to_equilibrium` is the discrete
cousin of `damped_glide` — "let it ride for N skips, then suppress."
Atmospheric drag provides natural damping to all of these — they are
bounded oscillations, not unbounded ones.

#### 12.3.1 Equilibrium glide (Tracy)

The simplest mode. At the pierce point (100 km on descent) the vehicle
is treated as already in equilibrium glide; no pull-up arc is modelled.
The EOM is integrated with constant L/D (or the polar trim) and a single
β throughout the glide phase. The integrator's natural dynamics keep the
vehicle near equilibrium because Tracy's Eq. (7) is satisfied at the
pierce point by construction (`trajectory.py–112`):

```
L · cos σ = m · (g − v² / r_e)              [Tracy 2020 Eq. (7)]
```

The equivalent statement: lift balances *gravity minus centripetal*. In
the EOM this appears as the trim condition (`trajectory.py`):

```python
_g_perp = g_mag - speed * speed / r_mag
lift_mag = min( rv_mass * _g_perp / cos(bank),
                drag_mag * glider_LD,
                _erv.glider_pullup_g_max * g_mag * rv_mass )
```

Three caps act in sequence:

1. **The Tracy analytic value** `m · g_⊥ / cos σ` — exactly Eq. (7) with
   the bank-angle factor.
2. **Aerodynamic supply cap** `D · L/D` — what the wings can actually
   deliver given the local dynamic pressure. Without this cap the
   analytic lift term would still be applied when q is negligible (high
   altitude, thin atmosphere), locking the vehicle at the handoff
   altitude and producing unrealistically long range (`trajectory.py`
   comment). With it, when `v < v_eq` and the analytic g_⊥ exceeds what
   the aero can supply, the vehicle descends naturally to denser air.
3. **Structural cap** `n_max · m · g` — the user-set maximum pull-up load
   factor (`glider_pullup_g_max`, default 10 g; settable per RV).

The full Tracy & Wright 3D EOM (their Eqs. 1–6) is recovered by
combining this lift trim with the drag and the standard ECEF gravity
field of Section 5. Maneuvering is by bank angle σ (Section 12.4); the
inverted dive is σ = π (Section 12.5).

The result is a *mild damped phugoid* oscillation about the equilibrium
altitude, matching the small oscillations visible in Figure 4 of Tracy
& Wright 2020 (their note: *"Minor oscillations about the equilibrium
flight altitude, called phugoid motion, result from the dynamics of this
process. These could be damped by active control of the vehicle."*). The
equilibrium trim is the closest the 3-DOF formulation can get to "active
control" without modelling explicit roll-rate dynamics.

#### 12.3.2 Equilibrium glide (Acton)

Same equilibrium-glide EOM as Tracy mode, but with an explicit *pull-up
arc* bridging the descent from the 100 km pierce point down to the
equilibrium-glide start altitude. Implements Acton 2015's three-phase
structure (his §"Pull-up Phase"):

| Phase | Span | Aero |
|---|---|---|
| 3. Direct re-entry | h₂ = 100 km → h₃ | Large angle-of-attack, drag-only, β = β_S |
| 4. Pull-up arc | h₃ → h₄ = h_eq | Smooth analytic arc, gradual rotation |
| 5. Equilibrium glide | h_eq → terminal | Constant L/D and β = β_L (Tracy mode) |

Acton's central simplification is that the pull-up is *deliberately
smooth and non-oscillatory*: the vehicle gradually rotates from high-α
direct-re-entry to low-α equilibrium glide such that ρ/β stays constant
through the pull-up (Acton Eq. 8):

```
ρ(h₃) / β_S  =  ρ(h_eq) / β_L                       [Acton 2015 Eq. (8)]
```

This is the rationale for the on-RV parameter `glider_beta_entry_kg_m2`
(β_S) on top of the glide-phase `beta_kg_m2` (β_L). The code uses
Acton's isothermal-atmosphere fit (his p. 197) over the relevant
altitude band:

```
ACTON_SCALE_HEIGHT_M  = 6970.0                      # trajectory.py
ACTON_SEA_LEVEL_RHO   = 1.46                        # trajectory.py
```

so that h_eq follows from the L/D, β_L, and pierce velocity by
inverting Acton's equilibrium-altitude formula
(`h_eq = H · ln(ρ_0 · r_e / (2·β_L · L/D) · v² / v_e²)`, his Table 1
equilibrium-glide column).

The Phase 3 → Phase 4 transition altitude h₃ is then fixed by Eq. (8):
`h₃ = H · ln(ρ_0 · β_S / β_L · exp(h_eq / H))`. The Phase 4 arc itself is
the analytic circular arc fitted between `(h₃, γ = −θ₂)` at the start
and `(h_eq, γ = 0)` at the end (Acton's small-angle Eqs. 18 / 21):

```
R = (h₃ − h_eq) / (1 − cos θ₂)
h(θ) = h₃ − R · (cos θ − cos θ₂)
```

implemented by `_acton_pullup_arc` (`trajectory.py`). The arc is
applied as a one-shot state reset at the Phase 3 → Phase 4 boundary
(detected by the descending crossing of h₃ via the event function
`_make_phase3_end_event`, `trajectory.py`), after which equilibrium
glide proceeds exactly as in Tracy mode.

Implementation detail: Acton mode falls back to Tracy mode if
`glider_beta_entry_kg_m2 ≤ 0`, since without a positive β_S the Phase 3
direct-re-entry segment is ill-defined. This guarantees the user always
gets an analytical pull-up rather than a phugoid in cases where the
small-β data is missing (`trajectory.py`).

#### 12.3.3 Phugoid / skip-glide

In this mode the guidance does *not* trim to suppress oscillation.
Instead the vehicle flies at the max-L/D angle of attack throughout, so
lift is proportional to dynamic pressure and the natural phugoid
oscillation about the equilibrium altitude is preserved at full
amplitude (`trajectory.py`):

```python
# skip_glide / phugoid: fly at max-L/D AoA (α*)
# so lift ∝ q and the natural phugoid oscillation
# is preserved.  C_L* = √(C_D0/k); C_D* = 2·C_D0.
```

In constant-L/D mode this collapses to `L = D · (L/D)` with no
equilibrium trim, so the lift force is set by whatever density and speed
the vehicle finds itself at — the classic phugoid restoring mechanism.

This is the "skip glider" of classical re-entry literature (e.g. Sänger,
Chapman 1958). Real boost-glide vehicles avoid this mode because the
large oscillations carry the vehicle in and out of severe heating
regimes; Acton explicitly notes (2015 p. 195) that DARPA's design
schematics show the glider "bouncing just once during the pull-up"
rather than continuing in unsuppressed phugoid. The mode is included
for pedagogical completeness, for true skip-vehicle modelling, and as a
sanity check against the equilibrium-glide modes.

The oscillations are *bounded and damped* — atmospheric drag removes
energy on every cycle (the up-swing into thinner air loses less than the
down-swing into denser air), and the amplitude decays over many cycles.
Bounded does not mean small: the first-cycle peak-to-trough altitude
swing can be tens of kilometres for a vehicle entering at v ≈ 6 km/s.

#### 12.3.4 Skip-to-equilibrium (Lewis)

A Thrusty-original hybrid that bridges the gap between Acton's idealized
smooth pull-up and the unsuppressed phugoid of skip-glide. The vehicle
flies in `skip_glide` (phugoid) mode for a user-specified number of
*upward crossings of the equilibrium-speed curve*, then transitions
one-way to `equilibrium_glide` (Tracy) mode for the remainder of the
flight (`trajectory.py`).

The skip count `glider_skip_count` (default 1; GUI spinbox 1–10) gates
the handoff. With `N = 1` the vehicle pulls up phugoidally once, then
settles into equilibrium glide on the first descent leg — a close match
to the DARPA HTV-2 schematic that Acton describes. With `N = 2 – 3` the
vehicle skips two or three times before settling, which is closer to
what real, marginally-damped boost-glide vehicles probably do. With
large N the mode approaches pure phugoid skip-glide; with `N = 1` and a
smooth initial trajectory it approaches Acton.

The motivating physical observation: a real boost-glide vehicle is
unlikely to either oscillate forever (skip-glide) or settle perfectly
on the first cycle (Acton's idealization). Some bounded number of
damped phugoid cycles before quasi-equilibrium is a more realistic
intermediate picture. The skip-to-equilibrium mode captures this
without requiring a full 6-DOF model with explicit roll-rate dynamics.

At the handoff the milestone logger emits a *"Skip-to-equilibrium
handoff"* event, and the trajectory dictionary's `flight_events` records
the time, altitude, and speed of the transition. The handoff itself is
implemented as a one-way mode flag flip — the EOM continues integrating
the same state vector, only the lift-trim rule changes (`trajectory.py:
2556`).

#### 12.3.5 Damped-phugoid glide (Lu)

The `damped_glide` mode (the Thrusty default for new glide RVs) fills the
physical middle of the spectrum between undamped `skip_glide` (ζ = 0) and
the analytic non-oscillatory capture of Acton (ζ → ∞). It reproduces what
a *guided* hypersonic glider actually does — a pull-up plus a few
**decaying** skips settling into equilibrium glide — with the number of
skips emerging from a single damping knob rather than a hand-set integer.

The vehicle flies the max-L/D trim angle α\* (identical to `skip_glide`)
plus a feedback term proportional to the altitude-rate error, which bleeds
energy out of the phugoid ([Lu 2013](#16-references) Eq. 33; equivalently
Yu & Chen 2011 Eq. 19):

```
L·cos σ_cmd = L·cos σ_nom − k_h·(ḣ − ḣ_eq)
```

with `ḣ = V·sin γ` the current altitude rate and `ḣ_eq = V·γ*` the
command, where `γ* = −2·H_ρ·g / (V²·cos σ·(L/D))` is the
quasi-equilibrium-glide flight-path angle (Lu Eq. 31 — L/D in the
denominator, so higher L/D glides shallower; matches Vinh, Coppola &
de-Olivé Ferreira 1996 and the classic γ = −1/(L/D)).

**The gain — derived, not fitted.** Linearising the planar equilibrium-glide
EOM about equilibrium gives, from first principles, a harmonic oscillator for
the altitude perturbation. At fixed angle of attack lift ∝ ρ ∝ e^(−h/H_ρ), so a
displacement δh changes the lift acceleration by `d a_L/dh = −g_eff/H_ρ`, giving
the open-loop mode

```
δḧ + ω_p²·δh = 0,    ω_p² = g_eff / H_ρ   (g_eff = g − V²/r)
```

where `H_ρ = −ρ/(dρ/dh)` is the local density scale height (the restoring force
is the density lapse). This linearisation is the second-order entry theory. The
primary source is Chapman (NACA TN 4276 / NASA TR R-11), who reduces the planar
entry equations to one second-order nonlinear ODE (Eq. 21) in a density-like
variable `Z(ū)`; its truncation neglecting vertical acceleration is the
equilibrium glide (his `Z_II` solution, attributed to Sänger), and the full
equation produces, for higher L/D, the oscillation he calls "numerous skips of
sizable intensity" (Fig. 6). Yaroshevskii's equation (Vinh, Busemann & Culp,
Ch. 10, Eq. 10-55) is a special case, with the same Sänger equilibrium glide
(Eq. 10-61) and the same numerically-shown oscillation (Fig. 10-10). The
oscillator above is the linearisation of that ODE about equilibrium glide; all
three sources show the oscillation but none writes the closed-form oscillator.
The frequency is also corroborated empirically by [Liu et al. 2025](#16-references)
(measured skip phugoid 0.021–0.037 rad/s). The altitude-rate feedback adds the `2ζω_p·δḣ`
damping term; matching the feedback contribution to `2ζω_p` fixes the gain for a
target damping ratio ζ:

```
k_h = 2·ζ·m·√(g_eff / H_ρ)
```

The gain is recomputed each integration step from the current state, so it
schedules down naturally as V and g_eff change (matching Lu's
velocity-scheduled gain, his Eq. 34).

**Why ζ = 0.7.** ζ is the single user knob (`glider_damping_zeta`, default
**0.7**). The 0.7 default is the classical second-order control value: it
sits in the desirable ζ = 0.4–0.8 band that the standard control texts
recommend for transient response — below 0.4 yields excessive overshoot,
above 0.8 responds sluggishly ([Ogata 2010](#16-references) §5-3, p. 171;
[Franklin, Powell & Emami-Naeini 2019](#16-references) §3.4.2 / Fig. 3.24,
which lists ζ = 0.7 → ~5 % overshoot as a "frequently used value"). The
first overshoot is `M_p = exp(−πζ/√(1−ζ²))` ≈ 4.3 % at ζ = 1/√2 (~5 % at
ζ = 0.7; Ogata Eq. 5-21, Franklin Eq. 3.72), and ζ ≈ 0.7 is very nearly
the *settling-time-minimizing* damping (Ogata p. 173 finds t_s bottoms out
near ζ = 0.68–0.76) — so it is the fastest settling without a sluggish
approach. (ζ = 1/√2 is also the "maximally flat" 2nd-order Butterworth
value in the *frequency* domain, but that is a separate characterisation,
not the time-domain transient-response argument above.) ζ = 0.7 is a
modelling choice describing a competently-guided vehicle, not a physical
constant of the airframe; it is freely dialled — ~0.3 gives several lazy
skips, ≥ 1.0 collapses to a smooth equilibrium capture. In short, ζ is a
property of the *guidance*, not the airframe — the bare vehicle's skip
phugoid is essentially undamped and is suppressed only by active control
([Tracy & Wright 2020](#16-references)) — so the designer chooses it,
dialling in more damping for a smoother, single-bounce capture
([Acton 2015](#16-references)) or less for a bouncier, longer-skipping
profile, with 0.7 the conventional well-guided midpoint
([Ogata 2010](#16-references); [Franklin et al. 2019](#16-references)).

**Nesting (the safety property).** ζ = 0 ⇒ k_h = 0 ⇒ the feedback term
vanishes ⇒ the lift law is *exactly* `skip_glide`. This is verified
bit-exact (`max|Δaltitude| = 0.000000 km` over a full integration, both
aero models) in `damped_glide_smoke_test.py`. Large ζ drives the
trajectory onto equilibrium glide, so `damped_glide` continuously
interpolates between the two existing endpoints.

**Validation.** Flying the repo's C-HGB glide body on a sub-circular
(MRBM-class) boost, entering at ~5.6 km/s:

| mode | fraction of glide above 100 km | range |
|---|---:|---:|
| `skip_glide` | 57 % (skips out of the atmosphere) | 2445 km |
| `damped_glide` ζ = 0 | 57 % (bit-identical to skip_glide) | 2445 km |
| `damped_glide` ζ = 0.7 | 14 % (glides in the atmosphere) | 6187 km |
| `equilibrium_glide` | 27 % | 6246 km |

Damping at ζ = 0.7 converts a skip that spends most of its flight *above*
the atmosphere into a true in-atmosphere glide, nearly tripling range to
match the analytic equilibrium glide.

**Limits.** Gated, like `skip_glide`, on being below the 100 km pierce
altitude and on dynamic pressure (`q > 1` Pa) — no aerodynamic control in
vacuum. The feedback is disabled when `g_eff ≤ 0` (at/above circular speed
the phugoid restoring force is undefined); the vehicle then flies plain
α\*. `H_ρ` is finite-differenced from the atmosphere and clamped to
4–12 km. Lift is bounded by the existing `glider_pullup_g_max` cap, so the
manoeuvre respects the structural g-limit automatically.

### 12.4 Bank-to-turn (cross-range manoeuvring)

Cross-range manoeuvring is by *bank-to-turn* with roll angle σ
(`bank_rad` in the code, settable via the RV's bank schedule or a target
point). Banking partitions the lift vector between vertical and
horizontal components:

```
F_lift = L · ( cos σ · n̂_up  +  sin σ · n̂_cross )
```

(`trajectory.py`). This is exactly the Tracy & Wright 2020 EOM
(their Eq. 2 with the `cos σ` factor and Eq. 3 with the `sin σ` factor).
Banking simultaneously reduces the vertical lift component to `L · cos σ`
— which forces the vehicle to fly at a lower equilibrium altitude in
denser air, raising drag — and steers horizontally by `L · sin σ`. This
is the source of the cross-range / range trade-off documented in Tracy &
Wright 2020 Figure 6.

A bank-angle schedule (lat, lon, time tuples) can be set directly on the
RV; alternatively, the GUI's "Aim at target" dialog computes the bank
trajectory that delivers the glider to a specified terminal lat/lon.
The dialog uses Brent's method on bank angle σ with the trajectory
integrator as the inner cost function — analogous to the `aim_missile`
function for ballistic flight (Section 10.4).

### 12.5 Terminal dive (inverted)

When the vehicle approaches the target it transitions to a terminal
dive by rolling to σ = π (inverted), which puts the lift force pointing
toward the ground rather than away from it (`trajectory.py–722`).
The dive is the most efficient way to traverse the dense lower
atmosphere while preserving as much speed as possible — matching the
HTV-2 test profile (Tracy & Wright 2020 §"Computational results",
inverted dive at end of glide).

Two trigger conditions are available:

- **Altitude trigger** (`glider_terminal_dive = True`,
  `glider_terminal_alt_km`): when `alt < terminal_alt_km` the dive
  begins. Default 30 km.
- **Range trigger** (`glider_dive_target_radius_km`): when the
  great-circle distance to a specified target lat/lon falls below
  `radius_km` the dive begins, regardless of altitude. This is the model
  for a glider that dives over the target rather than at a fixed
  altitude band.

Once `_dive_now = True`, the bank angle is held at π for the rest of
the flight. The inverted-dive section integrates with the same EOM as
the glide phase but with the lift sign flipped; the vehicle accelerates
slightly under combined gravity and downward lift, traversing 30 km of
atmosphere in a few seconds.

### 12.6 RV library

The reentry-vehicle library is a directory of JSON files
(`rv_library/*.rv.json`), each fully specifying one RV's mass, geometry,
ballistic coefficient, L/D, drag polar, nose-tip radius, emissivity, and
default guidance mode. The shipped library covers a spectrum of
public-domain reference vehicles:

| File | Class | L/D | β (kg/m²) |
|---|---|---:|---:|
| `C-HGB.rv.json` | Common-Hypersonic-Glide-Body | 2.0 | 15 000 |
| `HGB.rv.json` | Generic hypersonic glider (HTV-2-class) | 1.8 | 15 000 |
| `HGB-LD3.rv.json` | Hypothetical high-L/D glider | 3.0 | 10 000 |
| `Generic-RV.rv.json` | Generic ballistic RV (no glide) | 0.0 | 10 000 |
| `Mk21.rv.json` | Mk-21 RV (LGM-30 / LGM-118) | 0.0 | 75 000 |

A user can copy any of these as a starting template, edit the JSON
fields, and drop the file into `rv_library/`; the GUI's RV dropdown
re-scans the directory on startup. This is one of the extension hooks
named in §1.1: the shipped vehicles are a starting set, not a closed
inventory.

The JSON file for an RV contains only the fields the author wanted to
set; anything omitted inherits the `RVParams` default. The shipped
`HGB.rv.json` is representative — twelve fields, no glider-polar
parameters and no Acton-mode β_S:

```json
{
  "name":                    "HGB",
  "mass_kg":                 450,
  "beta_kg_m2":              15000,
  "shape":                   "cone",
  "diameter_m":              0.5,
  "length_m":                2.0,
  "glider_enabled":          true,
  "glider_LD":               1.8,
  "glider_guidance":         "equilibrium_glide",
  "glider_pullup_g_max":     10,
  "glider_terminal_dive":    false,
  "glider_terminal_alt_km":  30
}
```

The full set of fields recognised by the JSON loader, with their
defaults:

| Field | Default | Notes |
|---|---|---|
| `mass_kg` | (required) | Vehicle mass |
| `diameter_m` | 0.5 | Reference diameter |
| `length_m` | (optional) | Used for slender-body checks |
| `shape` | `"cone"` | Nose-shape selector |
| `nose_radius_m` | 0.0 = auto | Stagnation radius for heating (Section 13.1); 0 ⇒ derived from nose shape + diameter, a positive value overrides |
| `emissivity` | 0.85 | Surface emissivity for T_eq calculation |
| `beta_kg_m2` | 10000 | β_L — equilibrium-glide ballistic coefficient |
| `glider_beta_entry_kg_m2` | 0.0 | β_S — Acton direct-re-entry β; 0 disables Acton mode |
| `glider_enabled` | `false` | Master toggle for HGV machinery |
| `glider_LD` | 0.0 | Lift-to-drag ratio |
| `glider_aero_model` | `"constant_LD"` | Or `"polar"` (Section 12.2) |
| `glider_CD0` | 0.05 | Zero-lift drag coefficient (polar mode) |
| `glider_k_polar` | 0.10 | Induced-drag factor (polar mode) |
| `glider_guidance` | `"equilibrium_glide"` | Section 12.3 |
| `glider_skip_count` | 1 | Skip-to-equilibrium handoff count |
| `glider_pullup_g_max` | 10.0 | Structural load-factor cap |
| `glider_terminal_dive` | `false` | Enable inverted terminal dive |
| `glider_terminal_alt_km` | 30.0 | Altitude trigger for terminal dive |

The full schema lives in the module-level loader `rv_from_dict`
(`missile_models.py`).

**Acton mode and β_S.** The shipped JSON RVs leave `glider_beta_entry_kg_m2`
at zero, since β_S (the high-α drag direct-re-entry ballistic coefficient)
is published for only a handful of vehicles. Selecting Acton mode for an
RV with β_S = 0 falls back to Tracy mode (Section 12.3.2). For the
HTV-2-class built-in missiles defined programmatically rather than via the
RV library — `Forden_HTV2` and its variants — the code sets
`glider_beta_entry_kg_m2 = 7.0` based on Acton 2015 Table 3
(`missile_models.py, 1635`). A user wanting to run Acton mode on a
custom RV will need to research and set β_S themselves; this is by design,
since silently fabricating β_S for arbitrary vehicles would be misleading.

The values shipped in the library are starting points calibrated against
public literature (e.g. `C-HGB.rv.json`'s L/D = 2.0 follows the open-source
DoD common-glide-body briefings, and `HGB.rv.json`'s L/D = 1.8 / β =
15 000 kg/m² approximates the HTV-2 class without claiming to be HTV-2);
users should re-verify them against the specific scenario being modelled
rather than treating them as endorsed estimates.

---

## 13. Stagnation heating

For HGV trajectories Thrusty reports a single stagnation-heating
milestone — the peak heat-flux value and the corresponding
radiative-equilibrium wall temperature — at the time during the glide
phase when stagnation heating is maximum. This is the minimal
information needed to compare a simulated trajectory against
public-source claims about peak nose-tip temperatures and material
limits, without taking on the substantially larger modelling burden of
a full transient aerothermal solution.

The model is convective heating only — no radiative gas-cap heating, no
ablation, no wall conduction, no transient temperature. For preliminary
trajectory work these omissions are appropriate; for vehicle thermal
design or material selection they are not, and a different tool
(CBAERO, MINIVER, or full CFD) is the right choice.

### 13.1 Sutton-Graves stagnation heat flux

The convective heat flux at the stagnation point is computed from the
Sutton-Graves correlation
([Sutton & Graves 1971](#16-references), NASA TR R-376):

```
q̇ = K · √(ρ / R_N) · V³                     [W/m²]
```

with `K = 1.7415×10⁻⁴ W·s³/(kg^½·m^(5/2))` for Earth atmosphere,
`ρ` the free-stream density in kg/m³, `R_N` the nose-tip radius of
curvature in m, and `V` the free-stream velocity in m/s
(`trajectory.py`).

The Sutton-Graves correlation is a fit to chemical-equilibrium boundary
layer calculations across a wide range of base gases, enthalpies (2.3 to
116.2 MJ/kg), and pressures. For Earth re-entry conditions it
agrees with the more rigorous Fay-Riddell relation to within ~10 % over
the speed range relevant to HGV glide.

Three implementation conventions:

1. **ECEF velocity, not inertial.** `V` is the vehicle's speed relative
   to the rotating atmosphere (i.e. the airspeed), not the inertial
   ECI speed. At the equator this differs by ~465 m/s; the difference
   matters at HGV velocities of 3–6 km/s. The relevant comment is at
   `trajectory.py`.

2. **Per-RV nose-tip radius.** `R_N` is the active RV's effective
   stagnation radius (`RVParams.effective_nose_radius_m`, used in
   `trajectory.py`), making the stagnation-point formula geometry-aware.
   An explicit `nose_radius_m` is authoritative; when it is 0/absent the
   value is a screening default derived from the nose shape and base
   diameter (`nose_tip_radius`, `missile_models.py`): `R_N ≈ 0.10·R_body`
   for a sharp cone, scaled up for blunter profiles (von Kármán, LV-Haack),
   clamped to [5 mm, R_body]. This is a transparent bluntness heuristic,
   not a geometric tip-curvature — the idealised profiles are all
   geometrically sharp at the tip, and real bluntness is a design choice
   the outer shape does not fix (every shipped library RV is a `cone` yet
   spans 1–5 cm tips). The shipped library sets explicit radii where known
   (HTV-2 1 cm, C-HGB 2 cm, AHW 5 cm). Two RVs at the same speed and
   density see different stagnation fluxes: `q̇ ∝ 1/√R_N`, so a 1 cm nose
   sees about 2.2× the flux of a 5 cm nose. The peak-heating *time* is
   independent of `R_N`; only the peak *magnitude* changes.

3. **Re-entry/descent phase.** Heating is computed for any re-entering
   terminal vehicle — a ballistic RV as well as a glider — over the
   descent arc after the 100 km crossing (Section 12.1; for sub-100 km
   profiles, after apogee). A steep ballistic RV is the high-flux regime
   the survivability FOM most needs to score, so it is no longer excluded.
   Glide-specific milestones (pull-up / glide-start / skips / terminal
   dive) remain gated on a lifting vehicle. Boost and ascent heating is
   not separately reported because for typical operational vehicles the
   boost-phase stagnation flux is much smaller than the descent peak, and
   the booster nose is not the surface that matters for payload survival.

### 13.2 Radiative-equilibrium wall temperature

The peak heat-flux value is converted to a radiative-equilibrium
stagnation-point temperature via the Stefan-Boltzmann law assuming
all incoming convective flux is balanced by surface re-radiation:

```
σ · ε · T_eq⁴ = q̇_peak             ⇒    T_eq = (q̇_peak / (σ · ε))^(1/4)
```

with `σ = 5.670374419×10⁻⁸ W/(m²·K⁴)` (CODATA 2019, Stefan-Boltzmann
constant) and `ε` from the RV's `emissivity` field (default 0.85; set
at `RVParams`, `missile_models.py`). The default is consistent with
[Anderson 2006](#16-references) §18.8 (HERMES emissivity example) and
the upper end of the operational range for reinforced carbon-carbon
(RCC) materials reported by [Williams & Curry 1992](#16-references) in
NASA RP-1289 (RCC ε(T) measurements).

This is a radiative-equilibrium estimate, not a transient temperature.
It assumes the surface has reached steady-state at peak-heating
conditions; in reality the wall temperature lags peak heating slightly
because of the surface's thermal mass. For a thin RCC leading edge the
lag is small (seconds); for a thick ceramic ablator it can be
significant. The radiative-equilibrium value is therefore an
*upper bound* on the actual wall temperature at peak heating — the true
surface is somewhat cooler because it is still warming when peak
flux occurs.

### 13.3 What is not modelled

Several aerothermal effects are deliberately *not* implemented, to keep
the heating model simple and the failure modes obvious:

- **No cone-flank or boattail heating.** Only the stagnation point is
  computed. For a slender HGV the stagnation-point flux is the
  worst-case value; cone-flank heating downstream of the nose is lower
  by a factor that depends on local-flow Reynolds number and pressure
  gradient. If a finer-grained heating breakdown is needed, Tauber's
  1989 sharp-cone heating relation (NASA TP-2914, Eq. 46) or Lees 1956
  blunt-body laminar-heating treatment would be natural extensions —
  but neither is in the current code.
- **No emissivity vs. temperature.** Real materials have temperature-
  dependent emissivity (e.g. RCC's ε rises from 0.78 at 1500 K to
  ~0.88 at 2200 K per [Williams & Curry 1992](#16-references)). The
  code uses a single user-set value across the full trajectory.
- **No radiative gas-cap heating.** At V > ~7 km/s the shock layer
  begins to radiate appreciably, contributing additional heating
  beyond pure convection. This is small for HGV velocities below
  ~6 km/s but becomes significant for steep ballistic re-entry from
  ICBM trajectories.
- **No ablation, no wall conduction.** The radiative-equilibrium
  treatment assumes a steady non-ablating surface. Ablating heat
  shields absorb part of the incoming flux as mass-loss enthalpy,
  reducing T_wall; transient conduction matters for vehicles with
  high-thermal-capacity TPS where the surface is still warming during
  peak heating.

All four omissions can be addressed by post-processing the trajectory
in a more capable aerothermal tool (CBAERO, MINIVER, or CFD); the
trajectory itself is what Thrusty is built to produce, and the
single-point heating estimate provides the validation hook against
published peak-temperature claims.

### 13.4 Output and milestone format

When a glide trajectory completes, the code computes `q̇` at every
glide-phase time step (`trajectory.py`), finds the argmax, and
emits a flight-events row:

```
Peak heating  ({q̇_peak/1e6:.1f} MW/m², T_eq ≈ {T_eq:.0f} K)
```

inserted chronologically into the trajectory's `flight_events` list at
the time of peak heating. This is the only heating-derived output; the
full `q̇(t)` time series is computed internally but not exposed to the
user. Adding a CSV column for `q_dot_W_m2` would be a small extension
if a user wanted the time series for plotting.

---

## 14. Outputs, events, and milestones

Every `integrate_trajectory()` call returns a single dictionary capturing
the full state-time history plus a chronological list of milestones, plus
ancillary data products (debris arcs from spent stages, orbital elements
for vehicles that achieve orbit, commanded pitch / azimuth time series).
The GUI consumes this dictionary to populate the on-screen plots and the
event timeline; the Excel, KML, CSV, Folium, and Cartopy exports all
read from the same structure.

### 14.1 The trajectory dictionary

The canonical return structure (`trajectory.py`):

| Key | Type | Meaning |
|---|---|---|
| `t` | array | Time array (s) |
| `lat`, `lon`, `alt` | arrays | Geodetic latitude (deg), longitude (deg), altitude (m) |
| `speed` | array | Airspeed (ECEF-frame, m/s) — used for drag and heating |
| `inertial_speed` | array | Inertial speed (ECI-frame, m/s) — used for orbital queries |
| `accel` | array | Acceleration magnitude (m/s²) |
| `mass` | array | Vehicle mass time history (kg) |
| `range` | array | Cumulative great-circle range from launch (km) |
| `pos_ecef`, `vel_ecef` | (N,3) arrays | Full ECEF state |
| `orbital` | bool | True if the vehicle achieved bound orbit (no impact) |
| `impact_lat`, `impact_lon`, `range_km` | scalars | Final impact point and total range (None if `orbital`) |
| `apogee_km`, `apogee_lat_deg`, `apogee_lon_deg` | scalars | Apogee summary |
| `time_of_flight_s`, `impact_speed_ms` | scalars | Mission summary |
| `milestones` | list of dicts | Chronological flight events (Section 14.2) |
| `debris_trajectories` | list | Spent-stage and shroud debris arcs (Section 14.3) |
| `orbital_elements` | dict | Six-element Keplerian set, if orbital |
| `pitch_cmd_deg`, `az_cmd_deg` | arrays | Commanded guidance time histories |

Distinguishing airspeed (ECEF) from inertial speed (ECI) explicitly in
two columns turns out to matter in practice — orbital-insertion checks
need ECI for energy and `√(μ/r)` comparisons, while drag and heating
need ECEF because the atmosphere co-rotates with Earth (Section 13.1).
Both are exposed rather than forcing the caller to convert.

### 14.2 Milestone catalogue

Milestones are inserted chronologically into the `milestones` list by
`_insert_chrono()` (`trajectory.py`). Each milestone is a dict with
`{t_s, alt_km, range_km, mass_t, speed_ms, event}` populated by
interpolation onto the trajectory array (`_interp_milestone`,
`trajectory.py`). The exhaustive set of event labels currently
emitted:

| Phase | Event |
|---|---|
| Boost | Stage `N` ignition / Stage `N` empty body — in orbit / Stage `N` empty impact |
| Boost | Shroud jettison / Shroud impact |
| Boost | Booster casing impact (strap-on, after separation) |
| Boost / coast | Apogee (`{alt} km`) |
| Coast | Perigee (`{alt} km`) — for orbital trajectories |
| Coast | Yaw segment start / Yaw segment end (azimuth before → after) |
| Coast | Burnout (each stage's natural exhaustion or commanded cutoff) |
| Glide entry | Re-entry (100 km) — pierce-altitude descent crossing |
| Glide | Pull-up start (`{alt} km`) — Acton mode Phase 4 entry |
| Glide | Glide start (`{alt} km`) — equilibrium-glide handoff (Tracy / Acton) |
| Glide | → Equilibrium glide (`{alt} km`) — skip-to-equilibrium handoff |
| Glide | Skip *N* pull-up (`{alt} km`) — Nth phugoid trough |
| Glide | Skip *N* apex (`{alt} km`) — Nth phugoid crest |
| Glide | Peak heating (`{q̇} MW/m², T_eq ≈ {T} K`) — Section 13 |
| Glide | Max-G (`{n} g`) — peak structural load factor |
| Terminal | Terminal dive (`{alt} km`) — start of inverted dive |
| Terminal | Re-entry query (`{alt} km`) — user-set diagnostic altitude crossing |
| End | Orbital insertion (`{a_km} km × {e}`) — for orbital final stages |
| End | Impact (`{mass} kg`) — final ground impact |

The `Re-entry query` event is a user-controllable diagnostic: setting
`reentry_query_alt_km` causes the integrator to emit a milestone at the
descending crossing of that altitude with the full state at that
instant. Useful for comparing computed vs. published "100 km re-entry
velocity" or "30 km terminal velocity" claims.

### 14.3 Debris arcs

After staging or shroud jettison, each spent body continues on a
ballistic arc until impact, computed with a tumbling-cylinder ballistic
coefficient (`tumbling_cylinder_beta`, `missile_models.py`). This
gives a more realistic descent than treating the spent body as a point
mass:

```
A_end   = π · D² / 4                          [end-on cross-section]
A_side  = D · L                               [broadside cross-section]
A_eff   = (A_end + A_side) / 2                [arithmetic mean]
β_tumble = m / (C_D · A_eff)                  [kg/m²]
```

The arithmetic mean of end-on and broadside projected areas
approximates the time-averaged area for a cylinder tumbling in the
pitch plane. `C_D = 1.0` (the function's default) is representative of
bluff-body turbulent flow ([Hoerner 1965](#16-references)). The function
returns 0 if either length or diameter is zero, so missiles without a
configured shroud or spent-body geometry simply have no debris arcs
computed.

Spent bodies that re-enter compute their own impact point and add a
"Stage N empty impact" or "Shroud impact" or "Booster casing impact"
event to the master milestone list. The full debris-arc time-series is
also stored in `debris_trajectories` for plotting on the same map as
the primary trajectory. Stages that reach orbit are flagged with
"Stage N empty body — in orbit" instead of an impact.

### 14.4 Export formats

Five export targets are exposed through the GUI's File and Cartopy
menus:

| Format | Trigger | Contents |
|---|---|---|
| **KML** | File → Export Trajectory KML | Primary trajectory polyline + milestone markers + debris arcs, opened in Google Earth |
| **CSV** | Save sweep / save scenario buttons | Range vs. burnout-angle sweep tables; per-scenario parameter dumps |
| **Folium HTML** | Cartopy → Open Folium Map | Interactive web map with all trajectories, milestones, and impact ellipses; opens in browser |
| **Cartopy PNG** | Cartopy → Export Cartopy Map | High-resolution flat or globe projection map, publication-ready |
| **Excel XLSX** | File → Save Missile to XLSX | Full missile parameter set in a structured workbook (see Section 14.5) |

The KML output uses a fixed colour scheme — primary trajectory red,
debris arcs grey, impact points white — and embeds altitude as the third
KML coordinate so the trajectory renders three-dimensionally in Google
Earth. Milestone markers are placed at their interpolated lat/lon and
labelled with the event string from Section 14.2.

### 14.5 Excel missile import/export

`missile_xlsx.py` provides a 2-way bridge between the in-memory
`MissileParams` tree and an Excel workbook. Each stage gets its own
worksheet with parameters labelled in the first column and values in
the second; the top sheet holds vehicle-level parameters (name, base
diameter, total stages, glider settings, RV selection). The
`export_missile_xlsx(path, params)` and `import_missile_xlsx(path)`
functions round-trip cleanly — a parameter set exported to XLSX and
re-imported produces an identical `MissileParams` object.

This is the *recommended* mechanism for sharing missile definitions with
collaborators who don't run Python: an analyst can edit cells in Excel
and re-import without touching the codebase. The JSON save format
(`missile_to_dict` / `missile_from_dict`) is the lower-level alternative,
preferred for version control because text diffs are readable.

### 14.6 Auxiliary GUI dialogs

Three secondary dialogs supplement the main trajectory view:

- **FootprintDialog** (`thrusty.py`). Sweeps the bank-angle schedule
  for an HGV across a range of cross-range maneuvers and computes the
  envelope of reachable terminal points. Output: an impact-zone polygon
  rendered on the map.
- **RangeRingDialog** (`thrusty.py`). Draws great-circle range
  rings at user-specified distances from the launch site or any other
  reference point. Useful for visualising the launch vehicle's
  performance envelope against named geographic features.
- **Thrust estimator**. Given mass-at-liftoff and an observed time to
  reach a sighted altitude milestone (e.g. "the missile crossed 30 km at
  t = 35 s"), the estimator back-computes the average thrust and Isp
  consistent with the observation. This is a forensic / OSINT tool for
  reconstructing motor parameters from telemetry-deprived launch
  observations.

---

## 15. Validation and built-in missile definitions

Thrusty does not ship with a formal regression-test suite — the only
automated tests in the repository are the two Mars-atmosphere smoke
scripts (`mars_smoke_test.py`, `mars_smoke_test2.py`) noted in §4.4.
Validation has instead been performed informally by reproducing
published trajectory parameters from the open arms-control literature
and confirming that the simulator's output matches the published flight
profiles within the modelling fidelity of a 3-DOF point-mass code.

The twelve builder functions in `missile_models.py` represent the
qualitative validation set. Each is documented in code comments with
its source citation, parameter table, and any reproduction-specific
notes. Only two are registered in the runtime `MISSILE_DB` and exposed
in the GUI's "Missile" dropdown at startup; the others are available
as builder functions and can be added to `custom_missiles.json` for
runtime registration by users who want them visible.

### 15.1 Forden Table 1 — the four reference vehicles

The four missiles from [Forden 2007](#16-references) Table 1 are
implemented as builder functions `_scud_b`, `_al_hussein`, `_nodong`,
`_taepodong_i` (`missile_models.py–1071`). Each carries the exact
mass, Isp, burn time, diameter, and payload values from Forden Table 1
in its source comment:

| Builder | Dry (kg) | Fueled (kg) | Isp (s) | Burn (s) | Dia (m) | Payload (kg) |
|---|---:|---:|---:|---:|---:|---:|
| `_scud_b` | 1 198 | 4 897 | 230 | 75 | 0.84 | 1 000 |
| `_al_hussein` | 1 334 | 6 073 | 230 | 90 | 0.84 | 191 |
| `_nodong` | 3 900 | 19 900 | 240 | 70 | 0.88 | 1 000 |
| `_taepodong_i` | (stage 1 = Nodong body) | | | | | 454 |

The thrust values are derived from these by the rocket identity
`T = I_sp · g₀ · m_p / t_b` (Section 7.1) so they match Forden's
implied thrust to within rounding. Running each missile with Forden's
documented launch elevation and pitch program reproduces the apogee
and range from Forden's Table 1 to within a few percent — the residual
being attributable to the difference between Forden's Cd table
treatment and Thrusty's decomposed wave + friction + base model
(Section 8).

These four missiles are attributable to **Forden (2007)**, not to
Thrusty's authors — they are reproductions of the published reference
cases, included to anchor the simulator's behaviour against an
independent published benchmark.

### 15.2 Extended literature missiles

Six additional builder functions extend the Forden reference set to
cover later vehicles documented in the open literature:

| Builder | Class | Primary source |
|---|---|---|
| `_shahab3` | Shahab-3 (Iranian Nodong derivative) | Open-source literature |
| `_generic_icbm` | Generic three-stage ICBM | Forden-style template |
| `_taepodong_ii` | Taepodong-II / Unha SLV | Forden 2007 discussion |
| `_zoljanah` | Zoljanah SLV (Iran) | Open-source launch reporting |
| `_zoljanah_slv` | Zoljanah configured as SLV | Same vehicle, SLV mode |
| `_minotaur_4_htv2` | Minotaur-IV Lite + HTV-2 glider | [Wright 2015](#16-references) Table 2 |

The `_minotaur_4_htv2` builder is the most heavily documented (and most
carefully validated) of these, because it serves as the test case for
the HGV machinery of Section 12. The booster parameters come from
Wright's [Science & Global Security 2015](#16-references) Table 2
(originally sourced from the Orbital ATK Minotaur-IV User's Guide), and
the glider parameters come from [Acton 2015](#16-references) Table 3
with `glider_beta_entry_kg_m2 = 7.0` set for the Acton-mode β_S
direct-re-entry segment (Section 12.3.2).

These vehicles also draw on parameters from the published literature;
they are *reconstructions* in the same sense as the Forden missiles,
not original work by Thrusty's authors.

### 15.3 AUR — Lewis original

The `_aur` builder (`missile_models.py`) and its variant
`_aur_hgb` (`missile_models.py`) are an original Thrusty
contribution by the author of this document. AUR is a hypothetical
two-stage solid-propellant ballistic missile assembled from
open-source body-dimension and propulsion-class data; it is *not* a
reconstruction of a specific named vehicle from the literature. The
HGB variant (`_aur_hgb`) carries a hypersonic glide body in place of a
conventional warhead and uses the constant-L/D guidance modes of
Section 12.

The AUR/HGB combination is registered in the runtime `MISSILE_DB` as
`"AUR+HGB"`, alongside the Minotaur-IV + HTV-2 reproduction. These
two are the GUI's default "Missile" dropdown entries.

### 15.4 Reproducibility caveats

The qualitative validation against Forden 2007 and Wright 2015 should
not be over-interpreted. What is established:

- The basic 3-DOF integrator produces the same apogee and range from
  the same propulsion and aerodynamic parameters as the published
  source, within the limits of the decomposed-drag model.
- The HGV trajectory machinery (Section 12) reproduces the Tracy &
  Wright 2020 equilibrium-glide ansatz and the Acton 2015 three-phase
  formulation to the level documented in the source papers.

What is *not* established:

- Thrusty has not been compared against actual flight-test telemetry
  for any of the modelled vehicles, because that telemetry is not
  publicly available for any of them.
- The decomposed-drag model (wave + friction + base, Section 8) has
  not been validated against wind-tunnel data for full-vehicle
  configurations; each piece has been validated against its primary
  source (Chin 1961, NACA TN 4201, Crowell 1996) but the full assembly
  has not been independently calibrated.
- The Schilling SLV performance method (§11) has been calibrated
  against historical launch records by Schilling himself; Thrusty
  inherits that calibration without adding its own.

The intended use case — preliminary trajectory analysis for arms-control
verification and threat modelling — is well served by this level of
validation. Use cases that require higher confidence (vehicle thermal
design, terminal-guidance algorithm development, propellant load
optimisation) should use higher-fidelity tools.

---

## 16. References

> **Draft status note.** Specific equation, section, and figure
> numbers for Chin (1961) appearing in Section 8 are taken from
> source-code docstrings; they have not been independently re-verified
> against the PDF in the current editing environment. Citations to
> Tracy & Wright 2020, Acton 2015, Schilling 2009, Sutton & Graves
> 1971, Forden 2007, Anderson 2006, and the NACA reports are verified
> to publication level. Readers preparing the document for publication
> should spot-check the Chin equation numbers before quoting.

### Primary aerodynamics and propulsion

- **Chin, S. S.** (1961). *Missile Configuration Design.*
  McGraw-Hill, New York. Source for cone wave drag (Eqs. 3-4 / 3-6),
  tangent ogive wave drag (Eq. 3-9 / Miles formula), Blasius laminar
  Cf (Eq. 4-1), Schoenherr turbulent Cf (Eq. 4-2), Frankl-Voishel
  compressibility correction (Eq. 4-6), and base pressure coefficient
  (Fig. 3-15). Used throughout Section 8.

- **NACA TN 4201**: Stoney, W. E. Jr. (1958). *Collection of Zero-Lift
  Drag Data on Bodies of Revolution from Free-Flight Investigations.*
  NACA Technical Note 4201. Source for the Von Kármán, LV-Haack /
  Sears-Haack, and parabolic wave-drag comparison data (Section
  8.2.3); also the bluntness extension to r/R ≈ 0.4 (Fig. 7, Section
  8.2.5).

- **Crowell, G. A.** (1996). *The Descriptive Geometry of Nose Cones.*
  Self-published. Source for wetted-area formulas for the five
  implemented nose shapes (Section 8.3.6).

- **Shafer, J. I.** (1959). "Solid Rocket Propulsion," Chapter 16 in
  *Space Technology*, ed. Howard S. Seifert. Wiley, New York. Source
  for the solid-motor grain profile catalogue and fill-factor ranges
  (Section 7.3).

- **Hoerner, S. F.** (1965). *Fluid-Dynamic Drag.* Self-published,
  Bricktown, NJ. Source for the tumbling-cylinder drag coefficient
  used in debris arc calculations (Section 14.3).

### Atmosphere

- **Picone, J. M., Hedin, A. E., Drob, D. P., & Aikin, A. C.** (2002).
  "NRLMSISE-00 empirical model of the atmosphere: Statistical
  comparisons and scientific issues." *Journal of Geophysical
  Research: Space Physics* 107(A12): 1468. The default upper-atmosphere
  model, accessed via the `pymsis` Python package (Section 4.1).

- **U.S. Standard Atmosphere 1976.** NOAA / NASA / USAF, Washington
  DC, NOAA-S/T 76-1562 (1976). Fallback atmosphere model, used when
  `pymsis` is unavailable or selected explicitly via
  `configure_atmosphere(model='std1976')` (Section 4.2).

### Trajectory and guidance

- **Forden, G.** (2007). "Reducing a Common Danger: Improving Russia's
  Early-Warning System." *Cato Policy Analysis* No. 399 (and the
  associated MATLAB tool that Thrusty was originally a Python port of).
  Source for the four reference vehicles (Section 15.1) and for the
  basic pitch-program guidance convention (Section 9.1).

- **Wheelon, A. D.** (1959). "Free Flight of a Ballistic Missile."
  *ARS Journal* 29: 915–926. Source for the optimal-burnout-angle
  formula `γ_opt = ½·arccos(Q/(2−Q))` (Section 10.2).

- **Tsiolkovsky, K. E.** (1903). *Issledovanie mirovykh prostranstv
  reaktivnymi priborami* [Exploration of Outer Space by Reaction
  Devices]. The rocket equation, used for the stack ΔV pre-estimate
  (Section 10.1) and as the basis for the Schilling SLV
  performance method (Section 11.1).

- **Vincenty, T.** (1975). "Direct and Inverse Solutions of Geodesics
  on the Ellipsoid with Application of Nested Equations." *Survey
  Review* 23(176): 88–93. The Vincenty inverse formula used for
  surface distance (Section 2.3).

### SLV performance

- **Schilling, J. R.** (2009). *Estimating the Performance of
  Hypothetical Foreign Space-Launch Vehicles.* International
  Assessment and Strategy Center, working paper. The empirical ΔV
  penalty formulation and `K_3`, `K_4` calibration (Section 11.1).

### Mass estimation

The dry-mass estimator (Section 6.6) draws on a larger source set; the
principal references are listed here, with the full collection (including the
underlying Heineman / MacConochie–Klich / Glatt lineage) in
[`MASS_ESTIMATOR.md`](MASS_ESTIMATOR.md).

- **Akin, D. L.** (2016). *Mass Estimating Relations*, ENAE 791, U. Maryland.
  The primary SI component-level MER set (tanks, engines, thrust structure,
  avionics, wiring) — the "Wilhite-school" relations of Section 6.6.1.
- **Rohrschneider, R. R.** (2002; AIAA 2001-4542). *MER Database for Launch
  Vehicle Conceptual Design*, Georgia Tech / SSDL. Cross-validation of the
  structure/skirt/thrust-structure forms.
- **Hutchinson, V. L. & Olds, J. R.** (2004). *Estimation of Launch Vehicle
  Propellant Tank Structural Weight* (GT-STRESS), AIAA 2004-3661. Basis for the
  physics-based tank option (Section 6.6.1).
- **Zandbergen, B. T. C.** (2015, 6th EUCASS). Pump-fed engine mass/size
  regression — the engine-MER cross-check.
- **Zandbergen, B. T. C.** (2026, TU Delft). *Simple Parametric Relations for
  Solid Rocket Stage Inert Mass Estimation* — the headline solid whole-stage
  estimate (Section 6.6.2).
- **Pietrobon, S. S.** (2009). *Analysis of Propellant Tank Masses* — hydrolox
  aggregate stage-mass law and the Al-Li tank factor.
- **Shu, J.-I., et al.** (2020). *Multistage Liquid Rocket Weight Estimation…*,
  J. Aerospace Eng. 33(6) — the engine-mass-ratio aggregate (κ_E).
- **Goldyn, P., et al.** (2025). *Preliminary Design of Expendable and Reusable
  Mixed-Staged Launch Vehicles*, J. Spacecraft & Rockets — the structural-index
  feasibility ceiling.
- **Northrop Grumman Propulsion Products Catalog** (Jan 2023). Per-motor data for
  the best-in-class solid inert-mass fit (Section 6.6.2).

### Hypersonic glide vehicles

- **Tracy, C. L. & Wright, D.** (2020). "Modeling the Performance of
  Hypersonic Boost-Glide Missiles." *Science & Global Security* 28(3):
  135–170. The equilibrium-glide ansatz `L·cos σ = m·(g − v²/r)`
  (Eq. 7), the 3D EOM with bank-to-turn (Eqs. 1–6), and the inverted-
  dive terminal phase (Section 12.3.1, 12.4, 12.5).
  Available at: https://scienceandglobalsecurity.org/archive/sgs28tracy.pdf

- **Acton, J. M.** (2015). "Hypersonic Boost-Glide Weapons." *Science
  & Global Security* 23(3): 191–219. The three-phase pull-up
  formulation, the ρ/β invariant (Eq. 8), the isothermal scale-height
  and sea-level density fits, and the small-angle circular-arc
  bridging solution (Eqs. 18 / 21). Pierce altitude 100 km
  (Section 12.1, 12.3.2).
  Available at: https://scienceandglobalsecurity.org/archive/sgs23acton.pdf

- **Wright, D.** (2015). "Analysis of the Boost Phase of the HTV-2
  Hypersonic Glider Tests." *Science & Global Security* 23(3):
  220–229. Source for the Minotaur-IV booster stage parameters in
  `_minotaur_4_htv2` (Section 15.2).

- **Lu, P., Forbes, S. & Baldwin, M.** (2013). "Gliding Guidance of High
  L/D Hypersonic Vehicles." AIAA 2013-4648. The altitude-rate feedback
  damping law (Eq. 33), the velocity-scheduled gain (Eq. 34), and the
  command flight-path angle γ\* (Eq. 31) for the `damped_glide` mode
  (Section 12.3.5).

- **Yu, W. & Chen, W.** (2011). "Guidance Scheme for Glide Range
  Maximization of a Hypersonic Vehicle." AIAA 2011-6714. The
  flight-path-angle feedback (Eq. 19) and the empirical gain / heating /
  range sweep corroborating the `damped_glide` gain magnitude
  (Section 12.3.5).

- **Liu, Z., Hu, Y., Gao, C., Jing, W. & Ji, X.** (2025). "Modeling and analysis
  of maneuver laws based on higher-order multi-resolution dynamic mode
  decomposition for hypersonic glide vehicles." *Defence Technology* 48, 34–47.
  Data-driven (DMD) decomposition of HGV skip-glide; independently measures the
  skip phugoid frequency (0.0207–0.0374 rad/s), corroborating the first-
  principles ω_p = √(g_eff/H_ρ) (Section 12.3.5).

- **Chapman, D. R.** (1958/1959). *An Approximate Analytical Method for Studying
  Entry Into Planetary Atmospheres.* NACA TN 4276 / NASA TR R-11. The **primary**
  second-order nonlinear entry ODE (Eq. 21, in the density-like variable `Z(ū)`,
  `ū = V/V_circ`); the equilibrium-glide `Z_II` truncation attributed to Sänger;
  and the lift-driven transition from non-oscillatory glide to "numerous skips of
  sizable intensity" (Fig. 6) — the phugoid, shown numerically. The §12.3.5
  oscillator is the small-perturbation linearisation of this ODE about
  equilibrium glide. (Pages 14, 15, 21, 22, 24, 25 read and verified.)

- **Vinh, N. X., Busemann, A. & Culp, R. D.** (1980). *Hypersonic and Planetary
  Entry Flight Mechanics.* University of Michigan Press. Ch. 10 (Yaroshevskii's
  theory) gives the second-order nonlinear entry ODE (Eq. 10-55) — a special case
  of Chapman's Eq. 21 — and its equilibrium-glide reference state (Eq. 10-61, the
  Sänger condition), and shows the oscillation about equilibrium glide numerically
  (Fig. 10-10). (Vinh §7-2 / §7-5 are the first-order steady-glide and skip
  solutions and contain no oscillator; the perturbation oscillation lives in the
  Ch. 10 second-order theory. Pages 158-162 and 172-176 read and verified.)

### Control theory

- **Ogata, K.** (2010). *Modern Control Engineering*, 5th ed. Prentice
  Hall. §5-3 "Second-Order Systems": standard form, max-overshoot
  Eq. (5-21), the desirable damping-ratio band ζ = 0.4–0.8 (p. 171), and
  the settling-time minimum near ζ = 0.68–0.76 (p. 173). Source for the
  `damped_glide` ζ ≈ 0.7 default (Section 12.3.5).

- **Franklin, G. F., Powell, J. D. & Emami-Naeini, A.** (2019). *Feedback
  Control of Dynamic Systems*, 8th ed. (Global). Pearson. §3.4.2
  "Overshoot and Peak Time": overshoot Eq. (3.72) and Fig. 3.24, which
  lists ζ = 0.7 → 5 % overshoot as a "frequently used value." Source for
  the `damped_glide` ζ ≈ 0.7 default (Section 12.3.5).

### Stagnation heating

- **Sutton, K. & Graves, R. A.** (1971). *A General Stagnation-Point
  Convective-Heating Equation for Arbitrary Gas Mixtures.* NASA
  Technical Report R-376. The stagnation heat-flux correlation
  `q̇ = K·√(ρ/R_N)·V³` with `K = 1.7415×10⁻⁴` for Earth (Section 13.1).

- **Anderson, J. D.** (2006). *Hypersonic and High-Temperature Gas
  Dynamics*, 2nd ed. AIAA Education Series. Reference for the
  emissivity default `ε = 0.85` (§18.8, p. 781, HERMES example,
  Section 13.2).

- **Williams, S. D. & Curry, D. M.** (1992). *Thermal Protection
  Materials: Thermophysical Property Data.* NASA Reference Publication
  1289. Source for the reinforced carbon-carbon emissivity
  temperature-dependence reference (Section 13.2, 13.3).

### Earth model

- **WGS-84:** *Department of Defense World Geodetic System 1984: Its
  Definition and Relationships with Local Geodetic Systems.* NIMA
  Technical Report TR8350.2, 3rd ed. (2000). Source for `GM`, `R_E`,
  flattening `f`, J₂, and the Earth rotation rate `Ω` used throughout
  the EOM (Section 3, Section 5).

### Numerical methods

- **scipy** (Virtanen et al. 2020, *Nature Methods* 17: 261–272).
  The `solve_ivp` ODE integrator with RK45 method (Section 5.6),
  `brentq` for cutoff/range root-finding (Section 10.4), and
  `minimize_scalar` for range maximisation (Section 10.3).

- **pymsis** (Lucas 2022). Python wrapper around the official NRL
  Fortran NRLMSISE-00 source, providing the atmosphere lookup
  (Section 4.1).
