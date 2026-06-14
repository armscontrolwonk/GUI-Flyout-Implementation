# Thrusty — Methods

This document is the technical reference for Thrusty's models and algorithms.
It complements the in-repo [`README.md`](README.md) (overview, source-file
guide, quick-start) and [`USER_GUIDE.md`](USER_GUIDE.md) (step-by-step usage).
Each section gives the governing equation(s), a brief description of the
implementation, and citations to primary sources where one exists.

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

The single ODE integrated by `integrate_trajectory` (`trajectory.py:1316`) is

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
longitude (`_enu_frame`, `trajectory.py:301`). The unit vectors are

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

The constants are the WGS-84 values (`gravity.py:10–13`):

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
| `lat`, `lon` | 0, 0 | Geographic location for the MSIS evaluation |

The user can override any of these via `configure_atmosphere(**kwargs)`,
which rebuilds the lookup table in ~10 ms (measured: 6 ms on a typical
laptop, with pymsis 0.12). This is the path to model a real launch date
and site with measured solar indices.

#### Design rationale

**Implementation summary.** The NRLMSISE-00 table covers 0–1000 km at
500 m intervals — `alts_km = np.arange(0.0, 1000.5, 0.5)` — built at
import time from a single `pymsis.calculate()` call
(`atmosphere.py:140`). The COESA 1976 code (`_atmosphere_std1976`,
`atmosphere.py:70`) remains available both as an automatic fallback if
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
US Std Atm 1976 Table I/II reference points at 91, 100, 110, 120, 150, 200,
300, 500, and 1000 km.

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

(`_eom`, `trajectory.py:604`). The right-hand side gives acceleration
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

`scipy.integrate.solve_ivp` with `method='RK45'`, default relative tolerance
`1e-8` and absolute tolerance `1e-6` on positions/velocities. Adaptive
step sizing concentrates evaluations during the boost phase (where mass and
thrust change rapidly) and stretches them during coast.

Event functions used:

| Event | Purpose |
|---|---|
| `_hit_ground` | Trajectory termination at altitude ≤ 0 |
| `_apogee` (sign change of `ṙ·r̂`) | Apogee detection |
| Milestone-altitude crossings | 100 km re-entry, shroud jettison, user-defined queries |
| `_glider_pierce_atmosphere` | HGV pull-up / equilibrium-glide handoff |
| Orbital-energy cutoff | Engine cutoff when specific orbital energy meets target |

Detected events become rows in the Flight Timeline output (Section 14).

---

## 6. Mass and staging

A missile is represented by a linked-chain `MissileParams` dataclass
(`missile_models.py:25`). Each stage carries its own propulsive and
geometric parameters; the `.stage2` attribute points to the next stage.
Top-level fields (payload, shroud, RV) apply to the whole vehicle.

### 6.1 Mass schedule

For a stage with initial mass `m₀`, propellant mass `m_p`, and burn time `t_b`,
the instantaneous mass during burn is linear in time:

```
m(τ) = m₀ − (m_p / t_b) · τ              0 ≤ τ ≤ t_b
```

where `τ` is local stage burn time. The burnout mass is `m_f = m₀ − m_p`.

The full-vehicle mass `missile_mass(params, t)` (`missile_models.py:2009`)
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
`missile_models.py:1976`). The spent casings then follow tumbling-cylinder
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
`missile_models.py:2205`):

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
`solid_motor` boolean on each stage gates this behaviour:
`integrate_trajectory` ignores cutoff commands for stages with
`solid_motor = True`.

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

(`_cd_nose_shape`, `missile_models.py:864`). Each component has a
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
(`_chin_pressure_coeff`, `missile_models.py:600`):

```
Δp/q          = (0.083 + 0.096 / M²) · (σ° / 10)^1.69
C_D,wave,cone = Δp/q
```

The half-angle is derived from the user-specified fineness ratio
`l/d = nose_length / body_diameter` as
`σ = arctan(1 / (2 · l/d))`, then converted to degrees for the formula.

The code applies a transonic linear ramp to avoid the formula's
behaviour near the shock-attachment limit
(`_cd_wave_cone`, `missile_models.py:607`):

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
(`_cd_wave_ogive`, `missile_models.py:618`):

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
`l/d_nose = 3` (`_cd_wave_table`, `missile_models.py:634`).

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
and a roughness allowance (`_cd_friction`, `missile_models.py:711`):

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
(`_mu_air`, `missile_models.py:690`). Sutherland's law provides
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
(`_s_wet_ratio`, `missile_models.py:676`, after [Crowell 1996](#16-references)):

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
from `_BASE_CPB` (`missile_models.py:517`):

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
(`missile_models.py:933`), so powered flight sees the same `C_pb(M)` as
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

**Fin lift slope** (`_cl_alpha_fins`, `missile_models.py:758`) — the lift-curve
slope per fin from slender-body theory:

```
C_Lα,fin = 2π · (S_fin / S_ref) · η_fin
```

where `η_fin` is a fin-efficiency factor that depends on the fin planform
aspect ratio. For preliminary work the slender-body limit is used.

**Fin drag** (`_cd_fins`, `missile_models.py:804`) — flat-plate skin friction
on the fin wetted area plus a thickness pressure-drag correction:

```
C_D,fins = (N_fins · A_fin / S_ref) · 2 C_f · (1 + 2 · t/c)
```

with `N_fins` the number of fins, `A_fin` the planform area per fin, `t/c`
the thickness-to-chord ratio, and `C_f` the friction coefficient from
Section 8.3. The factor of 2 accounts for both fin surfaces. This is the
USAF DATCOM-style formulation.

For strategic and theatre-range missiles flying mostly through the upper
atmosphere, fin drag is a second-order effect and is sometimes set to
zero by leaving the fin parameters at their default zero values. For
short-range tactical missiles or atmospheric flight the fin term matters
and should be enabled.

### 8.6 Aerospike correction

An aerospike is a forward-projecting spike (sometimes terminated in a
small aerodisk) that creates a slender bow shock to replace the strong
detached shock of a blunt body, reducing wave drag at supersonic Mach
(`_aerospike_effective_LD`, `missile_models.py:738`).

The implementation replaces the actual nose's wave drag with the
*minimum* of (actual nose drag) and (effective-body cone drag), where
the effective body is a cone whose half-angle is determined by the spike
length-to-diameter ratio:

```
spike L/D = L_spike / D_body
spike d/D = D_aerodisk / D_body
effective half-angle ≈ arctan( 1 / (2 · spike_LD ) )      (sharp spike)
```

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
(`_cd_blunted_cone_newtonian`, `thrusty.py:130`). For a sharp cone the
exact Newtonian result is `C_D = 2 sin²θ`. For blunted cones (half-angle
10°–40°, bluntness ratio `ε = r_N/r_b` from 0 to 1) the table is taken
from a published Newtonian-hypersonic Cd chart cited in the source code
as "Ref (4) Ch. 5". The full primary reference for "Ref (4)" is not
defined in the repository; given the surrounding source chain (Section 8.2)
it is plausibly Chin (1961) Ch. 5, but this should be verified before
citing the table as Chin in publication-grade work.

For hypersonic glide vehicles the constant-β model is augmented by a
lift term and (optionally) a polar-drag model — see Section 12.

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

`_gravity_turn_thrust_dir` (`trajectory.py:329`) implements the linear
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

`_true_gravity_turn_thrust_dir` (`trajectory.py:443`) implements a
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

`_orbital_insertion_thrust_dir` (`trajectory.py:364`) implements a
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

The companion planner `plan_orbital_insertion` (`trajectory.py:2896`)
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

`_yaw_program` (`trajectory.py:513`) implements a multi-segment azimuth
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

(`_tsiolkovsky_dv`, `trajectory.py:3050`). This is used as a fast estimate
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

(`_wheelon_gamma_opt`, `trajectory.py:3063`). The dimensionless ratio
`Q` is bounded above by 1 (orbital velocity); for sub-orbital
trajectories Q < 1 and γ_opt is well-defined and lies between 0° (Q = 1,
flat orbit) and 45° (Q → 0, throw the rock from a tower).

Used as the centre of a coarse-grid search window
`γ ∈ [γ_opt − 10°, γ_opt + 10°]`. This roughly two-thirds reduction in
search volume relative to a 5°–80° unbounded scan brings the
range-maximisation cost down to a level where a parallel grid search
finishes in seconds for typical missiles.

### 10.3 Range maximisation algorithm

`maximize_range` (`trajectory.py:3075`) is a two-phase search over
(burnout angle, turn-stop time):

1. **Coarse parallel grid.** A grid of candidate `(γ, t_stop)` pairs
   is evaluated on a thread pool (up to 8 workers, capped to avoid
   hyperthreading thrash). The angle window is bounded by Wheelon
   (Section 10.2). The turn-stop window covers the powered-flight
   duration, with a 3600 s outer cap to bail out on degenerate cases
   that never impact.
2. **Brent polish.** The best coarse candidate is refined by
   `scipy.optimize.minimize_scalar` (Brent's method) over the burnout
   angle, with the turn-stop fixed at the coarse-grid optimum.

The result dictionary returns the maximum range plus the optimal
`(burnout_angle, turn_stop)` and the full trajectory at the optimum.
A `cancel_event` parameter allows GUI cancellation between coarse-grid
evaluations.

### 10.4 Aim at target

`aim_missile` (`trajectory.py:2840`) finds the engine cutoff time that
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

`find_range` (`trajectory.py:2877`) is a trivial wrapper around
`integrate_trajectory` that returns only the range; used by GUI calls
that don't need the full trajectory dictionary back.

---

## 11. SLV performance estimation

For space-launch vehicles the question is *can this rocket put a given
payload into a given orbit*, not *how far does it fly ballistically*.
This is answered by the Schilling/Townsend method, an algebraic ΔV
budget that does not require trajectory integration
(`slv_performance.py`).

### 11.1 The Schilling/Townsend method

The available ΔV from a stack of stages is the Tsiolkovsky sum
(Section 10.1). The required ΔV to reach a target orbit is

```
ΔV_req  = V_inj  +  ΔV_pen  −  V_rot
V_inj   = √( μ · (2/r_p − 1/a) )                vis-viva at perigee
a       = (r_p + r_a) / 2                       semi-major axis
V_rot   = R_E · Ω · cos(lat) · cos(azimuth)     Earth-rotation assist
```

`V_inj` is the inertial-frame injection speed at perigee from the
vis-viva equation; for a circular orbit this collapses to
`√(μ/r)`, the circular orbital speed. For an elliptical transfer the
formula gives the perigee speed of the target orbit. `V_rot` is the
ground-frame velocity at the launch site, which is "free" — a launch
toward the east adds Earth's rotation to the inertial speed at no
propellant cost.

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

### 11.2 Maximum payload — iterative solve

Computing the maximum deliverable payload is more subtle than checking
margin at a single payload because payload appears on both sides:

- Payload affects `ΔV_avail` directly through the Tsiolkovsky stage-burnout
  masses.
- Payload affects `ΔV_req` indirectly through `T_actual` (heavier
  payload → longer burn → larger penalty).

The implementation uses a binary search on payload between 0 and an
upper bound (default 2 × the configured payload), with an outer
fixed-point iteration to handle the `T_actual ↔ ΔV_pen` coupling. ~5
outer iterations and ~12 inner binary-search steps converge to <1 kg
precision.

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
| Can the full SLV reach the claimed orbit with the claimed payload? | Schilling/Townsend (Section 11) |
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
drag polar); four guidance modes spanning a spectrum of phugoid
suppression; a bank-to-turn cross-range model; an inverted terminal dive;
and a JSON-based RV library for shipping or extending vehicle
definitions.

The trajectory machinery and physical conventions follow
[Tracy & Wright 2020](#16-references) and
[Acton 2015](#16-references); the four guidance modes implement (i) the
Tracy & Wright equilibrium-glide ansatz, (ii) the Acton three-phase
analytic pull-up, (iii) a pure phugoid skip-glide, and (iv) a
Thrusty-original hybrid that lets the vehicle skip phugoidally for a
user-set number of cycles before settling into equilibrium glide.

### 12.1 The pierce-altitude latch and glide-mode state machine

Glide-mode aero forces are *not* active throughout the flight; they
activate only when a specific re-entry condition is met. The latch lives
in `effective_rv()` and the active-glider gate at `trajectory.py:683`:

```
glide_mode = ( RV has separated
            ∧ glider_enabled
            ∧ glider_LD > 0
            ∧ has-crossed-up-then-down-through ACTON_PIERCE_ALT_M )
```

The pierce altitude is

```
ACTON_PIERCE_ALT_M = 100_000.0          # trajectory.py:113
```

— the conventional 100 km Kármán-line value used by Acton 2015 (p. 204) as
the start of his Phase 3 (direct re-entry). The latch (`_gl_above_pierce`
at `trajectory.py:647`) requires the vehicle to have first crossed *up*
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
and a derived reference area (`_aero_polar`, `trajectory.py:554`). The
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
(`_C_L_lim` at `trajectory.py:727`), representing the slender-body
small-angle relation `C_L ≈ 2α` evaluated at α_max = 25°. At 25° the
linearization is starting to lose validity — the exact Newtonian value
`sin 2α` gives 0.766 — so this is best read as a conservative upper bound
on the trim solution rather than a precise aerodynamic limit.

### 12.3 Guidance modes

The `glider_guidance` field on each RV selects one of four guidance laws,
exposed by the GUI dropdown labelled "Glider guidance":

| GUI label | `glider_guidance` value | Origin |
|---|---|---|
| Equilibrium glide (Tracy) | `equilibrium_glide` | Tracy & Wright 2020 |
| Equilibrium glide (Acton) | `equilibrium_glide_acton` | Acton 2015 |
| Phugoid / skip-glide | `skip_glide` | Sänger / classical skip-glide |
| Skip → equilibrium (auto-handoff) | `skip_to_equilibrium` | Lewis (Thrusty-original) |

The four modes form a spectrum of *how aggressively the guidance
suppresses phugoid amplitude*, ranging from "fully suppress it" (Tracy
and Acton, mild residual phugoid only) through "let it ride at max-L/D
α*" (skip-glide, larger-amplitude damped phugoid) to "let it ride for N
skips, then suppress" (skip-to-equilibrium). Atmospheric drag provides
natural damping to all of these — they are bounded oscillations, not
unbounded ones.

#### 12.3.1 Equilibrium glide (Tracy)

The simplest mode. At the pierce point (100 km on descent) the vehicle
is treated as already in equilibrium glide; no pull-up arc is modelled.
The EOM is integrated with constant L/D (or the polar trim) and a single
β throughout the glide phase. The integrator's natural dynamics keep the
vehicle near equilibrium because Tracy's Eq. (7) is satisfied at the
pierce point by construction (`trajectory.py:108–112`):

```
L · cos σ = m · (g − v² / r_e)              [Tracy 2020 Eq. (7)]
```

The equivalent statement: lift balances *gravity minus centripetal*. In
the EOM this appears as the trim condition (`trajectory.py:780`):

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
   altitude and producing unrealistically long range (`trajectory.py:738`
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
ACTON_SCALE_HEIGHT_M  = 6970.0                      # trajectory.py:115
ACTON_SEA_LEVEL_RHO   = 1.46                        # trajectory.py:116
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

implemented by `_acton_pullup_arc` (`trajectory.py:945`). The arc is
applied as a one-shot state reset at the Phase 3 → Phase 4 boundary
(detected by the descending crossing of h₃ via the event function
`_make_phase3_end_event`, `trajectory.py:931`), after which equilibrium
glide proceeds exactly as in Tracy mode.

Implementation detail: Acton mode falls back to Tracy mode if
`glider_beta_entry_kg_m2 ≤ 0`, since without a positive β_S the Phase 3
direct-re-entry segment is ill-defined. This guarantees the user always
gets an analytical pull-up rather than a phugoid in cases where the
small-β data is missing (`trajectory.py:1517`).

#### 12.3.3 Phugoid / skip-glide

In this mode the guidance does *not* trim to suppress oscillation.
Instead the vehicle flies at the max-L/D angle of attack throughout, so
lift is proportional to dynamic pressure and the natural phugoid
oscillation about the equilibrium altitude is preserved at full
amplitude (`trajectory.py:760`):

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
flight (`trajectory.py:1521`).

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

### 12.4 Bank-to-turn (cross-range manoeuvring)

Cross-range manoeuvring is by *bank-to-turn* with roll angle σ
(`bank_rad` in the code, settable via the RV's bank schedule or a target
point). Banking partitions the lift vector between vertical and
horizontal components:

```
F_lift = L · ( cos σ · n̂_up  +  sin σ · n̂_cross )
```

(`trajectory.py:800`). This is exactly the Tracy & Wright 2020 EOM
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
toward the ground rather than away from it (`trajectory.py:704–722`).
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
| `nose_radius_m` | 0.05 | Used for stagnation heating (Section 13.1) |
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

The full schema lives in `RVParams.from_dict`
(`missile_models.py:395`).

**Acton mode and β_S.** The shipped JSON RVs leave `glider_beta_entry_kg_m2`
at zero, since β_S (the high-α drag direct-re-entry ballistic coefficient)
is published for only a handful of vehicles. Selecting Acton mode for an
RV with β_S = 0 falls back to Tracy mode (Section 12.3.2). For the
HTV-2-class built-in missiles defined programmatically rather than via the
RV library — `Forden_HTV2` and its variants — the code sets
`glider_beta_entry_kg_m2 = 7.0` based on Acton 2015 Table 3
(`missile_models.py:1487, 1635`). A user wanting to run Acton mode on a
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
(`trajectory.py:2625`).

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
   `trajectory.py:2621`.

2. **Per-RV nose-tip radius.** `R_N` is read from the active RV's
   `nose_radius_m` field (`trajectory.py:2619`), making the stagnation-
   point formula geometry-aware. The default in `RVParams` is 0.05 m
   (5 cm; `missile_models.py:238`). The shipped library overrides this
   for `C-HGB.rv.json` (2 cm, matching the sharper conical glide body)
   and otherwise inherits the 5 cm default. Two RVs at the same speed
   and density therefore see different stagnation fluxes:
   `q̇ ∝ 1/√R_N`, so the C-HGB nose sees about 1.58× the stagnation
   flux of an HGB-class nose. The peak-heating *time* is independent
   of `R_N`; only the peak *magnitude* changes.

3. **Glide phase only.** Heating is computed and reported only during
   the post-pierce glide phase (after the 100 km descent crossing of
   Section 12.1). The boost and ascent heating is not separately
   reported because for typical operational vehicles the boost phase
   stagnation flux is much smaller than the glide-phase peak — and
   because the booster nose is not the surface that matters for
   payload survival.

### 13.2 Radiative-equilibrium wall temperature

The peak heat-flux value is converted to a radiative-equilibrium
stagnation-point temperature via the Stefan-Boltzmann law assuming
all incoming convective flux is balanced by surface re-radiation:

```
σ · ε · T_eq⁴ = q̇_peak             ⇒    T_eq = (q̇_peak / (σ · ε))^(1/4)
```

with `σ = 5.670374419×10⁻⁸ W/(m²·K⁴)` (CODATA 2019, Stefan-Boltzmann
constant) and `ε` from the RV's `emissivity` field (default 0.85; set
at `RVParams`, `missile_models.py:330`). The default is consistent with
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
glide-phase time step (`trajectory.py:2625`), finds the argmax, and
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

The canonical return structure (`trajectory.py:2811`):

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
`_insert_chrono()` (`trajectory.py:2243`). Each milestone is a dict with
`{t_s, alt_km, range_km, mass_t, speed_ms, event}` populated by
interpolation onto the trajectory array (`_interp_milestone`,
`trajectory.py:274`). The exhaustive set of event labels currently
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
coefficient (`tumbling_cylinder_beta`, `missile_models.py:1900`). This
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

- **FootprintDialog** (`thrusty.py:3153`). Sweeps the bank-angle schedule
  for an HGV across a range of cross-range maneuvers and computes the
  envelope of reachable terminal points. Output: an impact-zone polygon
  rendered on the map.
- **RangeRingDialog** (`thrusty.py:2493`). Draws great-circle range
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
`_taepodong_i` (`missile_models.py:946–1071`). Each carries the exact
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

The `_aur` builder (`missile_models.py:1405`) and its variant
`_aur_hgb` (`missile_models.py:1471`) are an original Thrusty
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
  (Section 10.1) and as the basis for the Schilling/Townsend SLV
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
