# Dry-Mass Estimator (`mass_estimator.py`)

A standalone, importable module (and a Thrusty GUI widget) that estimates the
**dry / inert mass** of a rocket stage from its parameters, and reports how far
a *stated* dry mass diverges from established mass relationships. Thrusty takes
each stage's burnout mass as a direct input; this tool is an independent
cross-check on that number — "is the structure factor I typed in physically
plausible, or have I built an unobtanium rocket?"

It is deliberately built around two complementary families, because no single
relation is trustworthy across the whole size/technology range:

| Family | Needs | Best for |
|---|---|---|
| **Component-level (Wilhite-school MERs)** | geometry, propellant split, thrust | large launch-vehicle stages; itemised "where is the mass" breakdown |
| **Aggregate (whole-stage relations)** | propellant mass only | quick sanity bound; small stages; solids |

Solid and liquid motors are handled separately: a solid's case is a
pressure-loaded vessel sized by chamber pressure × grain volume, while a
liquid's tanks are near-unloaded and sized by propellant volume.

All inputs and outputs are **SI** (kg, m, m³, N, Pa, s).

---

## Usage

### As a program

```
python mass_estimator.py --demo

python mass_estimator.py liquid --prop LOX/RP1 --prop-mass 280000 \
        --thrust 7.6e6 --engines 1 --diameter 3.7 --length 40 --stated-dry 18000

python mass_estimator.py solid --prop-mass 87710 --thrust 4.5e6 \
        --casing composite --stated-dry 8533
```

### As a library

```python
import mass_estimator as mest
inp = mest.LiquidStageInputs(propellant="LOX/LH2", prop_mass_kg=135_800,
                             thrust_n=6*324_900, n_engines=6, diameter_m=4.2)
estimates, divergence = mest.analyse_liquid(inp, stated_dry_kg=12_240)
```

### In Thrusty

**Analysis → Dry Mass Estimator…** opens a dialog pre-filled from the currently
selected missile. Pick the stage, confirm liquid/solid, choose the propellant
combination (liquid) or casing material (solid), and press **Compute**. The
stage's stated dry mass (recovered from the missile's per-stage mass
bookkeeping) is compared against every estimate, with a verdict
(`consistent` ≤ 15 %, `marginal` ≤ 35 %, otherwise `optimistic` / `conservative`).

---

## Methods and coefficients

### Liquid — component-level MERs (primary set: Akin / UMD ENAE 791, 2016)

The component coefficients are the SI compilation taught in D. L. Akin's
*Mass Estimating Relations* (U. Maryland ENAE 791), which descends from the
historical Heineman / MacConochie–Klich / Glatt (WAATS) launch-vehicle MERs —
the same lineage as A. W. Wilhite's relations used in the SSDL conceptual-design
toolset.

| Component | Relation | Units |
|---|---|---|
| LH₂ tank | `M = 9.09·V` (or `0.128·m_LH2`) | V m³ |
| Other tank | `M = 12.16·V` ; LOX `0.0107·m`, RP-1 `0.0148·m` | V m³ |
| LH₂ insulation | `M = 2.88·A_tank` | A m² |
| LOX insulation | `M = 1.123·A_tank` | A m² |
| Pump-fed engine | `M = 7.81e-4·T + 3.37e-5·T·(Ae/At) + 59` | T N, per engine |
| Thrust structure | `M = 2.55e-4·T` | T N total |
| Gimbals / TVC | `M = 237.8·(T/P₀)^0.9375` | T N, P₀ Pa |
| Fairing / shroud | `M = 4.95·A^1.15` | A m² |
| Avionics | `M = 10·M₀^0.361` | M₀ = vehicle GLOW |
| Wiring | `M = 1.058·√M₀·L^0.25` | M₀ stage gross, L m |

**Avionics is one package per vehicle.** Guidance avionics is the flight
computer / IMU suite, carried on the **upper stage only** (never on lower
boosters and never on the bus / PBV), and is sized on the **vehicle gross
liftoff mass** (the size of vehicle it guides), not the stage it rides on. In
the Thrusty dialog the "carries guidance avionics" box defaults **on** for the
last stage and **off** for boosters; the CLI exposes `--no-avionics`. Wiring,
by contrast, is present on every stage and is sized on the stage's own gross
mass.

Tank volumes are obtained by splitting the total propellant load with a
representative mixture ratio (O/F by mass) and dividing by stored density;
cryogenic insulation is added only for cryogenic fluids. Tank surface area uses
a sphere, or a cylinder-with-hemispherical-domes when a body diameter is given.

**Tank material** (`tank_material`) scales the tank MER — the tanks are the
material-sensitive part; engines, thrust structure and avionics are not scaled:

| Material | Factor | Basis |
|---|---|---|
| Aluminium (Al 2219) | 1.00 | Akin baseline |
| Al-Li 2195 | 0.74 | Pietrobon (−26%, from specific-yield-strength ratio) |
| Composite (Gr/Ep) | 0.45 | Rohrschneider/SSDL 1970→2015 tank coefficients (≈0.43–0.47×) |
| Steel | 1.60 | thin-gauge / pressure-fed tankage; rare on pump-fed stages |

The **tank sizing model** (`tank_model`) selects how the two tanks are massed:

- `akin_volume` *(default)* — the volume MER above, material-scaled.
- `akin_offset` — an alternate Akin vintage with a fixed-mass term
  (`LOX 0.0152·m+318`, `LH2 0.0694·m+363`), better-behaved for small stages.
  (The source's dedicated storables relation was unreadable in the scan, so
  storables fall back to the RP-1 coefficient.)
- `physics` — the GT-STRESS load/material shell sizing (below).
- `averaged` — the mean of `akin_volume` and `physics`, following the SPSP
  (Scher & North) multi-estimate philosophy: averaging independent estimates is
  more robust than any single one. (SPSP independently validates the
  pressure-vessel-plus-correction tank approach used here.)

> **Note on the engine MER.** Akin's lecture *table* lists 373 kg per engine in
> the worked SSTO example, but his own printed formula gives ≈ 641 kg at the
> example's 324.9 kN / ε = 30 — and that agrees with the independent Zandbergen
> (2015) hydrolox fit (≈ 609 kg) to within 5 %. The implementation follows the
> **formula** (and `test_mass_estimator.py` pins it against Zandbergen), treating
> the 373 kg slide figure as an arithmetic slip.

**Engine cross-check — Zandbergen (EUCASS 2015)**, from a regression over 45+
pump-fed engines (reported as a note, not summed):

- hydrolox `M = 0.00514·T^0.92068` (RSE ≈ 13 %)
- kero-lox / storable `M = 1.104e-3·T + 27.702` (RSE ≈ 26 %)

**Cross-reference set — Rohrschneider / SSDL (AIAA 2001-4542; MER database
2002).** The SSDL "Reference 6" inline-expendable-LV relations (English units)
were used to validate the structure/skirt/thrust-structure forms used above:
thrust structure `K·T^1.0687`, intertank/interstage/skirt `K·S·b_body^K2`,
TVC `0.001185·T`, tanks `K·V·(1−ullage)` with shuttle `K_LH2 = 0.5595`,
`K_LOX = 0.8086` lb/ft³ (≡ the Akin SI values).

### Liquid — physics-based tank option (Hutchinson & Olds, GT-STRESS, 2004)

Instead of the empirical tank MER, the liquid tanks can be sized from **loads
and material properties** — a simplified single-critical-station version of the
GT-STRESS beam/shell method (Hutchinson & Olds, AIAA 2004-3661, the same
Georgia Tech SSDL lineage as Rohrschneider). Each tank is sized as a thin
circular shell under the worst of two load cases:

- **burnout / max axial** — peak axial compression (≈ stage thrust reacted at
  the base), tank nearly empty so head pressure is small;
- **liftoff / max-q-α** — full tank (full hydrostatic head + ullage) plus a
  lateral inertial bending load from the design **lateral-g**.

Shell thickness at each station is the max over **ultimate-tensile, yield,
axial buckling** (stiffened wide-column, Table-1 efficiencies) and **minimum
gauge**; the governing thickness × tank area × material density × a correlation
factor (`TANK_CORRELATION = 1.50`, folding in frames + secondary structure +
the single-station simplification) gives the structural mass. Material
properties (ρ, σ_yield, σ_ult, E, min gauge) are taken from `MATERIALS` for
aluminium 2219 / Al-Li 2195 / steel / composite, so material choice is physics,
not a multiplier. Internal pressure relieves axial compression, as in flight.

Inputs: design **lateral-g** (default 0.5; axial load comes from thrust, so no
trajectory run is needed — peak axial ≈ T/(m_burnout·g₀) is implicit), design
**ullage pressure** (default 0.25 MPa), and shell configuration. Calibrated to
reproduce the aluminium EELV and Shuttle-ET tanks of the source paper to ≈±30%
(comparable to the paper's own 11–29% pre-correlation scatter). Cryogenic
insulation is added on top, as in the empirical path. Enable with
`--physics-tank` (CLI) or the "Physics tank (GT-STRESS)" box in the dialog.

> The workflow this supports: estimate cold (axial from thrust, design
> lateral-g), then fly the trajectory in Thrusty and refine the loads from the
> actual peak axial-g and max-q to trim mass. Pulling loads directly from the
> last trajectory is a planned follow-on; today the lateral-g is a manual field.

### Liquid — aggregate relations

- **Pietrobon (2009) LOX/LH₂ stage-mass power law** (stage mass *less engines*,
  tonnes): `ms = a·mp^0.848`, `a = 0.19` (all-stages average), `0.1583`
  (common bulkhead, through Saturn S-II), `0.1171` (Al-Li 2195 common bulkhead).
  Strictly valid only for hydrolox; shown only when the propellant is LOX/LH₂.
- **Assumed structural coefficient** `ε = m_inert / (m_inert + m_prop)` ⇒
  `m_inert = ε·m_prop/(1−ε)`. **Opt-in only and not a prediction** — it merely
  restates an ε you supply as kilograms, so it can never tell you anything you
  didn't already assume. It is omitted unless explicitly given; ε's real role
  in this tool is as the *reporting unit* of the divergence table (below).
- **Engine-mass-ratio method (Shu et al. 2020)** — the predictive aggregate
  for *any* liquid, including non-hydrolox. The structural mass is
  `M_struct = M_engine / κ_E`, where κ_E = M_engine/M_struct is taken from
  historical data; total inert = `M_engine·(1 + 1/κ_E)`. Engine mass is
  *predicted* from thrust (the Akin engine MER), so — unlike assuming ε — this
  carries real information; κ_E varies over a much narrower, more stable band
  than ε (Shu's Fig. 2). κ_E defaults by stage role (lower ≈ 0.25, upper ≈ 0.12;
  anchors: KSLV-II 0.252/0.177/0.094, Titan II 0.250/0.111) and is overridable.
  Shown for every liquid stage with a thrust; this is what fills the
  non-hydrolox gap that ε could not.
- **Feasibility ceiling (Goldyn et al. 2025)** — `ε_max = 1/exp(Δv/(g₀·Isp))`.
  Above it the rocket equation drives propellant mass negative, so a stated or
  estimated ε exceeding the ceiling is physically impossible. When a stage Δv
  and Isp are supplied (`--delta-v`, `--isp`), any estimate breaching the
  ceiling is flagged with a ⚠ note.

### Solid — whole-stage inert mass (Zandbergen 2026 / 2019)

The headline solid estimate uses the dedicated regressions in B. T. C.
Zandbergen, *Simple Parametric Relations for Solid Rocket Stage Inert Mass
Estimation* (TU Delft, 2026), fit to 17 + 17 flown stages (excludes upper
stages). Masses in tonnes:

| Casing | Relation | R² | RMSPE |
|---|---|---|---|
| Steel (power) | `m_i = 0.2851·m_p^0.9030` | 0.993 | 22 % |
| Steel (linear) | `m_i = 0.1689·m_p + 0.509` | 0.997 | 23 % |
| Composite (power) | `m_i = 0.1275·m_p^0.9678` | 0.966 | 24 % |
| Composite (linear) | `m_i = 0.1110·m_p` | 0.978 | 25 % |

These are **whole-stage** inert masses (case, nozzle, insulation, igniter,
skirts, TVC, avionics, separation). An *assumed* propellant-mass-fraction
option `m_inert = m_prop·(1/ζ − 1)` is available opt-in (`--zeta`) — like the
liquid ε, it restates your assumption in kg and is not a prediction.

**Best-in-class cross-check — Lewis 2026 (NG catalog).** A second whole-stage
fit (`source="lewis"`) is reported alongside Zandbergen: a Lewis regression of
the *Northrop Grumman Propulsion Products Catalog* (Jan 2023, pp. 9–39), 29
flight-proven Orion/Castor/GEM motors. Coefficients are in lbm as published
(power law `m_i = k·m_p^0.947`, k = 0.172 composite / 0.258 steel; or constant
inert fraction f = 0.092 / 0.132) and converted at the kg boundary. Because
these are mature, mass-optimised flight motors, this fit runs **~10 % lighter**
than the broader Zandbergen sample — it is the lower (best-in-class) edge of the
inert band, the right reference for "fanciest US motor" sizing. Same scope as
Zandbergen (whole-stage, nozzle included). Material penalty steel/composite
≈ 1.50× (size-only), independently matching Zandbergen's 1.52×. ~20 % (1σ)
scatter; steel rests on 3 Castor-IV motors (~20–35 k lbm) — do not trust steel
outside that band.

**Slenderness (L/D) correction.** When a stage length and diameter are both
supplied, the Lewis power law adds a slenderness term refit on the 26-motor
composite group:

    m_inert = 0.24087 · m_prop^0.8832 · (L/D)^0.1834      [lbm]    (composite)

The L/D exponent is **positive** — a more slender motor carries *more* inert per
unit propellant, because case wall and insulation scale with surface area, not
enclosed volume. It is significant (p ≈ 0.006) and cuts RMS from 21.6 % to
18.1 %. This is what makes a catalogue skewed toward stubby launcher motors
usable on **slender missile stages**, which it otherwise under-predicts. With
the L/D term in, the steel material penalty drops to **×1.347** (from 1.50): the
slender steel Castor-IV motors' extra inert was largely geometry, not material.
Stage type (upper vs first) was tested and is redundant once L/D is present.
When length/diameter are absent the estimator falls back to the size-only
coefficient above. *Note:* the fit is on **propellant** mass, not the dataset
README's loaded-mass form (RMS 16.6 %) — loaded mass embeds the inert mass being
predicted, so it is circular for a tool that validates a stated inert mass.
Largest residual remains the Castor 30XL at −26 % (submerged nozzle / advanced
composite, outside the regressors). Provenance dataset: `mass_data/ng_motors.csv`.

### Solid — component-level (partial)

Open-literature component MERs for solids are sparse. The component view sums:

- **Motor case** — Akin `M = 0.135·m_prop` (case only).
- **Thrust structure / gimbal** — as for liquids.
- A **first-principles pressure-vessel case** cross-check (reported, not summed):
  for a thin-wall membrane vessel the mass is set by the pressure–volume product
  over the material specific strength,
  `m = k·SF·P·V/(σ/ρ)` (k = 1.5 sphere, 2.0 cylinder) — the physics the
  "Wilhite Solid" spreadsheet approximates via yield strength.

Nozzle and internal-insulation MERs are intentionally **not** invented; for a
complete solid inert figure the Zandbergen whole-stage regressions are
preferred. (This matches the "doesn't work very well yet" caveat on the
component-level solid approach in the source spreadsheet.)

---

## Reading the result as a structural coefficient

When a stated dry mass is supplied, the divergence report leads with the
**structural coefficient** ε = dry / (dry + propellant) implied by that mass
(and λ = dry/propellant for reference), then lists each method's *estimated* ε
next to the percentage divergence and verdict. This lets you judge a design in
the units you think in — "ε = 0.068; a composite stage of this size estimates
ε ≈ 0.065, so the stated mass is reasonable (+6 %)" — and, because tank material
moves the estimated ε, directly answer *is this dry mass plausible for this
material?* For dense-propellant and small stages ε runs higher; for large
hydrolox stages it falls (the Pietrobon `mp^0.848` size dependence).

## Accuracy and caveats

- Component MERs are tuned for **large launch-vehicle stages**. For small
  tactical-scale stages the fixed terms (engine `+59 kg`, avionics, wiring)
  dominate and the component total over-predicts — compare against the
  aggregate ε estimate in that regime.
- Every aggregate relation carries a 10–25 % standard error; a divergence
  inside ±15 % is "consistent", and a large negative divergence means the
  stated structure is lighter than any flown analogue (e.g. Peacekeeper-derived
  motors read as `optimistic`).
- The estimator says nothing about *feasibility of materials/processes* — only
  whether the mass is consistent with historical practice.

---

## Sources (collected in the project reference folders)

1. D. L. Akin, *Mass Estimating Relations*, ENAE 791, U. Maryland, 2016
   (`791S16L08.MERsx.pdf`). Underlying: Glatt WAATS (NASA CR-2420, 1974);
   MacConochie & Klich (NASA TM-78661, 1978); Heineman (NASA TN-D-6349, 1971;
   JSC-26098, 1994).
2. R. R. Rohrschneider, *Development of a Mass Estimating Relationship Database
   for Launch Vehicle Conceptual Design*, Georgia Tech / SSDL, 2002, and
   AIAA 2001-4542 (`RohrschneiderR-8900.pdf`, `rohrschneider2001.pdf`).
3. B. T. C. Zandbergen, *Simple mass and size estimation relationships of
   pump-fed rocket engines …*, 6th EUCASS, 2015 (`PaperLREenginemassandsizing.pdf`).
4. B. T. C. Zandbergen, *Simple Parametric Relations for Solid Rocket Stage
   Inert Mass Estimation*, TU Delft, 2026
   (`Revisit_of_solid_rocket_stage_inert_mass_estimation.pdf`).
5. S. S. Pietrobon, *Analysis of Propellant Tank Masses*, 2009
   (`382034main_…Analysis_of_Propellant_Tank_Masses.pdf`).
6. V. L. Hutchinson Jr. & J. R. Olds, *Estimation of Launch Vehicle Propellant
   Tank Structural Weight Using Simplified Beam Approximation* (GT-STRESS),
   AIAA 2004-3661, Georgia Tech SSDL, 2004 (`hutchinson2004.pdf`) — basis for
   the physics-based tank option.
7. J.-I. Shu, J.-W. Lee, S. Kim, et al., *Multistage Liquid Rocket Weight
   Estimation and Optimization for Early Design Stages*, J. Aerospace Eng.
   33(6), 2020 — basis for the engine-mass-ratio method (code:
   github.com/jshu004/Rocket-Weight-Estimation-and-Optimization).
8. P. Goldyn, A. Marwege, et al. (DLR), *Preliminary Design of Expendable and
   Reusable Mixed-Staged Launch Vehicles*, J. Spacecraft & Rockets, 2025
   (`goldyn-et-al-2025…pdf`) — structural-index feasibility ceiling.
9. M. D. Scher & D. North, *The Space Propulsion Sizing Program* (SPSP),
   NIA / Georgia Tech (`10.1.1.588.5523.pdf`) — pressure-vessel tank sizing
   with multi-estimate averaging; validates the physics tank approach.
10. *Northrop Grumman Propulsion Products Catalog* (Jan 2023, pp. 9–39) — per-motor
    data sheets for 29 Orion/Castor/GEM solid motors; basis for the "Lewis 2026
    (NG catalog)" best-in-class solid inert-mass regression. Case-material
    classification from external Pegasus/Castor/GEM literature.
10. D. M. Gaspar, *A Tool for Preliminary Design of Rockets*, IST Lisbon, 2014
    (`Thesis.pdf`) — independent confirmation of the Akin MER coefficients.
11. J. B. Nowell Jr., *Missile Total and Subsection Weight and Size Estimation
    Equations*, NPS thesis, 1992 (`a256081.pdf`) — tactical-missile regressions,
    consulted for context.
