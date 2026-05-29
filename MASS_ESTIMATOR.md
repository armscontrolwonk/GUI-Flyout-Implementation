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
| Avionics | `M = 10·M₀^0.361` | M₀ kg gross |
| Wiring | `M = 1.058·√M₀·L^0.25` | M₀ kg, L m |

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

### Liquid — aggregate relations

- **Pietrobon (2009) LOX/LH₂ stage-mass power law** (stage mass *less engines*,
  tonnes): `ms = a·mp^0.848`, `a = 0.19` (all-stages average), `0.1583`
  (common bulkhead, through Saturn S-II), `0.1171` (Al-Li 2195 common bulkhead).
  Strictly valid only for hydrolox; shown only when the propellant is LOX/LH₂.
- **Structural coefficient** `ε = m_inert / (m_inert + m_prop)` ⇒
  `m_inert = ε·m_prop/(1−ε)`. Default ε = 0.08.

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
skirts, TVC, avionics, separation). A propellant-mass-fraction option
`m_inert = m_prop·(1/ζ − 1)` is available for ad-hoc comparison.

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
6. J. B. Nowell Jr., *Missile Total and Subsection Weight and Size Estimation
   Equations*, NPS thesis, 1992 (`a256081.pdf`) — tactical-missile regressions,
   consulted for context.
