# SWERVE object revision — questionnaire

Working document for revising `ro_library/SWERVE.ro.json`. Fill it in
in place: tick one box per question with an `x`, write into the **Notes**
blocks. Nothing here changes the repo on its own.

Ordered by how far each answer propagates — **mass first, because β is
derived from it**, so Q2 moves Q3 and everything downstream.

Provenance grades used below:

| Grade | Meaning |
|---|---|
| **sourced** | traced to a citable document |
| **assumed** | standard estimator or generic default |
| **fit** | reconstruction fit — no primary source |

Current state: 23 fields set, 34 at defaults.
Companion file: `reentry_plans/SWERVE.reentryplan.json`.

---

## 01 · `name` — which vehicle is this object meant to be?

**Grade: identity conflict**

| | |
|---|---|
| this file | Sandia SWERVE · Strypi · Kauai→Johnston 1979–85 · 2.6 m · 5.25° half-angle · "small wings/elevons" |
| `docs/cl_margin_references.md` §3 | "SWERVe, the public C-HGB predecessor" (Gulan 2024 Tbl 2) · 2.75 m · 5° · 0.87 m span · 4 fins · US Navy booster |

The repo currently holds two different vehicles under one name. The object file
is unmistakably *Sandia's* SWERVE — Winged Energetic Re-entry Vehicle, three
Strypi flights, Williamson's AIAA 92-3989 flight-3 record. But the reference
base describes a "SWERVe" with a Navy booster and four fins, and cites it as the
C-HGB's ancestor. The dimensions are close enough to look like the same airframe
and different enough that they cannot both be this file.

**Propagates to:** every other answer here. Nothing below can be settled until
this is.

- [ ] Sandia SWERVE (1979–85, Strypi). The Gulan §3 entry is a *different*
      vehicle and should be renamed in the reference base.
- [ ] The C-HGB predecessor. Re-source the object file to Gulan Table 2.
- [ ] Same airframe lineage — keep one object, reconcile the dimensions, note
      both sources.
- [ ] Split into two objects (`SWERVE.ro.json` plus a C-HGB predecessor).

**Notes** — which dimension set governs, and what should the other entry be called?

```


```

---

## 02 · `mass_kg` — can 300 kg be sourced, or does it stay a fit?

**Grade: fit**

| | |
|---|---|
| current | 300.0 kg |
| rests on | trajectory reconstruction — "no primary-source SWERVE mass has been found" |
| rejected | 450 kg (program-class figure) — under-flew the ~1,180 km flight-3 range and the Mach-12 pull-out |

This is the file's largest admitted hole, and its notes are self-aware about it:
300 kg was chosen because it reproduces the flight-3 range to about 10 %, not
because a document says so. Fitting a mass to a range and then using the object
to check ranges is circular — defensible as a stated assumption, not as evidence.

**Propagates to:** β directly (β = m/C_D·A, rescaled by 300/450 to hold C_D·A
constant), and through β into every heating, deceleration and range result.

- [ ] Keep 300 kg as an explicit fit; sharpen the notes so no reader mistakes it
      for sourced.
- [ ] I have a sourced mass — given below.
- [ ] Revert to 450 kg and accept the range shortfall.
- [ ] Carry a band (e.g. 300–450 kg) rather than a point value.

**Notes** — value and citation:

```


```

---

## 03 · `beta_kg_m2` — is C_D·A held fixed, or is β set independently?

**Grade: assumed (derived)**

| | |
|---|---|
| current | 24,733 kg/m² |
| construction | m / (C_D·A) with C_D·A = 0.0121 m² held fixed |
| implied C_D | ≈ 0.068 on the 0.178 m² cone base — "a realistic sharp-5.25°-cone hypersonic value" |

β was not measured; it was rescaled from 37,100 when the mass moved 450 → 300,
holding C_D·A constant. That makes β a function of Q2. The separate question is
whether C_D ≈ 0.068 is right — it is quoted as realistic rather than derived
from the repo's own cone drag routines.

**Propagates to:** deceleration, peak-heating altitude, the whole down-leg. A
wrong β moves the flown trajectory more than a wrong L/D does.

- [ ] Keep C_D·A = 0.0121 m² fixed and rescale β with whatever mass Q2 settles.
- [ ] Recompute C_D from the repo's cone drag model rather than carrying 0.068.
- [ ] Set β from an independent source and let C_D·A fall out.

**Notes** — any source for C_D or β, or a preferred derivation path:

```


```

---

## 04 · `glider_LD` — which L/D regime should the shipped object carry?

**Grade: fit (by direction)**

| | |
|---|---|
| current | 1.8 — "the sustained-glide L/D set here (per user direction)" |
| flight record | ≈ 0.6 maneuvering, from the "small wings" (Williamson AIAA 92-3989, Fig. 20) |
| rationale | 1.8 matches the AHW glide body of the same airframe |

Two defensible numbers a factor of three apart, describing different things:
what the vehicle demonstrated while maneuvering, versus what the airframe can
trim to in sustained glide. The file ships the higher one on your instruction,
with the flight value recorded only in prose. For a tool whose purpose is
checking reported claims against open evidence, which number belongs in the
object matters.

**Propagates to:** glide range (roughly linear in L/D), and `commanded_LD: 1.8`
in the reentry plan, which must move with it.

- [ ] Keep 1.8 (sustained glide / AHW-matched), with 0.6 in the notes.
- [ ] Ship 0.6 — the flight-demonstrated value — as the conservative default.
- [ ] Two objects: a maneuvering SWERVE and a sustained-glide variant.
- [ ] A different value or band — below.

**Notes** — if 1.8 stays, what is the citation for the sustained-glide figure on
*this* airframe?

```


```

---

## 05 · `glider_control_surfaces` — should the tier stay `"unknown"`?

**Grade: assumed**

| | |
|---|---|
| current | `"unknown"` → damping estimator returns its widest band (ζ ≈ 0.35, 0.00–0.75) |
| documented | "small wings/elevons for lift and control, Mach 2–14 maneuvering" |
| tiers | none 0° · small 5° · substantial 15° · unknown 10° (reported as an assumption) |

Every shipped reentry object declares `"unknown"`, so nothing is lost by leaving
it — but SWERVE is one of the few with a documented control architecture. A
vehicle described as flying elevons through Mach 2–14 is not really an unknown.
Declaring `small` would narrow the damping band from a shrug to an estimate.

**Propagates to:** the phugoid damping ζ. It does **not** currently reach the
trajectory — that path needs `separation_mode "body"` with `glider_LD` at 0, and
this is a separating RV with L/D 1.8.

- [ ] Leave `"unknown"` — the widest band is the honest answer.
- [ ] `"small"` — "small wings/elevons" is close to a direct quote.
- [ ] `"substantial"` — Mach 2–14 maneuvering implies real authority.

**Notes** — anything in Williamson or Murbach that sizes the elevons?

```


```

---

## 06 · `glider_flap_deflection_deg` — enter the documented ~4°?

**Grade: documented but unused**

| | |
|---|---|
| current | 0.0 in the reentry plan → estimator falls back to its 12° default |
| documented | "~4 deg control deflection" — Williamson Fig. 20, flight-3 maneuver |

The object's own `source` field records a measured deflection from the flight
record, and the plan then ignores it in favour of a generic 12°. That is a
sourced number sitting unused next to an unsourced one — the exact pattern that
produced the Kumar & Stollery problem.

**Propagates to:** ΔC_L and hence ζ in the damping estimator. 4° against the 12°
default is a large reduction in commanded margin.

- [ ] Set 4.0° from the Williamson figure and cite it.
- [ ] Leave 0 — 4° was one maneuver, not the vehicle's usable travel.
- [ ] Re-read Fig. 20 first; not sure 4° is the peak deflection.

**Notes** — is ~4° the commanded peak, the trim offset, or something else?

```


```

---

## 07 · `wing_*` and `body_span_m` — the winged vehicle has no wings entered

**Grade: fit (empty)**

| field | value |
|---|---|
| `wing_area_m2` | 0.0 |
| `wing_span_exposed_m` | 0.0 |
| `wing_root_chord_m` / `wing_sweep_deg` / `wing_aspect_ratio` | 0.0 / 0.0 / 0.0 |
| `body_span_m` | 0.0 |
| `n_wings` | 4 (default, not stated in the file) |
| available | Gulan Tbl 2: 0.87 m span, 4 fins, lift referenced to *fin* area |

The W in SWERVE is "Winged", and the factsheet ~0.61 m diameter is explicitly
noted as including the wings — yet every lifting-surface field is zero. For a
separating RV these feed the 3-D depiction and the drag polar rather than the
derived L/D, so nothing is currently *wrong*; the geometry is simply absent, and
the object cannot be drawn or re-derived from its own shape.

**Propagates to:** the schematic and 3-D views now; the derived-L/D path if this
object is ever flown in body mode.

- [ ] Populate from Gulan Table 2 (0.87 m span, 4 surfaces) — if Q1 says that is
      this vehicle.
- [ ] Populate from a Sandia source supplied below.
- [ ] Derive from the ~0.61 m over-wing width and the 0.476 m cone base.
- [ ] Leave empty — better blank than invented.

**Notes** — span, chord, sweep, count, and where each comes from:

```


```

---

## 08 · `nose_radius_m` — is the R_n/R_base = 0.07 ratio sourced?

**Grade: assumed (derived from a ratio)**

| | |
|---|---|
| current | 0.0167 m |
| construction | R_base = L·tan(5.25°) ≈ 0.238 m, × published ratio 0.07 |

The arithmetic is clean and reproducible. What the notes do not say is *which*
publication gives 0.07 — it is described as "published" without a citation, the
same provenance shape as the figure that turned out to be wrong elsewhere in
this repo.

**Propagates to:** stagnation heating (q ∝ R_n^−1/2), the transition gate's
Re_Rn, and the TPS verdict.

- [ ] I can cite the 0.07 ratio — below.
- [ ] Can't cite it; mark it an assumption in the notes.
- [ ] Replace with a directly measured nose radius.

**Notes** — where does R_n/R_base = 0.07 come from?

```


```

---

## 09 · `body_tps_thickness_m` and `structure_*` — materials sourced, rest unset

**Grade: fit (empty)**

| field | value |
|---|---|
| `nose_tps_material` | `carbon_carbon` — **sourced**, Murbach 1993 p.3 |
| `body_tps_material` | `silica_phenolic` — **sourced**, Murbach 1993 p.3 |
| `body_tps_thickness_m` | 0.0 |
| `structure_material` | `""` (empty) |
| `structure_limit_K` | 0.0 |

The materials are among the best-sourced things in the file. The thickness and
the structural limit behind them are blank, which caps what the survivability
report can say — a bondline verdict needs a thickness and a backface limit, not
just a material name.

**Propagates to:** the bondline calculation and the survivability report's
burn-through bound.

- [ ] I have a thickness and/or structure material — below.
- [ ] Use the repo's screening-tier default for a silica-phenolic RV of this class.
- [ ] Leave blank; accept the reduced survivability output.

**Notes** — thickness, structure material, backface limit, with citation:

```


```

---

## 10 · `glider_beta_entry_kg_m2` — where does 20 kg/m² come from?

**Grade: assumed (inert)**

| | |
|---|---|
| current | 20.0 kg/m² |
| read by | `equilibrium_glide_acton` only — *ignored* under this object's plan (`dynamic_equilibrium_glide`) |
| reference value | Acton's HTV-2 fit gives β_S ≈ 7 kg/m² (Table 3, p. 206) |

A set value that nothing currently reads, roughly three times the only cited
figure for the same quantity, with no note explaining it. Harmless today because
the guidance mode never asks for it — and exactly the kind of dormant number
that becomes load-bearing the moment someone switches modes.

**Propagates to:** nothing today. Phase-3 entry drag if the plan is ever
switched to the Acton mode.

- [ ] 20 is deliberate — reasoning below.
- [ ] Replace with Acton's ≈ 7 kg/m² and cite it.
- [ ] Set to 0 (disable Phase 3) rather than carry an unexplained value.

**Notes** — origin of the 20, if known:

```


```

---

## 11 · `emissivity` — vehicle-specific or house default?

**Grade: assumed (generic)**

| | |
|---|---|
| current | 0.85 |
| surfaces | bare carbon-carbon (nose, leading edges) · silica phenolic (body) |

0.85 is a reasonable screening value and widely used, but the object carries two
quite different surfaces and one number. Worth a line in the notes saying which
it represents, even if the value does not change.

**Propagates to:** radiative equilibrium wall temperature, and so the TPS tier
verdict.

- [ ] Keep 0.85; add a note that it is a screening default, not measured.
- [ ] Set to the carbon-carbon value — the nose drives the verdict.
- [ ] Different value — below.

**Notes:**

```


```

---

## Anything else — sources in hand, or fields not asked about

If you have Murbach 1993, Williamson AIAA 92-3989, or a Sandia SWERVE factsheet
to hand, say which — several answers above collapse instantly with the document
open.

**Notes** — documents available, other fields to revisit, or how you want the
`notes` field rewritten:

```


```
