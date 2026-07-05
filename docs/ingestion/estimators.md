# Estimator-table spec — "fixing" incomplete data during ingest

This is the data behind the **resolution ladder** (`given → derived → estimated →
default → unresolved`). It says, for every fillable field, *how* a missing value is
recovered and *how much to trust the result*. Every value the resolver fills is
tagged in `provenance` (see `missile.schema.json`) so nothing is filled silently.

The resolver is **deterministic and auditable**: same input → same output → same
provenance. No model-by-model guessing in code; all heuristics live in the tables
below so they can be reviewed, versioned, and improved by the community.

---

## 0. Confidence scale

| Confidence | Meaning | Report treatment |
|-----------:|---------|------------------|
| `1.0` | `given` or exactly `derived` from given fields | silent |
| `0.7` | `estimated` from a tight, well-sourced relationship | noted |
| `0.5` | `estimated` from a class/propellant typical (wide band) | **VERIFY** flag |
| `0.3` | last-resort `default` standing in for a required field | **VERIFY** + lowers completeness |
| `0.0` | `unresolved` | blocking if required |

`completeness = Σ(weightᵢ · confidenceᵢ) / Σ(weightᵢ)` over the scored fields,
where required fields carry higher weight than cosmetic ones.

---

## 1. Derivations (deterministic — confidence 1.0)

Applied first, whenever the inputs they need are present. These are exact, not
guesses.

| Target field | Rule (`method`) | Formula |
|--------------|-----------------|---------|
| `thrust_N` | `isp_relation` | `Isp · g₀ · m_prop / t_burn`  (g₀ = 9.80665). Thrusty does this internally regardless; we record it for cross-check. |
| `mass_final` | `m0_minus_prop` | `mass_initial − mass_propellant` |
| `mass_propellant` | `mpf_known` | `mass_initial · MPF`  (when a mass fraction is given but prop mass isn't) |
| `mass_initial` | `final_plus_prop` | `mass_final + mass_propellant` |
| `length_m` | `ld_ratio` | `LD · diameter_m`  (when an L/D ratio is supplied instead of length) |
| `rv.beta_kg_m2` | `beta_from_geometry` | `m / (C_d · A)`, `A = π·d²/4`, `C_d` from RV shape (cone ≈ 0.10–0.25 hypersonic) |
| `boosters.length_m` | `two_diameters` | `2 · diam_m` (matches Thrusty's own fallback) |
| `nozzle_exit_area_m2` | (leave 0) | 0 triggers the legacy 2% back-pressure model — acceptable default, not an estimate |

A derivation only fires if **all** its inputs are themselves at confidence ≥ 0.7,
so estimates don't silently cascade into apparently-exact derived values (the
derived field inherits `min(input confidences)` when any input was estimated).

---

## 2. Isp by propellant type (estimated — confidence 0.5)

Keyed off `stage.propellant_type`. Midpoint used as the fill value; band recorded
in the provenance note so reviewers see the uncertainty.

> **Canonical source.** The `Reference` sheet in `booster_xlsx.py` is the single
> source of truth for these Isp/mass-fraction bands. The table below is a copy
> for readability and MUST be regenerated from that sheet rather than edited
> independently, so the two cannot drift.

| `propellant_type` | Isp band (s) | Fill (s) | Typical MPF | Notes |
|-------------------|-------------:|---------:|-------------|-------|
| `solid_composite`   | 230–290 | 260 | 0.85–0.92 | HTPB/AP; most modern SRBM/MRBM |
| `solid_double_base` | 200–240 | 220 | 0.80–0.88 | older / smaller motors |
| `liquid_storable`   | 280–320 | 295 | 0.85–0.92 | N₂O₄/UDMH or MMH; Scud, Nodong, DF-series |
| `liquid_lox_rp1`    | 340–360 | 350 | 0.88–0.94 | kerosene; fuelled at launch |
| `liquid_lox_lh2`    | 420–460 | 440 | 0.86–0.92 | highest Isp; SLV upper stages |
| `liquid_lox_ch4`    | 340–380 | 360 | 0.87–0.93 | methalox; newer systems |
| `unknown`           | — | fall through to §3 | — | use class default |

> **Sea-level vs vacuum.** These are vacuum figures (what Thrusty wants). If a
> source gives sea-level Isp for stage 1, the resolver bumps it by a class factor
> (≈ +8–12%) to a vacuum estimate and records `method: sl_to_vac`, confidence 0.5.

---

## 3. Class defaults (estimated — confidence 0.3–0.5)

Used when `propellant_type` is unknown too. Coarser; mainly to keep a model
*runnable* with a loud VERIFY flag rather than to be right.

| `missile_class` | assumed propellant | default Isp (s) | default stage MPF | default guidance |
|-----------------|--------------------|----------------:|-------------------|------------------|
| `srbm` | solid_composite | 260 | 0.88 | `pitch_program` |
| `mrbm` | solid_composite | 265 | 0.89 | `pitch_program` |
| `irbm` | liquid_storable  | 290 | 0.88 | `true_gravity_turn` |
| `icbm` | solid_composite (stage 1–2), storable (PBV) | 270 | 0.90 | `true_gravity_turn` |
| `slv`  | by stage: lox_rp1 → lox_lh2 upper | 320 / 440 | 0.91 | `orbital_insertion` (final) |
| `sounding` / `other` | unknown | 250 | 0.85 | `pitch_program` |

A glide vehicle has no class row: it is an RV (`rv_kind=glider`, §6b) on a booster
whose class is one of the rows above (e.g. an MRBM-class stack lofting an HGB).
Cruise is out of scope.

---

## 4. Aerodynamics defaults

| Field | Rule | Value / source | Confidence |
|-------|------|----------------|-----------:|
| `mach_table` / `cd_table` | `forden_default` | Mach `[0, 0.85, 1.0, 1.2, 2.0, 4.5]`, Cd `[0.20, 0.20, 0.27, 0.27, 0.20, 0.20]` | 0.5 |
| `nose_shape` | `class_typical` | RV/RBM → `cone`; SLV → `von_karman`; booster → `tangent_ogive` | 0.3 |
| `booster.cd` | `default` | 0.20 (tangent ogive) — matches Thrusty | 0.5 |
| `rv.nose_radius_m` | `default` | 0.05 m | 0.5 |
| `rv.emissivity` | `default` | 0.85 | 0.7 |

---

## 5. Geometry / β fallbacks

| Field | Rule | Notes | Confidence |
|-------|------|-------|-----------:|
| `rv.beta_kg_m2` | `beta_from_geometry` (§1) if shape+diam+mass present | else class typical below | 1.0 / 0.4 |
| `rv.beta_kg_m2` | `class_typical` | RV ≈ 30 000–100 000; HGB ≈ 5 000–15 000; decoy ≈ 100–1 000 | 0.4 |
| `length_m` (stage) | `ld_ratio` if LD given | else `class_typical` LD (booster ≈ 8–12, upper ≈ 2–4) × diameter | 1.0 / 0.4 |
| `payload_diameter_m` | `use_body_diameter` | 0 → body diameter (Thrusty's own behaviour) | 1.0 |

---

## 6. Validation / sanity checks (run after resolution)

These don't *fill* values — they catch physically impossible or implausible models
and emit warnings/errors into the report.

| Check | Severity | Condition |
|-------|----------|-----------|
| prop < initial | **error** | `mass_propellant ≥ mass_initial` |
| final positive | **error** | `mass_initial − mass_propellant ≤ 0` |
| burn positive | **error** | `burn_time_s ≤ 0` |
| Isp plausible | warn | Isp outside 150–480 s |
| stage MPF plausible | warn | MPF outside 0.5–0.96 |
| T/W at ignition | warn | stage 1 (+boosters) liftoff T/W < 1.1 or > 4 |
| upper-stage T/W | info | upper-stage T/W < 0.5 (often fine in vacuum, but flag) |
| ΔV plausibility | info | total Tsiolkovsky ΔV wildly off for the class (ICBM ≈ 6–7.5 km/s) |
| payload balance | warn | `bus_mass + num_ros·rv_mass` differs from `payload_kg` by > 10% |
| stage ordering | warn | upper stage heavier than the stage below it |

Errors block catalog inclusion; warnings lower `completeness` and surface in the
report with a VERIFY flag; info is logged only.

---

## 6b. Reentry vehicles (separate resolver pass)

RVs are resolved independently (they're shared via `ro_library`). Required: `name`,
`rv_kind`, `beta_kg_m2`. The `rv_kind` discriminator controls which fields are
required vs. inherited — getting this right is what prevents false "missing data"
flags.

| `rv_kind` | also required | inherited (DO NOT flag/estimate) |
|-----------|---------------|----------------------------------|
| `ballistic` | `mass_kg` | — |
| `marv_body` | maneuver props | `mass_kg`, `diameter_m`, `length_m` (from missile last-stage burnout via `effective_ro()`) |
| `glider` | `glider_LD`, `glider_guidance` (+`glider_beta_entry_kg_m2` if Acton) | — |
| `decoy` | `mass_kg` | — |

β resolution ladder:

| Rung | Method | Notes | Confidence |
|------|--------|-------|-----------:|
| given | — | always preferred | 1.0 |
| derived | `beta_from_geometry` | β = m/(C_d·A); hypersonic C_d-by-shape is uncertain | 0.4 ← VERIFY |
| estimated | `rv_kind_typical` | ballistic 30 000–100 000; marv lower; glider 5 000–15 000; decoy 100–1 000 kg/m² | 0.4 ← VERIFY |

Heating defaults (`nose_radius_m`=0.05, `emissivity`=0.85) and `glider_*` defaults
(`pullup_g_max`=10, `terminal_alt_km`=30, `aero_model`=constant_LD) are filled
**silently** (confidence ≥ 0.7) — they have sound physical defaults.

## 6c. Satellites / orbital payloads (NOT RVs)

A payload with `terminal_mode: orbital` does not reenter and has **no RV**.
`effective_ro()` returns `None` by design. The resolver behaves asymmetrically:

- **Skip the entire RV pass.** Absence of `beta_kg_m2`, `rv`, heating fields is
  *correct, not a gap* — never flag it, never estimate it, never let it lower
  `completeness`.
- **Required instead:** `payload.payload_kg` (satellite mass) and a final stage with
  `guidance: orbital_insertion`. `target_orbit_alt_km` optional (omit → report the
  naturally achieved orbit; solid final stages run to burnout regardless).
- **Default fill:** `terminal_mode` ← `orbital` when `missile_class='slv'`, else
  `reentry`.

Orbital sanity checks (replace the impact-side checks):

| Check | Severity | Condition |
|-------|----------|-----------|
| reaches orbit | **error** | achieved ΔV well short of orbital velocity (~7.8 km/s LEO) / perigee ≤ 0 |
| "doesn't reenter" | warn | perigee < `STABLE_ORBIT_PERIGEE_KM` **or** `orbital_lifetime_estimate` below horizon → "decays within N orbits, not a stable orbit" |
| eccentricity | info | flag highly elliptical vs intended circular |
| target met | info | achieved orbit vs `target_orbit_alt_km` |

The "doesn't reenter" threshold is a **single global pipeline constant**
(`STABLE_ORBIT_PERIGEE_KM`, default **200 km**) — *not* a per-model field and
*not* a runtime toggle. It is a QA-only lint that fires solely on
`terminal_mode: orbital` catalog entries at ingest time and never affects a
trajectory or anything shown in a normal run. A per-model override is
deliberately omitted for now; add one only if a real exception (e.g. a
deliberately-decaying tech demo) appears.

Note: every LEO orbit eventually decays. "Doesn't reenter" within Thrusty's scope
means the run **terminates at insertion and reports orbital elements + a decay
lifetime estimate**, rather than propagating to an impact point.

## 7. Worked example (what the resolver emits)

Source doc gives, for a notional 2-stage solid SRBM: name, class=`srbm`, both
stages' `mass_initial`, `mass_propellant`, `diameter_m`, `burn_time_s`, and a body
length for stage 1 only. No Isp, no thrust, no stage-2 length, no RV β.

```
scud-like.thrusty.json     COMPLETENESS 0.71   ✓ importable
  GIVEN     name, class, S1/S2 mass_initial, mass_propellant, diameter, burn_time; S1 length
  DERIVED   S1/S2 mass_final (m0_minus_prop), S1/S2 thrust_N (isp_relation)
  ESTIMATED S1/S2 isp_s = 260 s (solid_composite typical, conf 0.5)   ← VERIFY
            S2 length_m = 2.3 m (class LD 3 × diameter, conf 0.4)     ← VERIFY
            rv.beta_kg_m2 = 800 (srbm class typical, conf 0.4)         ← VERIFY
            cd_table = Forden defaults (conf 0.5)
  DEFAULT   nozzle_exit_area_m2 = 0 (legacy back-pressure)
  WARN      S1 liftoff T/W = 4.3 (>4 — check thrust/mass)
```

The model runs immediately; the three VERIFY lines are exactly the to-do list a
contributor (or I) works down to raise the model's completeness over time.
