# Form A ablator — Phase 2: H_eff calibration & uncertainty bands

Goal (plan §Phase 2): retire the bare `H_eff_MJ_kg` screening *guesses* by
confirming they are not wild against the flight/handbook literature, and emit
`{low, nominal, high}` bands instead of naked point values. **Non-goal (plan §2,
§5): do NOT tune H_eff to the recovered capsules.**

## The unit trap: "effective heat of ablation" is enthalpy-dependent, not a constant

The recession screen uses `δ = Q / (ρ · H_eff)`, so `H_eff` here is the
**effective heat of ablation Q\*** — the heat load absorbed per unit mass removed
— in the flight-relevant boundary-layer enthalpy regime. It is **not** a single
material constant:

- Q\* rises steeply with boundary-layer enthalpy: arc-jet characterization of
  carbon-phenolic-class ablators spans freestream stagnation enthalpies
  ~0.8–30 MJ/kg, and HARLEM/PICA arc-jet campaigns run to ~70 MJ/kg enthalpy
  ([HARLEM, *Sci. Rep.* 2023](https://www.nature.com/articles/s41598-023-40351-x);
  [PICAv3.3 arc-jet validation, *J. Spacecraft & Rockets*, DOI 10.2514/1.42949](https://arc.aiaa.org/doi/10.2514/1.42949),
  107 W/cm²@2.3 kPa → 1100 W/cm²@84 kPa).
- So any single `H_eff` is a **regime-specific engineering value**, and a
  screening model should sit at the **low (conservative) end** of the band:
  lower `H_eff` → *more* predicted recession → the model over-predicts, which is
  exactly the bounding direction the Phase 3 capsule tests enforce.

## Reentry-F back-out: a derived BRACKET (source obtained), not a calibration

Initially declined for lack of a paired Q+δ source; the source then arrived:
**Berry, "Deep Dive of Reentry F Nose Tip Step and Gap" white paper v2** (NASA
Langley; in the project Google Drive, `ReentryF_White_Paper_v2.pdf`), which
reproduces the primary-report numbers and figures (NASA CR-154044, LWP-460,
TM X-1856 Fig. 11).

**Cited inputs** (all via the white paper's quotes/figures):
- Nosetip: ATJ graphite shell, initial R_n 0.1 in, 8.5 in long [white paper §intro].
- Axial stagnation recession: **0.77 in = 19.6 mm at 49,000 ft** with nose radius
  0.171 in [CR-154044 quote — preflight prediction, consistent with the
  TM X-1856 postflight curve-1/curve-2 band, see below].
- Test-window environment, 100,000→50,000 ft: stagnation heating **9,000–28,000
  BTU/ft²·s = 102–318 MW/m²**, stagnation pressure 5–60 atm, enthalpy ~8,000
  BTU/lbm ≈ 18.6 MJ/kg [LWP-460 nominal-trajectory figure].
- Window duration: **~12–14 s** (TM X-1856 Fig. 11 time axis spans 448–462 s;
  the 60,000 ft anomaly is at 458.7 s) — read from figure, flagged.
- ATJ density ~1.73 g/cc (vendor-nominal, flagged; model's carbon_carbon uses 1800).

**Bracket arithmetic** (ours, deliberately widest — no time-averaging assumption):
`Q ∈ [102 MW/m² × 12 s, 318 MW/m² × 14 s] = [1.2, 4.5] GJ/m²`, so
`H_eff = Q/(ρ·δ) ∈ [1.2e9, 4.5e9]/(1730 × 0.0196) ≈` **36–130 MJ/kg** for
flight-regime graphite (oxidation + mechanical-erosion, 5–60 atm).

**Reading:** the model's carbon_carbon nominal **40 sits at the low (conservative)
edge** of the flight-derived bracket — the screen over-predicts recession
in-envelope too, same sign as the capsule bounds.  ⚠ This is a *derived bracket*
(inputs cited, arithmetic ours, spread carried); it is **not** a point
calibration, and the nominal was not changed.

**Radius-history spread, now quantified** (TM X-1856 Fig. 11, read from the
white paper's reproduction): curve 1 (thermochemical-only) ends near
R_n ≈ 0.17–0.2 in; curve 2 (mechanical-erosion-corrected) near ~0.3 in;
curve 3 (worst case, monotonic growth to the 0.5 in plug-exposure radius at
458.7 s) is **refuted** by the report itself (plug exposure would have shown in
thermocouples, body motions, surface pressures); pressure-matching preliminary
estimates (with uncertainty bars) fall between curves 1 and 2.  So the
demonstrated-survival blunting spread is **R_n 0.10 → 0.17–0.30 in**
(~0.7–2 R_n radial growth), worst-case 0.5 in excluded.
Corroboration: Malta/Langley full-scale ablation tests measured graphite
recession rates within **±15% of theory** at sublimation conditions
(0.27/0.59 atm), with irregular stagnation shapes forming only at 6–10+ atm
[LWP-460 summary].

## H_eff bands (replaces the bare point placeholders)

`nominal` = the retained screening value (kept stable so verdicts don't shift and
the Phase 3 bounds stay valid). `low`/`high` = literature-informed engineering
spread. **These are conservative screening constants, NOT fits.**

| material | ρ (kg/m³) | H_eff low | **nominal** | H_eff high | basis / provenance |
|---|---|---|---|---|---|
| carbon_phenolic | 1450 | 10 | **15** | 30 | flight-regime CP effective-heat-of-ablation band ~10–30 MJ/kg (plan §Phase 2 handbook guidance; enthalpy-dependence corroborated by CP/PICA arc-jet literature above). Nominal 15 at the conservative low end. |
| pica | 270 | 25 | **35** | ~100+ | PICA Q\* is higher than CP and rises sharply with enthalpy (peak "enthalpy of ablation" figures reach the hundreds of MJ/kg at Orion/return enthalpies). Screening nominal 35 is a deliberately conservative low-regime value — it over-predicts Stardust ~5× (Phase 3), vs FIAT's ~1.5×, which is *safe* for a screen. **Cited arc-jet point:** Winter et al. AIAA 2014-1151 (mArc, NASA Ames) — flat-face flux 1036 W/cm² (10.36 MW/m², ±10%, converted from a 2575 W/cm² hemispherical probe), PICA recession rate 0.05–0.06 cm/s by tracer spectroscopy, corroborated by typical large-facility rates 0.05–0.1 cm/s at similar conditions, surface T ≥ 2800 K. Implied Q\* = q̇/(ρ·ṡ) with ρ_virgin = 270: **38–77 MJ/kg at ~10 MW/m²** (77 at 0.5 mm/s ↔ 38 at 1.0 mm/s). The nominal 35 sits at/below the low edge of this cited band → conservative-low is now *cited*, not just argued. (Caveats: cold-wall calorimeter flux; feasibility-demo rate estimate.) |
| carbon_carbon | 1800 | 25 | **40** | 60 | bare C/C nosetip, oxidation→sublimation regime ([OSTI: carbon/graphite ablation correlation for RV nosetips](https://www.osti.gov/biblio/4729765); [NTRS 19790010869, C/C nosetip ablative performance](https://ntrs.nasa.gov/search.jsp?R=19790010869)). Table endpoints remain engineering brackets, but the **Reentry-F flight-derived bracket 36–130 MJ/kg** (next section — inputs cited, arithmetic ours) now contains the nominal 40 at its conservative low edge. |

**Provenance honesty:** the CP and C/C *band endpoints* are literature-informed
engineering brackets, not values lifted from one retrieved table (the authoritative
Q\*-vs-enthalpy curves — FIAT/PICAv3.3, the OSTI carbon-graphite correlation — are
paywalled/403 this session).  The PICA band is the exception: the Winter 2014
arc-jet point above is a firsthand, cited Q\* datum (38–77 MJ/kg at ~10 MW/m²). The *direction* and *magnitude sanity* (CP ~10–30, PICA higher,
Q\* enthalpy-dependent) ARE literature-grounded. The nominals are unchanged from
the prior screening values, now justified as conservative-low rather than
arbitrary, and independently bound-checked in Phase 3.

## Acceptance check

- Nominals within literature bands? **Yes** (CP 15 ∈ [10,30]; PICA 35 conservative-low;
  C/C 40 in sublimation-regime bracket).
- Direction conservative (over-predict)? **Yes** — Phase 3 bounds: Stardust 5.1×
  (vs firsthand Core 1 = 5.7±0.3 mm, Kontinos & Stackpoole AIAA 2008-1197),
  Hayabusa 44×, both predicted ≥ measured.
- Reentry-F honored within radius-history spread? **Yes, as a bracket** — the
  Berry white paper (project Drive) supplied the paired environment + recession
  numbers; the derived H_eff bracket 36–130 MJ/kg contains the C/C nominal 40 at
  its conservative edge, and the TM X-1856 curve-1/2/3 spread is quantified
  (0.17–0.30 in best-supported, 0.5 in worst case refuted) rather than collapsed
  to a point.
