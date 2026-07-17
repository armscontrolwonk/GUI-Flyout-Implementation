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

## Why NO Reentry-F point back-out was performed

The plan's Phase 2 step 1 suggests backing `H_eff` out of the Reentry-F graphite
nosetip. **This was deliberately not done, for two rule-compliant reasons:**

1. Reentry-F is wired in the repo only as a **δ/R_n shape-change ladder** anchor
   (survivability_report.py: ~0.7 R_n radial blunting survived, ~7.7 R_n axial
   solid-tip length; NASA CR-154044 / Berry). There is **no wired
   integrated-heat-load Q paired with a measured graphite recession δ** to invert
   `H_eff = Q/(ρ·δ)` from.
2. The plan §6 hazard is explicit that the Reentry-F nosetip recession history is
   **uncertain after ~60,000 ft** (thermochemical-only vs. mechanical-erosion vs.
   worst-case radius histories). Manufacturing a single-point back-out from
   paywalled secondary numbers would violate the standing rule ("do not invent
   citations; every number must trace to a source or be flagged"). A fabricated
   precise calibration is worse than an honest band.

The honest substitute is the literature cross-check below plus the Phase 3
bounding tests, which together confirm the nominals are conservative and
not wild — the actual acceptance criterion the plan names ("confirm the tuned
value is not wild").

## H_eff bands (replaces the bare point placeholders)

`nominal` = the retained screening value (kept stable so verdicts don't shift and
the Phase 3 bounds stay valid). `low`/`high` = literature-informed engineering
spread. **These are conservative screening constants, NOT fits.**

| material | ρ (kg/m³) | H_eff low | **nominal** | H_eff high | basis / provenance |
|---|---|---|---|---|---|
| carbon_phenolic | 1450 | 10 | **15** | 30 | flight-regime CP effective-heat-of-ablation band ~10–30 MJ/kg (plan §Phase 2 handbook guidance; enthalpy-dependence corroborated by CP/PICA arc-jet literature above). Nominal 15 at the conservative low end. |
| pica | 270 | 25 | **35** | ~100+ | PICA Q\* is higher than CP and rises sharply with enthalpy (peak "enthalpy of ablation" figures reach the hundreds of MJ/kg at Orion/return enthalpies). Screening nominal 35 is a deliberately conservative low-regime value — it over-predicts Stardust ~7× (Phase 3), vs FIAT's ~1.5×, which is *safe* for a screen. |
| carbon_carbon | 1800 | 25 | **40** | 60 | bare C/C nosetip, oxidation→sublimation regime ([OSTI: carbon/graphite ablation correlation for RV nosetips](https://www.osti.gov/biblio/4729765); [NTRS 19790010869, C/C nosetip ablative performance](https://ntrs.nasa.gov/search.jsp?R=19790010869)). Specific endpoint values NOT retrieved from a single table (sources paywalled/403) — band is engineering-judgement bracketing the sublimation regime, flagged as such. |

**Provenance honesty:** the *band endpoints* are literature-informed engineering
brackets, not values lifted from one retrieved table (the authoritative Q\*-vs-enthalpy
curves — FIAT/PICAv3.3, the OSTI carbon-graphite correlation — are paywalled/403
this session). The *direction* and *magnitude sanity* (CP ~10–30, PICA higher,
Q\* enthalpy-dependent) ARE literature-grounded. The nominals are unchanged from
the prior screening values, now justified as conservative-low rather than
arbitrary, and independently bound-checked in Phase 3.

## Acceptance check

- Nominals within literature bands? **Yes** (CP 15 ∈ [10,30]; PICA 35 conservative-low;
  C/C 40 in sublimation-regime bracket).
- Direction conservative (over-predict)? **Yes** — Phase 3 bounds: Stardust 7.2×,
  Hayabusa 44×, both predicted ≥ measured.
- Reentry-F reproduced within radius-history spread? **N/A** — no wired Q+δ pair;
  documented above rather than fabricated.
