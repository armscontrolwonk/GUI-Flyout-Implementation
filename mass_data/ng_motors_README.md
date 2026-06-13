# Northrop Grumman Solid Motor Dataset

Per-motor data for the 29 individually-specified solid rocket motors in the
Northrop Grumman Propulsion Products Catalog (Jan 2023), catalog **pp. 9–39**
(Orion, Castor, and GEM families). Built to support a dry-mass / inert-mass
estimator and the supporting scaling analysis.

## Files
- `ng_motors.csv` / `ng_motors.json` — primary dataset, 29 rows (identical content).
- `ng_motors_external_supplement.csv` — partial expansion-ratio / chamber-pressure
  data from external sources. **Low confidence, incomplete — see warning below.**
- `ng_motors_README.md` — this file.

## Column dictionary (primary dataset)

| column | units | provenance | notes |
|---|---|---|---|
| `motor` | — | catalog | Motor name as printed |
| `diameter_in` | in | catalog | Motor diameter |
| `length_in` | in | catalog | Overall length incl. nozzle |
| `LD_ratio` | — | computed | length_in / diameter_in |
| `mass_loaded_lbm` | lbm | catalog | Total loaded ("total motor") weight |
| `mass_propellant_lbm` | lbm | catalog | Propellant weight |
| `mass_burnout_lbm` | lbm | catalog | Burnout weight (some marked "est") |
| `mass_inert_lbm` | lbm | computed | loaded − propellant (the "dry mass") |
| `inert_fraction` | — | computed | mass_inert / mass_loaded |
| `case_material` | — | research-classified | Steel / Graphite-epoxy / Carbon-epoxy / "(inferred)" |
| `case_class` | — | research-classified | Steel or Composite (the modeling group) |
| `stage_type` | — | research-classified | Upper or First — see caveat |
| `thrust_bt_avg_lbf` | lbf | catalog | Burn-time average thrust |
| `burn_time_s` | s | catalog | See `burn_time_basis` |
| `burn_time_basis` | — | catalog | `action_to_burnout` or `to_30_psia` (definition differs by sheet) |
| `nozzle_exit_dia_in` | in | catalog | Nozzle exit cone diameter |
| `expansion_ratio_catalog` | — | catalog | Present on only 2 sheets (Orion 32, 32XL); else null |
| `chamber_pressure_psia_catalog` | psia | catalog | Present on only 2 sheets; else null |

### Provenance tiers
- **catalog** — read directly from the catalog data sheets. High confidence.
- **computed** — arithmetic from catalog fields. High confidence.
- **research-classified** — assigned from external literature, not in the catalog:
  - `case_material`/`case_class`: Orion & GEM = graphite-epoxy composite; Castor IVA/IVA-XL/IVB = steel;
    Castor 120/120XL = carbon-epoxy composite; Castor 30/30B/30XL = composite (inferred from Castor 120 lineage).
  - `stage_type`: "Upper" = in-line, altitude-ignition, minimal airloads; "First" = first stage / strap-on.
    **Caveat:** Orion 32 and 32XL are development motors of ambiguous role; both classified "Upper" here.
    Reclassifying them does not change the modeling conclusions.

## Recommended estimator (ships with this data)

Power law fit to the 26-motor composite group, with L/D correction:

```
m_inert = k * mass_loaded^{b} * (L/D)^{c}        # lbm, inches
  k (composite) = 0.20049
  b (loaded mass exponent) = 0.8954
  c (L/D exponent) = 0.1668
  steel: multiply k by 1.31
  fit quality: R^2 = 0.9736, RMS error = 16.6%
```

Notes for the implementer:
- Exponent b is statistically indistinguishable from 1 (scale-invariant inert fraction).
- The L/D term is positive: slender motors carry MORE inert mass per unit size
  (surface-area-to-volume). It is significant (p<0.05) and lowers RMS from ~20% to ~16.6%.
- `stage_type` was tested and is NOT a useful predictor once L/D is included (redundant). Provided for reference only.
- Report estimates with a +/- ~17% (1-sigma) band. Worst single-motor miss is the Castor 30XL
  (still ~28% under-predicted after size+L/D — it sits at the efficient corner of the design space).
- `mass_burnout` ≈ `mass_inert`; the unburned-propellant sliver is not separately recoverable and is ≈ 0.

## WARNING on the external supplement
`ng_motors_external_supplement.csv` holds expansion ratios and chamber pressures
gathered from third-party aggregators (Encyclopedia Astronautix, press releases).
It is **incomplete** (8 of 29 motors) and **low confidence**: several chamber
pressures appear to be nominal family-level values (e.g. 58 bar repeated across all
Pegasus stages), not per-motor measurements. We concluded these data are too coarse
to test a chamber-pressure effect on dry mass. Do NOT merge them into the primary
dataset or treat them as catalog-grade. They are included only for traceability.
