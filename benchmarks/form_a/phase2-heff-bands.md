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

**Bracket arithmetic — now DIGITIZED from the clean nominal-trajectory figure.**
History, kept for the audit trail: the repo's earlier Reentry-F heat-load
analysis (`HEATING_TPS_REFERENCES.md`: TM X-2253/X-2560/X-2282 read from
primary) found the stagnation q̇(t) exists only as the preflight LWP-460 curve,
made an order-of-magnitude integration (~1 GJ/m²) from Berry's congested
reproduction, and decided **Q_MJ stays None**.  A first pass here used a
constant-flux bracket (Q ∈ [~1, 4.5] GJ/m² → H_eff 29–130 MJ/kg).  A clean
full-resolution copy of the figure ("Figure 1 — Nominal Reentry 'F' trajectory,
γ_E = 21.2°, V_E = 20,300 ft/s"; the figure Berry reproduces as his Fig. 6
[LWP-460]) was then supplied in-chat, and the curve was digitized point-by-point
— reads and integration in `reentryf_nominal_qdot.csv`:

- **Q ≈ 2.85 GJ/m² cold-wall stagnation (±~25% read error)**, superseding both
  the ~1 GJ/m² order-of-magnitude and the 4.5 GJ/m² ceiling (both bracket it).
- Peak q̇ ≈ 28.5×10³ Btu/ft²·s ≈ **324 MW/m² at ~431.5 s (~45–49 kft)** —
  confirms the `_BENCHMARKS` 318 MW/m² pin to within read error.
- **Window correction:** 100→50 kft takes **~8 s** (423→431 s at ~6.5–7 kft/s
  sink), not the 12–14 s previously read off the TM X-1856 axis span (that
  figure's axis covers more altitude; also note the nominal figure's time base
  runs ~20 s earlier than the actual flight's).

`H_eff = Q/(ρ·δ)` with ρ 1.73 g/cc and δ carrying the radius-history spread
(0.6–1.0 in axial, centered on CR-154044's 0.77 in = 19.6 mm):
**H_eff ≈ 50–135 MJ/kg, central ≈ 84 MJ/kg** for flight-regime graphite
(oxidation + mechanical-erosion, 5–60 atm).

**Reading:** with the digitized Q, the carbon_carbon nominal **40 over-predicts
the preflight-predicted recession ~2×** (41 mm vs 19.6 mm) — the conservative
sign is restored for the sharp-tip regime, somewhat beyond Schneider 72-705's
±25%/1.6× ablation-model spread but on the safe side for a screen.  The
first-pass statement that the screen might under-predict by ~25% rested on the
~1 GJ/m² order-of-magnitude read and is withdrawn with it.  ⚠ Still a derived
bracket (nominal-preflight environment, figure-read Q, δ spread carried);
**not** a point calibration; nominal unchanged; `_BENCHMARKS` Q_MJ stays None
(the 2.85 GJ/m² is a preflight prediction — no flight-measured stagnation
heating exists).
(TM X-2584 — uploaded in-chat and in the project Drive — firsthand-confirms the
~18 MJ/kg (8,000 Btu/lbm) total enthalpy and the Mach-20 edge conditions.)

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
| carbon_carbon | 1800 | 25 | **40** | 60 | bare C/C nosetip, oxidation→sublimation regime ([OSTI: carbon/graphite ablation correlation for RV nosetips](https://www.osti.gov/biblio/4729765), still unretrieved; **Nestler 1979, NTRS 19790010869 — now READ FROM PRIMARY**, PDF in repo `data/`, see "Severe-regime cap" below). Table endpoints remain engineering brackets for the moderate-pressure regime; the **Reentry-F flight-derived bracket 50–135 MJ/kg (central ≈84, digitized Q)** sits above the nominal 40, which over-predicts recession ~2× — conservative for a screen (next section). **Validity floor:** at stagnation pressures ≥80 atm the band does not apply — see the Nestler severe-regime cap. |

**Provenance honesty:** the CP and C/C *band endpoints* are literature-informed
engineering brackets, not values lifted from one retrieved table (the authoritative
Q\*-vs-enthalpy curves — FIAT/PICAv3.3, the OSTI carbon-graphite correlation — are
paywalled/403 this session).  The PICA band is the exception: the Winter 2014
arc-jet point above is a firsthand, cited Q\* datum (38–77 MJ/kg at ~10 MW/m²). The *direction* and *magnitude sanity* (CP ~10–30, PICA higher,
Q\* enthalpy-dependent) ARE literature-grounded. The nominals are unchanged from
the prior screening values, now justified as conservative-low rather than
arbitrary, and independently bound-checked in Phase 3.

## Severe-regime cap: Nestler 1979 (C/C at 80–168 atm) — read from primary

Nestler, "Ablative Performance of Carbon-Carbon Nosetips in Simulated Re-Entry
Environments" (GE RESD; NTRS 19790010869 / N79-19040; PDF in repo `data/`)
gives **measured steady-state recession rates for 3-D carbon-carbon** in the
AFFDL 50 MW arc and HIP facility — the verbatim table (p. 400):

| facility | P_s (atm) | H_CL kJ/kg (Btu/lb) | ṡ (cm/s) | cone θ | T_w (K) | Ch/Cho |
|---|---|---|---|---|---|---|
| 50 MW | 80 | 11,600 (5,000) | 0.635 | 45° | 4,000 | 1.4 |
| HIP | 124 | 6,914 (2,980) | 0.508 | 57° | 4,167 | 1.4 |
| HIP | 168 | 8,027 (3,460) | 0.787 | 57° | 4,167 | 1.5 |

**Derived implication (⚠ labeled, assumptions stated):** using the paper's own
steady-state energy balance (its Eq. 3, `q_hot-wall = q_RR + ṁ·H_w`), the
effective heat of ablation is `Q* = q_RR/ṁ + H_w`.  With ṁ = ρ·ṡ at a *nominal*
3-D C/C density ~1.9 g/cc (not stated in the paper — flagged), q_RR = εσT_w⁴
(~13 MW/m² at 4,000 K), and H_w read from the paper's Fig. 5 at
sublimation-regime B′ (~9–19 MJ/kg): **Q\* ≈ 10–20 MJ/kg at 80–168 atm** —
*below* our band low (25) and nominal (40).  Physical reading: at these extreme
pressures ablation is sublimation- plus thermomechanically-dominated (the paper
measures roughness-augmented heating 1.4–1.5× smooth-wall theory, and its ramp
tests show surface gouging onset at transition pressures ~60–77 atm, biased
along the 45° weave rays), so effective heat collapses.  Consistent with
Schneider 72-705's mechanical-erosion regime bound (>55 atm).

**Consequence for the screen:** the C/C H_eff band applies to the
moderate-pressure regime (Reentry-F class, ≤~60 atm).  For a sharp, very-high-β
RV whose stagnation pressure reaches ≥80 atm, H_eff = 40 would UNDER-predict
recession several-fold.  Logged as a validity limit, not folded into the band —
the screening envelope's blunt-RV cases sit well below this regime.

## Acceptance check

- Nominals within literature bands? **Yes** (CP 15 ∈ [10,30]; PICA 35 conservative-low;
  C/C 40 in sublimation-regime bracket).
- Direction conservative (over-predict)? **Yes** — Phase 3 bounds: Stardust 5.1×
  (vs firsthand Core 1 = 5.7±0.3 mm, Kontinos & Stackpoole AIAA 2008-1197),
  Hayabusa 44×, both predicted ≥ measured.
- Reentry-F honored within radius-history spread? **Yes, as a bracket** — the
  Berry white paper (project Drive) supplied the paired environment + recession
  numbers, and the clean nominal-trajectory figure (supplied in-chat) let the
  heat pulse be digitized: Q ≈ 2.85 GJ/m² ±25%
  (`reentryf_nominal_qdot.csv`), giving H_eff 50–135 MJ/kg (central ≈84).  The
  C/C nominal 40 over-predicts ~2× (conservative); the TM X-1856 curve-1/2/3
  spread is quantified (0.17–0.30 in best-supported, 0.5 in worst case refuted)
  rather than collapsed to a point; `_BENCHMARKS` Q_MJ stays None (preflight
  prediction, not flight measurement).
