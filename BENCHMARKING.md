# Benchmarking shopping list — Reentry Survivability report

What flight data to find, per reentry mode, to benchmark the survivability
report (SURVIVABILITY_REPORT_DESIGN.md §10).  Fill in the "found" columns;
each row becomes a validation case run at the vehicle's entry conditions.

## The three forms and their data needs

### Form A — Ballistic RV (the recession/accuracy chain)
**Computed:** Sutton–Graves q̇(t) → peak flux, pulse width, load Q, T_eq;
recession δ = Q/(ρ·H_eff); δ/R_n accuracy band; lumped heat-sink.
**Material constants consumed:** `is_ablator`, `density_kg_m3`, `H_eff_MJ_kg`,
nose solid depth (`nose_solid_depth_m` / R_n default).

Per benchmark vehicle, find:
| Datum | Unit | Why |
|---|---|---|
| entry velocity & flight-path angle | km/s, ° | reproduce the arc |
| mass, β or (C_D·A), nose radius R_n | kg, kg/m², m | flux scale (q̇ ∝ 1/√R_n) |
| TPS material + thickness | —, cm | pick catalog entry; burn-through depth |
| published peak flux and/or load | MW/m², MJ/m² | check q̇/Q directly |
| **measured recession / char depth** | cm | the δ chain end-to-end |
| survived? accuracy/dispersion notes | — | band placement |

Candidates: **Hayabusa** (recovered; carbon-phenolic; ~12.0 km/s, γ≈−12°,
R_n≈0.20 m — ABOVE the 9 km/s convective envelope: radiative adds ~30%+, so
expect measured ≥ predicted; a bounding pair with **Stardust** (PICA,
recovered, ~12.9 km/s)).  In-envelope anchor already wired: **Reentry-F**
(6.1 km/s, measured recession, flew its mission).  Accuracy bands already
anchored: PANT, Lin 1982.

### Form B — Glider / HGV (the stopwatch)
**Computed:** q̇ history; sustained T_eq; soak (dwell above `continuous_K`);
oxidation dwell; t_fail vs glide time; NRC 300/800/3000-s tier.
**Material constants consumed:** `peak_K`, `continuous_K`,
`oxidation_dwell_s`, emissivity.

Per benchmark vehicle, find:
| Datum | Unit | Why |
|---|---|---|
| glide duration | s | the NRC stopwatch axis |
| measured surface temperature | K / °C | T_eq check (radiative equilibrium) |
| structure / bondline temperature | K / °C | soak criterion |
| survived duration, or failure time + mode | s, — | t_fail calibration |

Candidates: Shuttle (canonical long soak, tiles/RCC); **AHW 2011** (flew its
full ~3,800 km glide — a success datum for the conventional-tier ladder);
IXV (C/SiC, single mission ~1,700 °C).  Already anchored: HTV-2
(~1,900 °C surface / 1,090 °C·3,600 s structure), SWERVE (Mach 8–14 band),
NRC-2008 tiers.

### Form C — Maneuvering quasi-ballistic (the envelope)
**Computed:** terminal-dive segment peak flux + duration vs airframe limit
(screening; windward/AoA probe is a later tier).
**Material constants consumed:** body/airframe `peak_K`, `c_J_kgK` (transient
heat-sink).

Find: pull-up altitude & speed, airframe material, survived-the-maneuver.
Candidates: **Pershing II** (operational MaRV pull-up), SWERVE (maneuvering
flight record — already anchored).

## Scalar benchmarks already in the code (`heating._BENCHMARKS`)
Check/extend — these drive the "N.N× Apollo" ratio line:
ICBM RV 318 MW/m²; Stardust 9.4 MW / 276 MJ; Apollo 7.9 MW / 468 MJ;
MSL 2.0 / 55; Shuttle 0.6 / 66.

## Band constants awaiting data (survivability_report.py §top)
| Constant | Current | Wants |
|---|---|---|
| shape-change onset δ/R_n | 0.10 | more shape-change → dispersion flight data |
| severe blunting δ/R_n | 0.50–1.0 | more recovered/tracked RV data |
| glider ablative-tip flag δ/R_n | 0.05 | any glider tip-recession tolerance data |
| **UHTC `oxidation_dwell_s`** | **120 s (rough)** | **top priority** — arc-jet / flight dwell life for ZrB₂/HfB₂-class tips; gates every long-glide sharp-tip verdict |
| ablator `H_eff_MJ_kg` (CP 15, PICA 35, C/C 40) | screening values | recovered-capsule recession back-out (Hayabusa, Stardust) |
| analytic-honesty factor | 2–4× | more paired analytic/numerical runs |

## Caveats to carry into every comparison
- Screening tier: cold-wall convective only; no radiative gas heating
  (>9 km/s under-predicts), no hot-wall correction, laminar-implicit.
- Compare like-to-like: our q̇ is nose-stagnation reference flux; published
  numbers are sometimes acreage or hot-wall values.

## UHTC survivability anchor table (the living dataset)

Seeds the anchor dataset behind the Form B envelope-coverage model
(SURVIVABILITY_REPORT_DESIGN.md §11).  Each row is one flight / arc-jet /
plasma-torch / furnace datum for a ZrB₂-SiC-class or carbide-boride UHTC.
The aero-convective rows (arc-jet, plasma torch, flight) are all SURVIVALS —
those tests stopped, they did not fail — so on that side the demonstrated
region is a *floor* (§11.1).  The Levine 2003 furnace rows supply the first
**FAILURES** (caps), with the caveat that they are 1-atm stagnant-air cyclic
furnace data: they cap the passive-oxidation regime, not the aero-convective
one.  A clean *aero-convective* failure (arc-jet or recovered flight hardware
burned through at a known T/dwell/flux) remains the highest-value missing
point.

Record schema (one per datum): `id · material_class · kind · tip_radius ·
flux (+kind, enthalpy, stag pressure) · peak_T (+source) · dwell above 1650 °C ·
recession · mass change · outcome (+mode) · source`.

| id | class | kind | tip R | flux | enthalpy / press | peak T (src) | dwell | recession / mass | outcome | source |
|---|---|---|---|---|---|---|---|---|---|---|
| Monteverde-2012-ZS | zrb2_sic | arcjet | 0.1 mm (sharp) | ~7 MW/m² hot-wall net | 16 MJ/kg | **2450 °C (CFD)**; 1577 °C pyro 3 mm back | ~575 s cum. | 0.10→0.14 mm tip; oxide 140→50 µm | **survived** | Monteverde & Savino, *J. Am. Ceram. Soc.* 2012, DOI 10.1111/j.1551-2916.2012.05226.x |
| Scatteia-2010 | zrb2_sic | arcjet | 10 mm (blunt) | 26.5 MW/m² cold-wall | 10 MJ/kg near specimen | 2000–2300 °C (pyro) | >600 s (10+ min) | ~3 mm stag; 7 % mass loss | survived (single-use) | Scatteia et al., *J. Spacecraft & Rockets* 47(2) 271, 2010, DOI 10.2514/1.42834 |
| Monteverde-2013-ZSL10 | zrb2_sic | arcjet | 5 mm (hemisphere) | — | 11.4 MJ/kg / 8 kPa | 1973 K = 1700 °C (pyro) | 300 s | R 5.02→5.14 mm (≈none); **+0.3 % mass** | survived | Monteverde, Alfano, Savino, *Corros. Sci.* 75, 443–453, 2013 — note: LaB₆ was *detrimental* vs plain ZrB₂-SiC |
| Xu-2026-HTS5 | carbide_boride | plasma_torch | bulk billet | H₂/Ar flame | — | 2500–2600 °C (pyro) | 1800 s | **−0.1 µm/s** (net oxide growth); −0.14 g/m²·s | survived | Xu et al., *J. Eur. Ceram. Soc.* 46, 117934, 2026, DOI 10.1016/j.jeurceramsoc.2025.117934 |
| Levine-2003-ZSTS-arcjet | zrb2_sic_tasi2 | arcjet | flat disc 2.54 cm | 3.5 MW/m² stag; ~6 MW/m² edge | 0.07 atm | ~1800 °C measured; edge 1950–2000 °C | 600 s | Δwt −1.4 % | survived | Levine, Opila et al., "Ultra-High Temperature Ceramic Composites for Leading Edges," 27th JANNAF APS, Dec 2003, NTRS 20040033992 |
| Levine-2003-ZS-1927 | zrb2_sic | furnace (1 atm, stagnant, cyclic 10 min hot/10 cool) | coupon | — | 1 atm | 1927 °C | 100 min (10 cycles) | oxidized, discolored, **intact** | survived | Levine, Opila et al., NTRS 20040033992 |
| **Levine-2003-ZSTS-1927** | zrb2_sic_tasi2 | furnace (1 atm, cyclic) | coupon | — | 1 atm | **1927 °C** | ≤10 min (1 cycle already slumped; 5-cycle a molten mass fused to setter) | destroyed | **FAILED — melt/slump** | Levine, Opila et al., NTRS 20040033992 |
| **Levine-2003-ZSTC-1627** | zrb2_sic_tac | furnace (1 atm, cyclic) | coupon | — | 1 atm | **1627 °C** | ≤100 min | ~20 mg/cm² gain; visible holes | **FAILED — breakaway oxidation** | Levine, Opila et al., NTRS 20040033992 |
| SHARP-B1 | hfb2_sic | flight | 3.5 mm (sharp) | ballistic reentry | — | — | short (ballistic) | non-ablating demonstrated | flew, **not recovered** | Johnson, Gasch, Lawson et al., "Recent Developments in UHTCs at NASA Ames" (AIAA); Kolodziej et al. NASA TM-112215, 1997 |
| SHARP-B2 | hfb2_sic / zrb2_sic | flight | strakes on Mk12A RV | ballistic reentry | — | designed to **multi-use limit (retract 47.9 km) / single-use limit (43.3 km)** | short (ballistic) | recovered | flew, **recovered**; some segments failed on **material quality**, not the T/dwell limit | Johnson et al. (AIAA), NASA Ames |

Review-level context (not a point datum): **Peters et al., *Nat. Commun.* 15,
2024, DOI 10.1038/s41467-024-46753-3** — ZrB₂/HfB₂-SiC oxidation ceiling
**~1650 °C** (the `continuous_K` anchor); an HfB₂-SiC nose cone at **80 min
cumulative** arc-jet; carbides (HfC/ZrC) push service **>2000 °C**.  (Its
14.75 MW/m² · 130 s point is a *coated C/C* X-43 edge — file under C/C, not
UHTC.)

**What the table pins:**
- `continuous_K` (glass ceiling) = **1650 °C** — 5+ independent sources agree.
- Demonstrated dwell floor: **≥300 s at 1973 K** (zero recession), extending to
  **~575 s at 2450 °C** (sharp, tip-blunting); plus **600 s at ~1800 °C**
  arc-jet (Levine ZSTS) and **100 min at 1927 °C** in 1-atm furnace air
  (plain ZrB₂-SiC).  Use the low end as the conservative floor.
- Demonstrated peak: **~2450 °C** sharp ZrB₂-SiC, **~2600 °C** carbide-boride.
- First caps (passive-oxidation regime, 1-atm furnace): TaSi₂-doped ZrB₂-SiC
  **melts/slumps at 1927 °C within 10 min**; TaC-doped fails by breakaway
  oxidation at 1627 °C within 100 min.
- Recession rate (flux-normalized): ~0.07 µm/s @ 7 MW/m² (sharp) / ~3.6 µm/s @
  26 MW/m² (blunt).
- **The additive-inversion trap** (envelope-scoping rule): dopant effects
  invert with temperature — TaSi₂ is the best performer at 1627 °C and fatal
  at 1927 °C (Levine 2003); TaC hurts even at 1627 °C (Levine 2003); LaB₆
  hurts at 1700 °C (Monteverde 2013).  Class envelopes must therefore be
  built from the PLAIN ZrB₂-SiC / HfB₂-SiC rows; doped variants get their own
  class ids (`zrb2_sic_tasi2`, `zrb2_sic_tac`, …), never averaged into the
  parent class.

**What it does NOT contain** (the honest gap, §11.6): a sharp UHTC tip held at
**1700–2000 °C for 1000 s+ at flight pressure under aero-convective heating**
— the actual HGV glide case.  The data brackets it but does not reach it.  The
aero-convective anchors are low pressure (~0.07–0.2 atm); the furnace rows are
1 atm but stagnant (no dissociated O, no shear).  The SiC active/passive
transition is pressure-sensitive, so flight may sit on the other side of both.

**Remaining wanted data, ranked by information value:**
1. An **aero-convective failure** — arc-jet run driven to burn-through, or
   recovered flight hardware failed at known conditions (SHARP-B2 quantitative
   post-flight recession would qualify).
2. A **long-dwell moderate-temperature point**: one continuous arc-jet run at
   ~1800–2000 °C for 20–30 min (the HGV-glide hole; Peters' "80 min" is
   cumulative across cycles, not continuous).
3. A **flight-pressure aero-convective point** (VKI Plasmatron class) to close
   the active/passive asterisk.
4. An **HfB₂-SiC arc-jet dwell/recession point**, so that class stands on its
   own data instead of assumed-same-as-ZrB₂-SiC.

**Caveats to carry:** Monteverde-2012's 2450 °C is a CFD tip estimate (measured
pyrometer, 3 mm back, was 1577 °C).  Xu's −0.1 µm/s is net oxide *growth*, an
extreme-temp ablation-survival datum for a better material class — do not use it
to set a generic UHTC number.  Never edit a citation.
