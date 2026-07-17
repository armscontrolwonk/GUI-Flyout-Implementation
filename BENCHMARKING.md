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
| **UHTC `oxidation_dwell_s`** | **120 s (rough → retired)** | superseded by the cited dwell floor (≥300 s @ 1973 K, ~575 s @ 2450 °C) in the anchor table; still want a plain-diboride *aero-convective failure* to cap the top |
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
plasma-torch / furnace datum for a ZrB₂-SiC-class, complex-boride, or
carbide-boride UHTC.  Most aero-convective rows are SURVIVALS — those tests
stopped, they did not fail — so on that side the demonstrated region is a
*floor* (§11.1).  Caps now come from three regimes: the **Levine 2003 furnace**
rows (1-atm stagnant cyclic — they cap the passive-oxidation regime for *doped*
classes only, not the aero-convective one); two doped/complex **oxide-detachment**
caps (**Di Maso HfB₂-TaSi₂** cone, **De Prisco 2026 ZrB₂-TiB₂-SiC** hemispheres,
both ~2700–2800 K); and — the keystone — the **plain-ZrB₂-SiC PA
transition** (loss of protective silica → active SiC oxidation → temperature
jump), now **triple-sourced and flux-bracketed**: Marschall 2012 (VKI, +400 K
jump at ~2 MW/m² / 10 kPa), Monteverde 2017 (SPES, onset ~2050 K), and Zhang
2008 (HIT, passive at 1.7 MW/m² / 0 % mass ↔ active at 5.4 MW/m² / 15.75 %
loss, 3 mm), plus SHARP-B1 flight corroboration.  The remaining, narrower gap
is a *total burn-through / complete consumption* of a plain tip (the PA
transition is runaway *onset*, not full loss), and an aero-convective cap
specifically for **HfB₂-SiC** (only in telemetry-limited flight so far).

Record schema (one per datum): `id · material_class · kind · tip_radius ·
flux (+kind, enthalpy, stag pressure) · peak_T (+source) · dwell above 1650 °C ·
recession · mass change · outcome (+mode) · source`.

| id | class | kind | tip R | flux | enthalpy / press | peak T (src) | dwell | recession / mass | outcome | source |
|---|---|---|---|---|---|---|---|---|---|---|
| Monteverde-2012-ZS | zrb2_sic | arcjet | 0.1 mm (sharp) | ~7 MW/m² hot-wall net | 16 MJ/kg | **2450 °C (CFD)**; 1577 °C pyro 3 mm back | ~575 s cum. | 0.10→0.14 mm tip; oxide 140→50 µm | **survived** | Monteverde & Savino, *J. Am. Ceram. Soc.* 2012, DOI 10.1111/j.1551-2916.2012.05226.x |
| Scatteia-2010 | zrb2_sic | arcjet | 10 mm (blunt) | 26.5 MW/m² cold-wall | 10 MJ/kg near specimen | 2000–2300 °C (pyro) | >600 s (10+ min) | ~3 mm stag; 7 % mass loss | survived (single-use) | Scatteia et al., *J. Spacecraft & Rockets* 47(2) 271, 2010, DOI 10.2514/1.42834 |
| Monteverde-2013-ZSL10 | zrb2_sic | arcjet | 5 mm (hemisphere) | — | 11.4 MJ/kg / 8 kPa | 1973 K = 1700 °C (pyro) | 300 s | R 5.02→5.14 mm (≈none); **+0.3 % mass** | survived | Monteverde, Alfano, Savino, *Corros. Sci.* 75, 443–453, 2013 — note: LaB₆ was *detrimental* vs plain ZrB₂-SiC |
| Xu-2026-HTS5 | carbide_boride | plasma_torch | bulk billet | H₂/Ar flame | — | 2500–2600 °C (pyro) | 1800 s | **−0.1 µm/s** (net oxide growth); −0.14 g/m²·s | survived | Xu et al., *J. Eur. Ceram. Soc.* 46, 117934, 2026, DOI 10.1016/j.jeurceramsoc.2025.117934 |
| Levine-2003-ZSTS-arcjet | zrb2_sic_tasi2 | arcjet | flat disc 2.54 cm | 3.5 MW/m² face (350 W/cm²); ~6 MW/m² edge (600 W/cm²) | 0.07 atm | ~1800 °C measured; edge est. 1950–2000 °C | 600 s | Δwt −1.4 % | survived | Levine, Opila et al., "Ultra-High Temperature Ceramic Composites for Leading Edges," 27th JANNAF APS, Dec 2003, NTRS 20040033992 |
| Levine-2003-ZS-1927 | zrb2_sic | furnace (1 atm, stagnant, cyclic 10 min hot/10 cool) | coupon | — | 1 atm | 1927 °C | 100 min (10 cycles) | oxidized, discolored, **intact** | survived | Levine, Opila et al., NTRS 20040033992 |
| **Levine-2003-ZSTS-1927** | zrb2_sic_tasi2 | furnace (1 atm, cyclic) | coupon | — | 1 atm | **1927 °C** | ≤10 min (1 cycle already slumped; 5-cycle a molten mass fused to setter) | destroyed | **FAILED — melt/slump** | Levine, Opila et al., NTRS 20040033992 |
| **Levine-2003-ZSTC-1627** | zrb2_sic_tac | furnace (1 atm, cyclic) | coupon | — | 1 atm | **1627 °C** | 10-min cycles | ~20 mg/cm² runaway gain; degraded to molten washer | **FAILED — TaC ineffective at 1627 °C (runaway oxidation)** | Levine, Opila et al., NTRS 20040033992 |
| SHARP-B1 | hfb2_sic | flight | 3.5 mm (sharp) | ballistic reentry | — | — | short (ballistic) | non-ablating demonstrated | flew, **not recovered** | Johnson, Gasch, Lawson et al., "Recent Developments in UHTCs at NASA Ames" (AIAA); Kolodziej et al. NASA TM-112215, 1997 |
| SHARP-B2 | hfb2_sic / zrb2_sic | flight | strakes on Mk12A RV | ballistic reentry | — | designed to **multi-use limit (retract 47.9 km) / single-use limit (43.3 km)** | short (ballistic) | recovered | flew, **recovered**; some segments failed on **material quality**, not the T/dwell limit | Johnson et al. (AIAA), NASA Ames |
| DePrisco-2026-ZTN-M6 | zrb2_tib2_sic (NbC) | arcjet (SPES, M=6) | 10 mm (hemi) | ~4 MW/m² stag | 4.5→20.3 MJ/kg / 3×10⁻³ atm | 1800 K (pyro) | ~300 s (stepped) | +0.31 % mass; tip white oxide, sides silica; 165 µm peak-valley | **survived** | De Prisco et al., *J. Eur. Ceram. Soc.* 46 (2026) 118184 |
| DePrisco-2026-ZTV-M6 | zrb2_tib2_sic (VC) | arcjet (SPES, M=6) | 10 mm (hemi) | ~4 MW/m² stag | 4.5→20.3 MJ/kg / 3×10⁻³ atm | 1700 K (pyro) | >400 s (stepped) | +0.16 % mass; dark borosilicate glass (better than Nb); 100 µm | **survived** | ibid. |
| **DePrisco-2026-ZT-M3-2700K** | zrb2_tib2_sic (NbC & VC) | arcjet (SPES, M=3) | 10 mm (hemi) | ~10 MW/m² stag | 5.1→14.4 MJ/kg / 2.3×10⁻² atm | **2700 K** (pyro) | stepped (30–120 s) | small net mass gain; **very tip of BOTH samples detached** (poor oxide adherence) | **DEGRADED — 2nd aero-convective cap** (oxide detachment on handling after 2700 K; not burn-through) | ibid. |
| Marschall-2012-ZrB2-30SiC-protected | zrb2_sic | plasmatron (VKI 1.2 MW, subsonic) | flat "mushroom" face | 110 W/cm² (1.1 MW/m²) cold-wall | 10 kPa static | 1800–1900 K steady | held at power | small mass change; stable protective oxide | **survived (passive/protected)** | Marschall, Pejaković, Fahrenholtz, Hilmas, Panerai, Chazot, *J. Thermophys. Heat Transfer* 26(4) 2012, DOI 10.2514/1.T3798 |
| **Marschall-2012-ZrB2-30SiC-jump** | zrb2_sic | plasmatron (VKI, subsonic) | flat "mushroom" face | **jump at q_cw ≈ 202 W/cm² (2.02 MW/m²)**; no jump at 185 W/cm² (1.85) | 10 kPa static; P_dyn 75–95 Pa | ~2215 K steady → **+400 K in 20–30 s** (after 30–45 s hold) | onset-defined | protective silica lost → active SiC oxidation → chemical heat-flux surge; accelerated mass loss / changing mold line | **DEGRADED/RUNAWAY — the plain-ZrB₂-SiC passive→active (PA) transition; first plain-diboride aero-convective cap** | ibid. |
| Zhang-2008-1.7MW | zrb2_sic | arcjet (subsonic) | flat face, sharp LE R 3.5 mm | **1.7 MW/m²** | — | **1640–1660 °C** (pyro) | 600 s | **0.00 % mass, 0.00 mm, oxide 25 µm, no SiC depletion** | **survived (passive — below PA)** | Zhang, Hu, Han, Meng, *Compos. Sci. Technol.* 68 (2008) 1718–1726 |
| **Zhang-2008-5.4MW** | zrb2_sic | arcjet (subsonic) | flat face, sharp LE R 3.5 mm | **5.4 MW/m²** | — | 2150–2330 °C (pyro) | 600 s | **−15.75 % mass; 2.98 mm recession (~5 µm/s); oxide 390 µm** | **DEGRADED — active oxidation/ablation (above PA)** | ibid. |
| Monteverde-2017-SiCZrB2-jump | zrb2_sic | plasmatron (SPES, supersonic) | disc 12.7 mm | 3.5 MW/m² cold-wall | 9–11 kPa static; pO+O₂ 1.7–2.1 kPa; ≤21 MJ/kg | instabilities onset T_F ~2020–2050 K → jumps/waves-of-radiance | held | endured "rather well"; SiC-depletion correlates with jumps | survived-with-instabilities (2nd-lab PA corroboration) | Monteverde, Cecere, Savino, *J. Eur. Ceram. Soc.* 37 (2017) 2325–2341 |

Review-level context (not point data):
- **Peters et al., *Nat. Commun.* 15, 2024, DOI 10.1038/s41467-024-46753-3** —
  ZrB₂/HfB₂-SiC oxidation ceiling **~1650 °C** (the `continuous_K` anchor); an
  HfB₂-SiC nose cone at **80 min cumulative** arc-jet; carbides (HfC/ZrC) push
  service **>2000 °C**.  (Its 14.75 MW/m² · 130 s point is a *coated C/C* X-43
  edge — file under C/C, not UHTC.)
- **Glass, D. E., "Physical Challenges and Limitations Confronting the Use of
  UHTCs on Hypersonic Vehicles," AIAA 2011-2304** (NASA Langley) — an
  independent oxidation-regime map for ZrB₂-SiC (from its Ref [17]): ~700–1200 °C
  B₂O₃-protected; **~1200–1600 °C SiO₂-protected** (the ceiling, 6th source);
  **~1600–1800 °C SiO₂ lost to active oxidation** (SiO gas, ZrO₂ non-protective)
  — the PA/PAT transition by temperature, consistent with Marschall's ~1942 °C
  jump; >1800 °C recrystallized ZrO₂ "may prevent catastrophic failure."
  Crucially, Glass reviews the literature UHTC *component* failures (SHARP-B2;
  the CIRA ZrB₂-SiC nose tip) and attributes them to **mechanical / attachment
  causes — a Ti retaining screw, thermocouple-drilling holes, processing
  quality — NOT oxidation**.  A CIRA duplicate withstood 300 W/cm² (3 MW/m²) ×
  108 s twice with only non-critical base damage; the stated bottleneck is
  "design of mechanical interfaces with subtending structures."  This is why
  the model excludes those component failures from the *thermal* envelope and
  keeps Marschall's PA transition as the clean plain-material oxidation cap.

**What the table pins:**
- `continuous_K` (glass ceiling) = **1650 °C** — 6+ independent sources agree (Monteverde 2012, Peters 2024, Fahrenholtz & Hilmas, Marschall, Li, Glass 2011).
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

## UHTC anchor table — additions (Savino/Di Maso corpus)

| id | class | kind | tip R | flux | enthalpy / press | peak T (src) | dwell | recession / mass | outcome | source |
|---|---|---|---|---|---|---|---|---|---|---|
| Savino-2010-SPES-hemi | zrb2_sic | arcjet (SPES, M=3) | 5 mm | — | ≤10 MJ/kg | 2053 K = 1780 °C (pyro) | ~165 s hold (~10 min campaign) | oxide 150→60 µm; **−0.5 % mass** (Di Maso thesis); undamaged | **survived** | Savino, De Stefano Fumo, Paterna, Di Maso, Monteverde, *Aerosp. Sci. Technol.* 14 (2010) 178–187 |
| Savino-2010-SPES-cone | zrb2_sic | arcjet (SPES, M=3) | **0.5 mm (sharp)** | — | ≤10 MJ/kg | 2083 K = 1810 °C (pyro) | ~224 s hold | oxide 190 µm tip; SiC-depleted tip layer 70→0 µm (shear-driven); **−0.6 % mass** (Di Maso thesis); undamaged | **survived** | ibid. |
| Savino-2010-DLR-wedge | zrb2_sic | arcjet (DLR L2K, M=3.9) | 5 mm wedge | — | 6.05–9.7 MJ/kg | up to ~2250 K (abstract) | 60–180 s × multiple runs, AoA 0/25° | survived | **survived** | ibid. |
| DiMaso-2009-HfB2TaSi2-hemi | hfb2_tasi2 | arcjet (SPES), **3 thermal cycles** | 5 mm | — | 8.7→12.9 MJ/kg / 7–11 kPa | 2010–2044 K (pyro, per cycle) | ~142–186 s per condition (~684 s cum.) | micro-cracks in Hf,Ta-oxide + HfO₂ from cycling (mass change not separately reported for this material) | **survived (cycled)** | Di Maso, A., *Plasma Wind Tunnel Testing of UHTC*, PhD thesis, Univ. Naples Federico II, XXII ciclo |
| **DiMaso-2009-HfB2TaSi2-cone** | hfb2_tasi2 | arcjet (SPES), 2 cycles | **0.5 mm (sharp)** | — | 8.7→12.9 MJ/kg / 7–11 kPa | 2279 K (pyro); **~2800 K tip (CFD)** | ~90 s holds | LE oxide **detached from bulk**; craters ~10 µm; Ta₂O₅·6HfO₂ extensively evaporated 2300–2800 K | **DEGRADED — the dataset's first aero-convective cap** (oxide-scale detachment = loss of protection; not burn-through) | ibid. |

**What the additions change:**
- **Two aero-convective caps now converge on one mode.**  The Di Maso HfB₂-TaSi₂
  sharp cone (oxide detachment ~2800 K CFD tip) and the De Prisco 2026
  ZrB₂-TiB₂-SiC hemispheres (both NbC and VC tips detached after 2700 K, Mach 3,
  ~10 MW/m²) fail the *same way* — **oxide-scale detachment / poor adherence to
  the unreacted bulk at ~2700–2800 K** — across two labs, two material classes,
  and two geometries.  That convergence is a stronger cap than either alone: for
  doped/complex diborides the ~2700 °C-class limit is oxide adherence, not melt
  or burn-through.  Both cap *doped/complex* classes, consistent with the
  additive-inversion rule; neither may be averaged into plain ZrB₂-SiC /
  HfB₂-SiC envelopes.
- **New class survives the mid-band, pressure-resolved.**  De Prisco's
  ZrB₂-TiB₂-SiC survived 1700–1800 K at low pressure (Mach 6, 3×10⁻³ atm,
  ~4 MW/m², 300–400 s) but detached at 2700 K under **10× higher pressure**
  (Mach 3, 2.3×10⁻² atm) — the same specimens, so it isolates the pressure
  axis: the SiC active/passive footnote (§11.6) is not hypothetical here.
  VC-doping out-performed NbC (darker, more-stable borosilicate glass).
- **The plain-diboride cap we were missing — and it's the model's keystone.**
  Marschall 2012 caught the **passive→active (PA) oxidation transition** of
  *plain* ZrB₂-SiC in the act: at 10 kPa the surface holds a stable protective
  silica scale up to a threshold, then at q_cw ≈ 202 W/cm² (2.02 MW/m², ~2215 K
  steady) the glass is lost, SiC oxidises actively, the chemical heat flux
  surges, and the surface **jumps +400 K in 20–30 s** — a self-amplifying
  runaway, not a soft limit.  Just 185 W/cm² (1.85 MW/m²) did *not* trigger it:
  the boundary is sharp.  This is:
  1. the first **plain** (undoped) aero-convective cap — the exact gap the
     additive-inversion rule said we could not fill from doped data;
  2. **flight-corroborated** — SHARP-B1 saw the same jump on a ZrB₂-SiC
     arcjet sample, 2360→2810 K in ~15 s (Kolodziej et al.); HyMETS arcjet
     likewise;
  3. **pressure- and flux-explicit** (10 kPa), so it feeds the §11.6 pressure
     axis with a real number, not a caveat;
  4. the **physical mechanism** under the whole envelope — the loss of the
     borosilicate glass that the 1650 °C `continuous_K` ceiling is *about*.
  Consequence for the model: the envelope's "too hot" upper edge is the **PA
  transition, a flux/pressure surface — not a fixed temperature**.  It also
  reconciles an apparent tension: Monteverde's sharp tip survived 2450 °C at
  7 MW/m² because a sharp, conducting tip can stay locally passive, whereas
  Marschall's flat face went active at ~2215 K / 2 MW/m² / 10 kPa.  Same
  material, different (flux, geometry, pressure) → different PA crossing.
- **The PA transition is now triple-sourced and flux-bracketed.**  Three
  independent labs see the same plain-ZrB₂-SiC boundary:
  * **Marschall 2012** (VKI Plasmatron) — the +400 K jump at ~2 MW/m² / 10 kPa.
  * **Monteverde 2017** (SPES, *JECS* 37, 2325) — "jumps-/waves-of-radiance"
    instabilities onset at T_F ~2020–2050 K, 3.5 MW/m², 9–11 kPa: a second-lab
    confirmation at the same pressure regime (its 200 Pa is the *base* chamber
    pressure, not the test static pressure).
  * **Zhang 2008** (HIT arc-jet, *CST* 68, 1718) — a clean *flux bracket* on a
    flat plain-ZrB₂-SiC face: at **1.7 MW/m²** the surface sat at 1650 °C with
    **0.00 % mass / 0.00 mm** (fully passive, right at the glass ceiling); at
    **5.4 MW/m²** it reached >2300 °C with **15.75 % mass loss / 3 mm recession
    (~5 µm/s)** (active).  So the PA crossing lies between 1.7 and 5.4 MW/m² for
    a flat face — consistent with Marschall's ~2 MW/m² — and the active-side
    recession is now *quantified* (~5 µm/s at 5.4 MW/m² / 2300 °C), not just
    named.
- **The mid-ladder fills in**: sharp (0.5 mm!) ZrB₂-SiC survives ~1810 °C for
  ~224 s undamaged — between the 1700 °C/300 s and 2450 °C/575 s anchors, and at
  a tip radius 200× sharper than the Scatteia nose.
- **Cycling datum**: 3 heat-up/cool-down cycles at ~2000 K survive with
  micro-cracking — the first reuse-relevant point.
- **Plain-diboride isothermal survival at 1927 °C** (Levine ZS furnace): plain
  ZrB₂-SiC held 10× 10-min cycles at 1927 °C, discolored but intact — a
  survival *above* every arc-jet hold, and directly above the temperature that
  melted its TaSi₂-doped sibling.  Illustrates why furnace caps don't transfer
  to aero-convection: the ZSTS arc-jet edge ran 1950–2000 °C for 600 s and
  survived, yet the same material melts at 1927 °C isothermal — the furnace
  soaks the whole coupon, the arc-jet edge is transient and conducts away.
- **Mechanism temperatures now citable** (Di Maso thesis conclusions + its
  refs): B₂O₃ liquid protection 670–1370 K, boils off >1370 K; silica melts at
  ~2100 K and the film is lost to volatility/shear (the SiC-depletion onset);
  Ta₂O₅·6HfO₂ evaporates 2300–2800 K.

## Threshold provenance audit

Standing rule: **every threshold the survivability model consumes is either
cited to a real source, or explicitly labeled as an internal inference /
placeholder — never given a fake citation.**  Status of every threshold:

| Threshold | Value | Provenance | Status |
|---|---|---|---|
| Sutton-Graves constant | 1.7415e-4 | Sutton & Graves 1971, NASA TR R-376 | ✔ cited |
| shape-change onset | δ/R_n = 0.10 | Lin, Grabowsky & Yelmgren 1982 (TRW/BMO): 0.1 R_N "mildly indented"; PANT (DTIC ADA019186): asymmetry→dispersion | ✔ cited |
| severe blunting | δ/R_n = 0.50–1.0 | Berry, Reentry-F (NASA CR-154044): flew R_n 0.10→0.171 in (~0.7 R_n), survived | ✔ cited |
| burn-through | δ > nose_solid_depth (≈R_n) | Reentry-F solid-tip length ~7.7 R_n axial (Berry) | ✔ cited |
| glider tip flag | δ/R_n ≥ 0.05 | **internal inference** from Murbach 1993/AEOLUS (SWERVE C-C nose) + AHW's move to non-ablating tips; no literature number | ⚠ labeled inference |
| NRC duration ladder | 300 / 800 / 3,000 / 3,600 s | NRC 2008, *U.S. Conventional Prompt Global Strike*, App. D Fig. D-2, pp. 119–121 (Mk-500, CSM-1 AMaRV, CSM-2 FALCON) | ✔ cited |
| ablation↔reradiation crossover | ~1,000 s | **derived** from the NRC CSM-1 (800 s ablative) vs CSM-2 (3,000 s C-C) pair | ⚠ labeled derived |
| UHTC glass ceiling | 1650 °C (protective SiO₂ up to ~1600 °C; active oxidation above) | Monteverde & Savino 2012; Peters 2024; Fahrenholtz & Hilmas 2012; Marschall 2009/2012; Li 2008; Glass 2011 (AIAA 2011-2304) | ✔ cited (6 sources) |
| silica melt / film loss | ~2100 K | Di Maso thesis concl. 5 (citing its [45]); mechanism distinct from the 1650 °C protectiveness ceiling | ✔ cited |
| UHTC dwell floor | ≥300 s @ 1973 K; ~575 s @ 2450 °C | Monteverde 2013 (Corros. Sci. 75); Monteverde & Savino 2012 | ✔ cited |
| UHTC demonstrated peaks | ~2450 °C sharp ZrB₂-SiC / ~2600 °C carbide-boride | Monteverde 2012 (CFD-source flagged); Xu 2026 | ✔ cited |
| **PA (passive→active) transition** — the "too hot" edge (plain ZrB₂-SiC) | ~2 MW/m² / ~2050–2215 K / 10 kPa; flux-bracketed 1.7 (passive) ↔ 5.4 MW/m² (active); active-side recession ~5 µm/s @ 5.4 MW/m² (flux/pressure surface, not fixed T) | Marschall 2012 (*JTHT* 26(4), DOI 10.2514/1.T3798); Monteverde 2017 (*JECS* 37, 2325); Zhang 2008 (*CST* 68, 1718); flight-corroborated SHARP-B1 (Kolodziej et al.) | ✔ cited (3 labs + flight) |
| HfB₂-TaSi₂ / complex-boride oxide-detachment cap | ~2700–2800 K tip | Di Maso thesis (HfB₂-TaSi₂ cone, CFD tip); De Prisco 2026 (ZrB₂-TiB₂-SiC hemi, *JECS* 46 118184) | ✔ cited |
| additive inversion | TaSi₂ best @ 1627 °C, destroyed @ 1927 °C | Levine et al. 2003 furnace (NTRS 20040033992); corroborated by the Di Maso cone at temperature | ✔ cited |
| acreage flux fraction | 0.13 × stagnation | Lu, Shi, Zhang et al. 2024 (IJHMT 225; validated <9 %) | ✔ cited |
| bondline limit | 250 °C | NASA NTRS 20060004824 (ablative TPS sizing); Orion 260 °C NTRS 20080013535 | ✔ cited |
| tile/RCC/material limits | per-material peak/continuous K | HEATING_TPS_REFERENCES.md §2 (TPSX, KSC STS ref, NTRS 19940030739, Peters 2024, …) | ✔ cited per entry |
| analytic-honesty factor | ×2–4 | **internal**: paired analytic/numerical C-HGB runs in this tool | ⚠ labeled internal |
| `NOTHING_SURVIVES_K` | 4000 K | **modeling-validity bound** (radiative-equilibrium model invalid above all usable materials), not an empirical limit | ⚠ labeled model bound |
| `uhtc` `oxidation_dwell_s` (current code) | 120 s | **uncited placeholder** — retired at §11 implementation in favor of the cited dwell floor above | ⚠ flagged for removal |

Rows marked ⚠ are the complete list of thresholds NOT backed by literature;
each is labeled with its true epistemic status in code and report text.  If a
future source covers one, replace the label with the citation — never the
reverse.
