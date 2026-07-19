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
R_n≈0.20 m) and **Stardust** (PICA, recovered, ~12.9 km/s) — both ABOVE the
9 km/s convective envelope, but post-flight data shows the model still
*over*-predicts them (equilibrium-chemistry conservatism beats the missing
radiative term — for Stardust the radiative part was only 9% of peak rate /
4% of load, Kontinos & Stackpoole AIAA 2008-1197): they are **bounding**
anchors, see "Form A recession anchors" below.  In-envelope anchor already
wired: **Reentry-F** (6.1 km/s, measured recession, flew its mission).
Accuracy bands already anchored: PANT, Lin 1982.

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

#### Form C anchor shopping list (the next campaign)

Form C is the one form with no dedicated anchor campaign yet.  Per-vehicle
datums wanted: **pull-up/maneuver altitude & Mach, g-load, AoA, airframe
material + measured/estimated windward temperature, and
survived-the-maneuver** — feeding the terminal-dive transient screen (peak
flux × duration vs heat-sink capacity) and, later, a windward/AoA heating
probe.

*Already in hand — mine first:*
| asset | where | expected yield |
|---|---|---|
| Regan, *Re-Entry Vehicle Dynamics* (AIAA, 1984) | repo `data/` (scan; searchable OCR corpus supplied in-chat + user Drive) | **MINED** — see the verification case below.  AMaRV appears only as a program name (App. B, Honeywell GG1328AA strapdown gyro "used in the AMARV project") — the trajectory reconstruction remains with the 1993 Regan & Anandakrishnan |
| Williamson, "Hypersonic Flight Testing," AIAA 92-3989 (Sandia) | repo `data/` (`williamson-1992-hypersonic-flight-testing-sandia-aiaa-92-3989.pdf`) | Sandia flight-test record incl. SWERVE-class vehicles: Mach/altitude corridors, maneuver profiles |
| Morrison & Vamos, "The Reentry Systems Application Program (RSAP)" (1996) | project Drive | MaRV flight-test program overview; candidate pull-up datums |
| Brooks, "Estimating Characteristics of a Maneuvering Reentry Vehicle" (2010) | project Drive | observable MaRV maneuver signatures — pull-up geometry/g estimates |
| Bunn, *Technology of Ballistic Missile Reentry Vehicles* | project Drive + cited | MaRV context: β, maneuver classes (already cited for Form A context) |
| Sandia Lab News (1981), Strypi/Wente 1982 | uploaded in-chat (Jul 3) | SWERVE program context, booster lineage |

*To find:*
- **Pershing II / MGM-31B**: the RADAG terminal maneuver — pull-up altitude/
  Mach/g, airframe skin material and temperature margin (DTIC: Pershing II
  development & flight-test reports; Army historical monographs).
- **AMaRV** (McDonnell Douglas Advanced Maneuvering Reentry Vehicle, 1979–80
  flights): the canonical open-literature MaRV; NRC-2008 already uses it as
  CSM-1 in the duration ladder.  **CORRECTION (claim checked and withdrawn):**
  an earlier note here asserted the AMaRV trajectory reconstruction appears in
  Regan & Anandakrishnan 1993 — the book (now read from primary, repo `data/`)
  contains **no AMaRV content at all** (the 1984 Regan mentions the program
  name only, via its gyro).  Do not cite the Regan books for it.
  **Now partially filled from three DTIC documents (user-supplied):**
  - *Auclair, AFIT/GST/OS/82M-2 (1982), DTIC ADA115704* (read from primary,
    repo `data/`): the ABRES program fielded **two AMaRV classes — Evader**
    (preprogrammed avoidance maneuvers, ending with an accuracy maneuver) **and
    Accuracy** (same size/shape, no evasion).  **Accuracy-AMaRV flight
    profile: dive to 30,000 ft → near-horizontal flight while a Terminal Fix
    System (radar-correlation / TERCOM / pulse-Doppler map-match; Goodyear /
    McDonnell Douglas / Raytheon) updates navigation → dive to target.**
    The thesis's own model is effectiveness-only (CEP/yield/Pk) — no
    trajectory dynamics.
  - *MDAC SBMO/TR-80-12 (1980), DTIC ADA090577* (read from primary, repo
    `data/`): program facts — contract F04701-76-C-0100 under ABRES/BMO
    (Maneuvering Vehicle Branch, Norton AFB), start Sept 1976, **three flight
    vehicles built**, ~**30-minute flight time**, part-level environments
    (shock/vibration/acceleration/temperature) "exceeded any previous program"
    at SAMSO.
  - *Critchlow & Williams, AFIT/GST/OS/82M-5 (1982), DTIC ADA115691* (in user
    Drive; Drive text extraction partial — front matter only): RV/ABM
    engagement simulation with a sample RV trajectory (Fig. 8, p. 43) —
    **the quantitative maneuver model is still unmined** (scanned pages; needs
    chat upload or page renders).
  Quantitative pull-up numbers (Mach/g at the 30-kft maneuver) remain the open
  want.
- **Mk 500 Evader**: NRC-2008 already cites (300-s ablative tier); any
  maneuver-profile specifics.
- **BGRV** (boost-glide reentry vehicle, 1968 flight): early high-β maneuver
  datum, DTIC.
- Any **windward/leeward heating split measured during a pull-up** (flight or
  wind tunnel) — the datum the later-tier AoA probe needs.  First piece in
  hand: Thompson 1989 (read from primary, repo `data/`) brackets the
  engineering-code uncertainty at AoA on slender cones (~40% near the windward
  local maximum at α=3°, ~15% overall at α=20°, vs VSL3D; Mach-10.6 15° cone
  wind-tunnel validation at α=20°) — the honesty band for a screening AoA
  probe, though not yet a flight maneuver datum.

*Drive-folder triage (2026-07-19, folder `1oUqtoFx02…`):*
- `regan1993.pdf` — the Regan & Anandakrishnan book: **the priority get**,
  blocked by the 10 MB connector cap (see above).
- `AD0376942.pdf` = DTIC, "Aerodynamics of Conical Bodies" — RV aero
  (trim/static-margin side); also >10 MB, same cap.
- `GEReentryVehicles.pdf` = AIAA Historic Site brochure, GE Re-entry Systems —
  archived to repo `data/`.  Gives the **MaRV lineage map for the DTIC hunt**
  (MBRV → MaRV studies → EP MaRV / HP MaRV / Mk 500 → MTV / HAVE STING /
  ENDO LEAP / PDV / MSV / HEART) and a citable early datum: **MBRV (GE,
  contract 1963): three successful maneuvering reentry flights over the
  Pacific mid-1960s, demonstrating aerodynamic controls** (no quantitative
  pull-up numbers in the brochure).
- `10.1016@j.dt.2015.06.003.pdf` = Rizvi, He & Xu, *Defence Technology* 11
  (2015) 350–361, waverider boost-glide optimal trajectory + heat load —
  Form B methodology context, archived to repo `data/`.
- `206-215.pdf` = NRC-2008 CPGS **Appendix G** excerpt (boost-glide why/how) —
  already cited in HEATING_TPS_REFERENCES §6; no new datum.
- `10.1016@0376-04217990001-0.pdf` = *Prog. Aerosp. Sci.* 1979 first article
  (46 MB scan) — unidentified beyond the journal; >10 MB cap.
- Subfolders X-51 / HTV / Defenses / General Hypersonics + the Heating-folder
  Martin-class "atmospheric reentry" book (13 MB, >cap) — unswept.

#### Form C modeling anchors: Regan & Anandakrishnan 1993 (read from primary)

*Dynamics of Atmospheric Re-Entry* (AIAA 1993; repo `data/`), Chapter 9
"Maneuvering Re-Entry Vehicles: Particle Motion" + Appendix D simulation:

- **Diveline guidance framework** — the MaRV shapes its trajectory by a
  preset lift schedule (pull-up followed by tuck/pull-down, essentially
  in-plane; sequential divelines produce large out-of-plane deviation,
  Figs. 9.13–9.14) — the published conceptual model matching the Form C
  terminal-dive design (`glider_terminal_*` + dive-at-target).
- **Representative MaRV evader parameters** (Table D.1 listing, OCR-decoded):
  mass 140 kg, diameter 0.4 m, C_D0 = 0.1, (L/D)max = 2.5, max side
  acceleration **100 g**, fraction-of-(L/D)max 0.85 → β ≈ 1.1×10⁴ kg/m² —
  consistent with the 1984 worked case's β = 10⁴ and with Ch. 5's "two orders
  of magnitude greater than gravity" maneuver-load statement.  Textbook
  *representative* values, not flight data — citable as the reference MaRV
  configuration for Form C verification runs.

#### Form C verification case: Regan 1984 Table 6.7/6.8 (in hand)

Regan, *Re-Entry Vehicle Dynamics* (AIAA Education Series, 1984), pp. 133–134
— a published worked maneuvering trajectory (BASIC listing + sample output):
**fixed L/D = 1.5, β = 10,000 kg/m² (≈ Mk12A class), V₀ = 6,000 m/s at
z₀ = 100 km, γ₀ = 10°, transverse-acceleration limit 4 g** — with the book's
stated result that the fixed-L/D maneuver **hits the 4-g transverse limit at
≈45 km and L/D must be feathered below it**.  This is exactly the Form C
dive/maneuver physics, so it is the natural first verification case when the
Form C anchor work lands: run the MaRV mode at these parameters and check the
transverse-g crossing altitude ≈45 km.  Supporting load-context quotes (same
book): maneuvering loads normal to the velocity vector "can be more than two
orders of magnitude greater than the gravitational force" (Ch. 5 intro), and
a Ch. 13 worked transient of 25.7 g at α = 60° with endo-maneuvering loads
"an order of magnitude larger" causing little bending (distributed load).

#### Form A — CLOSED (changelog)

The Form A ablator campaign (plan: retire the `H_eff_MJ_kg` screening
placeholders) is **closed**: all three nominals cited and regime-labeled
(CP 15 = Sutton char-removal regime; PICA 35 ≤ Winter arc-jet point; C/C 40
under the Reentry-F flight bracket + Scala/Perini theory, Nestler validity
floor), both capsules firsthand (Stardust ×2, Hayabusa + Yamada caveats +
densities), bound tests enforcing predicted ≥ measured, P3-radiative and
P3-chemistry logged.  Mirrors the UHTC `oxidation_dwell_s` retirement.
*Remaining optionals, non-blocking:* **LWP-460** (would upgrade the
pixel-traced Reentry-F pulse to a published curve — user is hunting a copy);
**user spot-check of the Figure-1 digitization** (peak/tail reads); both land
as data edits when they arrive.

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
| ablator `H_eff_MJ_kg` (CP 15, PICA 35, C/C 40) | ~~screening values~~ → **conservative-low nominals within cited Q\* bands; bound-tested** (see "Form A recession anchors") | a finite-rate chemistry option to retire the built-in equilibrium conservatism (P3-chemistry) |
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
loss, 3 mm), plus SHARP-B1 flight corroboration.  **HfB₂-SiC** now has two ground-test
anchors: moderate-T (Gasch & Johnson 2010, ~1690 °C / 2.5 MW/m² / 600 s) and —
the strong one — Sevastyanov 2014, **HfB₂-45SiC survived 2500–2700 °C for
15–18 min** (>30 min total) with only **1.5 % mass loss and no cracking**.  That
confirms Peters' point empirically: Hf pushes the transition **~700 °C above**
plain ZrB₂-SiC's ~1942 °C — so HfB₂-SiC's floor extends to ~2700 °C (with the
caveats that this is a *high-SiC*, ~20 %-porous variant at 10–30 kPa).  The
remaining gaps are a *total burn-through* of a plain tip (the PA transition is
runaway *onset*, not full loss) and the **HfB₂ PA cap itself** — Sevastyanov's
sample never went into runaway, so HfB₂-45SiC's active-oxidation boundary is
only bounded *below* (> 2700 °C), not pinned.

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
| Monteverde-2017-SiCZrB2-jump | zrb2_sic | plasmatron (SPES, supersonic) | disc 12.7 mm | 3.5 MW/m² cold-wall | 9–11 kPa static; pO+O₂ 1.7–2.1 kPa; ≤21 MJ/kg | instabilities onset T_F ~2020–2050 K → jumps/waves-of-radiance | held | endured "rather well"; SiC-depletion correlates with jumps | survived-with-instabilities (2nd-lab PA corroboration) | Monteverde, Cecere, Savino, *J. Eur. Ceram. Soc.* 37 (2017) 2325–2341, DOI 10.1016/j.jeurceramsoc.2017.01.018 |
| Gasch-2010-HfB2SiC | hfb2_sic | arcjet (NASA Ames AHF) | flat face | ~250–280 W/cm² (2.5–2.8 MW/m²) cold-wall | 0.10 atm | ~1690 °C (baseline; pyro) | 600 s | thin oxide + SiC-depleted subsurface zone; smooth surface | **survived (passive/protected)** — first HfB₂-SiC ground-test anchor | Gasch & Johnson, *J. Eur. Ceram. Soc.* 30 (2010) 2337–2344 |
| Gasch-2010-HfB2SiC-TaSi2 | hfb2_sic_tasi2 | arcjet (NASA Ames AHF) | flat face | 250 W/cm² (2.5 MW/m²) | 0.10 atm | 1515–1590 °C | 600 s | oxide 3–7 µm, SiC-depl 6–34 µm; TaSi₂ *reduced* oxide/depletion at this T | survived (TaSi₂ helps at ~1500–1690 °C — additive-inversion *low* side, vs Levine's 1927 °C melt) | ibid. |
| **Sevastyanov-2014-HfB2-45SiC** | hfb2_45sic (high-SiC, ~20 % porosity) | induction plasmatron (VGU-4, subsonic dissociated air) | flat-end cyl, 15 mm | 45–64 kW anodic | 100–300 hPa (10–30 kPa) | **2500–2700 °C** (pyro; parts 1700–1800 °C) | **>15–18 min at 2500–2700 °C; ~20 min >2000 °C; >30 min total** | **−1.5 % mass; no cracking/exfoliation** (X-ray µCT: no bulk defects); oxide HfO₂ 300–400 µm + borosilicate 200–300 µm + SiC-depleted HfB₂ | **survived — the high-T, long-dwell HfB₂ anchor** | Sevastyanov, Simonenko, Gordeev et al., *Russ. J. Inorg. Chem.* 59(11) 1298–1311, 2014, DOI 10.1134/S0036023614110217 |
| Savino-2008-HB5 | hfb2_mosi2 | arcjet (SPES, ~1 atm) | hemi R 7.5 mm | 5–8 MW/m² | **114–122 kPa (~1 atm)**; 20–28 MJ/kg | exceeding 2000 °C (up to ~2400 °C) | ~30 s | HfO₂ + SiO₂ scale; low catalytic | **survived** — highest-pressure UHTC point in the set | Savino, De Stefano Fumo, Silvestroni, Sciti, *J. Eur. Ceram. Soc.* 28(9) 1899–1907, 2008, DOI 10.1016/j.jeurceramsoc.2007.11.021 |
| Savino-2008-HC5 | hfc_mosi2 | arcjet (SPES, ~1 atm) | hemi R 7.5 mm | ~10 MW/m² | ~1 atm; 20–22 MJ/kg | ~2000 °C+ | ~40 s | HfO₂ scale; **catalytic transition low→partial near 2000 °C** (heat-flux rise; not oxidation runaway) | survived — first **HfC** (carbide) datum | ibid. |

Review-level context (not point data):
- **Peters et al., *Nat. Commun.* 15, 3328 (2024), DOI 10.1038/s41467-024-46753-3** —
  ZrB₂/HfB₂-SiC oxidation ceiling **~1650 °C** (the `continuous_K` anchor); an
  HfB₂-SiC nose cone at **80 min cumulative** arc-jet; carbides (HfC/ZrC) push
  service **>2000 °C**.  (Its 14.75 MW/m² · 130 s point is a *coated C/C* X-43
  edge — file under C/C, not UHTC.)
- **Glass, D. E., "Physical Challenges and Limitations Confronting the Use of
  UHTCs on Hypersonic Vehicles," AIAA 2011-2304, DOI 10.2514/6.2011-2304** (NASA Langley) — an
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
- **Squire, T. H. & Marschall, J., "Material property requirements for analysis
  and design of UHTC components in hypersonic applications," *J. Eur. Ceram.
  Soc.* 30 (2010) 2239–2251** (read from primary; PDF in repo `data/`) — the
  design-methodology companion to the anchors: the canonical statement of the
  **cold-wall vs hot-wall flux distinction** (arc-jet/Plasmatron conditions are
  characterized by fully-catalytic cold-wall flux, always higher than the
  hot-wall flux in flight — the like-for-like caveat our comparisons carry),
  the **catalysis uncertainty** (deriving recombination efficiency γ can be off
  by an order of magnitude, strongly emissivity-sensitive), and roughness-
  induced transition.  Contains **no aero-convective failure datum** — checked
  as a candidate for the missing plain-diboride cap — but explains *why* that
  datum is so scarce from the design side: sharp UHTC components are governed
  by **thermally-induced stress limits** (CTE/thermal-shock; components are
  redesigned smaller/lower-aspect when predicted stresses approach allowables),
  so hardware fails or is redesigned *mechanically* before the material reaches
  a thermochemical aero-convective limit — independently corroborating Glass's
  mechanical-failures-first finding.  The standing want (a plain-diboride
  aero-convective failure to cap the envelope top) therefore REMAINS OPEN, and
  may be practically unfillable below the PA transition — which is itself the
  reason the PA transition serves as the cap.

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

## Form A recession anchors (the H_eff chain)

The ballistic-RV recession chain is `δ = Q/(ρ·H_eff)`.  Its anchors split into a
**tuning** anchor (in-envelope) and two **bounding** anchors (recovered capsules),
and the distinction is load-bearing — details in `benchmarks/form_a/`.

*Paper archive:* the Form A primary sources live in repo `data/`
(`suzuki-fujita-yamada-2014-hayabusa-postflight-tps-jsr-a32549.pdf`,
`kontinos-stackpoole-2008-stardust-postflight-entry-aiaa-2008-1197.pdf`,
`winter-stackpoole-2014-remote-recession-sensing-pica-aiaa-2014-1151.pdf`,
`stackpoole-sepka-cozmuta-kontinos-2008-stardust-forebody-heatshield-aiaa-2008-1202.pdf`,
`yamada-inatani-ishii-2011-hayabusa-reentry-postflight-heatshield-aiaa-2011-3322.pdf`,
`johnston-hayabusa2-spectral-measurements-vs-simulations-nasa-larc-preprint.pdf`,
`sutton-1970-carbon-phenolic-ablation-experimental-nasa-tn-d-5930-ada309608.pdf`,
`johnson-2013-tps-materials-past-present-future-ntrs-20130014035.pdf`,
`schneider-teter-coleman-heath-1972-graphite-nosetip-design-aiaa-72-705.pdf`,
`sutton-graves-1971-stagnation-point-convective-heating-nasa-tr-r-376-ntrs-19720003329.pdf`,
`nestler-1979-carbon-carbon-nosetip-ablation-high-pressure-arcs-ntrs-19790010869.pdf`,
`perini-1971-graphite-ablation-review-jhuapl-ansp-m-1-osti-4286220.pdf`)
and in the project Google Drive reentry library (Berry,
`ReentryF_White_Paper_v2.pdf` — kept in Drive rather than the repo because its
underlying postflight reports carry a "may still be CUI/ITAR" caveat; also
there: TM X-2584 (`19790075224.pdf`), Fay–Riddell 1958, Tauber 1991, Olynick
1997 Stardust aerothermo, Thompson 1989, and others).  The Reentry-F primary
reports (TM X-2253, X-2560, X-2282, X-2584) were read from primary earlier in
this project — that analysis lives in `HEATING_TPS_REFERENCES.md` (Reentry-F
entry) and `HEATING_MODEL_CROSSCHECK.md` §10.6, and governs here.

**Anchor roles**
| anchor | role | what it fixes |
|---|---|---|
| **Reentry-F** (Mach ~20 entry, ATJ graphite nosetip R_n 0.1 in; NASA CR-154044 / TM X-1856 / LWP-460, all via Berry's nose-tip white paper in the project Drive) | *tuning* — the in-envelope δ/R_n shape-change ladder, now with the radius-history spread quantified from the clean TM X-1856 Fig. 11 (digitized: `benchmarks/form_a/reentryf_tmx1856_fig11.csv`): R_n 0.105 → **~0.20 in (thermochemical) / ~0.31 in (erosion-corrected)** at end of window, pressure-derived bars mostly straddling the lower curve; worst case reaching the **~0.39 in exposure radius** at 458.7 s **refuted** by the report itself | the accuracy-band ladder plus a **derived H_eff bracket 70–175 MJ/kg, central ≈114** (Q ≈ 3.87 GJ/m² ±20%, pixel-traced from the nominal-trajectory figure — `benchmarks/form_a/reentryf_nominal_qdot.csv` + full trace `reentryf_qdot_trace_full.csv` — ÷ 0.6–1.0 in axial-recession spread; apex 348 MW/m² @ ~47 kft, the 318 MW/m² pin = LWP window-max quote, traced window range 10–30×10³ vs quoted 9–28×10³ Btu/ft²·s; nominal 40 over-predicts ~2.9× → conservative; Q_MJ stays None in code — preflight prediction, never flight-measured) |
| **Stardust** (PICA, recovered, 12.8 km/s inertial; Q 276 MJ/m² wired, design upper-bound ~360 MJ/m²) | *bounding* | model must predict ≥ the measured near-stagnation maximum, Core 1 = 5.7±0.3 mm (no core exists at the geometric stagnation point — the SRC impacted off-center) |
| **Hayabusa** (carbon-phenolic, recovered, >12 km/s; peak convective 5.3 MW/m² + ~1 MW/m² radiative at 70 s, calc peak surface ~3200 K) | *bounding* | model must predict ≥ measured ~0.3 mm (laser scan, error <10%; no recession downstream — slight thermal expansion instead).  Shape caveats firsthand (Yamada AIAA 2011-3322): recession confined to stagnation, swelling ±0.3 mm elsewhere, char layer **uniform** by X-ray CT — no transition scars or local anomalies.  Char/virgin densities 1125/1325 kg/m³ and CT layer depths (char 5.5/4.0 mm, virgin 16.5/18.5 mm, stag/downstream) now firsthand from Suzuki Fig. 12, numerically corroborated by Johnston (NASA LaRC, Hayabusa 2 spectral preprint, repo `data/`).  All firsthand: Suzuki et al. *JSR* 51(1) 2014, DOI 10.2514/1.A32549 + Yamada, Inatani & Ishii AIAA 2011-3322, PDFs in repo `data/` |

**Bounds, not fits (read before touching H_eff).**  Post-flight analysis found
equilibrium-style ablation chemistry *over*-predicts capsule recession
(Hayabusa calc/measured ≈ 3×, Suzuki *JSR* [DOI 10.2514/1.A32549](https://doi.org/10.2514/1.A32549);
Stardust 51–61% over at the near-stagnation core and 22–25% at mid-flank —
**firsthand**, Kontinos & Stackpoole AIAA 2008-1197 Table 1, reproducing
Stackpoole et al. AIAA 2008-1202).  That chemistry conservatism exceeds the
radiative-gas heating the convective-only screen omits above ~9 km/s — for
Stardust the radiative component was only **9% of peak heat rate and 4% of heat
load** at stagnation (Kontinos auxiliary computations; including it moves the
calculated recession just 9.6 → 10.4 mm) — so the net bias is over-prediction.
The capsules therefore validate the chain only as a **lower bound**: predicted δ
must exceed measured δ.  **Do not "fix" the over-prediction by raising H_eff** —
that is the specific failure mode this bounding-vs-tuning split exists to prevent.

**Stardust firsthand recession table** (held firsthand from BOTH the primary —
Stackpoole, Sepka, Cozmuta & Kontinos AIAA 2008-1202, repo `data/` — and its
identical reproduction in Kontinos & Stackpoole AIAA 2008-1197,
Table 1, from its Ref. 22 = Stackpoole et al. AIAA 2008-1202; full rows +
environment in `benchmarks/form_a/stardust_recession.csv`):

| location | measured | calc (conv-only) | calc (conv+rad) | over-prediction |
|---|---|---|---|---|
| stagnation point | — (no core; off-center impact) | 9.6 mm | 10.4 mm | — |
| Core 1 (near-stagnation) | 5.7±0.3 mm | 8.6 mm | 9.2 mm | 51% / 61% |
| Core 2 (mid-flank) | 3.2±0.2 mm | 3.9 mm | 4.0 mm | 22% / 25% |

**Bound-test results** (`test_form_a_bounds.py`, run through the real
`heating.heating_figure_of_merit` path at the documented entry environments):

| case | material | predicted δ | measured δ | ratio | verdict |
|---|---|---|---|---|---|
| `stardust_bound` | PICA (ρ 270, H_eff 35) | 29.1 mm | 5.7 mm (Core 1) | **5.1×** | bound holds (≥1) |
| `hayabusa_bound` | carbon-phenolic (ρ 1450, H_eff 15) | 9.2 mm (firsthand 5.3 MW/m² peak, τ 60 s labeled estimate) | ~0.3 mm | **31×** | bound holds (≥1) |

The large ratios are *expected and safe*: the screening chain uses full-load Q ×
a single conservative H_eff, far cruder than FIAT (~1.5× on Stardust), and a
lower bound wants headroom.  Ratio < 1 would signal a broken Q pipeline or bad
H_eff — halt and investigate, not a radiative shortfall.

**H_eff bands** (replacing the bare screening points; nominals unchanged so
verdicts are stable, now justified as conservative-low within the literature Q\*
band — Q\* is enthalpy-dependent, not a constant).  Full derivation +
provenance: `benchmarks/form_a/phase2-heff-bands.md`.
| material | ρ (kg/m³) | low | **nominal** | high | basis |
|---|---|---|---|---|---|
| carbon_phenolic | 1450 | 10 | **15** | 30 | **cited (Sutton, NASA TN D-5930, 1970, read from primary — `benchmarks/form_a/sutton_cp_qstar.csv`)**: clean rows Q\* ≈ 68–195 MJ/kg; **char-removal regime (onset ≥2.4 atm air) collapses to 14–20 MJ/kg** — the nominal 15 equals the measured severe-regime value, conservative for RV entries and ~10× conservative for capsule pulses (Hayabusa 31× ✓); zero recession in pure N₂ |
| pica | 270 | 25 | **35** | ~100+ | PICA Q\* higher than CP, rises sharply with enthalpy; nominal 35 conservative-low (over-predicts Stardust 5×). **Cited arc-jet point:** Winter et al. AIAA 2014-1151 — 10.36 MW/m² flat-face, recession 0.5–1.0 mm/s → implied Q\* **38–77 MJ/kg**; nominal sits at/below the low edge |
| carbon_carbon | 1800 | 25 | **40** | 60 | C/C oxidation→sublimation regime — **theory anchor now cited**: Scala's CO-diffusion-limit with blowing offset, ṁ = 0.1725·ρₑuₑC_H₀ → Q\* ≈ h_t/0.1725 ≈ **5.8 × total enthalpy** (Perini 1971, JHU/APL ANSP-M-1, read from primary, repo `data/`; data scatter +10/−50% about theory).  The 25–60 band ↔ h_t ≈ 4–10 MJ/kg — deliberately conservative-low for RV-class enthalpies.  **Reentry-F flight-derived bracket 70–175 MJ/kg (central ≈114, pixel-traced Q ≈ 3.87 GJ/m²)** independently corroborated by the Scala/Perini form: 18.6 MJ/kg ÷ 0.1725 ≈ 108, within ~6% of 114.  Nominal 40 over-predicts recession ~2.9×, the conservative side for a screen (`benchmarks/form_a/phase2-heff-bands.md`). **Validity floor:** Nestler 1979 (read from primary) shows the band does NOT apply at ≥80 atm stagnation pressure — severe-regime Q\* collapses to ~10–20 MJ/kg; Perini corroborates (JANAF sublimation predictions under-estimate measured loss by up to 70% at the p = 4 atm / 4000 K extreme — mechanical erosion / higher-order carbon species) |

**Two P3 items surfaced honestly by the capsules (logged, not hidden):**
- **P3-radiative:** the convective-only screen ends ~9 km/s; above it radiative
  gas heating is unmodeled.  The magnitude is **size-dependent**: for the small
  (0.827 m) Stardust capsule it was only 9% of peak rate / 4% of load, but for a
  CEV-scale (5 m) blunt body ~40% of peak flux is radiative (both Kontinos &
  Stackpoole AIAA 2008-1197) — larger shock volume, more radiant gas.  Partially
  masked by P3-chemistry at capsule scale.  Affects only >9 km/s cases — no
  operational Form A trajectory in the current use set exceeds this, which is
  why it is P3.
- **P3-chemistry:** equilibrium-style recession is conservative vs. flight
  (Hayabusa ×3, Stardust 51–61% near-stagnation) — the *larger* bias at capsule
  scale.  A finite-rate option (Park/Milos lineage) is the eventual fix; until
  then the model is honestly conservative and the bound tests enforce the sign.

## Threshold provenance audit

Standing rule: **every threshold the survivability model consumes is either
cited to a real source, or explicitly labeled as an internal inference /
placeholder — never given a fake citation.**  Status of every threshold:

| Threshold | Value | Provenance | Status |
|---|---|---|---|
| Sutton-Graves constant | 1.7415e-4 | Sutton & Graves 1971, NASA TR R-376 | ✔ cited |
| shape-change onset | δ/R_n = 0.10 | Lin, Grabowsky & Yelmgren 1982 (TRW/BMO): 0.1 R_N "mildly indented"; PANT (DTIC ADA019186): asymmetry→dispersion | ✔ cited |
| severe blunting | δ/R_n = 0.50–1.0 | Berry, Reentry-F (NASA CR-154044): flew R_n 0.10→0.171 in (~0.7 R_n), survived; clean TM X-1856 Fig. 11 (digitized, `benchmarks/form_a/reentryf_tmx1856_fig11.csv`) widens the survived-blunting spread to ~0.20–0.31 in final radius (**1–2.1 R_n₀ radial growth**), pressure-derived bars nearer the lower curve; worst case (exposure radius ~0.39 in at 458.7 s) refuted | ✔ cited (spread quantified) |
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
| additive inversion | TaSi₂ best @ 1627 °C, destroyed @ 1927 °C | Levine et al. 2003 furnace (NTRS 20040033992); low-side corroborated by Gasch & Johnson 2010 (TaSi₂ *reduces* oxide/depletion at 1500–1690 °C, HfB₂-SiC arcjet) and the Di Maso cone at temperature | ✔ cited |
| acreage flux fraction | 0.13 × stagnation | Lu, Shi, Zhang et al. 2024 (IJHMT 225; validated <9 %) | ✔ cited |
| bondline limit | 250 °C | NASA NTRS 20060004824 (ablative TPS sizing); Orion 260 °C NTRS 20080013535 | ✔ cited |
| tile/RCC/material limits | per-material peak/continuous K | HEATING_TPS_REFERENCES.md §2 (TPSX, KSC STS ref, NTRS 19940030739, Peters 2024, …) | ✔ cited per entry |
| analytic-honesty factor | ×2–4 | **internal**: paired analytic/numerical C-HGB runs in this tool | ⚠ labeled internal |
| `NOTHING_SURVIVES_K` | 4000 K | **modeling-validity bound** (radiative-equilibrium model invalid above all usable materials), not an empirical limit | ⚠ labeled model bound |
| `uhtc` `oxidation_dwell_s` (current code) | 300 s (demonstrated floor, floor-not-cliff) | **implemented** — the uncited 120 s placeholder is retired; the code now carries the cited floor (Monteverde 2013, 300 s @ 1973 K zero recession; sharp-tip extension 575 s, Monteverde 2012) and the Form B coverage verdict treats crossing it as extrapolation, not failure (SRD §11.4) | ✔ cited |
| ablator `H_eff_MJ_kg` nominals | CP 15 / PICA 35 / C/C 40 | Q\* is enthalpy- and regime-dependent; every nominal now has a cited basis. **CP 15 = the measured char-removal-regime value** (Sutton TN D-5930 Table I: 14–20 MJ/kg at ≥2.4 atm; clean rows 68–195). **PICA**: Winter et al. AIAA 2014-1151 arc-jet point, implied Q\* 38–77 MJ/kg at 10.36 MW/m² — nominal 35 at/below its low edge. **C/C**: Perini/Scala diffusion-limit theory (5.8·h_t) + Reentry-F flight bracket + Nestler severe-regime floor | ✔ all three nominals cited (regime-labeled) |
| CP char-removal onset | ≥2.4 atm stagnation (air, K_O₂ 0.23); ~6 atm (air-N₂ mixes); zero recession in pure N₂ | Sutton, NASA TN D-5930 (1970), Langley ceramic-heated + arc tunnels, Narmco 4028 CP, ρ 1392 kg/m³ — read from primary, PDF repo `data/`; the CP analogue of the C/C mechanical-erosion bounds (Nestler ≥80 atm gouging, Schneider >55 atm) | ✔ cited |
| Form A capsule bounds | Stardust 5.1×, Hayabusa 31× (predicted/measured) | Stardust measured 5.7±0.3 mm near-stagnation Core 1 — **firsthand ×2**: Stackpoole, Sepka, Cozmuta & Kontinos AIAA 2008-1202 (the primary; Table 1, error ±3–5%, calc FIAT v2.44 + PICA v3.3 conv+rad; PDF in repo `data/`) and Kontinos & Stackpoole AIAA 2008-1197 (identical reproduction).  Primary's own reading: the flank 25% discrepancy is within the model's arc-jet calibration scatter; the 61% near-stagnation over-prediction "not fully understood."  Hayabusa measured ~0.3 mm (laser scan, error <10%), calc/meas ≈3×, peak convective 5.3 MW/m² — **firsthand**, Suzuki et al. *JSR* 51(1) 2014, DOI 10.2514/1.A32549 (Hayabusa pulse duration in the bound test is a labeled 60 s estimate from the paper's heating window) | ✔ cited (both capsules firsthand; Stardust doubly) |
| Reentry-F H_eff bracket | 70–175 MJ/kg, central ≈114 (flight graphite, 5–60 atm) | derived: Q ≈ 3.87 GJ/m² ±20%, **pixel-traced** from the nominal-trajectory figure (γ_E 21.2°, V_E 20,300 ft/s; the figure Berry reproduces as his Fig. 6 [LWP-460]; embedded scan extracted from the Berry PDF, per-ruler tick calibration, apex-first slope tracking, overlay-QC'd — method + summary `benchmarks/form_a/reentryf_nominal_qdot.csv`, full trace `reentryf_qdot_trace_full.csv`) ÷ (ρ 1.73 g/cc vendor-nominal × 0.6–1.0 in axial-recession spread, CR-154044 0.77 in central); apex 348 MW/m² @ ~431.7 s (~47 kft), the 318 MW/m² `_BENCHMARKS` pin = LWP window-max quote; validation: traced in-window range 10.2–30.2×10³ vs LWP's quoted 9–28×10³ Btu/ft²·s + Sutton-Graves apex check ~340–380 MW/m²; 100→50 kft window ~8 s; supersedes both the ~1 GJ/m² order-of-magnitude read and the intermediate 2.85 GJ/m² eyeball table (wrong curve through the mid-rise, caught by overlay QC); Q_MJ stays None in code — preflight-nominal, no flight-measured stagnation heating exists (TM X-2560) | ⚠ labeled derived (pixel-traced; nominal 40 over-predicts ~2.9×, conservative) |
| C/C severe-regime Q\* cap | ~10–20 MJ/kg at 80–168 atm (band validity floor) | Nestler 1979 (NTRS 19790010869, read from primary, repo `data/`): measured 3-D C/C steady-state recession 0.508–0.787 cm/s at 80–168 atm / T_w 4,000–4,167 K / H_CL 6.9–11.6 MJ/kg, roughness-augmented heating 1.4–1.5×, gouging onset ~60–77 atm along 45° weave rays; Q\* derived via the paper's own energy balance with a flagged nominal ρ ≈ 1.9 g/cc and Fig.-5 H_w.  Corroborated by Perini 1971: JANAF sublimation theory under-estimates measured loss up to 70% at the p = 4 atm / 4000 K extreme (erosion / higher-order species) | ⚠ measured rates cited; Q\* labeled derived |
| C/C diffusion-limit Q\* (theory anchor) | Q\* ≈ h_t/0.1725 ≈ 5.8 × total enthalpy (cold-wall, moderate regime) | Scala's CO-diffusion-limit correlation with blowing offset (ṁ = 0.1725·ρₑuₑC_H₀), via Perini, *Review of Graphite Ablation Theory and Experimental Data*, JHU/APL ANSP-M-1, Dec 1971 (OSTI 4286220), Eq. 28 — read from primary, repo `data/`; diffusion-limit data scatter +10/−50% about theory.  Independent check on the Reentry-F flight-derived central: 18.6/0.1725 ≈ 108 vs 114 MJ/kg (~6%) | ✔ cited |
| Stardust radiative fraction | 9% of peak rate / 4% of load (stagnation); CEV-scale ~40% of peak flux | Kontinos & Stackpoole AIAA 2008-1197 (auxiliary computations; §II for CEV comparison) — scales the P3-radiative item | ✔ cited |

Rows marked ⚠ are the complete list of thresholds NOT backed by literature;
each is labeled with its true epistemic status in code and report text.  If a
future source covers one, replace the label with the citation — never the
reverse.
