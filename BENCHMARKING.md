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

#### HTV-2 primary/reconstruction corpus: Acton 2015 + Wright 2015 + DARPA FOIA 14-F-0122 (read from primary, archived 2026-07-20)

The X-51/HTV Drive folders (user-supplied) yielded the full open-literature
HTV-2 source set, now archived to repo `data/` (S&GS self-distributes free
PDFs from scienceandglobalsecurity.org — the exact files here; the FOIA set is
Distribution A / DARPA public releases):

- **Acton, "Hypersonic Boost-Glide Weapons," *Science & Global Security*
  23:191–219 (2015), DOI 10.1080/08929882.2015.1087242**
  (`acton-2015-hypersonic-boost-glide-weapons-sgs23.pdf` + the online
  supplement `acton-2015-sgs23-appendix-online-supplement.pdf`): the published
  HTV-2 trajectory reconstruction the shipped `HTV-2.ro.json` is built from —
  **A-flight fit: L/D = 2.6, glide β_L = 13,000 kg/m² (his β is the metric
  mass form, matching the repo convention), R = 4.9×10³ km, glide start
  h₄ ≈ 47 km ~1,100 km downrange; entry (t₂=435 s) γ₂ = 5.03°, h₂ = 100 km,
  v₂ = 6,010 m/s; endo end 30.5 km at 5,900 km.**  B-flight (2011): entry
  γ₂ = 3°, v₂ = 7,170 m/s, ~2,300 km design cross-range, same L/D/β_L
  assumed.  He flags the fitted entry-phase β_S = 7.0 kg/m² as physically
  unreliable (Newtonian flat-plate gives ≫; the RO notes already carry this).
  L/D 2.6 vs NRC-2008's assumed 2.2.
- **Wright, "Research Note … Boost Phase of the HTV-2 Tests," *S&GS*
  23:220–229 (2015), DOI 10.1080/08929882.2015.1088734**
  (`wright-2015-htv2-boost-phase-analysis-sgs23.pdf`): NOTMAR-constrained
  Minotaur IV-Lite (Peacekeeper 3-stage) boost reconstruction → **HTV-2 mass
  ≈ 1,000 kg** (the shipped RO's mass source), stage table, 16° stage-3
  dogleg (ΔV ≈ 1.0 km/s), A-flight burnout 6.0 km/s / 123 km / 3.7°.
  *Flag:* Wright's Table 1 pairs the pierce-point angles opposite to Acton's
  tables (A: −3° / B: −5.03° vs Acton A: 5.03° / B: 3°) — a crossed column in
  one of the two; the repo RO follows Acton's pairing (his fit used it).
  Not resolved here.
- **DARPA/USAF FOIA release 14-F-0122**
  (`foia-14-f-0122-darpa-htv2-x51-documents.pdf`), all Distribution A:
  - HTV-2 program brief: objectives, **"high lift-to-drag ratio, advanced
    carbon-carbon aeroshell"** — primary confirmation of the C/C aeroshell
    (repo `cc_hot_structure` body); Flight 1 (22 Apr 2010) roll-yaw-coupling
    anomaly during pull-up (adverse yaw exceeded flap roll authority; fix:
    CG shift, lower AoA, RCS augmentation); AoA capability to 89° demoed at
    release.
  - **HTV-2 Flight-2 ERB conclusion (DARPA release, 20 Apr 2012) — the
    citable primary for the Form B "too hot" flight anchor**: ~3 minutes of
    stable aero-controlled flight at up to Mach 20, then "unexpected
    aeroshell degradation — larger than anticipated portions of the vehicle's
    skin **peeled from the aerostructure**," the gouges creating impulsive
    shock waves ~100× design tolerance that repeatedly rolled the vehicle
    until the flight-safety system terminated.  Confirms the failure was
    thermal-material (skin/TPS loss → aero upset), not aerodynamic design.
  - DARPA Integrated Hypersonics (2012): Mach-20 endoatmospheric skin
    temperatures "**exceeding 3,500 °F**" (~2,200 K) — DARPA's own figure,
    consistent with the C/C–UHTC-class band the Form B screen uses.
  - **X-51A Waverider fact sheets (USAF)** — first ledger entry for X-51A:
    airbreathing scramjet cruiser (not a glider; Form B context datum only).
    TPS: "primarily standard aerospace materials (aluminum, steel, inconel,
    titanium); some carbon/carbon composites on the leading edges of fins and
    cowls; Boeing-designed **silica-based TPS and Boeing Reusable Insulation
    tiles, similar to the Shuttle Orbiter's**."  Flight 4 (1 May 2013):
    **Mach 5.1 at ~60,000 ft, 240-s burn, >230 nmi in ~6 min, 370 s of
    data — the longest airbreathing hypersonic flight** — i.e. a demonstrated
    ~370-s Mach-5-class dwell on Shuttle-class insulation + C/C edges,
    a low-Mach corroboration point for the NRC duration-ladder's
    conventional tier.

#### Defenses folder sweep (2026-07-20, folder `1YWhSDy_…`) — the target as a class

Single file: **Peace, Pulimidi, Umapathy, Singh, Lu (UT Arlington) & Barnard
(Lockheed Martin), "Mid-Tier Defense Against Hypersonic Glide Vehicles During
Cruise," AIAA 2018-5254, DOI 10.2514/6.2018-5254** (archived to repo
`data/peace-pulimidi-2018-…-aiaa-2018-5254.pdf`, AIAA-conference-paper
precedent).  A conceptual THAAD-baseline hit-to-kill interceptor design — the
defense-side value here is its **stated threat-HGV class requirements** (the
defense community's characterization of the Form B vehicle as a target):

- **Threat HGV class: mass 2,000–5,000 lb (907–2,268 kg); cruise Mach 12–15;
  cruise altitude 120–140 kft (36.6–42.7 km)**; maneuvering or
  non-maneuvering, with "a maneuver possible at any stage of the engagement"
  (maneuver *magnitude* unquantified — no g-level given); raid model 3 HGVs
  at 15-s spacing.  The mass band brackets the HTV-2's ~1,000 kg (Wright) at
  its low end.
- **Corroboration check (run 2026-07-20, repo atmosphere)**: Thrusty's
  equilibrium-glide altitude for the shipped HTV-2-class glider
  (β_L = 13,000 kg/m², L/D = 2.6) is **125 kft at Mach 12 and 139 kft at
  Mach 15 — the paper's 120–140 kft threat band is precisely the
  equilibrium-glide corridor Thrusty computes** for that class.  Independent
  (adversarial-design) confirmation of the Form B glide-corridor physics,
  complementing the CBO 30–40 km band check.
- Defense-side context facts (not screening anchors): ground-radar horizon at
  that altitude ≈ 420–460 mi → ~7-km defended radius with a Mach-6-class
  single-stage interceptor, ~60-s time-to-target; pure proportional
  navigation N′ = 4 in their engagement model.

The threat table cites the X-43 / X-51 / HTV-2 / AHW lineage as the class
exemplars — consistent with the ledger's Form B anchor set.

#### Cone-aero databook: Eastman, Boeing D2-36139-1 (read from primary, archived in 3 parts)

**Eastman, D. W., "Aerodynamics of Conical Bodies (U)," Boeing D2-36139-1,
rev. A (approved Dec 1965 / rev 10-12-66), 187 sheets; DTIC AD0376942,
declassified from Confidential 10 Dec 1978** — a compilation/evaluation of
theoretical + experimental cone aero, Mach 0–25, "emphasis on slender cones
such as might be used for ballistic missile re-entry," several hundred
references indexed by test condition and geometry.  Mined findings:

- **Sharp-cone center of pressure (the trim-gate anchor)**: Eq. 3.2,
  **X_cp/l = 2/(3 cos²δ)** from the apex — from ray geometry (conical-flow
  *and* Newtonian), **Mach-independent (shock attached) and constant with α
  to ≤90°**; "experimental results substantiate equation 3.2."  Fig. 3.6
  (M = 6.8, cg at 0.75 l, δ = 5–50°) encodes a self-check: the δ = 20° cone
  reads C_m ≈ 0 for all C_N, and 2/(3cos²20°) = 0.755 ≈ the 0.75 l cg ✓.
  **Thrusty comparison (run 2026-07-20):** the Barrowman buildup
  (`grid_fin_sizing._nose_cp_fraction`, cone = 0.666) sits **+0.9% forward
  of the substantiated value at δ = 5°, +3.2% at 10°, +5.0% at 12.5°,
  +13.4% at 20°, +33% at 30°** — fine at screening tier for slender
  (RV-class) noses, degrades for fat cones; limitation now noted in
  `trim_gate.py` / `grid_fin_sizing.py` with this citation.  Fig. 3.7 gives
  the Newtonian blunt-cone c.p. chart (vs d/D and L/D) if a blunted
  correction is ever wanted.
- **Newtonian C_N validated**: sharp cones "at most angles of attack"
  (overpredicts α > 60°); blunt cones at AoA good to α = 45° at M 9.75
  (Fig. 2.11, r_N/r_b to 0.763).  But **Newtonian fails for blunted-cone
  C_Nα at α = 0** (reductions much larger than 2sin²δ predicts; Modified
  Shock Expansion needed) — the same slender-blunt caveat class the repo's
  cone wave-drag validation notes for blunt/low-L/D shapes.
- **Base pressure** (§4): ~constant with α, ~independent of cone half-angle,
  larger for a cone than a cone-cylinder; nose blunting / base rounding
  negligible; **strongly sensitive to Reynolds number and sting mounting,
  and free-flight base drag differs significantly from wind-tunnel values**
  (their Figs. 4.13–4.15) — an honesty band to carry for `_cd_base`
  (Chin 1961 / DATCOM are tunnel-derived curves).
- **Boundary-layer transition** (§7): the ten-factor list (Re, M, roughness,
  wall/edge enthalpy ratio, nose bluntness, tunnel turbulence, pressure
  gradient, injection, vehicle dynamics, ablation shape change; "Mach and
  Reynolds generally conceded the most important"); **flight transition data
  were Secret** (their ref 536, "transition data from flights of actual
  reentry vehicles") and ground facilities could not reach flight transition
  Re — the 1966 statement of exactly the open-literature scarcity the repo's
  transition-uncertainty treatment (Thompson band, PANT-era anchors) works
  around.
- **Ablation/blowing effects on aero** (§9): simulated-ablation blowing
  **reduces C_N by a large amount** (α = 0), **reduces skin friction but
  increases wave drag**; small ablation rates reduce total axial force,
  larger rates can go either way; "most of the meager experimental data are
  classified SECRET."  A cited mechanism note for the Form A
  ablation-alters-aero coupling (alongside the δ/R_n shape-change ladder).
- **"LORV" identified**: the databook's recurring "Summary of all LORV
  vehicle flight tests" (data spanning **Mach 1–25**) references the **AVCO
  RAD "Low-Observable Reentry Vehicle" flight-test program** — per-flight
  evaluation reports for vehicles **L-1** (RAD-SR-64-307 Add., AD-363497)
  and **L-4** (RAD-SR-65-259, AD-366403) and the program summary
  **RAD-SR-66-31 Vol. I (AD-370054)**, all Secret at the time — an early
  (mid-1960s) low-observable/penetration-aids RV flight program; candidate
  DTIC declassification hunts if that lineage ever matters here.
- §5 (dynamic damping in pitch, C_mq-class derivatives) noted but not mined —
  beyond the screening tier (Thrusty's ζ is a guidance-law knob, not an aero
  damping derivative).

#### General Hypersonics folder sweep (2026-07-20, folder `1uGJ-ok4…`)

Four substantive finds, three archived to repo `data/`:

- **Tracy & Wright, "Modeling the Performance of Hypersonic Boost-Glide
  Missiles," *S&GS* 28 (2020), DOI 10.1080/08929882.2020.1864945**
  (`tracy-wright-2020-modeling-hypersonic-boost-glide-sgs28.pdf`) — an
  independent published model using **the same methodology chain as Thrusty's
  Form B screen**: stagnation-point heating + radiative-equilibrium wall
  temperature, ε = 0.85 carbon aeroshell, HTV-2 aero parameters from Acton's
  flight fit, leading-edge radius 0.034 m.  **Cross-check run against
  Thrusty's chain**: their Fig. 10 stagnation-region temperature ~3,200 K at
  v = 6 km/s / h = 49.7 km; Thrusty's cold-wall Sutton-Graves +
  radiative-equilibrium at the same point gives **3,408 K (+6.5%)** — high in
  exactly the conservative direction expected, since Tracy applies a hot-wall
  enthalpy correction and the repo deliberately stays cold-wall (screening
  bias).  Their Fig. 11 (centerline 1 m aft of the nose): ~1,200–2,000 K
  across the glide for 5–7 km/s entry speeds.
- **Glass, Dirling, Croop, Fry & Frank, "Materials Development for Hypersonic
  Flight Vehicles," AIAA 2006-8122, NASA NTRS 20070004792**
  (`glass-2006-…-ntrs-20070004792.pdf`) — the **Falcon MIPT paper** (same
  Glass as the 2011 C/C–SiC survey already cited in METHODS): the HTV-2/HCV
  TPS technology areas with their design-class temperatures — **C-C leading
  edges at nominal use <3,000 °F (= 1,922 K — independently matching the
  repo's `uhtc`/hot-structure 1,923 K continuous ceiling almost exactly)**,
  a >3,000 °F task targeting **3,600 °F (2,255 K) refractory composites**
  (Ir/HfO₂ MLOP, PCP-SiC), high-temperature multi-layer insulation to
  3,000 °F (backface ≤350 °F), acreage C/SiC & C-C rib-stiffened panels, and
  ~3,000 °F wafer seals.  Design-practice corroboration of the Form B
  material-class bands, from the HTV-2 program itself.
- **Spravka & Jorris, "Current Hypersonic and Space Vehicle Flight Test and
  Instrumentation," AFTC 412TW-PA-15264 / DTIC ADA619521 (2015, Dist. A)**
  (`spravka-jorris-2015-…-dtic-ada619521.pdf`) — the embedded Hypersonic-CTF
  brief carries **independent HTV-2 endo numbers: Mission A endo flight time
  1,363 s / endo range 3,180 nm (5,889 km); Mission B 1,409 s / 3,079 nm +
  1,250 nm cross-range (2,315 km); coast altitude 450 kft** — the A/B endo
  times match Acton's table-derived durations (1,798−435 = 1,363 s;
  1,785−376 = 1,409 s) **exactly**, and A's endo range matches Acton's
  x₅−x₂ = 5,900 km and the B cross-range his ~2,300 km.  Strong independent
  corroboration of the Acton reconstruction the HTV-2 RO is built on.  Also:
  X-51A max altitude 71 kft / 310 nmi; AHW flew ~2,500 mi in <30 min
  (first flight 3,700 km).
- **RAND RR-2137, Speier, Nacouzi, Lee & Moore, *Hypersonic Missile
  Nonproliferation* (2017) — facts-only, NOT archived** (RAND's rights page
  prohibits online posting; personal-use duplication only).  Appendix A
  presents the total-heat-transfer scaling **Q ∝ ∫√(ρ/R_n)·V³ dt — the same
  Sutton-Graves form Thrusty integrates**, plus the larger-nose-radius and
  trajectory-shaping mitigation arguments; the rest is a global
  hypersonic-facility survey and export-control analysis (no anchor datums).

Also present, identified but not mined: Rana & Chudoba, AIAA 2016-5319
(lifting-reentry-vehicle design historiography, Tsien → Dream Chaser; 19.5 MB
>connector cap; survey, no screening datums — parked); Micol, "Experimental
Hypersonics at NASA Langley" (legacy binary .ppt, unreadable via the
connector); duplicate copies of the Bedke and Sponable decks triaged in the
X-51/HTV sweep.

*X-51/HTV Drive-folder triage (2026-07-20):* mined & archived as above.  Also
present, triaged but **not mined**: Davies, "Infrasonic Characterization of
the Falcon HTV-2" (.pptx, 12.8 MB — geophysical observable of the flight, no
aerothermal datum expected); Bedke "High Speed Weapons — What is Different
Today" and Sponable "Reusable Space Systems" (legacy binary .ppt, unreadable
via the Drive connector; briefing-deck context, low datum value);
`FalconHTV2FlightPath.jpeg` (the DARPA trajectory graphic — the same image
Wright cites as his pierce-point source).

*X-51 folder re-check (2026-07-20, second pass):* the folder holds exactly two
files — the FOIA 14-F-0122 PDF (byte-identical to the HTV-folder copy already
mined/archived) and `4217801.ppt`, now **identified** (it is OOXML despite the
.ppt extension — text extracted): the **X-51A Collier Trophy nomination deck**
(X-51A WaveRider team, post-2013).  Facts-only (promotional briefing, no
distribution statement — not archived).  One datum refines the X-51A ledger
entry — the **Flight-4 mission clock (slide 7, 1 May 2013): total flight
448.0 s = boost 31.0 s + engine start 12.0 s + scramjet run 212.0 s + descent
193.0 s; max Mach 5.1, max altitude 63,500 ft, distance 240.6 nm** — i.e. the
airframe's demonstrated hypersonic dwell on Shuttle-class TPS is ~420 s
(scramjet run + descent), consistent with and sharper than the fact sheet's
"370 s of data."  Design slide corroborates: "conventional materials + high
Mach thermal protection."

### Form C — Maneuvering quasi-ballistic (the envelope)
**Computed:** terminal-dive segment peak flux + duration vs airframe limit
(screening; windward/AoA probe is a later tier).
**Material constants consumed:** body/airframe `peak_K`, `c_J_kgK` (transient
heat-sink).

Find: pull-up altitude & speed, airframe material, survived-the-maneuver.
Candidates: **Pershing II** (operational MaRV pull-up), SWERVE (maneuvering
flight record — already anchored).

> **ENCODED (2026-07-20):** the maneuver-load anchors below now live in code as
> `survivability_report.MANEUVER_ANCHORS` (same data-edit philosophy as
> `UHTC_ANCHORS`; METHODS §13.7), and the Form C report prints a
> demonstrated-envelope context block comparing the plan's commanded
> `glider_pullup_g_max` to the ladder (≤25 g operational class / ≤100 g AMaRV
> flight-demonstrated ceiling / beyond = extrapolation).  Context only — never
> a survivability verdict.  This file remains the citation of record.

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

#### Form C maneuver numbers: Yengst, *Lightning Bolts* (2010) — the keystone

William Yengst (ABRES/RV engineer), *Lightning Bolts: The Race to Develop
Maneuvering Reentry Vehicles* (2010) — a full-length development-and-test
history of U.S. (and foreign) MaRVs.  **Copyright: the book is NOT archived to
the repo** (commercial title; it stays in the user's library) — only the cited
facts are recorded here, with attribution.  The maneuver-load numbers the
Form C screen wanted, per Yengst:

- **AMaRV maneuver load ≈ 100 g** — the guidance unit was required to hold
  accuracy through ~100-g reentry maneuvers, and Bell XI accelerometers
  measured >100-g reentry maneuver levels on the flights (three: 20 Dec 1979,
  8 Oct 1980, 4 Oct 1981; CEP ~1,250 ft).  This is the AMaRV-specific g the
  ledger was missing — it quantifies Allen 1997's "high-G maneuvers" and
  matches Regan 1993's 100-g evader cap.  Geometry re-confirmed: 10.4° / 6°
  biconic (= Allen).
- **Pershing II MaRV**: ~**Mach 8** ballistic reentry, then below ~50,000 ft
  it retained energy for a **~25-g pullout + ~30-mile range extension** (during
  which RADAG ran its radar map-match accuracy update), or alternatively a
  high-g evasion maneuver.  This is the Pershing II RADAG pull-up datum
  (altitude/Mach/g) — the other open want, now filled.
- **BGRV** (boost-glide RV): Atlas boosted to ~130,000 ft, pitched to
  horizontal to build speed, then separated BGRV at **>Mach 15** onto a glide
  toward ~110,000–125,000 ft; components qualified to 25 g.  Fills the "BGRV
  early high-β maneuver datum" want.
- **MBRV**: corroborates the four-vehicle Atlas/Vandenberg account (Lin) — the
  qualification loads were the high-g reference the BGRV/AMaRV programs scaled
  against.

*AMaRV TPS resolved:* Allen 1997 states the terrestrial AMaRV used
**carbon-carbon** (see the Allen section above), nose radius 2.34 cm — so the
AMaRV reference reentry object can be fully specified (biconic 10.4°/6°, 470 kg,
β 13,485 kg/m², AoA 10°, R_n 2.34 cm, nose material `carbon_carbon`, ~100-g
maneuver plan).  *Pershing II TPS/maneuver — now resolved from primary* (Lund
1984, see the Pershing II subsection below): **ablative radome** over the RADAG
antenna + **velocity-control pullup/pulldown** terminal maneuver; only a
quantitative windward-temperature/thickness datum stays open (unpublished in
the open sources on hand).  *Still lower-priority:* primary DTIC flight-test
reports if publication-grade quantitative citation is needed (Yengst/Lund are
program/test histories, authoritative insider syntheses but light on thermal
data).

#### Contemporaneous trade-press corroboration: AW&ST (facts only)

Several *Aviation Week & Space Technology* articles (period trade press; **not
archived — copyrighted; facts recorded with attribution**) independently
corroborate and date the Form C programs:

- **AMaRV first flight — 20 Dec 1979**, Minuteman-1 from Vandenberg to Kwajalein
  Atoll; first of three planned launches (Smith, AW&ST 11 Feb 1980).  Matches
  Yengst's flight dates (1979-12-20 / 1980-10-08 / 1981-10-04) exactly.
  ~50–75% of objectives met; the vehicle **did perform a pull-up** but the
  planned pull-down/diving turn was marginal/abandoned after a Minuteman-1
  booster anomaly left it too high and not fully restabilized — it coasted and
  impacted the ocean.  Full three-axis self-contained inertial guidance +
  digital computer; nearest predecessor the ACE (Advanced Control Experiment,
  completed 1976; MDAC prime for both).  Program mgrs Lt Col M. Buchen /
  Maj J. Traeger.  Maneuvering-RV investment since 1963 ≈ $224 M; PGRV renamed
  AMaRV (Miller, AW&ST 24 May 1976).
- **ABRV (Advanced Ballistic Reentry Vehicle)** — a *ballistic*-RV anchor,
  new to the ledger: MX-program vehicle / option for Mk.12A, ~5 ft length,
  **carbon-carbon nosetip that ablates symmetrically** during reentry —
  deliberately, to avoid the asymmetric nosetip shapes that would produce
  unwanted aerodynamic lift and accuracy loss.  Program 1977, three flights on
  Boeing/Minuteman-1 (first two successful), completed ~1983 (AW&ST 16 Jun
  1980).  **This directly corroborates the Form A δ/R_n shape-change ladder**
  (asymmetric recession → dispersion; SHAPE_CHANGE_ONSET, Lin 1982/PANT).
- **Nosetip TPS genealogy**: Mk.12 / Mk.12A use **carbon-phenolic** nosetips;
  a **transpiration-cooled** nosetip was an early (1976) AMaRV candidate to
  prevent tip ablation (vs the carbon-carbon actually used per Allen).
- **MBRV / BGRV / Mk.500 corroboration** (Miller 1976): MBRV <3,000 lb,
  flaps + open-loop autopilot, 3-of-4 GD Atlas launches; BGRV (MDAC) long
  slender high-L/D, ~3,000-lb class, two Atlas launches, low-level hypersonic
  glide; Mk.500 "Evader" (GE) Navy-flight-tested 5×, Trident option, simpler/
  less-accurate than the Lockheed Mk.400.  Gas-jet (reaction-control) RV
  maneuvering was flight-demoed on Celesco Athena at White Sands but judged
  not to scale to operational size without a high weight penalty (→ the
  aero-control choice for MBRV/BGRV/AMaRV).
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

*Parked for later (identified, archived to repo `data/`, not yet mined):*
- Richie, "The Common Aero Vehicle: Space Delivery System of the Future,"
  AIAA 1999-4435 — CAV concept lineage (feeds the NRC-2008 CSM-2/FALCON tier
  context; Form B/C future-vehicle framing).

#### AMaRV reference configuration: Allen 1997 (read from primary)

Allen (NASA Ames), "Composite Heat Shields Revisited," AIAA 97-0471 (repo
`data/`) — initially parked, then found to treat **AMaRV extensively** (user
catch): it uses the AMaRV geometry as the maneuverable second stage of a Mars
composite heat shield, citing information "declassified for supporting DC-X
and other unclassified projects" (source appears to be his McDonnell Douglas
personal-communication reference; superscript ambiguous in the scan).  The
**open-literature AMaRV configuration**, firsthand:

- **Biconic-cut body**: leading-cone half-angle **10.4°**, trailing cone
  **6°**; **split windward flap + two side flaps** for yaw control.
- **Mass 470 kg; ballistic coefficient β = 13,485 kg/m²** — note the
  Regan-1993 textbook evader (β ≈ 1.1×10⁴) sits right at this class ✓.
- **Nominal angle of attack 10.0°, "the middle of AMaRV's operating range"**;
  "designed for rapid and accurate reentry … while performing high-G
  maneuvers"; "successfully tested weapon system prototype."
- Allen's Traj code carries a coded AMaRV aero model; in his Mars
  application the AMaRV second stage sees peak convective **153 W/cm²
  (1.53 MW/m²)**, stagnation T ≤ 2500 K, with RCC judged tolerant without
  significant recession — and he notes the *terrestrial* AMaRV design
  environment was more severe.
- **AMaRV TPS = reinforced carbon-carbon** — stated directly: Allen selects
  RCC for the Mars second stage *because* "carbon-carbon TPS was used" on the
  terrestrial AMaRV (SLA-561V would recede at 153 W/cm²; RCC tolerates it).
  So the AMaRV airframe TPS is **carbon-carbon** — which ties Form C straight
  back to the Form A C/C ablator anchors (Perini/Scala theory, Nestler
  severe-regime, Reentry-F bracket; `heating.py` `carbon_carbon`).
- **AMaRV second-stage nose radius = 2.34 cm** (Allen's parameter table; a
  sharp maneuvering-biconic tip) — the last geometry field for a Form C
  reference reentry object.

Together with Auclair's flight profile (dive → 30 kft near-horizontal TFS
segment → terminal dive) and the MDAC program facts, the AMaRV anchor now has
geometry, mass, β, trim AoA, control layout, and profile — the remaining open
number is the maneuver Mach/g itself.

#### MaRV loads & RV-TPS genealogy: Lin 2003 (read from primary)

Lin, "Development of U.S. Air Force ICBM Weapon Systems," *J. Spacecraft &
Rockets* 40(4) 2003, DOI 10.2514/2.3990 (Northrop Grumman RV engineer; repo
`data/`) — a survey with several citable Form C / Form A data points:

- **MaRV lateral-acceleration datum**: a piloted interceptor pulls 6–8 g;
  "a MaRV … must pull approximately 10–20 times as much gravitational-
  acceleration to evade a typical ABM interceptor such as the U.S. SPRINT."
  (Wording ambiguous — reads as either ~10–20 g absolute or 10–20× the
  6–8 g; flagged.)  Brackets the maneuver-load range alongside Regan's 4 g
  (gentle) and 100 g (extreme evasion) figures.
- **MBRV refinement** (vs the GE brochure's "three, over the Pacific,
  mid-1960s"): "**Four MBRVs** developed and launched by **Atlas** boosters
  from **Vandenberg** in the late 1960s," with **cruciform trailing flaps at
  the base** for high-g attitude control; program started 1964.  (Slight count/
  site discrepancy with the GE brochure — noted, not resolved.)
- **β convention corroborated**: Lin defines β = **W/(C_D·A)**, the weight/
  Pascal form — same as Regan, confirming the RV-literature convention split
  documented in `HEATING_MODEL_CROSSCHECK.md` (the repo uses the ÷g metric form).
- **RV-TPS genealogy** (Form A context): Teflon (low-β); **silica phenolic**
  on MK5 (MM I) and MK11 (MM II), "a melting and pyrolysis ablator, adequate
  for low-to-medium β" — corroborates the `silica_phenolic` catalog entry;
  asbestos/quartz phenolic; first recovered ablative RV Thor/Able 9 Apr 1959.

#### Pershing II terminal-maneuver RV: Lund 1984 (Martin Marietta / AIAA, read from primary)

Lund, "Evolution of the Pershing II Missile System," Martin Marietta (Copyright
1984 Martin Marietta, **released to AIAA to publish** — a public technical paper,
hence archived to repo `data/lund-1984-evolution-of-pershing-ii-missile-system-martin-marietta-aiaa.pdf`).
Pershing II is the operational-MaRV candidate named at the top of Form C; this
is the primary account of its reentry vehicle and terminal maneuver:

- **First terminally-guided MaRV fielded**: Pershing II was "the first ballistic
  missile deployed with a terminally guided maneuvering reentry vehicle"
  (Martin Marietta prime; deployed Dec 1983).  Program payoff: >doubled the
  Pershing Ia's ~740 km range and cut delivery error by an order of magnitude —
  the accuracy step that *required* the terminal maneuver + map-match.
- **RV structure (three sections)**: radar section + warhead section + guidance-
  and-control / adapter section.  The **radar antenna sits behind an ablative
  radome** — the one firsthand Pershing II TPS datum (a forebody ablator over
  the RADAG antenna; the report is qualitative, no thickness/temperature
  margin).  This partly fills the Yengst "Pershing II skin material" open want:
  the guided nose uses an **ablative** radome, not a passive heat sink.
- **Terminal maneuver = velocity-control pullup/pulldown under inertial
  guidance**: after ballistic reentry the RV executes a "velocity control
  maneuver (pullup/pulldown)" to bleed to the proper terminal velocity, then
  **RADAG** (radar area guidance) runs its radar map-match against a stored
  reference, and the RV is steered to the target by a **vane (aerodynamic)
  control system**; an exoatmospheric **reaction control system** handles
  attitude above the sensible atmosphere.  This corroborates the Yengst
  Pershing II datum (~Mach 8 reentry → sub-50-kft ~25-g pullout during the
  RADAG update) and matches the Auclair Accuracy-AMaRV profile shape
  (dive → near-horizontal map-match segment → terminal dive) — the same
  velocity-bleed-then-map-match kinematics in an operational vehicle.

This closes the last open Form C *want* (Pershing II TPS + terminal-maneuver
kinematics) at the screening tier: the vehicle is a velocity-control pullup
MaRV (~25 g, Mach-8-class reentry) with an **ablative** guided forebody.
Quantitative windward temperature / TPS thickness remain unpublished in the
open sources on hand (Yengst + Lund are program/test histories, not thermal
data reports).

*Modern corroboration — the terminal-velocity end of the profile (read from
primary; archived to repo
`data/wang-tang-zhang-2019-short-range-reentry-guidance-pershing-ii-hgrv-ieee-access-7.pdf`,
merged from a 4-part chat upload — IEEE Access is open access):* **Wang,
Tang & Zhang (NPU
Xi'an), "Short-Range Reentry Guidance With Impact Angle and Impact Velocity
Constraints for Hypersonic Gliding Reentry Vehicle," IEEE Access 7 (2019)
47437, DOI 10.1109/ACCESS.2019.2909589** (open access) — a Chinese guidance
study that **"uses the Pershing II HGRV as the research object."**  Its
mission profile is the same three-phase shape as Lund/Auclair (initial-
descent pull-up → gliding velocity-bleed under overload/field-of-view
constraints → terminal forcing-down), and it publishes the *design*
terminal constraints for the Pershing II-class problem: **impact angle
−90° ≤ θ ≤ −70° (near-vertical dive), impact velocity 550–650 m/s
(≈ Mach 1.7–2), miss distance ≤ 6 m, guidance completed within several
hundred km**.  It also states the published engineering *rationale* for the
velocity-control maneuver Lund described: excessive terminal velocity both
raises the required overload and generates an aeroheating **plasma that
blocks seeker transmission and refractively distorts the IR/radome path**,
up to destroying the seeker — i.e. why the RV must "slow to proper terminal
velocity" before the RADAG map-match.  Status: simulation-study design
values for a Pershing II-modeled vehicle (modern secondary), NOT flight
data — corroborates and bounds the terminal-velocity end of the maneuver;
the ~25-g pullout datum remains Yengst/Lund.

*Full-paper tables (complete page set supplied in-chat):* the simulation
profile locks onto the flight-history datums at two independent points —
**Table 1's process constraint is acceleration ±25 g** (the same figure as
Yengst's ~25-g pullout, now as the modern engineering overload bound for the
Pershing II-modeled class; encoded as a `textbook`-kind record in
`MANEUVER_ANCHORS`), and **the gliding/velocity-bleed phase runs at ~16 km
altitude** (handover 16 km / 50 km-to-go; Fig. 13) — matching Yengst's
"below ~50,000 ft" (15.2 km) energy-retention band.  Full profile: reentry
range 300 km, h₀ 55 km, V₀ 3,400 m/s, γ₀ −22° (Table 4); initial-descent
pull-up peaking ~11 g normal (Fig. 13d); quasi-level bleed glide 16–18 km
with ±5-g bank maneuvers; terminal attack from 20 km / 1,200 m/s / 50 km
(Table 2) ending at 600 m/s / −85° / ~1 m miss (Tables 3/5); robustness 300
Monte-Carlo runs at ±10% density, ±15% C_L/C_D → 296 hits.  Their trajectory
taxonomy: short-range HGRV flies a **C-shaped** lateral trajectory (large
seeker FOV, few bank reversals) vs the **S-shaped** trajectory of long-range
Shuttle-class gliders.  Notable context: the stated target "is to hit the
ship at sea" (moving-target anti-ship application); their Pershing II source
is ref [16] = Zhao & Chen 1993, now read from primary — see next entry.

*The 1993 Chinese-literature primary (read from primary, archived 2026-07-20;
user-supplied):* **Zhao Hanyuan (赵汉元) & Chen Kejun (陈克俊), "再入机动弹头的速度控制"
["Velocity Control of Maneuvering Reentry Vehicle(s)"], *Journal of National
University of Defense Technology* 15(2) (1993) 11–17** (Dept. of Automatic
Control, NUDT, Changsha; archived as
`data/zhao-chen-1993-velocity-control-of-maneuvering-reentry-vehicle-j-nudt-15-2.pdf`).
**CITATION CORRECTION**: Wang 2019's IEEE-style "Z. Hanyuan" is **Zhao**
Hanyuan, not Zhang — the same Zhao Hanyuan as Wang's ref [25] textbook
(*Spacecraft Reentry Dynamics Guidance*, NUDT Press 1997) and the author of
the 1980/1985 reentry-maneuver-trajectory design papers this one builds on
(J. NUDT 1980(4):73–105; Acta Astronautica Sinica 1985(1):1–10).  Mined
findings:

- **Pershing II + RADAG named explicitly as the exemplar**: accuracy-class
  maneuvering warheads require terminal guidance — "for example the
  Pershing II missile uses a radar area-correlation terminal guidance
  system" — and must decelerate first or the warhead "will be enveloped in
  plasma generated by severe aerodynamic heating and the signal cannot be
  transmitted"; "reentry deceleration is one of the prerequisites for
  implementing terminal guidance of a maneuvering reentry warhead."  This
  1993 paper is thus the primary for the plasma/velocity-control rationale
  Wang 2019 restates, and an independent Chinese-literature corroboration of
  Lund 1984's "slow the RV to proper terminal velocity" purpose statement.
- **Two-class MaRV taxonomy** (opening paragraph, in the RV community's own
  1993 words): (1) **penetration-focused** — the trajectory is designed to
  evade the defense system, terminal velocity is maximized, and **landing
  angle is not constrained**; (2) **accuracy-focused** — requires a terminal
  guidance system (→ the RADAG/deceleration chain above).  This is the same
  penetration-vs-accuracy split the ledger carries from Auclair's AMaRV
  Evader-vs-Accuracy classes and Lin/Yengst — here stated as the organizing
  dichotomy by a Chinese RV-guidance group, and the reason the two Form C
  regimes (max-velocity ballistic-ish evader vs decelerated map-matching
  accuracy MaRV) are genuinely different design points.
- **Method** (context, not a screening datum): optimal diveline guidance law
  for a fixed ground target derived by optimization (separate dive-plane and
  turn-plane motion), an *ideal velocity curve* design for the terminal
  velocity, and a vertical **additional-angle-of-attack** deceleration scheme
  (extra induced drag) — the exact analytic ancestor of Wang 2019's
  "additional angle of attack" deceleration control, confirming that method
  lineage.  Simulation only (no flight data; anti-fixed-ground-target here,
  vs Wang's anti-ship moving-target extension).

*Pershing II Drive-folder triage (2026-07-20, folder `11Iqc63JuDov…`):* only
**Lund 1984** was archivable and is mined above.  Also present:
`ADA121622` — **now MINED from primary (user chat-upload, 2 parts; archived
to repo `data/knight-1982-pershing-ii-simulation-studies-rd-cr-82-27-…`)**:
Knight, Lynch, Pyles, Seitz & Thornton (Georgia Tech EES for the US Army
Missile Command), "Pershing II Simulation Studies," RD-CR-82-27, July 1982,
Distribution Unlimited.  **NEGATIVE RESULT for the open quantitative want**
(published terminal-maneuver Mach/g): the aerodynamic flight validation "could
not be completed because Pershing II flight test data were not available in
time," and the nine Tactical-Ballistic-Missile trajectory profiles were
delivered to the sponsor **on tape**, not printed in the report — so the
report documents simulation *infrastructure*, not RV maneuver data (the
Appendix F aero tables are worked interpolation examples with made-up
numbers).  Yengst 2010 (+ Lund 1984 corroboration) therefore remains the
citation of record for the ~25-g / sub-50-kft / Mach-8 pullout datum.
What the report *does* give:
- The Army's **U70 simulation program** covers Pershing II "through boost and
  re-entry including the maneuvering re-entry vehicle" (separate BOUT/RVOUT
  boost/RV output stages) — confirmation that MICOM's own tool chain modeled
  the terminal maneuver, plus the TRW advanced simulator with
  multidimensional aero-pressure/temperature/coefficient tables.
- **A boost-phase structural/breakup criterion used in the Army's WSMR
  range-safety analysis: missile breakup assumed when total AoA exceeds 15°
  or normal acceleration exceeds 5 g during boost** (nozzle-deflection
  failure cases at t = 30 s [max-q] and 49 s [near first-stage burnout]) — a
  citable in-boost structural-limit class datum (Thrusty currently applies
  no boost structural gate; parked as a possible future anchor).
`Maneuvering Warheads.pdf` = a
1983 NYT article (**copyrighted, ProQuest — facts-only, not archived**; no new
technical datum beyond the above); FM 6-11 / FM 6-11 FD(84) Army field manuals
and an operator's technical manual (doctrine, not TPS/kinematics — no datum);
four Pershing II Space Force photographs (imagery, no datum).

#### Form C transition/AoA context: Francis 2024 (parked, archived)

Francis, Dylewicz, Klothakis, Theofilis, Jewell, "Instability Measurements on
a Cone-Slice-Flap in Mach-6 Quiet Flow," AIAA 2024-0500 (repo `data/`) —
modern boundary-layer-transition experiment on a **cone-slice-flap** (the
MaRV control-flap geometry) in the Boeing/AFOSR Mach-6 quiet tunnel; cites
AMaRV and Shuttle as the flight-test motivation for control-surface
heating/transition prediction.  The empirical face of the same control-surface
transition uncertainty Thompson 1989 flags for the Form C windward/AoA probe —
context, not a screening-tier anchor number.
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

#### Coated hot-structure dwell floors (RCC / C-SiC / C-C-HS) — campaign OPEN (2026-07-20)

**Goal**: replace the generic 120-s soak-dwell surrogate (the one uncited
number in the verdict machinery; `heating.py` labels it "empirical damage
surrogate, not an oxidation-kinetics closure") with **cited, per-material
demonstrated dwell floors** for the coated hot structures — mirroring the
UHTC campaign (U1–U4).  The coverage machinery is already property-gated
(`oxidation_dwell_s` present → envelope verdict), so each material lights up
as its data lands.  Status by material:

- **`cc_hot_structure` (HTV-2-class C/C) — sources in hand, mining pending**:
  the DARPA Flight-2 ERB primary (skin "peeled from the aerostructure" after
  ~3 min of stable Mach-20 flight; FOIA 14-F-0122, archived) + NTRS
  20090004576 (oxidation of C/C through coating cracks — the failure
  mechanism).  A *flight* degradation-time datum.
- **`rcc` (Shuttle) — REUSE-LIFE DATUM NOW IN HAND (Jenkins 2013, below)**;
  also NTRS 19940030739 ("Analysis of the Shuttle Orbiter RCC Oxidation
  Protection System," coating mass-loss / reuse limits — still to mine for
  the per-cycle mass-loss number).
  *Checked-and-empty (2026-07-20, not archived):* Smith, Soares et al.
  (Boeing/NASA JSC), "Space Shuttle TPS Repair Flight Experiment Induced
  Contamination Impacts" (AIAA/JSC, STS-114/121 DTO 848) — an STS-114 RCC
  leading-edge document, but entirely about **NOAX repair-material
  outgassing/contamination** (49 °C dispense, −34 °C deposit surfaces, ASTM
  E 1559, EVA visor fogging); no RCC entry environment, no arc-jet exposure,
  no time-at-temperature.  Right material/mission, wrong subject — carries no
  dwell datum.  Do not re-mine.

*RCC reuse-life anchor (read from primary, archived
`data/jenkins-2013-protecting-the-body-orbiter-tps-nasa-history-ch5.pdf`):*
Jenkins, D. R., "Protecting the Body: The Orbiter's Thermal Protection
System," Ch. 5 of the NASA Shuttle history (AIAA book DOI
10.2514/5.9781624102172.0111.0136).  The authoritative Shuttle-TPS history,
and it hands over the RCC reuse-life numbers **and** the epistemic caveat:
- **RCC operating range −250 °F to 3000 °F** (3000 °F = **1922 K** — matches
  the catalog `rcc` `peak_K` 1922 K exactly).  Pyrolyzed-carbon/carbon with a
  **silicon-carbide conversion coating** (diffusion reaction at 3200 °F =
  2033 K); "**not an insulator — the backface was essentially as hot as the
  frontface**" (so the RCC limit really is a bulk-material limit, exactly how
  the `hot_structure` group treats it, not a through-thickness soak).
- **Reuse-life spec (Lockheed TPS): "100 normal operational entries at
  2500 °F (= 1644 K), or a single-contingency entry at 3000 °F (= 1922 K)."**
  → the RCC anchor: **kind=design/flight, reuse floor 100 entries at 1644 K,
  single-entry cap 1922 K, coating = SiC conversion, outcome survived (flew
  the full 135-mission program on the wing LE / nose cap)**.  This is a
  *cycle-life* floor (the right RCC metric — reusability, not single-entry
  dwell), sitting **below** the current `continuous_K` 1811 K — i.e. the
  routine-reuse temperature is lower than the material's one-shot limit.
- **The honesty caveat, in the program's own words**: 1970s coatings
  "appeared to permit an upper-limit temperature of about 2500 °F for 100
  cycles, but **actual real-world data was lacking**."  The Shuttle's own
  coating-life numbers were *design estimates*, not measured oxidation-life —
  precisely why the campaign wants an at-limit arc-jet time-to-failure and
  why "floor, not fence" is the honest verdict semantics.
- Corroborates the **silica-tile** limits too: LI-900/LI-2200 tiles ran
  650–2300 °F acreage (**2300 °F = 1533 K** = catalog `silica_tile`
  `continuous_K`), cyclable to 2500 °F, backface held to 350 °F (450 K =
  the aluminum airframe limit) — a clean independent check on three catalog
  numbers at once.
- **`c_sic` — first flight dwell record NOW IN HAND (below); at/near-limit
  arc-jet oxidation-life still the open get.**

*C/SiC model-half (facts-only from abstract; full text not yet supplied):*
Huang, Yang & Huang, "Oxidation and Sublimation Ablation of C/SiC Ceramics in
Hypersonic Environments," *J. Spacecraft & Rockets*, DOI 10.2514/1.A36501
(online 18 Aug 2025) — a **theoretical/numerical** wide-temperature ablation
model for C/SiC covering **passive oxidation → active oxidation →
sublimation** with a dimensionless ablation-rate system; reports pressure has
weak effect on surface temperature but a **significant positive correlation
with mass-ablation rate** — the C/SiC statement of the same SiC
pressure-sensitivity caveat the UHTC envelope carries (§11.6).  A regime-map
source (the Jacobson-and-Harder role), **not** a demonstrated dwell (no
specimen was heated); its bibliography is the hunting ground for the
experimental arc-jet papers.

*SHEFEX I flight heritage + a conservatism datum (read from primary, archived
`data/barth-eggers-2006-shefex-…-dlr-stab.pdf`):* Barth & Eggers (DLR),
"SHEFEX — A First Aerodynamic Post-Flight Analysis," STAB 2006.  SHEFEX I
(S30 + Improved Orion, apogee 211 km): **45 s of experimental reentry between
90 and 14 km**, Mach ≈5.6 held from 100→50 km rising to its maximum lower
down; facetted ceramic-composite sharp-edged TPS.  **Deliberately NOT flown
at the material thermal boundary** — the stated objective was "to prove in
flight that the temperature peaks at the edges of the ceramic-composite
panels are **lower than those predicted based on a radiation equilibrium
hypothesis**" — i.e. the program was *designed around* rad-eq over-prediction
at sharp features.  Benign-regime flight heritage for facetted CMC (45-s
survival), not a dwell-at-limit floor.  Bonus data: boundary layer assessed
laminar >40 km / turbulent <30 km with the 33.8 km station reading turbulent
(a Mach-6 sharp-body flight transition band), and a flight-data caveat (AoA/
sideslip extraction "much more demanding than expected").

*IXV — the first C/SiC-class flight dwell record (read from primary, archived
`data/buffenoir-…-eucass-2017-330.pdf`):* Buffenoir, Pichon & Barreteau
(Ariane Group), "IXV Thermal Protection System Post-Flight Preliminary
Analysis," EUCASS 2017-330.  The windward + nose **SepcarbInox C/SiC-class
CMC** assemblies (non-ablative OML), flown Feb 2015 and recovered:
- **Design spec: max heat flux 650 kW/m² (0.65 MW/m²), estimated max TPS
  outer-skin temperature 1650 °C (1923 K), reentry duration 20 minutes
  (~1200 s), 3 g.**
- **Post-flight: measured temperatures 200–600 K BELOW predictions** (hottest
  windward sensor WT80 on panel 27: −600 K vs calculation; hottest nose
  sensor NT2: −400 K); heating slopes/durations matched, absolute levels did
  not; temperature gradients smoother than expected; **no visible damage**
  on the panels at first inspection; "re-entry conditions were less severe
  than predicted."
- **Draft anchor shape**: C/SiC-class CMC, kind=flight, dwell ≈ 1200 s
  (full reentry), design-spec environment 0.65 MW/m² / ≤1923 K, measured
  well below spec, outcome = survived (no visible damage; *preliminary*
  analysis — margins consumed not quantified).  A **moderate-temperature
  1200-s flight floor**, not an at-limit datum: the at/near-limit
  time-to-failure (arc-jet) remains the open want before `c_sic` gets a
  cited `oxidation_dwell_s`.

**Cross-cutting conservatism corroboration**: two independent European flight
programs — DLR SHEFEX (by design objective) and ESA/Ariane IXV (by measured
−200…−600 K) — both report **flight temperatures below
radiation-equilibrium-class predictions**.  Direct flight-side corroboration
that Thrusty's screening chain (cold-wall Sutton-Graves + radiative
equilibrium) errs hot, i.e. conservative — consistent with the Tracy &
Wright hot-wall cross-check (+6.5%) logged in HEATING_MODEL_CROSSCHECK §6.

#### Windward/AoA heating probe — BUILT (2026-07-20)

**Status: implemented** (`heating.windward_flank_flux`, wired into the Form C
report).  The screening model, from the source pack below:

```
q̇_flank0 = BODY_FLUX_FRACTION · q̇_stag(ρ,V,R_body)     # α=0 acreage flux (cited, 0.13)
A(α)      = sin(δ+α)/sin(δ)                             # windward amplification
q̇_wind    = q̇_flank0 · A(α)   →   T_eq,w = (q̇_wind/σε)^¼
```

evaluated over the **glide sub-arc** (the low-AoA terminal dive is masked out —
that segment keeps the nose-stagnation block), reported as a **T_eq band across
α = 5–20°** (the Thompson error-anchor ends) with the trimmed operating AoA
marked inside when a non-sep body glider supplies one (from the static-margin
gate `alpha_glide_deg`; a separating RV → band-only).  **`A(α)` reuses the
already-cited `BODY_FLUX_FRACTION = 0.13`** (Lu/Shi & Zhang 2024, `heating.py`)
for the α=0 flank and the modified-Newtonian pressure ratio through the
reference-enthalpy `q̇ ∝ √p_e` scaling for the amplification — **`CP_MAX`
cancels**, so the factor is purely geometric.  **Inference labels carried in
code:** the closed-form sin-ratio reduction is an inference (the *method family*
Van Driest + Eckert-Tewfik and the windward-vs-leeward *ordering* are the cited
part — AGARD-R-754, Tracy M=7.95 cone); `ρ_e∝p_e` holds edge temperature fixed
so the ratio mildly **over-predicts** (conservative); the δ≥5° floor is a
numerical guard.  Turbulent flank (~3–5×) and control-fin gap interference
(Alviani 10–80×) are **flags, not computed** — screening can't place the
transition or reattachment line; Murray & Russell 2002 (MASCC) is the named
computed-value upgrade.  Verdict role is gated by
`heating.WINDWARD_DRIVES_VERDICT` (default **off** → context overlay; on →
downgrade survive→degraded past the body soak limit at the gentlest α, or flag
needs-analysis past the peak limit — never a hard fail).  Pinned by
`test_windward_flank.py` (α=0 reduction, Tracy 2.46/3.81 magnitudes, windward>
leeward ordering, stagnation-approach guard, band/stamp).  *Still genuinely
open:* a **flight** windward/leeward split during a real maneuver — all four
sources below are wind-tunnel / CFD / ground-code.

##### Source pack (read from primary, archived 2026-07-20)

Four references (user-supplied) that between them **provisioned this tier**.
Each is archived to repo `data/`.

- **Kapp, Mathauer & Rieger, "Aerodynamic Heating of Missiles," AGARD-R-754
  paper 10 (Special Course on Missile Aerodynamics, 1988; DTIC ADA199172,
  Distribution Unlimited)** — the method template for the probe: engineering
  aeroheating via **Van Driest** (Stanton-number, geometry-dependent) for
  general bodies and **Wing** for ogival noses, with the **Eckert–Tewfik
  reference-enthalpy** adaptation of Lee's momentum-integral for the laminar
  distribution.  Validated windward-vs-leeward: an **ogive-cylinder swept
  through α = −10°…+10°** shows "the increase on the windward side is much
  higher than on the leeward side" (weak on the cylinder, strong on the
  ogive), and a **pointed cone at M = 7.95, α = 12°/24°** matches Tracy's
  measured *circumferential* (windward→leeward) heat-transfer distribution —
  the **same Tracy test case Thompson 1989 uses**, so the two anchors align.
  This is exactly the reference-enthalpy windward estimator a Form C AoA
  probe would implement.
- **Richards, "Kinetic Heating of High Speed Missiles," AGARD-LS-98 paper 9
  (Missile Aerodynamics, 1979; DTIC ADA068808, public release)** — the
  companion review: kinetic heating for *tactical* missiles (the class with
  the sparsest open literature), covering attached vs separated flows,
  high-incidence effects, transient/arbitrary-wall vs steady-isothermal, and
  the shock-surface-interaction localized-heating problem — the framing and
  caveat set for the probe.
- **Alviani, Fano, Poggie & Blaisdell, "Aerodynamic Heating in the Gap
  Between a Missile Body and a Control Fin," J. Spacecraft & Rockets (2022),
  DOI 10.2514/1.A35183** — the **fin-LE / control-surface interference
  anchor** the terminal-dive block explicitly punts to "later tier": RANS,
  fully-turbulent **Mach 6**, validated against **AEDC VKF Tunnel B 1979**
  wind-tunnel data.  Quantifies the shock-interaction augmentation in the
  body-fin gap: local heat transfer **10–80× the fin-off baseline** (surface
  pressure up to ~20×) at the fin-root reattachment — i.e. the control-flap
  leading edge, not the nose, is the Form C thermal driver, and by a large
  factor.  The cited flag "AoA probe is a later tier" now has a numbered
  severity to attach when built.
- **Murray & Russell (ITT Aerotherm / US Army AMCOM), "Coupled
  Aeroheating/Ablation Analysis for Missile Configurations," JSR 39(4) 2002**
  — the **maneuvering + windward + ablation coupled method**: the Maneuvering
  Aerotherm Shape Change Code (MASCC) computes surface heat flux on
  **windward streamlines as a function of velocity, altitude and angle of
  attack** through the trajectory (axisymmetric-analogy streamline tracing;
  Dahm–Love / other windward-pressure options), feeding the CMA charring-
  ablator transient — a missile-specific, AoA-aware counterpart to Duffa
  that couples **both Form A (recession) and Form C (windward aeroheating)**.
  The reference design for a future coupled tier above the current screening
  split.

*Together these retire the "provisioning" half of the AoA-probe want*: the
method (Van Driest/Eckert reference-enthalpy windward estimate; MASCC-class
coupling for a later tier), the validation cases (Tracy cone circumferential;
AEDC fin-gap Mach 6), and the severity (fin-LE 10–80× baseline) are now all in
hand and archived.  Still genuinely open: a **flight** windward/leeward split
measured during an actual maneuver (all four here are wind-tunnel/CFD or
ground-code), and the probe itself is **not yet built** — this is the source
pack for that build, not the build.

*Drive-folder triage (2026-07-19, folder `1oUqtoFx02…`):*
- `regan1993.pdf` — the Regan & Anandakrishnan book: **the priority get**,
  blocked by the 10 MB connector cap (see above).
- `AD0376942.pdf` = "Aerodynamics of Conical Bodies" — **MINED from primary
  (2026-07-20, user chat-upload in three parts; archived to repo
  `data/eastman-1966-…-d2-36139-1-…-part-{a,b,c}.pdf`).**  See the dedicated
  cone-aero databook subsection below.
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
- `10.1016@0376-04217990001-0.pdf` = *Prog. Aerosp. Sci.* 1979 first article —
  **identified and MINED from primary (2026-07-20, user chat-upload of a clean
  1.9 MB copy; archived to repo
  `data/townend-1979-research-and-design-for-lifting-reentry-pas-18.pdf`):
  Townend, L. H. (UK MoD Procurement Executive), "Research and Design for
  Lifting Reentry," *Prog. Aerospace Sci.* 18 (1979) 1–80** — the UK
  caret-wing / waverider "flow containment" lifting-reentry survey.  Form B
  design-space context, with several citable datums:
  - **"The supersonic and hypersonic L/D need not exceed 2±1 (rather than
    5±1, say)" for lifting-reentry gliders** (vs hypersonic cruise vehicles;
    §3) — a 1979 design-community statement of exactly the L/D class the
    repo's Form B objects ship with (C-HGB 2.0, HTV-2 2.6 flight-fit,
    NRC-2008's assumed 2.2) and the reason reentry gliders are "substantially
    bulkier" than cruise vehicles.
  - **Flow-containment lift data (AASU gun tunnel, Southampton)**: an
    anhedral "high-wing" modification of the NASA-MSC/040A Shuttle-class
    orbiter produced **~20% more lift than the low-wing version at α =
    40–50°, M = 9.7, Re = 4×10⁵** (East 1976; corroborating Davies &
    Townsend 1972 at M 8.4); caret undersurface ω ≈ 5° gave ~10% higher C_L
    at α = 55°, M = 12.2.  C_L ≈ 0.7–0.9 at α = 40–60°, C_L/C_D ≈ 0.4–1.2 —
    the high-α Shuttle-class reentry regime (25–70°), distinct from the
    low-α MaRV/HGV regime.
  - **The heat-transfer logic of raising C_L** (his Refs. 8/11 argument): at
    given wing loading, higher C_L ⇒ deceleration at higher altitude ⇒ lower
    ambient density ⇒ reduced stagnation heating; and beneath the stronger
    contained shock, lower local flow velocity ⇒ reduced undersurface
    heating per unit wetted area.  The same corridor physics Thrusty's
    equilibrium-glide altitude embodies (higher C_L/(W/S) → higher glide
    altitude → lower flux, longer duration — the NRC stopwatch trade).
  - Stability: the high-wing orbiter's aerodynamic centre sat ~10% of length
    further aft than the low-wing's, and for both, **a.c. position was
    almost invariant with attitude for α = 40–60°**; experiments confirmed
    the concave-undersurface C_L gain across 6 < M < 22, 25° < α < 70°.
  Copyright: Pergamon/Elsevier journal article — archived per the repo's
  technical-journal precedent (Lin 2003, Francis 2024).
- Subfolders X-51 / HTV / General Hypersonics / Defenses — **all swept
  2026-07-20** (see the Form B sweep subsections).  The two big scans are
  now also mined via chat upload: AD0376942 (Boeing cone-aero databook, 3
  parts) and the PAS-1979 article (Townend lifting-reentry survey).  The
  only remaining cap-blocked item is the Heating-folder Martin-class
  "atmospheric reentry" book (13 MB).

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
transverse-g crossing altitude ≈45 km.

**RUN (2026-07-20) — PASS.**  `regan1984_marv_check.py` (planar point-mass
EOM with Thrusty's own `atmosphere()` and the constant_LD glide force law
`a_lift = min(L/D·q̄/β, n_max·g)`): the 4-g cap first binds at **46.4 km**
(V = 5,882 m/s) — within 1.4 km of Regan's ≈45 km.  The case also exercises
the β-convention rule (HEATING_MODEL_CROSSCHECK.md): Regan's β = 10⁴ is his
**Pa weight form** and must be ÷ g → 1,019.7 kg/m² in repo units; read
wrongly as kg/m² the bind altitude comes out **31.2 km** — a 14-km miss.
Pinned by `test_form_c_anchors.py::test_regan_1984_worked_case` (±5 km
band).  This verifies the atmosphere + force-law + convention chain against
a published table; the 3-D integrator embeds the same law
(`trajectory.py` damped_glide constant_LD branch).  Supporting load-context quotes (same
book): maneuvering loads normal to the velocity vector "can be more than two
orders of magnitude greater than the gravitational force" (Ch. 5 intro), and
a Ch. 13 worked transient of 25.7 g at α = 60° with endo-maneuvering loads
"an order of magnitude larger" causing little bending (distributed load).

#### Heating-chain verification: Finke, IDA P-2395 (read from primary, archived 2026-07-22)

Reinald G. Finke, *Calculation of Reentry-Vehicle Temperature History*, IDA
Paper P-2395, September 1990 (SDIO/ENA contract MDA 903 89 C 0003; DTIC
**ADA231552**, approved for public release; uploaded to the project archive).
Built to support POET interceptor-seeker detection analysis, it is an
**independent 1990 implementation of Thrusty's exact heating architecture**:
trajectory (RANGE) → stagnation flux → "inertialess" radiative-equilibrium
T_eq → per-location heating ratios → 1-D transient material response (TRIDE),
on a hypothetical high-β ICBM RV (R_n 0.077 m, β ≈ 1500 lb/ft² ≈ 7,320 kg/m²,
V_entry ≈ 7.06 km/s, γ −24.8°, range 10,020 km; glass-fiber-phenolic shield,
0.5 cm).  What it contributes:

- **T_eq chain verification (PINNED, two tiers).**  His laminar correlation
  (q ∝ √ρ·V^3.15, stated "in numerical agreement with Detra, Kemp, and
  Riddell as validated … in Perini, 1975") is printed in closed form, so the
  load-bearing check is **exact, digitization-free**: ratioed at identical
  (ρ, V), S-G/DKR flux = 1.01 at 3 km/s → **0.89 at 7 km/s** (−11% flux =
  −3% T at ICBM speed; both ∝ √ρ, so the ratio is velocity-only).  S-G sits
  mildly on the LOW (optimistic) side of the DKR family at high speed — a
  documented family spread inside the screening tier's stated uncertainty,
  with the sign now on the record.  The Fig.-2 curve comparison is now
  **pixel-traced** (tick-calibrated axes, nearest-run tracker, frame masking;
  `benchmarks/verification/digitize_finke_fig2.py` + QC overlay + the CSV),
  replacing an earlier eyeball read whose ±3–4 km x-axis uncertainty on a
  ~2%/km curve alone injected ±6–8% in T — which had made 60 km look like a
  9% outlier.  Digitized, the residual collapses to a **steady ~5% across
  37 / 60 / 80 km** (Thrusty 4,144 / 2,834 / 1,969 K vs trace 4,324 / 3,006 /
  2,076 K; ratios 0.958 / 0.943 / 0.949) — the SAME sign and size as the
  exact correlation ratio (S-G ~3% low on T at 7 km/s) plus the 1962-vs-modern
  atmosphere.  No 9% outlier survives the trace.  Both tiers pinned in
  `test_finke_check.py` (exact ratio band 0.86–0.92; traced-curve band ±8%).
- **Hemisphere heating distribution** (his Fig. 3, Kemp-Riddell 1959 theory +
  shock-tube data): q/q_s = 1.0 / 0.93 / 0.72 / 0.45 / 0.22 at s/R = 0° /
  20° / 40° / 60° / 80°, with the **conical surface held at the shoulder value
  0.22 × nose-stagnation** (constant-pressure-on-cone argument).  Frame
  conversion: 0.22 of the R_n = 0.077 m nose flux ≈ 0.5 × a *body-radius-
  referenced* stagnation flux for his geometry — vs our Lu/Shi & Zhang
  cone-TAIL 0.13.  Not a conflict: near-shoulder (his, held constant = an
  aft-conservative bound) vs far-tail (ours).  The pair brackets laminar
  acreage heating ~0.13–0.5 × body-referenced stagnation, corroborating the
  existing "flank can run above the 0.13 tail value" warning with a cited
  forward-cone number.
- **Rarefied/transition bridging** (logged as a model caveat, METHODS §13.3):
  free-molecule heating (Gilbert & Scala) crosses laminar-continuum at
  ρ_c/ρ₀ = (2.023×10⁻⁸/R_n)·V^0.3 — ≈ 92 km for his geometry; he bridges with
  q̄ = (q_FM⁻ⁿ + q_L⁻ⁿ)^(−1/n), n = 2 (vs Matting 1971).  Thrusty applies
  Sutton-Graves (continuum) everywhere, which **over-predicts above the
  crossover** (free-molecule q ∝ ρ is the lesser there) — conservative in
  sign and negligible in integrated load, but onset-altitude timing reads
  slightly early/hot above ~90 km.
- **Screening-convention corroboration:** his "ablation temperature,
  arbitrarily taken as an even 2000 K" for the phenolic shield equals our
  phenolic-family ablation-onset `continuous_K` = 2000 K (pinned in the same
  test); and his emissivity sweep (0.25–1.0 changes surface T only ~100–300 K
  in the heating window) documents that conduction, not reradiation, disperses
  the early-entry heating — context for how weakly ε drives the screening
  verdicts.  (The paper's own purpose — low-ε coatings to cut IR
  detectability — is outside Thrusty's scope but explains the SDIO interest.)

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
| ablator demonstrated-load record (verdict driver) | graphite/C-C 3,870 MJ/m² (Reentry-F); PICA 276 MJ/m² (Stardust); carbon-phenolic 60 MJ/m² (Pioneer Venus LP) | The Form A ablator verdict compares flown load to these (like the UHTC dwell floor), NOT a computed δ (see METHODS §13.6). Graphite: Reentry-F Q ≈ 3.87 GJ/m² (pixel-traced, ±20%). PICA: Stardust Q 276 MJ/m² wired, recovered. **CP: Pioneer Venus Large Probe ~60 MJ/m²** (Cabrera & West 2026, DOI 10.2514/1.A36431, coupled reconstruction validated to ≤6% vs flight TCs; figure-integrated ±25%; short radiation-heavy CO₂ pulse → deliberately conservative as a load record; Hayabusa CP corroborates ~2–3× higher, pulse-duration-soft — see the CLOSED note below) | ✔ all three cited (CP labeled figure-integrated) |
| ablator `H_eff_MJ_kg` — role now = tripwire bound only | nominal CP 15 / PICA 35 / C/C 40; **optimistic bound** CP 20 / PICA 77 / C/C 175 | `H_eff` no longer sets a verdict via a δ point-estimate; it (a) brackets the reported δ *band* (nominal, conservative-low edge) and (b) at its most OPTIMISTIC cited value gates the **burn-through tripwire** (red only if the shield is consumed even there). Nominals: **CP 15** = measured char-removal-regime (Sutton TN D-5930: 14–20 MJ/kg at ≥2.4 atm; clean rows 68–195). **PICA 35 / bound 77**: Winter AIAA 2014-1151 arc-jet, implied Q\* 38–77 MJ/kg. **C/C 40 / bound 175**: Reentry-F flight bracket 70–175 + Scala/Perini theory; Nestler severe-regime floor | ✔ nominals + bounds cited (regime-labeled) |
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

### User-adjustable screening thresholds (`thresholds.py`) + deferred spreadsheets

A curated **~9-number envelope subset** of the thresholds above is now
user-editable at runtime (Analysis ▸ Screening Envelope…): the UHTC dwell
floor, the two MaRV g-ceilings, the three **ablator demonstrated-load records**
(graphite/C-C, PICA, carbon-phenolic), and the two model-conservatism knobs (acreage flux
fraction, windward AoA band).  (The `δ/R_n` accuracy-ladder steps + glider-tip
flag were retired from the dialog when the ablator verdict moved from a computed
recession to a load-vs-record comparison — METHODS §13.6 — and survive as the
cited δ ladder in the report's warning text.)  These are the numbers a **policy
modeler** is likeliest to move when a new open-source flight/test lands — the
*envelope*, not the material coupons.
The registry (`thresholds.REGISTRY`) carries each default's citation of record
(the same provenance as the audit table above); a user edit lives only in an
overlay (`benchmark_overrides.json`) and always restores, and any modified
number self-discloses in the report (headline asterisk + *Modified benchmarks*
block).  `test_thresholds.py` pins the registry defaults to the live constants.

**Deferred to a future spreadsheet project (explicit scope decision):** the
full **material catalog** (14 TPS materials × ~7 numeric fields) and the
**anchor datasets** (`UHTC_ANCHORS`, `MANEUVER_ANCHORS`, the Form A recession
anchors) remain code/data-edit surfaces, *not* runtime-editable.  Rationale: a
policy modeler integrates *events* (a new glide time, a new demonstrated g),
not new coupon data for one material; the catalog/anchor tables are a
heavier, spreadsheet-shaped import job better served by an XLSX round-trip
(mirroring the booster/RO XLSX templates) than by a threshold dialog.  When
that project happens, it slots beneath this same frozen-default + self-disclose
discipline.

**CLOSED (2026-07-22) — carbon-phenolic demonstrated load: Pioneer Venus Large
Probe, ~60 MJ/m².**  Was OPEN ("NRC gives durations, not loads").  Closed by
**Cabrera & West 2026** (read from primary, uploaded to the project archive):
Jannuel V. V. Cabrera & Thomas K. West IV, "Pioneer Venus Large Probe
Stagnation Point Entry Heating with Coupled Ablation," *J. Spacecraft &
Rockets* 63(2), Mar–Apr 2026, DOI 10.2514/1.A36431 (NASA Langley; presented as
AIAA 2024-3560).  A trajectory-based LAURA/HARA **coupled-ablation
reconstruction** of the Dec 9, 1978 entry (11.584 km/s, γ −31.829°,
β 190 kg/m², Rₙ 0.355 m, CP heatshield 1.0 cm stagnation / 0.75 cm flank),
validated against the flight stagnation thermocouple to a **24 K maximum
discrepancy (≤6%)** — the paper's own hypothesis for the fit is exactly the
coupled finite-rate chemistry.  Coupled peak fluxes: **radiative
2,027 W/cm² (20.3 MW/m²) at 18.6 s; convective 1,382 W/cm² (13.8 MW/m²) at
19.2 s**; peak wall temperature **4,032 K** (sublimation regime); ablation-
induced convective blockage 47–63% vs non-ablating; stagnation recession
"only 20% of the Galileo result."  **Integrated load: ~60 MJ/m²**
(trapezoid integration of the Fig. 7 coupled curves, ~33 rad + ~31 conv
MJ/m², labeled **figure-integrated ±~25%** — same discipline as the
Reentry-F pixel trace, coarser method).  Regime caveats carried in the
source string: **97.4% CO₂ atmosphere, radiation-heavy, ~4–6 s pulse** — this
anchor demonstrates CP's *flux/temperature* capability (34 MW/m² combined,
4,032 K wall on a 1 cm shield) far more than its long-dwell load capability,
so as a LOAD record it is deliberately conservative.  **Hayabusa (CP,
recovered; Suzuki JSR 2014, already in repo) corroborates a higher load** —
our half-sine reconstruction integrates ~200 MJ/m² at the nominal 60 s pulse
(~100 at the conservative 30 s end) — but its pulse duration is a labeled
estimate (3× window), so the PV number, with the tighter provenance chain,
sets the wired record and Hayabusa rides as corroboration.  (This also
corrects the earlier OPEN note: a CP load anchor *was* derivable from the
in-repo Suzuki data; what was missing was one with tight provenance.)
Bonus datum: the paper's 4,032 K coupled peak wall temperature is consistent
with (slightly above) our CP `peak_K` 3,900 — the catalog value reads as
mildly conservative against a validated flight reconstruction.
