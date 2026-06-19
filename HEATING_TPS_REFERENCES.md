# Heating / TPS-Survivability References

Curated bibliography behind the reentry-heating figure-of-merit work (peak-flux +
heat-soak survivability for RVs and gliders). Compiled from a multi-source
research pass.

> **Provenance note.** Automated fetching (WebFetch) was network-blocked (HTTP 403)
> for NTRS, ScienceDirect, AIAA, Springer, Wikipedia, etc. in the build
> environment, so the numbers gathered during research came from search-index
> extracts of these sources, **not** direct page reads. The URLs below are the
> canonical primary sources — verify exact figures/wording against them before
> formal citation. ⭐ marks the highest-value primaries to verify first.

## Obtained & verified from primary PDFs (Drive "Heating" folder)

These four were read directly and their key results confirmed:

- **Fay & Riddell (1958)** — equilibrium stagnation-point heat transfer, their Eq. 63:
  `q̇ = 0.76·Pr^-0.6·(ρ_s μ_s)^0.4 (ρ_w μ_w)^0.1·[1+(Le^0.52−1)(h_D/h_s)]·(h_s−h_w)·√(du_e/dx)_s`,
  with modified-Newtonian `(du_e/dx)_s ∝ 1/R_N` ⇒ the q̇∝1/√R law. Coefficient
  **0.76 sphere / 0.53 swept cylinder (leading edge)**; Lewis exponent 0.52
  equilibrium, 0.63 frozen; Pr=0.71. (Sutton–Graves is its engineering reduction.)
- **Tauber & Sutton (1991)** — radiative heating tabulated only for V=9–16 km/s
  (air); **negligible below ~9 km/s** ⇒ ignored for boost-glide (~6 km/s).
- **Allen & Eggers, NACA Report 1381** (= NTRS 19930091020) — blunt-body
  principle; convective heating minimized by high-drag shapes (heat load ∝ 1/drag).
- **ADA396928.pdf = Sims SP-3004** — the same DTIC scan already used to validate
  the cone wave drag (`validate_cone_wave_drag.py`); duplicate, not re-processed.
- **Reynerson, C.M. (2006), AIAA 2006-6275** — *Reentry Envelope Determination
  Part II: Structural Failure Due to Atmospheric Heating* (Boeing). The
  reentry-debris-survival "burn-up" method and **the cleanest figure of merit
  for the "does it burn up / at what point" question**:
  - heating rate (Detra–Kemp–Riddell / Bertin p.258):
    `q̇ = (11030/√R_n)·(ρ/ρ_sl)^0.5·(V/V_c)^3.15`  (V_c = circular orbital
    velocity; **verify the 11030 constant's units — Bertin's is W/cm² with R_n
    in ft, Reynerson writes R_n in m**);
  - accumulated heat `Q = Σ q̇·A_p·Δt`;
  - **melt/burn-up criterion `Q ≥ m·c·(T_melt − T₀)`** (lumped heat sink) ⇒
    heat-sink margin `= m·c·(T_melt−T₀)/Q_absorbed`; burn-up point = first
    crossing. Aluminum c=0.22 BTU/lb/°F; melt: 2024-T3 940 °F, 6061-T6 1080 °F,
    7079-T6 900 °F. This is the whole-body (unprotected / heat-sink) criterion,
    complementary to surface-T_eq (TPS surface) and time-at-temperature (TPS soak).

---


## 1. Heating correlations & entry-heating methods

- ⭐ Sutton, K. & Graves, R.A., *A General Stagnation-Point Convective-Heating
  Equation for Arbitrary Gas Mixtures*, NASA TR R-376 (1971) — q̇=K√(ρ/Rₙ)V³,
  K=1.7415×10⁻⁴ (SI), validity: enthalpy 2.3–116 MJ/kg, p 0.001–100 atm,
  Tw 300–1111 K. https://ntrs.nasa.gov/citations/19720003329
- ⭐ Fay, J.A. & Riddell, F.R., *Theory of Stagnation Point Heat Transfer in
  Dissociated Air*, J. Aero. Sci. 25(2):73 (1958) — origin of the 1/√R law
  (coeff 0.76 sphere / 0.53 swept cylinder). https://arc.aiaa.org/doi/10.2514/8.7517
- Tauber, M.E. & Sutton, K., *Stagnation-Point Radiative Heating Relations for
  Earth and Mars Entries*, J. Spacecraft & Rockets 28(1):40 (1991) — radiative
  heating, valid ~6.5–9 km/s, ±20–30%. https://arc.aiaa.org/doi/10.2514/3.26206
- ⭐ Allen, H.J. & Eggers, A.J., *A Study of the Motion and Aerodynamic Heating of
  Ballistic Missiles…*, NACA Report 1381 (1958) — blunt-body principle; heat
  load ∝ 1/C_D. https://digital.library.unt.edu/ark:/67531/metadc65613/
- NASA TFAWS Aerothermodynamics Course (Sutton-Graves/Fay-Riddell forms).
  https://tfaws.nasa.gov/TFAWS12/Proceedings/Aerothermodynamics%20Course.pdf
- Regan & Anandakrishnan, *Dynamics of Atmospheric Re-Entry* (AIAA Education
  Series, 1993), Ch. 11 "Flowfield Description" — foundational reentry-
  aerothermo text (governing equations, laminar/turbulent boundary layers,
  surface-temperature/reradiation coupling).  Corroborates the framework; the
  stagnation-heating correlations themselves are the Sutton-Graves/Fay-Riddell
  forms above.
- NASA Mars aeroheating correlations, NTRS 20200002354.
  https://ntrs.nasa.gov/citations/20200002354
- NASA SP-4201 ch.3-3 (blunt-body history; RV shock-layer ~12,000 °F).
  https://www.hq.nasa.gov/pao/History/SP-4201/ch3-3.htm

## 2. TPS materials — peak vs continuous-use temperature limits

- ⭐ Peters, A.B. et al., *Materials design for hypersonics*, Nature Communications
  15 (2024) — C/C oxidation onset, UHTC, hot-structure taxonomy, creep limits.
  https://www.nature.com/articles/s41467-024-46753-3 (PMC11026513)
- NASA TPSX material database — LI-900 / HRSI limits.
  https://tpsx.arc.nasa.gov/
- Reinforced carbon-carbon — limits, SiC coating, TEOS sealant (Wikipedia, sourced).
  https://en.wikipedia.org/wiki/Reinforced_carbon%E2%80%93carbon
- Space Shuttle TPS — tile/RCC limits, reuse thresholds.
  https://en.wikipedia.org/wiki/Space_Shuttle_thermal_protection_system
- NASA KSC STS Systems Reference — aluminum 350 °F airframe limit, RCC 3000 °F.
  https://science.ksc.nasa.gov/shuttle/technology/sts-newsref/sts_sys.html
- ⭐ *Analysis of the Shuttle Orbiter RCC Oxidation Protection System*, NASA NTRS
  19940030739 — coating mass-loss / reuse limits.
  https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19940030739.pdf
- ⭐ Cedillos-Barraza, O. et al. (2016), *Investigating the highest melting
  temperature materials* (HfC/TaC) — PMC5131352.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5131352/
- Opila, E. et al., *Oxidation of ZrB2/HfB2-based UHTCs*.
  http://li.mit.edu/Archive/Papers/05/ME2/Papers/Opila04.pdf
- *Toward Oxidation-Resistant ZrB2-SiC UHTCs*, Springer 10.1007/s11661-010-0540-8.
  https://link.springer.com/article/10.1007/s11661-010-0540-8
- NASA Ames UHTC review, NTRS 20150022996.
  https://ntrs.nasa.gov/citations/20150022996
- *SHARP-B2: Flight Test Objectives, Implementation, and Initial Results*,
  NTRS 20010046981 (UHTC sharp-edge flight test).
  https://ntrs.nasa.gov/citations/20010046981
- *Carbon-Carbon Composites: Emerging Materials for Hypersonic Flight*,
  NTRS 19900016764. https://ntrs.nasa.gov/citations/19900016764
- RCG tile coating patent US4093771 (softening ~2700 °F).
  https://patents.google.com/patent/US4093771A/en

## 3. TPS sizing & failure criteria (bondline, load, recession)

- ⭐ Myers, Martin & Blosser, *Parametric Weight Comparison of Current and
  Proposed Thermal Protection System (TPS) Concepts*, AIAA 99-3459 / NTRS
  20040086853 (NASA LaRC) — 1-D thermal-FE TPS sizing for metallic-panel,
  ceramic-tile and ceramic-blanket concepts over two reusable-vehicle entry
  profiles (Access-to-Space SSTO, RLV).  Uses **peak heat flux (Btu/ft²·s) and
  total unit heat load (Btu/ft²) as the sizing drivers** — corroborating the
  FOM's peak-flux + integrated-load pair — and an aluminium-structure limit of
  300 °F.  Corroborates the §2 tile ladder: RCG-coated tile operational
  2300 °F (=1533 K, our silica_tile continuous limit), AETB+TUFI tile 2500 °F.
  Heat-flux/load magnitudes are plotted, not tabulated, so it adds no new
  anchor.  https://ntrs.nasa.gov/citations/20040086853
- ⭐ *An Approximate Ablative TPS Sizing Tool*, NASA NTRS 20060004824 — 250 °C
  bondline criterion, recession+insulation split. https://ntrs.nasa.gov/citations/20060004824
- Beck, R., *Ablative TPS Fundamentals*, TFAWS 2017 / NTRS 20170011453 — sizing
  & margin methodology. https://ntrs.nasa.gov/citations/20170011453
- *Defining Ablative TPS Margins for Planetary Entry Vehicles* — 50%/10% margins,
  bondline-exceedance failure metric. https://www.researchgate.net/publication/268558113
- Rickman, *Ablation TPS Margin Study*, NASA JSC, NTRS 20200005815.
  https://ntrs.nasa.gov/citations/20200005815
- Orion bondline (260 °C, HT-424), NTRS 20080013535.
  https://ntrs.nasa.gov/citations/20080013535
- Jeng, M., *Aerothermodynamics and TPS Sizing of Skip Re-Entry and Aerocapture
  Vehicles*, SJSU thesis (2017) — peak-flux→material / load→thickness.
  https://www.sjsu.edu/ae/docs/project-thesis/Max%20Jeng%20-%20F17.pdf
- *Atmospheric Reentry* review (MDPI/Encyclopedia) — entry-angle & β trade,
  integrated loads. https://encyclopedia.pub/entry/37190
- CHAR / 1-D ablation-with-pyrolysis, NASA NTRS 20070022357, 20160005889.
  https://ntrs.nasa.gov/citations/20070022357

## 4. Glide vs ballistic thermal physics, oxidation, HTV-2

- ⭐ Tracy, C.L. & Wright, D., *Modeling the Performance of Hypersonic Boost-Glide
  Missiles*, Science & Global Security 28(3) (2020) — glide surface-temp-vs-time,
  reradiative-equilibrium model. https://scienceandglobalsecurity.org/archive/sgs28tracy.pdf
- ⭐ Acton, J.M., *Hypersonic Boost-Glide Weapons*, Science & Global Security 23(3)
  (2015) — trajectory model, HTV-2 L/D≈2.6. https://scienceandglobalsecurity.org/archive/sgs23acton.pdf
- *Oxidation of Carbon/Carbon through Coating Cracks*, NASA NTRS 20090004576 —
  SiC CTE-mismatch crack oxidation (HTV-2 skin-peel analog).
  https://ntrs.nasa.gov/citations/20090004576
- Jacobson & Harder, *SiC active/passive oxidation transition*, J. Am. Ceram.
  Soc. (2013) — pO₂-dependent transition (1620 K@2.5 Pa → 1816 K@123 Pa).
  https://ceramics.onlinelibrary.wiley.com/doi/10.1111/jace.12108
- US Patent 12,491,700 — *Shielded multi-layer ablative/insulative material for
  hypersonic flight* (glide thermal-load ~20× longer; thicker-ablator limits).
- ⭐ Rizvi, He & Xu, *Optimal trajectory and heat load analysis of different
  shape lifting reentry vehicles for medium range application*, Defence
  Technology 11(4), 2015 (ScienceDirect S2214914715000471) — **read from
  primary, incl. Table 3**.  Stagnation model is q̇ = C·W·R⁻⁰·⁵·V³·⁰⁵ (Scott
  et al.) with Q = ∫q̇dt — our exact peak-flux + integrated-load forms — under a
  4 MW/m² (=2900 K) heat-rate limit.  Table 3 absolute integrated loads
  (1600 km medium-range, β≈400 kg/m², burn-out 3.7 km/s): **waverider
  ≈1.66–1.86 GJ/m²** (22–26 min glide), wing-body ≈0.32–0.44, lifting-body
  ≈0.21–0.27, **bi-conic (conventional) ≈0.20 GJ/m² with a 14 MW/m² peak**.
  Confirms the cited ordering (waverider ≈8× bi-conic, ≈4–5× wing-body) and the
  glide long-soak physics (load grows ~exponentially with L/D / glide time).
  These are medium-range design-study configs, not flown-vehicle anchors, so
  they corroborate but are not added to _BENCHMARKS.  https://www.sciencedirect.com/science/article/pii/S2214914715000471
- GE Re-entry Systems (AIAA Historic Aerospace Site brochure) — qualitative RV
  history corroborating the heat-sink→ablative progression (Atlas Mk 2 heat
  sink, 1958 → Mk 3 first operational ablative sphere-cone → Titan II Mk 6),
  consistent with Bunn 1984.  No quantitative heating data.
- AIAA-2008-2539, *The DARPA/AF Falcon Program: HTV-2 Flight Demonstration Phase*.
  https://arc.aiaa.org/doi/10.2514/6.2008-2539
- DARPA ERB / HTV-2 Flight-2 cause ("unexpected aeroshell degradation").
  https://spacenews.com/darpa-engineering-review-board-concludes-review-of-htv-2-second-test-flight/

## 5. Reentry heating benchmarks (peak flux / integrated load)

- Apollo 4 stagnation-point radiation & heating (793 W/cm² peak, 46,792 J/cm²).
  https://www.researchgate.net/publication/271370890
- *Post-Flight Aerothermal Analysis of the Stardust Sample Return Capsule*
  (12.6 km/s, ~942 W/cm², ~27.6 kJ/cm²). https://www.researchgate.net/publication/27541461
- *Mars Science Laboratory Entry Capsule Aerothermodynamics*, NTRS 20070016625
  (197 W/cm² design, 5,477 J/cm²). https://ntrs.nasa.gov/citations/20070016625
- MSL heatshield aerothermo design, NTRS 20090024218 (entry-angle flux trade).
  https://ntrs.nasa.gov/citations/20090024218
- ⭐ Ried, Goodrich, Li, Scott, Derry & Maraia, *Space Shuttle Orbiter Entry
  Heating and TPS Response: STS-1 Predictions and Flight Data*, NTRS
  19820015618 — **read from primary** (JSC).  Fig. 11 gives the STS-1 windward-
  centerline (x/L=0.4) surface-heat-flux **history** vs entry time (axis
  0–20 W/cm², pulse ~0–1600 s), with flight data derived by **radiation
  equilibrium** — directly validating the FOM's q̇↔T_eq inversion — plus
  bondline/surface-T comparisons (Figs 13–15).  The integrated load is not
  tabulated, but the Fig. 11 flight curve has an **absolute** W/cm² axis, so we
  digitised and integrated it: peak ≈6 W/cm² (0.06 MW/m²) over a ~1500 s pulse →
  **∫q̇dt ≈ 6.6 kJ/cm² ≈ 66 MJ/m²** (±~20% reading error).  **This pins the
  Shuttle Q_MJ = 66 MJ/m²** at the windward-centerline acreage (the right
  location for the load metric; the q_MW=0.6 anchor remains the RCC-nose hot
  spot).  https://ntrs.nasa.gov/citations/19820015618
- ⭐ *Benchmark aerodynamic heat-transfer data from the first flight of the
  Space Shuttle Orbiter*, NTRS 19820036242 — flight-reconstructed convective
  rates: windward mid-body tiles ~5 Btu/ft²·s (≈0.06 MW/m²); RCC nose-cap /
  wing-leading-edge stagnation peak ~50 Btu/ft²·s (≈0.6 MW/m², surface
  ~1650 °C).  **Shuttle benchmark now pinned to 0.6 MW/m² (RCC stagnation),
  conf='solid'.** https://ntrs.nasa.gov/citations/19820036242
- STS-3 windward analysis, NTRS 19820020699. https://ntrs.nasa.gov/citations/19820020699
- Horvath et al., *Shuttle Entry Imaging Using Infrared Thermography*, AIAA
  2007-4267 (NASA LaRC HYTHIRM) — flight IR thermography: windward acreage
  (excl. nose/wing-LE) surface T generally **600–1100 K** over Mach 25→6;
  imagery saturates ~1480–1500 K.  Corroborates the *temperature* (T_eq) side,
  not flux: 1100 K at RCG ε≈0.89 back-computes via εσT⁴ to ≈0.07 MW/m² —
  consistent with the ~0.06 MW/m² windward-tile benchmark above.
- Taylor et al., *Global Thermography of the Space Shuttle During Hypersonic
  Re-entry*, AIAA 2011-xxxx (HYTHIRM) — 3-D windward surface-T maps for
  STS-119/125/128/132/133 near closest approach (Mach 8.4–14.3).  Independent
  multi-flight validation of the radiative-equilibrium T_eq = (q̇/εσ)^¼ method
  the FOM uses; RCC nose/LE confirmed as the hot region.  **Neither HYTHIRM
  paper pins peak flux or integrated load — Shuttle Q_MJ stays open.**
- Olynick & Tam, *Trajectory-Based Validation of the Shuttle Heating
  Environment*, J. Spacecraft & Rockets 34(2), 1997 — 3-D reacting Navier–
  Stokes over the orbiter vs the STS-2 flight database at 8 trajectory points;
  computes surface/bond-line T, heating profiles and **integrated heat loads**,
  validated against flight.  Pins the **heat-pulse duration ≈1350 s** (75,140→
  76,490 s; entry V 7.44 km/s @ 79 km) — i.e. the long-soak character — but
  reports its Table 3 integrated loads *normalised by q_ref*, in **units of
  seconds** (Q/q_ref; e.g. windward HRSI 99,341 = 535.6 s for STS-2), with
  q_ref never printed — so it corroborates the long soak and the load
  methodology but cannot itself give an absolute MJ/m².  The absolute Shuttle
  Q_MJ instead comes from integrating the STS-1 Fig. 11 flight curve (≈66 MJ/m²,
  above).
- Naved, Hermann & McGilvray, *Numerical Simulation of Transpiration Cooling
  for a High-Speed Vehicle with Substructure*, AIAA J. 59(8), 2021 — applies
  the **space shuttle reentry trajectory** (first ~900 s) to a 15° wing-leading-
  edge, 0.1 m nose radius, via Sutton–Graves; a transpiration-cooling study,
  so it confirms the trajectory/duration but reports no clean absolute Shuttle
  integrated load either.  (The Shuttle Q_MJ was ultimately pinned by
  integrating the STS-1 Fig. 11 flight curve → ≈66 MJ/m², windward centerline.)
- ⭐⭐ **Reentry F** flight experiment (1968) — the ICBM-RV peak-flux pin.
  5° half-angle slender cone, 156 in long, **R_n = 2.54 mm (0.10 in)** ablative
  ATJ-graphite nosetip, Mach ~20, V≈20,000 ft/s (6.1 km/s), ballistic.
  Flight-measured **stagnation-point heating 9,000–28,000 Btu/ft²·s
  (≈102–318 MW/m²)** over the 50,000–100,000 ft test window (stag. pressures
  5–60 atm, enthalpy ~8,000 Btu/lbm).  **ICBM-RV anchored at the 318 MW/m²
  peak, conf='solid'.**  The very sharp tip makes this far above a blunter
  operational RV (q̇∝1/√R_N → a 1–5 cm nose scales to ~70–160 MW/m²).
  Sources: NASA TM X-2584 (flight data); Berry, *Deep Dive of Reentry F Nose
  Tip Step and Gap* (NASA LaRC white paper, the 9k–28k Btu/ft²·s figures);
  Thompson, Zoby, Wurster & Gnoffo, *Aerothermodynamic Study of Slender
  Conical Vehicles*, J. Thermophysics 3(4), 1989 (VSL/engineering validation
  vs the Reentry F laminar & turbulent flight heating).
- Bunn, M., *Technology of Ballistic Missile Reentry Vehicles* (MIT STIS,
  1984) — regime context for the above (not a flux source): reentry V≳7 km/s
  at γ≈20–22°, >50 g, surface "thousands of °C", small ablating nosetip the
  most severely heated point, β≈1800 lb/ft² (Mk4) / ≈2000 lb/ft² (Mk12A).  A
  Sutton-Graves estimate at these parameters (Allen-Eggers peak-heating density
  ρ≈0.16 kg/m³, V_pk≈5.9 km/s) gives ≈45–100 MW/m² for a 2–10 cm nosetip —
  consistent with the Reentry F flight value scaled from 2.5 mm to cm-class
  tips.  https://scholar.harvard.edu/matthew_bunn/publications/technology-ballistic-missile-reentry-vehicles
- Ogasawara & Nishioka (MHI), *Proposal of the Reentry Vehicle Design Index to
  Minimize Integrated Heat Load*, AIAA 2001-1109 — integrated-heat-load design
  methodology: stagnation heating via a reference-sphere correlation
  (Detra–Kemp–Riddell), an **equivalent nose radius** for shape comparison, and
  the result that blunter shapes minimise stagnation heating.  Corroborates our
  ∫q̇dt-over-trajectory integrated-load approach and the shape→nose-radius→flux
  link, but reports a *normalised* index (no absolute MJ/m²), so it does not pin
  Q_MJ.
- Nikaido, D'Souza & Hays, *Pterodactyl: Aerodynamic and Aeroheating Database
  … Mechanically Deployed Entry Vehicle*, NASA Ames (AIAA 2020) — uses the same
  two metrics our FOM does (peak heat rate + integrated heat load over the
  nominal trajectory, CBAero-anchored).  Notes a carbon-fabric-TPS arc-jet
  heat-rate limit of **<250 W/cm² (2.5 MW/m²)**; vehicle class (deployable
  decelerator) differs from our RV/glider focus, so it is methodological
  corroboration, not a benchmark anchor.

## 6. Policy / survivability framing (independent corroboration)

- National Research Council, *U.S. Conventional Prompt Global Strike: Issues
  for 2008 and Beyond* (2008), Appendix G "The Why and How of Boost-Glide
  Systems" — semiquantitative boost-glide framing (range extension, dogleg
  maneuver, defense penetration); corroborates the glide-vehicle framing.
  https://doi.org/10.17226/12061
- ⭐ CBO, *U.S. Hypersonic Weapons and Alternatives* (Jan 2023) — heating caps
  boost-glide range ~10,000 km; sustained T ~3000 °F.
  https://www.cbo.gov/system/files/2023-01/58255-hypersonic.pdf
- GAO-21-378, *Hypersonic Weapons* (Mar 2021) — exterior T >2000 °F.
  https://www.gao.gov/assets/gao-21-378.pdf
- CRS R45811, *Hypersonic Weapons: Background and Issues for Congress* — thermal
  management "the fundamental remaining challenge."
  https://www.congress.gov/crs_external_products/R/PDF/R45811/R45811.53.pdf
- RAND RR2137, Speier et al., *Hypersonic Missile Nonproliferation* (2017).
  https://www.rand.org/pubs/research_reports/RR2137.html
- UCS, *Slowing the Hypersonic Arms Race* (Tracy, 2021) — glide centerline
  surface-temp-vs-time. https://www.ucs.org/sites/default/files/2021-04/slowing-the-hypersonic-arms-race.pdf
- Wright & Tracy, *The Physics and Hype of Hypersonic Weapons*, Scientific
  American (Aug 2021). https://www.scientificamerican.com/article/the-physics-and-hype-of-hypersonic-weapons/

---

## Verify-first shortlist (load-bearing numbers)

Done (read from primary): Fay-Riddell 1958, Tauber-Sutton 1991, Allen-Eggers
NACA 1381, Sims SP-3004. Still outstanding:

1. **Sutton-Graves TR R-376** — K constant & validity range (NTRS 19720003329).
2. ~~STS-1/STS-3 Shuttle peak heat flux~~ — RESOLVED: pinned to 0.6 MW/m²
   (RCC nose-cap/leading-edge stagnation, STS-1 benchmark NTRS 19820036242),
   conf='solid'.  ~~ICBM-RV peak flux~~ — RESOLVED: pinned to the **Reentry F**
   flight experiment, 318 MW/m² peak (9k–28k Btu/ft²·s, R_n=2.54 mm, Mach 20;
   NASA TM X-2584 / Berry white paper / Thompson 1989), conf='solid'.  Both
   former 'rough' anchors are now flight-pinned.  ~~Shuttle integrated load~~ —
   RESOLVED: Q_MJ≈66 MJ/m² from integrating the STS-1 Fig. 11 flight curve
   (windward centerline, ±~20%).  Still outstanding: an integrated load (∫q̇dt)
   for a ballistic RV — Reentry F gives instantaneous flux only (steep,
   short ablative pulse); the Reentry F Q_MJ remains the sole blank.
3. **Cedillos-Barraza 2016** — UHTC melting points (PMC5131352).
4. **Jacobson/Harder 2013** — SiC passive→active transition (jace.12108).
5. **NTRS 19940030739** — RCC oxidation/coating reuse limits.
6. **Tracy & Wright 2020 (SGS)** — glide surface-temperature-vs-time curve.
