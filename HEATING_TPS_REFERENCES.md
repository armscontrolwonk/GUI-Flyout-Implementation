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
- *Optimal trajectory and heat load analysis of different shape lifting reentry
  vehicles*, ScienceDirect S2214914715000471 (waverider ~10× lifting-body load).
  https://www.sciencedirect.com/science/article/pii/S2214914715000471
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
- ⭐ STS-1 entry heating, NTRS 19820015618 — **primary Shuttle windward-flux
  reconstruction (re-verify the disputed peak value here).**
  https://ntrs.nasa.gov/citations/19820015618
- STS-3 windward analysis, NTRS 19820020699. https://ntrs.nasa.gov/citations/19820020699

## 6. Policy / survivability framing (independent corroboration)

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

1. **Sutton-Graves TR R-376** — K constant & validity range (NTRS 19720003329).
2. **STS-1/STS-3 reconstructions** — the disputed Shuttle peak heat flux
   (NTRS 19820015618 / 19820020699).
3. **Cedillos-Barraza 2016** — UHTC melting points (PMC5131352).
4. **Jacobson/Harder 2013** — SiC passive→active transition (jace.12108).
5. **NTRS 19940030739** — RCC oxidation/coating reuse limits.
6. **Tracy & Wright 2020 (SGS)** — glide surface-temperature-vs-time curve.
