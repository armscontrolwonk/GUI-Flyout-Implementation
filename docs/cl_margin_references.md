# Lift-coefficient margin & control-surface effectiveness — reference base

Background literature for estimating a hypersonic glide vehicle's usable
**lift-coefficient margin** — the extra C_L it can pull above its maximum-L/D
trim by deflecting controls / increasing angle of attack — which sets the
achievable phugoid **damping ratio ζ** (see `DAMPED_GLIDE_MEMO.md` and the
`estimate_damping()` design in `docs/damping_estimate_spec.md`).

**Verification status.** Entries marked **[verified]** were read against the
actual primary-source PDF. Entries marked **[snippet]** come from web-search
extracts (full-text fetch was blocked during the search) and should be
spot-checked against the source before publication-grade quoting. Entries
marked **[derived]** are standard closed-form results computed directly.

---

## 1. Implementable Newtonian relations (the math the estimator runs)

- **Grant, M. J. & Braun, R. D., "Analytic Hypersonic Aerodynamics for
  Conceptual Design of Entry Vehicles," AIAA 2010-1212.** **[verified]**
  Newtonian sine-squared law `Cp = 2 sin²θ` (their Eq. 1) with leeward
  shadowing (`Cp = 0`); closed-form force/moment/stability coefficients for
  sharp cones, blunted biconics, and flat plates vs α and β. States explicitly
  that *"stationary fins and moving control surfaces can be approximated as
  flat plates"* — the basis for the flap ΔC_L below.
- **Anderson, J. D., *Hypersonic and High-Temperature Gas Dynamics*, Ch. 3.**
  **[snippet]** Canonical statement of classical and **modified** Newtonian
  `Cp = Cp,max·sin²θ`, with `Cp,max = (2/γM²)(p₀₂/p∞ − 1) → ≈ 1.84` at γ=1.4,
  M→∞. Flat-plate `C_L = 2sin²α cosα`, `C_D = 2sin³α`.
- **Flap lift increment (implementable form).** Treating a control surface as a
  flat plate at local inclination (Grant & Braun), the lift increment over the
  undeflected (θ = α) condition is
  `ΔC_L ≈ (S_flap/S_ref)·Cp,max·[sin²(α+δ) − sin²α]·cosα`,
  with `Cp,max ≈ 1.84` (modified) or 2 (classical). **[derived]**

## 2. Cone / biconic lift, trim, and the C_L,max ceiling

- **Penland, J. A., "Aerodynamic Force Characteristics of a Series of Lifting
  Cone and Cone-Cylinder Configurations at M = 6.83…," NASA TN D-840 (1961).**
  **[verified]** Slender-cone **C_L,max ≈ 0.5** (planform-area reference),
  nearly constant for semivertex ≤ 30°; **(L/D)_max ≤ ~3.5**, achievable only
  for semivertex < 5°; L/D = 2/1/0.5 at semivertex ≈ 8°/16.5°/26°. C_L,max
  occurs at high α (empirically α + θ_v ≈ 56°). Gives implementable cone
  Newtonian forms (`C_N = cos²θ_v·sin2α`, etc.); modified Newtonian Cp,max =
  1.822 at M = 6.83.
- **Harris, J. E., "Aerodynamic Characteristics of a Spherically Blunted 25°
  Cone at M = 20," NASA TN D-4098 (1967).** **[verified]** Blunt 25° cone
  (bluntness 0.2): **C_L,max = 0.395 at α = 25°** (base-area reference);
  (L/D)_max = 0.563 at α = 20°; lift-curve slope 0.023/deg; modified Newtonian
  `Cp = Cp,max·sin²(α+φ)` (Eq. 3) matches experiment.
- **Net ceiling:** a cone/biconic tops out at **C_L,max ≈ 0.4–0.5**, but only at
  high α (~25–50°); max-L/D trim C_L (at α ≈ 8–10°) is far lower (~0.1–0.15).
- **NASA TN, "Aerodynamic characteristics at M = 6 of a hypersonic
  configuration," NTRS 19770017117.** **[snippet]** Biconic-class **trim at
  α ≈ 8–10.5°, (L/D)_max 2.8–3.3.**
- **Tracy, C. L. & Wright, D., *Science & Global Security* 28(3), 2020.**
  HTV-2-class glide **L/D ≈ 2.6** (flight-derived). (Already in `DAMPED_GLIDE`.)
- **Küchemann L/D barrier:** `(L/D)_max ≈ 4(M+3)/M`. **[snippet]**

## 3. The C-HGB / SWERVe vehicle (the actual subject)

- **Gulan, A. E., "Conceptual, Trajectory-Based Structural Sizing Method for
  Hypersonic Glide Vehicles," M.S. thesis, Georgia Tech, Dec. 2024.**
  **[verified]** — but **SECOND-HAND for the vehicle, and wrong on the booster**;
  read the correction below before using this row.
  **Table 2 (SWERVe, the public C-HGB predecessor):** length **2.75 m**,
  half-cone angle **5°**, span **0.87 m**, **4 fins**, US Navy 0.876 m booster.

  **CORRECTION (2026-09-08).** "SWERVe" here is not a separate vehicle: it is
  Sandia's SWERVE, rendered second-hand through the Murbach / Aeolus
  Mars-derivative literature. Every Gulan dimension traces there —
  2.75 m to Murbach, AIAA 93-0313, p. 1 ("flown three times in the 2.75 m long
  version") and Murbach/Keese/Farmer, SSC97-V-2, p. 6; the 5° half-angle and the
  four cruciform wings to SSC97-V-2 p. 6 ("a sharp 5 deg half-cone with four
  wings arranged in a cruciform"). Treat this row as a **downstream citation of
  Murbach**, not as independent corroboration, and prefer the primaries.

  Two specific traps:

  * **The "US Navy booster" attribute is wrong.** SWERVE flew on Sandia's
    **STRYPI VIII-R** (Sandia LAB NEWS, 23 Jan 1981). The Navy association
    belongs to the AHW/CPS *descendants* — AvWeek (11 Oct 2018) reports the 2011
    AHW flight on a Polaris-derived stack, and the Navy leads CPS. Gulan has
    collapsed SWERVE and the C-HGB programme into one row.
  * **The 5° half-angle is a rounding.** The flown article is **5.25°**
    (Iliff & Shafer, AIAA 93-0311), corroborated by Griswold/Stein/Redding,
    SAE 820850, Fig. 2 (10.5° included) and by Murbach's own 1993 paper
    ("a sharp 10.5 degree cone") — the same author who wrote 5° in 1997.

  `ro_library/SWERVE.ro.json` carries the primary-sourced values and its own
  provenance breakdown.
  Derived base diameter ≈ 0.48 m (max cross-section ≈ 0.18 m²); span ≈ 1.8× body
  diameter. **Key modelling fact:** the **lift reference area is the fin area**
  (drag reference is the max cross-section) — i.e. the C-HGB's lift is generated
  by its four fins; they are primary lifting/control surfaces, not trim tabs.
  (Exact fin planform area / chord is in the thesis Figure 24 / its ref [35],
  not the body text — still to be pulled for an exact S_fin/S_ref.)

## 4. Control-surface sizing & effectiveness (real vehicles)

- **Scallion, W. I., "Aerodynamic Characteristics and Control Effectiveness of
  the HL-20 Lifting Body at Mach 10," NASA/TM-1999-209357.** **[verified]**
  Fin-mounted elevons (δ to ±40°) + lower body flaps (to 30°) + yaw controller;
  S_ref = 11.9 in², CG at 54 % length. **Elevons could not trim above α = 23.5°
  (target entry α = 30°)** — a documented *area-limited* control-authority
  shortfall; body-flap/yaw-controller pitching increments −0.009 / −0.006.
  (Flap planform areas are in the figures, not the text.)
- **Ferretto, Gori, Fusaro & Viola, "Integrated Flight Control System
  Characterization Approach for Civil High-Speed Vehicles in Conceptual
  Design," *Aerospace* 2023, 10(6):495.** **[verified]** *"Control surface
  deflection can cause a reduction in the aerodynamic efficiency of a hypersonic
  aircraft of up to 30 %"* (cited from their refs [1]–[4]; applied to STRATOFLY,
  body-flap δ = −15°). Use as a corroborating anchor for the L/D-vs-trim cost.
- **Bornemann et al., "Aerodynamic Design of the Space Shuttle Orbiter," NASA
  19790013835.** **[snippet]** Hypersonic L/D ≈ 1.3 at α = 34°; entry α = 40°;
  body flap is the primary pitch-trim device. Standard public areas: S_ref =
  2,690 ft², total elevon ≈ 210 ft² (**≈ 8 % of S_ref**), body flap ≈ 135 ft²
  (**≈ 5 %**) — anchors for a typical S_flap/S_ref.
- **Pezzella et al., ESA IXV aerodatabase, *Acta Astronautica* 94 (2014).**
  **[snippet]** L/D ≈ 0.7; twin body flaps, deflection −10° to +15°.
- **arXiv 2510.08275 (DLR GHGV-2 control allocation); STRATOFLY aerodatabase
  (ResearchGate 337981727).** **[snippet]** Hypersonic flap lift ≈ **linear in
  deflection**; deflection/rate limits **scale with dynamic pressure**; trim
  authority is **area-limited** ("extend the elevons to trim at the desired α").

## 5. Caveats that bound the usable margin

- **Needham, D. A. & Stollery, J. L., "Boundary Layer Separation in Hypersonic
  Flow," AIAA Paper No. 66-455, AIAA 4th Aerospace Sciences Meeting, 1966
  (Imperial College, London).** **[verified]** — read against the PDF
  2026-09-07.  Incipient-separation criterion for a deflected ramp/flap.

  Their **Fig. 11** ("Correlation of laminar, transitional and turbulent
  incipient separation on the wedge compression corner") is the usable result.
  It plots **α_i/√M∞ against Re_L** — the running-length Reynolds number to the
  corner — which collapses Mach out of the laminar branch entirely, and shows
  **three distinct branches**:

  | Branch | α_i/√M∞ | Notes |
  |---|---|---|
  | laminar | `80·(C/Re_L)^¼` | the closed form below; ≈4.5 at Re_L 10⁵ falling to ≈2.1 at 2×10⁶ |
  | transitional | dips below laminar, then climbs steeply past Re_L ≈ 2×10⁶ | dashed, M 8.2 data |
  | **turbulent** | **≈ 9.7–13**, nearly flat over Re_L 10⁶–2×10⁷ | Kuehn M 3, Sterrett & Emery M 6 |

  The turbulent branch sits **5-8× above** the laminar one — 5.1× at
  Re_L = 10⁶ widening to ~8× at 2×10⁷, because the laminar line is much the
  steeper of the two (slope −¼ in Re_L, against roughly −0.1 for the turbulent).
  A turbulent boundary layer needs a far larger pressure rise to separate.

  Needham's own text flags the laminar correlation as "a very tentative
  correlation", emphasised by "our lack of knowledge concerning the effect of
  T_w" — so band it rather than trusting it to a decimal.

- **Kumar, D. & Stollery, J. L., "Hypersonic control flap effectiveness,"
  *The Aeronautical Journal* **100**(996), June/July 1996, pp. 197–208.
  Paper No. 2151, DOI 10.1017/S0001924000067154.  College of Aeronautics,
  Cranfield University.  Also published as ICAS-94-4.4.3, 19th ICAS Congress,
  1994, pp. 1194-1204.** **[verified]** — read against the PDF 2026-09-07.

  RETRACTION OF THE 2026-09-04 "CORRECTION".  An earlier pass here asserted that
  the *Aeronautical Journal* 100(996), 1996 citation was WRONG and that this is
  "an ICAS congress paper, not Aeronautical Journal".  **That assertion was
  itself wrong and is withdrawn.**  The PDF is the Aeronautical Journal article:
  June/July 1996, pp. 197–208, Paper No. 2151, manuscript received 17 August
  1995 and accepted 1 March 1996.  The ICAS-94 congress paper is the earlier
  printing of the same study, not a replacement for it.  The original citation
  in this file was correct all along.

  (Precision, so the next reader does not have to re-derive it: what the PDF
  itself prints is the journal name, "June/July 1996", the page range, and the
  paper number — the **volume and issue are not printed on the article pages**.
  Volume 100 is The Aeronautical Journal's 1996 volume, so `100(996)` is
  consistent with the article and with the pre-existing citation, but it is
  carried over rather than read off the PDF.)

  The same pass also introduced two numbers that are **not in the paper** —
  "incipient separation 7.8° at α = 5°" and "eq. 12 predicts 8.4° at α = 10°".
  Neither appears as a deflection angle anywhere in it.  See below.

  What the paper actually reports.  Hypersonic gun tunnel, **M∞ = 8.2**
  (NOT "M ≈ 10"), Re∞/cm = 9.0 × 10⁴, quasi-2D flat plate with a full-span
  trailing-edge control flap, hingeline length **L = 15.9 cm** and flap chord
  4.4 cm (their Fig. 2), flap deflection **0 ≤ β ≤ 30°**, incidence
  0 ≤ α ≤ 10°, sharp and hemi-cylindrically blunted leading edges.

  * **Incipient separation flap angle: 6.6°** — "Under the present test
    conditions, the flap deflection for incipient separation is 6·6°" (their
    §5.1.2, α = 0°, sharp leading edge).  This is the paper's only
    separation-onset number.

    **Read what that number is, carefully** — this is where two earlier passes
    went wrong.  6.6° is **not a measurement**.  It is Eq. (6) below (the
    Needham & Stollery criterion) *evaluated at the tunnel conditions*, and the
    paper says so: the sentence before it credits Inger's triple-deck theory,
    and the sentence after concludes "This supports the prediction of the above
    criterion."  What Kumar & Stollery *measured* is only a bracket around it —
    attached at β = 5°, separated at β = 10°.  Consequence for us: the 6.6° and
    the criterion are **one source, not two**, so recomputing 6.62° from Eq. (6)
    checks our transcription and unit convention, **not** the physics.
  * At α = 0° with a **sharp** leading edge the Schlieren brackets it: "while
    for β = 5°, the flow is attached, the dual shock structure for β = 10°
    indicates that the flow has separated."
  * **Bluntness, not incidence, is what makes β = 10° attach.**  At α = 0°,
    β = 10°, with a hemi-cylindrical blunt leading edge of d = 4.0 and 6.0 mm,
    "the single flap shock implies that the suppression of separation is
    complete and the flow is attached" (their §5.3.2, Figs. 16 and 17(a),
    p. 206) — while the sharp configuration at the same condition is "well
    separated".  This is *not* a free gain in control authority: the same
    bluntness "reduces the pressure recovered downstream of the hingeline and
    hence causes significant loss of control effectiveness".
  * Incidence separately DELAYS separation — for β = 10° with a sharp leading
    edge, their Fig. 11 "shows a delay in separation as the incidence is
    increased to α = 5°" (§5.2.3) — but the sharp β = 10° case stays separated
    at every incidence tested.  A 2026-09-04 pass claimed the paper observed
    *attached* flow at β = 10° through incidence at α = 10°; that specific
    mechanism is not what the paper reports, though attached β = 10° flow does
    exist in it under bluntness.
  * Flap boundary-layer state with deflection: "The flap boundary layer changes
    from laminar for β = 5° to transitional at β = 15° and to a turbulent
    structure at β = 25°."  **Scope matters**: this sentence is in §5.3.2, the
    **blunt** leading-edge results (d = 6 mm, α = 0°, Fig. 20) — it is not a
    general law.  On the *sharp* leading edge (§5.1.2) transition is already
    under way earlier: at β = 15° transition is complete by the flap trailing
    edge and the local turbulent heat-transfer level is attained there.  This
    sentence is nevertheless the likely origin of the old "5–15°" band, since
    it is the only place those two numbers appear together.
  * Conclusions: "Flap deflection promotes separation of laminar boundary
    layers"; "Incidence promotes transition of the flap boundary layer.  It
    delays separation"; large bluntness "substantially delays separation" but
    "reduces the pressure recovered downstream of the hingeline and hence causes
    significant loss of control effectiveness".

  **It carries the Needham & Stollery criterion in implementable form**, as
  their Eqs. (6) and (7):

  ```
  M∞·β_i   = 80·χ̄_L^(1/2)        (6)   incipient separation flap angle
  M∞·β_sep = 50·[χ̄_sep]^(1/2)     (7)   separation streamline angle, well-separated
  χ = M³·√(C/Re_x)                      viscous interaction parameter (their nomenclature)
  ```

  **β in DEGREES**, Re at the hingeline length, C the Chapman-Rubesin constant.
  Transcription check (**not** an independent validation — see the caveat above,
  the paper's 6.6° is itself this equation): at the paper's own conditions
  (M 8.2, Re∞/cm 9.0 × 10⁴, L = 15.9 cm → Re_L = 1.43 × 10⁶, C = 1) Eq. (6)
  returns **6.62°** against the paper's stated 6.6°, which pins the unit
  convention (degrees, not radians) and the choice of length scale.

  **⚠ Which χ, exactly — read this before implementing.** The overbar in χ̄_L is
  **never defined in the paper**. Its nomenclature defines a bare
  `χ = M³√(C/Re_x)` and, separately, a wall-temperature-weighted
  `χ_e = ε[0.664 + 1.73(T_w/T_0∞)]·χ` with `ε = (γ−1)/(γ+1)` — carried by a
  subscript, not an overbar. Their own Fig. 6 annotation writes the same
  relation as `M∞β_i = 80·χ_L^(1/2)` with **no** overbar. The ambiguity is
  resolved numerically, and it matters a lot:

  | reading of χ̄_L | value at the paper's conditions | β_i |
  |---|---|---|
  | **bare χ, C = 1** | 0.461 | **6.62°** ✓ matches their 6.6° |
  | χ_e, T_w/T_0∞ = 0.3 | 0.091 | 2.94° ✗ |

  So **χ̄_L is the bare viscous-interaction parameter**. Anyone implementing this
  from the nomenclature alone would reach for χ_e and land 2.2× low.  Eq. (6) is exactly the laminar branch of Needham's
  Fig. 11 — it reduces to `β_i/√M∞ = 80·(C/Re_L)^¼`, Mach-independent, which is
  why that figure's ordinate collapses.

  There is **no "usable deflection ≈ 5–15°" statement in the paper**, and no
  "critical deflection ≈ 15°".  The old 5–15° band conflated the
  boundary-layer-state sequence (laminar 5° / transitional 15°) with a usable
  limit.  As for "M ≈ 10": this study's own condition is M 8.2 throughout, but
  M ≈ 10 does appear in the paper several times — always about *other* work or
  as borrowed data.  Their §2 cites Townsend at M∞ = 10.0 and Coet et al. at
  M∞ = 10.0, §2 also cites Sanator et al. at M∞ = 10.55, and §§3.4/5.3.3 build
  the entropy-layer model on Stone's Mach 10.4 pitot data.  So the misattributed
  Mach number has several candidate sources, not one; what is certain is that it
  is never this paper's test condition.

  The "8.4" is narrower and can be pinned exactly: it occurs **once** in the
  paper, as `Re∞/cm = 8·4 × 10⁴` in "In tests at M∞ = 10·0 and
  Re∞/cm = 8·4 × 10⁴ on a compression corner, Coet et al. found…" (§2).  It is a
  **Reynolds number**, not a deflection angle — and it is where the 2026-09-04
  pass got the "8.4°" it reported as a separation onset.

  WHAT THIS MEANS FOR THE CODE.  Applying Eq. (6) to a Thrusty body gives 2–6°
  across the glide envelope — but that is the **laminar** branch, and a flight
  vehicle is not laminar at the hingeline.  Re_L for a 6 m body over M 3–10 and
  25–40 km runs **1.4 × 10⁶ to 4.9 × 10⁷**, at or beyond where Needham's laminar
  branch ends.  On the turbulent branch the same correlation gives

  | Condition (L = 6 m) | Re_L | laminar | turbulent |
  |---|---|---|---|
  | 40 km, M 3 | 1.4 × 10⁶ | 4.0° | 17–23° |
  | 35 km, M 5 | 5.0 × 10⁶ | 3.8° | 22–29° |
  | 30 km, M 8 | 1.8 × 10⁷ | 3.5° | 27–37° |
  | 25 km, M 10 | 4.9 × 10⁷ | 3.0° | 31–41° |

  So the evidence does **not** support tightening the deflection cap below 15°;
  it points the other way.  `trim_gate._DELTA_MAX_BY_CONTROL['substantial']` and
  `damping_estimate.DELTA_MAX_DEG` at 15° are **conservative against the
  turbulent onset**, which is the regime Thrusty actually flies in.  The right
  long-run fix is not another constant but a branch-selected β_max(M, Re_L, BL
  state); see TODO.md item 9(d).

  (Note `kumar2015.pdf` in the Drive Thrusty folder is a DIFFERENT paper —
  Kumar & Mahulikar, TPS materials, ASME JTSEA 8(2), 2016 — a name collision.)

- **Maus, Griffith, Szema & Best, "…Real Gas Effects on Space Shuttle Orbiter
  Aerodynamics," *J. Spacecraft & Rockets* 21(2), 1984 (and the STS-1 trim
  anomaly, DOI 10.2514/3.26680).** **[snippet]** Real-gas γ reduction shifted
  the center of pressure; body flap needed ≈ 16° vs ~11° predicted — Newtonian
  flap predictions are optimistic above M ≈ 5–7 and must be derated.
- **Induced-drag / range cost.** For a parabolic polar `C_D = C_D0 + k·C_L²`,
  flying at `C_L = n·C_L,opt` gives **`(L/D)/(L/D)_max = 2n/(1+n²)`** **[derived]**:
  n = 1.5 → 0.92 (~8 % loss); n = 2 → 0.80 (20 %); n = 3 → 0.60 (40 %). Flat
  near the optimum, steep beyond ~2×. Range ∝ L/D for equilibrium glide
  (Eggers, Allen & Neice, NACA TN 4046, 1957). **[snippet]**

---

## Headline numbers (for the estimator)

| Quantity | Value | Source | Status |
|---|---|---|---|
| Newtonian Cp,max (modified) | ≈ 1.84 | Anderson Ch.3 | snippet |
| Flap ΔC_L | (S_flap/S_ref)·Cp,max·[sin²(α+δ)−sin²α]·cosα | Grant & Braun | verified+derived |
| Cone/biconic C_L,max | ≈ 0.4–0.5 (at α ≈ 25–50°) | TN D-840, D-4098 | verified |
| Biconic max-L/D trim α | ≈ 8–10° | NTRS 19770017117 | snippet |
| Laminar incipient separation flap angle | **6.6°** at M 8.2, Re_L 1.43×10⁶ — Eq. (6) evaluated, not measured; measurement is the bracket β=5° attached / β=10° separated (sharp LE, α=0°) | Kumar & Stollery 1996 §5.1.2 | **verified** |
| Incipient separation criterion | `M∞·β_i = 80·χ̄_L^(1/2)`, χ = M³√(C/Re_x), β in degrees | Kumar & Stollery Eq. (6) / Needham & Stollery | **verified** |
| Turbulent incipient separation branch | `β_i/√M∞ ≈ 9.7–13` → **17–41°** over M 3–10 at flight Re_L | Needham & Stollery Fig. 11 | **verified** |
| L/D cost at 1.5× C_L,opt | ≈ 8 % (20 % at 2×) | parabolic polar | derived |
| Trim deflection L/D cost | up to ~30 % | Ferretto 2023 | verified |
| Typical S_flap/S_ref | ~5–8 % (Shuttle elevon 8 %, body flap 5 %) | Bornemann | snippet |
| C-HGB / SWERVe | 2.75 m, 5° half-cone, 0.87 m span, 4 fins, base ≈0.48 m; lift ref = fin area | Gulan 2024 | verified |
