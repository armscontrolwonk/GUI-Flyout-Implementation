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
  **[verified]** The repo's cited source for C-HGB/SWERVe dimensions.
  **Table 2 (SWERVe, the public C-HGB predecessor):** length **2.75 m**,
  half-cone angle **5°**, span **0.87 m**, **4 fins**, US Navy 0.876 m booster.
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
  Flow," AIAA 66-455 (1966).** **[snippet]** Incipient-separation criterion for
  a deflected ramp/flap; incipient angle decreases with Mach, increases with Re.
- **Kumar, D. & Stollery, J. L., "Hypersonic control flap effectiveness,"
  *Aeronautical Journal* 100(996), 1996.** **[snippet]** M = 8.2, flap 0–30°:
  effectiveness falls once the boundary layer separates ahead of the flap.
  Practical **usable laminar-hypersonic deflection ≈ 5–15°** (M≈10 "critical
  deflection" ≈ 15°).
  *Acquisition status (checked 2026-09-04): NOT in the repo and NOT in the Drive
  library.* The citation above is itself unverified — it comes from the same
  web-search extract as the finding, so the volume/issue, the exact title and the
  author initials have never been checked against the paper, and page numbers and
  a DOI are missing. Anyone completing it should treat every field as provisional.
  Note `kumar2015.pdf` in the Drive Thrusty folder is **a different paper**
  (Kumar & Mahulikar, TPS materials, ASME JTSEA 8(2), 2016) — a name collision,
  not this reference. This band is load-bearing: `trim_gate._DELTA_MAX_BY_CONTROL`
  and `damping_estimate.DELTA_MAX_DEG` both rest on it.
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
| Usable flap deflection | ≈ 5–15° (laminar, before separation) | Kumar & Stollery | snippet |
| L/D cost at 1.5× C_L,opt | ≈ 8 % (20 % at 2×) | parabolic polar | derived |
| Trim deflection L/D cost | up to ~30 % | Ferretto 2023 | verified |
| Typical S_flap/S_ref | ~5–8 % (Shuttle elevon 8 %, body flap 5 %) | Bornemann | snippet |
| C-HGB / SWERVe | 2.75 m, 5° half-cone, 0.87 m span, 4 fins, base ≈0.48 m; lift ref = fin area | Gulan 2024 | verified |
