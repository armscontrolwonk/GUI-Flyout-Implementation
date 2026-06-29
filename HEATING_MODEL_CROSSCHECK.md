# Heating-Module Cross-Check Memo

Consolidated validation record for the Thrusty reentry-heating / TPS-benchmark
module (design in `heating_module_spec.md` + `thermo_scaffold.py`; current code in
`heating.py`, wired at `trajectory.py:2788`).

This memo records what was checked against primary sources, what survived, what
needs changing, and the decisions still open. It is the bridge from the
research/validation phase to implementation. Source abbreviations:

- **Duffa** — Duffa, *Ablative Thermal Protection Systems Modeling* (AIAA, 2013) — `data/`
- **Fletcher** — Fletcher (ed.), *Aerodynamic Heating and Thermal Protection Systems* — `data/`
- **Regan** — Regan, *Re-Entry Vehicle Dynamics* (AIAA Education Series, 1984) — `data/`
- Flight/transition primaries: **TM X-2584** (Johnson et al. 1972), **Berry** (Reentry-F
  step/gap white paper, ~2025), **Williamson** (SWERVE, AIAA 92-3989, 1992),
  **Thompson** (AIAA/J.Thermophys. 1989), **Simeonides** (26th ICAS, 2008),
  **Dujardin** (Liège MSc, 2023), **Ling et al.** (arXiv 2511.16511, 2025).

---

## 1. Equations & constants — verified

| Quantity | Form used | Primary check | Status |
|---|---|---|---|
| Stagnation flux (Sutton-Graves) | `q̇ = k√(ρ/R_n)·V³`, `k = 1.7415e-4` SI | **Duffa Eq. 4.66** derives it as the **cold-wall** limit of Fay-Riddell (4.63→4.65→4.66) | ✔ form + cold-wall basis confirmed |
| Reference enthalpy (Eckert) | `h* = 0.5(h_e+h_w) + 0.22(h_aw−h_e)` | **Fletcher p.175**: `h*/h_e = 0.5(1+h_w/h_e)+0.11 r(γ−1)M_e²` — algebraically identical | ✔ exact |
| Windward laminar | `C_h = 0.332 Pr*^(−2/3) Re*^(−1/2)` | **Fletcher p.112** (Eckert/Hung) | ✔ exact |
| Windward turbulent | spec uses `0.0296 …Re*^(−1/5)` | **Fletcher p.112** uses **0.0288** | ⚠ pick one + cite (both in literature; ~3%) |
| Reradiation wall temp | `T_w = (q̇/εσ)^¼` | **Duffa p.36 Fig 1.8** (ceramic ≤ ~2000 K oxidizing; ε matters) | ✔ |
| Heat ∝ ρV³ (windward avg) | `q̇ ∝ ρV³` via Reynolds analogy | **Regan Eq. 6.110** (`q̇ = c_f ρV³/4`) | ✔ corroborates n=3 exponent |
| β convention | `β = W/(C_D A)` (repo: `RVParams.beta_kg_m2`) | **Regan Ch. VI** (Cases I–V parameterized by β, L/D) | ✔ |

**Net:** the correlation forms and constants are correct and mutually consistent
across Duffa (ablation/stagnation), Fletcher (windward/reference-enthalpy), and
Regan (trajectory/heat-load). Only the turbulent coefficient needs a one-line
decision (0.0288 vs 0.0296).

---

## 2. The apples-to-apples invariant — corroborated by Regan

Regan Ch. VI derives **ballistic and lifting re-entry as special cases of one EOM**
(6.8a/6.8b), parameterized only by **β** and **L/D** (Case I horizontal → … → Case V
lifting). This is the textbook foundation for the spec's hard requirement: *one
shared heating evaluator both trajectories call; they differ only in the state
history fed in.* Supporting cross-checks from Regan §6.8:

- Peak heating sits at a **fixed velocity fraction independent of β** (`V_m = 0.72 V_E`,
  Eq. 6.113) — the ballistic analog of the transcript's "peak flux is velocity-locked."
- Max-heating altitude ≈ **1.10× max-load altitude** (milestone-ordering check).
- **Blunt-body / heat-load ∝ 1/C_D** (Eqs. 6.120–6.122) — Allen-Eggers, re-derived.
- Driving potential is **`(T_r − T_w)`** (Eq. 6.98) — hot-wall-aware by construction
  (supports fix #1, below).

**Caveat for the validation harness:** Regan's closed forms (and Allen-Eggers, and
the equilibrium-glide anchors) are derived on the **exponential atmosphere**. The
continuity/closed-form tests (831 s / 3021 s) must therefore run the evaluator on an
exponential atmosphere to compare like-for-like, then switch to US-Std-1976 for
production (invariant #1).

---

## 3. Concern resolutions (#1–#6)

**#1 — Cold-wall `T_w` over-predicts the classifying temperature. → FIX (agreed).**
Duffa shows `q̇ = k√(ρ/R)V³` is the cold-wall limit; the true flux carries
`(h_a − h_w)/h_a`, and the radiative-equilibrium wall temperature is the root of the
**surface energy balance** `q̇_conv(T_w) = εσT_w⁴ (+ ablation)` (Duffa Eq. 4.67), with a
hot-wall-reduced `q̇_conv` — not `(q̇_coldwall/εσ)^¼`. At ~1900 °C / ~6 km/s the cold-wall
`T_w` runs ~3–4 % (≈70–90 K) too hot — enough to flip a material verdict and to
undermine the HTV-2 anchor. **Action:** classify material on the hot-wall-corrected
equilibrium `T_w`; keep cold-wall `q̇` as the cross-trajectory currency.

**#2 / #3 — Constant `H_eff` recession and omitted blowing. → use the B′ formalism
(not over-conservative, not a full solver).** Duffa §4.3/§4.7 + Ch. 5:
- Blowing: `B′ = ṁ/(ρ_e u_e C_M)`; correction `C_H/C_H0 = f(B′)` validated **≤3%**
  (Putz-Bartlett, Duffa p.158; 10³–1.5×10⁷ Pa, 11.6–30 MJ/kg, graphite/carbon-phenolic).
  For carbon in the diffusion-limited regime `B′ ≈ 0.17–0.2` → `φ ≈ 0.91`, i.e. blowing
  is only a **~10 %** effect — small, but one algebraic line, so include it.
- Recession without a constant `H_eff`: the "effective heat of ablation" is an **output**
  `∝ (h_a − h_p)` (Duffa Eq. 4.74), not a constant. Two algebraic routes:
  diffusion-limited plateau `ṁ ≈ ρ_e u_e C_M · Y_O/s`, or the steady-state-ablation
  balance (Eqs. 4.75–4.79, "good approximation for severe reentries"). Recession
  `δ = ∫(ṁ/ρ_abl)dt`.
- Carbon (our two ablators) is temperature-banded (Duffa Ch. 5): kinetic oxidation →
  **O₂-diffusion-limited plateau** (dominant for reentry) → sublimation (Knudsen-Langmuir
  Eq. 5.9). Implement either a small `B′_c(T_w,p)` table or the 3-regime analytic form.
- **Honesty note:** cold-wall + no-blowing over-predicts heating into an ablator
  (conservative on recession). Label any `H_eff` surrogate as order-of-magnitude with
  its source conditions.

**#4 — Turbulent coefficient 0.0288 vs 0.0296.** Pick one, cite it; second-order next
to the laminar/turbulent decision itself.

**#5 — Transition. → bracket, not a prediction (see §4).**

**#6 — Radiative gas heating.** Correctly omitted < ~9 km/s (Tauber-Sutton); keep the
velocity guard `heating.py` already has. New from Regan: add a **Knudsen/altitude
validity guard** — continuum convective correlations are valid only below ~80 km
(Kn < 0.01); flag the high-altitude early-soak as out-of-continuum.

---

## 4. Transition — the irreducible uncertainty (full evidence chain)

The single largest physical uncertainty, and the field has **no flight-reliable
predictor** for slender cones/RVs. Evidence, oldest → newest:

- **Reentry-F flight (TM X-2584, 1972, read from primary):** local transition Reynolds
  number is **multi-parameter** (local Mach `M_e`, wall cooling `t_w/t_e`, unit Re),
  with a **cooling-induced reversal** ("hooklike": `R_s,t` ≈ 43×10⁶ at 100 kft / M_e 15.1,
  peaks **65.6×10⁶** at 80 kft / M_e 14.4, collapses to 0.81×10⁶ by 60 kft). Correlation
  is a polynomial in `M_e` and `t_w/t_e` (Eq. 1), **not** a single `Re_θ/M_e` constant.
  Wall-to-total **enthalpy** ratio ≈ 0.03 (≠ Berry's ~0.1 *temperature* ratio).
- **Berry (Reentry-F deep-dive, ~2025):** the classic anchor is **roughness-contaminated**
  — N-factor at transition ≈ 7.5 (vs smooth-wall 9–11) from the nose-tip step/gap; the
  physical step/gap was never modeled in modern eᴺ analyses. The notorious **×12 units
  error** (Schneider/Zoby) is in the θ tables of TM X-2253 / AIAA 77-719, **not** in
  X-2584's `R_s` data. (Note: X-2584's Mach-8 ground test called the step/gap "negligible";
  Berry argues the flight N-anomaly leaves it open — a genuine source disagreement.)
- **SWERVE / Williamson (AIAA 92-3989, 1992):** flew photodiodes; the boundary layer
  **"jumped back and forth between laminar and turbulent"**, and flight agreed with
  **neither** standard method — the G.E. Low Mass Addition technique **nor the NASP
  `Re_θ = 150 M_e`** form (= our spec §4.3 criterion). Verdict: *"our inability to predict
  transition."* This directly tests and breaks our exact criterion form.
- **Simeonides (26th ICAS, 2008):** the most sophisticated correlation in this lineage
  (organized on bluntness Reynolds number `Re_b/M²`, strong/modest-bluntness forms
  eqs. 1–6; documents shock/BL-interaction transition promotion that can drop `Re_tr` by
  **>1 order of magnitude** at flap reattachment). Its own conclusion: *"underestimates
  much of the flight data over cones"*; the AoA-vs-bluntness reversal "causes remain
  unclear"; calls for new experiments. **Eqs. (3)/(4) are a defensible upgrade for the
  *nominal line* over `Re_θ/M_e`, but still under-predict cone flight.**
- **Post-2008 (2020–2025 literature):** production-CFD transition models "insufficient";
  the eᴺ N-factor is **not universal** (depends on unit Re); bypass/receptivity transition
  escapes linear theory; the community still flies dedicated experiments (HIFiRE, BOLT).
- **The "different way" (Dujardin 2023; Ling et al. 2025):** WMLES wall models that span
  laminar→turbulent. **High-fidelity CFD, low-speed/incompressible, not hypersonic**
  (Ling lists compressible as future work, blocked by data scarcity). Crucially, even the
  2025 SOTA does **not eliminate** transition uncertainty — its headline feature is to
  **quantify** it (epistemic + **aleatoric** error + confidence score). Dujardin's
  transition still hinges on a case-dependent sensor threshold.

**Conclusion / treatment.** Transition is a **bracketed, tunable, flight-unvalidated**
quantity. The module must:
1. Run heating **both fully-laminar and fully-turbulent** and report the **band** as the
   primary result.
2. Treat any criterion (`Re_θ/M_e`, or Simeonides eqs. 3/4) as a **low-confidence nominal
   line inside the band**, flagged "disagrees with SWERVE/Reentry-F flight."
3. Apply the transition rule **identically to ballistic and glide** (invariant #2) so the
   *comparison* is internally consistent even where the absolute is wrong.
4. Frame the uncertainty as **epistemic** (reducible — better correlation / flight data
   moves the nominal) vs **aleatoric** (irreducible — receptivity, roughness, the SWERVE
   jumping); the band *is* the aleatoric report.
5. Take the **turbulent bound seriously** for deflected control surfaces (SWBLI
   reattachment, Simeonides) — not a remote worst case.

---

## 5. How the band relates to the core goal (glide-time ↔ TPS type)

The band attaches to absolute **magnitudes**; the glide-time→TPS-type **linkage** is a
structural relationship that survives it.

- **Glide-time → TPS *approach*/regime (robust to the band).** Rides on integrated load
  `Q`/dwell (the "stopwatch") and the velocity-locked peak. Transition multiplies `Q` by a
  ~constant laminar/turbulent factor but does **not reorder** it; under invariant #2 you
  compare laminar-to-laminar and turbulent-to-turbulent, so the 300 < 800 < 3000 s ordering
  and the ablation→reradiation step structure hold in **both** bound-worlds.
- **Glide-time → specific *material* rung (band-sensitive).** Set by peak `T_w`. The
  **stagnation/nosetip is always laminar → transition-independent**, partially shielding the
  class call; the **windward acreage** carries the band — turbulent `q` ≈ 3–5× laminar →
  `T_w ∝ q^¼` ≈ **1.3–1.5×** in K, enough to straddle a ladder rung (e.g. RCC↔C/C).
- **Deliverable shape:** per glide time report **regime** (robust) + **(Q, T_w, dwell)
  bands** + a material verdict that is either *"class X"* (band within one rung) or
  *"X-to-Y straddle"*. The straddle is the **actionable** output — it pinpoints the glide
  times where the TPS-type decision is transition-limited (design to the turbulent bound,
  or buy a flight test / WMLES point there).
- **Why this *serves* the goal:** the coarse linkage (the NRC-panel "seconds → tier"
  result) is the robust, duration-driven part; the band only sets the **resolution** of the
  fine (material-rung) call. A single point estimate would have *hidden* exactly the
  transition-sensitivity that decides whether a glide time flies on existing TPS or forces
  the carbon-phenolic→C/C step.
- **One caveat:** at the ~1000 s ablation↔reradiation crossover a wide band can straddle the
  **regime** itself — flag that glide-time neighborhood as "regime-ambiguous under
  transition." Away from it (most 300 s / 3000 s cases) the regime call is solid.

---

## 6. Benchmark anchors — status

- **ICBM-RV 318 MW/m²** is the NASA **preflight-predicted** Reentry-F stagnation peak
  (LWP-460), corroborated by Berry (9,000–28,000 Btu/ft²·s); the ablating tip was never
  calorimetered, so it is a flight-*validated prediction*, not a measurement. Keep, but
  label epistemic class distinct from the Shuttle flight reconstruction.
- **Thompson (1989)** validates the **windward reference-enthalpy + momentum-thickness-Re**
  method against Reentry-F flight to **10–20%** — i.e. the heating correlation is
  flight-validated *given* a transition location.
- **HTV-2** (~1900 °C surface, 1090 °C / 3600 s structure) is the flown boost-glide anchor:
  the model's laminar/turbulent **band should bracket** ~1900 °C on a CSM-2-class trajectory.
- Existing `heating.py` `_BENCHMARKS` / `TPS_MATERIALS` are consistent with Duffa Table 1.4
  (flight anchors are "rough cold wall") and must be reconciled with the spec's
  `NAS_TIERS` / `MATERIAL_LADDER` (one table, not two).

---

## 7. Open design decisions (pending)

1. **File layout / reuse.** Recommend **(a)** evolve `heating.py` into the shared evaluator
   + small `thermo` helpers, reuse `atmosphere.py`, rewire `trajectory.py:2788` — vs **(b)**
   the spec's standalone `thrusty/thermo/` package. (a) honors "single source of truth" and
   minimizes churn; **unify the two material tables either way.**
2. **Regime classification by physics, not glide-time.** Derive ablation↔reradiation from
   `(Q→δ, T_w)` outputs; make the ~1000 s crossover an **emergent output**, not the
   `CROSSOVER_S` input threshold (removes the circularity).
3. **TPS-type grouping = (temperature regime) × (architecture).** Material ladder is the
   reradiative/ablative spine; add a **backface/bondline axis** (from Hu et al. 2025 review)
   to separate "thin reradiative skin OK" from "needs insulation stack." Keep the verdict on
   **passive reradiative + ablative**; name **active cooling out of scope** (cruise/propulsion).
4. **Velocity convention:** airspeed (ECEF, co-rotating atmosphere) everywhere — matches
   `trajectory.py:2765`; reconcile the `v_c`-based closed forms accordingly.
5. **Hot-wall `T_w` as the benchmark default** (fix #1); cold-wall `q̇` stays the comparison
   currency.
6. **Transition nominal:** `Re_θ/M_e` (simple) or Simeonides eqs. (3)/(4) (bluntness-Re,
   better) — either way a tunable, bracketed, flagged nominal.

---

## 8. Source ledger (what each contributed)

| Source | Read | Contribution |
|---|---|---|
| Duffa (textbook) | primary | Sutton-Graves cold-wall basis; surface energy balance; B′/blowing; carbon ablation regimes; "rough cold wall" anchors |
| Fletcher (textbook) | primary | Eckert reference-enthalpy definition; windward laminar/turbulent coefficients; heat-blockage |
| Regan (textbook) | primary | unified ballistic↔lifting EOM; ρV³ + Reynolds analogy; blunt-body 1/C_D; (T_r−T_w) potential; exponential-atmosphere caveat |
| TM X-2584 | primary | actual Reentry-F transition Re data; multi-parameter correlation + cooling reversal |
| Berry | primary | roughness/step-gap contamination; N≈7.5; ×12 units-error scope |
| Williamson (SWERVE) | primary | flight breaks `Re_θ=150 M_e`; laminar↔turbulent jumping |
| Thompson | primary | windward method validated vs flight (10–20%) |
| Simeonides | primary | bluntness-Re correlation; SWBLI promotion; cone-flight discrepancy unresolved |
| Dujardin; Ling et al. | primary | WMLES "different way"; low-speed; SOTA quantifies (not eliminates) transition uncertainty |
| Hu et al. 2025 | primary | large-area TPS architectures; bondline axis; >800 °C acreage < leading edge |
