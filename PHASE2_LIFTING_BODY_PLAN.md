# Phase 2 — Lifting-Body Trim Estimator: Consolidated Plan

Status: **plan, agreed for build** (Phase 1 — `body_form` data model + honest
depiction — shipped).  This document consolidates the literature review into
the design we will implement, with the anchor tests specified before the code
is written.  Governing rule, as everywhere in Thrusty: **derive, don't
invent** — every input is image-measurable, every output states the conditions
it was computed at, and every test is an identity, a measured datum, or a
direction — never a fit.

## 1. Objective

Give the β estimator an angle-of-attack sweep so that, from measurable
geometry alone, it produces for each `body_form` (wedge, half-cone, and the
existing cone/biconic):

- **C_L(α), C_D(α)** — modified-Newtonian pressure + skin friction + base drag;
- a single **consistent trim row**: α*, C_L*, C_D*, (L/D)max, β at trim, and
  β at zero lift — never a peak L/D detached from the α that produces it
  (the Tracy & Wright error, as diagnosed by Candler & Leyva: an assumed
  attitude and an assumed L/D that were mutually inconsistent);
- the **camber offset C_L0** (the C_L at minimum drag — nonzero for
  asymmetric bodies, per Lobanovskii), stored for the Phase-3 offset polar;
- a stated **evaluation condition** (Mach, Reynolds number, laminar/turbulent,
  base-drag convention) on every output — Fetterman: L/D comparisons are
  meaningful only at constant Re and boundary-layer state.

The trajectory physics remain untouched in Phase 2: β and L/D stay the
canonical carriers.  The estimator is where the geometry content lives,
exactly as with the biconic (§8.8).

## 2. Method skeleton (what we adopt, from where)

### 2.1 Pressure: modified Newtonian with component build-up (AEDC-TDR-64-25)

`Cp = K·cos²η` on unshadowed surfaces, `Cp = 0` in shadow; force coefficients
by (closed-form) integration over each component; components summed after
rescaling to the configuration reference area (same superposition-with-
rescaling as Grant & Braun Eq. 23, which we already use for the biconic).

**K is an explicit knob** (AEDC §1): `K = 2` (classic Newtonian, slender body
with attached shock — the default, matching the existing estimators);
`K = γ+1` (flat plate, attached shock — Love); `K = Cp,max ≈ (γ+3)/(γ+1) ≈
1.83` (blunt body, detached shock — Lees).  The AEDC Mach-8.1 delta-wing
validation shows K = 2 is good up to shock detachment (~57° α for that wing)
and K = 1.83 recovers agreement above it.  **Validity flag**: results beyond
the estimated detachment angle are flagged, not silently extrapolated.

**Wedge** (`body_form = "wedge"`): a sharp-leading-edge delta lifting body —
flat bottom, ridge-sectioned top (two facets from the centerline ridge down
to the leading edges; the natural closed solid when the LE is sharp, and
exactly AEDC's swept-wedge §2.1.5 upper half).  Composition per surface,
each a plane carrying **constant** Cp (no integration):

```
lower (flat delta plate, lit for α > 0):   Cp = K·sin²α
upper facets (AEDC Eq. 72, lit for α < ε): Cp = K·sin²(ε−α) / (1 + tan²Λ·sin²ε)
base:                                       Cp = 0 (base-drag term separate)
```

with ridge angle ε = atan(t/L), planform sweep Λ = atan(2L/b), and force
contributions computed as Cp × the facet's projected areas taken directly
from the geometry (planform projection → C_N, frontal projection → C_A) —
not AEDC's rectangular-planform area formulas, which don't apply to a delta.
At α = 0 the flat bottom carries no load and the facets push DOWN: the wedge
has negative C_N at α = 0, so the zero-lift and minimum-drag points sit at
small nonzero α — the C_L0 camber offset emerges rather than being assumed.
Leading-edge radius: not modeled in the Phase-2 sweep (sharp LE, flagged on
output); the swept-cylinder component (§2.1.3) is a later refinement.

**Half-cone** (`body_form = "half_cone"`): flat side down (windward) —
Fetterman TN D-2942 shows flat-bottom superiority for the body alone.  Two
surfaces: the flat triangular underside (a plate, Cp = K·sin²α, lit α > 0)
and the retained **upper** half of the cone lateral surface.  CAUTION: the
AEDC full-cone closed forms (Eqs. 130–135) may NOT simply be halved at
incidence — windward and leeward halves carry different loads, so ½× the
full cone is an identity **only at α = 0**.  The φ-integrals must be
re-derived over the half-range with the shadow clip `sinφ₀ = tanδ/tanα`;
they stay elementary (`∫(a − b·sinφ)²·{1, sinφ}·dφ` with a = cosα·sinδ,
b = sinα·cosδ), so closed forms survive — implemented as a general
frustum-**sector** integral (arbitrary φ-limits), of which the full cone is
a special case (its own identity test).

**Cone / biconic sweep — Phase 2b**, after the lifting forms.  Continuity
rules fixed now: for SHARP cones the sweep at α = 0 must equal
`cd_cone_hypersonic` exactly (the sector integral over the full range); for
BLUNTED noses the spherical-segment closed form will not byte-match the
shipped chart table (two pressure models for the cap) — it is a cross-check
within the chart's own accuracy band, which is precisely the validation that
retires the METHODS §8.8 chart-provenance wart.  The cap's force is treated
as axial and α-independent at screening level (second-order for small caps;
stated on output).

**Nose blunting**: spherical-segment closed form (AEDC §2.2.1, tangency
condition) — cross-checking the chart table whose provenance METHODS §8.8
flags as unverified (see the Phase-2b continuity rules above).  AEDC's own
Ref. 5 confirms the conic/spheric closed forms trace to Wells & Armstrong,
NASA TR R-127 (1962) — the citation that retires the "Ref (4) Ch. 5" wart.

### 2.2 Friction: Eckert reference-temperature Cf (Corda & Anderson 1988)

Pure Newtonian has no drag floor — L/D → cot α unbounded.  Skin friction is
what makes the ceiling honest (Candler: turbulence alone costs ~8% of L/D on
an HTV-2-class shape; Fetterman: at Re_ℓ = 1.4×10⁶ the *viscous drag at α=0
was 2–5× the inviscid drag*).  Cf stays **directly enterable** (the shipped
estimator behavior — continuity); an Eckert reference-temperature helper
PRE-FILLS it from (M, Re_ℓ, laminar/turbulent, wall-to-freestream temperature
ratio) — the method Corda validated to within 10% of a full integral
boundary-layer calculation even at high hypersonic Mach.  The wall-temperature
assumption is a stated parameter, never implicit (cold-wall for wind-tunnel
anchors, hot-wall for flight).  The viscous increment is applied over each
component's true wetted area and treated as α-independent (additive to C_D)
per Lobanovskii's experimental observation.  With Cf = 0 the sweep is the
inviscid ceiling — valid ONLY for friction-off anchor comparisons, and the
output says so.

### 2.3 Base drag

`C_D,base = 2/(γM²)` as in the shipped build-up, **switchable off** for anchor
comparisons: Fetterman's data are corrected to free-stream base pressure
(base drag excluded), and Corda excludes it entirely.  The output states which
convention was used.

## 3. Inputs (all image-measurable; span is REQUIRED for the wedge)

| form | inputs | derived |
|---|---|---|
| wedge | length L, base depth t (= the ⌀ field), **planform span b (required)**, optional LE radius | sweep Λ from planform, wedge angle ε from t/L, areas |
| half_cone | ⌀, length (existing fields) | δ from geometry; flat-bottom orientation |
| cone/biconic | existing fields (⌀, L, fore length, break ⌀, nose radius) | θ1/θ2/ε via `biconic_angles` |

Reference area is **declared in the output** (planform for the wedge, base
area for bodies of revolution) — the cruise side is invariant to the choice,
but the pull limit `q·C_L,max·A_ref/m` is not, so the convention is never
implicit.

## 4. Outputs

One table per run: α sweep from **−10° to +25°** (C_L, C_D, L/D) — the sweep
must extend below zero because the asymmetric forms cross C_L = 0 at nonzero
α (measured: TN D-2942 Fig. 6b) and C_L0 is read off the minimum-drag point.
Operational definitions: **α\*** = argmax L/D over the sweep; **C_L0** = C_L
at minimum C_D.  Plus the consistent trim row —

```
α*  C_L*  C_D*  (L/D)max   β(α=0)   β(α*)   C_L0   [M, Re, laminar/turb, base on/off, A_ref, K]
```

- **β(α=0)** is what the drag polar wants (`C_D0 = m/(β·A_ref)`) — the
  "Use β" button writes THIS one, closing the trim-β/zero-lift-β trap
  documented in METHODS §8.8.
- **(L/D)max** pre-fills glider L/D, with α* displayed so the user can
  sanity-check it (Candler: the generic HGV peaks at α ≈ 13–14°, and at
  α = 5.5° the same vehicle gives L/D 0.4 — the curve is steep).
- **C_L0** stored on the RO for Phase 3.
- **Ceiling guard**: (L/D)max above the viscous-optimized-waverider band
  (≈7 at M6 falling toward ≈6 at M14, per Bowcutt/Corda) is flagged as
  exceeding optimized-waverider physics.

## 5. Anchor tests (specified before implementation)

Identities (exact):
1. Flat plate (wedge with t → 0): `C_N = K·sin²α` on the planform, hence
   `C_L = K·sin²α·cosα`, `C_D,pressure = K·sin³α`, inviscid L/D = cot α.
2. Wedge upper facets shielded exactly at α = ε (Cp continuous through zero);
   at α = 0 the wedge's C_N is negative (facets push down) — the camber
   offset is present by construction.
3. Frustum sector over the FULL φ-range at α = 0 ≡ `cd_cone_hypersonic`
   (sharp cone); the α = 0 blunted build-up cross-checks the shipped chart
   within its accuracy band (Phase 2b).
4. Half-cone lateral integral = ½ × full cone's **at α = 0 only** — and the
   test also asserts the inequality at α > 0 (windward ≠ leeward), guarding
   against the halving shortcut.
5. K is multiplicative: K = 1.83 vs K = 2 scales every pressure component by
   exactly 1.83/2 (friction and base terms unaffected).

Measured / CFD anchors (screening band ±30% unless noted):
6. **Fetterman TN D-2942, body alone** (M 6.86, Re_ℓ 1.43×10⁶, laminar,
   base drag off): flat-bottom half-cone (L/D)max ≈ 4.6 (θ=3°), ≈ 4.0 (θ=5°),
   ≈ 3.5 (θ=9°); the flat-bottom-superior *direction* over round-bottom must
   hold, and θ=5° geometry is fully specified (ℓ=6.404 in, r_b=0.558 in — the
   longer of the two θ=5° entries in Fig. 1).
7. **Fetterman wing-body**: Λ=75°/θ=5° (L/D)max ≈ 5.4 (Fig. 6a; measured L/D
   uncertainty ±0.2), Λ=81°/θ=5° ≈ 5.0.  (Composite = half-cone + our wing
   planform machinery.)
8. **Fetterman TN D-2942 Fig. 6(b) — component-level anchor** (the strongest
   single anchor): the modified-Newtonian theory tracks the *measured* C_N(α)
   and C_A(α) for θ=5°, Λ=75° and 81° across the swept α range.  Our
   estimator must reproduce that curve, not merely the peak L/D it implies.
   The measured C_N crosses zero at slightly **negative** α — direct
   experimental confirmation of the camber offset C_L0 (§1; Lobanovskii's
   asymmetric-body trinomial polar), so the Phase-3 offset polar is anchored
   to data, not only theory.  (Axial force in TN D-2942 is corrected to
   free-stream base pressure from measured base pressures — hence anchor
   with base drag OFF.)
9. **Candler & Leyva CFD wedge** (HTV-2-class, 6 km/s): (L/D)max ≈ 2.4–2.6
   at α ≈ 13–14°, turbulent Cf at flight Re; our α* must land in ~10–18°.
10. **Grant & Braun biconic contours** (friction off, K=2, Phase 2b): peak
    L/D ≈ 1.86 (d = 21 in family) and ≈ 2.01 (d = 19.6 in, δ1=17°, δ2=10°).
11. **Viscous share** (Fetterman fig. 2): at Re_ℓ ≈ 1.4×10⁶, laminar,
    viscous/inviscid drag ratio at α=0 in the 2–5× range for slender forms.
12. **Fetterman wedge≡delta equivalence**: our swept wedge at AR 0.707 / 1.46
    tracks his 80°/70°-sweep delta-wing curves (direction + magnitude band).

## 6. Build stages and code layout

- **Phase 2a (next commit): estimator core + tests, no GUI, no trajectory
  change.**  In `booster_models.py`: `frustum_sector_newtonian(...)` (the
  general φ-limited pressure integral, full cone = special case),
  `wedge_newtonian(...)` (facet build-up), `cf_reference_temperature(...)`
  (Eckert helper), and `lifting_body_sweep(form, geom, M, Re, ...)` returning
  the α-table + trim row.  Tests in `test_lifting_body_estimator.py`:
  identities 1–5 first, then measured anchors 6–8, 11–12 (anchor 9/Candler
  needs a flight-Re Cf case; anchor 10/biconic is 2b).
- **Phase 2b: cone/biconic sweep** on the same sector machinery (continuity
  rules in §2.1), the blunted-nose chart cross-check that retires the METHODS
  §8.8 provenance wart, Grant contours (anchor 10).
- **Phase 2c — GUI**: the β-estimator dialog grows a body-form-aware mode —
  for wedge/half-cone it takes the §3 inputs, shows the sweep table + trim
  row, and "Use these values" writes β(α=0), glider L/D = (L/D)max, and stores
  α*/C_L0 on the RO; cone/biconic mode gains the same display.  The conditions
  line (M, Re, laminar/turbulent, base on/off, A_ref, K, wall temp) is always
  visible.

## 7. Phase-3 hooks (explicitly out of Phase-2 scope)

- Shape-derived C_L,max replacing the universal 0.873 body ceiling — for
  lifting forms only; every axisymmetric vehicle stays byte-identical.
- Offset polar `C_D = C_D,min + k·(C_L − C_L0)²` for asymmetric forms.
- Wedge span input removes the schematic's "span not modeled" flag.

Out of scope entirely: moment/trim-by-CG (requires a CG we'd have to invent),
wing-body interference (Fetterman: dissipates by ~M 11 and flat-bottom wins
in the glide regime anyway — our non-interference superposition is the
defensible choice), Bezier bodies of revolution.

## 8. References

- **AEDC-TDR-64-25** — Clark, E. L. & Trimmer, L. L., *Equations and Charts
  for the Evaluation of the Hypersonic Aerodynamic Characteristics of Lifting
  Configurations by the Newtonian Theory*, Arnold Engineering Development
  Center, 1964.  The primary source: modified-Newtonian K·cos²η, closed-form
  swept-wedge (§2.1.5) and cone-frustum (§2.2.5) relations, flat-bottom
  bookkeeping, spherical-segment blunting, composite superposition, and a
  Mach-8.1 75° delta-wing validation.
- **NASA TR R-127** — Wells, W. R. & Armstrong, W. O., *Tables of Aerodynamic
  Coefficients Obtained from Developed Newtonian Expressions for Complete and
  Partial Conic and Spheric Bodies…*, 1962.  Source of the conic/spheric
  closed forms (AEDC Ref. 5); the citation that retires the METHODS §8.8
  "Ref (4) Ch. 5" provenance wart.
- **NASA TN D-2942** — Fetterman, D. E., *Favorable Interference Effects on
  Maximum Lift-Drag Ratios of Half-Cone Delta-Wing Configurations at Mach
  6.86*, 1965.  Numeric half-cone anchor (geometry Fig. 1; L/D Figs. 3–6;
  theory-vs-measured C_N/C_A Fig. 6b).  Base drag corrected to free stream.
- **NASA TN D-2956** — Fetterman, Henderson, Bertram & Johnston, *Studies
  Relating to the Attainment of High Lift-Drag Ratios at Hypersonic Speeds*,
  1965.  Delta-wing ≡ equal-AR/area wedge equivalence; flat-bottom
  superiority; Reynolds-dependence discipline; interference dissipation by
  ~M 11.
- **AIAA 2010-1212** — Grant, M. J. & Braun, R. D., *Analytic Hypersonic
  Aerodynamics for Conceptual Design of Entry Vehicles*, 2010.  Superposition-
  with-rescaling method (Eq. 23) and the sharp-biconic peak-L/D contours
  (friction-off anchors ≈1.86 / 2.01).
- **AIAA 88-0369** — Corda, S. & Anderson, J. D., *Viscous Optimized
  Hypersonic Waveriders Designed from Axisymmetric Flow Fields*, 1988.  Eckert
  reference-temperature Cf (within 10% of integral BL); base drag excluded;
  the Fetterman half-cone/delta validation case; viscous-waverider L/D ceiling
  (≈7 at M6 → ≈6 at M14).
- **S&GS 30(3)** — Candler, G. V. & Leyva, I. A., *CFD Analysis of the
  Infrared Emission from a Generic Hypersonic Glide Vehicle*, 2022.  CFD
  L/D(α): (L/D)max ≈ 2.58 laminar / 2.39 turbulent at α ≈ 13–14°; the
  attitude/L-D consistency critique of Tracy & Wright.
- **Izv. AN SSSR MZhG 1983 / BF01090577** — Lobanovskii, Yu. I., *Maximal
  Lift-Drag Ratio of Wing-Cone and Wing-Half-Cone Combinations…*.  Asymmetric-
  body drag polar is a quadratic trinomial (minimum drag at nonzero lift —
  the C_L0 offset); friction increment ≈ α-independent in the range of
  interest.