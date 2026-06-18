# Damped‑Phugoid Glide (`damped_glide`)

A glide‑guidance mode for high‑L/D vehicles that reproduces what a *guided*
hypersonic glider actually does: a pull‑up plus a few **decaying** skips settling
into equilibrium glide.

The **four core** reentry models form one physical spectrum in how strongly the
re‑entry **phugoid** is damped, and `damped_glide` is its tunable middle:

| Core reentry model | Phugoid behavior | Damping |
|---|---|---|
| Ballistic | no lift, no glide | — |
| Phugoid / skip‑glide (`skip_glide`) | undamped — oscillates indefinitely (skips out for energetic entries) | ζ = 0 |
| **Damped phugoid glide (`damped_glide`)** | a pull‑up plus a few decaying skips into equilibrium | ζ ≈ 0.7 |
| Non‑oscillatory glide (`equilibrium_glide_acton`, "Acton") | analytic capture, no oscillation | ζ → ∞ (limit) |

`damped_glide` is a **new mode** that fills the physical middle of this spectrum.
Two further **legacy** models are retained for comparison: `equilibrium_glide`
(Tracy's steady equilibrium glide) and `skip_to_equilibrium` (skip‑glide with a
one‑way hand‑off to equilibrium after a set number of skips). The pre‑existing
modes are otherwise unchanged — `equilibrium_glide_acton` is now surfaced as the
core "Non‑oscillatory glide (Acton)" mode.

## The control law

> **Dynamic EOM, no free lift** (see `GLIDE_CAPTURE_DESIGN.md` §8). Drag is
> coupled to the actual commanded lift and lift is bounded by the aerodynamic
> ceiling, so the old "free" pull‑up (effective L/D above the vehicle's) is gone.
> There are now **two** dynamic glide modes — this one (`damped_glide`) genuinely
> *damps the phugoid*, and `dynamic_equilibrium_glide` captures *smoothly*
> (see below).

`damped_glide` flies the **max‑L/D trim α\*** (identical to `skip_glide`, so the
natural phugoid is preserved) plus a feedback term on the altitude‑rate error
that **damps the skips**:

```
L·cos σ_cmd = L_α*·cos σ − k_h·(ḣ − ḣ_eq)        (Lu 2013 Eq. 33;
                                                  equiv. Yu & Chen 2011 Eq. 19)
```

- `L_α*` = the max‑L/D lift (q·A·C_L\* for the slender‑body polar; (q/β)·m·L/D for
  constant_LD) — i.e. exactly the `skip_glide` lift.
- `ḣ = V·sin γ`; `ḣ_eq = V·γ*`, with the quasi‑equilibrium‑glide angle (Lu Eq. 31)
  `γ* = −2·H_ρ·g / (V²·cos σ·(L/D))` (L/D in the denominator — higher L/D glides
  shallower).
- `k_h = 2·ζ·m·√(g_eff/H_ρ)` (derived below). **Drag is coupled** to the actual
  commanded C_L (polar: `C_D = C_D0 + k·C_L²`, capped at C_L,max; constant_LD:
  `drag = L/(L/D)`, capped at the β‑available lift) — no free lift.

Because the nominal is the α\* lift (∝ q), the vehicle over/undershoots
equilibrium — the **phugoid** — and ζ damps it: **ζ = 0 ≡ `skip_glide`** (undamped
skips), and increasing ζ gives **fewer, smaller decaying skips** into equilibrium.
On an uncapturable (lofted) entry it **plunges**, exactly like `skip_glide`.

> **Sibling mode — `dynamic_equilibrium_glide`.** Same EOM/no‑free‑lift framework,
> but the nominal is the **equilibrium trim** `L·cos σ = m·(g − V²/r)`, so it
> captures **smoothly with no oscillation** (the ζ knob there is a *tracking gain*,
> not a damping ratio). Use `damped_glide` to model the realistic guided pull‑up
> *with* decaying skips; use `dynamic_equilibrium_glide` for the smooth capture.

## The gain — derived, not fitted

Linearising the planar equilibrium‑glide EOM about equilibrium gives, from first
principles, a simple harmonic oscillator for the altitude perturbation. At the
equilibrium glide the (fixed‑α) lift acceleration balances `g_eff = g − V²/r`;
since lift ∝ ρ ∝ e^(−h/H_ρ), a small displacement δh changes it by
`d a_L/dh = a_L·(d ln ρ/dh) = −g_eff/H_ρ`, so the open‑loop altitude mode is

```
δḧ + ω_p²·δh = 0,    ω_p² = g_eff / H_ρ      (g_eff = g − V²/r)
```

where `H_ρ = −ρ/(dρ/dh)` is the local density scale height — the restoring force
is the density lapse (drop below equilibrium → denser air → more lift → pushed
back up, the mechanism Yu & Chen describe explicitly). This linearisation is not
ad hoc: the planar entry equations collapse to a single second‑order nonlinear
ODE whose **primary source is Chapman** (NACA TN 4276 / NASA TR R‑11, 1958–59,
Eq. 21), reducing the two motion equations to one ODE in a density‑like variable
`Z(ū)` (`ū = V/V_circ`, `Z ∝ ρ`). Its truncation neglecting vertical acceleration
is the **equilibrium glide** — Chapman's `Z_II` solution, in his words
"equilibrium gliding flight originally discussed by Sänger" — and the *full*
equation produces, for higher L/D, the oscillation Chapman calls "numerous skips
of sizable intensity" (Fig. 6). **Yaroshevskii's equation** (Vinh, Busemann &
Culp, Ch. 10, Eq. 10‑55, `y″ = −K + (e^{2x}−1)/y`, `y ∝ ρ`, `K = √(βr₀)·C_L/C_D`)
is a special case of Chapman's, with the same equilibrium‑glide solution
(Eq. 10‑61, the Sänger condition) and the same numerically‑shown oscillation
(Fig. 10‑10). Linearising the second‑order ODE about equilibrium glide gives
precisely the oscillator above; **all three sources exhibit the oscillation but
none writes the closed‑form oscillator** — that one‑line step is taken here. The
frequency is also corroborated empirically: Liu et al. (2025) measure the skip
phugoid at 0.021–0.037 rad/s, bracketing ω_p ≈ 0.034 rad/s here. The altitude‑rate feedback
(Lu 2013, Eq. 33) adds the damping term, giving `δḧ + 2ζω_p·δḣ + ω_p²·δh = 0`;
matching to `2ζω_p` fixes the gain for a **target damping ratio ζ**:

```
k_h = 2·ζ·m·√(g_eff / H_ρ)
```

`ζ` is the single user knob (`glider_damping_zeta`, default **0.7**). The gain is
computed *each step* from the current state, so it schedules down naturally as V
and g_eff change — matching Lu's velocity‑scheduled gain (his Eq. 34). The 0.7
default is the classical second‑order control value: it lies in the desirable
ζ = 0.4–0.8 band (Ogata §5‑3, p.171; Franklin §3.4.2 / Fig. 3.24, ζ = 0.7 → ~5 %
overshoot) and is very nearly settling‑time‑optimal (Ogata p.173, t_s minimum at
ζ ≈ 0.68–0.76). See `DAMPED_GLIDE_MEMO.md` §5 for the full rationale.

Crucially, ζ is a property of the *guidance*, not the airframe: the bare
vehicle's skip phugoid is essentially undamped — even mildly unstable
open‑loop (Liu et al. 2025) — and is suppressed only by active control
(Tracy & Wright 2020), so the designer sets it, dialling in more damping for a
smoother, single‑bounce capture (Acton 2015) or less for a bouncier,
longer‑skipping profile, with 0.7 the conventional well‑guided midpoint.

### Three independent groundings of the gain (the curated library)

1. **First‑principles (the linearisation above):** the derivation — `k_h = 2ζm√(g_eff/H_ρ)`.
2. **Empirical (Yu & Chen 2011, Table 1):** their flight‑path‑angle‑feedback gain
   sweep (k_γ = 0→15) shows strong damping at k_γ ≈ 3–5 with large reductions in
   peak heating/q/load for a few‑% range cost. Evaluating our derived gain at
   representative HGV conditions (C_L\*≈0.15, V=6 km/s, g_eff≈4 m/s², H_ρ≈7 km,
   ζ=0.7) lands at k_γ ≈ 3.7 — squarely in their strong‑damping band.
3. **Analytical (Lu 2013):** his nondimensional `k₀≈20` schedule, as a magnitude
   sanity bound.

All three agree on the magnitude, which is why ζ≈0.7 is a defensible default.

## Nesting (the safety property)

`ζ = 0` ⇒ `k_h = 0` ⇒ the feedback term vanishes ⇒ the law is **exactly
`skip_glide`** (the α\* lift). Verified bit‑exact (`max|Δalt| = 8.4e‑12 km` over a
full integration) in `damped_glide_smoke_test.py`. Increasing ζ damps the
phugoid: the skip amplitude falls monotonically (e.g. **43 → 8 → 3 → 0 km** of
re‑climb as ζ = 0 → 2 on a capturable insertion), so the number of decaying skips
*emerges* from ζ rather than being a hand‑set integer. Whether the vehicle
*captures* at all is set by the entry geometry and aero model, not by ζ (a lofted
entry plunges at every ζ) — reported by the diagnostic glide‑regime classifier
(`glide_regime.py`).

## Relation to the analytic equilibrium modes (Acton / Tracy)

The two analytic equilibrium modes are the limiting cases this mode generalises.
Tracy & Wright (2020) describe the phugoid as *"minor oscillations about the
equilibrium flight altitude… [that] could be damped by active control of the
vehicle"* and then model the steady glide (`equilibrium_glide`, "Tracy"). Acton
(2015) goes further and assumes the oscillation away, *"assuming that the vehicle
does not oscillate during the transition to equilibrium gliding"* — so his
closed‑form pull‑up arc (`equilibrium_glide_acton`, the primary "Non‑oscillatory
glide (Acton)" mode) is the **infinitely‑damped (ζ → ∞) endpoint** of the
spectrum above. Acton justifies this as design intent (*"the DARPA
schematic shows the glider bouncing just once during the pull‑up"*) and notes
that a model which *permitted* oscillation would lose more speed in the pull‑up —
which is exactly what a finite ζ does.

Empirically, on the C‑HGB no single ζ bit‑reproduces the Acton mode (different
model: his βS blunt‑entry plus ρ/β‑matched rotation vs. dynamic lift feedback),
but **ζ ≈ 0.7–1.0 matches Acton's range to a few percent**, and ζ ≈ 0.7
reproduces it almost exactly *while showing the single bounce* Acton attributes
to the real vehicle. Prefer the Acton mode when you want a parsimonious,
auditable, parameter‑free range estimate matching the published reference method;
prefer `damped_glide` when the realism of the pull‑up transient is the object of
study. See **`DAMPED_GLIDE_MEMO.md`** for the full derivation, the ζ=0.7
rationale, and the when‑to‑use comparison.

## Validation

`damped_glide_smoke_test.py` flies the repo's **C‑HGB** glide body
(`rv_library/C-HGB.rv.json`, SWERVe/AHW‑descendant; Gulan, Georgia Tech 2024) on
a **lofted** sub‑circular boost, and the **AUR** on a **depressed (shallow)
insertion**; it pins the behaviour of both dynamic modes (7/7 checks):

- **ζ = 0 ≡ `skip_glide`** — bit‑exact (`max|Δalt| = 8.4e‑12 km`).
- **Decaying skips:** on the capturable shallow insertion the skip amplitude
  falls monotonically with ζ — **43 → 8 → 3 → 0 km** of re‑climb as ζ = 0 → 2.
  This is the realistic guided pull‑up: a few decaying skips into equilibrium,
  the count set by ζ.
- **No free lift:** a captured damped glide never out‑ranges the analytic
  equilibrium glide (effective L/D ≤ vehicle L/D). The old law glided ~20 % *too
  far* (effective L/D ≈ 2.4 on an L/D = 2 vehicle); that artifact is gone.
- **Honest plunge on a lofted entry:** the C‑HGB on a *lofted* ballistic boost
  **plunges at every ζ** (both aero models) — the thin‑air lift ceiling is too
  small high up to arrest the steep entry. Matches Lu's deep‑dive ballistic
  launch and the observed real C‑HGB falling deeply.
- **Capturability is entry‑geometry dependent:** same law, lofted → plunge,
  shallow → capture. Real boost‑glide vehicles are inserted shallow, not lofted.
- **Consistent across aero models:** `constant_LD` is capped at the β‑available
  lift `(q/β)·m·(L/D)`, so it matches the physical polar model (plunges lofted,
  captures shallow, no zoom‑climb) instead of over‑pulling out via the lumped
  model's missing C_L,max ceiling.

(`dynamic_equilibrium_glide` captures the shallow insertion **smoothly** — no
skips, monotonic dive‑arrest onto the descending glide — and likewise plunges
the lofted entry.)

Whether a given boost produces capture is read off by the diagnostic
**glide‑regime classifier** (`glide_regime.py`, attached to each trajectory
result as `glide_regime`): {`skip`, `capture`, `plunge`}.

## Limits & notes

- Gated, like `skip_glide`, on being below the 100 km re‑entry pierce altitude
  and on dynamic pressure (`q > 1` Pa) — no aerodynamic control in vacuum.
- The feedback is disabled when `g_eff = g − V²/r ≤ 0` (at/above circular speed
  the phugoid restoring force is not defined); the vehicle then flies plain α\*.
- `H_ρ` is computed by finite‑differencing the COESA atmosphere and clamped to
  4–12 km for robustness.
- Lift is bounded by the existing `pullup_g_max` cap, so the maneuver respects
  the structural g‑limit automatically (unlike the analytical arc).
- **Aero model (`polar` is now the default).** `polar` charges induced drag
  (`C_D = C_D0 + k·C_L²`) and only realizes the vehicle's full `(L/D)_max` at
  the trim point `C_L = C_L*`; the equilibrium‑trim lift command generally sits
  *off* that point, so the realized L/D — and hence range — is below nominal.
  `constant_LD` is the idealized fixed‑L/D upper bound: it asserts full L/D at
  all times and never pays the off‑design induced‑drag penalty, so it
  over‑ranges the polar by ~15% on the same insertion. `constant_LD` is kept
  for cross‑checking the closed‑form Sänger/Tracy/Acton range solutions (which
  assume constant L/D), but the default is now `polar` so the out‑of‑the‑box
  number is the realistic one. The two coincide exactly at `C_L = C_L*`.
  *Range ordering on the same shallow insertion:* `damped_glide` (skip/boost‑
  glide, lofts through thin air at max‑L/D AoA) > `dynamic_equilibrium_glide`
  (steady equilibrium descent, the range floor of the family); the gap closes
  as ζ→large damps the skips — i.e. it is the physical Sänger–Bredt skip premium,
  not free lift.
- **Dynamic vs. analytic equilibrium glide — capture loss.** The dynamic modes
  generally range shorter than the analytic `equilibrium_glide` /
  `equilibrium_glide_acton`, but **not** because the analytic models neglect the
  pull‑up energy loss — they account for it analytically. Acton (2015, Eq. 11)
  treats the speed bled during the pull‑up as the closed‑form turn‑loss integral
  `v₄ = v₃·exp(−θ₂/(L/D))` (descent angle θ₂ at atmospheric piercing), then seeds
  the glide at the equilibrium altitude `h_eq(v₄)`; our `_acton_pullup_arc`
  implements this for both analytic modes (the Acton‑specific addition is the
  Phase‑3 β_S direct‑reentry descent that sets v₃, h₃). The residual gap is
  therefore *idealized* vs *dynamic* capture loss: Acton's relation is the
  efficient lower bound — a perfect constant‑L/D arc arriving exactly at `h_eq`
  with γ=0, which always captures — whereas the dynamic EOM integrates the real
  density profile, typically overshoots `h_eq` into denser air (extra drag loss),
  and for a steep enough entry fails to capture at all (plunge). So the analytic
  models bound the capture loss optimistically; they do not ignore it.
- **Reconciliation with Tracy's 10–20% pull‑up penalty.** Tracy & Wright (2020)
  assert a 10–20% range penalty for the pull‑up maneuver. Our dynamic
  `damped_glide` at ζ=0.4 on the shallow AUR insertion reproduces this from first
  principles: pull‑up velocity loss (v₃ at the 100 km descending pierce → speed at
  glide onset) is **14%** for `constant_LD` (Tracy's modeling assumption) and 19%
  for `polar`, with corresponding range penalties of **14%** and 20% vs the
  analytic equilibrium glide (5834 km → 5002 / 4643 km). Equivalently, knocking
  Tracy's 10–20% off the idealized glide gives [4667, 5251] km, which brackets
  both dynamic results. **Crucially, this penalty is the dynamic capture
  *overshoot* loss, not Acton's idealized turn loss.** For this geometry θ₂ at
  piercing is only 3.5°, so Acton Eq. 11 alone gives `exp(−θ₂/(L/D)) = 0.967` —
  just **3.3%**. The remaining ~11–16% comes from the vehicle plunging below its
  eventual glide altitude (to ~36 km `polar` / ~26 km `constant_LD`, vs a ~44–48 km
  glide) and bleeding speed through that dense‑air excursion and the climb back.
  The dynamic EOM captures this; the closed‑form turn integral by itself
  under‑counts it for shallow entries. The exact figure is ζ‑ and
  geometry‑dependent (deeper entries and lower ζ → larger overshoot loss); ζ=0.4
  on this insertion happens to sit mid‑band.
- **Relationship between the two ζ knobs (damped_glide ↔ dynamic_equilibrium_glide).**
  The two control laws are *identical* except for the nominal lift they damp
  toward; the feedback term is the same:

  ```
  damped_glide              : L = L_skip(α*)   − k_h·(ḣ − V·γ*)
  dynamic_equilibrium_glide : L = m·g_eff/cosσ − k_h·(ḣ − V·γ*)
                                  └─ anchor ──┘   └── same feedback ──┘
  ```

  Both use the **same gain** `k_h = 2ζ·m·√(g_eff/H_ρ)` and the **same target**
  descent rate `V·γ*`. The only difference is the nominal anchor: `damped_glide`
  anchors on the skip / max‑L/D lift (which is what makes ζ=0 reduce exactly to
  `skip_glide`), while `dynamic_equilibrium_glide` anchors on the exact
  force‑balance equilibrium lift `m·g_eff/cosσ`.

  Two distinct things converge at different rates, and it matters which you mean:

  1. **Steady‑glide *tracking* error.** At the settled glide `damped_glide`'s lift
     is `m·g_eff/cosσ + (L_skip − L_eq)/k_h`, so the off‑equilibrium error in
     descent rate / lift decays as **1/ζ** — doubling ζ halves it. In this
     (steady‑state, force‑balance) sense `dynamic_equilibrium_glide` is the exact
     ζ→∞ limit of `damped_glide`, anchored directly so it sits there at any gain.

  2. **Range.** The *range* difference does **not** follow 1/ζ and does **not**
     close at usable ζ. It is dominated by a **ζ‑independent capture‑transient
     offset**: `damped_glide`'s max‑L/D skip nominal pulls out of the first dip
     higher and banks a roughly fixed capture‑energy advantage. On the shallow AUR
     insertion (`constant_LD`, dyn‑eq ≈ 4878 km) the damped range is **flat at
     ~5010 km (gap ~131 km, ≈2.7%) for ζ = 1…16**, and only begins eroding above
     ζ≈32 (5003 @ ζ=32, 4988 @ 64, 4963 @ 128, 4928 @ 256 — a log‑log slope of
     about −0.3, far shallower than 1/ζ). So bit‑level range convergence would need
     ζ in the hundreds–thousands.

  The practical consequence: **at any usable damping ratio `dynamic_equilibrium_glide`
  is a genuinely distinct, ~2–3% shorter capture maneuver — not "damped_glide at
  high ζ."** The two are correctly separate modes. (For `polar` the gap closes
  somewhat faster and crosses slightly below dyn‑eq near ζ≈16 via the additional
  C_L↔L/D coupling.) This is intended behavior — not free lift and not a bug.

## Sources

- P. Lu, S. Forbes, M. Baldwin, *Gliding Guidance of High L/D Hypersonic
  Vehicles*, AIAA 2013‑4648 — feedback damping law (Eq. 33), gain schedule
  (Eq. 34), command flight‑path angle γ\* (Eq. 31).
- W. Yu, W. Chen, *Guidance Scheme for Glide Range Maximization of a Hypersonic
  Vehicle*, AIAA 2011‑6714 — flight‑path‑angle feedback (Eq. 19) and the
  empirical gain/heating/range sweep (Table 1, Figs. 6–10).
- J. M. Acton, "Hypersonic Boost‑Glide Weapons," *Science & Global Security*
  23:191–219, 2015 (DOI 10.1080/08929882.2015.1087242) — non‑oscillatory pull‑up
  assumption (the infinitely‑damped limit of this mode).
- C. L. Tracy, D. Wright, "Modeling the Performance of Hypersonic Boost‑Glide
  Missiles," *Science & Global Security* 28, 2020 (DOI 10.1080/08929882.2020.1864945)
  — equilibrium‑glide formulation; phugoid "damped by active control."
- N. X. Vinh, A. Busemann, R. D. Culp, *Hypersonic and Planetary Entry Flight
  Mechanics*, University of Michigan Press, 1980 — Ch. 10 (Yaroshevskii's
  theory): the second‑order nonlinear entry ODE (Eq. 10‑55, a special case of
  Chapman's Eq. 21) and its equilibrium‑glide reference state (Eq. 10‑61, the
  Sänger condition); the oscillation about equilibrium glide is exhibited
  numerically (Fig. 10‑10). Our `δḧ + ω_p²·δh = 0` oscillator is the
  small‑perturbation linearisation of Eq. 10‑55 about Eq. 10‑61. (Read and
  verified pp. 158–162, 172–176; the §7‑2 first‑order solution is steady
  equilibrium glide and contains no oscillator — the linearisation lives in the
  Ch. 10 second‑order theory.)
- D. R. Chapman, *An Approximate Analytical Method for Studying Entry Into
  Planetary Atmospheres*, NACA TN 4276 (1958) / NASA TR R‑11 (1959) — the
  **primary** second‑order nonlinear entry ODE (Eq. 21, in the density‑like
  variable `Z(ū)`); the equilibrium‑glide `Z_II` truncation (Eq. 41,
  `Z_II = (1−ū²)/(ū√(βr)·L/D)`, which reduces to our `ρ_eq = 2β·g_eff/(V²·L/D)`),
  attributed to Sänger; and the lift‑driven transition from non‑oscillatory glide to "numerous
  skips of sizable intensity" (Fig. 6) — the phugoid, shown numerically. Yaroshevskii
  (Vinh Ch. 10) is a special case of this equation. (Read and verified pp. 14, 15,
  21, 22, 24, 25.) H. J. Allen & A. J. Eggers, NACA Report 1381 (1958) — companion
  ballistic/skip grounding (Chapman's `Z_I`/`Z_III` truncations).
- A. E. Gulan, *Conceptual, Trajectory‑Based Structural Sizing Method for
  Hypersonic Glide Vehicles*, M.S. thesis, Georgia Tech, 2024 — SWERVe / C‑HGB
  vehicle dimensions used as the validation glide body.
- Z. Liu, Y. Hu, C. Gao, W. Jing, X. Ji, "Modeling and analysis of maneuver laws
  based on higher order multi‑resolution dynamic mode decomposition for
  hypersonic glide vehicles," *Defence Technology* 48 (2025) 34–47
  (DOI 10.1016/j.dt.2024.12.018) — data‑driven (DMD/Koopman) decomposition of
  HGV skip‑glide; independently measures the skip phugoid frequency
  (0.0207–0.0374 rad/s) corroborating ω_p ≈ 0.034 rad/s, and shows the open‑loop
  altitude mode is mildly unstable (motivating active damping). Not a source for
  the ζ = 0.7 default.
- K. Ogata, *Modern Control Engineering*, 5th ed., Prentice Hall, 2010 — §5‑3:
  desirable damping‑ratio band ζ = 0.4–0.8 (p.171), overshoot Eq. (5‑21), and the
  settling‑time minimum near ζ = 0.68–0.76 (p.173). Source for the ζ = 0.7 default.
- G. F. Franklin, J. D. Powell, A. Emami‑Naeini, *Feedback Control of Dynamic
  Systems*, 8th ed., Pearson, 2019 — §3.4.2 / Fig. 3.24: overshoot Eq. (3.72) and
  ζ = 0.7 → 5 % overshoot as a "frequently used value." Source for the ζ = 0.7
  default.
