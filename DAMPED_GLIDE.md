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

> **Rebuilt as a pure dynamic EOM law** (see `GLIDE_CAPTURE_DESIGN.md` §8 for the
> full rationale and verification). The earlier law used the α\* (skip) lift as
> its nominal with an *uncapped, drag‑decoupled* feedback term — which gave a
> non‑physical "free" pull‑up (effective L/D above the vehicle's). That is fixed.

The nominal is the **equilibrium‑glide trim** `L·cos σ = m·(g − V²/r)` — the
force‑balance command that actually *captures* the glide — plus a feedback term
proportional to the altitude‑rate error that damps the residual phugoid:

```
L·cos σ_cmd = m·(g − V²/r)/cos σ − k_h·(ḣ − ḣ_eq)    (trim: Tracy 2020 Eq. 7;
                                                     damping: Lu 2013 Eq. 33)
```

- `ḣ = V·sin γ` is the current altitude rate; `ḣ_eq = V·γ*` is the command.
- `γ*` = the quasi‑equilibrium‑glide flight‑path angle (Lu Eq. 31, dimensional):
  `γ* = −2·H_ρ·g / (V²·cos σ·(L/D))` — the small negative descent angle
  (L/D in the denominator: higher L/D glides shallower).
- **Lift is bounded by the aerodynamic ceiling and drag is coupled to the actual
  commanded lift** — no free lift. For the slender‑body **polar**: `C_L` is
  capped at `C_L,max` and `C_D = C_D0 + k·C_L²` (induced drag). For the lumped
  **constant_LD** model: `drag = L/(L/D)` (the lumped model has no aerodynamic
  C_L,max ceiling, so it over‑predicts capturability — the polar model is the
  trustworthy one).

An α\*/max‑L/D nominal was tried and **verified unable to capture at any ζ** (it
targets a kinematic ḣ unachievable in thin air and plunges even at ζ=30), so the
nominal must be the equilibrium trim.

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

## Nesting

`ζ = 0` ⇒ `k_h = 0` ⇒ the law reduces to the **equilibrium‑glide trim**
(`L·cos σ = m·g⊥`) with no damping — the baseline this mode damps about. (It no
longer reduces to `skip_glide`; that was the old α\*‑nominal law. `skip_glide`
remains the separate ζ = 0 *undamped‑phugoid* endpoint, selected explicitly.)
Increasing ζ adds damping of the residual phugoid. Whether the vehicle *captures*
a glide at all is set by the entry geometry and the aero model, not by ζ — see
**Validation** and `GLIDE_CAPTURE_DESIGN.md` §8 — and is reported by the
diagnostic glide‑regime classifier (`glide_regime.py`).

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
(`rv_library/C-HGB.rv.json`) — the SWERVe/AHW‑descendant Common Hypersonic Glide
Body (Gulan, Georgia Tech, 2024) — and pins the honest behaviour of the rebuilt
law (5/5 checks). Key results on a **lofted** sub‑circular boost (~5.6 km/s):

- **No free lift:** a captured glide never out‑ranges the analytic equilibrium
  glide (effective L/D ≤ vehicle L/D). The old law glided ~20 % *too far*
  (effective L/D ≈ 2.4 on an L/D = 2 vehicle); that artifact is gone.
- **Honest plunge (polar aero):** the C‑HGB on a *lofted* ballistic boost
  **plunges at every ζ** (verdict `plunge`). The thin‑air lift ceiling
  (max lift ∝ q·A·C_L,max) is too small high up to arrest the steep entry —
  independent of L/D and launch angle. This matches Lu's deep‑dive ballistic
  launch and the observed real C‑HGB falling deeply. **A glider must be inserted
  shallow** (depressed/equilibrium‑glide insertion), not lofted, to glide.
- **Damping improves capture (constant_LD):** the lumped model has no aero lift
  ceiling, so it *can* pull out; range grows with ζ (2550 → 5606 km, ζ = 0 → 2)
  and captures at high ζ. It **over‑predicts capturability** — the polar model
  is the trustworthy one.
- **ζ = 0** is the equilibrium‑trim baseline; **no zoom‑climb**.

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
