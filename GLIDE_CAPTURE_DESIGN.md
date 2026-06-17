# Glide Capture & the Two-Phase Glide Law — Design Memo

**Status:** design record + build log. The γ\* correction (§2) is **committed**;
the diagnostic capture classifier (§5) is **built** (`glide_regime.py`); and the
`damped_glide` rebuild is **built** as a **pure dynamic EOM** law (§8 — the
*decided* approach, which supersedes §4's literature-derived two-phase plan after
verification showed the EOM cannot capture a lofted entry and the user chose
honest dynamic behaviour over analytical-arc capture).

This memo records a long, primary-source-verified investigation. Every equation
below is cited to a page that was read directly (see §7). §1–§7 are the analysis
and literature; **§8 is the implemented result.**

---

## 1. The two regimes a glide model must represent

A boost-glide / re-entry vehicle's post-entry behaviour splits into **two
physically distinct regimes** that the original `damped_glide` conflated:

- **(A) Capture** — getting *onto* the equilibrium glide from the entry state.
  Large-amplitude, nonlinear, and **geometry-/lift-limited**: depending on entry
  angle, speed, ballistic coefficient and L/D, the vehicle either **skips**
  (lofts back out), **captures** (settles into a glide), or **plunges**
  (over-penetrates without ever gliding).
- **(B) Phugoid damping** — damping the small **residual oscillation** once near
  equilibrium. Small-perturbation, linear, governed by ω_p and the damping
  ratio ζ.

All of the prior phugoid work — the ζ knob, ω_p² = g_eff/H_ρ, the ζ ≈ 0.7
default, the Chapman/Yaroshevskii/Vinh-Coppola grounding — is **regime (B)** and
is unaffected. The defects found in this investigation are all in **regime (A)**,
which the code never represented explicitly.

## 2. The γ\* bug (fixed, committed `0d76c80`)

The quasi-equilibrium-glide flight-path angle was computed with L/D in the
**numerator**. The correct form has L/D in the **denominator** — higher L/D
glides *shallower*:

```
γ* = −2·H_ρ·g / (V²·cos σ·(L/D))
```

Verified three ways: **Lu, Forbes & Baldwin (AIAA 2013-4648) Eq. 31** (the cited
source) and its concrete Eq. 37 (γ = −1/(250·L/D)); **Vinh, Coppola &
de-Olivé Ferreira (1996) Eqs. 50–52**; and the classic subsonic glide result
γ = −1/(L/D). It is also self-consistent with this repo's own equilibrium-glide
density and range formulas, which already place L/D in the denominator.

## 3. The "free lift" diagnosis (why the old `damped_glide` was wrong)

The feedback law set the **lift magnitude** (`lift = lift_nom − k_h·(ḣ − V·γ*)`,
bounded only by the 10 g structural cap) but **never coupled drag to that lift**.
Consequence, measured on the C-HGB:

- In **steady glide** the effective L/D is exactly the vehicle L/D (2.0) — the
  damping core is sound.
- In the **entry pull-up** the feedback commanded enormous lift in thin air (e.g.
  ~8800 N where the air can supply ~4 N) with negligible drag → an **effective
  L/D of thousands**. The vehicle bought altitude for free and glided ~20 %
  farther than a true equilibrium glide (effective L/D ≈ 2.4 on an L/D = 2.0
  vehicle — physically impossible).

The "glide" the old mode showed on a lofted boost was therefore a **free-lift
artifact**. The prior validation claim ("ζ ≈ 0.7 matches equilibrium glide to
< 0.5 %") was largely a coincidence of this plus the (then-uncorrected) too-steep
γ\*; it is **stale** and must be regenerated against the corrected model.

A first fix attempt — couple drag to the actual lift and bound lift by the
aerodynamic C_L,max — removed the free lift but revealed the deeper issue: with
honest aero the vehicle **cannot capture** a steep lofted entry and **plunges**
(the polar model) — which is *correct physics* (see §4), not a regression.

## 4. The capture maneuver — the corrected glide law

The entry-corridor literature (§5) prescribes the capture maneuver exactly.
**Chapman (Vinh Ch. 12, p. 223), verbatim:**

> *"a constant high lift-to-drag ratio may lead to a skip trajectory. Hence, the
> constant C_L/C_D program is only maintained until the flight path is essentially
> horizontal, γ ≈ 0, near the point where maximum deceleration is reached. After
> this point the lift-to-drag ratio is modulated to maintain the flight inside the
> atmosphere in order to complete entry in a single pass."*

**Lees, Hartwig & Cohen (1959)** quantify exactly this program for entry from
orbital/escape speed: fly positive lift (constant or modulated) up to peak G,
then **modulate the lift beyond peak G** — only a small, possibly *negative*
(downward) C_L/C_D, bounded by `|C_L/C_D| ≤ 1/(−V/g)` — to hold G ≤ G\* and
**eliminate the skip phase entirely**. (Their objective is controlled descent to
landing, so they then cut lift; a boost-glide vehicle instead transitions to the
sustained equilibrium glide — but the capture maneuver itself is identical, and
they too warn that a constant high L/D skips.)

This dictates a **two-phase** glide law:

1. **Pull-out:** fly the max-L/D trim α\* with **aerodynamically-limited** lift
   (bounded by `q·A·C_L,max`, not the structural cap) and **drag coupled to the
   actual C_L** (induced drag). Arrest the descent toward γ ≈ 0. This phase
   *succeeds* (reaches γ ≈ 0 above the surface) → **capture**; *fails* (over-
   penetrates) → **plunge**. Flying a *constant* high L/D here → **skip**.
2. **Glide:** once near γ ≈ 0 / equilibrium, **modulate the lift to the
   equilibrium trim** (`L·cos σ = m·g⊥`, exactly what `equilibrium_glide`
   already computes) **plus the ζ phugoid damping** (regime B). The modulation
   is what prevents re-skipping.

This **unifies the existing modes**:

| existing mode | role in the two-phase picture |
|---|---|
| `skip_glide` (constant α\*) | the un-modulated pull-out → correctly **skips** |
| `equilibrium_glide` (trim to m·g⊥) | the **modulated glide** phase, without an explicit pull-out |
| `damped_glide` (target) | aero-limited pull-out → equilibrium trim **+ ζ damping**, drag coupled |

So the corrected `damped_glide` = **`equilibrium_glide`'s trim + a physical
aero-limited pull-out + ζ damping of the residual phugoid**, with drag always
following the actual lift. ζ = 0 then nests to "equilibrium glide with an
undamped residual phugoid" (arguably more correct than the old "≡ skip_glide").

## 5. The capture classifier (entry corridor)

Whether the vehicle skips / captures / plunges is the **entry corridor** problem
(Chapman TR R-55; Vinh Ch. 12), bounded by the **overshoot** (skip-out) and
**undershoot** (excessive-deceleration / plunge) boundaries. Two routes:

**Predictive (closed form).** Chapman's dimensionless **periapsis parameter**
(Vinh Eq. 12-10):

```
F_p = (ρ_p·S·C_D / 2m)·√(r_p/β)
```

with the corridor `F_p,ov < F_p ≤ F_p,un` (ballistic parabolic Earth example:
0.06 < F_p ≤ 0.31 at 10 g; corridor width Δh_p = (1/β)·log(F_p,un/F_p,ov)).
Lift widens the corridor (higher L/D pushes the undershoot boundary to much
higher F_p; Figs. 12-12/12-13).

> **Critical caveat.** Chapman's corridor and F_p are built for **supercircular
> entry from space** (Vinh p. 216: valid for V̄_i > 1.05; the overshoot boundary
> is *defined* by the exit speed reaching circular speed — the vehicle leaving
> the atmosphere and returning). Our boost-glide C-HGB is **sub-circular**
> (V̄ ≈ 0.71): it is suborbital and cannot "skip out to space." So Chapman's
> **numerical** boundaries do not transfer to deeply sub-circular boost-glide.
> The framework and physics do; the predictive boundaries for sub-circular
> entry come instead from equilibrium-glide capturability (entry γ vs γ\* + lift
> authority — Lu's QEGC / Vinh-Coppola), and Chapman's corridor applies to the
> near-/super-circular boost-glide cases (e.g. HTV-2-class).

*Future extension (TR R-55 §"Guidance Requirements"):* the corridor width also
sets the **entry-state precision** needed to capture — Chapman converts Δh_p into
allowable (V, γ) errors (a 10-mile Earth corridor ⇒ ≈ ±0.01° flight-path-angle
accuracy at 10 R⊕ for supercircular entry). The boost-glide analog — how
sensitively capture depends on burnout/insertion γ — quantifies the lofted-vs-
shallow-insertion sensitivity that this whole investigation turned on, and could
later feed a "deliverability" check on the boost.

**Diagnostic (exact, speed-agnostic — recommended to build first).** Thrusty
integrates the full trajectory, so it can read the regime off it directly, using
Chapman's own criterion (Vinh p. 209: skip ⇔ Z̄_f = Z̄_i; descend/capture ⇔
Z̄ > Z̄_i; with deceleration setting undershoot):

| verdict | test on the integrated trajectory |
|---|---|
| **skip** | after the first dip, altitude climbs back above the entry interface (ρ returns toward ρ_i) |
| **capture** | ρ grows monotonically toward ρ_eq and a sustained glide is held (ḣ arrested, γ → γ\*) |
| **plunge** | peak deceleration exceeds the structural limit, or steep impact without ever holding a glide |

The classifier's verdict should **gate the glide modes against the selected
mode's intent** — a `skip` is *not* inherently a failure. Skipping is a
legitimate, guided flight regime (Sänger skip-glide for range; deliberate
skip-entry for landing-site access — Tigges et al. 2006), and capture can occur
on a *later* entry after a Kepler coast. So:

- `skip_glide` (the ζ=0 phugoid endpoint) + `skip` verdict → **expected** (the
  point of the mode);
- `equilibrium_glide` / `damped_glide` + `skip` or `plunge` verdict → **mismatch**
  to flag: the vehicle cannot hold the sustained glide the user asked for on this
  entry (wrong insertion geometry / insufficient lift authority).

The verdict reports *trajectory shape*; whether that shape is the desired one
depends on the selected mode.

## 6. Implications for validation

The lofted-boost glide demo rested on the free-lift artifact. With honest aero,
the lofted C-HGB **plunges** — which matches Lu's ballistic-launch case (*"little
the unpowered gliding vehicle can do to reduce this deep dive"*) and the observed
real C-HGB falling deeply into the atmosphere. A correct glide demonstration needs
a **shallow insertion** (Lu's equilibrium-glide-insertion: low burnout altitude,
near-zero flight-path angle). Validation should therefore use **both aero models
side by side**, a **plunge** case (lofted) and a **capture** case (shallow
insertion), regenerated against the rebuilt law.

## 7. Sources (all read and verified; pages noted)

1. **Chapman, D. R.**, *An Approximate Analytical Method for Studying Entry into
   Planetary Atmospheres*, NACA TN 4276 (1958) / NASA TR R-11 (1959). Primary
   second-order entry ODE (Eq. 21); Z_II equilibrium glide (Eq. 41 = Sänger =
   our ρ_eq); skip/descent criterion. Read pp. 14, 15, 21, 22, 24, 25.
2. **Chapman, D. R.**, *An Analysis of the Corridor and Guidance Requirements for
   Supercircular Entry into Planetary Atmospheres*, NASA TR R-55 (1959/1960).
   The entry-corridor formulation (overshoot/undershoot, perigee parameter F_p);
   corridor width ∝ F_p,un/F_p,ov; lift widens the corridor (Earth V̄_i=1.4, 10 g:
   7 mi at L/D=0 → 51 mi at L/D=1 → 65 mi modulated); multi-planet tables;
   cross-validates its undershoot boundary against Lees, Hartwig & Cohen. The
   second half — **guidance requirements** — converts corridor width into
   allowable entry (V, γ) errors (Eqs. 20–22): a 10-mile Earth corridor needs
   ≈ ±0.01° flight-path-angle accuracy at 10 R⊕. **Read directly** (main text:
   abstract, perigee-parameter, corridor-width, guidance-requirements, lifting
   boundaries; appendices not read). Supercircular only (V̄_i ≥ 1.05).
3. **Vinh, N. X., Busemann, A. & Culp, R. D.**, *Hypersonic and Planetary Entry
   Flight Mechanics*, Univ. of Michigan Press (1980). Ch. 7 (first-order
   solutions, pp. 123–126); Ch. 10 (Yaroshevskii, pp. 158–162, 172–176);
   **Ch. 12 (Entry Corridor, pp. 205–225 — read in full)**: F_p (12-10), corridor
   (12-19/12-20), lifting corridor (12-28–12-31), the two-phase capture maneuver
   (p. 223).
4. **Vinh, N. X., Coppola, V. T. & de-Olivé Ferreira, L.**, *Phugoid Motion for
   Grazing-Entry Trajectories at Near-Circular Speeds*, J. Spacecraft & Rockets
   33(2):206–213 (1996). Closed-form damped glide phugoid: frequency
   ω̄² = βr₀(C_L/C_D)² − 1.6358, damping envelope ζ(u) = u^{1/4}(1−u)^{−1/4},
   N = ω̄/4 oscillations (lightly damped → guidance supplies the damping). Read in full.
5. **Lu, P., Forbes, S. & Baldwin, M.**, *Gliding Guidance of High L/D Hypersonic
   Vehicles*, AIAA 2013-4648. γ_QEGC (Eq. 31, the corrected γ\*); altitude-rate
   feedback (Eq. 33); equilibrium-glide-insertion. Read in full; Eq. 31 verified.
6. **Lees, L., Hartwig, F. W. & Cohen, C. B.**, *Use of Aerodynamic Lift During
   Entry Into the Earth's Atmosphere*, ARS Journal 29(9):633–641 (Sept. 1959;
   STL Report GM-TR-0165-00519, 1958). The quantified two-phase capture maneuver:
   positive lift to peak G (Eqs. 8–24), then lift modulation beyond peak G to
   hold G ≤ G\* and eliminate skip (`|C_L/C_D| ≤ 1/(−V/g)`; Figs. 7, 8, 16);
   L/D = 2 widens the 10 g entry-angle limit from < 3° to 9.5° (12.5° modulated).
   Read in full (pp. 633–641).
7. **de-Olivé Ferreira, L., Vinh, N. X. & Greenwood, D. T.**, *Critical Cases of
   Ballistic Entry: New, Guidance-Oriented, Higher-Order Analytic Solutions*,
   J. Spacecraft & Rockets 37(5):630–637 (2000). Chapman-variable framework;
   ballistic critical cases. Read in full.
8. **Tigges, M. A., Crull, T., Rea, J. & Johnson, W.**, *Numerical Skip-Entry
   Guidance*, AAS 06-080 (2006). Orion/CEV lunar-return low-L/D (0.3–0.4)
   deliberate skip-entry: EI → lift-up skip → Kepler coast → second entry →
   Apollo final phase. Confirms the overshoot/undershoot corridor with bank
   modulation, and the supercircular-skip vs sub-circular-plunge energy
   distinction; establishes that **skip is a legitimate guided regime** (range/
   site access), capture occurring on a later entry. Its predictor-corrector,
   target-seeking guidance is beyond Thrusty's open-loop trajectory scope but is
   the reference for any future "guided skip-entry to target" mode. Read in full.

---

## 8. Implementation status (decided & built)

**Decision (user):** `damped_glide` is a **pure dynamic EOM** law — *honest
plunge*, no analytical-arc capture; capturability is **entry-geometry (and
aero-model) dependent**, read off by the diagnostic classifier (§5), not forced.

**Built law** (`trajectory.py`, `damped_glide` branch):

```
g_eff = max(g − V²/r, 0)
L_target = m·g_eff/cosσ  −  k_h·(ḣ − V·γ*)        # equilibrium trim + ζ damping
  polar:        C_L = clip(L_target/(q·A), 0, C_L,max);  L = q·A·C_L
                drag = q·A·(C_D0 + k·C_L²)             # induced drag (coupled)
  constant_LD:  L = clip(L_target, 0, pull-up g cap);  drag = L/(L/D)
```

with `γ* = −2H_ρg/(V²cosσ·(L/D))`, `k_h = 2ζm√(g_eff/H_ρ)`. **No free lift**
(drag follows the actual commanded lift), and lift is bounded by the
*aerodynamic* ceiling (polar C_L,max), not the structural cap.

**Validated behaviour (C-HGB, lofted sub-circular boost; `damped_glide_smoke_test.py`):**

- **polar (physical aero): PLUNGES at every ζ.** The thin-air lift ceiling
  (max lift ∝ q·A·C_L,max) is too small high up to arrest a steep lofted entry —
  independent of L/D (tested to L/D=8) and launch angle. This is the honest
  dynamic result and matches Lu's deep-dive ballistic-launch case and the
  observed real C-HGB falling deeply. *To glide, a glider needs a shallow /
  depressed insertion (Lu's equilibrium-glide-insertion), not a lofted ballistic
  arc — a boost-trajectory capability Thrusty's gravity-turn boost does not
  currently expose.*
- **constant_LD (lumped): captures, ζ-dependent** (range 2550→5606 km as
  ζ=0→2, `glide_frac` 0→0.94). The lumped model has **no aerodynamic lift
  ceiling**, so it can pull out where the physical polar model cannot — it
  **over-predicts capturability**. Polar is the trustworthy aero model for
  capture; constant_LD is a lumped approximation.
- **No free lift:** a captured glide never out-ranges the analytic equilibrium
  glide (effective L/D ≤ vehicle L/D) — the ~20 % overshoot of the old free-lift
  law is gone.
- **ζ = 0** is the equilibrium-trim baseline (no damping); it no longer nests to
  `skip_glide`. `skip_glide` (α* lift, undamped) remains the separate
  phugoid/skip endpoint.
- **No zoom-climb** (the equilibrium trim descends monotonically).

**Why not the α\*-nominal (verified):** flying α\*/max-L/D lift + damping
*cannot capture at any ζ* (it targets a kinematic ḣ unachievable in thin air and
plunges to ζ=30) — so the nominal must be the equilibrium trim.

**Pending:** GUI surfacing of the classifier verdict + mode-gating (judge verdict
vs selected-mode intent, §5); regenerating the headline validation tables; and a
depressed/shaped-insertion boost capability to exercise polar capture.
