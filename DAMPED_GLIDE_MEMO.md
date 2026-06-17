# MEMORANDUM — Damped‑phugoid glide guidance

**Date:** 13 June 2026
**Re:** Modeling approach for the `damped_glide` reentry mode
**Vehicle class:** High‑L/D hypersonic glide bodies (C‑HGB / SWERVe lineage)

---

## 1. Problem

A boost‑glide vehicle does not settle into a steady glide the instant it
re‑enters. Lift initially exceeds the local weight‑minus‑centrifugal force, so
the vehicle **pulls up**, overshoots its equilibrium altitude, and oscillates —
the **phugoid** (or "skip") mode. A *guided* glider damps this oscillation and
settles into equilibrium glide after a few skips.

The reentry mode is the modeling choice for **how strongly that phugoid is
damped**. The four primary modes form a single physical spectrum in the damping
ratio ζ:

| Reentry mode | Phugoid behavior | Damping |
|---|---|---|
| Ballistic | no lift, no glide | — |
| Phugoid / skip‑glide | undamped — oscillates indefinitely (skips out of the atmosphere for energetic entries) | ζ = 0 |
| **Damped phugoid glide** | a pull‑up plus a few decaying skips into equilibrium | ζ ≈ 0.7 |
| Non‑oscillatory glide (Acton) | analytic capture, no oscillation | ζ → ∞ (limit) |

`damped_glide` fills the physical middle of this spectrum with a tunable decay
rate; Acton's analytic model (`equilibrium_glide_acton`) is its no‑oscillation
endpoint (§6). Two further modes — `equilibrium_glide` (Tracy) and
`skip_to_equilibrium` — remain available for comparison.

## 2. The oscillation being damped

Equilibrium glide balances lift against the net normal force (Tracy & Wright
2020, Eq. 7):

    L = m (g − V²/r) ≡ m·g_eff

Linearizing about this equilibrium gives, from first principles, a simple
harmonic oscillator in the altitude perturbation δh, restored by the **density
lapse**. At fixed angle of attack lift ∝ ρ ∝ e^(−h/H_ρ), so a displacement δh
changes the lift acceleration by d a_L/dh = a_L·(d ln ρ/dh) = −g_eff/H_ρ — dip
below the equilibrium altitude, the air is denser, lift rises, and the vehicle
is pushed back up:

    δḧ + ω_p²·δh = 0 ,     ω_p² = g_eff / H_ρ

with H_ρ = −ρ/(dρ/dh) the local density scale height. Typically
ω_p ≈ 0.034 rad/s — a skip period of order 180 s.

This linearization is the second-order entry theory in closed form. The primary
source is **Chapman** (NACA TN 4276 / NASA TR R-11, 1958–59), who reduces the two
planar motion equations to a single second-order nonlinear ODE (Eq. 21) in a
density-like variable `Z(ū)` (`ū = V/V_circ`, `Z ∝ ρ`):

    ū·d/dū(Z′ − Z/ū) − (1−ū²)cos⁴φ/(ūZ) + √(βr)·(L/D)·cos³φ = 0   (Chapman Eq. 21)

Its truncation neglecting vertical acceleration is the equilibrium glide —
Chapman's `Z_II` solution (Eq. 41), "equilibrium gliding flight originally
discussed by Sänger" — the same Sänger root as the first-order relation in
Vinh §7-2:

    Z_II = (1 − ū²) / (ū·√(βr)·(L/D))           (Chapman Eq. 41, ū = V/V_circ)

With Chapman's Z ∝ ρ (his Eq. 24a) and 1 − ū² = g_eff/g, this reduces exactly to
the equilibrium-glide density relation the tool uses:

    ρ_eq = (2m/C_D A)·g_eff/(V²·L/D) = 2β·g_eff/(V²·L/D)

so the primary source confirms our reference state algebraically, not just by
attribution. The *full* equation produces, for higher L/D, the oscillation
Chapman calls "numerous skips of sizable intensity" (Fig. 6). **Yaroshevskii's equation** (Vinh, Busemann
& Culp, Ch. 10, Eq. 10-55, `y″ = −K + (e^{2x}−1)/y`, `y ∝ ρ`, `K = √(βr₀)·C_L/C_D`)
is a special case of Chapman's, whose quasi-steady solution is the same Sänger
equilibrium glide (Eq. 10-61) and which shows the same oscillation numerically
(Fig. 10-10). Linearizing the second-order ODE about equilibrium glide gives
δ″ + k²δ = 0 with k² > 0 — precisely the oscillator above. **All three sources
(Chapman, Yaroshevskii, Vinh) exhibit the oscillation but none writes the
closed-form oscillator** — that one-line step is taken here. (Vinh §7-2/§7-5 are
the *first-order* steady-glide/skip solutions and contain no oscillator. Read and
verified: Chapman pp. 14, 15, 21, 22, 24, 25; Vinh pp. 158-162, 172-176.)

This analytical frequency is corroborated empirically by Liu et al. (2025), who
decompose CAV-H skip-glide trajectories with a higher-order multi-resolution
dynamic mode decomposition (HMDMD) and measure the skip oscillation's dominant
frequency directly from the data: **0.0207–0.0374 rad/s** across entry speeds of
3000–5000 m/s (their Table 9), bracketing the ω_p ≈ 0.034 rad/s used here. Their
data also show the frequency *falling* as entry speed rises — exactly the
g_eff = g − V²/r dependence of ω_p (faster ⇒ smaller g_eff ⇒ lower ω_p).

A caveat worth recording: the linearization above treats the phugoid as a
*neutrally stable* oscillator (ω_p² > 0, no growth term) that guidance then
damps. Liu et al. find the open-loop altitude mode is in fact mildly
**unstable** — their DMD eigenvalues lie outside the unit circle (|λ| up to
1.098; "aperiodic and unstable," their Table 8 / Fig. 25). The open-loop skip
therefore tends to *grow*, not merely persist, which only strengthens the case
that a guided glider must actively damp it (§3) — the feedback is removing a
real (slightly negative-damped) mode, not just shaping a neutral one.

## 3. The pull‑up is the first half‑cycle — and the damping ratio shapes it

The initial pull‑up is **not a separate event**: it is the leading half‑cycle of
this oscillation. The vehicle arrives with γ < γ\* (descending faster than
equilibrium), lift exceeds m·g_eff, and the trajectory arcs upward. What happens
*next* is set entirely by the damping:

- **ζ = 0** — the pull‑up overshoots fully and the vehicle skips back out
  (undamped; this is exactly `skip_glide`).
- **0 < ζ < 1** — the pull‑up overshoots modestly, then a few decaying skips
  settle onto the glide. The realistic guided pull‑up.
- **ζ ≥ 1** — the pull‑up is arrested with no overshoot and merges directly into
  equilibrium glide.

So in this model **you "fly" the pull‑up by choosing the damping ratio.** The
*magnitude* of the pull‑up is separately bounded by the structural limit
`glider_pullup_g_max` (lift is capped at that load); ζ governs its *character* —
how hard it captures, and whether it overshoots into skips.

## 4. Control law

We fly the maximum‑L/D trim angle α\* (identical to `skip_glide`) and add
feedback on the **altitude‑rate error**, after Lu, Forbes & Baldwin (Eq. 33);
equivalently the angle‑of‑attack feedback of Yu & Chen (Eq. 19):

    (L cos σ)_cmd = (L cos σ)_nom − k_h·(ḣ − ḣ_eq)

with ḣ = V·sin γ, the commanded altitude rate ḣ_eq = V·γ\*, and the
quasi‑equilibrium flight‑path angle (Lu, Eq. 31)

    γ\* = − 2·(L/D)·H_ρ·g / (V²·cos σ).

## 5. The gain, and why ζ = 0.7

Inserting the feedback turns the oscillator into a damped one, and matching the
damping coefficient fixes the gain for a chosen damping ratio:

    δḧ + 2ζω_p·δḣ + ω_p²·δh = 0    ⟹    k_h = 2·ζ·m·√(g_eff / H_ρ)

ζ is the single user knob, and k_h is recomputed each integration step from the
current state, so the gain schedules down with V and g_eff (consistent with Lu's
velocity‑scheduled gain).

**Why 0.7 by default.** ζ ≈ 0.7 is the classical second‑order control default:
it sits in the middle of the damping‑ratio band (**ζ = 0.4–0.8**) that the
standard control texts recommend for a *desirable* transient response — below
0.4 gives excessive overshoot, above 0.8 responds sluggishly (Ogata, *Modern
Control Engineering*, 5th ed., §5‑3, p.171; Franklin, Powell & Emami‑Naeini,
*Feedback Control of Dynamic Systems*, 8th ed., §3.4.2 / Fig. 3.24, which lists
ζ = 0.7 → ~5 % overshoot as a "frequently used value"). Physically it is the
lightest damping that still reads as a *guided capture* rather than a skip: the
first overshoot is

    M_p = exp(−πζ/√(1−ζ²)) ≈ 4.3 % at ζ = 1/√2  (~5 % at ζ = 0.7)
                                        (Ogata Eq. 5‑21; Franklin Eq. 3.72)

i.e. the vehicle overshoots the equilibrium altitude by only a few %, then
settles within roughly one damped period (t_s ≈ 4/ζω_p ≈ 170 s) — **one
pronounced pull‑up plus one or two shallow, decaying skips.** ζ ≈ 0.7 is also
very nearly the *settling‑time‑minimizing* damping: Ogata (p.173) finds t_s
bottoms out around ζ = 0.68 (5 % criterion) / 0.76 (2 % criterion), so 0.7 is the
fastest settling without a sluggish approach. (Note this is the *time‑domain*
justification; ζ = 1/√2 is also the "maximally flat" value in the *frequency*
domain — the 2nd‑order Butterworth response — but that is a separate Butterworth
characterization, not what the transient‑response texts above argue.) Importantly,
**0.7 is a modeling choice describing a competently‑guided vehicle, not a physical
constant of the airframe** — it is freely dialed: ~0.3 gives several lazy skips,
≥ 1.0 collapses to a smooth equilibrium capture.

## 6. Relation to Acton and Tracy

Both *Science & Global Security* treatments frame the phugoid as a real effect
suppressed by guidance. Tracy & Wright describe it directly —

> "Minor oscillations about the equilibrium flight altitude, called phugoid
> motion, result from the dynamics of this process. These could be damped by
> active control of the vehicle."

— and then model the steady equilibrium glide (the tool's `equilibrium_glide`,
"Tracy"). Acton goes further and assumes the oscillation away during the
transition, in his words *"assuming that the vehicle does not oscillate during
the transition to equilibrium gliding."* `damped_glide` is the mode that
actually represents the "active control" both papers invoke.

Acton's assumption is a deliberate judgment about how HTV‑2 was *designed*:

> "the designers of a practical boost‑glide vehicle would want [oscillations] to
> be as highly damped as possible… if they have designed the vehicle for
> equilibrium gliding rather than skip gliding, then it is to avoid precisely
> this kind of transient behavior,"

anchored by the observation that *"the DARPA schematic shows the glider bouncing
just once during the pull‑up,"* and damped by a gradual reorientation that holds
ρ/β constant (his Eq. 8). His Phase 4 then bleeds velocity through the turn,
dV/V = −(D/L)·dγ, depositing the glider directly onto the equilibrium curve of
Phase 5 (the tool's `equilibrium_glide_acton`, which prepends the
blunt‑orientation direct‑entry phase, βS). Because no overshoot is permitted,
**Acton's construction is the infinitely‑damped (ζ → ∞) limit of the present
model — the no‑oscillation endpoint of the §1 spectrum**, which is why it is now
a primary reentry mode ("Non‑oscillatory glide (Acton)") rather than a legacy
one.

Acton himself flags where that idealization fails — a ~10× discrepancy in βS:

> "This failure is probably associated with the simplification of assuming that
> there is no oscillatory behavior during the pull‑up. If the model permitted
> such behavior… more of the glider's speed would be lost during the pull‑up."

That is, in effect, a description of `damped_glide`: a finitely‑damped pull‑up
bleeds the extra energy Acton's closed form misses.

## 7. When to use Acton vs. damped phugoid

Flying the C‑HGB through both at sub‑circular entry:

| mode | range | bounces | matches |
|---|---|---|---|
| `equilibrium_glide_acton` | 6159 km | 0 | Acton's published method |
| `damped_glide` ζ = 0.7 | 6187 km | **1** | Acton's range (<0.5 %) **and** his "bouncing just once" |
| `damped_glide` ζ = 1.0 | 6392 km | 1 | Acton's range to ~4 % |
| `damped_glide` ζ = 2.0 | 6965 km | 0 | no oscillation, but range diverges |

No single ζ *bit*‑reproduces the Acton mode — they are different models (his
analytic βS blunt‑entry plus ρ/β‑matched rotation, vs. the dynamic lift feedback
here). But ζ ≈ 1 is the natural "Acton's assumption, integrated" setting
(fastest non‑oscillatory capture), reproducing his range to a few percent; and
ζ ≈ 0.7 reproduces his range almost exactly *while showing the single bounce he
attributes to the real vehicle*. Note that forcing zero oscillation with ζ ≳ 2
**overshoots** Acton's range — "more damping = more Acton‑like" is false past
ζ ≈ 1, because the mechanisms differ.

**Prefer Acton (`equilibrium_glide_acton`) when:**

1. **It is the reference method.** Reproducing Acton's published HTV‑2 figures,
   auditing other SGS‑style analyses, or producing results directly comparable
   to that literature. The closed form is, in his words, "easy to scrutinize";
   a reviewer can check the algebra but cannot independently verify a chosen ζ.
2. **Guidance quality is unknown — the usual case for an adversary vehicle.**
   Acton needs no control‑system data and adds no free parameter beyond one
   clearly‑stated, design‑justified assumption. Choosing ζ for a black‑box
   foreign system asserts knowledge you do not have; Acton's parsimony is the
   more defensible posture when all you have is β and L/D. (And since ζ = 0.7
   reproduces his range, little is gained by adding the knob.)
3. **You want analytic insight or bulletproof speed.** The closed form yields
   explicit scaling (range vs. L/D, his Eq. 13) and is instant and robust for
   large sweeps, Monte Carlo, or optimization.
4. **The blunt‑orientation direct‑entry (βS) phase matters.** `damped_glide`
   flies the glide trim throughout and omits that high‑drag deceleration phase.

**Prefer `damped_glide` when:**

- The pull‑up **transient itself** is the object of study — speed/energy lost in
  the bounce (which Acton says his model under‑predicts), heating during
  capture, or the in‑atmosphere corridor.
- The vehicle is genuinely under‑damped/skipping (Acton cannot represent this).
- You want to **vary** guidance quality or run a sensitivity analysis on ζ,
  rather than assume perfect capture.

**One‑line guidance:** prefer Acton when you are *citing a method* — a
parsimonious, auditable, parameter‑free range estimate that matches the
published reference; prefer damped phugoid when you are *modeling a maneuver* and
the realism of the pull‑up transient is what you are after. For range alone the
two barely differ.

## 8. Safety property

ζ = 0 ⟹ k_h = 0 ⟹ the law is **bit‑for‑bit identical** to `skip_glide`
(verified to max|Δaltitude| = 0 over a full trajectory, for both aero models).
Feedback is bounded by the structural g‑limit, disabled above the 100 km
re‑entry interface, and disabled when g_eff ≤ 0 (the phugoid restoring force is
undefined at/above circular speed).

## 9. Validation

On the C‑HGB glide body (SWERVe/AHW descendant; Gulan 2024) at sub‑circular entry
(~5.6 km/s), the undamped skip spends **57 %** of its glide *above* 100 km —
outside the atmosphere, where it cannot glide. At ζ = 0.7 that falls to **14 %**
(the remainder being the legitimate ballistic arc to apogee), and range rises
**~2.5×** (2445 → 6187 km) to match the analytic equilibrium glide. See
`damped_glide_smoke_test.py`.

## References

1. P. Lu, S. Forbes, M. Baldwin, *Gliding Guidance of High‑L/D Hypersonic
   Vehicles*, AIAA 2013‑4648 — feedback law (Eq. 33), gain schedule (Eq. 34),
   command flight‑path angle (Eq. 31).
2. W. Yu, W. Chen, *Guidance Scheme for Glide Range Maximization of a Hypersonic
   Vehicle*, AIAA 2011‑6714 — angle‑of‑attack feedback (Eq. 19); empirical
   gain/heating/range sweep.
3. J. M. Acton, "Hypersonic Boost‑Glide Weapons," *Science & Global Security*
   23:191–219, 2015 (DOI 10.1080/08929882.2015.1087242) — multi‑phase analytical
   trajectory; non‑oscillatory pull‑up assumption (Phase 4; Appendix A); ρ/β‑match
   (Eq. 8); HTV‑2 L/D ≈ 2.6 fit (Table 3).
4. C. L. Tracy, D. Wright, "Modeling the Performance of Hypersonic Boost‑Glide
   Missiles," *Science & Global Security* 28, 2020
   (DOI 10.1080/08929882.2020.1864945) — equilibrium‑glide formulation (lift
   balance L = m·g_eff, Eq. 7); phugoid "damped by active control."
5. A. E. Gulan, *Conceptual, Trajectory‑Based Structural Sizing Method for
   Hypersonic Glide Vehicles*, M.S. thesis, Georgia Tech, 2024 — SWERVe / C‑HGB
   dimensions (validation body).
6. Z. Liu, Y. Hu, C. Gao, W. Jing, X. Ji, "Modeling and analysis of maneuver
   laws based on higher order multi‑resolution dynamic mode decomposition for
   hypersonic glide vehicles," *Defence Technology* 48 (2025) 34–47
   (DOI 10.1016/j.dt.2024.12.018) — data‑driven (DMD/Koopman) decomposition of
   HGV skip‑glide trajectories. Not a guidance/damping law, but independently
   measures the skip phugoid frequency (0.0207–0.0374 rad/s, their Table 9 —
   corroborating ω_p ≈ 0.034 rad/s here) and finds the open‑loop altitude mode
   mildly unstable (|λ| up to 1.098, their Table 8 / Fig. 25). CAV‑H validation
   body: 907 kg, 0.48 m², 50 km / 5.5 km/s.
7. K. Ogata, *Modern Control Engineering*, 5th ed., Prentice Hall, 2010 — §5‑3
   "Second‑Order Systems": standard form Eq. (5‑10); max‑overshoot Eq. (5‑21)
   M_p = e^(−(ζ/√(1−ζ²))π); the desirable damping‑ratio band ζ = 0.4–0.8 (p.171);
   settling‑time minimum near ζ = 0.68–0.76 (p.173). Source for the ζ ≈ 0.7
   default.
8. G. F. Franklin, J. D. Powell, A. Emami‑Naeini, *Feedback Control of Dynamic
   Systems*, 8th ed. (Global), Pearson, 2019 — §3.4.2 "Overshoot and Peak Time":
   overshoot Eq. (3.72) M_p = e^(−πζ/√(1−ζ²)); Fig. 3.24 lists ζ = 0.7 → 5 %
   overshoot as a "frequently used value." Source for the ζ ≈ 0.7 default.
9. N. X. Vinh, A. Busemann, R. D. Culp, *Hypersonic and Planetary Entry Flight
   Mechanics*, University of Michigan Press, 1980 — Ch. 10 (Yaroshevskii's
   theory): the second‑order nonlinear entry ODE (Eq. 10‑55) and its
   equilibrium‑glide reference state (Eq. 10‑61, the Sänger condition); the
   oscillation about equilibrium glide is shown numerically (Fig. 10‑10). The
   `δḧ + ω_p²·δh = 0` oscillator (§2) is the small‑perturbation linearization of
   Eq. 10‑55 about Eq. 10‑61. (§7‑2 / §7‑5 are first‑order steady‑glide / skip
   solutions with no oscillator; read and verified pp. 158‑162, 172‑176.)
10. D. R. Chapman, *An Approximate Analytical Method for Studying Entry Into
    Planetary Atmospheres*, NACA TN 4276 (1958) / NASA TR R‑11 (1959) — the
    **primary** second‑order nonlinear entry ODE (Eq. 21, in the density‑like
    variable `Z(ū)`, `ū = V/V_circ`); the equilibrium‑glide `Z_II` truncation
    attributed to Sänger; and the lift‑driven transition from non‑oscillatory
    glide to "numerous skips of sizable intensity" (Fig. 6). Yaroshevskii (ref. 9)
    is a special case of this equation; the §2 oscillator is its linearization.
    (Read and verified pp. 14, 15, 21, 22, 24, 25.)
