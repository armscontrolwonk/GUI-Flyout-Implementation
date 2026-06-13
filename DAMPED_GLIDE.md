# Damped‑Phugoid Glide (`damped_glide`)

A glide‑guidance mode for high‑L/D vehicles that reproduces what a *guided*
hypersonic glider actually does: a few **decaying** skip oscillations settling
into equilibrium glide — rather than the two unphysical extremes the tool
already had (instant‑damped analytical arc, or an undamped phugoid forever).

It is a **new fifth mode**; the existing `skip_glide`, `skip_to_equilibrium`,
`equilibrium_glide`, and `equilibrium_glide_acton` are unchanged.

## The control law

The vehicle flies the max‑L/D trim angle α\* (identical to `skip_glide`) plus a
feedback term proportional to the altitude‑rate error, which bleeds energy out
of the phugoid:

```
L·cos σ_cmd = L·cos σ_nom − k_h·(ḣ − ḣ_eq)          (Lu 2013, Eq. 33;
                                                     equiv. Yu & Chen 2011, Eq. 19)
```

- `L_nom` = the α\* lift (β·L/D for the constant‑L/D aero model; q·A·C_L\* for the
  slender‑body polar) — i.e. exactly the `skip_glide` lift.
- `ḣ = V·sin γ` is the current altitude rate; `ḣ_eq = V·γ*` is the command.
- `γ*` = the quasi‑equilibrium‑glide flight‑path angle (Lu Eq. 31, dimensional):
  `γ* = −2·(L/D)·H_ρ·g / (V²·cos σ)` — the small negative descent angle.

## The gain — derived, not fitted (Vinh §7‑2)

Linearising the planar equilibrium‑glide EOM about equilibrium (Vinh, Busemann &
Culp, *Hypersonic and Planetary Entry Flight Mechanics*, 1980, §7‑2; the lift
balance `L = m(g − V²/r)` and first‑order framework, pp. 109–111) gives a simple
harmonic oscillator for the altitude perturbation:

```
δḧ + 2ζω_p·δḣ + ω_p²·δh = 0,    ω_p² = g_eff / H_ρ      (g_eff = g − V²/r)
```

where `H_ρ = −ρ/(dρ/dh)` is the local density scale height. The restoring force
is the density lapse (drop below equilibrium → denser air → more lift → pushed
back up — the mechanism Yu & Chen describe explicitly). Adding the feedback term
contributes the damping coefficient, and matching to `2ζω_p` gives the gain for
a **target damping ratio ζ**:

```
k_h = 2·ζ·m·√(g_eff / H_ρ)
```

`ζ` is the single user knob (`glider_damping_zeta`, default **0.7**). The gain is
computed *each step* from the current state, so it schedules down naturally as V
and g_eff change — matching Lu's velocity‑scheduled gain (his Eq. 34).

### Three independent groundings of the gain (the curated library)

1. **First‑principles (Vinh §7‑2):** the derivation above — `k_h = 2ζm√(g_eff/H_ρ)`.
2. **Empirical (Yu & Chen 2011, Table 1):** their flight‑path‑angle‑feedback gain
   sweep (k_γ = 0→15) shows strong damping at k_γ ≈ 3–5 with large reductions in
   peak heating/q/load for a few‑% range cost. Evaluating our derived gain at
   representative HGV conditions (C_L\*≈0.15, V=6 km/s, g_eff≈4 m/s², H_ρ≈7 km,
   ζ=0.7) lands at k_γ ≈ 3.7 — squarely in their strong‑damping band.
3. **Analytical (Lu 2013):** his nondimensional `k₀≈20` schedule, as a magnitude
   sanity bound.

All three agree on the magnitude, which is why ζ≈0.7 is a defensible default.

## Nesting (the safety property)

`ζ = 0` ⇒ `k_h = 0` ⇒ the feedback term vanishes ⇒ the lift law is **exactly**
`skip_glide`. This is verified bit‑exact (`max|Δaltitude| = 0.000000 km` over a
full integration, for both aero models) in `damped_glide_smoke_test.py`. Large ζ
drives the trajectory onto equilibrium glide. So `damped_glide` continuously
interpolates between the two existing endpoints, with ζ controlling how many
decaying skips occur — the count *emerges* from the damping rather than being a
hand‑set integer (the limitation of `skip_to_equilibrium`, which is retained).

## Relation to the analytic equilibrium modes (Acton / Tracy)

The two analytic equilibrium modes are the limiting cases this mode generalises.
Tracy & Wright (2020) describe the phugoid as *"minor oscillations about the
equilibrium flight altitude… [that] could be damped by active control of the
vehicle"* and then model the steady glide (`equilibrium_glide`, "Tracy"). Acton
(2015) goes further and assumes the oscillation away, *"assuming that the vehicle
does not oscillate during the transition to equilibrium gliding"* — so his
closed‑form pull‑up arc (`equilibrium_glide_acton`) is the **infinitely‑damped
limit** of `damped_glide`. Acton justifies this as design intent (*"the DARPA
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
Body (Gulan, Georgia Tech, 2024; SWERVe is the publicly‑available C‑HGB
predecessor) — on a sub‑circular (MRBM‑class) boost, where it genuinely glides
*in* the atmosphere. Entering at ~5.6 km/s:

| mode | fraction of glide **above 100 km** (no air) | range |
|---|---|---|
| `skip_glide` | 57 % — skips out of the atmosphere | 2445 km |
| `damped_glide` ζ=0 | 57 % — **bit‑identical to skip_glide** | 2445 km |
| `damped_glide` ζ=0.7 | **14 %** — glides in the atmosphere | 6187 km |
| `equilibrium_glide` | 27 % | 6246 km |

Damping at ζ=0.7 converts a skip that spends most of its flight *above* the
atmosphere into a true in‑atmosphere glide (the residual 14 % is the legitimate
ballistic arc to apogee), nearly tripling range to match the analytic
equilibrium glide. (The undamped near‑orbital HTV‑2‑on‑Minotaur built‑in is a
poor glide test — at ~7.6 km/s it really is skip‑entry, reaching 200 + km where
there is no atmosphere to glide on, and no atmosphere‑gated lift law can hold it
down. Use a sub‑circular boost‑glide vehicle to exercise this mode.)

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
- N. X. Vinh, A. Busemann, R. D. Culp, *Hypersonic and Planetary Entry Flight
  Mechanics*, Univ. Michigan Press, 1980 — §7‑2 equilibrium‑glide linearisation
  (phugoid frequency); Ch. 16–17 lift modulation.
- J. M. Acton, "Hypersonic Boost‑Glide Weapons," *Science & Global Security*
  23:191–219, 2015 (DOI 10.1080/08929882.2015.1087242) — non‑oscillatory pull‑up
  assumption (the infinitely‑damped limit of this mode).
- C. L. Tracy, D. Wright, "Modeling the Performance of Hypersonic Boost‑Glide
  Missiles," *Science & Global Security* 28, 2020 (DOI 10.1080/08929882.2020.1864945)
  — equilibrium‑glide formulation; phugoid "damped by active control."
- D. R. Chapman, NACA TN 4276 / NASA TR R‑11 (1958–59); H. J. Allen &
  A. J. Eggers, NACA Report 1381 (1958) — independent phugoid/skip grounding.
- A. E. Gulan, *Conceptual, Trajectory‑Based Structural Sizing Method for
  Hypersonic Glide Vehicles*, M.S. thesis, Georgia Tech, 2024 — SWERVe / C‑HGB
  vehicle dimensions used as the validation glide body.
