# Where the reentry drag polar gets its numbers — and which half is trustworthy

Scope: `trajectory._aero_polar`, `booster_models.lifting_body_sweep`, and the
question of replacing an asserted `glider_LD` with a geometry-derived one.
No vehicle-specific content; the findings below are properties of the two
methods, measured on generic slender shapes.

---

## 1. What the code does today

`_aero_polar` (trajectory.py) builds `C_D = C_D0 + k·C_L²` from **two scalar
user inputs**:

```
C_D0 = m / (beta_kg_m2 · A_ref)        # zero-lift drag, from the entered beta
k    = 1 / (4 · C_D0 · glider_LD²)     # back-solved so (L/D)max IS the input
C_L* = sqrt(C_D0 / k) = 2·C_D0·glider_LD
```

Two consequences follow directly from the algebra:

- The polar attains `glider_LD` at exactly **one** lift coefficient, `C_L*`.
  Everywhere else L/D is lower. Only `skip_glide` and the nominal branch of
  `damped_glide` fly at `C_L*`; `equilibrium_glide*` trims to
  `m·(g − V²/r)/cos σ`, a commanded pull-up trims to the structural g-limit,
  and banking raises the required `C_L` — all of which sit off the peak.
  (`glider_aero_model="constant_LD"` is the exception: there lift is literally
  `drag × L/D` at all times, by design, as the closed-form-range cross-check.)
- The peak always lands at `C_D = 2·C_D0`, as an identity of the construction
  rather than a result.

Note also that the reentry integrator carries **no angle of attack**. The polar
is parameterised in `C_L`; α appears only in boost-phase code
(`_alpha_limited_dir`, `_boost_alpha_aero_force`) and, on the reentry side,
solely as the fixed 25° cap that sets `C_L_max`.

## 2. The over-determination

`beta_kg_m2` is a statement about zero-lift drag. `glider_LD` is a statement
about peak efficiency. They are entered independently, and the back-solve
constructs whatever `k` reconciles them. **The model cannot disagree with the
user** — any pair yields a self-consistent vehicle.

A bottom-up estimator changes this. Integrating a pressure law and a friction
law over the geometry yields `C_L(α)` and `C_D(α)` as a coupled pair, so `C_D0`
*and* `k` both fall out of one shape. β and L/D then stop being independent
inputs; they become two projections of the same curve. Entering both would
over-determine the vehicle, and the schema question becomes which one is data
and which is an override with a stated reason.

## 3. Measured: C_D0 is well conditioned, k is not

Comparing the back-solve against an APAS-style impact method (tangent-cone
body, tangent-wedge fins, Eckert reference-temperature friction, and the
existing `cd_cone_hypersonic` base term) on slender shapes:

- **C_D0 / β agree to roughly 10%.** Zero-lift drag decomposes into terms that
  are separately estimable and individually citable — forebody pressure,
  wetted-area friction, base drag. A representative slender-cone split is
  ~39% pressure / ~37% friction / ~24% base. The residual spread between the
  two routes is dominated by the **friction** assumption (transition state,
  wall-temperature ratio, Reynolds number), not by the pressure law: taking
  `cd_cone_hypersonic`'s internal friction instead of an Eckert value at
  Re_L = 1e7 moves C_D0 by ~16% and β with it. That is an arguable modelling
  choice, not an unknown.

- **k can differ by a factor of ~2.** Drag-due-to-lift is never estimated in
  the back-solve; it is *inferred* from the asserted L/D. When the asserted
  value sits below what the geometry supports, the back-solve absorbs the
  entire deficit into `k`. That is a physically different vehicle from one
  carrying the same deficit as extra parasite drag: same peak, same β,
  different behaviour at every other lift coefficient — and off-peak is where
  most of a maneuvering trajectory lives.

The parabolic **form** is not the problem. Fitted against an impact-method
sweep, `k` holds to ±8% over α = 2–15°, and the `C_D = 2·C_D0` identity at the
peak emerges independently from the bottom-up build-up (~2.0× measured). The
disagreement is calibration, not functional form.

## 4. The pressure law fixes C_L and C_D, not their ratio

Relevant to any proposal to replace the Newtonian estimator. Exact
Taylor–Maccoll surface Cp (NACA 1135 relations, already implemented in
`validate_cone_wave_drag.py`) divided by Newtonian `Cp = 2 sin²θ`:

| cone half-angle | M3 | M6 | M10 |
|---|---|---|---|
| 5°  | 2.21 | 1.44 | 1.24 |
| 10° | 1.45 | 1.19 | 1.11 |
| 20° | 1.22 | 1.09 | 1.06 |

Substantial, and worst where the hypersonic similarity parameter `K = M·θ` is
smallest — which is exactly the regime a small fin at low incidence occupies,
and where the `sin²` law is furthest from the near-linear behaviour of real
thin-surface lift.

But the same factor multiplies windward lift and windward pressure drag, so it
largely cancels in the ratio. Measured on the same panels, changing only the
pressure law:

- **C_L** moves by 10–30%.
- **L/D** moves by ≤1% on a bare cone, ≤6% on a cone with a wing.

So a better pressure law buys real accuracy in `C_L`, in `C_D`, and therefore
in **β** — and essentially nothing in `glider_LD`. If a derived L/D looks
wrong, the pressure law is not the place to look. The terms that move L/D are
the ones that add drag without adding lift: appendages, control-surface trim
deflection, nose bluntness, and the laminar/turbulent state. Since
`(L/D)max ∝ 1/sqrt(C_D0·k)`, doubling parasite drag costs ~30% of the peak.

Corollary for diagnosis: if C_D0 agrees between the two routes and `k` does
not, that is a finding, not just a discrepancy. It localises the deficit to
drag-producing hardware absent from the geometry description, or to a
control-authority limit holding the vehicle off its aerodynamic peak — and
both of those are checkable against open evidence in a way that a bare L/D
number is not. `trim_gate` already expresses the second as
`LD_max` vs `LD_achievable`.

## 5. Validation status — read carefully

The impact method above reproduces the lift curves of the NASA TM 102610
generic winged-cone simulation database to within ~1% for α ≥ 6°, against
0.84–0.90 for Newtonian on the same panels. **This is method reproduction, not
validation.** TM 102610 p. 15 states its own database was generated by
APAS/HABP using "the tangent-cone (fuselage component) and tangent-wedge (wing
and tail components) methods… Prandtl-Meyer expansion… for all shadow
surfaces… viscous shear forces… estimated using the reference enthalpy
method." Agreement therefore confirms a correct implementation of the law and
says nothing about whether the law is right.

The actual accuracy claim is one step further out: Cruz & Wilhite
(AIAA-89-2173) put APAS within 10% of Space Shuttle databook values, with an
~11.8% overprediction of pressure drag against VSL3D. That is the bound any
screening-tier estimator built on this method inherits, and it is the number
to quote — not the 1%.

## 6. Open items

- `_calc_beta` (thrusty.py) routes only `wedge` and `half_cone` body forms to
  `lifting_body_sweep`; an axisymmetric form reaches the β-only dialog and
  never sees the α-sweep estimator, even though `'cone'` is a member of
  `_LIFTING_SWEEP_FORMS`.
- The wing composite in `lifting_body_sweep` is gated to `half_cone`.
- Wing-body carryover and shock-layer interference are absent from both
  routes. NACA 1307 supplies slender-body carryover factors but assumes a
  circular **cylinder**, which a cone frustum is not; the error grows with
  the radius change over the fin chord. Unquantified, and the most likely
  home for a residual in `k`.
- β is a constant in the schema; a Mach-dependent β would need a schema and
  integrator pass, not a data edit. `_beta_of_mach` and `_ld_of_mach` already
  exist for derived no-separation bodies, but the setup gate at
  trajectory.py:2115 requires `glider_LD <= 0` — so an *entered* L/D is
  constant across Mach by construction.

## References

- NACA Report 1135, *Equations, Tables, and Charts for Compressible Flow* —
  θ-β-M relation and the Taylor–Maccoll conical-flow solution.
- Cruz, C. I. & Wilhite, A. W., "Prediction of High-Speed Aerodynamic
  Characteristics Using the Aerodynamic Preliminary Analysis System (APAS),"
  AIAA-89-2173 — accuracy bounds for the tangent-cone/tangent-wedge method.
- Shaughnessy, J. D., Pinckney, S. Z., McMinn, J. D., Cruz, C. I. & Kelley,
  M.-L., *Hypersonic Vehicle Simulation Model: Winged-Cone Configuration*,
  NASA TM 102610, November 1990 — Table I geometry, Fig. 7 lift curves, and
  the p. 15 statement of method.
- Eckert, E. R. G., reference-temperature method — implemented as
  `cf_reference_temperature` in `booster_models.py`.
- Munk 1924; Ashley & Landahl §6-7 — the slender-body polar form cited in the
  `_aero_polar` docstring.
