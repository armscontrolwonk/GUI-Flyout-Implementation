# Trim / Control Gate — Approach and Provenance

**Status:** implemented and in use; two items open. **Date:** 2026-09-04.
**Code:** `trim_gate.py`, `glider_ld.py`, `trajectory.py`, `thrusty.py`
(previews), `validation/datcom/compare_datcom.py`, `test_ld_calibration.py`.
**Governing rule, as everywhere here:** derive, don't invent. Where a number
could not be derived or sourced, this document says so rather than dressing it up.

---

## 1. What the gate is for

A non-separating ("body") reentry vehicle has no designed L/D. The gliding
object *is* the airframe, so its lift-to-drag ratio is an emergent geometric
property, derived by `glider_ld.py` from the Jorgensen / Allen-Perkins /
Nielsen-Kaattari-Pitts build-up.

That build-up gives the **aerodynamic ceiling**: the best L/D the shape could
reach *if it flew at its best-glide angle of attack*. Whether it can actually
get there is a different question, and it is the one that decides the flown
answer. A Scud-B-class airframe does not glide, not because its aerodynamics are
poor, but because nothing commands the incidence at which those aerodynamics
would pay.

The gate answers that second question. It is the difference between "what could
this shape do" and "what will this vehicle do".

---

## 2. The defect this replaced

The previous gate linearised the pitching moment:

```
alpha_trim,max = (C_Nd / C_Na,total) * (x_fin - x_CG)/(x_CP - x_CG) * delta_max
```

This assumes a constant normal-force slope and a **fixed** centre of pressure.
Both fail on a slender finned body, and the failure was not marginal.

Because `x_CP` is a normal-force-weighted average of `x_body` and `x_fin`, the
lever `(x_fin - x_CG)/(x_CP - x_CG)` exceeds 1 for **every** stable centre of
gravity. So the trim angle had a hard floor of
`control_eff * (C_Na,fin / C_Na,total) * delta_max`, which for a Scud-B body is
12.8 deg — already past its 11 deg best-glide angle. Sweeping the centre of
gravity across the entire stable range returned the full aerodynamic peak every
single time, with trim angles running from 33 deg to over 600 deg as the static
margin approached zero.

The gate was therefore **structurally incapable** of limiting that vehicle at any
centre of gravity. Its failure mode was to hand back the unconstrained ceiling.
A gate that fails open is not a gate.

---

## 3. The approach

### 3.1 Solve the moment, don't linearise it

The build-up's own normal force is strongly nonlinear in incidence, and its
three terms act at three different stations. So the moment is summed term by
term rather than from a single slope:

```
C_m(a,d)*d = -[ C_N,body(a) * (x_body     - x_CG)     slender-body potential
              + C_N,cross(a)* (x_planform - x_CG)     Allen-Perkins crossflow
              + C_N,fin(a)  * (x_fin      - x_CG)     fin + N-K-P carryover
              + control_eff * C_Na,fin * d * (x_fin - x_CG) ]   commanded control
```

`alpha_trim(d)` is the root, found by bisection **on the build-up's own 1-59 deg
sweep**, so nothing is extrapolated. `glider_ld.cn_components()` supplies the
split, and it is an exact regrouping of the same `C_N` the L/D curve is formed
from — verified to 2e-15 and pinned by a test. There is no second model of the
lift hiding behind the moment.

The physics that makes this work: the crossflow term grows as `sin^2(a)` on the
body **planform**, well aft of the nose centre of pressure, so the centre of
pressure migrates aft and the airframe stiffens with incidence. That is the
restoring moment the fixed-c.p. linearisation could not see, and it is why the
nonlinear solve converges where the linear one ran to 600 deg.

### 3.2 Read control authority from the vehicle

The old gate assumed a 25 deg all-moving control surface on every finned
airframe. Nothing in the repo justified that number, and it is what let a
fin-stabilised ballistic body be credited with a glide.

Authority now comes from the reentry object's existing
`glider_control_surfaces` descriptor, so no field was added and the interface
already edits it. `none` means no commanded deflection at all: the body trims at
zero incidence and does not glide, however good its ceiling. That is the branch
a fin-stabilised missile body belongs in, and the gate previously could not
express it.

### 3.3 Distinguish the ways a body fails to glide

Three outcomes that had been conflated are now separate, because they fly
differently:

| Outcome | Cause | Consequence |
|---|---|---|
| Tumbles | Statically unstable (SM <= 0) | Attitude lost; **beta re-derived** as a tumbling cylinder |
| Ballistic nose-first | Stable, but nothing commands incidence | No lift, but **keeps its aeroshell beta** |
| Not a glide | Control moment beats the restoring moment everywhere | Reported as a non-result, never as unlimited authority |

The middle row matters. An earlier version of this work routed it through the
tumbling branch, which would have handed a fin-stabilised body a
tumbling-cylinder drag it does not have. The two paths differ measurably in
flight (247 km vs 209 km on a Scud-B body), so callers branch on the explicit
`tumbles` flag, never on a zero achievable L/D.

### 3.4 Score the reachable band, not its endpoint

Deflection is commandable, so a vehicle that can reach `alpha_trim,max` can also
hold anything below it. Achievable L/D is therefore the **best** L/D over
`(0, alpha_trim,max]`, not the value at the endpoint. This matters because L/D
is not monotonic in incidence: it peaks at best glide and falls away beyond, so
scoring the endpoint would penalise a vehicle for authority it need not use.

---

## 4. What is derived, what is sourced, what is assumed

This is the part worth being blunt about.

### Derived

- **The moment balance.** No new coefficient. Same normal force as the L/D
  sweep, regrouped exactly, taken about the CG at each term's own station.
- **The planform centroid**, from the same nose-plus-afterbody decomposition
  that already sets the planform area, so the two cannot drift apart.
- **The fin panel centroid**, closed form for a straight-tapered panel; reduces
  exactly to the root mid-chord for an unswept rectangular fin.
- **Control effectiveness** `k_W(B)/K_W(B)`, both NACA 1307 slender-body factors
  on a common normalisation. See §5.

### Sourced

- **Station of the viscous crossflow term** — Simon & Blake, AIAA 99-4258
  (AFRL): "the center of pressure of the body at large angles of attack is
  effectively at the planform centroid", with the two-station moment form (their
  Eq. 6) this gate uses, and the fin's viscous part at the panel centroid.
- **The construction of control effectiveness** — Moore, McInville & Hymer,
  JSR 33(3) 1996, which also confirms the body-alone c.p. is the
  normal-force-weighted sum of stations, i.e. the form used here.
- **Reference accuracy** — Sooy & Schmidt, JSR 42(2) 2005: Digital DATCOM's own
  centre-of-pressure error against wind tunnel is below 2% of body length. This
  is what licenses treating a disagreement with DATCOM as model error.

### Assumed, and flagged as such

- **The deflection tiers** (5 / 15 / 10 deg). Laying tier names onto deflection
  angles is a modelling choice; no document here grades those three words. The
  `unknown` tier is reported as an assumption in the gate's own verdict string
  wherever it produces a glide.
- **The upper tier specifically is now unsourced.** See §6.
- **The 25 deg linear-fin validity threshold**, beyond which a trim solution is
  reported but marked indicative because the fin term is linear wing theory with
  no stall.

---

## 5. Deriving control effectiveness

`C_Nd = control_eff * C_Na,fin`. This was a hard-coded 0.85 with no backing
document. It is now:

```
control_eff = k_W(B) / K_W(B)
```

Both are NACA Rep. 1307 slender-body factors, and both are normalised by the
**same** wing-alone `(C_La)_W` (their Eqs. 5 and 8), so the ratio is exact
within that theory and the wing-alone slope cancels.

**The body carryover cancels rather than being neglected.** The full fin
construction is `[k_W(B) + k_B(W)] / [K_W(B) + K_B(W)]`, matching the carryover
`c_na_fin` already carries. NACA 1307 Eq. (34) gives
`k_B(W) ~= k_W(B) * K_B(W)/K_W(B)`, stated to differ from the exact Eq. (33)
value by no more than 0.01. Substituting, the bracket divides out exactly:

```
[k_W(B) + k_W(B)*K_B(W)/K_W(B)] / [K_W(B) + K_B(W)]  =  k_W(B) / K_W(B)
```

`K_W(B)` was already implemented. `k_W(B)` is `nkp_deflection_factor()`, the
closed form of NACA 1307 **Eq. (19)** in `tau = s/r`.

**The result is a function of geometry, not a constant** — about 1.0 for a
vanishing body down to about 0.52 for a fin nearly buried in one. No constant
could have been right in form. For real tail fins it lands well below the old
value: a Scud-B's fins give 0.66, so 0.85 had been overstating control
authority, in the non-conservative direction.

---

## 6. Validation, and its limits

The transcription of a six-term equation is exactly the kind of thing that fails
silently, so it is checked two independent ways:

1. **Theoretical limits.** `k_W(B) -> 1` as the body vanishes (the fin *is* the
   wing alone), and again as the fin is buried in the body (it simply moves with
   it).
2. **Chart 1 of the same report.** It plots all four factors against `r/s`. The
   computed curve reproduces the distinctive feature — a shallow minimum near
   0.93 at `r/s ~ 0.4` between endpoints of 1.0 — in both **depth and location**.
   A copying error would not survive that.

Both are pinned by tests. The same caution paid off earlier: an inferred column
from a different paper's table read 0.921 where the true factor is exactly 1.0,
so implementing from it would have been wrong.

**The gate's dependency is the moment, so the centre of pressure is checked
directly.** The committed DATCOM output had carried `CM` and `XCP` columns since
it was added, and nothing read them; `compare_datcom.py` now does, using the
convention `x_cp = X_mrc - (CM/CN)*L_ref` (published as Sooy & Schmidt Eq. 4,
and confirmed against DATCOM's own printed column, 29 of 30 rows to 0.002).

| Mach | DATCOM x_cp, a 2->20 deg | model | error |
|---|---|---|---|
| 2 | 1.43 -> 1.91 m | 0.93 -> 1.73 m | -12.4 .. -4.6 % of L |
| 3 | 1.61 -> 1.99 m | 0.93 -> 1.84 m | -16.9 .. -3.7 % of L |
| 5 | 1.72 -> 2.01 m | 0.93 -> 1.75 m | -19.7 .. -6.3 % of L |

Both curves migrate aft with incidence, so the two-station mechanism is the
right shape. But the model's c.p. is **forward of DATCOM everywhere**, by up to
20% of body length, worst at low incidence and high Mach. The cause is
structural: slender-body theory puts the whole potential normal force at the
nose c.p., while DATCOM uses empirical charts and Van Dyke hybrid theory there.

**The direction is non-conservative.** A c.p. too far forward understates the
restoring moment, so a given deflection trims to a higher incidence than it
really would, and the gate over-grants glide. **No correction is applied** —
fitting one would be a calibration with no source behind it. The bias is
measured, pinned so it cannot silently widen, and left open.

**Scope limit, stated plainly:** the committed DATCOM deck is body-alone and
finless. The fin normal-force term and the control-deflection term are validated
against **no data at all**. Only the fin's viscous *station* is sourced.

### 3.5 Answer the CG question with a station

The gate's stability verdict is only useful if the user can act on it. "Set the
CG forward" is not actionable, and the auto CG (a uniform-tube centroid) makes a
warhead-forward missile look unstable, so the failure was easy to hit and hard
to fix.

`cg_targets()` inverts the gate's own static-margin definition, which is exact,
and reports the **neutral point** the CG must sit forward of, the current margin
in metres as well as calibers, and the CG that would buy a requested margin.

It also reports the **trade**, which the old advice hid: moving the CG forward
buys stability but stiffens the airframe, so the same deflection trims to a
smaller incidence and achievable L/D falls. There is a window, not a maximum.
An estimator showing only the stability side would walk a user straight through
it, so both directions are pinned by tests.

Note the other lever, which is easy to miss: the warhead-forward CG only engages
when the reentry object declares a **body nose length**. Left at zero, the
estimate falls back to the uniform tube regardless of the declared payload.

---

## 7. Open items

1. **Centre-of-pressure bias.** The fix is to distribute the potential normal
   force along the body rather than concentrating it at the nose c.p., validated
   by regenerating a DATCOM case — not tuned.
2. **A finned, deflected DATCOM run.** The single highest-value missing
   evidence. It would move the fin and control paths from unvalidated to
   measured, exactly as reading the `CM`/`XCP` columns did for the body.
3. **The upper deflection tier.** Kumar & Stollery was read against the primary
   on 2026-09-04 and does **not** support the 5-15 deg band it had been cited
   for. It is at M 8.2 (not ~10), states no usable-deflection band, and reports
   laminar incipient separation at **7.8 deg** (a = 5 deg) and 8.4 deg
   (a = 10 deg). The old band appears to have mistaken the boundary-layer-state
   sequence (laminar 5 / transitional 15) for a usable limit. So the
   `substantial` tier at 15 deg is unsourced, as is the same figure in
   `damping_estimate.DELTA_MAX_DEG`. Re-anchoring is a physics decision with
   flown consequences and the two modules must move together; separation onset
   is also not zero effectiveness, so 8 deg is a floor on usable travel rather
   than a cap.

---

## 8. Where things live

| Concern | Location |
|---|---|
| Governing equations, outcomes, provenance | `METHODS.md` §8.10 |
| The gate itself | `trim_gate.py` |
| Build-up, component split, the two N-K-P factors | `glider_ld.py` |
| Flight wiring, the `tumbles` branch | `trajectory.py` |
| Centre-of-pressure comparison | `validation/datcom/compare_datcom.py` |
| Behaviour pinned | `test_ld_calibration.py` |
| Citations, with Drive links | `data/REFERENCES.md` |
| Verified deflection entry | `docs/cl_margin_references.md` |
| Open items with context | `TODO.md` item 9 |

A note on the sources: they are not in the repo, because `data/` holds no PDFs
by standing policy — they are in the Drive library, and `data/REFERENCES.md` now
carries a direct link for each. Several code comments previously claimed they
were "in `data/`", which was false; that has been swept.
