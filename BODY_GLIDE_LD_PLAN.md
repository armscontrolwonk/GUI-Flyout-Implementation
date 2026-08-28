# Non-Separating Body Glide L/D — Investigation & Plan

**Status:** ceiling reassessed and locked (no change); trim + mode levers
identified; two follow-ups open. **Date:** 2026-08-28.
**Code touched:** `trajectory.py` (ballistic=no-lift guard), `glider_ld.py`
(comment only), `test_ballistic_no_lift.py`, `test_ld_calibration.py`,
`validation/datcom/compare_datcom.py` (bugfix), `TODO.md` item 9.

---

## 1. How this started

A user reported that **non-separating (body-reenters) vehicles flying a phugoid
or damped-phugoid glide land much farther than seems realistic**, and suspected
the derived lift-to-drag ratio was too high. This document records what the
investigation actually found (which is not what the hypothesis predicted), what
was changed, and a concrete plan for the parts still open.

## 2. TL;DR

1. **The L/D *ceiling* is not too high.** `glider_ld.whole_booster_LD` is
   cross-validated against Digital DATCOM (USAF, AFFDL-TR-79-3032) and sits
   within **5 / 9 / 10 %** at M2 / 3 / 5, on the **conservative** side
   (under-predicts). A trial de-rate would have *broken* that validation
   (§4). No ceiling change was made.
2. **The "too high" impression came from the wrong anchors.** The low
   free-flight numbers are either a **different shape class** (a flared
   projectile, a blunt capsule) or a **different quantity** (the trimmed L/D at
   a low trim angle, not the L/D-max ceiling). See §5.
3. **The real levers on flown range are trim and mode, not the ceiling:**
   - **cg / static margin** → sets the trim angle → sets the L/D the body
     *actually flies at* (`trim_gate` already does this).
   - **glide vs ballistic mode** — a fin-stabilized *drag-driven* body should
     be flown ballistic (now enforced: ballistic = no lift), not as an active
     glider that trims to best-glide α.
4. **Two follow-ups remain** (§7): a possible over-generous control-deflection
   assumption in the trim gate, and a glide-*law* energy-loss audit. Both are
   separate from the L/D build-up.

## 3. What sets a non-separating body's glide L/D

L/D is derived, not entered. Two stages:

**(a) The aerodynamic ceiling — `glider_ld.whole_booster_LD`.** A semi-empirical
component build-up (the theoretical core of Missile DATCOM): body normal force
from Jorgensen (NASA TR R-474) — slender-body potential + Allen–Perkins (NACA
1048) viscous cross-flow — plus Pitts–Nielsen–Kaattari (NACA 1307) wing-body
carryover for fins, with a `sin(2α)/2` high-α correction. It reports
**L/D_max**, the maximum over α, at the best-glide angle.

Decomposition at the best-glide α (finless slender body, M5):

| Contribution | Effect on L/D_max |
|---|---|
| Full build-up | 3.06 |
| Cross-flow term **off** | **1.44** |
| Cross-flow de-rated 50 % (`_ETA` 1.0→0.5) | 2.49 |

So the cross-flow (nonlinear) lift is a large part of the ceiling, but it is
bounded below by the slender-body potential slope (2.0/rad, physically exact for
a blunt-based body — the Munk base-area term) over Cd0.

**(b) The operative cap — `trim_gate`.** The body only *flies* at L/D_max if it
can trim there. `trim_gate` computes cg, cp, static margin, and the trim angle
reachable at full control deflection, then returns `LD_achievable` = **L/D
evaluated at the trim α**:

- unstable (cp ahead of cg) → tumbles → L/D 0 (ballistic);
- stable but control-limited → L/D at the (low) trim α;
- stable + control-rich → reaches best glide → full ceiling.

This is exactly the "evaluate L/D at α_trim (C_m = 0), which is cg-driven" point
that any correct treatment insists on. It is already in the model.

## 4. Evidence: the ceiling is DATCOM-validated (and a de-rate breaks it)

`validation/datcom/compare_datcom.py` parses committed Digital DATCOM output for
the finless slender reference body (D = 0.5 m, L = 4 m, 1.5 m tangent-ogive
nose) and compares L/D_max:

| Mach | `glider_ld` (shipped, η=1) | Digital DATCOM | gap |
|---|---|---|---|
| 2 | 2.13 | 2.23 | −5 % |
| 3 | 2.48 | 2.71 | −9 % |
| 5 | 3.17 | 3.51 | −10 % |

A trial cross-flow de-rate (`_ETA` 1.0 → 0.50) moved these to **−19 / −21 /
−22 %** — i.e. ~⅕ below the industry-standard tool. That is the wrong direction
and was reverted. The regression suite now includes a **DATCOM-agreement guard**
(`test_ld_calibration.py`) that fails loudly if anyone re-introduces such a
de-rate.

## 5. Why the low free-flight anchors don't demand a lower ceiling

| Anchor | Shape | L/D | Why it doesn't lower our ceiling |
|---|---|---|---|
| Intrieri, NASA TM X-569 | Mercury blunt capsule, M5.5 | ~0.38 | Wrong shape class (blunt lifting body). |
| CAN-4 (Dupuis & Edwards DREV TM-9525; Fournier & Dupuis AIAA 96-3399; coeffs in Yates & Chapman AIAA 96-3360) | cone-cylinder-**flare**, L/d 5.84 | ~1.25 | Stubby, high base+flare drag — not a clean slender body. |
| Fin-stabilized slender body (external brief) | slender + small tail fins | ~1 | **Trimmed** L/D at a low trim α — not the ceiling. `trim_gate` produces this. |
| Seiff & Wilkins, NASA TN D-341 | slender ogive-cylinder + **3 large wings** | ~4–6.7 | Winged glider (upper end). Our winged-body case reads ~5.3 — in band. |

The one transferable *physics* caution — Seiff & Wilkins measured the nonlinear
(Newtonian/cross-flow) body lift over-credited near best-glide α (measured C_L
below even the linear term at α = 5.7°, M6) — is real, but a **weak L/D lever**
here (§3a) and would push us further below DATCOM, so it is not acted on at the
ceiling. It is noted in `glider_ld._ETA`'s comment.

Correct high-Mach body method for reference (Vukelich & Jenkins recommend it for
M > 1.2): second-order shock-expansion, Syvertson & Dennis NACA Rept 1328 — held
as the validation method, not a runtime change.

## 6. What was changed

- **Ballistic = no lift, enforced in the EOM** (`trajectory.py`, committed
  separately, `test_ballistic_no_lift.py`). A body whose reentry guidance is
  `ballistic` now produces zero lift regardless of `glider_enabled` /
  `glider_LD`. Previously the lift block relied on `glider_enabled` being off,
  and a body-setup that derived `glider_LD > 0` could leak a hidden skip-glide
  (Scud-class body: 258 → 427 km, +65 %). This is the single largest realism fix
  for a *drag-driven* body mis-flown as a glider.
- **No L/D-ceiling change.** `glider_ld._ETA` stays 1.0; the edit is comment
  only, documenting why it is not a free knob.
- **Regression harness** (`test_ld_calibration.py`): DATCOM-agreement guard +
  Mach plateau + winged anchor + lifting-surface ordering + two trim tests.
- **Validator bugfix**: `whole_missile_LD` → `whole_booster_LD` in
  `compare_datcom.py` (it had been silently broken by a rename).

## 7. Open follow-ups (only if range still looks high with a correct cg)

### 7.1 Trim gate assumes full control deflection to reach best glide
`trim_gate` computes the trim α reachable at **δ = 25° full deflection** and, if
that reaches the best-glide α, grants the full ceiling. That is right for an
**actively maneuvering** HGV/MaRV but over-generous for a body meant to be
**passively stabilized** (which would fly near α ≈ 0, L/D ≈ 0). 
**Plan:** decide whether a body's "active vs passive" intent should be an
explicit switch (or inferred from the reentry mode: glide laws ⇒ active,
ballistic ⇒ passive). If passive, cap the trim α at the cg-offset trim, not the
full-deflection trim. **Validate** against the fin-stabilized-body expectation
(~1) and leave the active-glider path (DATCOM ceiling) unchanged.

### 7.2 Glide-law energy budget (phugoid / skip-glide)
Range at a given L/D is set by how much energy the glide law bleeds per skip. If
the user still sees over-range with a correct cg and an active-glide intent, the
suspect is the **guidance law**, not the aero.
**Plan:** instrument a single damped-phugoid descent — track specific energy vs
range and compare the range achieved to the closed-form equilibrium-glide
estimate for the same L/D (Tracy/Acton). If the numerical law out-ranges the
closed form materially, the skip energy loss is under-modeled. This is a
`trajectory.py` guidance-law question, separate from `glider_ld`.

### 7.3 cg / static margin UX
Since cg is the dominant lever and no paper supplies it, make sure the user can
set it clearly. The burnout-vs-full-tank cg option exists (TODO item 15 / done);
confirm the RO editor surfaces `reentry_cg_m` and the resulting static margin so
a user can see *why* a body reaches (or doesn't reach) best glide.

## 8. Validation checklist for any future change here

- [ ] `python validation/datcom/compare_datcom.py` stays within ~10 %,
      conservative, at M2/3/5.
- [ ] `pytest test_ld_calibration.py` green (DATCOM guard + trim tests).
- [ ] `pytest test_ballistic_no_lift.py` green (no lift leak).
- [ ] Body-glide trajectory suite (`test_depressed_glide.py`,
      `test_glide_regime.py`, `test_pullup.py`) green.

## 9. References (mirror to Drive per `data/REFERENCES.md`)

- Jorgensen, *NASA TR R-474* (1977) — body normal force / cross-flow build-up.
- Allen & Perkins, *NACA Rept. 1048* (1951) — viscous cross-flow.
- Pitts, Nielsen & Kaattari, *NACA Rept. 1307* (1959) — wing-body carryover.
- Gowen & Perkins, *NACA TN 2960* (1953) — cross-flow C_dn vs M_n.
- Digital DATCOM, *AFFDL-TR-79-3032* — cross-validation (in `validation/datcom/`).
- Syvertson & Dennis, *NACA Rept. 1328* (1957) — SOSE (high-Mach body method).
- Vukelich & Jenkins, *AIAA 81-1893R / J. Spacecraft 19(6) 1982* — method
  selection + validity envelope (M ≤ 6, α ≤ 30°).
- Seiff & Wilkins, *NASA TN D-341* (1961) — winged hypersonic glider; nonlinear
  body-lift over-prediction.
- Dupuis & Edwards, *DREV TM-9525* (1996); Fournier & Dupuis, *AIAA 96-3399* —
  CAN-4 flared projectile.
- Yates & Chapman, *AIAA 96-3360* (1996) — CAN-4 coefficients (inverse method).
- Intrieri, *NASA TM X-569* (1961) — Mercury blunt-body (scale floor).
