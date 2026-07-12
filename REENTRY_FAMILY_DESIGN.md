# Reentry integration families: making the numerical/analytic line the identity

Design document for elevating the **numerical-EOM vs. closed-form-analytic**
distinction from a *visual grouping* (shipped) to a *structural boundary* — the
"no switching modes" rule from flight-plan law, applied at the family level
instead of the individual-mode level.  Status: **plan, pre-implementation**.
Companion to `GLIDE_CAPTURE_DESIGN.md` and `BODY_REENTRY_DESIGN.md`.

---

## 1. Motivation

Reentry modes divide by **how the trajectory is integrated**, and that boundary
is a genuine capability fork, not cosmetics:

| | Numerical (EOM) | Closed-form analytic |
|---|---|---|
| Modes | ballistic, skip_glide, damped_glide, dynamic_equilibrium_glide | equilibrium_glide_acton, equilibrium_glide |
| Banking | yes | **no** |
| Dive-at-target | yes | **no** |
| Mach-varying L/D | yes | **no** (formula needs constant L/D) |
| Capturability | honest (lofted entry plunges) | always captures (arc imposed) |
| Character | integrate every step | one formula, deterministic, fast |

Two problems today:

1. **The boundary is leaky and silent.** When an *analytic* plan turns on a bank
   schedule or dive-at-target, `trajectory.py` quietly drops to the numerical
   EOM (`_need_numerical`, `trajectory.py:1933/1985`), plus a numerical bridge
   for degenerate arc geometry (`:1947`).  The user picked "Acton" and got a
   numerical integration — an invisible cross-over.
2. **The editors advertise controls a family can't honour.** The analytic modes
   show ζ, banks, and dive-at-target that either do nothing or force the silent
   fallback; the constant-L/D requirement isn't surfaced.

The conceptual question (raised by the user): *what if the hard "no switching"
rule became just the high-level EOM-vs-analytic question?*  This doc works that
through, including how **ballistic** fits.

## 2. Conceptual model

Three physical buckets, but only **two families** for the UI:

```
Reentry
├─ Ballistic (no lift)            ─┐
├─ skip / damped / dynamic-eq      ├─ NUMERICAL family (EOM integrated)
│                                 ─┘
└─ Acton / Tracy                  ─── ANALYTIC family (closed form)
```

**Ballistic stays inside the numerical family** even though it is conceptually a
third thing ("no glide"): it is numerically integrated (drag + gravity, lift
term off), and turning gliding on/off within a vehicle is a natural in-family
switch (ballistic ↔ dynamic glide).  Making ballistic its own family would give
a degenerate one-item dropdown and forbid the common "what if this RV glided?"
tweak.  So the family split is **binary** (numerical / analytic); the trinary
informs the framing, not the control structure.

**Family = f(mode).**  The family is a pure function of `glider_guidance`
(plus `glider_enabled` for ballistic), so it need not be a new stored field —
it is derived.  Identity is enforced by *scoping the choices*, not by a flag.

## 3. Two levels (the fork to decide)

### Level 1 — Honest feature-gating (lighter)
Keep the single switchable dropdown (shipped grouping), but make the boundary
**honest instead of silent**:
- When an analytic mode is active, the editor **hides/disables** banking,
  dive-at-target, and the Mach-L/D affordance, and shows the constant-L/D note.
- **Delete the silent `_need_numerical` fallback**: an analytic run is purely
  analytic.  (The degenerate-arc numerical *bridge* stays — it is a visual
  continuity fix, not a capability cross-over.)
- Cross-family switching on the strip still allowed (one click).

Delivers: the honesty + the fallback deletion (most of the simplification),
**without** touching plan identity or the comparison workflow.

### Level 2 — Family as plan identity (the "full suite")
Everything in Level 1, plus the "no switching across families" rule:
- **New Reentry Plan picks a family first** (Dynamic / Analytic), then the law
  within it — exactly the flight-plan "name + law" parallel, but the *family*
  is the fixed identity and the *law within it* stays switchable.
- The **strip mode dropdown is family-scoped**: it lists only the current plan's
  family's laws (numerical: Ballistic/skip/damped/dynamic; analytic:
  Acton/Tracy).  Crossing families is impossible from the strip.
- **Cross-family comparison** (the analytic modes' reason to exist) is preserved
  at the *plan* level: keep a "C-HGB dynamic" and a "C-HGB analytic" plan and
  flip between them with the Reentry Plan dropdown (variant machinery already
  supports this).

Delivers: everything in Level 1, plus coherent per-family plans and a simpler
strip — at the cost of one extra plan to A/B across the family line.

**Recommendation:** Level 1 captures ~80% of the simplification (the honesty +
the fallback deletion) with far less ripple and no new restriction.  Level 2 is
the fuller realization of the user's idea and is clean given the variant
machinery, but it trades the one-click cross-family flip for a two-plan setup.
The doc plans **Level 2** (the requested full suite); Level 1 is the fallback if
the added restriction proves annoying.

## 4. Detailed change list (Level 2)

### 4.1 Model layer (`booster_models.py`)
- `glide_family(guidance, enabled) -> {'numerical','analytic'}` — pure derive.
  (Ballistic and the three EOM laws → numerical; Acton/Tracy → analytic.)
- No new stored field.  `_norm_glide_mode` already aliases retired modes.

### 4.2 Strip (`thrusty.py`, Reentry Plan section)
- The mode combobox values are rebuilt on populate from the **active plan's
  family** (derived from its mode): numerical → the 4 numerical labels;
  analytic → the 2 analytic labels.  No group headers needed within a
  family-scoped list.
- `_reentry_plan_kwargs` / `_current_reentry_mode_key` unchanged (they already
  map label→key); they only ever see in-family labels now.
- The "family" is shown as a small read-only caption ("Numerical (EOM)" /
  "Closed-form analytic") above the dropdown, mirroring the flight-plan
  "law fixed when the plan was created" caption.

### 4.3 New Reentry Plan (`_ask_new_reentry_plan_name_and_mode`)
- Add a **family radio** (Dynamic numerical / Closed-form analytic) above the
  mode combobox; the mode list filters to the chosen family.  Seed the family
  from the object's current mode.

### 4.4 Editor (`ReentryPlanDialog`) — family-conditional
- Analytic family: show commanded L/D (constant), β_S (Acton Phase-3), terminal
  dive, attitude, provenance.  **Hide** ζ, bank schedule, dive-at-target
  (with a one-line "closed-form: constant L/D, no banking/target" note).
- Numerical family: show ζ, bank schedule, dive-at-target as today; hide β_S
  unless relevant.

### 4.5 Trajectory (`trajectory.py`) — delete the silent cross-over
- Remove `_bank_active` / `_target_trigger_active` → `_need_numerical` and the
  `if _need_numerical:` numerical-glide branch (`:1985–2003`).  An analytic run
  is purely analytic (arc + `_analytical_equil_glide`).
- Keep the degenerate-arc numerical **bridge** (`:1947`) — continuity, not a
  capability cross-over.
- Guard: an analytic plan should never carry a bank schedule / dive-target after
  4.4, but assert/ignore them defensively if present in legacy data.

### 4.6 Migration
- Legacy plans that named an analytic mode **and** carried banking/dive-target
  were *already running numerically* via the silent fallback.  Migrate them to
  the **numerical** family with the behaviour-preserving equivalent law
  (`dynamic_equilibrium_glide`), so no run changes.  Analytic-without-banking
  stays analytic.  Marker-gated one-shot, mirroring `.dive_default_0`.
- Shipped plans: all already coherent (analytic ones carry no banking).

## 5. How ballistic is affected
- Ballistic remains a **numerical-family** law (glider off).  A ballistic plan's
  strip dropdown offers {Ballistic, skip, damped, dynamic}, so turning gliding
  on/off is a normal in-family switch — no new plan needed.
- Ballistic never touches the analytic path, so the fallback deletion doesn't
  affect it.  It is the numerical family's "no-lift" member, framed apart in the
  docs but not in the control structure.

## 6. Verification
1. Family derive: unit-check `glide_family` over all modes.
2. Strip scoping: a numerical plan lists 4 modes, an analytic plan lists 2;
   cross-family labels never appear.
3. New-plan family filter: choosing Analytic offers only Acton/Tracy.
4. Fallback deletion: an analytic plan with banking in legacy data migrates to
   numerical and reproduces its *previous* (already-numerical) trajectory
   byte-for-byte; a clean analytic plan runs the closed form unchanged.
5. Editor: analytic editor hides ζ/banks/target; numerical shows them.
6. reentry-identity + hybrid + declutter guards; 37 tests; Xvfb GUI smoke.

## 7. Docs
- METHODS §12.3: note the family boundary is now structural (plan identity),
  and the analytic family is purely analytic (no silent numerical fallback).
- README: New-Reentry-Plan family choice; strip is family-scoped.

## 8. Open decisions for the user
1. **Level 1 vs Level 2** (honest-gating vs full identity) — §3.
2. Whether an existing plan should be *convertible* across families in place
   (an explicit "convert to numerical/analytic" action) as an escape hatch, or
   strictly new-plan-only.
3. Ballistic caption: leave it inside the numerical dropdown (recommended) or
   surface a top-level "Glide: none / dynamic / analytic" selector.
