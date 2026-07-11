# Body-Reentry Redesign: separation, attitude, drag, and L/D

Design document for the "B1" separation redesign and the reentry-drag work
that grew out of it.  Status: **approved direction, pre-implementation**.
Companion to `GLIDE_CAPTURE_DESIGN.md` / `DAMPED_GLIDE.md`.

---

## 1. Problems being solved

1. **The phantom object.** A `separation_mode == 'body'` reentry object in the
   library carries a `mass_kg` that is *always* discarded at run time
   (`effective_ro` overrides mass/diameter/length from the last-stage burnout
   state).  The stored number is a lie, editing it does nothing, and the
   object is meaningless without a booster to fuse with.

2. **Two separation flags.** `ro.separation_mode` and `booster.ro_separates`
   both exist and must agree; the GUI referees mismatches with warnings.

3. **The drag graft.** Body mode keeps the aeroshell's β while swapping in the
   stage's mass.  The implied drag area `stage_mass / aeroshell_β` corresponds
   to no physical object.  For an **uncontrolled** (tumbling) spent stage the
   error is qualitative: a tumbling stage has a *low* β (huge mean projected
   area), the opposite of a streamlined RV's β.

4. **The tumble trap.** The geometry L/D estimator reports the trimmed
   aerodynamic *ceiling* with no check that the vehicle can trim there or is
   statically stable.  A marginally-stable body-mode vehicle is silently
   credited with its finned max L/D when it would really tumble (L/D ≈ 0 and
   a different β regime).  This is the one failure mode that is binary, not a
   percentage.

5. **Frozen L/D.** The derived L/D is evaluated once at `GLIDE_MACH_REF = 5`
   and flown as a constant.  Measured swing for the Scud/Hwasong-11 airframe:
   2.28 (M2) → 2.56 (M5), ~12%.  Negligible for aeroballistic *range*
   (sub-1%), but 5–10% for glider range and largest for terminal-phase
   quantities flown at M2–4.

## 2. Decisions (approved)

* **Separation is a run-level choice, not a stored object property.**  The
  Reentry Object sidebar section gains a two-value **Separation** control:
  *"Separates at burnout"* / *"Non-separating — reenters with final stage"*.
  It replaces both `ro.separation_mode` and `booster.ro_separates` as the
  authority.  The RO dropdown keeps naming the aeroshell (front-end TPS,
  emissivity, nose shape, β for the stable case, maneuvering plan); in
  non-separating mode the mass/geometry always come from the last stage and
  the same aeroshell can be A/B'd separating vs. integrated in two clicks.

* **Reentry attitude is a reentry-plan field**: `reentry_attitude` ∈
  `{'trim', 'tumbling'}`.  *Trim* = stable, controlled body (Iskander/MaRV
  class): aeroshell β as given, L/D from geometry (non-sep) or designed value
  (separating).  *Tumbling* = uncontrolled body (spent stage, failed RV):
  L/D = 0 and **derived tumbling β** (below).  Default: `trim` for separating
  objects; for non-separating bodies the static-margin gate (below) suggests.

* **Tumbling β from primary Hoerner data** (Fluid-Dynamic Drag, 1965),
  replacing the flat `Cd = 1.0` for the reentry case:

  ```
  (C_D·A)_eff = ½ [ C_D,broadside · d·L  +  C_D,end · π d²/4 ]
  β_tumble    = m / (C_D·A)_eff
  ```

  Hypersonic coefficients, transcribed from the source:
  - Impact pressure  C_p• = 1.84 − 0.76/M²          (Ch. XVIII eq. 41)
  - Cross-flow cylinder C_D = ⅔·C_p• ≈ **1.2**       (eq. 44, Fig. 24)
  - Blunt cylinder head C_D = 0.89·C_p• ≈ **1.6**    (Fig. 22)

  Continuum anchors (§3-5/§3-6, Figs. 12/28): 2-D cross-flow cylinder
  C_D ≈ 1.17–1.2 subcritical; normal plate 1.98 (2-D) / disc 1.17 (3-D);
  finite-length relief C_D = C_D∞·[1 − k·(d/b)], k ≈ 5.  The existing
  `tumbling_cylinder_beta` (debris arcs, Cd = 1.0 on the two-orientation mean
  area) either gains the two-term form or keeps 1.0 with a documented note;
  decide at implementation — default: **unify on the two-term form** and note
  the change in METHODS.

* **Static-margin gate** for non-separating trim mode: reuse the §8.9
  fin/body static-margin machinery (Barrowman/DATCOM build-up, already used
  for boost-phase stability).  SM ≤ 0 → warn and default the plan's attitude
  to `tumbling`; SM > 0 but `α_trim,max < α_LDmax` → control-limited L/D (the
  curve value at `α_trim,max`), per the scheme already sketched in METHODS
  §12.  The METHODS text describes this gate; implementation must confirm how
  much is wired and finish it.

* **Auto-derived L/D: non-separating only.**  Already wired at integration
  setup (`trajectory.py` ~1531): body mode + `glider_LD ≤ 0` sentinel →
  `derive_glider_LD` at M5.  Scope stays non-sep because that is the only
  case where Thrusty knows the flying geometry (nose + cylinder + fins); a
  separating HGV's L/D is a designed property of an aeroshape Thrusty does
  not store, and the slender-missile build-up would be wrong for it.
  Changes:
  - Shipped body-mode objects (Hwasong-11) migrate to the sentinel (drop the
    explicit 1.0) so the derivation actually runs; `commanded_LD` remains the
    way to fly it worse.
  - Optional **"estimate from geometry"** button for *separating conical*
    objects in the object editor (body-only build-up, no fins) — a sanity
    check that never auto-applies.

* **Mach-varying L/D** (non-sep derived path only): at integration setup,
  precompute L/D_max(M) over M ∈ {1.5, 2, 3, 4, 5, 6, 8, 12} via
  `whole_booster_LD`, interpolate per EOM step on local Mach in the
  **numerical** glide modes; hold the M=1.5 value below M 1.5 (linear wing
  theory invalid).  `commanded_LD` caps the entire curve.  Analytical
  Tracy/Acton modes keep constant L/D (the closed form requires it),
  evaluated at the glide-entry Mach, and say so in their notes.  Separating
  objects with designed scalar L/D are untouched.

## 3. Data model

* `ROParams.separation_mode` — **removed** (after migration).  `_norm_sep_mode`
  retained for reading legacy files.
* `BoosterParams.ro_separates` — becomes derived/default only: seed for the
  new Separation control when a booster is selected; no longer consulted by
  the run path.
* Reentry plan gains `reentry_attitude` (plan key, default `'trim'`);
  `_REENTRY_PLAN_KEYS` updated; ReentryPlanDialog gains the field with the
  gate's suggestion shown.
* GUI run-level state: `_separation_var` in the Reentry Object section;
  persisted as a GUI-only key in the *flight* plan (it is a mission choice of
  the boost/deployment side) — decide final home at implementation; leading
  candidate: flight plan (deployment is an up-leg event).
* Sentinel: `glider_LD = 0` on body-mode objects = "derive from geometry".

## 4. Physics changes

* `effective_ro(params, separation)` — fusion keyed off the run-level control
  instead of the stored field.  Non-sep: mass/diameter/length from last-stage
  burnout (unchanged mechanics), β per attitude:
  - `trim`: aeroshell β (as today).
  - `tumbling`: `β_tumble` from the two-term Hoerner form on stage geometry;
    `glider_enabled = False`.
* L/D(M) table + per-step interpolation in `_eom`'s glide branches (numerical
  modes); plumbed via the effective RO or a session cache, NOT recomputed in
  the hot loop.
* Static-margin gate at integration setup for non-sep trim mode; emits a
  milestone/warning and (if SM ≤ 0) flips the effective attitude to tumbling
  unless the plan explicitly pinned `trim`.

## 5. Migration

* `Hwasong-11.ro.json`: `separation_mode` dropped; `glider_LD` 1.0 → 0
  (sentinel).  Its reentry plan pins nothing new; flying values preserved via
  plan `commanded_LD` if the curator wants the old 1.0 (decide: keep 1.0 as
  commanded_LD in the shipped plan so behaviour is unchanged until the user
  says otherwise).
* Marker-gated one-shot migration for user files (same pattern as
  `.dive_default_0`): strip `separation_mode` into the new control's default,
  preserve explicit `glider_LD` values as plan `commanded_LD` when they were
  below the derived ceiling.
* All regression scripts re-run; the byte-identity guard
  (`reentry_identity.py`) extended with the separation control's states.

## 6. Verification

1. Fusion identity: non-sep selection reproduces today's body-mode
   mass/geometry inheritance exactly (trim attitude, explicit β).
2. Tumbling β: hand-check `β_tumble` for a Scud-class stage against the
   formula; trajectory shows the expected steeper, slower impact vs. trim.
3. Gate: an artificially finless/aft-CG config triggers SM ≤ 0 → tumbling
   suggestion; the shipped finned case passes.
4. L/D(M): table interpolation matches `whole_booster_LD` at the nodes; a
   numerical-mode run with the table differs from constant-M5 in the expected
   direction (~12% L/D swing → few-% range for aeroballistic).
5. Existing suites: 37 tests + law-identity + reentry-variant + run-identity
   scripts stay green.

## 7. Documentation plan

* **METHODS**: new subsection under reentry drag — the two-orientation
  tumbling β with the Hoerner figure-level citations (eq. 41/44, Figs. 22/24,
  §3-5/§3-6 Figs. 12/28); the attitude flag semantics; the L/D(M) table and
  its validity window; the SM gate's outcomes table (already sketched) marked
  as implemented.
* **References**: Hoerner FDD 1965 entry extended with the hypersonic chapter
  cites.  (Plate page images optionally archived under `docs/` — decide with
  the user; the transcription is in this doc regardless.)
* **README**: the separation control and attitude flag in the narrative +
  left-panel sections.

## 8. After this

Next stop (user-directed): **heating** — the aerothermal side picks up the
same fused-body geometry (nose radius from the stage vs. aeroshell) and the
attitude flag (a tumbling body heats very differently from a trimmed one);
carry both into the heating review.
