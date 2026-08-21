# Front-End Redesign: the non-separating reentry body

Design document for making Thrusty's depiction and modeling of a **unitary,
non-separating** missile (V-2, Scud, KN-23 / Iskander, Pershing II MaRV) mutually
consistent. Companion to `BODY_REENTRY_DESIGN.md` (which established
`separation_mode` and the run-level loadout) and `GLIDE_CAPTURE_DESIGN.md`.

Status: **Phases 0–2 implemented** (2026-08-21); Phase 3 deferred (see §6).
Decisions settled with the user 2026-08-21. Governing rule, as everywhere:
derive, don't invent.

---

## 1. The principle this serves

> **The schematic must be accurate because it is how a human user oversees the
> code. If what is drawn does not match what is flown, the human has no way to
> exercise authority over the model.**

This is the whole motivation. The schematic is not decoration and not a
convenience preview — it is the oversight surface. A screening tool whose
picture disagrees with its physics silently launders modeling errors past the
one reviewer (the human) who could catch them. So the redesign's success
criterion is a single invariant, stated in §3, and every phase is measured
against it.

---

## 2. The bug, precisely

Reconstructing the user's KN-23 (single stage ⌀1.1 × 6.7 m; reentry object
"KN23 front end" ⌀1.1 × 2.0 m, Von Kármán, `separation_mode = "body"`) exposes
**three distinct defects that happen to overlap** on this vehicle:

| # | Layer | What happens | Evidence |
|---|-------|--------------|----------|
| **A** | Schematic — fabrication | With no fairing, `draw_booster` ignores the reentry object entirely and draws a generic **1.6 × ⌀ = 1.76 m cone** on top of the 6.7 m stage → an **8.46 m** stack that exists nowhere in the data. | `booster_schematic.py:436` (`nl = 1.6 * nd`) |
| **B** | Schematic — wrong shape | Even the to-scale RO drawn in the corner is rendered by `_reentry_shape()`, which **always draws a straight cone** and ignores `ro.shape`. A Von Kármán RV shows as a sharp triangle. | `booster_schematic.py:258` (`_reentry_shape`, unconditional) |
| **C** | Physics ↔ editor mismatch | For `separation_mode = "body"`, `effective_ro()` **overrides the RO's length with the stage length**: the RO editor shows L = 2.0 m, but the body actually flown is ⌀1.1 × **6.7 m**. The number the user typed is discarded, silently. | `booster_models.py:836–839` |

Defect **C** is the deep one and the reason a plan is needed rather than two
patches. The `fairing_fit` "0.24 m too long" warning is a *fourth* symptom: the
containment check compares the real 2.0 m RO against the fabricated 1.76 m cone
(defect A) — a warning generated entirely from invented geometry.

### Why C exists (and why it is half-right)

`BODY_REENTRY_DESIGN.md` established the correct doctrine: *there is always a
reentry object; a V-2 has a warhead, it just doesn't separate.* For a body-mode
vehicle the reentering object **is** the last stage, so `effective_ro` inherits
the stage's **mass** and **diameter** — which is right, because the body's mass
is the burnout mass and its width is the airframe width. But it *also* inherits
**length**, and that is where the model and the human's input diverge: the user
was given a length field, told it means "the reentry body," and then the code
throws it away in favor of the motor-tube length. The physics then treats the
entire 6.7 m tube as the reentering body (`_boost_front_geometry` returns
`ro.length_m = 6.7`), which is defensible for a tumbling spent stage but wrong
for a shaped MaRV whose actual reentry body is the forward ~2 m.

---

## 3. The invariant

Every phase below is in service of one testable statement:

> **DRAWN ≡ FLOWN.** The geometry rendered by `booster_schematic.draw_booster`
> is the same geometry consumed by the physics through `effective_ro` /
> `_boost_front_geometry` — same overall length, same body diameter, same nose
> shape and nose length, same body form. No element is drawn that the physics
> does not use; no element is flown that the schematic does not show. Where a
> value is a fallback, it is flagged identically in both.

This is enforceable as a test (Phase 1) and is the acceptance gate for the whole
effort.

---

## 4. The design decision: subtractive length

The user chose the **subtractive** model for a unitary body (over the additive
"motor tube + separate nose section" alternative):

> **The last-stage length IS the whole airframe. The nose is the forward taper
> carved out of the top of that length — not an extra section stacked on top.**

For a KN-23 that means: one true length, 6.7 m. The forward (say) 2.0 m is the
shaped ogive/Von-Kármán nose; the aft 4.7 m is the cylindrical motor body. Total
drawn height = 6.7 m, matching the airframe and the flown body. There is no
8.46 m anywhere.

### Why subtractive, not additive

- **A unitary missile has one length.** A V-2/Scud/KN-23 is a single tube with a
  pointed top. "Motor length 6.7 m" already *includes* the ogive on any real
  drawing; asking the user to enter 4.7 m of motor + 2.0 m of nose forces them to
  hand-subtract and invites the exact double-count we are removing.
- **It makes DRAWN ≡ FLOWN natural.** `effective_ro` already inherits the stage
  length as the body length. Subtractive keeps that: the body length is the stage
  length, and the nose length is a *portion* of it, so nothing is overridden or
  discarded — the RO's role narrows honestly to *shape + nose fraction*, not a
  competing length.
- **Additive re-introduces the mismatch.** If the nose were a separate stacked
  length the user sets, the physics would again have to choose between the RO
  length and the stage length — the very fork that produced defect C.

### Consequence for the data model

For `separation_mode = "body"`:

- **Body length** = last-stage `length_m` (authoritative; the airframe).
- **Body diameter** = last-stage `diameter_m` (already inherited; unchanged).
- **Body mass** = last-stage burnout mass (already inherited; unchanged).
- **Nose shape** = `ro.shape` (the RO contributes this — now actually used).
- **Nose length** = the forward taper carved from the body. Sourced, in order:
  (a) an explicit nose-length field if the RO carries one; else (b) a
  shape-appropriate default *fraction* of the body length, **flagged** as a
  fallback. It is bounded to `≤ body length`, so a nose can never exceed the
  airframe (the class of defect A can never recur).
- The RO editor's **length field becomes derived/read-only in body mode** (it
  displays the inherited body length) so the human is never shown an input that
  the code will discard. In `separating_ro` mode the length field stays a live,
  independent input exactly as today — separating RVs are unaffected by all of
  this.

Resolved (user, 2026-08-21): the nose length is a **new stored field**,
`ROParams.body_nose_length_m` — additive-free, separating RVs ignore it, and it
does not overload `nose_radius_m` (which means something else). It defaults to a
flagged shape-appropriate fraction when 0, and is bounded to the body length so a
nose can never exceed the airframe.

---

## 5. Scope boundaries

**In scope:** the non-separating (`body`) unitary missile — how its front end is
stored, flown, and drawn, and the guarantee that those three agree.

**Explicitly out of scope (unchanged by this work):**

- **Separating RVs** (`separating_ro`). Their length is a real independent input;
  the corner-drawn RO already uses its own geometry. Only defect **B** (wrong
  shape in the corner drawing) touches them, and its fix is shape-only.
- **Multi-object loadouts** (N > 1). The bus-face blunt-cylinder nose
  (`_boost_front_geometry`, `_multi`) is a deliberate conservative choice and
  stays. Body mode already pins N = 1.
- **Lifting-body forms** (wedge / half-cone). Their depiction and the "span not
  drawn" honesty are already correct; the invariant test will cover them but no
  behavior changes.
- **The trajectory/aero numbers themselves.** This work makes the *depiction*
  match the *existing* physics and stops `effective_ro` from discarding a user
  input. Where the flown body length changes (a shaped MaRV whose reentry body
  becomes the forward taper rather than the whole tube), that is a physics change
  and is called out as its own phase (Phase 3) with before/after ranges reported,
  not folded in silently.

---

## 6. Phased plan

Each phase is independently shippable, ends green, and is gated on the invariant.
Phases 0–2 are **done**; Phase 3 is deferred until a shaped-MaRV physics change is
actually wanted; Phase 4 is folded into this doc's status.

### Phase 0 — the invariant test (write first, RED) — DONE
`test_front_end_consistency.py`. `draw_booster` returns a `front_end`
dict `{kind, shape, nose_length_m, body_diameter_m}`; the test asserts it equals
what `effective_ro` flies, for the body-mode KN-23 fixture and every library
booster. Red on the pre-fix 8.46 ≠ 6.7 / cone ≠ Von Kármán, green after Phase 1.

### Phase 1 — schematic truth (fix A + B) — DONE
Body mode draws the last stage with its nose carved subtractively from the top
(the airframe length is the total height); the corner reentry object and the
containment check are skipped for a body (nothing contains it); the corner RO and
the stack nose draw the declared analytic profile (`_nose_profile`) instead of an
unconditional cone. KN-23: 8.46 m → 6.7 m, Von Kármán, no phantom "too long".

### Phase 2 — data model + editor — DONE
`ROParams.body_nose_length_m` (JSON + xlsx round-trip); the RO editor gains a
"Body nose length (m)" field active only in body mode, while mass/diameter/length
already grey out there (inherited from the last stage). Legacy files default the
field to 0 → the schematic's flagged fraction.

### Phase 3 — physics reconciliation (deferred, only if body-length semantics change)
Today the *drawing* uses `body_nose_length_m` (the taper), while the *aero*
(`_boost_front_geometry`) still treats the whole airframe as the reference body —
consistent, because the aero consumes only body diameter (for area) and nose
shape (for Cd), not the taper length. If we later decide the flown reentry body
for a shaped MaRV is the forward taper rather than the whole tube, that changes
`_boost_front_geometry`'s returned body length and therefore drag/heating: treat
it as a distinct, measured change — report before/after range and heating for the
KN-23 and the body-mode library entries, keep axisymmetric byte-identity for
everything not in body mode, and pin with tests. Do **not** fold it into the
drawing work above.

### Phase 4 — docs — DONE
This doc's status and §4 resolution updated; the DRAWN ≡ FLOWN invariant and its
test (`test_front_end_consistency.py`) stand as the guarantee.

---

## 7. Acceptance

- The invariant test (Phase 0) is green and runs in CI over every library
  vehicle plus the body-mode fixture.
- The KN-23 draws at 6.7 m with a Von Kármán nose and no "too long" warning.
- No input shown to the user in the RO editor is silently discarded by the code.
- Separating-RV, multi-object, and lifting-body behavior is unchanged (pinned).
- Any change to flown numbers (Phase 3) is reported with before/after, never
  silent.

---

# Part II — Ownership and derivation (the booster-default RO)

Status: **proposed** (decisions settled with the user 2026-08-21; not yet built).
Part I made the *depiction* honest. Part II makes the *authorship* honest: for a
non-separating body the "reentry object" is not an independent thing you design
in isolation — it is a **view onto this booster's front end**, with its geometry
inherited from the airframe and its aero (β, L/D, stability) *emergent from the
whole stack*. The current separate-RO editor invites you to type those emergent
quantities as if they were free inputs, which is the root of every inconsistency
in this document (the CG double-count of §2/the 2fb6cf5 fix was the same disease
in the physics path).

## 8. Ownership — the booster-default reentry object

**Decision (user, 2026-08-21): keep the "there is always a reentry object"
doctrine; a non-separating plan auto-seeds a booster-default RO.** When
`separation_mode = "body"` is selected, the RV is not a blank object the user
must invent — it is seeded as *this booster's front end* and presented as such:

- **Inherited-from-booster fields** (mass, diameter, length) are shown
  **read-only**, labeled "from booster" — meaning **from the last stage's own
  Stage-panel fields** (`diameter_m`, `length_m`, and the burnout mass
  `mass_final`), which the user already enters there. This is NOT a
  fairing-style parallel entry: a fairing is separate, jettisonable hardware
  with its own `shroud_mass_kg`/`shroud_length_m`/`shroud_diameter_m` (additive,
  sits on top), whereas a unitary body's front end adds no mass and no length —
  the airframe *is* the last stage, and the nose is the forward taper carved
  from it (subtractive). So there is no new "front-end mass/diameter/length"
  box to add; `effective_ro` already inherits those, and the editor only
  surfaces them read-only so the inheritance is visible.
- **Front-end fields** (nose shape, `body_nose_length_m`, nose radius, TPS
  material) are editable and labeled **"this airframe's nose"** — they are the
  only geometry a unitary body actually adds, and they live here rather than in
  a phantom detached object.
- **Emergent aero** (β, L/D, static-margin verdict) is **derived, not typed**
  (§9–§10): shown with a live value and an Estimate/preview, `0 = derive`.

Rejected alternative (A): moving the front end into the booster editor's Front
End panel. It reads intuitively for a unitary missile but forks the reentry
model — β/L-D/TPS/attitude would have to migrate onto the booster too — and
undoes the BODY_REENTRY_DESIGN.md consolidation. Option B keeps one home for all
reentry physics and one code path (`effective_ro`), while fixing the *framing* so
the object visibly belongs to the booster.

## 9. The derived-vs-entered matrix

The heart of the user's concern: for a non-separating body, be explicit about
which quantities are inherited, which are genuine front-end inputs, and which are
**derived from the full booster+RO stack** and must never be free inputs.

| Quantity | Non-separating body | Why |
|---|---|---|
| mass, diameter, length | **inherited** (read-only, "from booster") | the airframe IS the last stage (`effective_ro`) |
| nose shape, `body_nose_length_m`, nose radius, TPS material | **entered** ("this airframe's nose") | the only geometry a unitary body adds |
| **β (ballistic coeff.)** | **derived β(Mach)** from the airframe Cd₀ (§10); `0 = derive` | β = m/(Cd·A) of the *whole* body, not a sub-cone |
| **L/D** | **derived** (`glider_ld` + `trim_gate`); `0 = derive` | emergent from nose+body+fins at trim (already built) |
| CG / CP / static margin | **derived** from the full stack (`grid_fin_sizing`) | position depends on the whole mass+area layout |
| boost front-end drag/area | **derived** (`_boost_front_geometry`) | the exposed nose is the front end |
| g-limit, bank, terminal dive, pull-up, ζ | **entered** | the maneuvering *plan*, not geometry |
| `separation_mode`, `reentry_attitude` | **plan-level** (sidebar) | how it is flown |

For a **separating RV** the same matrix flips the top three rows to *entered*:
mass/diameter/length/β/L-D are the RV's own designed properties (it is a real,
detachable object), so nothing here changes for that case.

## 10. Deriving β(Mach) from the airframe

Today β is the one emergent quantity still typed on a body (the design comment
even says "no clean way to derive a single scalar β from a Mach-dependent body Cd
table"). That is exactly right — and the fix is to stop asking for a single
scalar. Mirror what L/D already does: derive a **β(Mach) table**, not a number.

The primitives exist: `glider_ld._body_cd0(last, mach)` (nose + skin-friction +
base Cd₀ of the airframe) and `booster_area(params)` (the front-end reference
area). So for a body with `beta_kg_m2 <= 0`:

```
β_body(M) = effective_ro.mass_kg / ( _body_cd0(last, M) · A_ref )
```

sampled over the same Mach grid the L/D table uses (`trajectory.py` ~2070),
stashed as `params._beta_of_mach` alongside `params._ld_of_mach`, and read by the
drag terms (`trajectory.py:1099/1232/1267/1347`) in place of the scalar
`_ero.beta_kg_m2`. A **scalar fallback** at `GLIDE_MACH_REF` fills the analytic
paths and any non-table read, exactly as L/D does. A separating RV, or any body
with β entered `> 0`, keeps its scalar untouched — `0 = derive` is opt-in and the
legacy default (β = 10000) still means "typed".

**Two β regimes — nose-first vs tumbling (user, 2026-08-21).** The derived β
above is the **nose-first** β: the airframe flying pointed, held there by its
nose and (especially) fins. A body that *cannot* hold that attitude **tumbles** —
a bluff spinning cylinder with a far lower β (heavy drag, near-terminal impact),
which `effective_ro` already derives from `tumbling_cylinder_beta` when
`reentry_attitude == 'tumbling'`. These are physically distinct — a finned
KN-23 strikes fast, a bare spent stage flutters down — and the tool already has
the discriminator: the **trim gate** (which includes the fins in the CP) decides
nose-first-vs-tumbling, and β simply follows it. So the nose-first β(Mach) table
is used only while the body is nose-first; the moment the gate (or the user)
declares tumbling, the table is dropped and the tumbling scalar wins. A separated
nose leaves the spent stage as its own tumbling debris object — a different case,
handled by the debris path, not this table.

Honesty caveats to surface (not hide): a Mach-dependent β makes the "ballistic
coefficient" a curve, so the reports/estimator must show β at the reference Mach
plus its range, and note it is a screening Cd₀ build-up, not a measured value —
the same disclosure standard as the L/D estimate. `result['derived_beta_kg_m2']`
carries the nose-first ref-Mach value (or `None` when not derived / tumbling).

## 11. Phasing (Part II)

Independently shippable, each gated on tests; **not started**.

- **P2-A — β(Mach) derivation.** Add `_beta_of_mach` for a body with β≤0,
  mirroring `_ld_of_mach`; wire the drag reads to prefer it; scalar fallback at
  the ref Mach. Pin byte-identity for separating RVs and for any body with β>0.
- **P2-B — booster-default RO seeding + editor framing.** Auto-seed the RO on
  `body` selection; relabel inherited fields "from booster" (read-only) and
  front-end fields "this airframe's nose"; show β and L/D as derived with an
  Estimate/preview and `0 = derive`. No physics change beyond P2-A.
- **P2-C — the derived-vs-entered guard.** One test asserting the §9 matrix: for
  a body, mass/dia/length/β/L-D are not independently honored (changing the typed
  value while `0`/inherited does not change the flown result); for a separating
  RV they are. This is the standing guarantee that emergent quantities stay
  emergent.
- **P2-D — docs.** METHODS §6.4/§12 + this doc's status.

Open question deferred to build time: the CG estimate is *full-tank* (§ Part I,
`estimate_cg` docstring) but reentry stability wants the *empty/burnout* CG;
with a near-neutral body (the KN-23 came out SM ≈ 0) that difference can flip the
verdict. Worth a burnout-CG option, but it is a separate correctness item from
the ownership/β framing here.
