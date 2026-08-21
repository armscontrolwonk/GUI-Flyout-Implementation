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
