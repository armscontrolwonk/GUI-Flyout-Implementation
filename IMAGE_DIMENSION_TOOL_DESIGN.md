# Image Dimensioning Tool — Design

Status: **Phase A1 shipped** (2026-07-30; RO-editor slice — see Phasing).
Load a picture of a
vehicle, declare what it is, and be walked through clicking its dimensions
off the image so that every measurement lands in an existing editor field —
the Schematic run backwards (image → data instead of data → picture).

Governing rule, applied at the click level:

> **Topology is declared.  Geometry is measured.  Symmetry is a stated
> convention.  Anything the view can't show is flagged, never guessed.**

The image proposes; the human commits.  A click fills a field the user can
edit; the image never silently writes data.

## Decisions (recorded 2026-07-30)

1. **No persistence.**  The image, scale, and click points are session-only —
   nothing binary is saved with the vehicle or the repo (the PDF-bloat lesson).
   What DOES survive is a text provenance stamp auto-appended to the object's
   notes on save (e.g. "D, L from image, side view, scale from claimed
   L=10.2 m, 1 px = 3.4 cm, 2026-07-30") — the audit trail without the bytes.
2. **Prompted drawing.**  The tool walks a checklist generated from the
   declared topology ("click the two ends of stage 1", "click the fin root
   chord"…), with skip allowed on every step.  No free-measure palette in
   Phase A (an expert escape hatch can come later if wanted).
3. **Populate first, audit second.**  The goal is filling dimensional fields;
   the schematic overlay is a supporting live cross-check, not the product.

## Architecture

No new data path.  The tool is an image pane + prompt engine attached to the
EXISTING stage and reentry-object editors: every accepted measurement lands in
a field those editors already own, inheriting their validation, flags, and
round-trip.  The overlay draws the current data through the existing schematic
renderer at the image scale.

Core loop:
1. Load image → declare **type** (booster / reentry object).
2. **Declare topology** (dropdowns/checkboxes, never measured): stage count,
   has-fairing + fairing shape, strap-ons + count, fins + count + **clocking**,
   RO body_form, grid-fin state (stowed/deployed).
3. **Set scale**: click two points + enter the real distance.  The anchor's
   provenance is entered alongside ("length from …") — see risk R2.
4. **Prompted measurement pass** over the declared topology; each step shows
   the target field, the click, the proposed value with its pixel quantum,
   and Accept / Edit / Skip.
5. **Overlay audit** (toggle): schematic over image at scale; mismatches are
   visible; stylized elements (fin tabs, flagged fallbacks) drawn hatched so
   nobody aligns to a stylization (R12).
6. **Save** through the normal editor save; provenance stamp appended to
   notes; completeness checklist shown (R7).

## GUI placement (recorded 2026-07-30)

**Entry points — two buttons, one dialog.**  A "Measure from image…" button
in each editor:
- **Booster editor**, top of the dialog near the stage frames (Phase A scope:
  stages, fairing, fins, strap-ons);
- **Reentry-object editor**, beside the geometry fields near "Estimate β…"
  (⌀/length/nose; span from a plan view in Phase B).

This follows from populate-first + editors-own-the-fields: the tool is
launched FROM the thing it fills in, writes into the live editor fields
already open (old→new deltas visible, R8), and the editor's own Save/Cancel
remains the single commit point.  No new save path; no ambiguity about which
vehicle is being measured.

**The tool window** is one modal dialog, identical from either entry point:

```
┌──────────────────────────────────────────────┬───────────────────────────┐
│                                              │ ① Topology  (declared)    │
│                                              │   stages, fairing, fins…  │
│          image canvas                        │ ② Scale                   │
│          (zoom / pan / click)                │   click 2 pts + distance  │
│                                              │   anchor provenance       │
│          [overlay: schematic at scale — ☐]   │ ③ Prompt checklist        │
│                                              │   ▸ "click stage-1 ends"  │
│                                              │     proposed: 5.02 m      │
│                                              │     (1 px = 3.4 cm)       │
│                                              │     [Accept][Edit][Skip]  │
├──────────────────────────────────────────────┴───────────────────────────┤
│  Apply to editor        Cancel                                            │
└──────────────────────────────────────────────────────────────────────────┘
```

"Apply to editor" pushes accepted values into the parent editor's fields
(still editable there); closing the dialog discards everything else — the
no-persistence decision, structurally enforced.

**Deliberately NOT entry points:** no Tools-menu launcher in Phase A (a
free-floating tool would need its own vehicle picker and a second write
path); the main Schematic tab stays untouched — the overlay lives inside the
measuring dialog (audit-in-support-of-populate).  A standing Schematic-tab
overlay for pure auditing would be a separate, deliberate Phase-C addition.

## The hard cases (from the scoping discussion)

- **Fairing**: has/shape declared (existing enum); length/⌀ measured.
- **Fins / strap-ons partially visible**: count is DECLARED, geometry is
  measured ONCE and replicated at equal spacing (the data model is already
  count + one geometry, so "I only want to add one" is exactly the model).
  Flagged: "N declared, 1 measured, instances assumed identical."
- **Wedge / half-cone RO**: needs **two named views** (side + plan), each with
  its own scale; every measurement is tagged with its view.  Span/sweep come
  from plan, depth from side.  A missing view leaves its fields UNSET and
  flagged (the schematic's "span not modeled" flag) — never interpolated.
- **Perspective**: not solved, flagged (see R9).  Prefer line drawings.

## Risk register (red-team, 2026-07-30) and mitigations

Silent data corrupters:
- **R1 Roll/clocking foreshortening.**  A ×-rolled fin set seen side-on
  under-measures span by cos45° ≈ 29% with full confidence.  → the fin-span
  prompt is clocking-sensitive; the dialog offers a clocking selector
  (in-plane / ×-rolled / unknown, default in-plane) that feeds
  `Measurement(clocking=…)`; the cos correction is offered explicitly and
  flagged, never inferred.  WIRED to the UI (A3, 2026-08-01).
- **R2 Scale-anchor circularity.**  The anchor is usually a CLAIMED overall
  length — often the most contested number (cf. the AUR 10.2 m error).  →
  anchor provenance recorded with the scale; the tool reports which derived
  quantities are anchor-free (angles, fineness/ratio quantities survive a
  wrong anchor; absolute lengths inherit it 1:1).
- **R3 Stage-joint illusion.**  Joints are routinely invisible (raceways,
  paint bands, canisters).  → stage-boundary clicks carry their own flag
  class ("boundary inferred from surface features"), distinct from hard
  outer-mold-line measurements.
- **R4 Nose-radius illusion.**  RN drives heating as 1/√RN and is a few
  pixels in any photo.  → resolution floor: the tool REFUSES to populate a
  feature smaller than ~5 px; every proposed value displays its pixel
  quantum ("1 px = 3.4 cm").
- **R5 Convention mismatches.**  Half-cone side-view depth = stored ⌀/2;
  wedge ⌀ field IS the depth; fin span is per-panel exposed; stowed grid
  fins swap apparent chord/height.  → each prompt embeds its conversion
  ("click the side-view depth; stored ⌀ = 2×"); conversions unit-tested.

Dependency gaps:
- **R6 Wedge span field.**  RESOLVED (2026-07-30): `body_span_m` exists —
  a separate field from the wing planform, wing_geometry-blind.  Phase B is
  unblocked.
- **R7 Dimensional draft ≠ vehicle.**  Images populate dimensions only.  →
  save shows a completeness checklist (mass, propulsion, materials still
  required from sources); the notes stamp says "dimensional draft".
- **R8 Mixed-provenance collisions.**  Existing hand-entered values vs new
  clicks → always show old/new delta; never silently overwrite.

Scope and plumbing:
- **R9 Photogrammetry creep.**  Vanishing points, camera pose, multi-view
  solving: OUT OF SCOPE, permanently.  Scaled-orthographic screening only;
  photo-derived values are flagged as carrying perspective error.
- **R10 Model-first.**  The tool measures only what the data model consumes.
  A visible-but-unmodeled feature (canards, raceways) is a model discussion,
  not a field the tool invents.
- **R11 Dependencies/UI.**  JPEG needs Pillow (new dependency; parade photos
  are JPEGs — accept Pillow, document it).  Zoom/pan + undo are most of the
  UI work.  Honest sizing: Phase A is the largest single GUI feature yet.
- **R12 Stylization trap.**  Overlay-audit against stylized schematic
  elements is false confirmation → stylized/flagged elements drawn hatched.

## Phasing

- **A — core loop, single view.**
  - **A1 (DONE, 2026-07-30): reentry-object slice.**  `image_measure.py`
    (pure, tested core: Scale, pixel quantum, resolution floor R4, convention
    conversions R5, clocking R1, anchor-free note R2, provenance stamp) +
    the Tk dialog on the RO editor ("Measure from image…"): load image, set
    scale + provenance, prompted per-body-form measurement (Accept/Edit/Skip
    with pixel quantum), Apply writes fields + stamps notes.  Pillow is now a
    dependency (JPEG/PNG).  Tested: test_image_measure.py (14),
    test_gui_image_measure.py (5).  NOT yet: canvas zoom/pan (fit-only),
    overlay toggle, the topology-declaration panel, count-replication.
  - **A2 (DONE, 2026-07-30): booster editor + measure-one-declare-count.**
    The RO dialog is extracted to a shared `_open_image_measure_dialog`
    (prompts + apply_fn); the booster editor gets the "Measure from image…"
    button.  KEY SIMPLIFICATION vs the original design: the editor ALREADY
    declares topology (stage combobox, fairing/fins checkboxes+counts,
    strap-on spinbox) and ALREADY stores count + one-geometry — so no
    separate topology panel and no replication logic; `booster_prompts()`
    reads the editor's declared topology, and each repeated-feature prompt
    states the count it will be replicated to ("measure ONE … the model
    replicates to the 4 declared fins, assumed identical").  Apply writes
    stage/fairing/one-fin/one-strap-on geometry; the declared counts are
    untouched.  Tests: booster_prompts unit tests + a booster apply GUI test.
  - **A3 (DONE, 2026-08-01): clocking control (R1) wired to the UI.**  The
    fin-span prompt now carries `clocking_sensitive`; the shared dialog builds
    a clocking selector (`image_measure.CLOCKING_OPTIONS`: in-plane / ×-rolled
    45° / unknown, default in-plane) ONLY when the declared topology has a
    clocking-sensitive span, and shows it only while that prompt is selected.
    The chosen value flows into `Measurement(clocking=…)`, so the previously
    unreachable cos45 correction now fires from the GUI — offered, flagged,
    never inferred.  Tests: `test_only_fin_span_is_clocking_sensitive`,
    `test_clocking_options_default_to_no_correction`,
    `test_clocking_control_present_for_fins`.
  - **A4 (DONE, 2026-08-01): full RO dimensional coverage.**  `ro_prompts`
    now honours the editor's OTHER topology declarations: the biconic
    checkbox adds fore-cone length + break diameter (axisymmetric only), and
    the Maneuvering section adds the wing PLANFORM — root chord + exposed
    span, the span clocking-sensitive (R1 applies to a ×-rolled fin/wing set
    exactly as to booster fins).  S and AR are never prompted: they derive
    from the written planform (`_sync_wing_derived` fires on the var writes —
    measure the planform, derive the area).  LE sweep is an angle, outside a
    two-point distance tool — hand entry.  Wedge never gets wing prompts (its
    wing rows are disabled/zeroed by design).  Tests: prompt-generation units
    + a GUI apply test asserting the derivation fires.
    STILL A-phase remainder: canvas zoom/pan (fit-only) and the overlay toggle.
- **B — multiple named views**: side + plan with per-view scale and tagging;
  unlocks wedge/half-cone ROs.  Requires the span field (R6).
- **C — audit polish**: opacity/drag alignment, discrepancy highlighting,
  hatched stylization, mixed-provenance delta view.

Out of scope: persistence of images/clicks (decision 1), photogrammetry (R9),
auto-detection of features (everything is declared or clicked), writing any
value the user did not accept.
