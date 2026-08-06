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
  BUILT (2026-08-01, Phase C): the Apply delta preview.

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
  - **A5 (DONE, 2026-08-01): length closure + format filter.**  The booster
    checklist ends with a CHECK-ONLY prompt (overall length — never stored;
    there is no editor field for the derived total), and the dialog shows a
    live closure line: Σ(stage lengths + fairing) vs the total, where the
    total is the check measurement or the scale anchor when the user declares
    it IS the overall length (asked once at scale time — the anchor then
    doubles as the total for free).  WARN-ONLY by design: a mismatch is
    information (wrong claimed total — the AUR 10.2 m case; a mis-clicked
    invisible joint, R3; a real gap) and auto-normalizing segments would
    launder the disagreement into the data.  Pending state lists unmeasured
    segments; complete state shows the signed error (red beyond ±2%).
    Diameters never pollute the sum (length conventions only).  Also: the
    load-image filter now offers WebP/TIFF (Pillow always read them; the
    filetypes list simply omitted them and macOS greys non-matching files).
  - **A6 (DONE, 2026-08-01): paste + opportunistic drag-and-drop.**
    "Paste image" button + ⌘V/Ctrl-V (Pillow grabclipboard: handles both a
    raw clipboard image, e.g. a screenshot, and a copied-file list, e.g. a
    Finder copy).  OS drag-and-drop onto the canvas is enabled exactly when
    the OPTIONAL tkinterdnd2 package is importable (it bundles the tkdnd Tk
    extension); silently absent otherwise — Load/Paste are the guaranteed
    paths, no new hard dependency.  The loader refactor also fixed a latent
    bug: loading a NEW image now RESETS the scale (m/px belongs to the image
    it was anchored on; carrying it to another picture was silently wrong).
    Tested: paste both forms, scale reset, opportunistic flag.
  - **A7 (DONE, 2026-08-01): Type value… — the promised Edit path.**  The
    original core loop said "Accept / Edit / Skip"; only Accept (after a
    click-measure) and skip (never select the prompt) existed, so a dimension
    the user already knew precisely could not be entered in the checklist.
    A "Type value…" button now takes the STORED value in metres for the
    selected prompt — gated on NOTHING (no image, no scale; the checklist is
    populated from dialog open, which also fixed prompts being empty until a
    scale was set).  Typed values are recorded as `HandEntry` (no pixel
    quantum, flagged) and the provenance stamp lists them SEPARATELY
    ("entered by hand (not measured): …") — the audit trail never claims a
    typed number came off the image.  Typed lengths count toward the length
    closure exactly like measured ones.  This subsumes the deferred
    anchor-as-field idea (type the known anchor-field value directly).
    Skipping remains structural: Apply writes only accepted fields (R8) and
    is never gated on completeness.
  - **A8 (DONE, 2026-08-01): zoom/pan + overlay toggle — A-phase complete.**
    Wheel zooms about the cursor (0.05–8×), right- or middle-drag pans, Fit
    resets; clicks are stored in ORIGINAL-image pixels so zoom is display-
    only and can never touch a measurement or its quantum (tested).  The
    overlay toggle draws every ACCEPTED measurement's clicked segment +
    field label on its view — the audit of what was clicked, mis-clicks
    visible at a glance.  SCOPE NOTE: the schematic-at-scale-over-image
    comparison originally sketched for the overlay needs renderer
    integration + alignment UX and stays in Phase C; the accepted-
    measurement overlay is the first-order audit of the tool's own outputs.
- **B — multiple named views: DONE (2026-08-01).**  Views are generated
  from the checklist ({p["view"]}): single-view checklists (booster,
  axisymmetric RO) are unchanged — one slot, no selector.  The wedge gets
  Side/Plan radio buttons; EACH view carries its OWN image and its OWN
  scale (two figures are never at the same resolution); loading an image
  into one view never touches the other's scale.  Measure is HARD-GATED on
  the prompt's view being loaded and scaled (auto-switches view; the old
  label-only warning let a span be clicked off a side elevation — garbage,
  the span runs into the page).  Cross-view audit: the plan-view length is
  a check-only prompt (never stored) compared live against the stored side
  length — disagreement means one scale anchor is wrong and the span would
  inherit it (red beyond ±2%).  Measurements are view-tagged; the stamp
  notes "views: plan+side" when both contributed.  The missing-view rule is
  structural: skip the plan view and the span field stays unset + flagged.
- **Wing/fin sweep is DERIVED, not measured (2026-08-05).**  Measuring the
  sweep as an ANGLE proved a recurring trap: on a nose-up image the LE↔root
  opening (~13°) and the sweep Λ (~77°) are complements, and every attempt to
  instruct the click order left the 90° flip easy to get backwards
  (rectangles instead of triangles).  Retired.  Sweep now derives from the
  planform LENGTHS the user clicks confidently: `tan Λ = (root − tip)/span`
  (straight-TE, as `wing_geometry` assumes).  The RO checklist gains a
  TIP-CHORD prompt (0 = pointed delta → Λ = atan(root/span), the 77° that
  finally draws a triangle; a trapezoid measures its real tip chord and Λ
  drops); the booster fin already measures root/span/tip so its sweep derives
  the same way.  Apply writes the derived `_wing_sweep_var` / `fin_sweep`.  A
  length RATIO → still anchor-free, and no angle to reverse.  No sweep angle
  prompt remains on either editor.
- **Cone flank angles (check-only, DONE 2026-08-01)** — the only measured
  angles left.  Three-click measurement (vertex + two rays),
  ANCHOR-FREE: needs an image but NO scale (R2 — an angle survives a wrong
  anchor completely); the resolution guard is on RAY length (2× the R4
  floor).  Stored targets: RO wing LE sweep and booster fin LE sweep
  (degrees — kept in separate field maps from the metre contract; angle
  prompts never touch CONVENTIONS).  Built-in cross-checks, warn-only like
  the closure line, because angles fail differently (non-uniform image
  stretch and perspective tilt corrupt them silently):
  - **identity twins**: measured sweep vs tan Λ = (c_r − c_t)/s_e from the
    accepted planform; mean cone flank vs tan θ = (⌀/2)/L from accepted
    lengths — disagreement on a clean orthographic image is impossible, so
    it diagnoses stretch/tilt/mis-click specifically;
  - **flank symmetry** (axisymmetric, check-only fields, never stored):
    upper vs lower half-angle must match; asymmetry impeaches the WHOLE
    image (tilt/perspective) — the R9 screening upgraded to a measurement.
- **C — audit polish**:
  - **Delta view (DONE, 2026-08-01)** — the R8 mitigation ("always show
    old/new delta; never silently overwrite"), previously promised but not
    built: Apply now opens a preview table — field → current editor value →
    proposed value → Δ% (red beyond ±5%, `DELTA_WARN_REL`), measured vs
    entered marked, biggest deltas first (findings on top), blank/zero
    fields shown as "new", check-only cross-checks counted as audit-only.
    Nothing is written until "Write N fields"; Back returns to measuring
    with everything untouched.  One field→var map per editor
    (`_img_field_var`) is shared by apply and the preview, so they can
    never disagree about what is writable.
  - **Interstage length (DONE, 2026-08-05)** — a stage whose interstage
    adapter is declared (its checkbox) gets an interstage-LENGTH prompt after
    that stage's len/dia.  Length is the only image-measurable interstage
    dimension: its ⌀ is inherited from the adjacent stages (not stored), and
    mass / jettison time aren't dimensions.  It carries the stage_length
    convention, so it tiles the overall-length closure sum; apply routes it to
    that stage frame's `_is_len_var` and keeps the interstage section enabled
    (measured-it-so-show-it).  Its diagram is a short segment at the stage top.
  - **What-to-click diagrams (DONE, 2026-08-05)** — every checklist prompt
    shows a tiny schematic under the selector: a stylized base outline
    (nose-up RV with delta fins / plan-view planform / two-stage stack) with
    the exact clicks NUMBERED in order — lengths as an arrowed segment,
    angles as vertex + two rays + arc.  Redraws on selection change,
    including checklist auto-advance.  Pure DATA in image_measure
    (`DIAGRAM_BASES`, `diagram_spec` — unit square, y down); the GUI only
    scales and draws, and tests pin that EVERY prompt has a diagram, click
    counts match the prompt kind, and the sweep diagram's rays trace the
    same fin corners the base art draws (the picture can't contradict the
    convention).  Unknown fields get a blank strip, never a wrong picture.
  - REMAINING: schematic-at-scale overlay with opacity/drag alignment +
    hatched stylization (R12) + live discrepancy highlighting.

Out of scope: persistence of images/clicks (decision 1), photogrammetry (R9),
auto-detection of features (everything is declared or clicked), writing any
value the user did not accept.
