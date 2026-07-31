# Image Dimensioning Tool — Design

Status: **design agreed, not yet built** (TODO item 1).  Load a picture of a
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
  under-measures span by cos45° ≈ 29% with full confidence.  → topology
  gains a clocking declaration (in-plane / ×-rolled / unknown); the cos
  correction is offered explicitly and flagged, never inferred.
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
- **R6 Wedge span field.**  Plan-view span has no ROParams home yet (Phase-3
  hook).  The span field must land before/with Phase B.
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

- **A — core loop, single view** (booster-first): image pane, scale + anchor
  provenance, prompted pass for stages/fairing/fins/strap-ons with clocking,
  pixel-quantum display, resolution floor, overlay toggle, notes stamp.
  Covers cylindrical/conical boosters end to end.
- **B — multiple named views**: side + plan with per-view scale and tagging;
  unlocks wedge/half-cone ROs.  Requires the span field (R6).
- **C — audit polish**: opacity/drag alignment, discrepancy highlighting,
  hatched stylization, mixed-provenance delta view.

Out of scope: persistence of images/clicks (decision 1), photogrammetry (R9),
auto-detection of features (everything is declared or clicked), writing any
value the user did not accept.
