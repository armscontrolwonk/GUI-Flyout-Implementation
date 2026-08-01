# Thrusty — To-Do / Parked Work

Living list of agreed-but-unbuilt work.  Items move here only with their
context attached (what pulls them in, where the plan lives), so any future
session can pick one up cold.  Governing rule as everywhere: derive, don't
invent.

## New — not yet planned

### 1. Image dimensioning tool — Phase A1–A4 SHIPPED (A3/A4 2026-08-01)
Working "Measure from image…" on BOTH editors (shared dialog).  RO: load →
scale+provenance → prompts from the FULL declared topology (body form +
biconic checkbox → fore-cone length/break ⌀ + Maneuvering → wing planform,
S/AR derived not measured) → Apply writes fields + notes stamp.
Booster: prompts generated from the editor's own declared topology (stages/
fairing/fins+count/strap-ons); measure-one-declare-count (counts untouched,
model replicates).  A3: the R1 clocking correction is wired to the UI —
fin-span and RO wing-span prompts are clocking-sensitive and the dialog
offers an in-plane / ×-rolled / unknown selector (default in-plane) feeding
the tested cos45 core; built only when the topology has such a span.  Core in
image_measure.py (tested); Pillow dependency.
A5: length-closure warning (stages+fairing vs overall length — check-only
prompt or anchor-declared total; warn-only, never normalizes) + WebP/TIFF in
the load filter.
REMAINING before B: canvas zoom/pan (fit-only today) and the overlay toggle;
LE sweep is an angle (not two-point-measurable) — hand entry by design.
Discussed & deferred (2026-08-01): drag-and-drop (optional tkinterdnd2,
opportunistic), clipboard paste, anchor-as-field (honest "declared, not
measured" fill of the field the scale anchor IS).
Phase B multi-view (wedge/half-cone plan span) unblocked (body_span_m).
Full design in IMAGE_DIMENSION_TOOL_DESIGN.md.
[superseded lead-in below kept for the risk register pointer]

### 1b. Image tool — original scoping pointer
Load a picture, declare type + topology, and be walked (prompted drawing)
through clicking dimensions off the image into the existing editor fields;
schematic overlay as live audit.  Decisions taken: NO persistence (text
provenance stamp in notes only), prompted checklist not free-measure,
populate-first with audit support.  Full design + 12-risk red-team register
(clocking foreshortening, scale-anchor circularity, stage-joint illusion,
nose-radius resolution floor, convention conversions, span-field dependency,
dimensional-draft status, photogrammetry excluded, …) in
**IMAGE_DIMENSION_TOOL_DESIGN.md**.  Phase A = single-view booster core loop
(largest single GUI feature yet); Phase B = multi-view for wedge/half-cone
(requires the ROParams span field, Phase-3 hook); Phase C = audit polish.

### 2. Blender export of the schematic (rough-draft 3D for modelers)
Export the vehicle as a simple 3D file to hand a modeler a dimensionally
correct starting point.  The schematic's geometry is mostly bodies of
revolution (stages = cylinders/frustums, noses/fairings = cones/ogives, RO =
cone/biconic) plus a few known non-revolved parts (planar fins, grid-fin
boxes, wedge/half-cone lifting bodies) — all revolve/extrude cleanly from
the same stored fields the 2-D schematic already draws.
- Format: a true .blend can only be written by Blender itself, so either
  (a) export a **bpy Python script** the modeler runs in Blender's scripting
  tab — builds named, editable primitives (S1, S2, fairing, fin_1…) at true
  dimensions, organized in a collection (most useful as a "rough draft"), or
  (b) export **OBJ/glTF** meshes we generate ourselves (zero dependencies,
  imports anywhere, but frozen geometry).  Doing (a) with (b) as fallback
  covers both kinds of modeler.
- Derive-don't-invent carries over: only specified geometry is exported;
  flagged/unset items (e.g. wedge span) are omitted or stubbed with a
  clearly-named placeholder, never silently guessed.

### 3. Better geospatial data sources
Improve the sources behind launch/target locations and the map layers.
- Open questions to scope first: which layer hurts today — the built-in
  locations database (coverage/accuracy of sites), basemap/coastline detail
  on trajectory plots, or terrain/elevation (impact-point and low-altitude
  glide realism)?  Candidate sources: Natural Earth (coastlines/borders),
  ETOPO/SRTM (elevation), curated site lists with citations.
- First step when picked up: inventory current sources in the code and note
  provenance for each (some may be as unattributed as the Ref-(4) chart was).

## Phase 2b — lifting-body estimator completion (deferred by agreement)
Plan + anchors pre-specified in PHASE2_LIFTING_BODY_PLAN.md §6.
- Cone/biconic α-sweep on the same sector machinery (upgrades the shipped
  zero-AoA estimators to L/D estimators; continuity rules already written).
- Wing-body composite (half-cone + delta wing) — where the Fetterman 2–5×
  viscous-share anchor properly lands.
- Grant & Braun biconic peak-L/D contours as a friction-off anchor.
- Swept-cylinder leading-edge component (AEDC §2.1.3) — the one remaining
  paper-driven physics upgrade: gives the wedge a real bluntness term and a
  sweep dependence (documented limitation: sharp-Newtonian is
  sweep-independent at trim).

## Phase 3 — polar upgrades for lifting forms (parked; only part that
## touches trajectory physics, gated to lifting forms only)
- Shape-derived C_L,max replacing the universal 0.873 body ceiling.
- Offset polar C_D = C_D,min + k·(C_L − C_L0)² (camber offset; measured
  support in Fetterman TN D-2942 fig. 6b).
- Store α*/C_L0 on the RO (2c displays them; storage deferred until this
  consumer exists).
- Wedge planform-span field: DONE (body_span_m, 2026-07-30) — separate
  from the wing planform, wing_geometry-blind (phantom-wing guard tested);
  schematic flag retires when set; estimator pre-fills/persists it.  This
  also satisfies the image tool's R6 dependency (Phase B unblocked).

## Parked earlier in the project (context in METHODS / chat)
- Biconic boost-phase wave drag (biconic Phase 2; β/L-D carry reentry today).
- Interstage / conical-stage drag (Phase 2 of interstage work; geometry+mass
  shipped, drag-neutral by design).
- Through-deck central upper stage (D5-style) and hammerhead /
  stage-enclosing fairing geometry.
- Descent-regime grid fins (Falcon-9-style forward fins; ascent-only today
  by agreement).
- Heating: the Candler-Tauber issue, SCOPED (see chat, 2026-07-30).
  Thrusty has NO direct exposure to the criticized correlation: Candler &
  Leyva's target is the Tauber et al. CONVECTIVE flat-plate/acreage
  correlation used by Tracy & Wright; Thrusty's only Tauber is
  Tauber-Sutton 1991 RADIATIVE gas heating (exactly zero below 9 km/s —
  inactive for HGV glide).  Thrusty's acreage/windward path is
  BODY_FLUX_FRACTION (0.13, cited vs Lu/Shi & Zhang 2024 + STS-1) ×
  Sutton-Graves stagnation × Newtonian windward amplification — a
  different, better-anchored method family.  Residual follow-ups, in
  priority order:
  1. Free anchor: DONE (2026-07-30, test_candler_windward_anchor.py +
     METHODS §13.8).  Result: at the Candler glide point the cone-flank
     windward model OVER-predicts a flat-bottom HGV by ×1.65 in T / ×7 in
     flux (1934 K vs CFD ~1175 K).  BODY_FLUX_FRACTION 0.13 is a CONE ratio;
     a flat lower surface implies ~0.018.  0.13 left unchanged (anchored for
     its cone domain).  FOLLOW-ON DONE (2026-07-30): windward heating is
     body_form-aware — BODY_FLUX_FRACTION_FLAT = 0.018 (single-point Candler
     anchor, stated on output) selected for wedge/half_cone, cone value
     untouched; trajectory forwards body_form; closure + wiring pinned by
     test_candler_windward_anchor.py and test_body_form.py.
  2. Consistency guard: the windward α band (5–20°) is user-tunable and
     could be set inconsistently with the vehicle's polar — consider
     pre-filling α_op from the polar/estimator trim α* (the Candler
     lesson: attitude must be consistent with L/D).
  3. Remember the propagation asymmetry when judging deltas: flux errors
     compress ×4-root into temperature (2× q → +19% T; 3× → +32%) but pass
     LINEARLY into ablator heat-load/thickness, and AMPLIFY through
     Planck-band radiance (Candler: T&W's stacked errors → 15–17× IR
     radiance, yet the detectability judgment only PARTIALLY flipped —
     below DSP, still visible to SBIRS).  Deltas matter where they cross a
     tier threshold; report margins, not just verdicts.
- XLSX round-trip: DONE (2026-07-30) — biconic, wing planform, body_form,
  and body_span_m all carry through ro_xlsx (appended rows; pre-upgrade
  workbooks import with defaults; test_ro_xlsx.py).
- Two pre-existing test failures: DIAGNOSED AND FIXED (2026-07-30).
  Bisected to db73fa1 (2026-07-10: terminal-dive default 30 km -> 0 =
  glide-to-impact), which let the marginal lofted case skip instead of
  plunge.  Fixtures now pin glider_terminal_alt_km at their 30 km
  calibration point; the zeta=0 == skip_glide identity is asserted with the
  dive off (the modes' dive handoff differs by one output sample —
  discretization, not physics).  Full suite green with NO deselects.

## Housekeeping
- References manifest: when the paper library moves to Drive, add
  data/REFERENCES.md mapping each citation key (R-127, AEDC-TDR-64-25,
  Fetterman D-2942/D-2956, Corda 1988, Candler S&GS 30, Lobanovskii 1983,
  Grant & Braun 2010, Heybey) → full citation, what the code uses it for,
  and the external location.  Policy: new papers go to Drive, not the repo.
- Note: deleting data/*.pdf later will NOT shrink clones (blobs stay in
  history); an actual shrink needs a deliberate history rewrite.
