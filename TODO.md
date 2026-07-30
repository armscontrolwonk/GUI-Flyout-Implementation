# Thrusty — To-Do / Parked Work

Living list of agreed-but-unbuilt work.  Items move here only with their
context attached (what pulls them in, where the plan lives), so any future
session can pick one up cold.  Governing rule as everywhere: derive, don't
invent.

## New — not yet planned

### 1. Image-overlay schematic sizing tool
A widget/tool where the user loads an image (parade photo, line drawing,
trade-press cutaway), enters a **single scale** (e.g. a known overall length,
or two clicked points + a distance), and the to-scale Schematic is drawn
**over the image** so every entered dimension can be checked against the
picture at a glance.  This is the Schematic's data-auditor role completed:
today it shows what the data says; this shows it against the evidence.
- Likely shape: an "Overlay…" control on the Schematic tab → file dialog →
  `imshow` the image behind the schematic axes at adjustable opacity; scale
  from one user-entered dimension (m per pixel); drag/arrow-key alignment.
- Derive-don't-invent: the tool must never write dimensions back from the
  image automatically — it makes mismatches visible; the user decides.
- Open questions: side-elevation only (matches schematic) or also planform?
  persist the image path + scale with the booster JSON?

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
- Wedge planform-span field on ROParams → removes the schematic's
  "span not modeled" flag and persists the estimator's span input.

## Parked earlier in the project (context in METHODS / chat)
- Biconic boost-phase wave drag (biconic Phase 2; β/L-D carry reentry today).
- Interstage / conical-stage drag (Phase 2 of interstage work; geometry+mass
  shipped, drag-neutral by design).
- Through-deck central upper stage (D5-style) and hammerhead /
  stage-enclosing fairing geometry.
- Descent-regime grid fins (Falcon-9-style forward fins; ascent-only today
  by agreement).
- Heating: Candler & Leyva found the Tauber et al. surface-temperature
  correlations over-predict at glide conditions — revisit the heating-side
  correlations against their CFD when heating work resumes.
- XLSX round-trip does not carry biconic, wing-planform, or body_form fields
  (pre-existing pattern: JSON is the primary store; extend ro_xlsx if the
  spreadsheet path starts being used for lifting bodies).
- Two pre-existing test failures, deselected in every run:
  damped_glide_smoke_test.py::test_lofted_plunges_both_modes and
  ::test_no_zoom_climb (predate this work; never diagnosed).

## Housekeeping
- References manifest: when the paper library moves to Drive, add
  data/REFERENCES.md mapping each citation key (R-127, AEDC-TDR-64-25,
  Fetterman D-2942/D-2956, Corda 1988, Candler S&GS 30, Lobanovskii 1983,
  Grant & Braun 2010, Heybey) → full citation, what the code uses it for,
  and the external location.  Policy: new papers go to Drive, not the repo.
- Note: deleting data/*.pdf later will NOT shrink clones (blobs stay in
  history); an actual shrink needs a deliberate history rewrite.
