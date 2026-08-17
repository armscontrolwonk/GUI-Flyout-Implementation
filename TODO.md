# Thrusty — To-Do / Parked Work

Living list of agreed-but-unbuilt work.  Items move here only with their
context attached (what pulls them in, where the plan lives), so any future
session can pick one up cold.  Governing rule as everywhere: derive, don't
invent.

## New — not yet planned

### 1. Image dimensioning tool — Phases A (complete) + B SHIPPED (2026-08-01)
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
A8: zoom/pan (wheel about cursor, right-drag pan, Fit; clicks stored in
original-image px so zoom never touches a measurement) + overlay toggle
(accepted measurements drawn on their view — audit of what was clicked).
Phase B: named Side/Plan views, per-view image + scale, hard view gating
for plan-only dimensions (wedge span), live cross-view length check
(check-only), view-tagged stamp.  LE sweep is an angle (not two-point-
measurable) — hand entry by design.
A6: clipboard paste (⌘V, image or copied file) + drag-and-drop when the
optional tkinterdnd2 package is present; new image now resets the scale.
A7: "Type value…" per prompt (the promised Edit path) — known dimensions
entered by hand, no image/scale needed, stamped separately ("entered by
hand, not measured"); checklist populated from dialog open.  Subsumes the
anchor-as-field idea.
Angles (2026-08-01, with Phase 2b): 3-click anchor-free angle measurement —
stores wing/fin LE sweep (degrees, separate from the metre contract), with
warn-only identity twins (sweep vs planform, flank vs ⌀/L) and the
two-flank symmetry check as a perspective/tilt detector (R9 → measurement).
Phase C delta view (2026-08-01): Apply now previews field → current →
proposed → Δ% (red beyond ±5%, findings first; measured vs entered marked;
check-only values counted audit-only) — the R8 "never silently overwrite"
promise, built.  REMAINING Phase C: schematic-at-scale overlay with
opacity/drag alignment + hatched stylization (R12) + live discrepancy
highlighting.  Full design in IMAGE_DIMENSION_TOOL_DESIGN.md.
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

### 2. Blender export of the schematic — SHIPPED (2026-08-07), option (a)
File → "Export Schematic to Blender…" writes a self-contained **bpy
script** (blender_export.py generates it from the composed as-it-will-fly
stack — exactly what the Schematic tab shows, cached at redraw).  Run in
Blender's Scripting tab → each element lands as a DISCRETE named editable
mesh (S1, Interstage_1, Fairing, Fin_1…, Strapon_n, RO_Body/RO_Wing_n) in
a collection, true metres, +Z up.  Real 3-D shapes: stages/interstages are
closed solids of revolution (cylinders/true frustums with the schematic's
derived adapter diameters); noses/fairings revolve their REAL analytic
profiles (tangent ogive, Haack series C=0/1/3, parabolic, blunt dome,
cone); an RO cone with a stored nose radius exports the true
tangent-sphere sphere-cone; biconic is piecewise; wedge extrudes across
its span; half-cone is a half-revolve closed by its flat deck; fins/grid
fins are plates placed at true angular spacing.  Derive-don't-invent:
same fallbacks as the 2-D schematic, every one listed in the script
header + the export dialog; unstored thicknesses (fins, aerospike stalk)
get thin nominals, each flagged.  Tests: dimension/stacking identities,
sphere-cone apex identity, compile() of the emitted script, and a full
exec under a bpy stub validating every mesh's face indices.
- OBJ export ADDED (2026-08-07): a .py isn't a model file (Blender only
  runs it from the Scripting tab), so the export now defaults to a
  Wavefront .obj Blender IMPORTS directly (File → Import → Wavefront),
  one named `o` group per element; the .py bpy script stays as the
  alternative extension.  Both paths share one tessellation
  (revolve_mesh / plate_mesh), pinned equal by test.  REMAINING (small):
  glTF, if a workflow ever wants materials/hierarchy in one file.

### 3. Better geospatial data sources — INVENTORIED (2026-08-16)
Current sources, from the code:
- **Launch sites**: `launch_sites.json` — 34 hand-curated sites (name,
  country, lat, lon).  NO provenance, NO site elevation.  The weakest
  layer for the modeling use case.
- **Borders/coastlines**: bundled Natural Earth **110m** countries GeoJSON
  (`data/ne_110m_countries.geojson`) — NE's coarsest tier; coastlines are
  visibly polygonal at trajectory-plot zoom.
- **Interactive maps**: folium with CartoDB positron tiles (online; fine).
- **Terrain/elevation**: NONE — trajectories end at h = 0 (sea level)
  everywhere; launch altitude unmodeled.
Upgrade path, in effort order:
  (a) swap the bundled GeoJSON to Natural Earth **50m** (drop-in, same
      format, ~few MB) — instant plot-quality win;
  (b) rebuild `launch_sites.json` with per-site provenance (citation
      field) + site elevation + expanded coverage.  NAME/COORDINATE
      SOURCE CHOSEN (2026-08-17): the two official public-domain US
      gazetteers, paired —
      **NGA GEOnet Names Server (GNS)** for foreign places
      (geonames.nga.mil/geonames/GNSData/ — BGN-approved names, WGS84
      coords, UFI unique ids, native + romanized variants, weekly
      updates; per-country ZIPs keep downloads manageable) and
      **USGS GNIS** for domestic
      (usgs.gov/us-board-on-geographic-names/download-gnis-data —
      Vandenberg, Wallops, Kodiak, WSMR).  Each rebuilt site stores
      {name, variants, lat, lon, elev_m, source: GNS|GNIS, id: UFI or
      GNIS-ID, retrieved: date} — provenance-first, matching the house
      rule.  Honest limits, stated up front: a gazetteer supplies
      authoritative NAMES and COORDINATES, not the judgment that a
      place is a launch facility — the curated site list stays curated,
      GNS/GNIS just anchors it; and the big complexes (Sohae, Semnan,
      Jiuquan, Plesetsk) are present but pad-level precision still
      comes from imagery/literature.  Third piece (added 2026-08-17):
      **BGN Antarctic names** via USGS staged products
      (prd-tnm.s3.amazonaws.com …/GeographicNames/Antarctica/ — take
      the GPKG: GeoPackage = SQLite, stdlib-readable, no GIS deps) for
      polar trajectories/FOBS ground tracks.  NOTE on acquisition from
      a Claude session (verified 2026-08-17): the USGS staged-products
      bucket prd-tnm.s3.amazonaws.com IS reachable through the egress
      proxy and carries BOTH GNIS domestic names (per-state ZIPs ~1 MB,
      AllStates ~38 MB, under …/GeographicNames/DomesticNames/) AND the
      Antarctic gazetteer — schema spot-checked (American Samoa file):
      pipe-delimited, feature_id | feature_name | feature_class |
      decimal lat/lon | BGN authority fields; NO elevation column in
      the current DomesticNames schema (the old NationalFile's
      ELEV_IN_M is gone — elevation comes from the (c) DEM plan
      regardless).  Only NGA GNS (foreign) is proxy-blocked (403) and
      needs a hand-download, committed like the DEM plan's baked
      lookups;
  (b2) SPIN-OFF the gazetteer enables (proposed 2026-08-17): nearest-
      populated-place annotation for trajectory outputs — bake a
      thinned populated-places extract (GNS/GNIS P-class, ranked by
      admin level so capitals/regional centres survive the thinning)
      into data/, nearest-neighbour by haversine over a lat/lon grid
      index, and impact/apogee/debris points get labels like
      "impact ~12 km SE of <place> (GNS UFI …)".  For the reporting
      audience this is the difference between a coordinate and a
      sentence someone can use;
  (c) DEM (agreed 2026-08-16; source chosen 2026-08-16: **Copernicus
      GLO-30**, not SRTM — ~2–4 m vs ~6–9 m vertical accuracy, and
      SRTM's 60°N cutoff misses Plesetsk/high-latitude Russia entirely;
      both are surface models, canopy-biased over forest — FABDEM is
      the bare-earth variant if that ever matters, license caveat):
      trajectories START at the launch site's real altitude and
      TERMINATE on real ground height (impact and the low-altitude
      glide floor) — the physics-relevant step, and the only one that
      touches trajectory code.  Architecture: bake one-time GLO-30
      lookups into the site database of (b) as elev_m + provenance (no
      runtime DEM needed for launch altitude); fetch GLO-30 tiles
      on demand along the ground track for impact/glide, cached, with
      a bundled coarse ETOPO-2022 fallback (~tens of MB) for offline;
  (d) air-launched missiles (agreed 2026-08-16, follows from (c)'s
      launch-state generalization): initial state = carrier release
      altitude + speed + flight-path angle instead of a ground pad —
      no vertical liftoff, the kick/loft schedule starts from release
      conditions (the existing launch_elevation_deg / guidance modes
      are the hooks).  DEM ground is then the floor, not the start.
      Scope when picked up: release-condition fields on the flight
      plan, launch-transient handling, and what "range" means measured
      from a moving release point.

### 5. Move the paper library from GitHub to an organized Drive
Agreed 2026-08-16.  The ~100 PDFs under `data/` (grid fins, TPS/ablation,
flight test, waveriders, lifting bodies) move to the Drive "Thrusty"
folder, organized into subfolders mirroring how the code cites them
(e.g. Aero — lifting bodies / Grid fins / Heating & TPS / Flight test &
programs / Trajectory & guidance), so data/REFERENCES.md and
HEATING_TPS_REFERENCES.md can link every entry.  Steps: (1) upload in
topic batches to Drive subfolders (needs an interactive Drive session —
bulk upload exceeds what the MCP connector can do); (2) extend
data/REFERENCES.md with the moved locations; (3) THEN the deliberate
repo slim — deleting data/*.pdf plus the history rewrite (separate,
careful step; deleting alone does not shrink clones).  Close the two
small gaps first: Fetterman D-2942/D-2956 PDFs into Drive (public NTRS),
TR R-127 + Sutton-Graves mirrored.

### 4. Satellite / orbital payload type (payload is not always a reentry object)
Today every payload is modeled as an `ROParams` reentry object; the payload
concept is implicitly "an RV/HGB that comes back down".  A **satellite** is
the natural second payload type: it separates, enters orbit, and does NOT
reenter — so it has no reentry shape, no β/L-D, no survivability analysis.
The fairing-vs-nose distinction is exactly this seam (recognised 2026-08-05):
- **bare nose shape ↔ reentry object** — the payload IS its own aerodynamic
  forebody and flies through reentry (Thrusty's whole current domain);
- **fairing ↔ enclosed payload** — a satellite/bus that can't fly bare, rides
  under a jettisonable shroud, revealed after jettison.  A fairing already
  implies "enclosed payload"; it just has no satellite object to enclose yet.
What already exists to build on: the `orbital_insertion` guidance mode and
the "Plan Orbit" solver — the trajectory half is done; the gap is a payload
TYPE that is orbital, not a reentry object.  Scope when picked up:
- a payload-type switch (reentry object | orbital payload) on the booster;
- an orbital payload carries mass + (optionally) a target orbit, pairs with
  the fairing, and its natural trajectory is orbital insertion;
- it SKIPS the reentry-survivability tab and the RO editor's β/L-D/TPS fields
  (nothing to survive) — derive-don't-invent: don't ask for reentry data a
  satellite doesn't have.
- Deferred question: deployment/station-keeping (out of scope — Thrusty ends
  at orbit insertion, as it ends at impact for an RV).

### 6. Jettisoned-fairing drag / debris trajectory (added 2026-08-17)
Today the fairing jettison only removes MASS from the ascending stack; the
fairing itself vanishes — no trajectory, no impact point.  For the
arms-control use case the fairing debris field is real information
(where do the halves land?), so the open question is how to model the
jettisoned fairing's own ballistic drop.  The user's framing: does it
matter whether it comes off as a CLAMSHELL (two halves) or a FULL object —
i.e. can we bound the accuracy loss of modeling it as one piece?
Sketch of the bounding argument (to be built, not asserted): the fall is
governed by ballistic coefficient β = m/(C_D·A).  A full fairing has the
whole shell's mass but, tumbling, roughly the same frontal area as one
half presented broadside — while a clamshell half has HALF the mass at a
comparable tumbling-average area, so β differs by roughly 2× between the
two idealizations (plus C_D differences between a closed shell and an
open half-shell scoop).  The honest Thrusty move is to RUN BOTH bounding
cases (full shell, single half) from the jettison state vector and report
the impact-point SPREAD as the error bar — if the spread is operationally
small (tens of km at ICBM jettison conditions, or dwarfed by wind/tumble
uncertainty anyway), one-piece modeling is justified BY the tool, not by
assumption.  Inputs already stored: fairing mass, dimensions (⌀, length,
shape), jettison time/altitude, full state vector at jettison.  Scope
when picked up: (a) tumbling-average C_D·A estimates for closed shell vs
half shell (literature: planetary-probe shell tumbling data, Falcon 9
fairing recovery numbers as sanity anchors); (b) a post-jettison
point-mass propagation from the jettison state (the existing 3-DOF core
reused with β-only drag); (c) report both impact points + spread on the
trajectory output/map.

## Phase 2b — lifting-body estimator completion: DONE (2026-08-01)
All four items shipped (details in PHASE2_LIFTING_BODY_PLAN.md §6):
cone/biconic α-sweep (α=0 continuity with the zero-AoA build-ups EXACT),
Grant & Braun anchors hit to 0.2%/0.05% (1.864 vs 1.86, 2.011 vs 2.01),
wing-body composite (Fetterman Λ75/81 within ~6%, directions correct),
swept-cylinder LE (cos³Λ exact; sweep dependence now real — penalty 15%
vs 28% more/less swept at test geometry).  GUI: wedge LE-radius row
(pre-filled from nose radius), half-cone composites the declared wing
planform.  BOR dialog stays zero-AoA until a consumer needs the sweep.

## Phase 3 — polar upgrades for lifting forms: DONE (2026-08-01)
The one phase that touches trajectory physics, gated to lifting forms
(axisymmetric byte-identity pinned by test; details in METHODS §8.8):
- Shape-derived C_L,max: Newtonian pressure C_L at the 25° cap from stored
  geometry (wedge needs body_span_m, else keeps 0.873 flagged; half-cone
  from ⌀/L + declared wings).  Force-level conversion makes the pull limit
  invariant to the A_ref convention (closes old limitation 3 too).
- Offset polar C_D = C_D0 + k·[(C_L−C_L0)²−C_L0²]: C_D(0)=C_D0 exactly (β
  keeps its zero-lift meaning), k back-solved on the offset parabola so
  (L/D)max stays exactly glider_LD; inconsistent C_L0 (> LD·C_D0/2) falls
  back to symmetric.  Anchored to Fetterman fig. 6b's negative-α C_N zero.
- trim_alpha_deg/trim_CL0 stored on the RO (sweep-native; "Use β and L/D"
  writes them; zeroed on save for bodies of revolution; json+xlsx
  round-trip).  α* now pre-fills the windward-heating operating AoA when no
  static-margin trim exists — the Candler attitude/L-D consistency guard
  (heating follow-up 2 CLOSED).
- Wedge planform-span field: DONE earlier (body_span_m, 2026-07-30).

## Parked earlier in the project (context in METHODS / chat)
- Boattail (V-2-style tapered aft body): NOT worth geometry modeling
  (agreed 2026-08-17).  Quantified from the model's own tables: a
  20%-necked boattail cuts base drag ~36% (ΔC_D ≈ 0.05–0.08 transonic,
  power-OFF), but nearly all atmospheric transonic passage happens under
  power, where the plume fills the nozzle exit and suppresses most base
  drag anyway → ascent effect ~10–15 m/s of ΔV, sub-1% of range.  For a
  non-separating airframe (V-2, Scud-B) the descent effect is real but
  already lives in the β the user assigns.  The USEFUL spin-off noticed
  while checking: `_cd_base()` has an unused `base_area_ratio` hook and
  the build-up currently charges FULL power-off base drag during the
  burn — with nozzle exit area + count now stored per stage, a power-on
  annulus correction (ratio = 1 − A_exit/A_base, floored at 0) is
  derivable from stored fields, and is a larger error than any boattail.
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
- References manifest: DONE (2026-08-16) — data/REFERENCES.md maps every
  core citation key → full citation → code use → repo file → Drive link
  (the core papers were uploaded to the Drive "Thrusty" folder
  2026-07-29).  Residual gaps, listed in the manifest: (a) Fetterman TN
  D-2942 / D-2956 have NO PDF anywhere (anchors were worked from
  scratchpad-extracted text; both are public NTRS docs — download into
  Drive to close); (b) TR R-127 and Sutton-Graves TR R-376 are repo-only,
  not yet mirrored to Drive; (c) the wider ~100-PDF data/ corpus
  (grid fins, TPS, flight test) is repo-only pending the deliberate
  move-and-history-rewrite.  Policy stands: new papers go to Drive.
- Note: deleting data/*.pdf later will NOT shrink clones (blobs stay in
  history); an actual shrink needs a deliberate history rewrite.
