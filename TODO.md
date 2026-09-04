# Thrusty — To-Do / Parked Work

Living list of agreed-but-unbuilt work.  Items move here only with their
context attached (what pulls them in, where the plan lives), so any future
session can pick one up cold.  Governing rule as everywhere: derive, don't
invent.

## New — not yet planned

### 9. Whole-body L/D "over-prediction" — REASSESSED, NO CEILING CHANGE (2026-08-28)
Chasing "non-separating bodies over-range in phugoid glide", the working
hypothesis was that `glider_ld.whole_booster_LD` over-predicts L/D_max.  It
does NOT.  Cross-checked against Digital DATCOM (validation/datcom/) the
build-up sits within 5/9/10% (M2/3/5) and CONSERVATIVE (finless slender
L/D_max: glider_ld 2.13/2.48/3.17 vs DATCOM 2.23/2.71/3.51).  A trial
crossflow de-rate (`_ETA` 1.0 -> 0.50) pushed the gap to -19/-21/-22% —
i.e. it BREAKS the documented validation.  Reverted; `_ETA` stays 1.0.

The premise was anchored on the WRONG shape classes: CAN-4 (~1.25) is a
stubby cone-cylinder-FLARE projectile (L/d 5.84, high drag); Seiff-Wilkins
(~4-6.7) is a WINGED glider; Intrieri (~0.38) is a blunt capsule.  None is a
clean slender body, for which ~3 at M5 is right (DATCOM).  The Seiff-Wilkins
nonlinear-lift finding is real but a weak L/D lever here (crossflow OFF ->
1.44, full -> 3.06; the potential slope + Cd0 dominate at best-glide alpha).

The low free-flight L/D (~1) of a fin-stabilized body is a TRIM effect, not
a lower ceiling: it is the L/D at the body's (low) trim alpha, set by cg,
which `trim_gate` evaluates (unstable -> tumbles/ballistic; no commanded
control surfaces -> trims at zero incidence -> ballistic NOSE-FIRST, keeping
its beta; control-limited -> best L/D over the reachable alpha band; only
stable + control-rich reaches best glide).  So the real over-range levers are (a) cg / static margin, and
(b) flying a drag-driven body BALLISTIC vs as an active glider — the latter
now enforced by the ballistic=no-lift guard (trajectory.py, committed
2026-08-28).

Harness (test_ld_calibration.py) rewritten to LOCK this in: a DATCOM-
agreement guard (L/D_max within 12% and conservative at M2/3/5 — fails
loudly if anyone re-introduces a de-rate), the Mach plateau, the winged
anchor, lifting-surface ordering, and two trim tests (small fins do NOT buy
best-glide L/D; a control-rich body does).  Also fixed the stale
`whole_missile_LD` name in validation/datcom/compare_datcom.py.

DONE: trim_gate's 25-deg control-deflection assumption no longer over-grants
best-glide trim.  Control authority is read from the reentry object's
glider_control_surfaces descriptor (none => no commanded deflection => trims at
zero incidence => no glide), deflection is capped at the Kumar & Stollery
separation limit damping_estimate.py already uses, and the trim angle is the
root of a nonlinear moment balance rather than a linearised relation that
returned 144 deg for a Scud-B and could not limit that vehicle at ANY cg.
See BODY_GLIDE_LD_PLAN.md 7.1 and METHODS.md 8.10.

Still open there, in rough order of how much they matter:

(a) CENTRE-OF-PRESSURE BIAS, MEASURED and now BOUNDED AGAINST THE REFERENCE.
compare_datcom.py reads the DATCOM CM/XCP columns (previously unread) and shows
the modelled body c.p. sits FORWARD of DATCOM at every alpha and Mach, by up to
~20% of body length, worst at low alpha / high Mach.  Direction is
NON-CONSERVATIVE for the gate (understates the restoring moment -> over-grants
trim alpha -> over-grants glide).

The reference is good enough to blame the model: Sooy & Schmidt (JSR 42(2),
2005) put DATCOM's own c.p. error against wind tunnel below 2% of body length
at any AoA (body-wing-tail M1.5/M4.6, body-tail M2.0), and Simon & Blake
(AIAA 99-4258) report c.p. well predicted at all AoA at supersonic speeds.  So
the 5-20% gap is model error, not reference noise.

CAUSE, now sourced rather than hypothesised.  Simon & Blake note that at low
alpha DATCOM determines the potential c.p. from empirical charts / Van Dyke
hybrid theory, NOT the nose-concentrated slender-body result Thrusty uses --
which is exactly where the gap is worst.  A real fix therefore means
distributing the potential normal force along the body instead of putting it
all at the Barrowman nose c.p.  Deliberately NOT corrected by a fitted factor;
pinned instead by test_cp_bias_is_forward_and_bounded.

DONE in this pass: the fin's VISCOUS normal force now acts at the panel centroid
rather than being lumped at the fin aerodynamic centre, per Simon & Blake Eq. 6.
Small effect on the shipped vehicles, but it makes the station structurally
right and sourced.

(b) The fin station and the control-deflection term are validated against
NOTHING: the committed DATCOM deck is body-alone and finless.  Generating a
finned/deflected deck (validation/datcom/README.md has the PDAS build steps)
would be the single highest-value piece of evidence for this subsystem.

(c) control_eff = 0.85 is unverified AND WRONG IN FORM.  Hemsch & Nielsen,
JSR 20(4) 1983 (read 2026-09-04) defines the two factors it is meant to be the
ratio of: the Beskin upwash factor K_w for the angle-of-attack case, and the fin
deflection factor A_ji (their Eqs. 11-12), tabulated for slender-body theory
against a/s_m (body radius / semispan) in their Table 1.  The headline point is
structural: control effectiveness is a FUNCTION of a/s_m, not a constant.  Using
their Table 1 self-deflection column over the repo's own NACA 1307 K_W(B),
the ratio runs

    a/s_m   0.0   0.2   0.4   0.5   0.6   0.8   0.9
    ratio   0.92  0.76  0.66  0.62  0.59  0.54  0.52

i.e. it varies by nearly 2x across fin geometries, and for the shipped Scud-B
fins (a/s_m = 0.467) it is 0.635 -- well BELOW the hard-coded 0.85, so the
constant currently OVERSTATES control authority for that vehicle, in the
non-conservative direction.

NOT IMPLEMENTED, deliberately, on two honest blockers:
  1. Table 1's column headers did not survive PDF text extraction.  The self
     column was identified by physical argument (two columns are exactly
     antisymmetric = the perpendicular fin pair; one is small and same-signed =
     the opposite fin; the remaining one goes to 1.000 at a/s_m = 1 where the
     exposed span vanishes = the self term).  Defensible, but it is an inference
     from values, not a read header.  A look at the printed table settles it.
  2. Thrusty's c_na_fin carries k_sum = K_W(B) + K_B(W), i.e. fin PLUS body
     carryover.  The deflection analogue needs the deflection-case carryover
     too (k_W(B) + k_B(W)), and whether A_44 already embeds that is exactly what
     the normalisation question turns on.

Settle both from Nielsen, Hemsch & Smith, "A Preliminary Method for Calculating
the Aerodynamic Characteristics of Cruciform Missiles to High Angles of Attack
Including Effects of Roll Angle and Control Deflections", ONR CR215-226-4F,
Nov. 1977 (their Ref. 13, the source of both the Table 1 factors and the
slender-body K_w values), or from Nielsen, *Missile Aerodynamics*, McGraw-Hill
1960 (their Ref. 12), which derives k_W(B) and K_W(B) in the same framework.

(d) The Kumar & Stollery deflection band is a [snippet] in
docs/cl_margin_references.md, not read against the primary; and laying the tier
names onto its endpoints is a modelling choice, not a measurement.

Open (only if a user still sees unrealistic glide range with correct cg):
audit whether the phugoid / skip-glide LAW loses too little energy per skip
(a guidance-law question, separate from L/D).  Papers in hand
(mirror to Drive per data/REFERENCES.md): Seiff-Wilkins TN D-341, Syvertson-
Dennis NACA 1328 (SOSE), Vukelich-Jenkins (Missile DATCOM feasibility),
Fournier-Dupuis AIAA 96-3399, Intrieri TM X-569, Yates-Chapman AIAA 96-3360.

### 8. Canards / lifting surfaces on a non-separating body — DEFERRED (2026-08-22)
Deferred with the user's agreement: "for a KN-23 the current approach is
fine.  But at some point we have to deal with canards."

Current state (verified this session).  `glider_ld.whole_booster_LD` — the
derived body L/D — reads exactly ONE lifting surface: the booster last
stage's TAIL FINS (`has_fins`, `fin_span_m`, `fin_root_chord_m`,
`fin_tip_chord_m`, `fin_sweep_deg`), carried over the body by Nielsen-
Kaattari-Pitts (`k_sum = (1+r/s)²`).  The body's potential normal-force
slope is the slender-body `2·(A_b/A_ref)` — always 2.0/rad, shape-
independent; only fins raise it.  The reentry object's OWN wing fields
(`wing_area_m2`, `wing_root_chord_m`, `wing_span_exposed_m`,
`wing_aspect_ratio`, `wing_sweep_deg`) are NOT read by this path — they
feed only the 3-D depiction and (for a SEPARATING RV) the trajectory drag
polar.  So a KN-23, whose control surfaces are stage-level tail fins, is
modeled correctly; a body with its OWN lifting surfaces cannot express
them in the derived L/D.  (The body-mode wing hint now says this, rather
than telling a body user to "enter the planform" — 2026-08-22.)

What a canard build needs (its own design pass):
  - A second lifting surface on the body (canards forward of the CG), with
    its own N-K-P carryover AND the right SIGN in the static-margin / trim
    gate (grid_fin_sizing): a canard ahead of the CG is DEstabilising and
    moves the CP forward — the opposite of a tail fin.  Getting the CP
    right matters more than the small L/D bump.
  - Decide whether a body's RO wing planform should feed
    `whole_booster_LD` as a lifting surface (so the editor's wing fields
    mean what they appear to), and how stage fins + RO wings + canards
    combine (sum vs. mutually-exclusive roles: tail vs. canard vs. wing).
  - Downwash / surface-to-surface interference between a forward canard and
    an aft fin (the KN-23 has only the tail set, so this is untested).
  - Schematic: draw canards forward of the CG, distinct from tail fins, so
    DRAWN ≡ FLOWN holds for the two-surface layout.
Not needed for the ballistic / tail-fin-controlled quasi-ballistic bodies
Thrusty models today; revisit when a canard-controlled MaRV or a winged
body is the subject.


### 7. Vehicle-derived structural α / q·α capacity — SHELVED (2026-08-20)
Prototyped and deliberately parked while building the ascent q·α load
model (METHODS §9.6).  The SHIPPED feature is: (a) q·α reporting + a
constant-q·α α-limit envelope (10° default at max-q, user-set; replaced
the arbitrary 100 Pa gate — the envelope self-deactivates in vacuum
because q·α→0), and (b) an applied lateral-g readout `n_lat =
q·A_ref·C_Nα·α/(m·g₀)`.  What is SHELVED is auto-DERIVING the structural
*capacity* (the ceiling) from vehicle data.
The idea, kept for whoever adds a structural model: anchor a thin-cylinder
bending capacity to the axial thrust the case demonstrably carries,
`M_cap ≈ F·R/2` (no material/skin data needed), convert to a lateral-g
ceiling, and use it to auto-set the limit.  Why parked (all validated
against real data this session):
  - It estimates CAPACITY; user guides publish EXPERIENCED load — equal
    only for loads-optimised vehicles.  Minotaur-IV (Peacekeeper SR118,
    over-built for its SLV role) estimates 1.05 g vs its <0.5 g nominal:
    consistent (cap ≥ nominal·FoS) but unvalidatable (capacity is
    proprietary).  Accuracy ≈ ±2×, single-point anchor (START-1 0.56 g
    est vs 0.7 g published).
  - `M_cr = P_cr·R/2` assumes monocoque — meaningless for a
    pressure-stabilised (balloon) tank; `P_cr≈thrust` ignores that
    ground-handling / combined max-q may size the structure.
  - Bending arm `≈0.25·L` swings the answer ~3× and was effectively fitted
    to the one anchor.  Real methods (SMC-S-004; CNES EUCASS 2013) carry
    distributed mass/aero to locate critical stations.
  - Limits the STEERING term only; SP-8099 p.10 + CNES put steering at
    0.05–0.15 of the WIND bending moment, and Thrusty models no winds.
Validation corpus gathered (in Drive / session): SP-8099; CNES EUCASS
2013 (Delorme et al.); NSSL Phase-2 CDRL (→ SMC-S-004, day-of-launch
placarding, limit-load allowables vocabulary); START-1 User's Handbook
(0.7 g); Cyclone-4 (0.3–0.6 g); Minotaur I / IV-V-VI (<0.5 g steady,
6–12 g CLA transient); Pegasus (−2.33 g winged pull-up).  Open lead for
an absolute q·α placard in Pa·rad: Bartos, AIAA 2001-0841 (Delta
load-relief wind model).  If picked up: add structural inputs (skin
gauge, material, construction type incl. balloon-tank flag) so capacity
is derived not fitted, and combine with a wind model so the limited term
is the sizing one.


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
- **Place lookup** (the "Find Location" picker shared by launch site /
  aim-at-target / estimate-azimuth — MISSED in the original inventory,
  corrected 2026-08-17): offline = GeoNames.org city data via the
  optional `geonamescache` package (name/lat/lon/country/population,
  population-ranked); online fallback = OSM Nominatim via `geopy`.
  Both are OPTIONAL pip packages — Thrusty bundles no gazetteer of its
  own, and on a plain install (user's machine, screenshot 2026-08-17:
  the picker shows the "pip install geonamescache" tip) the offline
  path is absent entirely, leaving online-only Nominatim or nothing.
  Both are also third-party crowd-maintained sources (GeoNames CC-BY,
  OSM ODbL + usage policy).  Exactly what the NGA GNS / USGS GNIS trio
  upgrades: official BGN names, public domain, baked into the repo so
  the offline path ALWAYS exists (no pip dependency), and covering
  facilities/spot features that city lists lack.  When 3(b) lands, the
  picker should search the baked gazetteer first and keep Nominatim as
  the online catch-all.
- **Borders/coastlines**: bundled Natural Earth **50m** countries GeoJSON
  (`data/ne_50m_countries.geojson`, 1.8 MB) — upgraded from 110m 2026-08-18.
- **Interactive maps**: folium with CartoDB positron tiles (online; fine).
- **Terrain/elevation**: SHIPPED 2026-08-20 (see (c)) — opt-in DEM:
  launch at real pad elevation, terminate on real ground height;
  default remains flat sea level (Forden benchmark condition).
Upgrade path, in effort order:
  (a) NE 50m coastline swap — SHIPPED (2026-08-18): bundled
      ne_50m_countries.geojson (geometry-only, properties stripped,
      coords 4-dp; 1.8 MB) replaces the 110m tier on every matplotlib
      map (Ground Track plot + Gazetteer Explorer via _load_borders,
      which now prefers 50m and falls back to 110m if present).
      Resolves the Bahamas, Lesser Antilles arc, Florida Keys the
      110m tier dropped entirely.  (The cartopy picker map draws its
      OWN coastlines at 110m — left as-is to avoid triggering cartopy's
      on-demand download, given the known cert issue.);
  (b) rebuild `launch_sites.json` with per-site provenance (citation
      field) + site elevation + expanded coverage.
      GAZETTEER PHASE 1 SHIPPED (2026-08-17): bundled offline gazetteer
      — gazetteer_build.py (reproducible bake, provenance in
      data/gazetteer/MANIFEST.md) + gazetteer.py (SQLite index cached
      in ~/.gui_missile_flyout, diacritic-folded variant-aware search,
      nearest()) + data/gazetteer/ packs: FULL GNIS domestic (974,023
      features, every class) + BGN Antarctic (14,353 + variants),
      16.7 MB total.  The Find Location picker now searches it first
      (always-present offline path; geonamescache demoted to legacy
      fallback, Nominatim still the online catch-all).  PHASE 2
      SHIPPED (2026-08-17): all NINE NGA GNS class files baked (user
      decision: everything, no thinning) — 9.45 M worldwide features
      (populated places, spot/facilities, hydro, terrain, admin, areas,
      vegetation, transport, undersea), transferred via the gns-staging
      branch (deleted after), verified byte-for-byte, class-aware
      ranking added (cities/facilities above creek noise).  Grand total
      with Phase 1: 10,435,205 features, ~196 MB packs, counts + the
      whole provenance chain in data/gazetteer/MANIFEST.md.
      REMAINING on this thread: the launch_sites.json rebuild proper
      (anchor the 34 sites to GNS/GNIS IDs) and the nearest-place
      impact annotation (b2) on top of gazetteer.nearest() — both now
      pure code, no more data acquisition.  VARIANTS ARE A
      HARD REQUIREMENT (user, 2026-08-17): romanized names differ
      across systems (Sŏhae/Sohae, Tongch'ang-ri/Dongchang-ri), so the
      baked extract KEEPS every GNS variant name and native-script
      form, the picker matches against ALL of them, and results show
      the BGN-approved primary with the matched variant — variants are
      never dropped in thinning or repacking.  NAME/COORDINATE
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
  (b2) nearest-populated-place lookup — SHIPPED (2026-08-18) as
      **Cartography ▸ Nearby Places…**, per the user's reframe after
      the overlay withdrawal (3e): not a map product anyone sees, but
      a lookup the user opens to put coordinates into words.  For
      every key trajectory event (launch, stage/fairing/debris
      impacts, apogee ground point, reentry, impact) it reports the
      nearest POPULATED place — gazetteer.nearest_populated(), merged
      across GNS PPL* and GNIS 'Populated Place', never a creek or a
      ridge — with great-circle distance, 8-point compass direction
      as seen FROM the place, and the GNS UFI / GNIS id, e.g.
      "Impact: ~7 km S of Ebeye (MHL) [GNS:10256638]".  Table view +
      one-click copyable sentence report.  No thinned extract was
      needed — the full index answers in milliseconds off the lat
      index.  Requires the offline index (gated on index_ready with a
      pointer to Analysis ▸ Reference Data).
      EXTENDED (2026-08-18, same session): a second column per event —
      nearest named feature of ANY class (ocean impacts get the
      seamount/trough/island; over land often a stream or hill, shown
      anyway, honestly), classes decoded to words via
      gazetteer.class_word() (NGA designation-code table, raw code
      fallback).  And a second tool, **Cartography ▸ Gazetteer
      Explorer…** (user request: "a light map… zoomable with ALL
      locations"): matplotlib + bundled NE coastlines (no cartopy, no
      network), every zoom/pan re-queries the index for the viewport
      in a worker thread; above a 20k point budget it draws an
      UNBIASED 1-in-k id-modulo sample and says so in the corner
      (gazetteer.viewport_sample(), guess-escalate-deescalate on
      MAX(id)); under it, everything, with names below ~150 in view;
      click-to-identify (class in words, source id, nearest populated
      place); family-coloured dots with per-family toggles ('other'
      catches admin/vegetation/transport — nothing hidden); optional
      ground-track overlay.  Timings on the worldwide index: global
      ~10 s (sampled 1-in-1024), region ~5 s, close-up instant;
  (c) DEM — SHIPPED 2026-08-20.  SOURCE DECISION REVISED (user,
      2026-08-20, after a pros/cons comparison): **AWS Terrarium
      terrain tiles** (elevation-tiles-prod; SRTM/GMTED2010/ETOPO1
      blend, PNG-encoded, zero new deps) replace the 2026-08-16
      GLO-30 choice for BOTH layers — GLO-30's COG format needs
      rasterio/GDAL, and Terrarium's blend has no 60°N cutoff
      (GMTED2010 covers Plesetsk).  GLO-30 (uniform 30 m TanDEM-X,
      ~2–4 m vertical, better void handling — the stronger source
      over African/high-relief terrain) SHIPPED as the third Reference
      Data source (2026-08-23).  The COG-reader blocker was DISPROVEN:
      the copernicus-dem-30m AWS bucket is proxy-reachable (the
      2026-08-20 "unavailable" call was wrong) and its float32 1°×1°
      COGs read with Pillow ALONE — no rasterio/GDAL — lat/lon→pixel
      from the ModelPixelScale/ModelTiepoint tags (poleward thinning
      handled, Plesetsk included).  terrain.py 'glo30' source: whole-
      tile fetch ~40 MB, disk-cached + in-process LRU, miss→Terrarium→
      coarse fallback; wired through MODEL_OPTIONS + Analysis ▸ Reference
      Data.  Governs GUI-side hi-res sampling only; the integrator keeps
      the deterministic offline coarse grid, so every benchmark is
      byte-identical.  test_terrain_dem.py, METHODS §2.5.  Follow-up only
      if 40 MB/tile bites over wide terminal glides: a COG range-reader
      fetching just the ~KB internal tile around a point.
      As shipped: trajectories START at the
      launch site's real altitude and TERMINATE on real ground height
      (integrate_trajectory(terrain_dem=True, launch_elev_m=…);
      "Use terrain (DEM)" checkbox + live pad readout in the Launch
      Site panel; default OFF = flat sea level, byte-identical to the
      Forden benchmarks).  Architecture as agreed: one-time hi-res
      (z11, ~76 m/px·cosφ) elevations baked into launch_sites.json as
      elev_m + elev_source provenance (rebake: `python3 dem_build.py
      sites`); bundled coarse 0.05° global grid (user-chosen
      resolution 2026-08-20; data/dem/terrain_0p05deg.npy, ~52 MB,
      baked reproducibly by dem_build.py from the z5 tile set,
      provenance in data/dem/MANIFEST.md) serves the integrator —
      always offline/deterministic, bilinear + lon-wrap; on-demand
      z11 tiles (disk-cached, coarse fallback on any failure) serve
      GUI-side pad sampling, source selectable under Analysis ▸
      Reference Data ▸ Terrain (DEM) via the MODEL_OPTIONS registry.
      Ocean floor is floored to sea surface (ground = max(elev, 0)).
      The low-altitude GLIDE FLOOR comes free: every HGV glide/bridge/
      equilibrium segment terminates through the same _hit_ground
      event (shared eom_args carry params._terrain_dem), so glides
      floor on real terrain too — verified end-to-end with the
      Minotaur-IV + HTV-2 stack.  terrain.py, tests in
      test_terrain_dem.py, METHODS §2.5;
  (d) air-launched missiles (agreed 2026-08-16, follows from (c)'s
      launch-state generalization): initial state = carrier release
      altitude + speed + flight-path angle instead of a ground pad —
      no vertical liftoff, the kick/loft schedule starts from release
      conditions (the existing launch_elevation_deg / guidance modes
      are the hooks).  DEM ground is then the floor, not the start.
      Scope when picked up: release-condition fields on the flight
      plan, launch-transient handling, and what "range" means measured
      from a moving release point.
  (e) gazetteer map overlays — BUILT 2026-08-18, then WITHDRAWN the
      same day on the user's field test (Canaveral→Atlantic map): the
      basemap tiles already label places, and the overlay cluttered —
      every Caribbean provincial seat at once, labels colliding, the
      US side dark (root causes diagnosed before withdrawal: rank ≤ 1
      grid bypass too generous, label boxes sized at mid-bucket zoom,
      lat-ascending fetch cap starving the bbox's north, and a REAL
      data asymmetry — NGA GNS is foreign-only, so US cities exist
      only as unranked GNIS 'Populated Place' rows and can never
      compete with foreign capitals).  USER DECISION: leave the maps
      to their tiles; the 10.4 M-name index serves lookups instead —
      see 3b2, SHIPPED as Cartography ▸ Nearby Places.  The full
      implementation (map_overlays.py + tests, browser-verified) lives
      in git history at commit 798132c if a map layer is ever wanted
      again.  Original design of record kept below for that case.
      DESIGN OF RECORD (agreed 2026-08-17, user decision:
      TRAJECTORY-AWARE by default on the trajectory map).
      Toggleable per-class layers of gazetteer features on the
      folium trajectory map and the cartopy picker map, each class in
      two flavors: dots, and dots + labels.  Class families: Populated,
      Facilities (GNS-S), Water (GNS-H), Terrain/Islands (GNS-T),
      Undersea (GNS-U).
      PRIORITY SCORE (all terms stated, no population data exists in
      the packs): designation-rank table per class (PPLC > PPLA >
      PPLA2 > PPL; SEA > GULF > BAY > STM; AIRB/INSM > generic) +
      variant-name-count bonus, capped (prominence proxy: places the
      world writes about accumulate romanizations) + CORRIDOR BONUS
      when a trajectory is plotted (≈1 tier within ~100 km of the
      ground track; impact zone prefers water/undersea names; launch
      area prefers facilities).  Corridor term ON by default for the
      trajectory map, OFF for the location-picker map (neutral
      reference there).
      DENSITY: zoom-tiered class gates (explicit table: continental =
      capitals + seas; country = +admin-1 seats, space centers, gulfs,
      major islands; regional = +districts, installations, bays,
      peaks; local = everything) decide ELIGIBILITY → best-per-grid-
      cell (~12×8, top tiers bypass the grid) guarantees SPREAD →
      greedy label-box deconfliction in priority order decides LABELS
      (dot stays when its label is culled; label budget ≈ ⅓ of dot
      budget).  Folium: precomputed zoom-bounded FeatureGroups (static
      HTML, Leaflet handles show/hide), hover tooltips free on every
      dot; cartopy: rules run live per redraw.  REJECTED: folium
      MarkerCluster (count-bubbles hide exactly the names the reader
      needs).

### 5. Move the paper library from GitHub to an organized Drive
STEPS 1–3 DONE 2026-08-17: all 107 data/*.pdf uploaded to the Drive
"Thrusty" folder (flat), verified file-by-file against the repo
(name-complete; sizes spot-checked on the largest volumes), repo copies
deleted, REFERENCES.md gaps 2–3 closed.  HISTORY REWRITE DONE
2026-08-17 (git filter-repo --invert-paths on data/*.pdf, HEAD tree
hash unchanged, all branches force-pushed; clone 730 → 61 MB) and
BRANCH CLEANUP DONE 2026-08-18 (all staging/stale branches deleted
after content verification; only main + the working branch remain).
REMAINING: (a) optionally sort the flat Drive upload into the planned
topic subfolders and add per-file Drive links for the non-core corpus;
(b) Fetterman D-2942/D-2956 PDFs (public NTRS) into Drive — the last
manifest gap.
Original plan follows.
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
  POWER-ON BASE BLEED: DONE (2026-08-22).  base_bleed_ratio = max(0, 1 −
  A_exit/A_base) scales the base-drag term while the active stage fires
  (gated on _eff_burn + global cutoff); coast/reentry and no-nozzle-data
  vehicles are byte-identical; only the decomposed _cd_nose_shape path (not
  the Forden table) gets it.  Isolated effect +~4% range on a solid-motor
  body with A_exit/A_base≈0.5.  test_base_bleed.py; METHODS §8.4.  Side effect
  RESOLVED: the base-bleed energy bump flipped test_pullup's fragile ~1% trough
  comparison, which investigation showed was a CONFOUNDED metric (first-trough
  altitude tracks the trigger altitude itself, not pull effectiveness).
  Re-anchored to the honest guard — a 90 km trigger is inert in the thin-air
  band (speed identical to no-pull-up to ~0.05 m/s) and active only once real
  air returns — which is boost-energy-robust.  test_triggering_too_high_
  conjures_no_lift.
- Biconic boost-phase wave drag — DONE (2026-08-23).  Unshrouded biconic
  bodies/bare RVs now fly a two-cone wave term on ascent (Chin framework:
  fore cone on br² + aft frustum on 1−br², each at its half-angle;
  friction/base unchanged), routed in drag_force_vector via the shared
  biconic_nose_geometry resolver.  Reduces exactly to the single cone at
  θ2=θ1; non-biconic and shrouded vehicles byte-identical.
  test_biconic_front_end.py, METHODS §8.2.
- Interstage / conical-stage flare drag — DONE (2026-08-23).  A conical
  stage or interstage that widens toward the aft (a flare) now adds a
  screening wave-drag increment (_flare_cd / _transition_wave_drag: cone-
  pressure Cp at the flare half-angle × frontal-area step / A_ref, Chin
  framework).  Boattails/same-diameter and plain stacks byte-identical;
  friction not separately counted (front-end model granularity).
  test_transition_drag.py, METHODS §6.7.
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
- Note: deleting data/*.pdf alone would not have shrunk clones (blobs
  stay in history); the deliberate history rewrite that actually shrank
  them was run 2026-08-17 (see item 5).
