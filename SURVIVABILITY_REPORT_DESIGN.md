# Reentry Survivability Report — working concept

The down-leg analogue of the Schilling/Townsend SLV panel: given a reentry
object's **profile** (geometry, materials, thicknesses) and its **reentry mode**,
Thrusty reports a mode-keyed table and a flux/load plot, and then **expresses a
judgement with consequences** — not survive/fail, but *what kind* of failure and
*when*: "nose ablation of this size/material likely degrades accuracy on this
ballistic trajectory," or "this TPS thickness of this type likely fails after
~N s of a M-s glide."  Status: **working concept, merged and approved
direction** — the band constants are explicitly the user-benchmarkable part.

Resolved decisions (user):
1. **New "Reentry Survivability" tab** paralleling SLV Performance; the current
   Heating Survivability tab stays until the new one proves out, then is
   deleted.
2. **Loft/MET context only on sweep** — no auto-comparison run per flyout.  The
   Parametric Sweep gains heating outputs (below) so the loft/depress trade is
   read off a sweep, exactly like range.
3. **Qualitative consequence bands first**; refine with data (user benchmarks).

Companions: `HEATING_MODEL_CROSSCHECK.md` (the verified physics + the three
vehicle classes), `HEATING_DATAFLOW.md`, the NRC-2008 session transcript (the
duration-ladder logic), `REENTRY_FAMILY_DESIGN.md` (mode identity).

---

## 1. The Schilling parallel (what "working concept" means here)

The SLV panel works because it commits to a *shape*: inputs echoed → a budget
table → a signed margin → a verdict, with method accuracy and the reference
stated at the bottom.  The survivability report commits to the same shape:

```
inputs echoed        object profile + trajectory/mode summary
budget table         the mode's binding quantities (flux, load, T, time)
signed margin        per-criterion margins, worst named
judgement            a consequence sentence (accuracy / time-to-failure / envelope)
method + refs        screening-tier honesty line + the anchors used
```

One report **form per vehicle class**, keyed automatically from the plan
(mode/family) and profile — because the classes bind on different physics
(crosscheck §0): a ballistic RV on peak-flux+load, a glider on duration, a
maneuvering quasi-ballistic on the transient pull-up spike.

## 2. Class keying (automatic, from data the run already has)

| Class | Key | Report form |
|---|---|---|
| **Ballistic RV** | glider disabled (mode `ballistic`) | A |
| **Glider / HGV** | numerical or analytic glide, no terminal pull-up emphasis | B |
| **Maneuvering quasi-ballistic (MaRV)** | glide with terminal dive / dive-at-target / pull-up-dominated profile (Hwasong-11 class) | C |

(C is form B plus the transient-pulse block; it ships after A and B.)

## 3. Form A — Ballistic RV: the accuracy ladder

The novelty vs today's screen: recession is judged on the **δ/R_n consequence
ladder** (flight-anchored, crosscheck §10.2), and the verdict names accuracy
before it names survival — a vehicle can pass survival and fail accuracy, and
accuracy fails first.

**Table:** entry conditions (V, γ at 100 km — names the loft/MET shaping),
peak flux + pulse width (FWHM), integrated load Q, peak T_eq, and per-location
(nose/body) material margins as today.

**Judgement ladder (δ/R_n, symmetric-recession proxy for the PANT asymmetric-
recession dispersion mechanism — stated honestly in the method line):**

| δ/R_n | Band | Consequence sentence | Anchor |
|---|---|---|---|
| < 0.1 | NOMINAL | "shape change negligible — accuracy preserved" | Lin 1982 ("mildly indented" at 0.1) |
| 0.1 – 0.5 | ACCURACY-DEGRADED | "shape-change onset — dispersion growth likely (CEP degradation), vehicle survives" | PANT (ADA019186); Lin 1982 |
| 0.5 – 1.0 | SEVERE BLUNTING | "large shape change — survivable (Reentry-F flew at ~0.7) but accuracy heavily degraded; β falls, range/dispersion shift" | Reentry-F (NASA CR-154044) |
| > overhang/R_n | BURN-THROUGH | "nosetip consumed at t≈…s — failure" | solid-tip length (Reentry-F ~7.7 R_n axial) |

Plus the trajectory-shaping note **when a sweep provides it** (decision 2): at
matched range, lofted raises peak flux and depressed raises load (No-dong pair:
+36% flux lofted, +42% load depressed) — the report says **which axis this
trajectory stressed** when sweep data exists, and omits the line otherwise.

**Mock (real numbers — No-dong + Generic RV, carbon-phenolic, R_n 5 cm, 744 km):**

```
─── Entry (ballistic) ──────────────────────────────────────
  Trajectory shaping:   LOFTED (burnout 47°; apogee 356 km)
  Entry at 100 km:      2.52 km/s at γ = −57°
  vs MET to same range: peak flux +36%, load −30%  (lofted = flux-stressed)
─── Heating budget ─────────────────────────────────────────
  Peak stagnation flux:  9.8 MW/m²   (pulse width 8 s)
  Integrated load:        97 MJ/m²
  Peak T_eq:            3,773 K      (ablator: informational)
─── Nose recession (carbon-phenolic, R_n 5.0 cm) ───────────
  Recession δ:           0.4 cm   →  δ/R_n = 0.08
  Band:                  NOMINAL  (< 0.1 shape-change onset)
─── Judgement ──────────────────────────────────────────────
  SURVIVES, ACCURACY PRESERVED.  Margin to accuracy band: δ/R_n
  0.08 vs 0.1 onset.  On the depressed trajectory to the same
  target the load rises ~40% → δ/R_n ≈ 0.11: shape-change onset —
  expect dispersion growth.  Screening tier: symmetric-recession
  proxy (PANT mechanism is asymmetric); Sutton-Graves cold-wall.
  Refs: Reentry-F, PANT, Lin 1982, Schneider 1972.
```

## 4. Form B — Glider: the stopwatch

The NRC-2008 insight *is* the report design: every glide reduces to a
measurement of time, so the glider verdict is expressed in **seconds** — the
time-to-failure t_fail (earliest criterion crossing, which `heating.py` already
computes as the compromise point) against the glide duration t_glide the
mission needs, placed on the NRC 300/800/3,000-s ladder (`tps_ladder`, already
wired).

**Table:** glide duration + mode (and family-honesty flag if analytic — an
idealized smooth capture reads peak flux 2–4× below the as-flown numerical
mode), peak flux, sustained T_w, integrated load, per-location margins,
oxidation-dwell, NRC-tier placement.

**Judgement:** one of —
- "TPS survives the full N-s glide (margin: fails at ~M s, ×1.8 the mission)."
- "TPS likely fails at t≈M s of an N-s glide (bondline soak) — thermal range
  ~X km of the Y-km aero range."  ← *the thermal-range cap: min(aero range,
  thermal range), crosscheck §0 — the single most decision-relevant number.*
- "Ablative nose/LE on a glider: any meaningful recession corrupts the
  aeroshape — needs a non-ablating tip (UHTC-class)." (SWERVE→AHW rule; any
  δ/R_n ≥ ~0.05 on a glider flags this regardless of survival.)

**Mock (real numbers — C-HGB, damped glide, UHTC nose R_n 2 cm, CP body):**

```
─── Glide (numerical: damped phugoid, ζ=0.7) ───────────────
  Glide duration:       1,434 s     range 6,295 km
  NRC TPS ladder:       past ADVANCED (3,000-s C/C tier; > 800-s ablative wall)
─── Heating budget ─────────────────────────────────────────
  Nose (UHTC, R_n 2 cm):  peak 29.4 MW/m², T_eq 4,968 K
  Body (carbon-phenolic): peak  1.0 MW/m², T_eq 2,136 K, Q 209 MJ/m²
─── Per-location margins ───────────────────────────────────
  Body: peak-surface margin 0.55, recession 5% of layer  → holds full glide
  Nose: T_eq 4,968 K ≥ 4,000 K reradiative screen        → beyond screening
─── Judgement ──────────────────────────────────────────────
  BODY SURVIVES THE FULL 1,434-s GLIDE (recedes 5% of layer;
  worst margin 0.55 on peak surface).  NOSE IS BEYOND THE
  SCREENING MODEL: a UHTC tip at ~5,000 K equilibrium needs an
  ablation/oxidation-life analysis this tier cannot provide —
  the 2-cm tip radius is the driver (q̇ ∝ 1/√R_n).  Numerical
  (as-flown) mode: phugoid troughs set the peak; an analytic run
  of this plan would read ~3× lower peak flux (family-honesty).
  Refs: NRC-2008 tiers, HTV-2 anchor (~1,900 °C / 3,600 s),
  Murbach/SWERVE materials.
```

## 5. Form C — Maneuvering quasi-ballistic (MaRV): the envelope

Form B plus a **transient-pulse block** for the terminal dive / pull-up
(Hwasong-11 class, crosscheck §0 row 3): the binding event is a low-altitude
high-q̄ flux spike on the **windward flank / fin leading edge**, not the nose,
and it is heat-sink-limited (too fast for equilibrium).  Screening version
(ships with the existing stagnation trace): isolate the dive segment, report
its peak flux/duration separately from the glide budget, and judge against the
airframe/hot-structure limit rather than the nose material.  Verdict language
is a **maneuver envelope**: "the commanded dive spikes X MW/m² at H km — within
/ beyond the airframe class; a steeper dive-at-target trigger would exceed it."
Full fidelity needs the windward/AoA probe (P3).

## 6. The plot — flux(t) and load(t)

Two curves on the reentry arc, the visual twin of the NRC load-vs-flux figure
but from the real EOM:
- **q̇(t)** — a single hump (ballistic), a plateau (equilibrium glide),
  **phugoid teeth** (skip/damped — each trough is a flux spike), or a **late
  terminal-dive spike** (MaRV).  The pulse *shape* is the mode's signature.
- **Q(t)** — the running integral (the ablator-sizing number; its slope shows
  where the soak accumulates).
- Overlays: material limit line(s), t_fail marker, and (glider) glide-time vs
  survival-time bars; NRC tier durations as reference ticks.

## 7. Sweep integration (decision 2)

The Parametric Sweep (`ParametricSweepDialog`) already sweeps **Burnout Angle /
Turn Stop / Azimuth / Cutoff** and records (param, range, apogee) with a live
plot + table, storing each run's trajectory.  Extension: record **q̇_peak and
Q** per point (one FOM call on each stored arc) and offer them as the plotted
y-axis — "flux and load across burnout angles and turn stops, exactly like the
range sweep."  This is where the loft/MET trade lives: the flux curve rises
with burnout angle while the load curve falls, and the crossing IS the trade
made visible.  Form A's "which axis this trajectory stressed" line pulls from
the most recent sweep when one exists.

## 8. What exists vs what the report needs

**Exists** (`heating.py` + `tps_ladder`, wired to the current tab):
Sutton-Graves flux / T_eq / Q / duration; two-location FOM with binding
location; three criteria + ablator recession branch with δ/R_n bands; material
catalog with limits and oxidation dwell; NRC ladder; compromise (t_fail) point.

**Needs building:**
1. **Report assembler** (new module, `survivability_report.py`): class keying
   from the plan, the three forms' body strings, consequence-band wording with
   anchors.  Pure presentation over existing numbers.
2. **New tab** ("Reentry Survivability", parallel to SLV Performance) hosting
   the report + the flux/load plot.  Old Heating tab untouched until proven.
3. **Sweep heating outputs** (§7).
4. **Consequence mapping**: δ/R_n ladder (Form A), t_fail vs t_glide + thermal
   range (Form B), dive-spike vs airframe (Form C screening).
5. *(physics, later — P3)*: hot-wall correction (removes the glide-side
   cold-wall conservatism), windward/AoA flank probe (Form C fidelity, conical
   glider acreage), terminal-dive pulse extraction refinement, transition
   criterion (makes the 300<800<3,000 ordering laminar/turbulent-honest).

## 9. Sequencing

- **P1 — report + plot + tab** (items 1, 2): reshape existing numbers into the
  mode-keyed forms with qualitative bands; flux/load plot.  No new physics.
- **P2 — sweep + consequence wording** (items 3, 4): heating sweep axes; the
  loft/MET context line fed from sweeps; band sentences with flight anchors.
  → **User benchmarks the bands here.**
- **P3 — physics** (item 5), as the numbers' accuracy demands.

## 10. Benchmarking hooks (the user-tunable part)

Every judgement cites its anchor inline so checking a number is a read, not a
code dive:

| Constant | Value (screening) | Anchor |
|---|---|---|
| shape-change onset | δ/R_n ≈ 0.1 | Lin 1982 "mildly indented" |
| severe blunting | δ/R_n 0.5–1.0 | Reentry-F flew ~0.7 |
| burn-through | δ > nose_solid_depth (≈R_n default) | Reentry-F 7.7 R_n axial |
| glider ablative-tip flag | δ/R_n ≥ ~0.05 | SWERVE→AHW UHTC rule |
| ablative-tier wall | ~800–1,000 s | NRC-2008 (CSM-1 vs CSM-2) |
| hot-structure anchor | ~1,900 °C surface / 1,090 °C · 3,600 s structure | HTV-2 |
| family-honesty factor | analytic peak ×(2–4) low | this session's C-HGB runs |
| acreage flux fraction | 0.13 × body-scale stagnation | Lu/Shi & Zhang 2024 |

## 11. Living survivability envelope (UHTC hot-structure gliders)

The screening constants in §10 are point values.  For the UHTC sharp-tip / hot-
structure case (Form B gliders), the flight and arc-jet record is too sparse and
too *one-sided* to support a single pass/fail number — it can only support a
statement about how much of a trajectory lies **inside what has been
demonstrated**.  This section defines that model.  It supersedes the single
`oxidation_dwell_s` cliff for the `uhtc` material; the other Form A / Form C
bands in §10 are unaffected.

### 11.1 The data is a floor, not a fence

Nearly every UHTC survivability datum we have (BENCHMARKING.md §UHTC) is a
**survival**: the arc-jet or flight test *stopped*, it did not *fail*.  So each
point is a **lower bound** — "at least this hot, this long, with this tip
radius, is fine" — and on the aero-convective side the demonstrated region is a
floor with **almost no ceiling**.  We know where it is safe; we are largely
guessing where it breaks.

The two outcomes bound the envelope from opposite sides:

* **Survival → lower bound** (extends the floor).  Most current anchors.
* **Failure → upper bound** (caps the ceiling).  We now have several caps, each
  narrow and to be applied only to its own regime and material class:
  * **Levine 2003 furnace failures** (NTRS 20040033992): ZrB₂-SiC-TaSi₂
    melted/slumped at 1927 °C; ZrB₂-SiC-TaC breakaway oxidation at 1627 °C.
    1-atm stagnant cyclic furnace — they cap the passive-oxidation regime for
    the *doped* classes, not the aero-convective regime, and not plain
    ZrB₂-SiC (which survived the same 1927 °C exposure).
  * **Two aero-convective oxide-detachment caps that converge on one mode.**
    The **Di Maso 2009** HfB₂-15TaSi₂ sharp cone (oxide detached from the bulk
    at ~2800 K CFD tip / 2279 K pyro, PhD thesis, Univ. Naples) and the
    **De Prisco 2026** ZrB₂-TiB₂-SiC hemispheres doped NbC/VC (both tips
    detached after 2700 K, Mach 3, ~10 MW/m²; *J. Eur. Ceram. Soc.* 46, 118184)
    fail the *same way* — oxide-scale detachment / poor adherence to the
    unreacted bulk at ~2700–2800 K — across two labs, two classes, two
    geometries.  So for doped/complex diborides the ~2700 °C-class limit is
    **oxide adherence, not melt or burn-through**.  Both cap doped/complex
    classes; the additive-inversion rule forbids averaging them into
    plain-diboride envelopes.
  * **Marschall 2012 — the plain-ZrB₂-SiC passive→active (PA) transition**
    (*J. Thermophys. Heat Transfer* 26(4), VKI Plasmatron): the keystone cap.
    A *plain* (undoped) ZrB₂-SiC surface holds a stable protective silica scale
    up to a threshold, then loses it — SiC oxidises actively, the chemical heat
    flux surges, and the surface **jumps +400 K in 20–30 s** (a self-amplifying
    runaway).  At 10 kPa the boundary is sharp: q_cw ≈ 202 W/cm² (2.02 MW/m²,
    ~2215 K steady) triggered it; 185 W/cm² did not.  This is the first
    **plain-diboride** aero-convective cap — the gap the additive-inversion rule
    said we could not fill from doped data — and it is **flight-corroborated**
    (SHARP-B1: 2360→2810 K jump in ~15 s, Kolodziej et al.).  It is also the
    *mechanism* under the whole model: the loss of the borosilicate glass that
    `continuous_K` is about.
  * SHARP-B2 segment failures remain excluded: **material-quality**, not a
    temperature/dwell limit.
  The remaining, narrower gap is a *total burn-through* of a plain tip (the PA
  transition is runaway *onset*) and an aero-convective cap specifically for
  HfB₂-SiC.

  Two anchors **isolate the pressure axis** (§11.6), so it is not hypothetical:
  the De Prisco pair (same specimens survived 1700–1800 K at 3×10⁻³ atm but
  detached at 2700 K under 10× the pressure), and Marschall's PA threshold is
  quoted *at a stated 10 kPa* because the transition is pO₂-dependent.

The report must therefore never say "survives" beyond the floor.  It says
"within the demonstrated envelope" up to the floor, and "beyond validated
data — extrapolation" past it.

### 11.2 The envelope is derived from an anchor dataset, not hardcoded

So that "a new flight strengthens the dataset" is a **data edit, not a code
change**, the anchors live in a structured, extensible table (BENCHMARKING.md
§UHTC seeds it; long-term it becomes a JSON/CSV the report reads).  One record
per flight/arc-jet/plasma-torch datum:

| Field | Meaning |
|---|---|
| `id` | e.g. `Monteverde-2012-ZS`, `SHARP-B2` |
| `material_class` | `zrb2_sic`, `hfb2_sic`, `carbide_boride`, … (the class, not an exact recipe) |
| `kind` | `flight` \| `arcjet` \| `plasma_torch` \| `furnace` |
| `tip_radius_m` | leading-edge / nose radius (sharp vs blunt matters) |
| `flux_MW_m2`, `flux_kind` | `cold_wall` \| `hot_wall_net`; enthalpy `MJ_kg`; `stag_pressure_Pa` |
| `peak_T_C`, `T_source` | `measured` \| `cfd` (Monteverde's 2450 °C tip is CFD, not pyrometer) |
| `dwell_s` | time held above the 1650 °C glass ceiling |
| `recession_um`, `mass_change_pct` | observed degradation (negative recession = net oxide growth) |
| `outcome` | `survived` \| `degraded` \| `failed` + `failure_mode` |
| `source` | exact citation (never paraphrased) |

The envelope for a class is then the bounding region of its points in
(peak_T, dwell) space — survivals as the reachable floor, failures as the cap.
The report cites *which* anchor(s) bound the current verdict.

### 11.3 Envelope-coverage report model (Form B)

The model already produces `T_eq(t)` along the glide arc (radiative-equilibrium
from Sutton-Graves q̇).  Against the class envelope it shades the trajectory:

* **Green — protected.**  `T_eq ≤ continuous_K` (the 1650 °C / 1923 K silica-
  glass ceiling; §11.4).  No dwell clock runs; indefinite for any glide length.
* **Amber — demonstrated-with-recession.**  `continuous_K < T_eq ≤ peak
  demonstrated tip temperature`, while cumulative dwell above the ceiling is
  within the demonstrated floor.  Inside the envelope, consuming recession
  margin.
* **Red — extrapolation.**  Left the envelope by **one of two named exits**,
  reported distinctly because they have different fixes:
  * **too hot** — the surface crosses the **passive→active (PA) oxidation
    boundary** for the class: the protective silica is lost and heating runs
    away (Marschall's +400 K jump).  This edge is a **flux/pressure surface,
    not a single temperature** — Marschall's plain ZrB₂-SiC went active at
    ~2215 K / 2 MW/m² / 10 kPa (flat face), while a sharp conducting tip stayed
    passive to 2450 °C at 7 MW/m² (Monteverde).  So "too hot" is evaluated
    against the PA threshold at *this run's* flux and pressure, not a fixed
    peak.  Fix: loft / blunt tip / lower flux.
  * **too long** — dwell above the glass ceiling outruns the demonstrated floor
    at that temperature → shorten exposure.

The verdict is a **coverage fraction plus the beyond-envelope segment**, not a
boolean.  Example: *"Nose above 1650 °C for 420 s: first ~300 s within the
demonstrated ZrB₂-SiC envelope (Monteverde 2013, 1973 K · 300 s), remaining
~120 s at 1900–2050 °C is beyond validated dwell — extrapolation."*  On the
flux/load plot this is a green/amber/red band along the arc.

### 11.4 The two thresholds and the dwell floor, defined precisely

Plain ZrB₂-SiC has **two distinct thermal thresholds**, not one; the model must
carry both:

1. **`continuous_K` = 1650 °C (1923 K)** — the borosilicate-glass
   *protectiveness* ceiling, confirmed by ≥5 independent sources (Monteverde
   2012, Peters 2024, Fahrenholtz & Hilmas, Marschall, Li).  Below it, no clock;
   the glass is stable and long-duration-protective.  (Was 1900 °C / a single
   rough value.)
2. **The passive→active (PA) transition** — the *upper* edge, where the glass is
   lost and oxidation runs away (Marschall's +400 K jump).  Not a fixed
   temperature: a **flux/pressure surface**, anchored at ~2215 K / 2.02 MW/m²
   cold-wall / 10 kPa for a flat plain-ZrB₂-SiC face, and pushed higher on a
   sharp conducting tip (passive to 2450 °C at 7 MW/m², Monteverde).  This is
   the "too hot" red exit (§11.3).

Between the two — the **amber band** — the material survives with recession,
bounded in time by:

* **`oxidation_dwell_s` = the demonstrated floor**, conservatively the low
  anchor (~300 s at 1973 K, zero recession — Monteverde 2013; sharp-tip
  survival extends to ~575 s at 2450 °C — Monteverde 2012).  Re-annotated as a
  **conservative floor, not a cliff**: within it → inside the envelope; past it
  → the report flags extrapolation, it does not assert failure.  (Was `120`,
  used as a hard fail line.)

The honest sharp-tip criterion is ultimately a **flux-normalized recession
rate** (≈0.07 µm/s at 7 MW/m² sharp; ≈3.6 µm/s at 26 MW/m² blunt) that blunts
the tip and degrades sharpness/accuracy — mapping onto the Form A δ/R_n ladder
applied to the tip radius.  That rework is a later step; the floor + coverage
shading is the near-term model.

### 11.5 Guessing the material, and the update loop

When something new flies, the exact TPS is usually unknown.  The tool brackets
by **material class** and shows where the observed (peak_T, dwell, outcome)
lands against that class's envelope.  The outcome then does one of three things,
and the report names which:

* **extends** the floor — survived past the current envelope → "this observation
  grows the demonstrated envelope; add the record."
* **caps** the ceiling — failed → the first real upper bound for the class.
* **contradicts** — failed *inside* the demonstrated envelope → a surprise:
  wrong material guess, or a regime effect (see §11.6) worth flagging.

Either way the dataset gets stronger, which is the whole point of keeping the
anchors as data.

### 11.6 The standing asterisk: pressure

The entire demonstrated envelope is **low-pressure ground testing** (~0.08–0.2
atm stagnation).  The SiC active/passive oxidation transition is pressure-
sensitive, so a real flight stagnation can sit on the other side of it.  Even
green/amber coverage carries a "demonstrated at ground-facility pressure"
footnote — it is not a clean guarantee.  The one regime none of the anchors
reaches is the actual HGV case: a sharp UHTC tip held at **1700–2000 °C for
1000 s+ at flight pressure**.  The data brackets it (1973 K · 300 s survived;
2450 °C · 575 s survived) but does not contain it; the report should say so
rather than interpolate silently.

### 11.7 Implementation steps (code, next)

1. `heating.py`: retune the `uhtc` entry per §11.4 (`continuous_K` 1900→1650 °C;
   `oxidation_dwell_s` re-annotated as a floor with the anchor citations).
   Verdict-affecting → re-baseline the shipped-object survivability runs.
2. Anchor dataset: seed from BENCHMARKING.md §UHTC (inline dict first; promote
   to a read-at-startup file once the schema settles).
3. `survivability_report.py` Form B: replace the boolean dwell verdict with the
   green/amber/red coverage shading (§11.3), the two named exits, the coverage
   fraction headline, and inline anchor citation.
4. Reentry Survivability tab: shade the flux/load plot band; render the coverage
   line + pressure asterisk.
