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
