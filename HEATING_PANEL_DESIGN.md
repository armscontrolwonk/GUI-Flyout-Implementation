# Heating survivability panel: a mode-keyed report + judgement

Working concept for a Townsend/Schilling-style **heating report** — a table and
plot keyed to the reentry mode, ending in a stated survival *judgement with
consequences* (accuracy degradation, burn-through time), not a bare pass/fail.
Status: **concept for discussion, pre-implementation.**  Companion to
`HEATING_MODEL_CROSSCHECK.md` (the physics/source backbone) and the SLV tab
(`thrusty.py` `_slv_*`, the presentation model to mirror).

---

## 1. The idea, by analogy to the SLV tab

The SLV Performance tab takes a booster, runs the Schilling/Townsend algebraic
ΔV budget, prints a **table** (ΔV available vs required, margin, payload,
timing parameters), and ends with a **boolean verdict** (`can_reach_orbit`)
plus a stated method accuracy.  The heating panel should read the same way for
a reentry object:

> take an object + its reentry plan → run the trajectory → compute the
> heating profile → print a mode-keyed table + plot → end with a judgement:
> *survives / survives-degraded / fails*, and **what that means**.

The engine (`heating.py`) already produces most of the numbers; the missing
piece is the **report layer** — selecting which numbers matter for this mode,
and turning margins into a consequence sentence.

## 2. The judgement is mode-shaped (the core design claim)

The reentry mode decides which physical failure question is even being asked,
so the panel must key its table columns, its plot, and its verdict language to
the mode's binding measure.  Three classes (from `HEATING_MODEL_CROSSCHECK.md`
§0, already the module's internal structure):

| Class (mode) | Binding measure | Verdict axis | Consequence language |
|---|---|---|---|
| **Ballistic RV** (`ballistic`) | peak flux **and** integrated load (trade off across loft/depress) | nose recession δ/R_n; heat-sink Q vs Q_melt | "recedes X cm (Y·R_n) → *shape-change/accuracy* onset at δ/R_n≈0.1; blunting at 0.5–1; burn-through at overhang" |
| **Glider / HGV** (`skip`/`damped`/`dynamic`/analytic) | **duration** — soak / oxidation-dwell (the stopwatch) | survival-time vs glide-time; bondline | "hot structure holds T_w K for the N s glide" **or** "exceeds oxidation limit after M s < glide time → fails at M s" |
| **Maneuvering quasi-ballistic** (glide + terminal dive; Hwasong-11) | transient low-altitude **pull-up flux spike** (heat-sink) | fin-LE/flank peak vs airframe limit | "terminal pull-up spikes to X MW/m² at H km → windward flank/airframe, not nose" |

The mode is already the plan's identity (family + law), so the panel *knows*
its class without asking.  Ballistic → the loft/MET trade (peak-vs-load) is the
headline; glider → the stopwatch (survival-time vs glide-time) is the headline;
MaRV → the terminal-dive spike is the headline.

## 3. The report — three sections (mirrors the SLV body string)

### 3.1 Header — what was flown
```
Reentry object:  C-HGB  (biconic, R_n 2 cm, ⌀ 0.58 m)
Reentry mode:    Damped phugoid glide   [numerical family]
Entry (100 km):  V 6.2 km/s   γ −8.4°   → glide, captured
TPS:             nose UHTC · body carbon-phenolic 2.0 cm
```

### 3.2 Profile table — the mode-keyed numbers
Ballistic RV (loft/MET/depress framing — the peak-vs-load trade this question
turns on):
```
                    peak q̇     pulse    load Q    peak T_eq   arc     nose recession
  This trajectory   9.8 MW/m²   8 s      97 MJ     3773 K      43 s    0.5 cm (0.10 R_n)
  (context: a depressed shot to the same range → 7.2 MW/m², 138 MJ — lower flux, MORE load)
```
Glider (the stopwatch — survival-time vs glide-time, per location):
```
  location   material          peak T_eq   binds at        vs glide 1434 s
  nose       UHTC              (needs ablation analysis > 4000 K screen)
  body       carbon-phenolic   2136 K      recession 5% depth   survives the glide
```

### 3.3 Judgement — the consequence sentence
Not "FAIL" but *what fails and what it costs*:
- Ballistic: "Nose recedes ≈0.5 cm (0.10 R_n) — at the shape-change onset:
  **accuracy degradation likely** (asymmetric recession → trim → dispersion,
  PANT/Lin), survival not threatened.  A lofted shot to the same range raises
  peak flux ~35%; a depressed shot raises integrated load ~40%."
- Glider: "Body TPS (carbon-phenolic 2.0 cm) holds for the full 1434 s glide.
  **Nose exceeds the reradiative screen (>4000 K)** — a UHTC tip needs an
  ablation/oxidation-life analysis this screen can't provide."
- Glider fail: "Oxidation-dwell exceeded after ≈900 s < 1434 s glide time →
  **fails ~530 s before impact**; needs a higher-limit tip or a shorter glide."

## 4. The plot — flux(t) and load(t), the mode reads off the shape
Two shared axes, both regimes on the same picture (the NAS-session
"apples-to-apples" invariant):
- **q̇(t)** — a single hump for a smooth glide/analytic; **phugoid teeth** for
  skip/damped (each trough dips into denser air → a flux spike); a **late
  spike** for a MaRV terminal dive.  The pulse *shape* is the mode's signature.
- **Q(t)** — the running integral; its final value is the ablator-sizing number;
  its slope shows where the soak accumulates.
- Overlays: peak-flux point, the material's limit line(s), and (glider) the
  glide-time vs survival-time bars.  This is the visual twin of the NAS
  load-vs-flux figure, but from the *real* EOM so the phugoid teeth show.

## 5. What exists vs what the panel needs

**Exists** (`heating.py`, wired to the "Heating Survivability" tab):
- Sutton-Graves stagnation flux, radiative-eq T_eq, Q, peak, duration.
- Two-location (nose + body-acreage) FOM, binding-location selection.
- Three criteria: peak-surface, soak-dwell, heat-sink; ablator→recession branch
  with δ/R_n shape-change bands (§10.2); NRC duration ladder (`tps_ladder`).
- Material catalog with limits, ablator H_eff, oxidation dwell.

**Needs building** (the report layer + a few physics gaps):
1. **Mode-keyed report assembler** — pick the columns/verdict language from the
   plan's class; the SLV-style body string + verdict.  (Pure presentation over
   existing numbers — the cheapest high-value piece.)
2. **flux(t)/load(t) plot** — two columns Thrusty already has per step; a new
   subplot.  (Cheap; the phugoid teeth are the payoff.)
3. **Loft/MET/depress context line** for ballistic — one auto-comparison run at
   the same range, different burnout angle, to state the peak-vs-load trade
   quantitatively.  (A small sweep; the lofted-vs-MET question made concrete.)
4. **Consequence mapping** — recession δ/R_n → accuracy/dispersion band;
   oxidation-dwell shortfall → "fails N s before impact".  (Bands already in
   §10.2; this is wording + the dwell-vs-glide-time compare.)
5. *(physics, later)* hot-wall correction (glide optimism); windward-flank probe
   at AoA (MaRV/glider); terminal-dive second pulse surfaced separately.

## 6. Benchmarking hooks (user offered to help)
Each verdict cites a flight anchor so a number can be checked:
- Ballistic recession: **Reentry-F** (0.7 R_n radial / 7.7 R_n axial, flew its
  mission) and **Lin/TRW-SCATHE** (0.1 R_n at 67 kft → "mildly indented").
- Glider hot structure: **HTV-2** (~1900 °C surface / 1090 °C structure for
  3600 s) and the **NRC 300/800/3000 s** ladder.
- Accuracy onset: **PANT** (asymmetric recession → dispersion below 0.7 R_n).
The panel prints the anchor next to the verdict so benchmarking is a read, not
a code dive.

## 7. Sequencing
- **P1 (report + plot):** items 1–2 — reshape existing numbers into the
  mode-keyed table + the flux/load plot + a first consequence sentence.  No new
  physics; immediately useful.
- **P2 (context + consequence):** items 3–4 — the loft/MET trade line and the
  recession→accuracy / dwell→fail-time wording, with flight anchors.
- **P3 (physics):** item 5 — hot-wall, windward AoA probe, terminal-dive pulse,
  as the accuracy of the numbers demands.

## 8. Open questions for the user
1. Report home: extend the existing **Heating Survivability** tab, or a new
   **Reentry Survivability** tab paralleling SLV Performance?
2. For ballistic, is the **loft/MET/depress auto-context** (one extra run) worth
   the compute, or should the trade be shown only when the user sweeps?
3. Accuracy overlay: report recession→dispersion **qualitatively** (band words:
   "shape-change onset / blunting / burn-through") for P1/P2, deferring a
   quantitative CEP-growth number to a later tier?  (Recommend qualitative
   first — the quantitative map needs the shape→drag→dispersion loop the
   crosscheck flags as its own uncertainty.)
