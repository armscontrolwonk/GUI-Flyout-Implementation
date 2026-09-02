# Provenance and correction record

Thrusty was developed with AI coding agents between May and September
2026. Raw session logs from that work (`chat_transcript.txt`,
`chat_transcript_full.txt`, `chat_transcript_verbatim.txt`,
`Thrusty_chat_transcript.md`; about 92,000 lines) were committed on
2026-08-21 and removed on 2026-09-02. This file is the curated record of
what those logs established that bears on trusting the numbers in this
repository. It replaces them.

**Why the logs were removed.** They were too long for any maintainer to
have read end to end, so shipping them meant vouching for content nobody
had reviewed. A full pass before removal found no design, targeting, or
countermeasure content in the dialogue, but it did find agent-chosen
test fixtures that used real-world coordinates and a recurring habit of
operational vocabulary for analytic work. None of that survives in the
code or documentation. The substantive value of the logs, the record of
what was claimed, caught, and corrected, is preserved below.

**A note on commit references.** The repository history was rebuilt on
2026-08-21. Commit hashes recorded in the May sessions do not resolve in
this history, so entries below give dates rather than hashes. The
authoritative record for any current value is the module's own git
history from 2026-08-21 onward, `METHODS.md` §16, `BENCHMARKING.md`,
and `data/REFERENCES.md`.

## 1. Corrections to sourced values

Each row is a case where a number or citation entered the work
incorrectly and was caught, either by the maintainer or by a later
verification pass. The last column says where the corrected value lives.

| Item | What was stated | What was wrong | Corrected to | Now in |
|---|---|---|---|---|
| Surface emissivity citation (May) | NASA TM-X-3508, "Apollo Heat Shield Design," Curry & Stephens 1976, ε = 0.79–0.86 | Fabricated by the agent: document number, title, authors, and range were all invented. Caught by the maintainer asking for the source. | Anderson 2006 §18.8 (HERMES example, ε = 0.85) and Williams & Curry 1992, NASA RP-1289 (RCC ε(T) data). Both read in full before re-citing. | `METHODS.md` §13; `ROParams` default `emissivity = 0.85` |
| RP-1289 emissivity value (May) | "0.54 at 3040 °F," attributed to RP-1289 | That figure is from Ohlhorst et al. 2007 (NTRS 20070031768), not RP-1289, whose table ends at 2800 °F (ε = 0.75). | Attribution corrected; Ohlhorst 2007 arc-jet data (0.88–0.91 at 2700–3000 °F) noted as the higher-temperature source. | `METHODS.md` §13 |
| Anderson emissivity value (May) | "Anderson uses ε = 0.8" | Cited from memory. The textbook value is 0.85, §18.8, p. 781. | 0.85 | `METHODS.md` §13 |
| Cone-flank heat flux fraction (May) | "10–20 % of stagnation flux" | Engineering estimate presented without a citation. The fraction is geometry-dependent. | Tauber 1989 Eq. 46: roughly 15–20 % at 10° half-angle, 25–30 % at 15°, 40–50 % at 25°. | `METHODS.md` §13 |
| Grid-fin literature (Aug) | Washington & Miller, DeSpirito, Kantrowitz cited for the grid-fin drag model | Cited from memory without reading the papers. Caught when the maintainer asked, "Did you actually read them?" | Each paper obtained and read; drag model recalibrated against Washington & Miller data. | `METHODS.md` §8.5 |
| Grid-fin chord heuristic (Aug) | Chord proportional to stage diameter, attributed to Kretzschmar & Burkhalter | The agent's own inference, not the paper's finding. | The chord "Estimate" button was reverted rather than shipped on an unsupported ratio. | Not in code |
| Crossflow drag coefficient (Aug) | Constant C_dn = 1.2 in the whole-missile L/D build-up | Unsourced constant; validation against public-domain Digital DATCOM showed a Mach-growing gap up to 20 %. | C_dn(M_n) read from Gowen & Perkins 1953, NACA TN 2960, Fig. 7, via the Jorgensen 1973 NASA TN D-7228 method. η = 1 confirmed by Jorgensen for supersonic free stream. Gap reduced to a roughly constant 10 % conservative bias. | `glider_ld.py`; `METHODS.md` §8.10 |
| Body planform area in L/D build-up (Aug) | Planform taken as a cone-only triangle, ½·L·d | Badly underestimates the side-projected area of a nose-plus-cylinder body. Caught by the DATCOM validation. | Nose-plus-cylinder planform. | `glider_ld.py` |
| C-HGB mass (May; source recorded Sep) | 450 kg, with an empty `source` field | The value was right but unattributed. | Sourced to the U.S. environmental documentation: the FE-2 EA/OEA (Navy SSP, Dec 2019) §2.5.6 uses "up to 454 kg (1,000 lb) of tungsten alloy" for the FE-2 payload analysis; the FT-3 Biological Assessment (Army RCCTO/SMDC, Sep 2020) §2.2.1 gives the FT-3 payload as about 350 kg (750 lb), "similar to" FE-2 but with 10 % of its tungsten. The maintainer reads these as FE-2 = 454 kg and FT-3 = 350 kg, the difference being 90 % of the tungsten. | `ro_library/C-HGB.ro.json`; `data/REFERENCES.md` |
| C-HGB total length (May) | 2.43 m | The source figure's labelled dimensions were mutually inconsistent. | 1.5 m, trusting the labelled half-angles and the 0.58 m base diameter, per the maintainer's correction. The figure's own source is not recorded. | `ro_library/C-HGB.ro.json` |
| C-HGB ballistic coefficient (May) | β = 15,000 kg/m² | No public primary source exists for this value. It is a placeholder carried over from the generic HGB entry. | Unchanged; flagged as a placeholder in the file's `source` field. Use the in-app β estimator as a first-principles check. | `ro_library/C-HGB.ro.json` |
| "Gravity turn" guidance mode (May) | Labelled a gravity turn | Was an east-north-up pitch program, not a velocity-aligned gravity turn. Found when the built-in Minotaur-IV + HTV-2 preset gave a physically implausible range. | Renamed `pitch_program`; a true gravity turn added following Wright 2015, with the preset recalibrated against Wright's published burnout state. | `trajectory.py`; `METHODS.md` |
| Footprint asymmetry diagnosis (May) | Attributed to Coriolis | Wrong; retracted after the maintainer questioned it. Actual cause was sweep ordering in the footprint envelope. | Convex-hull envelope. | `thrusty.py` |

## 2. Process errors

These did not put a wrong number in the code, but they shaped the
working rules in `CLAUDE.md`.

- **Unauthorised fidelity change (2026-05-15).** An agent changed the
  glider lift gate from a latched 100 km descent rule to an instantaneous
  descent test without surfacing the trade-off. The maintainer rejected
  it as an unacceptable fidelity loss and it was reverted the same day.
  Rule: fidelity trade-offs are surfaced before they are implemented.
- **Real-world coordinates in test fixtures (2026-05-09).** While smoke
  testing the dive-at-target field, an agent chose coordinates in
  inhabited China as the aim point and reported a miss distance. No such
  fixture exists in the test suite. Rule: synthetic tests use synthetic
  coordinates; real launch sites and impact zones appear only when
  reproducing a specific documented test event.
- **Refusal of a benign bug report (2026-05-08).** The message "KN-23
  won't skip glide." was refused by the API usage-policy filter. The
  project statement at the top of `CLAUDE.md` exists partly so that
  ordinary simulator work on named vehicles is understood for what it is.

## 3. Where to look instead

- `git log` from 2026-08-21 for the history of any current value.
- `METHODS.md` §16 and `data/REFERENCES.md` for the citation of record
  behind every model.
- `BENCHMARKING.md` for validation against published figures.
- `MASS_ESTIMATOR.md` for the mass estimator, which postdates the logs.
- The `source` field of each file in `booster_library/` and `ro_library/` for vehicle-level provenance. `C-HGB.ro.json` and `STARS-1.booster.json` were sourced on 2026-09-02 to the U.S. environmental documentation held in the maintainer's Drive library.
