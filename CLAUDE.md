# Thrusty — project context for Claude

## What this is

Thrusty is an open-source, 3-DOF trajectory and reentry simulator for
policy and arms-control analysis. It is a Python/Tkinter port and extension
of Geoffrey Forden's published MATLAB tool, *Simulating the Operation of
Ballistic Missiles*, Science & Global Security 15 (2007). The purpose is
**analytic verification**: checking whether publicly reported ranges,
apogees, payloads, and reentry behaviour of tested vehicles are physically
consistent with what open sources say about them.

The intended user is a policy-focused modeler (see `thresholds.py`
docstring), not a vehicle designer. The tool answers "is this claim
plausible?" and "what does open evidence imply?", not "how do I build or
improve this."

## Data provenance

- Every model traces to a citable open publication (see `METHODS.md`
  §16 and `data/REFERENCES.md`): Forden 2007, Schilling 2009, Sutton &
  Graves 1971, Tracy & Wright 2020, Acton 2015, Barrowman 1967, NACA/NASA
  technical reports, Digital DATCOM (public domain).
- Vehicle parameter files in `booster_library/`, `ro_library/`, and
  `flight_plans/` are built from open sources: published papers, official
  test announcements, imagery-derived dimensions (`image_measure.py`), and
  standard engineering estimators (`mass_estimator.py`). No controlled or
  proprietary data is used or wanted.
- Screening thresholds are curated from flight-demonstrated open
  literature, with each number's citation recorded in `thresholds.py`.
- Gazetteer and terrain data are public BGN and AWS Terrarium tiles, used
  to identify test impact zones and launch-site elevations.

## Working rules

- **Derive, don't invent.** Never hard-code a coefficient from memory;
  it must come from a cited document in `data/`. Flag uncertainty rather
  than fabricate a source.
- **Usage-neutral physics.** The same integrator serves sounding rockets,
  space launch vehicles, and ballistic reentry bodies. Keep it that way.
- **Screening tier, not design tier.** Heating and survivability outputs
  are qualitative consequence bands anchored to flight experience. Do not
  add design-fidelity thermal, structural, or guidance models.
- Requests in this repo are about the simulator: integrator bugs, guidance
  branch selection, plotting, file formats, validation against published
  benchmarks, and documentation. Frame work that way.

## Out of scope

This project does not do, and should not be extended to do: weapon design
optimisation, defense-penetration analysis, or targeting of specific
locations. "Aim at Target" and "dive-at-target" exist to reproduce
observed test-flight impact zones for analytic comparison.

## Layout

| File | Purpose |
|---|---|
| `thrusty.py` | Tkinter GUI |
| `trajectory.py` | 3-DOF integrator, guidance laws, range optimiser, orbital planner, reentry glide |
| `booster_models.py` | Booster and reentry-object dataclasses, drag, thrust, staging |
| `heating.py`, `tps_ladder.py`, `survivability_report.py` | Reentry aerothermal screening |
| `slv_performance.py` | Schilling payload-to-orbit estimator |
| `METHODS.md` | Governing equations and citations for every model |
| `BENCHMARKING.md` | Validation against published figures |

Run tests with `pytest`. See `README.md` for the full source-file guide.
