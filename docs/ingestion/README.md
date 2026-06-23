# Thrusty model ingestion — design docs

Planning artifacts for populating Thrusty with a community-grown catalog of
missile models. **Design only — none of this is wired into the app yet.**

## The problem
- **Two authoring workflows:** (A) Claude ingests documents and builds models;
  (B) users author their own, mainly in spreadsheets.
- **A growing community catalog:** many preloaded models, held in a separate
  versioned library, imported by others.
- **Incomplete data is the norm:** missing values get *fixed during ingest*
  (estimated + flagged), not rejected.

## The pipeline
```
 source (doc / xlsx / CSV / Google Sheet)
   → parse → normalize units (→SI) → RESOLVE (fill gaps) → VALIDATE (sanity)
   → *.thrusty.json  +  provenance  +  ingest report  →  catalog.json
```

## Documents
| File | What it covers |
|------|----------------|
| `missile.schema.json` | Canonical model format (JSON Schema 2020-12). Required vs optional fields, `stages` array, payload (`terminal_mode`: reentry/orbital/suborbital) / rv / booster / shroud blocks, `source`, `provenance`, `completeness`. |
| `rv.schema.json` | Independently-loadable reentry-vehicle / glide-body format (`rv_library/*.rv.json`). `rv_kind` discriminator (ballistic / marv_body / glider / decoy) drives required-vs-inherited fields. Satellites are NOT here — they're an orbital payload on the missile. |
| `estimators.md` | The resolution ladder data: derivations, Isp-by-propellant, class defaults, aero/β fallbacks, RV pass (§6b), orbital-payload handling (§6c), sanity checks, confidence scoring, worked example. This is how "fix during ingest" actually works. |
| `spreadsheet-and-sheets.md` | Spreadsheet redesign (keep the detailed workbook, add a long-format catalog sheet) + three-level Google Sheets integration plan. |

## Key design decisions captured
- **Single canonical format** (`*.thrusty.json`); spreadsheets/Sheets are authoring
  surfaces, not storage.
- **Resolution ladder** `given → derived → estimated → default → unresolved`, with
  **per-field provenance** so nothing is filled silently and each model gets a
  **completeness score**.
- **7 hard-required fields** per stage (name, mass_initial, mass_propellant,
  burn_time_s, isp_s, diameter_m, length_m); everything else derivable or defaulted.
- **Catalog as a separate versioned library** (one file per model, `catalog.json`
  index, CI validator) so the community can contribute via PRs.

## Decisions (closed 2026-06-14)
All design decisions are settled; the docs above reflect them.
1. **Missing data** → estimate-with-VERIFY-flag: models import immediately with
   defaults filled and flagged, never blocked on a missing required field.
2. **missile_class** → `srbm / mrbm / irbm / icbm / slv / sounding / other`. It
   classifies the boost vehicle only and selects estimator fallbacks. A glide
   body is an RV (`rv_kind=glider`) on a booster, not a class; cruise is out of
   scope.
3. **rv_kind** → `ballistic / marv_body / glider / decoy`.
4. **Catalog location** → seed in-repo (`models/`) now; split to a standalone
   `thrusty-models` repo once the shape is proven.
5. **Google Sheets** → build Level 2 first (published-CSV "Refresh from Sheet").
6. **Orbital lint** → single global constant `STABLE_ORBIT_PERIGEE_KM` (200 km),
   QA-only, no per-model override.
7. **Isp/mass-fraction table** → `missile_xlsx.py` Reference sheet is canonical;
   `estimators.md` copies from it.
8. **Schema version** → `1.0`.

## When we move to "doing"
1. Lock the schema (this file set) — done.
2. Build the resolver + reporter module (pure functions; golden-test against the
   existing built-ins: Scud, AUR, Minotaur).
3. Wire the front-ends: extend `missile_xlsx.py`, add `import_catalog_csv()`
   (Level 2 published-CSV pull).
4. Seed the first batch in-repo `models/` (Forden's validated set is the natural
   start), then split to `thrusty-models`.
