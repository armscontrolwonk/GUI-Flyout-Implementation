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
| `missile.schema.json` | Canonical model format (JSON Schema 2020-12). Required vs optional fields, `stages` array, payload/rv/booster/shroud blocks, `source`, `provenance`, `completeness`. |
| `estimators.md` | The resolution ladder data: derivations, Isp-by-propellant, class defaults, aero/β fallbacks, sanity checks, confidence scoring, worked example. This is how "fix during ingest" actually works. |
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

## Open questions for the next pass
1. Estimation aggressiveness — confirmed leaning: estimate-with-VERIFY-flag so
   models run immediately (vs. block on any missing required field).
2. Final class taxonomy (current draft: srbm/mrbm/irbm/icbm/slv/hgb/cruise/
   sounding/other).
3. Catalog repo location — folder in this repo vs a standalone `thrusty-models` repo.
4. Google Sheets level to build first (recommended: Level 2, published-CSV pull).

## When we move to "doing"
1. Lock the schema (this file set).
2. Build the resolver + reporter module (pure functions; golden-test against the
   existing built-ins: Scud, AUR, Minotaur).
3. Wire the front-ends: extend `missile_xlsx.py`, add `import_catalog_csv()`.
4. Stand up the catalog repo + CI validator.
5. Seed the first batch (Forden's validated set is the natural start).
