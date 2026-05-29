# Spreadsheet redesign + Google Sheets workflow

Two goals here:

1. **Revisit the spreadsheet format** so it scales from "carefully hand-tune one
   missile" to "bulk-author a catalog of hundreds."
2. **Add Google Sheets functionality** so you (and contributors) can author models
   in a browser, collaboratively, and pull them into Thrusty.

Nothing below is wired into the app yet — this is the design.

---

## 1. Why the current format doesn't scale

`missile_xlsx.py` today is **one workbook = one missile**, laid out
*fields-as-rows, stages-as-columns* (D–G). It's excellent for tuning a single
complex vehicle — rich notes, live computed cross-checks, dropdowns — but:

- **One file per model** means a 200-model catalog is 200 workbooks.
- The transposed layout (parameters down, stages across) can't be filter/sorted
  or diffed like a normal table.
- It carries no `missile_class`, `propellant_type`, `source/citation`, or
  provenance — the fields the ingest pipeline and catalog need.
- It targets the legacy nested structure, not the new `stages` array + schema.

## 2. Proposal: keep the detailed sheet, add a "catalog" long-format

Support **two complementary layouts**, both feeding the same resolver (§ docs/ingestion/estimators.md) → same `*.thrusty.json`.

### A. Detailed single-model workbook (evolve the existing one)
Keep what's there; just extend it:
- Add an **IDENTITY** block: `missile_class`, `country`, `propellant_type` per stage.
- Add a **SOURCE** block: citation, url, author, license.
- Switch the import target to the new schema/array + emit an **ingest report** sheet
  (the resolver's findings: GIVEN/DERIVED/ESTIMATED/WARN per field).
- Best for: HGBs, MIRV buses, anything with boosters/shroud/glide detail.

### B. Catalog long-format sheet (new — the bulk workhorse)
**One row per stage**, models stacked, designed for hundreds of rows and for
Google Sheets. This is what makes a community catalog practical.

| model_id | name | class | country | stage_no | prop_type | mass_initial | mass_prop | mass_final | diameter_m | length_m | burn_s | isp_s | thrust_N | solid | grain | guidance | source_citation | source_url | license |
|----------|------|-------|---------|---------:|-----------|-------------:|----------:|-----------:|-----------:|---------:|-------:|------:|---------:|-------|-------|----------|-----------------|-----------|---------|
| df21 | DF-21 | mrbm | China | 1 | solid_composite | 14700 | 11000 |  | 1.4 | 10.7 | 60 |  |  | YES | tubular | pitch_program | Lewis 2024 | … | CC-BY |
| df21 | DF-21 | mrbm | China | 2 | solid_composite | 2500 | 1800 |  | 1.4 | 2.6 | 55 |  |  | YES | star | pitch_program | | | |
| mm3  | Minuteman III | icbm | USA | 1 | solid_composite | … | … | | … | … | … | | | YES | | true_gravity_turn | … | | |

Conventions:
- **`model_id`** groups a model's stages; rows sharing an id collapse into one
  `stages` array (sorted by `stage_no`). This is the same array→chain link the
  current importer already does internally.
- **Blank cells are intentional** — they're exactly what the resolver fills and
  flags. Authors leave Isp/thrust blank for the estimator rather than guessing.
- **Separate tabs** for the one-per-model blocks: `payload`, `rv`, `boosters`,
  `shroud`, `cd_overrides` — each keyed by `model_id`. Most models only need the
  `stages` tab + a `payload`/`rv` row.
- A `README`/`enums` tab documents the allowed values (mirrors the schema enums)
  and feeds Data Validation dropdowns so browsers enforce them.

### Why long-format wins for ingest
- Sorts, filters, and **diffs in git** as plain rows.
- One paste-friendly grid for batch entry; export the whole tab to CSV → batch
  ingest → N JSON files + one combined report.
- Trivially round-trips with Google Sheets (CSV is the lingua franca).

---

## 3. Google Sheets functionality — three integration levels

Pick based on how much "live" you want vs how much auth plumbing you'll tolerate.

### Level 1 — Template + manual download  *(zero code, do this first)*
- Publish the **catalog long-format** as a shared, view-only Google Sheet.
- Workflow: `File → Make a copy` → fill rows → `File → Download → .csv` (or
  `.xlsx`) → Thrusty's existing import button.
- Pros: no API, no credentials, works today once the CSV/xlsx importer speaks the
  new long-format. Cons: manual export step.

### Level 2 — Published-CSV pull  *(thin code, no auth)  ← recommended default*
- Author publishes their sheet via `File → Share → Publish to web → CSV`.
- Thrusty (or the ingest CLI) takes the **published CSV URL** and fetches it
  directly: `import_catalog_csv(url_or_path)`.
- Pros: one-click "Refresh from Sheet" in Thrusty; no OAuth; read-only and safe;
  the same code path handles a local `.csv` file. Cons: sheet must be
  publish-enabled; read-only (no write-back).
- This is the sweet spot for *"give myself Google Sheets functionality"*: you keep
  a master Sheet, hit refresh in Thrusty, models flow in.

### Level 3 — Full Sheets API  *(OAuth / service account)*
- `gspread` / Google Sheets API for authenticated read **and write-back**
  (e.g. Thrusty writes resolved Isp/thrust and completeness scores back into the
  sheet so the grid shows what was filled).
- Pros: live two-way sync, private sheets, multi-user catalog curation.
- Cons: credentials, token storage, network policy, a real dependency. Worth it
  only once the catalog is active and collaborative.

> **Network note:** Levels 2–3 require outbound network access, governed by this
> environment's network policy (see the web docs). Level 1 needs none.

### Suggested path
1. Define the long-format columns (table in §2.B) and the CSV importer →
   resolver. *(enables Level 1 immediately, and is the core of Level 2)*
2. Add `import_catalog_csv(url_or_path)` + a "Refresh from Google Sheet" action.
   *(Level 2)*
3. Only if/when collaborative curation demands it, add `gspread` write-back.
   *(Level 3)*

---

## 4. How the pieces connect

```
 Detailed workbook ─┐
                    ├─▶ parse ─▶ normalize units ─▶ RESOLVE ─▶ validate ─▶ *.thrusty.json  ┐
 Catalog sheet ─────┤        (one row/stage)        (estimators.md)         + provenance    ├─▶ catalog.json
 (xlsx / CSV / URL)─┘                                                        + ingest report ┘      (+ rv_library/)
```

One resolver, one schema, one report format — three ways in (detailed xlsx,
long-format CSV/xlsx, Google Sheet URL). That keeps the "two workflows" (Claude
from documents; users from spreadsheets) on a single validated pipeline.
