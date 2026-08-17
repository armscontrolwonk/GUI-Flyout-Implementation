# Bundled gazetteer packs — provenance

Baked by `gazetteer_build.py` (repo root) from official U.S. gazetteer
products.  The packs ARE the shipped data; the raw files stay outside
the repo.  Every record carries its source's own feature ID
(`GNIS:…` / `ATA:…` / `GNS:…`), so any coordinate here is traceable to
an official record.  Variants are kept and searchable by design (user
requirement 2026-08-17): romanized spellings differ across systems.

| Pack | Source | Retrieved | Features |
|---|---|---|---|
| `gnis_us.txt.gz` | USGS GNIS DomesticNames AllStates (public domain), prd-tnm.s3.amazonaws.com /StagedProducts/GeographicNames/DomesticNames/DomesticNames_AllStates_Text.zip | 2026-08-17 | 974,023 (all feature classes) |
| `antarctica.txt.gz` | BGN/ACAN Antarctic gazetteer GPKG (public domain), prd-tnm.s3.amazonaws.com /StagedProducts/GeographicNames/Antarctica/Gazetteer_Antarctica_GPKG.zip | 2026-08-17 | 14,353 (+ AllNames variants) |

## Pending (Phase 2): NGA GNS worldwide

The foreign half — GNS `Populated_Places.zip` (399 MB) and
`Spot_Features.zip` (75 MB), archived in the Drive "Thrusty NGA" folder
(retrieved from geonames.nga.mil 2026-08-17) — cannot reach a Claude
session directly (both NGA and Drive are egress-blocked).  Transfer
plan: from a local clone, `split -b 90m` the zips, push the parts to a
throwaway `gns-staging` branch, and a session bakes the packs onto main
(`gazetteer_build.py` already handles the GNS schema) and deletes the
branch — main's history never carries the raw blobs.

## Format

One line per feature, pipe-delimited, gzipped:
`ext_id|primary_name|variants(';'-joined)|lat|lon|admin|feature_class|source`

Runtime: `gazetteer.py` builds a local SQLite cache at
`~/.gui_missile_flyout/gazetteer.db` on first use (safe to delete —
it rebuilds from the packs).
