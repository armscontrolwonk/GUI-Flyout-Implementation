# Bundled gazetteer packs — provenance

Baked by `gazetteer_build.py` (repo root) from official U.S. gazetteer
products.  The packs ARE the shipped data; the raw files stay outside
the repo.  Every record carries its source's own feature ID
(`GNIS:…` / `ATA:…` / `GNS:…`), so any coordinate here is traceable to
an official record.  Variants are kept and searchable by design (user
requirement 2026-08-17): romanized spellings differ across systems.

All NGA GNS packs bake from the nine whole-world class files retrieved
from geonames.nga.mil 2026-08-17 (archived in the Drive "Thrusty NGA"
folder; transferred via the throwaway `gns-staging` branch and verified
byte-for-byte against the originals before baking).  Decision
2026-08-17: bake ALL classes — thinning is a judgment call the data
policy avoids; search ranking (gazetteer._tier) keeps cities and
facilities above hydrographic/terrain noise.

| Pack | Source | Retrieved | Features |
|---|---|---|---|
| `gnis_us.txt.gz` | USGS GNIS DomesticNames AllStates (public domain), prd-tnm.s3.amazonaws.com /StagedProducts/GeographicNames/DomesticNames/DomesticNames_AllStates_Text.zip | 2026-08-17 | 974,023 (all feature classes) |
| `antarctica.txt.gz` | BGN/ACAN Antarctic gazetteer GPKG (public domain), prd-tnm.s3.amazonaws.com /StagedProducts/GeographicNames/Antarctica/Gazetteer_Antarctica_GPKG.zip | 2026-08-17 | 14,353 (+ AllNames variants) |
| `gns_pp.txt.gz` | NGA GNS Populated_Places (public domain) | 2026-08-17 | 4,768,980 |
| `gns_hydro.txt.gz` | NGA GNS Hydrographic | 2026-08-17 | 1,741,086 |
| `gns_hypso.txt.gz` | NGA GNS Hypsographic | 2026-08-17 | 1,265,349 |
| `gns_spot.txt.gz` | NGA GNS Spot_Features (facilities, installations, airfields) | 2026-08-17 | 860,385 |
| `gns_areas.txt.gz` | NGA GNS Areas_Localities | 2026-08-17 | 377,271 |
| `gns_admin.txt.gz` | NGA GNS Administrative_Regions | 2026-08-17 | 276,692 |
| `gns_veg.txt.gz` | NGA GNS Vegetation | 2026-08-17 | 118,800 |
| `gns_transport.txt.gz` | NGA GNS Transportation_Networks | 2026-08-17 | 31,868 |
| `gns_undersea.txt.gz` | NGA GNS Undersea | 2026-08-17 | 6,398 |

Total: 10,435,205 features, ~196 MB of packs.  First index build on a
machine takes ~2½ minutes (SQLite cache, ~2.5 GB, in
`~/.gui_missile_flyout/`); afterwards search is instant.

## Format

One line per feature, pipe-delimited, gzipped:
`ext_id|primary_name|variants(';'-joined)|lat|lon|admin|feature_class|source`

Runtime: `gazetteer.py` builds a local SQLite cache at
`~/.gui_missile_flyout/gazetteer.db` on first use (safe to delete —
it rebuilds from the packs).
