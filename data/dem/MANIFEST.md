# Thrusty coarse DEM — provenance

- **File**: terrain_0p05deg.npy — int16 metres, 3600×7200 (lat×lon), 0.05° equirectangular,
  row 0 = +90° lat, col 0 = −180° lon, cell centres.
- **Source**: AWS `elevation-tiles-prod` Terrarium tiles, zoom 5
  (`s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png`),
  a global blend of SRTM, GMTED2010, ETOPO1, and national DEMs.
- **Encoding decoded**: elev_m = R*256 + G + B/256 − 32768.
- **Resample**: z5 native Mercator (8192²) → inverse-Mercator to
  0.05° equirectangular; poleward of ±85.05° holds the edge value.
- **Reproducible**: `python3 dem_build.py`.
- **Baked**: elevation range -10644..6693 m.
