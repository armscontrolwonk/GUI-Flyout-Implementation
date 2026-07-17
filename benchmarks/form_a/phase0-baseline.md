# Form A ablator — Phase 0 baseline (pre-change)

## Recession chain (heating.py)
- `is_ablator` materials recede: `δ = Σ q̇·dt / (ρ·H_eff)` over the ablating
  portion (T_eq ≥ continuous_K onset); burn-through when δ ≥ available depth
  (default = nose radius).  Grounded in Duffa §4.3/§4.7.  heating.py ~L293-330.
- Peak-surface T_eq is INFORMATIONAL for ablators (they ablate at their surface
  temp), not a failure.
- Stagnation flux: Sutton-Graves `q̇ = 1.7415e-4·√(ρ/R_n)·V³`, convective-only
  (no radiative gas cap — the >9 km/s P3 item).

## H_eff placeholders (heating.py L85-95) — the targets
| material         | ρ (kg/m³) | H_eff (MJ/kg) |
|---|---|---|
| pica            | 270  | 35 |
| carbon_phenolic | 1450 | 15 |
| carbon_carbon   | 1800 | 40 |

## Already-anchored asset
- `_BENCHMARKS['Stardust']` = q̇ 9.4 MW/m², **Q 276 MJ/m²**, conf 'solid'.

## Baseline predictions (screening chain, full-Q)
- Stardust PICA: δ = 276 MJ/m² / (270 · 35 MJ/kg) = **29.2 mm** (full-load).
  Measured stagnation recession = **4.06 mm** → predicted/measured ≈ **7.2×**.
  Bounding direction (predicted ≥ measured) holds strongly.  The 7× (vs FIAT's
  ~1.5×) reflects the screening chain's crudeness (full-Q × single H_eff), which
  is conservative-safe for a screen.
