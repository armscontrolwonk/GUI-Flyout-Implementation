"""
Reentry-heating survivability figure of merit.

Given a reentry/glide arc (time, density, airspeed), evaluate three
complementary failure criteria and return per-criterion margins, the earliest
"compromise point", and a verdict.  Grounded in HEATING_TPS_REFERENCES.md:

  1. Peak surface temperature — radiative-equilibrium T_eq = (q̇/εσ)^¼ vs the
     TPS material's SHORT-DURATION limit.  Surface melts/ablates at the peak
     (pull-out).  q̇ is Sutton-Graves stagnation heating at the NOSE radius
     (the engineering reduction of Fay-Riddell; radiative gas heating is
     negligible <~9 km/s per Tauber-Sutton and omitted).

  2. Heat soak (oxidation) — time the surface holds above the material's
     CONTINUOUS-use limit.  TPS oxidizes/degrades over a long glide
     (the HTV-2 aeroshell mode); the literature-standard reusable-TPS
     criterion is time-at-temperature.

  3. Lumped heat-sink burn-up — Reynerson, AIAA 2006-6275.  Integrated heat
     that the surface cannot re-radiate soaks into the body mass; the body
     melts when that accumulated heat exceeds m·c·(T_melt−T₀).  Driven by the
     BODY-radius stagnation flux (blunter → lower than the nose).  The
     re-radiation cap is taken at the continuous limit, so a surface that can
     reject its flux never accumulates (correct for a radiating glider), while
     a flux spike above the cap soaks in (the bare-body / short-pulse case).

All temperatures are kelvin.  This is a screening figure of merit, not a
through-wall conduction solution (that backface upgrade is future work).
"""
import numpy as np

SIGMA = 5.670374419e-8            # Stefan-Boltzmann, W/m²/K⁴
_SG_K = 1.7415e-4                 # Sutton-Graves constant (SI, W/m²), Earth air
_T0   = 298.0                     # reference temperature, K
NOTHING_SURVIVES_K = 4000.0       # above all usable materials

# TPS material ladder — HEATING_TPS_REFERENCES.md §2 (peak vs continuous limits).
#   peak_K       : short-duration surface limit (melt / ablation onset)
#   continuous_K : sustained / oxidation (soak) limit, also the re-radiation cap
#   melt_K       : bulk melt/sublimation for the lumped heat sink (None → continuous_K)
#   c_J_kgK      : specific heat for the lumped heat sink
TPS_MATERIALS = {
    "aluminum":       dict(peak_K=775,  continuous_K=450,  melt_K=775,  c_J_kgK=900,  label="Aluminum"),
    "titanium":       dict(peak_K=1900, continuous_K=870,  melt_K=1900, c_J_kgK=520,  label="Titanium"),
    "steel":          dict(peak_K=1700, continuous_K=1100, melt_K=1700, c_J_kgK=500,  label="Steel"),
    "silica_tile":    dict(peak_K=1811, continuous_K=1533, melt_K=None, c_J_kgK=1000, label="Silica tile (LI-900)"),
    "rcc":            dict(peak_K=1922, continuous_K=1811, melt_K=None, c_J_kgK=1200, label="Coated carbon-carbon (RCC)"),
    "uhtc":           dict(peak_K=2700, continuous_K=1900, melt_K=3500, c_J_kgK=600,  label="UHTC (ZrB2/HfB2-SiC)"),
    "carbon_ablator": dict(peak_K=3900, continuous_K=2000, melt_K=3900, c_J_kgK=1500, label="Ablative carbon-carbon"),
}

# Reentry heating benchmarks — HEATING_TPS_REFERENCES.md §3.  Per entry:
#   q_MW : peak stagnation flux (MW/m²);  Q_MJ : integrated load (MJ/m², per
#   unit area, = ∫q̇ dt);  conf : 'solid' (CFD reconstruction) or 'rough'.
# Apollo (793 W/cm², 46,792 J/cm²), Stardust (942 W/cm², 27.6 kJ/cm²) and MSL
# (197 W/cm² design, 5,477 J/cm²) are CFD-solid.  Shuttle is now pinned to the
# STS-1 flight reconstruction (NASA LaRC benchmark, NTRS 19820036242 /
# 19820015618): windward tiles ~5 Btu/ft²·s (≈0.06 MW/m²), RCC nose-cap /
# wing-leading-edge stagnation peak ~50 Btu/ft²·s (≈0.6 MW/m², surface ~1650 °C);
# we anchor on the RCC stagnation peak.  No clean public integrated-load value,
# so Q_MJ stays None.  ICBM-RV peak flux still lacks a clean public primary
# (Bunn 1984 not machine-readable) — order-of-magnitude only, flagged 'rough'.
_BENCHMARKS = {
    "ICBM RV":  dict(q_MW=30.0, Q_MJ=None,  conf="rough"),
    "Stardust": dict(q_MW=9.4,  Q_MJ=276.0, conf="solid"),
    "Apollo":   dict(q_MW=7.9,  Q_MJ=468.0, conf="solid"),
    "MSL":      dict(q_MW=2.0,  Q_MJ=55.0,  conf="solid"),
    "Shuttle":  dict(q_MW=0.6,  Q_MJ=None,  conf="solid"),
}


def _nearest_benchmark(value, key):
    """Nearest benchmark to *value* by log-distance on field *key* ('q_MW' or
    'Q_MJ'), as a ratio string; '(rough)' appended for flagged anchors."""
    cands = [(n, b[key], b["conf"]) for n, b in _BENCHMARKS.items() if b.get(key)]
    if value <= 0 or not cands:
        return "n/a"
    n, v, conf = min(cands, key=lambda c: abs(np.log(value / c[1])))
    return f"{value / v:.1f}× {n}" + (" (rough)" if conf == "rough" else "")


def _stag_flux(rho, V, radius_m):
    return _SG_K * np.sqrt(np.asarray(rho) / max(radius_m, 1e-4)) * np.asarray(V) ** 3


def heating_figure_of_merit(t, rho, V, alt, rng, *, nose_radius_m=0.05,
                            body_radius_m=0.0, emissivity=0.85, material="",
                            mass_kg=0.0, frontal_area_m2=0.0, soak_dwell_s=120.0):
    """Evaluate the heating survivability figure of merit over a reentry arc.

    t, rho, V, alt, rng : 1-D arrays over the reentry/glide phase (SI).
    Returns a dict (peak flux, peak T_eq, integrated load, per-criterion
    margins, compromise point, verdict, benchmark ratio).
    """
    t = np.asarray(t, float); rho = np.asarray(rho, float); V = np.asarray(V, float)
    alt = np.asarray(alt, float); rng = np.asarray(rng, float)
    eps = max(float(emissivity), 1e-3)

    q_surf = _stag_flux(rho, V, nose_radius_m)               # nose stagnation flux
    T_eq = (q_surf / (SIGMA * eps)) ** 0.25
    ipk = int(np.argmax(q_surf)) if q_surf.size else 0
    q_peak = float(q_surf[ipk]) if q_surf.size else 0.0
    T_peak = float(T_eq[ipk]) if q_surf.size else 0.0
    Q_area = (float(np.sum(0.5 * (q_surf[1:] + q_surf[:-1]) * np.diff(t)))
              if q_surf.size > 1 else 0.0)

    out = {
        "q_peak_MW_m2": q_peak / 1e6,
        "T_eq_peak_K": T_peak,
        "integrated_load_MJ_m2": Q_area / 1e6,
        "benchmark": _nearest_benchmark(q_peak / 1e6, "q_MW"),
        "benchmark_load": _nearest_benchmark(Q_area / 1e6, "Q_MJ"),
        "material": material,
        "criteria": {},
        "compromise": None,
        "verdict": "",
    }

    if T_peak >= NOTHING_SURVIVES_K:
        out["verdict"] = f"nothing survives (peak T_eq ≈ {T_peak:.0f} K ≥ {NOTHING_SURVIVES_K:.0f} K)"
        return out

    mat = TPS_MATERIALS.get(material)
    if not mat:
        out["verdict"] = ("physical numbers only — set the RV's tps_material "
                          "for a survivability verdict")
        return out

    crossings = []   # (index, mode label)
    dt = np.diff(t, prepend=t[0])

    # 1. Peak surface temperature vs short-duration limit
    out["criteria"]["peak_surface"] = {
        "margin": T_peak / mat["peak_K"], "limit_K": mat["peak_K"],
        "T_eq_peak_K": T_peak}
    ex = np.where(T_eq > mat["peak_K"])[0]
    if ex.size:
        crossings.append((int(ex[0]), "surface melt/ablation (pull-out)"))

    # 2. Heat-soak: dwell above the continuous (oxidation) limit
    above = T_eq > mat["continuous_K"]
    cum_above = np.cumsum(np.where(above, dt, 0.0))
    time_above = float(cum_above[-1]) if cum_above.size else 0.0
    out["criteria"]["soak"] = {
        "margin": time_above / soak_dwell_s, "time_above_s": time_above,
        "limit_K": mat["continuous_K"], "dwell_s": soak_dwell_s}
    js = np.where(cum_above >= soak_dwell_s)[0]
    if js.size:
        crossings.append((int(js[0]), "TPS oxidation soak (glide)"))

    # 3. Lumped heat-sink burn-up (Reynerson): flux above the re-radiation cap
    #    soaks into the body mass; melt when accumulated heat ≥ m·c·(T_melt−T₀).
    if mass_kg > 0 and frontal_area_m2 > 0 and body_radius_m > 0:
        q_body = _stag_flux(rho, V, body_radius_m)
        q_cap = eps * SIGMA * mat["continuous_K"] ** 4          # re-radiation cap
        q_excess = np.maximum(q_body - q_cap, 0.0)
        Q_in = np.cumsum(q_excess * dt) * frontal_area_m2       # J absorbed by mass
        Tm = mat["melt_K"] or mat["continuous_K"]
        Q_melt = mass_kg * mat["c_J_kgK"] * (Tm - _T0)
        out["criteria"]["heat_sink"] = {
            "margin": Q_melt / max(float(Q_in[-1]), 1e-9),
            "Q_absorbed_MJ": float(Q_in[-1]) / 1e6,
            "Q_melt_MJ": Q_melt / 1e6, "melt_K": Tm}
        jh = np.where(Q_in >= Q_melt)[0]
        if jh.size:
            crossings.append((int(jh[0]), "bare-body melt (heat sink)"))

    if crossings:
        ci, mode = min(crossings, key=lambda c: c[0])
        out["compromise"] = {
            "t_s": float(t[ci]), "alt_km": float(alt[ci]) / 1000.0,
            "range_km": float(rng[ci]) / 1000.0, "V_kms": float(V[ci]) / 1000.0,
            "mode": mode}
        out["verdict"] = (f"COMPROMISED — {mode} at t={t[ci]:.0f}s, "
                          f"{alt[ci]/1000:.0f} km, {V[ci]/1000:.1f} km/s "
                          f"({mat['label']})")
    else:
        out["verdict"] = f"survives ({mat['label']})"
    return out
