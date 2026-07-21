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
NOTHING_SURVIVES_K = 4000.0       # T_eq above all usable materials → the
                                  # no-ablation equilibrium model is invalid here
                                  # (ablators survive by recession/blowing, not by
                                  # staying below a fixed surface limit)

# TPS material ladder — HEATING_TPS_REFERENCES.md §2 + HEATING_MODEL_CROSSCHECK.md §10.4.
# CONSUMED by the current FOM (do not change for existing keys — verdict stability):
#   peak_K       : short-duration surface limit (melt / ablation onset)
#   continuous_K : sustained / oxidation (soak) limit, also the re-radiation cap
#   melt_K       : bulk melt/sublimation for the lumped heat sink (None → continuous_K)
#   c_J_kgK      : specific heat for the lumped heat sink
#   label        : display name
# Phase-1 metadata (NOT yet consumed — used by the per-location evaluator, Phase 2+):
#   group        : 'metal' | 'hot_structure' | 'insulative' | 'ablative'.  Architecture flag:
#                  'metal'/'hot_structure' = the material IS the load path → bondline collapses
#                  onto its own limit (§10.1, non-separating hot-structure case); 'insulative'/
#                  'ablative' = a layer over a separate structure → bondline applies.  Also the
#                  GUI-flyout dropdown grouping.
#   is_ablator   : recedes (δ = Q/(ρ·H_eff)); else reradiative/heat-sink limited
#   density_kg_m3: for the lumped heat-sink mass and ablator recession
#   H_eff_MJ_kg  : effective heat of ablation Q* (ablators only; None otherwise).
#                  Q* is enthalpy-dependent, NOT a fixed constant; these nominals sit at the
#                  conservative (low) end of the flight/handbook band (CP ~10-30, PICA higher,
#                  C/C sublimation regime) so the recession screen OVER-predicts.  Bands +
#                  provenance: BENCHMARKING.md "Form A anchors"; benchmarks/form_a/phase2-heff-bands.md.
#                  Direction is bound-checked in test_form_a_bounds.py (predicted delta >= measured
#                  for the recovered Stardust/Hayabusa capsules).  NOT tuned to those capsules.
#   oxidation_dwell_s : representative oxidation-limited dwell life at its severe-use temperature
#                  (UHTC ~60-140 s at 2700 °C, §10.4/Tului); None if not dwell-limited
# NEW entries (carbon_carbon, carbon_phenolic, c_sic, cc_hot_structure, silica_phenolic, sirca,
# pica) carry SCREENING-tier peak/continuous estimates pending a verification pass (§10.4).
# The `uhtc` entry is RETUNED per SURVIVABILITY_REPORT_DESIGN.md §11.4 (grounded glass ceiling +
# demonstrated dwell floor) — see the inline comment on the entry; the envelope-coverage verdict
# that consumes it lives in survivability_report.py (green/amber/red, §11.3).
TPS_MATERIALS = {
    # --- structural metals (heat-sink / bare hot structure) ---
    "aluminum":        dict(peak_K=775,  continuous_K=450,  melt_K=775,  c_J_kgK=900,  label="Aluminum",
                            group="metal", is_ablator=False, density_kg_m3=2700, H_eff_MJ_kg=None, oxidation_dwell_s=None),
    "titanium":        dict(peak_K=1900, continuous_K=870,  melt_K=1900, c_J_kgK=520,  label="Titanium",
                            group="metal", is_ablator=False, density_kg_m3=4500, H_eff_MJ_kg=None, oxidation_dwell_s=None),
    "steel":           dict(peak_K=1700, continuous_K=1100, melt_K=1700, c_J_kgK=500,  label="Steel",
                            group="metal", is_ablator=False, density_kg_m3=7800, H_eff_MJ_kg=None, oxidation_dwell_s=None),
    # --- non-ablating hot structures (the material IS the structure) ---
    "rcc":             dict(peak_K=1922, continuous_K=1811, melt_K=None, c_J_kgK=1200, label="Coated carbon-carbon (RCC)",
                            group="hot_structure", is_ablator=False, density_kg_m3=1600, H_eff_MJ_kg=None, oxidation_dwell_s=None),
    "c_sic":           dict(peak_K=1970, continuous_K=1970, melt_K=None, c_J_kgK=1200, label="C/SiC (coated CMC)",
                            group="hot_structure", is_ablator=False, density_kg_m3=2000, H_eff_MJ_kg=None, oxidation_dwell_s=None),
    "cc_hot_structure":dict(peak_K=2170, continuous_K=2170, melt_K=None, c_J_kgK=1200, label="C/C hot structure (HTV-2)",
                            group="hot_structure", is_ablator=False, density_kg_m3=1800, H_eff_MJ_kg=None, oxidation_dwell_s=None),
    # uhtc: retuned per SURVIVABILITY_REPORT_DESIGN.md §11.4 (was continuous_K 1900, dwell 120 s
    # hard line).  continuous_K = 1923 K (1650 °C) — the borosilicate-glass PROTECTIVENESS
    # ceiling, ≥5 sources (Monteverde 2012, Peters 2024, Fahrenholtz & Hilmas, Marschall, Li).
    # oxidation_dwell_s = 300 s — the DEMONSTRATED FLOOR above the ceiling, conservatively the
    # low anchor (Monteverde 2013: 300 s at 1973 K, zero recession; sharp-tip survival extends
    # to ~575 s at 2450 °C, Monteverde 2012).  A floor, NOT a cliff: past it the report flags
    # extrapolation, it does not assert failure (§11.1).  peak_K 2700 ≈ the demonstrated sharp
    # ZrB2-SiC tip peak (2450 °C, CFD-sourced).  Anchor dataset: survivability_report.UHTC_ANCHORS.
    "uhtc":            dict(peak_K=2700, continuous_K=1923, melt_K=3500, c_J_kgK=600,  label="UHTC (ZrB2/HfB2-SiC)",
                            group="hot_structure", is_ablator=False, density_kg_m3=6000, H_eff_MJ_kg=None, oxidation_dwell_s=300),
    # --- reusable insulator (a layer over a separate structure) ---
    "silica_tile":     dict(peak_K=1811, continuous_K=1533, melt_K=None, c_J_kgK=1000, label="Silica tile (LI-900)",
                            group="insulative", is_ablator=False, density_kg_m3=144, H_eff_MJ_kg=None, oxidation_dwell_s=None),
    # --- ablators (sacrificial layer; recede) ---
    "carbon_ablator":  dict(peak_K=3900, continuous_K=2000, melt_K=3900, c_J_kgK=1500, label="Ablative carbon-carbon",
                            group="ablative", is_ablator=True, density_kg_m3=1450, H_eff_MJ_kg=15, oxidation_dwell_s=None),
    "carbon_carbon":   dict(peak_K=3900, continuous_K=2000, melt_K=3900, c_J_kgK=1500, label="Bare carbon-carbon (nose)",
                            group="ablative", is_ablator=True, density_kg_m3=1800, H_eff_MJ_kg=40, oxidation_dwell_s=None),
    "carbon_phenolic": dict(peak_K=3900, continuous_K=2000, melt_K=3900, c_J_kgK=1500, label="Carbon phenolic",
                            group="ablative", is_ablator=True, density_kg_m3=1450, H_eff_MJ_kg=15, oxidation_dwell_s=None),
    "silica_phenolic": dict(peak_K=1700, continuous_K=1700, melt_K=1700, c_J_kgK=1000, label="Silica phenolic",
                            group="ablative", is_ablator=True, density_kg_m3=1700, H_eff_MJ_kg=10, oxidation_dwell_s=None),
    "sirca":           dict(peak_K=1700, continuous_K=1700, melt_K=1700, c_J_kgK=1000, label="SIRCA (low-density ablator)",
                            group="ablative", is_ablator=True, density_kg_m3=270,  H_eff_MJ_kg=15, oxidation_dwell_s=None),
    "pica":            dict(peak_K=3600, continuous_K=2000, melt_K=3600, c_J_kgK=1500, label="PICA (low-density ablator)",
                            group="ablative", is_ablator=True, density_kg_m3=270,  H_eff_MJ_kg=35, oxidation_dwell_s=None),
}

# Dropdown groups for the GUI flyout (§10.1/§10.4) — order = display order.
TPS_MATERIAL_GROUPS = ("metal", "hot_structure", "insulative", "ablative")


def materials_by_group():
    """Return {group: [(key, label), ...]} for building the per-location material dropdown.

    Both the nose and body selectors draw from the full catalog (a non-separating warhead
    needs metals/hot-structure in the nose slot too, §10.1); the GUI orders by group.
    """
    out = {g: [] for g in TPS_MATERIAL_GROUPS}
    for key, m in TPS_MATERIALS.items():
        out.setdefault(m.get("group", "ablative"), []).append((key, m["label"]))
    return out


def is_hot_structure(material_key):
    """True if the material is its own load path (metal / hot structure) → bondline collapses
    onto its own limit (§10.1).  Layers (insulative / ablative) sit over a separate structure."""
    m = TPS_MATERIALS.get(material_key or "")
    return bool(m) and m.get("group") in ("metal", "hot_structure")


# Sentinel material keys for user-defined (bespoke) materials, one per location.
CUSTOM_NOSE_KEY = "custom_nose"
CUSTOM_BODY_KEY = "custom_body"


def register_custom_material(key, props):
    """Inject a user-defined material into TPS_MATERIALS so the key-based FOM can consume it.

    `props` is a partial catalog entry (from ROParams.nose_tps_custom / body_tps_custom);
    missing fields are filled with safe defaults so a bespoke material is always well-formed.
    Called for any RV carrying custom props before the heating FOM runs.  A falsy `props`
    removes any stale registration for `key`.  Returns the resolved key (or "" if cleared).
    """
    if not key or not props:
        TPS_MATERIALS.pop(key, None)
        return ""
    is_abl = bool(props.get("is_ablator", False))
    limit = props.get("continuous_K") or props.get("peak_K") or 0.0
    try:
        limit = float(limit)
    except (TypeError, ValueError):
        limit = 0.0
    def _num(v, default):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default
    entry = dict(
        label=str(props.get("label") or "Custom material"),
        group="ablative" if is_abl else "hot_structure",
        is_ablator=is_abl,
        peak_K=_num(props.get("peak_K"), limit),
        continuous_K=_num(props.get("continuous_K"), limit),
        melt_K=_num(props.get("melt_K"), limit) if props.get("melt_K") else None,
        c_J_kgK=_num(props.get("c_J_kgK"), 1200.0),
        density_kg_m3=_num(props.get("density_kg_m3"), 1800.0),
        H_eff_MJ_kg=(_num(props.get("H_eff_MJ_kg"), 15.0) if is_abl else None),
        oxidation_dwell_s=None,
    )
    TPS_MATERIALS[key] = entry
    return key

# Reentry heating benchmarks — HEATING_TPS_REFERENCES.md §3.  Per entry:
#   q_MW : peak stagnation flux (MW/m²);  Q_MJ : integrated load (MJ/m², per
#   unit area, = ∫q̇ dt);  conf : 'solid' (CFD reconstruction) or 'rough'.
# Apollo (793 W/cm², 46,792 J/cm²), Stardust (942 W/cm², 27.6 kJ/cm²) and MSL
# (197 W/cm² design, 5,477 J/cm²) are CFD-solid.  Shuttle is now pinned to the
# STS-1 flight reconstruction (NASA LaRC benchmark, NTRS 19820036242 /
# 19820015618): windward tiles ~5 Btu/ft²·s (≈0.06 MW/m²), RCC nose-cap /
# wing-leading-edge stagnation peak ~50 Btu/ft²·s (≈0.6 MW/m², surface ~1650 °C);
# we anchor q_MW on the RCC stagnation peak (the hot spot).  Q_MJ≈66 MJ/m² is
# the integrated load at the windward centerline x/L=0.4 — the acreage/tile
# location where the soak and TPS mass live, the right place for the load metric
# — obtained by integrating the STS-1 flight-data heat-flux history (radiation-
# equilibrium reduction, Ried et al. NTRS 19820015618 Fig. 11: peak ~6 W/cm²
# over a ~1500 s pulse → ∫q̇dt ≈ 6.6 kJ/cm²; ±~20% from digitizing the plot).
# So the Shuttle's two metrics intentionally reference their most-relevant
# locations (peak→RCC nose, load→windward acreage).  ICBM-RV is anchored on
# Reentry F (5° half-angle cone, R_n=2.54 mm initial, Mach ~20, V≈6.1 km/s).
# Its stagnation heating 9,000–28,000 Btu/ft²·s (≈102–318 MW/m²) is the NASA
# PREFLIGHT-predicted nominal (LWP-460, via Berry Fig. 6) — the ablating
# graphite tip was not calorimetered, so the flight-MEASURED data is the
# cone-flank turbulent heating (15–50 Btu/ft²·s, TM X-2253) and the nose
# recession.  The flight flew near-nominal and the prediction methods were
# validated against the flank flight data (Thompson et al. 1989), so we keep
# the 318 MW/m² peak but it is a flight-validated PREDICTION, not a measurement;
# blunter 1–5 cm operational tips scale to ~70–160 MW/m² (1/√R_N).  conf='solid'.
# Q_MJ omitted (steep, ~few-second ablative pulse; no clean ∫q̇dt value).
_BENCHMARKS = {
    "ICBM RV":  dict(q_MW=318.0, Q_MJ=None, conf="solid"),
    "Stardust": dict(q_MW=9.4,  Q_MJ=276.0, conf="solid"),
    "Apollo":   dict(q_MW=7.9,  Q_MJ=468.0, conf="solid"),
    "MSL":      dict(q_MW=2.0,  Q_MJ=55.0,  conf="solid"),
    "Shuttle":  dict(q_MW=0.6,  Q_MJ=66.0,  conf="solid"),
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
                            mass_kg=0.0, frontal_area_m2=0.0, soak_dwell_s=120.0,
                            recession_depth_m=0.0):
    """Evaluate the heating-survivability SCREENING figure of merit over a
    reentry arc.  This is a stagnation-point convective indicator, not a
    through-wall TPS response model; verdicts are screening flags (see the
    "warnings" key), not survival guarantees.

    t, rho, V, alt, rng : 1-D arrays over the reentry/glide phase (SI).
    Returns a dict: q_peak (NOSE-STAGNATION reference flux), peak T_eq,
    integrated load, per-criterion margins, compromise point, verdict,
    benchmark ratios, and screening-validity 'warnings'.
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

    _duration_s = float(t[-1] - t[0]) if np.asarray(t).size > 1 else 0.0
    out = {
        "q_peak_MW_m2": q_peak / 1e6,
        "T_eq_peak_K": T_peak,
        "integrated_load_MJ_m2": Q_area / 1e6,
        "duration_s": _duration_s,       # reentry/glide-arc span (feeds the TPS ladder)
        "benchmark": _nearest_benchmark(q_peak / 1e6, "q_MW"),
        "benchmark_load": _nearest_benchmark(Q_area / 1e6, "Q_MJ"),
        "material": material,
        "criteria": {},
        "compromise": None,
        "verdict": "",
    }

    # Validity guards — this is a screening indicator, NOT a through-wall TPS
    # response model; the labels below keep callers from over-reading it.
    out["warnings"] = [
        "Screening model: nose-stagnation Sutton-Graves convective flux + "
        "radiative-equilibrium wall temperature.  No ablation/recession/pyrolysis, "
        "backface conduction, or off-stagnation (shoulder/leading-edge/acreage) "
        "heating.  Verdicts are screening flags, not TPS-survival guarantees.",
        "q_peak is a NOSE-STAGNATION reference flux, not the heat over the whole "
        "vehicle.",
        "Benchmark ratios are single-scalar (flux or load) matches; entry regime, "
        "heating location (stagnation/acreage/leading-edge), beta, lift and TPS "
        "type are not matched.",
    ]
    _Vmax = float(np.max(V)) if V.size else 0.0
    if _Vmax > 9000.0:
        out["warnings"].append(
            f"Convective-only model; radiative gas heating NOT assessed "
            f"(peak V {_Vmax/1000:.1f} km/s exceeds the ~9 km/s screening envelope).")

    mat = TPS_MATERIALS.get(material)
    is_ablator = bool(mat and mat.get("is_ablator"))

    # NOTHING_SURVIVES guard applies to RERADIATIVE surfaces (and the no-material
    # case): a non-ablating wall above ~4000 K is beyond the no-ablation
    # equilibrium model.  An ABLATOR is *supposed* to sit above it — it survives
    # by receding at its ablation temperature, not by staying below a surface
    # limit — so ablators skip this guard and are judged by RECESSION below.
    if not is_ablator and T_peak >= NOTHING_SURVIVES_K:
        out["verdict"] = (f"outside no-ablation model validity (peak T_eq ≈ {T_peak:.0f} K "
                          f"≥ {NOTHING_SURVIVES_K:.0f} K) — requires ablation/material-response analysis")
        return out

    if not mat:
        out["verdict"] = ("physical numbers only — set the RV's tps_material "
                          "for a screening verdict")
        return out

    crossings = []   # (index, mode label)
    dt = np.diff(t, prepend=t[0])

    if is_ablator:
        # === ABLATOR: recession criterion δ = ∫q̇dt / (ρ·H_eff) ===
        # An ablator does NOT fail by exceeding a surface temperature — it
        # ablates AT its surface temperature and carries the heat away.  The
        # screening survival question is whether cumulative recession δ exceeds
        # the available material depth (burn-through).  The effective heat of
        # ablation H_eff already bundles reradiation and blowing; recession
        # accrues only while the surface is ablating (T_eq ≥ the material's
        # onset limit).  Grounded in Duffa §4.3/§4.7 (δ = Q/(ρ·H_eff)); the
        # peak-surface T_eq is reported but is INFORMATIONAL, not a failure.
        H_eff = float(mat.get("H_eff_MJ_kg") or 0.0) * 1e6     # J/kg
        rho_abl = float(mat.get("density_kg_m3") or 0.0)
        T_onset = float(mat["continuous_K"])                   # ablation/oxidation onset
        depth = (float(recession_depth_m) if recession_depth_m and recession_depth_m > 0
                 else float(nose_radius_m))
        out["criteria"]["peak_surface"] = {
            "margin": T_peak / mat["peak_K"], "limit_K": mat["peak_K"],
            "T_eq_peak_K": T_peak,
            "note": "informational for an ablator — it ablates at its surface "
                    "temperature; exceeding T_eq is expected, not a failure"}
        if H_eff > 0 and rho_abl > 0 and depth > 0:
            q_recess = np.where(T_eq >= T_onset, q_surf, 0.0)  # incident load while ablating
            delta_cum = np.cumsum(q_recess * dt) / (rho_abl * H_eff)   # cumulative recession (m)
            delta_final = float(delta_cum[-1]) if delta_cum.size else 0.0
            out["criteria"]["recession"] = {
                "margin": delta_final / depth,
                "recession_m": delta_final,
                "recession_over_Rn": delta_final / max(float(nose_radius_m), 1e-6),
                "depth_m": depth,
                "H_eff_MJ_kg": mat.get("H_eff_MJ_kg"), "rho_kg_m3": rho_abl,
                "basis": "effective-heat-of-ablation screen δ = ∫q̇dt/(ρ·H_eff) "
                         "over the ablating portion (T_eq ≥ onset); burn-through "
                         "when δ ≥ available depth (default = nose radius)"}
            bt = np.where(delta_cum >= depth)[0]
            if bt.size:
                crossings.append((int(bt[0]), "burn-through (recession exceeds depth)"))
        else:
            out["warnings"].append(
                "Ablator recession not evaluated (missing H_eff / density / depth) "
                "— physical numbers only for this location.")
    else:
        # === RERADIATIVE / non-ablating: the existing screen (unchanged) ===
        # 1. Peak surface temperature vs short-duration limit
        out["criteria"]["peak_surface"] = {
            "margin": T_peak / mat["peak_K"], "limit_K": mat["peak_K"],
            "T_eq_peak_K": T_peak}
        ex = np.where(T_eq > mat["peak_K"])[0]
        if ex.size:
            # T_eq is the zero-thermal-mass EQUILIBRIUM wall temperature, so this
            # timestamp is when the flux first EXCEEDS the surface limit — the
            # real (finite-thermal-mass) skin reaches failure seconds to tens of
            # seconds later (τ ≈ ρcδ·ΔT/q̇).  Label accordingly.
            crossings.append((int(ex[0]), "flux above surface melt/ablation limit"))

        # 2. Heat-soak: dwell above the continuous (oxidation) limit.
        #    A material-supplied oxidation_dwell_s is a DEMONSTRATED FLOOR
        #    (SURVIVABILITY_REPORT_DESIGN.md §11.4: within it → inside the
        #    demonstrated envelope; past it → extrapolation, NOT asserted
        #    failure — the report layer renders that distinction).  Materials
        #    without one keep the generic 120-s screening surrogate.
        _mat_dwell = mat.get("oxidation_dwell_s")
        _dwell_lim = float(_mat_dwell) if _mat_dwell else float(soak_dwell_s)
        _is_floor = bool(_mat_dwell)
        above = T_eq > mat["continuous_K"]
        cum_above = np.cumsum(np.where(above, dt, 0.0))
        time_above = float(cum_above[-1]) if cum_above.size else 0.0
        out["criteria"]["soak"] = {
            "margin": time_above / _dwell_lim, "time_above_s": time_above,
            "limit_K": mat["continuous_K"], "dwell_s": _dwell_lim,
            "floor": _is_floor,
            "basis": ("demonstrated oxidation-dwell floor (anchor dataset; "
                      "past it = extrapolation, not asserted failure)"
                      if _is_floor else
                      "empirical dwell-above-continuous-limit damage surrogate "
                      "(not an oxidation-kinetics closure)")}
        js = np.where(cum_above >= _dwell_lim)[0]
        if js.size:
            crossings.append((int(js[0]),
                              "past demonstrated oxidation-dwell floor "
                              "(extrapolation)" if _is_floor
                              else "TPS oxidation soak (glide)"))

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

    out["is_ablator"] = is_ablator
    if crossings:
        ci, mode = min(crossings, key=lambda c: c[0])
        out["compromise"] = {
            "t_s": float(t[ci]), "alt_km": float(alt[ci]) / 1000.0,
            "range_km": float(rng[ci]) / 1000.0, "V_kms": float(V[ci]) / 1000.0,
            "mode": mode}
        out["verdict"] = (f"COMPROMISED — {mode} at t={t[ci]:.0f}s, "
                          f"{alt[ci]/1000:.0f} km, {V[ci]/1000:.1f} km/s "
                          f"({mat['label']})")
    elif is_ablator and "recession" in out["criteria"]:
        # survives by ablation: report the recession it took to get there
        rc = out["criteria"]["recession"]
        out["verdict"] = (f"no screened burn-through ({mat['label']}); "
                          f"recedes {rc['recession_m']*100:.1f} cm "
                          f"({rc['recession_over_Rn']:.2f} R_n, "
                          f"{rc['margin']:.0%} of {rc['depth_m']*100:.1f} cm depth)")
    else:
        out["verdict"] = f"no screened thermal failure ({mat['label']})"
    return out


# ---------------------------------------------------------------------------
# Two-location screening verdict (HEATING_MODEL_CROSSCHECK.md §11, lean Phase 2)
# ---------------------------------------------------------------------------
# Acreage flux fraction: windward cone-flank/tail heating as a fraction of the
# nose stagnation flux.  Screening constant: Lu/Shi & Zhang 2024 give a
# cone-tail/stagnation ratio ≈ 0.13 (engineering distribution validated vs
# NASA TN D-5450 to <9%); the STS-1 windward-acreage-vs-RCC-stagnation ratio
# (~0.1) is consistent.  One number, flagged — not a heating distribution.
BODY_FLUX_FRACTION = 0.13

# --- Windward-flank heating band (Form C AoA probe) --------------------------
# The α=0 acreage flux above times a windward amplification A(α)=sin(δ+α)/sin(δ):
# the modified-Newtonian surface-pressure ratio Cp ∝ sin²θ fed through the
# reference-enthalpy laminar scaling q̇ ∝ √ρ_e ∝ √p_e (Eckert/Hung C_h ∝ Re*^−½,
# HEATING_MODEL_CROSSCHECK.md reference-enthalpy section).  CP_MAX cancels in the
# ratio, so A(α) is purely geometric.  The METHOD FAMILY (Van Driest St∝1/√Re
# laminar + Eckert-Tewfik reference enthalpy) and the windward-vs-leeward
# ORDERING it reproduces are CITED (AGARD-R-754 Kapp/Mathauer/Rieger 1988,
# validated on the Tracy M=7.95 cone at α=12°/24°); the closed-form sin-ratio
# reduction itself is an INFERENCE (BENCHMARKING.md "Windward/AoA heating
# probe").  ρ_e∝p_e holds edge temperature fixed, so behind the stronger
# windward shock the true ρ_e ratio is below the p_e ratio — the sin-ratio
# mildly OVER-predicts (conservative for a screen).
_WINDWARD_ALPHA_BAND  = (5.0, 20.0)   # deg; ends coincide with Thompson 1989's cited error anchors
_WINDWARD_DELTA_FLOOR = 5.0           # deg; numerical guard — A(α) diverges as δ→0 (RV forebodies are 5–15° cones)
# Gate: keep windward heating as a CONTEXT overlay by default (does not change
# any shipped verdict).  Flip to True to let it downgrade survive→degraded /
# flag needs-analysis.  Screening cannot place the transition or fin-gap
# reattachment line, so it never asserts a hard fail even when True.
WINDWARD_DRIVES_VERDICT = False


def _severity(res):
    """Rank a single-location FOM result for binding-location selection
    (higher = closer to / past failure; ≥1 means a criterion is exceeded).
    -inf = no material set (no verdict to bind on)."""
    if not res or not res.get("material"):
        return float("-inf")
    crit = res.get("criteria") or {}
    # Ablator: severity is the recession margin (δ/depth); T_eq and peak_surface
    # are informational, NOT failures — an ablator is meant to run hot and recede.
    if res.get("is_ablator") and "recession" in crit:
        return crit["recession"]["margin"]
    # Reradiative: outside-validity is worst, else the worst criterion margin.
    if res.get("T_eq_peak_K", 0.0) >= NOTHING_SURVIVES_K:
        return float("inf")
    worst = float("-inf")
    if "peak_surface" in crit:
        worst = max(worst, crit["peak_surface"]["margin"])
    if "soak" in crit:
        worst = max(worst, crit["soak"]["margin"])
    if "heat_sink" in crit:
        m = crit["heat_sink"]["margin"]
        worst = max(worst, (1.0 / m) if m > 0 else float("inf"))
    return worst


def heating_fom_per_location(t, rho, V, alt, rng, *, nose_radius_m=0.05,
                             body_radius_m=0.0, emissivity=0.85,
                             nose_material="", body_material="",
                             mass_kg=0.0, frontal_area_m2=0.0,
                             soak_dwell_s=120.0, nose_solid_depth_m=0.0,
                             body_thickness_m=0.0,
                             body_flux_fraction=None):
    """Two-location screening verdict: the SAME evaluator run at the nose
    (stagnation flux, nose material) and at the body acreage
    (body_flux_fraction × stagnation, body material); the headline verdict is
    the BINDING location — outside-validity first, else earliest compromise,
    else worst margin.

    The acreage call reuses heating_figure_of_merit unchanged by exploiting
    q̇ ∝ 1/√R: a flux fraction f is exactly an effective radius R_ref/f²,
    where R_ref is the BODY radius (flank heating carries no tip-radius term).
    The lumped heat-sink criterion (mass/frontal-area) runs on the body call
    only (it is a whole-body bulk criterion; running it at the nose too would
    double-count).

    Returns the binding location's dict (same top-level keys as
    heating_figure_of_merit, so existing consumers are unchanged) plus:
      binding_location : 'nose' | 'body'
      locations        : {'nose': <full result>, 'body': <full result>}
    and compromise (when present) gains a 'location' key.
    """
    # Recession depth per location (only used if the material is an ablator):
    #   nose  → solid-tip depth (default = nose radius, a conservative screen);
    #   body  → the designed TPS-layer thickness (MUST be passed explicitly —
    #           the body call's inflated effective radius is not a depth).
    nose = heating_figure_of_merit(
        t, rho, V, alt, rng, nose_radius_m=nose_radius_m, body_radius_m=0.0,
        emissivity=emissivity, material=nose_material, mass_kg=0.0,
        frontal_area_m2=0.0, soak_dwell_s=soak_dwell_s,
        recession_depth_m=nose_solid_depth_m)
    if body_flux_fraction is None:      # resolve the live module attr at call time
        body_flux_fraction = BODY_FLUX_FRACTION
    f = min(max(float(body_flux_fraction), 1e-3), 1.0)
    # Acreage reference scale: the flank/acreage boundary layer is set by the
    # BODY scale and contains no tip-radius term — referencing the fraction to
    # the sharp-tip stagnation flux would inflate body heating by
    # sqrt(R_body/R_n) (3.8x for SWERVE's 1.7 cm tip on a 24 cm body).  So the
    # fraction multiplies the body-scale stagnation flux (same reference the
    # heat_sink criterion already uses); sharp tip and blunt capsule then give
    # the same acreage estimate for the same body.
    _R_ref = body_radius_m if body_radius_m > 0.0 else nose_radius_m
    # Body ablator depth: the designed layer thickness, or a 2 cm screening
    # default when the RV does not specify one (flagged in warnings below).
    _body_depth = float(body_thickness_m) if body_thickness_m and body_thickness_m > 0 else 0.02
    body = heating_figure_of_merit(
        t, rho, V, alt, rng, nose_radius_m=_R_ref / f ** 2,
        body_radius_m=body_radius_m, emissivity=emissivity,
        material=body_material, mass_kg=mass_kg,
        frontal_area_m2=frontal_area_m2, soak_dwell_s=soak_dwell_s,
        recession_depth_m=_body_depth)

    # Binding location: validity breach > earlier compromise > worst margin.
    sn, sb = _severity(nose), _severity(body)
    if sn == float("inf") or sb == float("inf"):
        name = "nose" if sn >= sb else "body"
    else:
        nc, bc = nose.get("compromise"), body.get("compromise")
        if nc and bc:
            name = "nose" if nc["t_s"] <= bc["t_s"] else "body"
        elif nc or bc:
            name = "nose" if nc else "body"
        else:
            name = "nose" if sn >= sb else "body"
    binding, other = (nose, body) if name == "nose" else (body, nose)
    other_name = "body" if name == "nose" else "nose"

    out = dict(binding)
    out["binding_location"] = name
    out["locations"] = {"nose": nose, "body": body}
    out["verdict"] = f"{name.upper()} binds — {binding['verdict']}"
    if other.get("material"):
        out["verdict"] += f"  [{other_name}: {other['verdict']}]"
    if out.get("compromise"):
        out["compromise"] = dict(binding["compromise"], location=name)
    out["warnings"] = list(binding.get("warnings", [])) + [
        f"Body acreage flux modeled as a single screening fraction "
        f"({f:.2f} × body-scale stagnation flux; Lu/Shi & Zhang 2024 cone-tail "
        f"ratio, referenced to the body radius so tip sharpness does not "
        f"inflate acreage heating) — not a heating distribution; windward/"
        f"turbulent flank heating can run 3–5× higher than the laminar value "
        f"used here.",
    ]
    return out


def windward_amplification(delta_deg, alpha_deg):
    """Windward / leeward laminar-heating amplification over the α=0 flank.

    A(α) = sin(δ+α)/sin(δ)  (windward generator);  leeward uses sin(max(δ−α,0)).
    Modified-Newtonian pressure ratio through the √p_e laminar scaling; CP_MAX
    cancels.  δ is floored at _WINDWARD_DELTA_FLOOR (A diverges as δ→0)."""
    d = np.radians(max(float(delta_deg), _WINDWARD_DELTA_FLOOR))
    a = np.radians(max(float(alpha_deg), 0.0))
    sd = np.sin(d)
    win = np.sin(min(d + a, np.pi / 2.0)) / sd
    lee = np.sin(max(d - a, 0.0)) / sd
    return float(win), float(lee)


def windward_flank_flux(t, rho, V, alt, rng, *, body_radius_m,
                        flank_half_angle_deg, alpha_band_deg=None,
                        alpha_op_deg=None, emissivity=0.85, body_material="",
                        nose_radius_m=0.0, body_flux_fraction=None,
                        glide_mask=None, delta_defaulted=False):
    """Screening windward-flank convective heating for a lifting RV at AoA.

    Model (all closed-form, Sutton-Graves altitude):
        q̇_flank0 = body_flux_fraction · q̇_stag(ρ,V,R_body)   (α=0 acreage flux)
        q̇_windward(α) = q̇_flank0 · A(α),  A(α)=sin(δ+α)/sin(δ)
        T_eq,w = (q̇_windward / (σ·ε))^¼
    Evaluated over the glide sub-arc (pass glide_mask to exclude the low-AoA
    terminal dive).  Reported as a band across alpha_band_deg with the operating
    AoA marked when supplied.  Windward is off-nose acreage — the nose remains
    the hard-fail driver; this is a body-location severity overlay.

    Returns a flat dict: q_windward_MW_m2 / T_eq_windward_K each {lo,op,hi},
    delta_deg, alpha_band_deg, alpha_op_deg, amplification {lo,op,hi},
    criteria {windward_surface: {...}}, verdict, warnings.
    """
    rho = np.asarray(rho, float); V = np.asarray(V, float)
    alt = np.asarray(alt, float); t = np.asarray(t, float)
    eps = max(float(emissivity or 0.85), 1e-3)
    warnings = []
    # Resolve the live module attrs at call time so thresholds.apply() drives them.
    if alpha_band_deg is None:
        alpha_band_deg = _WINDWARD_ALPHA_BAND
    if body_flux_fraction is None:
        body_flux_fraction = BODY_FLUX_FRACTION

    delta = max(float(flank_half_angle_deg or 0.0), _WINDWARD_DELTA_FLOOR)
    if float(flank_half_angle_deg or 0.0) < _WINDWARD_DELTA_FLOOR:
        warnings.append(
            f"Forebody half-angle {float(flank_half_angle_deg or 0.0):.1f}° "
            f"floored to {_WINDWARD_DELTA_FLOOR:.0f}° (windward amplification "
            f"diverges as δ→0; screening guard, not a cited threshold).")
    if delta_defaulted:
        warnings.append(
            f"Forebody geometry unset — half-angle defaulted to {delta:.0f}° "
            f"(flagged inference).")

    R_body = float(body_radius_m) if body_radius_m and body_radius_m > 0 else float(nose_radius_m or 0.0)
    if R_body <= 0:
        return dict(verdict="windward not evaluated (no body radius)",
                    warnings=["no body/nose radius for the windward acreage scale"],
                    delta_deg=delta, criteria={})

    # α=0 flank flux over the (glide-masked) arc, then peak.
    q_flank0 = float(body_flux_fraction) * _stag_flux(rho, V, R_body)
    q_stag_nose = _stag_flux(rho, V, float(nose_radius_m)) if nose_radius_m and nose_radius_m > 0 else None
    mask = np.ones(t.shape, bool) if glide_mask is None else np.asarray(glide_mask, bool)
    if mask.shape != q_flank0.shape or not mask.any():
        mask = np.ones(q_flank0.shape, bool)
        if glide_mask is not None:
            warnings.append("glide-phase mask empty — windward evaluated over the full arc.")
    q_flank0_pk = float(np.max(q_flank0[mask])) if q_flank0[mask].size else 0.0

    a_lo, a_hi = float(alpha_band_deg[0]), float(alpha_band_deg[1])
    A_lo, _ = windward_amplification(delta, a_lo)
    A_hi, _ = windward_amplification(delta, a_hi)
    A_op = windward_amplification(delta, alpha_op_deg)[0] if alpha_op_deg is not None else None

    def _T(q):    return float((q / (eps * SIGMA)) ** 0.25) if q > 0 else 0.0
    q_lo, q_hi = q_flank0_pk * A_lo, q_flank0_pk * A_hi
    T_lo, T_hi = _T(q_lo), _T(q_hi)
    q_op = (q_flank0_pk * A_op) if A_op is not None else None
    T_op = _T(q_op) if q_op is not None else None

    # Stagnation-approach guard: windward should stay below the nose stagnation
    # flux for the screening regime.
    if q_stag_nose is not None and q_stag_nose[mask].size:
        q_stag_pk = float(np.max(q_stag_nose[mask]))
        if q_stag_pk > 0 and q_hi >= q_stag_pk:
            warnings.append(
                f"Windward estimate ({q_hi/1e6:.1f} MW/m²) approaches the nose "
                f"stagnation flux ({q_stag_pk/1e6:.1f} MW/m²) — sharp tip + small "
                f"δ + high α is outside the screening regime; treat as a flag.")

    # Body material limits → the windward criterion.
    mat = TPS_MATERIALS.get(str(body_material or "")) if body_material else None
    criteria = {}
    verdict = ""
    if mat and not mat.get("is_ablator"):
        cont, peak = float(mat["continuous_K"]), float(mat["peak_K"])
        criteria["windward_surface"] = {
            "limit_continuous_K": cont, "limit_peak_K": peak,
            "T_lo_K": T_lo, "T_hi_K": T_hi, "T_op_K": T_op,
            # margins: robust (even at the gentlest α=lo) and worst (α=hi)
            "margin_soak_lo": (T_lo / cont) if cont else 0.0,
            "margin_peak_hi": (T_hi / peak) if peak else 0.0}
        if T_lo > cont:
            verdict = (f"windward flank exceeds the body continuous limit "
                       f"({mat['label']} {cont:.0f} K) even at α={a_lo:.0f}° — "
                       f"body TPS runs beyond its soak limit on the windward side")
        elif T_hi > peak:
            verdict = (f"windward flank can exceed the body peak limit "
                       f"({mat['label']} {peak:.0f} K) at α≤{a_hi:.0f}° — "
                       f"needs a dedicated AoA/transition analysis")
        else:
            verdict = (f"windward flank within the body limits across α "
                       f"{a_lo:.0f}–{a_hi:.0f}° ({mat['label']})")
    elif mat and mat.get("is_ablator"):
        verdict = (f"body is an ablator ({mat['label']}) — windward reported as "
                   f"flux/T_eq only (recession is the failure mode, not T_eq)")
    else:
        verdict = "windward flux/T_eq only (no body TPS material set)"

    warnings.append(
        "Laminar windward estimate — turbulent flank runs ~3–5× higher "
        "(T_eq ~1.3–1.5×); flag, not a computed value.")
    warnings.append(
        "Control-fin / body-fin gap interference heating can reach 10–80× the "
        "fin-off flank value at reattachment (Alviani 2022, AEDC Mach-6); "
        "screening cannot place the reattachment line — flagged, not computed. "
        "Computed value needs a coupled solver (Murray & Russell 2002, MASCC).")
    if alpha_op_deg is None:
        warnings.append(
            "No trimmed operating AoA (separating RV or geometry unset) — "
            "windward reported as a band only.")

    return dict(
        q_windward_MW_m2={"lo": q_lo / 1e6, "op": (q_op / 1e6 if q_op is not None else None), "hi": q_hi / 1e6},
        T_eq_windward_K={"lo": T_lo, "op": T_op, "hi": T_hi},
        delta_deg=delta, alpha_band_deg=(a_lo, a_hi), alpha_op_deg=alpha_op_deg,
        amplification={"lo": A_lo, "op": A_op, "hi": A_hi},
        q_flank0_peak_MW_m2=q_flank0_pk / 1e6, body_material=str(body_material or ""),
        criteria=criteria, verdict=verdict,
        thompson_band="engineering-code AoA uncertainty ~15–40% (Thompson 1989)",
        warnings=warnings)


# ---------------------------------------------------------------------------
# Survivability summary — reduce a FOM to a screening pass/fail for display
# (feeds the GUI "Heating Survivability" panel; kept here so it is testable
# without the GUI).  Deliberately hedged wording: this is a rough screen.
# ---------------------------------------------------------------------------

def _loc_status(res):
    """Classify one location's FOM result.
    Returns (status, mark, T_eq_K, q_peak_MW, detail):
      status ∈ 'survive' | 'fail' | 'analysis' | 'none'."""
    if not res:
        return ("none", "–", 0.0, 0.0, "no data")
    T = float(res.get("T_eq_peak_K", 0.0) or 0.0)
    q = float(res.get("q_peak_MW_m2", 0.0) or 0.0)
    mat = res.get("material", "")
    if not mat:
        return ("none", "–", T, q, "no TPS material set")
    cmp = res.get("compromise")
    if res.get("is_ablator"):
        # Ablator: judged by recession, not T_eq — high T_eq is expected, so no
        # "beyond the model" branch here.  Fail = burn-through; else survives
        # with a stated recession.
        if cmp:
            return ("fail", "✗", T, q, f"{cmp['mode']} at t={cmp['t_s']:.0f} s")
        rc = (res.get("criteria") or {}).get("recession")
        if rc:
            return ("survive", "✓", T, q,
                    f"ablates: recedes {rc['recession_m']*100:.1f} cm "
                    f"({rc['recession_over_Rn']:.2f} R_n, {rc['margin']:.0%} of depth)")
        return ("survive", "✓", T, q, "no screened burn-through")
    # Reradiative (non-ablating): T_eq beyond the no-ablation model is unassessable.
    if T >= NOTHING_SURVIVES_K:
        return ("analysis", "⚠", T, q, "peak T_eq beyond the no-ablation model — needs ablation analysis")
    if cmp:
        return ("fail", "✗", T, q, f"{cmp['mode']} at t={cmp['t_s']:.0f} s")
    return ("survive", "✓", T, q, "no screened thermal failure")


# Headline wording per overall status (deliberately hedged — screening tier).
_SURV_HEADLINE = {
    "survive":  "LIKELY SURVIVES",
    "fail":     "LIKELY FAILS",
    "analysis": "MARGINAL — REQUIRES DEDICATED ANALYSIS",
    "none":     "NO VERDICT — set a TPS material",
}


def survivability_summary(fom):
    """Reduce a heating FOM (single-location or per-location) to a screening
    survivability assessment for display.  Returns a dict:
      status   : 'survive' | 'fail' | 'analysis' | 'none'
      headline : hedged one-line verdict
      lines    : per-location [{loc,label,T,q,mark,detail,binds,status}]
      nose_q_MW, load_MJ : headline reentry numbers (nose stagnation)
      notes    : screening caveats (warnings[])
    """
    if not fom:
        return dict(status="none", headline="No reentry heating was computed for this flight",
                    lines=[], nose_q_MW=None, load_MJ=None, notes=[])

    def _label(res):
        key = (res or {}).get("material", "") or ""
        return (TPS_MATERIALS.get(key, {}) or {}).get("label", key or "—")

    locs = fom.get("locations")
    if locs:
        binding = fom.get("binding_location")
        lines = []
        for name in ("nose", "body"):
            res = locs.get(name)
            st, mark, T, q, detail = _loc_status(res)
            lines.append(dict(loc=name, label=_label(res), T=T, q=q, mark=mark,
                              detail=detail, binds=(name == binding), status=st))
        overall = _loc_status(locs.get(binding))[0]
        nose = locs.get("nose") or {}
    else:
        st, mark, T, q, detail = _loc_status(fom)
        lines = [dict(loc="nose", label=_label(fom), T=T, q=q, mark=mark,
                      detail=detail, binds=True, status=st)]
        overall = st
        nose = fom

    return dict(
        status=overall,
        headline=_SURV_HEADLINE[overall],
        lines=lines,
        nose_q_MW=nose.get("q_peak_MW_m2"),
        load_MJ=nose.get("integrated_load_MJ_m2"),
        duration_s=fom.get("duration_s") or nose.get("duration_s"),
        notes=list(fom.get("warnings", [])),
    )
