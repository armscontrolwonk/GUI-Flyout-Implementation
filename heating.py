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
#   H_eff_MJ_kg  : effective heat of ablation (ablators only; None otherwise)
#   oxidation_dwell_s : representative oxidation-limited dwell life at its severe-use temperature
#                  (UHTC ~60-140 s at 2700 °C, §10.4/Tului); None if not dwell-limited
# NEW entries (carbon_carbon, carbon_phenolic, c_sic, cc_hot_structure, silica_phenolic, sirca,
# pica) carry SCREENING-tier peak/continuous estimates pending a verification pass (§10.4); they
# are not referenced by any existing rv.json, so they do not affect current verdicts.  The `uhtc`
# temperature limits are the LEGACY screening values, kept for verdict stability; the grounded
# regrade (recede >~2000 °C, dwell-limited) lands when the oxidation-dwell criterion is built.
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
    "uhtc":            dict(peak_K=2700, continuous_K=1900, melt_K=3500, c_J_kgK=600,  label="UHTC (ZrB2/HfB2-SiC)",
                            group="hot_structure", is_ablator=False, density_kg_m3=6000, H_eff_MJ_kg=None, oxidation_dwell_s=120),
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
                            mass_kg=0.0, frontal_area_m2=0.0, soak_dwell_s=120.0):
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

    if T_peak >= NOTHING_SURVIVES_K:
        out["verdict"] = (f"outside no-ablation model validity (peak T_eq ≈ {T_peak:.0f} K "
                          f"≥ {NOTHING_SURVIVES_K:.0f} K) — requires ablation/material-response analysis")
        return out

    mat = TPS_MATERIALS.get(material)
    if not mat:
        out["verdict"] = ("physical numbers only — set the RV's tps_material "
                          "for a screening verdict")
        return out

    crossings = []   # (index, mode label)
    dt = np.diff(t, prepend=t[0])

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

    # 2. Heat-soak: dwell above the continuous (oxidation) limit
    above = T_eq > mat["continuous_K"]
    cum_above = np.cumsum(np.where(above, dt, 0.0))
    time_above = float(cum_above[-1]) if cum_above.size else 0.0
    out["criteria"]["soak"] = {
        "margin": time_above / soak_dwell_s, "time_above_s": time_above,
        "limit_K": mat["continuous_K"], "dwell_s": soak_dwell_s,
        "basis": "empirical dwell-above-continuous-limit damage surrogate "
                 "(not an oxidation-kinetics closure)"}
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


def _severity(res):
    """Rank a single-location FOM result for binding-location selection.
    inf  = outside no-ablation validity (worst — the model can't even clear it);
    -inf = no material set (no verdict to bind on);
    else the worst criterion margin (heat_sink margin is inverted: safe > 1)."""
    if res.get("T_eq_peak_K", 0.0) >= NOTHING_SURVIVES_K:
        return float("inf")
    if not res.get("material"):
        return float("-inf")
    worst = float("-inf")
    crit = res.get("criteria") or {}
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
                             soak_dwell_s=120.0,
                             body_flux_fraction=BODY_FLUX_FRACTION):
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
    nose = heating_figure_of_merit(
        t, rho, V, alt, rng, nose_radius_m=nose_radius_m, body_radius_m=0.0,
        emissivity=emissivity, material=nose_material, mass_kg=0.0,
        frontal_area_m2=0.0, soak_dwell_s=soak_dwell_s)
    f = min(max(float(body_flux_fraction), 1e-3), 1.0)
    # Acreage reference scale: the flank/acreage boundary layer is set by the
    # BODY scale and contains no tip-radius term — referencing the fraction to
    # the sharp-tip stagnation flux would inflate body heating by
    # sqrt(R_body/R_n) (3.8x for SWERVE's 1.7 cm tip on a 24 cm body).  So the
    # fraction multiplies the body-scale stagnation flux (same reference the
    # heat_sink criterion already uses); sharp tip and blunt capsule then give
    # the same acreage estimate for the same body.
    _R_ref = body_radius_m if body_radius_m > 0.0 else nose_radius_m
    body = heating_figure_of_merit(
        t, rho, V, alt, rng, nose_radius_m=_R_ref / f ** 2,
        body_radius_m=body_radius_m, emissivity=emissivity,
        material=body_material, mass_kg=mass_kg,
        frontal_area_m2=frontal_area_m2, soak_dwell_s=soak_dwell_s)

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
    if T >= NOTHING_SURVIVES_K:
        return ("analysis", "⚠", T, q, "peak T_eq beyond the no-ablation model — needs ablation analysis")
    if not mat:
        return ("none", "–", T, q, "no TPS material set")
    cmp = res.get("compromise")
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
