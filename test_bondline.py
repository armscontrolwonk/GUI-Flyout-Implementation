"""Interior (bondline) survivability screen — heating.bondline_screen and its
report wiring (METHODS §13.10).

The "does the inside survive" axis: 1-D transient conduction through the body
TPS to the structure bondline, vs the ablative-TPS sizing limit (Dec & Braun
NTRS 20060004824, 250 °C).  Crossing it is BEYOND DESIGN ENVELOPE (yellow) — a
design criterion, never red.  Evaluated only for an ablative body with a cited
through-thickness conductivity.
"""

import numpy as np

import heating
import survivability_report as sr


def _flat(q_MW, dur_s, n=201):
    t = np.linspace(0.0, dur_s, n)
    return t, np.full(n, q_MW * 1e6)


def test_thick_short_pulse_interior_safe():
    # A fast ballistic pulse through a thick shield never reaches the bondline.
    t = np.linspace(0.0, 25.0, 51)
    q = 3e6 * np.sin(np.pi * t / 25.0)
    r = heating.bondline_screen(t, q, material="carbon_phenolic", thickness_m=0.10)
    assert r["evaluated"] and not r["crossed"]
    assert r["T_bond_peak_C"] < 30.0          # essentially untouched


def test_thin_layer_long_soak_cooks_interior():
    # A thin layer under a sustained glide-class soak crosses the limit.
    t, q = _flat(1.0, 800.0)                   # 1 MW/m² for 800 s
    r = heating.bondline_screen(t, q, material="carbon_phenolic", thickness_m=0.003)
    assert r["evaluated"] and r["crossed"]
    assert r["t_cross_s"] is not None and r["t_cross_s"] < 800.0


def test_uncited_material_not_evaluated():
    # PICA has no cited k → the screen honestly declines rather than guessing.
    t, q = _flat(1.0, 500.0)
    r = heating.bondline_screen(t, q, material="pica", thickness_m=0.05)
    assert r["evaluated"] is False


def test_sirca_now_evaluates_with_tpsx_k():
    # SIRCA-14A k = 0.0629 W/m·K (TPSX id 41; supersedes the Murbach-1997
    # ratio estimate 0.04) → the bondline screen runs for SIRCA bodies.
    # The low DENSITY matters more than the low k: SIRCA's thermal
    # diffusivity k/(ρc) slightly EXCEEDS silica phenolic's (2.3e-7 vs
    # 2.1e-7 m²/s), so at EQUAL thickness it soaks through FASTER under a
    # sustained load — SIRCA's insulation lever is thickness bought with its
    # 7.6× lower density (per-mass, not per-centimeter).
    m = heating.TPS_MATERIALS["sirca"]
    sp_m = heating.TPS_MATERIALS["silica_phenolic"]
    assert abs(m["k_W_mK"] - 0.0629) < 1e-9
    a_s = m["k_W_mK"] / (m["density_kg_m3"] * m["c_J_kgK"])
    a_sp = sp_m["k_W_mK"] / (sp_m["density_kg_m3"] * sp_m["c_J_kgK"])
    assert a_s > a_sp                     # the counterintuitive ordering, pinned
    t, q = _flat(0.5, 800.0)
    s2 = heating.bondline_screen(t, q, material="sirca", thickness_m=0.02)
    s5 = heating.bondline_screen(t, q, material="sirca", thickness_m=0.05)
    assert s2["evaluated"] and s5["evaluated"]
    assert s5["T_bond_peak_C"] < s2["T_bond_peak_C"]   # thicker = cooler
    assert "TPSX" in m["k_source"]


def test_steady_state_bounds_and_insulated_back():
    # Long constant flux → surface approaches T_eq; insulated back face → the
    # bondline converges up toward the surface temperature (never above it).
    q = 0.5e6
    t, qa = _flat(q / 1e6, 6000.0, n=1201)
    r = heating.bondline_screen(t, qa, material="carbon_phenolic", thickness_m=0.02)
    T_eq = (q / (0.85 * heating.SIGMA)) ** 0.25
    assert r["T_surf_peak_K"] <= T_eq * 1.02
    assert r["T_bond_peak_K"] <= r["T_surf_peak_K"] + 1.0


def test_bondline_limit_follows_threshold():
    # The dialog knob (BONDLINE_LIMIT_C) drives the crossing: the same run
    # crosses below the peak and clears above it.
    import thresholds
    thresholds.reset(); thresholds.apply()
    t, q = _flat(0.5, 500.0)                    # peaks ~360 °C, inside the clamp
    TH = 0.04
    peak_C = heating.bondline_screen(
        t, q, material="carbon_phenolic", thickness_m=TH)["T_bond_peak_C"]

    thresholds.set_override("bondline_limit_C", peak_C - 50.0); thresholds.apply()
    below = heating.bondline_screen(t, q, material="carbon_phenolic", thickness_m=TH)
    thresholds.set_override("bondline_limit_C", peak_C + 50.0); thresholds.apply()
    above = heating.bondline_screen(t, q, material="carbon_phenolic", thickness_m=TH)

    assert below["crossed"] and not above["crossed"]
    assert below["limit_C"] == peak_C - 50.0 and above["limit_C"] == peak_C + 50.0
    thresholds.reset(); thresholds.apply()


def _green_ballistic():
    """A shipped ballistic RV that reads WITHIN EXPERIENCE — the clean base to
    test bondline escalation against (nose + body both fine)."""
    import copy
    import json
    from booster_models import get_booster, ro_from_dict
    from trajectory import integrate_trajectory
    ro = ro_from_dict(json.load(open("ro_library/Generic-RV.ro.json")))
    p = get_booster("No-dong")
    p.ro = copy.deepcopy(ro)
    return integrate_trajectory(p, 0.0, 0.0, 90.0, max_time_s=6000.0, dt_output=2.0)


def test_report_bondline_escalates_a_green_case_to_beyond():
    # Take a genuinely WITHIN-EXPERIENCE result and inject a crossed bondline
    # (a thin body under a long soak) — the report must escalate green→yellow
    # on the interior axis alone, and say the skin holds but the inside cooks.
    import thresholds
    thresholds.reset(); thresholds.apply()
    r = _green_ballistic()
    assert sr.build_report(r)["tier"] == "experience"       # clean baseline

    t, q = _flat(1.0, 800.0)
    r["heating_fom"]["bondline"] = heating.bondline_screen(
        t, q, material="carbon_phenolic", thickness_m=0.003)
    assert r["heating_fom"]["bondline"]["crossed"]
    rep = sr.build_report(r)
    assert rep["tier"] == "beyond"
    assert "BEYOND DESIGN ENVELOPE" in rep["headline"]
    assert "bondline" in rep["body"].lower()
    assert ("interior does not" in rep["body"]
            or "structure behind it cooks" in rep["body"])
    thresholds.reset(); thresholds.apply()
