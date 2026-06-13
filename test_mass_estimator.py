"""
Smoke / regression tests for mass_estimator.py.

Run:  python test_mass_estimator.py        (no third-party dependencies)

Each test pins the estimator against a published worked example or the source
regression so coefficient typos are caught.  Tolerances are loose where the
underlying relation is itself a regression (RSE 10–25%).
"""

import math
import mass_estimator as mest


def _close(a, b, rel=0.02, msg=""):
    assert abs(a - b) <= rel * abs(b) + 1e-9, f"{msg}: {a} vs {b} ({rel:.0%})"


def test_akin_individual_mers():
    # Akin ENAE 791 worked-example component values (SSTO LOX/LH2 stage).
    _close(mest.tank_mass_by_propmass(116_400, "LOX"), 1245, rel=0.01,
           msg="LOX tank")
    _close(mest.tank_mass_by_propmass(19_390, "LH2"), 2482, rel=0.01,
           msg="LH2 tank")
    # Engine: the printed Akin MER gives ~641 kg at 324.9 kN, ε=30
    # (Akin's slide *table* says 373 kg each, but that contradicts his own
    # formula and the independent Zandbergen fit — see note below).
    _close(mest.engine_mass_akin(324_900, 30), 641, rel=0.02, msg="engine MER")
    # Independent cross-check: Zandbergen hydrolox must agree within its RSE.
    _close(mest.engine_mass_zandbergen(324_900, "hydrolox"),
           mest.engine_mass_akin(324_900, 30), rel=0.20, msg="engine cross-check")
    _close(mest.thrust_structure_mass(6 * 324_900), 497, rel=0.02,
           msg="thrust structure")
    _close(mest.avionics_mass(153_000), 744, rel=0.02, msg="avionics")
    # Gimbal: Akin example ~81 kg at Pc≈6.9 MPa, T=324.9 kN/engine ×6.
    _close(mest.gimbal_mass(6 * 324_900, 6.9e6), 81, rel=0.15, msg="gimbals")
    # Fairing exponent check (4.95·A^1.15): 69.03 m² → ~645 kg in Akin example.
    _close(mest.fairing_mass(69.03), 645, rel=0.05, msg="fairing")


def test_liquid_stage_matches_akin_total():
    inp = mest.LiquidStageInputs(
        propellant="LOX/LH2", prop_mass_kg=135_800, thrust_n=6 * 324_900,
        n_engines=6, expansion_ratio=30, diameter_m=4.2, length_m=40,
        chamber_pressure_pa=6.9e6, gross_mass_kg=153_000)
    est = mest.estimate_liquid_stage(inp)
    # Akin's first-pass inert estimate is ~10.9 t; we should land within 10%.
    _close(est.total_kg, 10_870, rel=0.10, msg="liquid stage total")


def test_pietrobon_hydrolox():
    # ms = 0.19·mp^0.848 (tonnes).  135.8 t propellant.
    expect = 0.19 * 135.8 ** 0.848 * 1000.0
    _close(mest.pietrobon_stage_mass(135_800, "average"), expect, rel=0.001,
           msg="Pietrobon avg")
    # Common-bulkhead variant must be lighter than the average.
    assert (mest.pietrobon_stage_mass(135_800, "common_bulkhead")
            < mest.pietrobon_stage_mass(135_800, "average"))


def test_solid_stage_regressions():
    # Zandbergen 2026, tonnes.  Steel S2: 0.1689·mp + 0.509.
    _close(mest.solid_stage_inert(100_000, "steel", "linear"),
           (0.1689 * 100 + 0.509) * 1000, rel=0.001, msg="steel linear")
    # Steel power S3: 0.2851·mp^0.903.
    _close(mest.solid_stage_inert(100_000, "steel", "power"),
           0.2851 * 100 ** 0.9030 * 1000, rel=0.001, msg="steel power")
    # Composite is lighter than steel for the same propellant load.
    assert (mest.solid_stage_inert(100_000, "composite", "power")
            < mest.solid_stage_inert(100_000, "steel", "power"))
    # Vega P80 (composite): 87.71 t prop, observed inert 8.533 t → within RSE.
    est = mest.solid_stage_inert(87_710, "composite", "power")
    _close(est, 8533, rel=0.25, msg="Vega P80")


def test_solid_lewis_ng_catalog():
    LB = 0.45359237
    # Spec sanity check: 30,000 lbm composite → m_inert ≈ 2,950 lbm.
    est_lb = mest.solid_stage_inert(30_000 * LB, "composite", "power",
                                    source="lewis") / LB
    _close(est_lb, 2950, rel=0.03, msg="NG spec sanity (composite 30k lbm)")
    # Constant-fraction model: inert/loaded = 0.092 (composite).
    mp_lb = 30_000.0
    mi_lb = mest.solid_stage_inert(mp_lb * LB, "composite", "linear",
                                   source="lewis") / LB
    _close(mi_lb / (mi_lb + mp_lb), 0.092, rel=0.001, msg="composite fraction")
    # Steel ≈ 1.5× composite at the same load (documented material penalty).
    c = mest.solid_stage_inert(50_000 * LB, "composite", "power", source="lewis")
    s = mest.solid_stage_inert(50_000 * LB, "steel", "power", source="lewis")
    _close(s / c, 1.50, rel=0.01, msg="steel/composite penalty")
    # Best-in-class runs lighter than the broader Zandbergen sample.
    assert (mest.solid_stage_inert(50_000 * LB, "composite", "power", source="lewis")
            < mest.solid_stage_inert(50_000 * LB, "composite", "power"))


def test_structural_coefficient_roundtrip():
    mp, eps = 100_000.0, 0.1
    mi = mest.inert_from_structural_coefficient(mp, eps)
    _close(mi / (mi + mp), eps, rel=0.001, msg="epsilon round-trip")


def test_mass_fraction():
    mp, zeta = 100_000.0, 0.9
    mi = mest.inert_from_mass_fraction(mp, zeta)
    _close(mp / (mp + mi), zeta, rel=0.001, msg="zeta round-trip")


def test_tank_material_scaling():
    base = mest.LiquidStageInputs(propellant="LOX/RP1", prop_mass_kg=300_000,
                                  thrust_n=7e6, n_engines=9, diameter_m=3.7)
    al = mest.estimate_liquid_stage(base).total_kg
    base.tank_material = "composite"
    comp = mest.estimate_liquid_stage(base).total_kg
    base.tank_material = "steel"
    steel = mest.estimate_liquid_stage(base).total_kg
    # Composite tanks lighten the stage; steel tanks make it heavier.
    assert comp < al < steel, f"{comp} < {al} < {steel}"
    # Factor sanity: composite tank coeff is 0.45× aluminium.
    _close(mest.material_tank_factor("composite"), 0.45, rel=0.001)
    _close(mest.material_tank_factor("aluminum"), 1.0, rel=0.001)


def test_avionics_scope():
    # Booster (no guidance avionics) must omit the Avionics line but keep wiring.
    booster = mest.LiquidStageInputs(
        propellant="LOX/RP1", prop_mass_kg=300_000, thrust_n=7e6, n_engines=9,
        diameter_m=3.7, length_m=40, include_avionics=False)
    names = [c.name for c in mest.estimate_liquid_stage(booster).components]
    assert not any("Avionics" in n for n in names), names
    assert any("Wiring" in n for n in names), names
    # Upper stage sizes avionics on vehicle GLOW, not its own gross mass.
    upper = mest.LiquidStageInputs(
        propellant="LOX/LH2", prop_mass_kg=20_000, thrust_n=1e6, diameter_m=4,
        length_m=10, gross_mass_kg=23_000, include_avionics=True,
        vehicle_gross_kg=500_000)
    av = next(c.mass_kg for c in mest.estimate_liquid_stage(upper).components
              if "Avionics" in c.name)
    _close(av, mest.avionics_mass(500_000), rel=0.001, msg="avionics on GLOW")


def test_physics_tank():
    # Sanity magnitude: aluminium kerolox tank, ~few mm wall, hundreds–few t.
    m, det = mest.tank_structural_mass(
        180_000, 1140.0, 3.81, 20.0, thrust_n=3.8e6, gross_mass_kg=284_000,
        lateral_g=0.5, material="aluminium")
    assert 0.5 < det["t_mm"] < 30, det
    assert 200 < m < 6000, m
    # Higher lateral load → heavier (bending drives thickness).
    m_hi, _ = mest.tank_structural_mass(
        180_000, 1140.0, 3.81, 20.0, thrust_n=3.8e6, gross_mass_kg=284_000,
        lateral_g=3.0, material="aluminium")
    assert m_hi >= m
    # Composite lighter than steel for the same tank.
    mc, _ = mest.tank_structural_mass(180_000, 1140.0, 3.81, 20.0, 3.8e6,
                                      284_000, material="composite")
    ms, _ = mest.tank_structural_mass(180_000, 1140.0, 3.81, 20.0, 3.8e6,
                                      284_000, material="steel")
    assert mc < ms
    # Physics-tank path runs end to end and labels its tanks.
    inp = mest.LiquidStageInputs(propellant="LOX/RP1", prop_mass_kg=284_000,
                                 thrust_n=3.8e6, diameter_m=3.81, length_m=32,
                                 physics_tank=True)
    names = [c.name for c in mest.estimate_liquid_stage(inp).components]
    assert any("GT-STRESS" in n for n in names), names


def test_shu_engine_mass_ratio():
    # M_struct = M_engine/κ_E; total inert = M_engine·(1 + 1/κ_E).
    m_eng = mest.total_engine_mass(3.8e6, 1, "kerolox", 30)
    inert, eng = mest.engine_mass_ratio_inert(3.8e6, 0.25, 1, "kerolox", 30)
    _close(eng, m_eng, rel=0.001, msg="engine mass passthrough")
    _close(inert, m_eng * (1 + 1 / 0.25), rel=0.001, msg="Shu inert")
    # Smaller κ_E ⇒ heavier structure relative to engine ⇒ larger inert.
    big, _ = mest.engine_mass_ratio_inert(3.8e6, 0.10, 1, "kerolox", 30)
    assert big > inert
    # Appears as an aggregate line for a (non-hydrolox) liquid stage.
    est, _ = mest.analyse_liquid(mest.LiquidStageInputs(
        propellant="LOX/RP1", prop_mass_kg=284_000, thrust_n=3.8e6,
        stage_role="lower"))
    assert any("Engine-mass-ratio" in e.method for e in est)


def test_akin_offset_tank():
    # LOX offset: 0.0152·m + 318 ; LH2: 0.0694·m + 363.
    _close(mest.tank_mass_offset(116_400, "LOX"), 0.0152 * 116_400 + 318,
           rel=0.001, msg="LOX offset")
    _close(mest.tank_mass_offset(19_390, "LH2"), 0.0694 * 19_390 + 363,
           rel=0.001, msg="LH2 offset")
    # Offset tank model runs end to end.
    est = mest.estimate_liquid_stage(mest.LiquidStageInputs(
        propellant="LOX/LH2", prop_mass_kg=135_800, thrust_n=1.9e6,
        tank_model="akin_offset"))
    assert est.total_kg > 0


def test_averaged_tank_is_mean():
    base = dict(propellant="LOX/RP1", prop_mass_kg=284_000, thrust_n=3.8e6,
                diameter_m=3.81, length_m=32)
    emp = mest.estimate_liquid_stage(mest.LiquidStageInputs(
        tank_model="akin_volume", **base))
    phys = mest.estimate_liquid_stage(mest.LiquidStageInputs(
        tank_model="physics", **base))
    avg = mest.estimate_liquid_stage(mest.LiquidStageInputs(
        tank_model="averaged", **base))
    # Averaged total sits between the two source models.
    assert min(emp.total_kg, phys.total_kg) <= avg.total_kg <= \
        max(emp.total_kg, phys.total_kg)


def test_goldyn_ceiling():
    # ε_max = 1/exp(Δv/(g0·Isp)); higher Isp/lower Δv ⇒ higher ceiling.
    e1 = mest.structural_index_ceiling(3000, 311)
    e2 = mest.structural_index_ceiling(3000, 450)
    assert 0 < e1 < 1 and e2 > e1
    _close(e1, 1 / math.exp(3000 / (9.80665 * 311)), rel=0.001, msg="ceiling")
    # An impossible stage (huge Δv, low Isp) flags the warning note.
    est, _ = mest.analyse_liquid(mest.LiquidStageInputs(
        propellant="LOX/RP1", prop_mass_kg=10_000, thrust_n=3.0e5,
        delta_v_ms=9000, isp_s=250))
    assert any("feasibility ceiling" in n for e in est for n in e.notes)


def test_gaspar_confirms_akin_coefficients():
    # Gaspar (2014) lists the same Akin MER coefficients we implement.
    _close(mest.thrust_structure_mass(1e6), 2.55e-4 * 1e6, rel=0.001)
    _close(mest.engine_mass_akin(1e6, 30),
           7.81e-4 * 1e6 + 3.37e-5 * 1e6 * 30 + 59, rel=0.001)
    _close(mest.tank_mass_by_propmass(1000, "LOX"), 10.7, rel=0.001)
    _close(mest.tank_mass_by_propmass(1000, "LH2"), 128.0, rel=0.001)
    _close(mest.tank_mass_by_propmass(1000, "RP1"), 14.8, rel=0.001)
    _close(mest.cryo_insulation_mass(100, "LH2"), 288.0, rel=0.001)
    _close(mest.fairing_mass(100), 4.95 * 100 ** 1.15, rel=0.001)


def test_epsilon_in_divergence():
    est = [mest.MassEstimate("x", 1000.0)]
    rep = mest.divergence_report(1000.0, est, prop_mass_kg=9000.0)
    # dry=1000, prop=9000 → ε = 1000/10000 = 0.10
    _close(rep[0].eps_stated, 0.10, rel=0.001, msg="eps stated")
    _close(rep[0].eps_estimate, 0.10, rel=0.001, msg="eps estimate")
    assert "ε =" in mest.format_divergence(rep)


def test_divergence_signs():
    est = [mest.MassEstimate("x", 1000.0)]
    rep = mest.divergence_report(1200.0, est)
    _close(rep[0].pct, 20.0, rel=0.001, msg="divergence +20%")
    assert rep[0].verdict() == "marginal"
    rep = mest.divergence_report(500.0, est)
    assert rep[0].pct == -50.0 and "optimistic" in rep[0].verdict()


def test_tank_surface_area_sphere():
    # Sphere of volume V has area 4π(3V/4π)^(2/3).
    v = 100.0
    r = (v / (4 / 3 * math.pi)) ** (1 / 3)
    _close(mest.tank_surface_area(v), 4 * math.pi * r ** 2, rel=0.001,
           msg="sphere area")


def test_analyse_liquid_and_solid_run():
    le, lr = mest.analyse_liquid(
        mest.LiquidStageInputs(propellant="LOX/RP1", prop_mass_kg=280_000,
                               thrust_n=7.6e6, n_engines=1, diameter_m=3.7,
                               length_m=40), stated_dry_kg=18_000)
    assert le and lr and all(d.estimate_kg > 0 for d in lr)
    se, sr = mest.analyse_solid(
        mest.SolidStageInputs(prop_mass_kg=45_000, thrust_n=2.2e6,
                              casing="composite"), stated_dry_kg=4000)
    assert se and sr and all(d.estimate_kg > 0 for d in sr)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} tests passed.")


if __name__ == "__main__":
    main()
