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
