"""Smoke/▢ tests for damping_estimate.estimate_damping (docs/damping_estimate_spec.md)."""
from damping_estimate import estimate_damping, estimate_for_ro


def _ordered(r):
    assert r.zeta_lo <= r.zeta <= r.zeta_hi + 1e-9, \
        f"band not ordered: {r.zeta_lo}/{r.zeta}/{r.zeta_hi}"


def test_ballistic_is_zero():
    r = estimate_damping(10000.0, 0.0)
    assert r.zeta == 0.0 and r.mode == "ballistic"
    print(f"  ok  ballistic (L/D=0) -> zeta=0")


def test_none_tier_floor():
    r = estimate_damping(15000.0, 2.0, control="none")
    _ordered(r)
    assert r.mode == "none" and r.zeta <= 0.2 and r.zeta_hi <= 0.25
    print(f"  ok  none -> {r.text()} (undamped/drag-only floor)")


def test_unknown_is_wide():
    r = estimate_damping(15000.0, 2.0, control="unknown")
    _ordered(r)
    assert r.zeta_lo <= 0.05 and r.zeta_hi >= 0.6, "unknown band should be wide"
    print(f"  ok  unknown -> {r.text()} (wide band)")


def test_tiers_increase():
    zn = estimate_damping(15000.0, 2.0, control="none").zeta
    zs = estimate_damping(15000.0, 2.0, control="small").zeta
    zb = estimate_damping(15000.0, 2.0, control="substantial").zeta
    assert zn < zs < zb, f"tiers not increasing: {zn} {zs} {zb}"
    print(f"  ok  tiers increase: none {zn:.2f} < small {zs:.2f} < substantial {zb:.2f}")


def test_substantial_competent():
    r = estimate_damping(15000.0, 2.0, control="substantial")
    _ordered(r)
    assert 0.4 <= r.zeta <= 0.8, f"substantial central out of band: {r.zeta}"
    print(f"  ok  substantial -> {r.text()}")


def test_computed_monotone_in_area():
    prev = -1.0
    for ratio in (0.0 + 1e-6, 0.05, 0.1, 0.2, 0.3):
        z = estimate_damping(15000.0, 2.0, flap_area_ratio=ratio).zeta
        assert z >= prev - 1e-9, f"non-monotone in flap area at {ratio}: {z} < {prev}"
        prev = z
    print(f"  ok  zeta increases monotonically with S_flap/S_ref")


def test_chgb_computed_band():
    # C-HGB: beta 15000, L/D 2, substantial fins (S_fin/S_ref ~ 0.30)
    r = estimate_damping(15000.0, 2.0, control="substantial", flap_area_ratio=0.30)
    _ordered(r)
    assert r.mode == "computed" and 0.45 <= r.zeta <= 0.8, f"C-HGB zeta {r.zeta}"
    assert r.zeta_hi >= 0.7, "C-HGB should reach the textbook 0.7 at the top of its band"
    assert 28.0 <= r.h_eq_km_lo and r.h_eq_km_hi <= 45.0, \
        f"equilibrium glide band {r.h_eq_km_lo}-{r.h_eq_km_hi} km should bracket 30-40 km"
    print(f"  ok  C-HGB computed -> {r.text()}  h_eq {r.h_eq_km_lo:.0f}-{r.h_eq_km_hi:.0f} km, M={r.margin_M:.2f}")


def test_flown_state_override():
    r = estimate_damping(15000.0, 2.0, flap_area_ratio=0.2, v_glide=4000.0, h_glide=36000.0)
    _ordered(r)
    assert r.zeta > 0
    print(f"  ok  flown-state override -> {r.text()}")


def test_ro_wrapper():
    class _RO:
        beta_kg_m2 = 15000.0; glider_LD = 2.0; glider_pullup_g_max = 10.0
        glider_control_surfaces = "substantial"; glider_flap_area_ratio = 0.0
        glider_flap_deflection_deg = 0.0
    r = estimate_for_ro(_RO())
    _ordered(r)
    print(f"  ok  estimate_for_ro -> {r.text()}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} damping-estimate checks passed.")


if __name__ == "__main__":
    main()
