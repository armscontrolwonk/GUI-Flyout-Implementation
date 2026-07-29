"""Lifting-body Newtonian α-sweep estimator (Phase 2a core).

Per PHASE2_LIFTING_BODY_PLAN.md, every test here is an IDENTITY (an exact
reduction the closed forms must satisfy), a MEASURED anchor (Fetterman
TN D-2942, screening band), or a DIRECTION — never a fit to a corridor.  The
physics of the trajectory is untouched by any of this; the estimator only
produces β and L/D inputs.

Phase 2a covers the pressure core (cone sector + flat plate), the Eckert Cf
helper, and the HALF-CONE composer/sweep.  The wedge composer (2a part 2) and
the cone/biconic sweep (2b) are not yet under test here.
"""

import math

import pytest

from booster_models import (
    cone_sector_newtonian, flat_plate_newtonian, cf_reference_temperature,
    cd_blunted_cone_newtonian, lifting_body_sweep,
    _wedge_coeffs, _planar_face_force, wedge_planform_area,
    NEWTON_K_SLENDER, NEWTON_K_BLUNT,
)

FULL = (-math.pi / 2.0, 3.0 * math.pi / 2.0)          # full 2π sector


# ── Identity 3: full sharp cone at α=0 ≡ the shipped 2·sin²θ ─────────────────
@pytest.mark.parametrize("theta", [5.0, 10.0, 20.0, 35.0, 45.0])
def test_full_cone_at_zero_alpha_reduces_to_sharp_cone(theta):
    c = cone_sector_newtonian(theta, 0.0, *FULL, K=NEWTON_K_SLENDER)
    assert c['C_A'] == pytest.approx(cd_blunted_cone_newtonian(theta, 0.0), rel=1e-9)
    assert c['C_A'] == pytest.approx(2.0 * math.sin(math.radians(theta)) ** 2, rel=1e-9)
    assert c['C_N'] == pytest.approx(0.0, abs=1e-12)   # axisymmetric → no lift


def test_full_cone_generates_lift_only_under_incidence():
    """A complete cone carries zero normal force at α=0 (axisymmetric to the
    axial flow) but DOES lift at incidence (the C_Nα every cone has), and the
    lift grows with α — the sign/direction check for the sector integral."""
    assert cone_sector_newtonian(12.0, 0.0, *FULL)['C_N'] == pytest.approx(0.0, abs=1e-9)
    cn = [cone_sector_newtonian(12.0, a, *FULL)['C_N'] for a in (5.0, 15.0, 30.0)]
    assert all(x > 0.0 for x in cn)                    # positive = lift
    assert cn[0] < cn[1] < cn[2]                        # grows with α


# ── Identity 4: half-shell = ½ full at α=0, but windward ≠ leeward at α>0 ────
def test_half_shells_split_the_cone_at_zero_alpha():
    full = cone_sector_newtonian(15.0, 0.0, *FULL)
    top = cone_sector_newtonian(15.0, 0.0, math.pi, 2.0 * math.pi)
    bot = cone_sector_newtonian(15.0, 0.0, 0.0, math.pi)
    assert top['C_A'] + bot['C_A'] == pytest.approx(full['C_A'], rel=1e-9)
    assert top['C_A'] == pytest.approx(bot['C_A'], rel=1e-9)      # symmetric at α=0
    assert top['C_A'] == pytest.approx(0.5 * full['C_A'], rel=1e-9)


def test_windward_and_leeward_halves_differ_under_incidence():
    """The guard against the halving shortcut my wash caught: at α>0 the two
    halves carry different loads (one faces the flow, one hides), so ½×full is
    NOT valid at incidence."""
    top = cone_sector_newtonian(15.0, 12.0, math.pi, 2.0 * math.pi)  # leeward
    bot = cone_sector_newtonian(15.0, 12.0, 0.0, math.pi)            # windward
    assert bot['C_A'] > top['C_A'] + 1e-3
    # their axial sum still equals the full cone (partition of the surface)
    full = cone_sector_newtonian(15.0, 12.0, *FULL)
    assert top['C_A'] + bot['C_A'] == pytest.approx(full['C_A'], rel=1e-9)


# ── Identity 1: the flat plate ──────────────────────────────────────────────
@pytest.mark.parametrize("al", [2.0, 5.0, 10.0, 20.0])
def test_flat_plate_newtonian_law(al):
    p = flat_plate_newtonian(al, K=NEWTON_K_SLENDER)
    assert p['C_N'] == pytest.approx(2.0 * math.sin(math.radians(al)) ** 2)
    assert p['C_A'] == 0.0


def test_flat_plate_leeward_is_shadowed():
    assert flat_plate_newtonian(-5.0)['C_N'] == 0.0
    assert flat_plate_newtonian(0.0)['C_N'] == 0.0


# ── Identity 5: K is a pure multiplier on pressure ──────────────────────────
def test_K_scales_pressure_linearly():
    lo = cone_sector_newtonian(20.0, 10.0, math.pi, 2.0 * math.pi, K=NEWTON_K_BLUNT)
    hi = cone_sector_newtonian(20.0, 10.0, math.pi, 2.0 * math.pi, K=NEWTON_K_SLENDER)
    f = NEWTON_K_BLUNT / NEWTON_K_SLENDER
    assert lo['C_A'] == pytest.approx(f * hi['C_A'], rel=1e-12)
    assert lo['C_N'] == pytest.approx(f * hi['C_N'], rel=1e-12)


# ── Eckert Cf helper: shape and direction ───────────────────────────────────
def test_cf_reference_temperature_direction():
    # turbulent > laminar at the same conditions; both fall with Reynolds number
    lam = cf_reference_temperature(8.0, 1e6, turbulent=False)
    turb = cf_reference_temperature(8.0, 1e6, turbulent=True)
    assert turb > lam > 0.0
    assert cf_reference_temperature(8.0, 1e7, turbulent=True) < turb   # ↓ with Re
    # compressibility thins the boundary layer → Cf falls with Mach
    assert cf_reference_temperature(15.0, 1e6, turbulent=True) < \
        cf_reference_temperature(3.0, 1e6, turbulent=True)


# ── Half-cone sweep: structure and the consistent trim row ──────────────────
def test_sweep_reports_consistent_trim_row():
    r = lifting_body_sweep("half_cone", theta_deg=5.0, mach=6.86,
                           reynolds_length=1.43e6, turbulent=False,
                           base_drag=False)
    tr = r['trim']
    # α* is the argmax of L/D over the very rows returned — internally consistent
    ld_at_star = max(row['L_D'] for row in r['alpha'])
    assert tr['LD_max'] == pytest.approx(ld_at_star)
    star_row = next(row for row in r['alpha']
                    if row['alpha_deg'] == tr['alpha_star_deg'])
    assert tr['C_L_star'] == star_row['C_L'] and tr['C_D_star'] == star_row['C_D']
    # conditions travel with the result (Fetterman discipline)
    assert r['conditions']['mach'] == 6.86 and not r['conditions']['base_drag']
    assert r['conditions']['cf'] > 0.0


def test_camber_offset_is_at_slightly_positive_alpha():
    """Flat-side-down: the flat underside only lifts for α>0, so both min-drag
    and zero-lift sit at small POSITIVE α — the C_L0 offset emerges from the
    geometry (measured in TN D-2942 Fig. 6b), it is not assumed."""
    r = lifting_body_sweep("half_cone", theta_deg=5.0, mach=6.86,
                           reynolds_length=1.43e6, turbulent=False,
                           base_drag=False)
    # C_L rises through zero with α; L/D peaks at a positive α
    assert r['trim']['alpha_star_deg'] > 0.0
    assert r['trim']['LD_max'] > 1.0


# ── Measured anchor 6: Fetterman TN D-2942 half-cone body alone ──────────────
# M 6.86, Re_ℓ 1.43×10⁶, laminar, base drag corrected out.  Screening band ±30%.
@pytest.mark.parametrize("theta,expected", [(3.0, 4.6), (5.0, 4.0), (9.0, 3.5)])
def test_fetterman_half_cone_body_alone_LDmax(theta, expected):
    r = lifting_body_sweep("half_cone", theta_deg=theta, mach=6.86,
                           reynolds_length=1.43e6, turbulent=False,
                           base_drag=False)
    ld = r['trim']['LD_max']
    assert 0.7 * expected <= ld <= 1.3 * expected, (theta, ld, expected)


def test_fetterman_LDmax_falls_with_cone_angle():
    """Direction (TN D-2942 Fig. 3): slenderer half-cones glide better."""
    lds = [lifting_body_sweep("half_cone", theta_deg=t, mach=6.86,
                              reynolds_length=1.43e6, turbulent=False,
                              base_drag=False)['trim']['LD_max']
           for t in (3.0, 5.0, 7.5, 9.0)]
    assert all(lds[i] > lds[i + 1] for i in range(len(lds) - 1)), lds


# ── Anchor 11 (body-alone form): friction is a first-order term ─────────────
def test_friction_is_first_order_at_test_reynolds():
    """Friction is not a small perturbation here: at Re_ℓ≈1.4×10⁶ laminar the
    viscous drag at α=0 is at least comparable to the inviscid (pressure) drag.

    NB: Fetterman Fig. 2's specific 2–5× viscous/inviscid ratio is for the thin
    half-cone DELTA-WING combination (the near-flat wing has tiny pressure drag
    and huge wetted area); for the body ALONE the half-cone's own shell
    pressure (∝ sin²θ) is larger, so the ratio is order-unity.  The 2–5× check
    belongs with the wing-body composite in Phase 2b."""
    invisc = lifting_body_sweep("half_cone", theta_deg=5.0, mach=6.86,
                                cf=0.0, base_drag=False)['trim']['C_D0']
    full = lifting_body_sweep("half_cone", theta_deg=5.0, mach=6.86,
                              reynolds_length=1.43e6, turbulent=False,
                              base_drag=False)['trim']['C_D0']
    viscous = full - invisc
    assert viscous > 0.0 and viscous / invisc >= 0.8, (invisc, viscous)


# ═══════════════════════════════════════════════════════════════════════════
# Wedge composer (Phase 2a part 2)
# ═══════════════════════════════════════════════════════════════════════════

# ── Identity 1: the sharp wedge → flat plate as depth → 0 ───────────────────
@pytest.mark.parametrize("al", [2.0, 5.0, 10.0, 18.0])
def test_wedge_reduces_to_flat_plate_at_zero_depth(al):
    """With no ridge (depth→0) only the flat bottom is lit for α>0, giving the
    exact plate law C_L = K·sin²α·cosα, C_D,pressure = K·sin³α."""
    L, b = 4.0, 1.4
    s_ref = wedge_planform_area(L, b)
    c = _wedge_coeffs(L, 1e-7, b, al, NEWTON_K_SLENDER, 0.0, False, 10.0, s_ref)
    a = math.radians(al)
    assert c['C_L'] == pytest.approx(NEWTON_K_SLENDER * math.sin(a) ** 2 * math.cos(a), abs=1e-4)
    assert c['C_D'] == pytest.approx(NEWTON_K_SLENDER * math.sin(a) ** 3, abs=1e-4)


# ── Identity: the top facet Cp equals AEDC-TDR-64-25 Eq. 72 exactly ─────────
@pytest.mark.parametrize("al", [0.0, 2.0, 4.0])
def test_top_facet_pressure_matches_aedc_eq72(al):
    """Independent check of the wedge's sign/geometry bookkeeping: the facet
    pressure built from vertices equals AEDC's closed form
    Cp = K·sin²(ε−α) / (1 + tan²Λ·sin²ε)  (lit region α < ε)."""
    L, b, t = 4.0, 1.4, 0.6
    eps = math.atan2(t, L)
    lam = math.atan2(2.0 * L, b)
    nose, ridge, rt = (0, 0, 0), (L, 0, t), (L, 0.5 * b, 0)
    a = math.radians(al)
    v_hat = (math.cos(a), 0.0, math.sin(a))
    F, area = _planar_face_force((nose, ridge, rt), v_hat, NEWTON_K_SLENDER, (0, 0, 1))
    cp = math.sqrt(sum(x * x for x in F)) / area
    aedc = NEWTON_K_SLENDER * math.sin(eps - a) ** 2 \
        / (1.0 + math.tan(lam) ** 2 * math.sin(eps) ** 2)
    assert cp == pytest.approx(aedc, abs=1e-9)


def test_top_facet_shadows_exactly_at_ridge_angle():
    """The top facet is lit for α < ε and shadowed for α > ε — the transition
    is exactly at the ridge angle ε = atan(t/L)."""
    L, b, t = 4.0, 1.4, 0.6
    eps_deg = math.degrees(math.atan2(t, L))
    nose, ridge, rt = (0, 0, 0), (L, 0, t), (L, 0.5 * b, 0)
    def facet_cp(al):
        a = math.radians(al)
        F, area = _planar_face_force((nose, ridge, rt),
                                     (math.cos(a), 0.0, math.sin(a)),
                                     NEWTON_K_SLENDER, (0, 0, 1))
        return math.sqrt(sum(x * x for x in F)) / area
    assert facet_cp(eps_deg - 1.0) > 0.0        # lit just below ε
    assert facet_cp(eps_deg + 1.0) == 0.0       # shadowed just above ε


# ── Camber offset emerges (not assumed): C_L(0) < 0 ─────────────────────────
def test_wedge_has_negative_lift_at_zero_alpha():
    """At α=0 the flat bottom is edge-on (no load) but the top facets face
    forward and push DOWN, so C_L(0) < 0 and both zero-lift and min-drag sit
    at small POSITIVE α — the camber offset is geometric, not assumed."""
    r = lifting_body_sweep("wedge", length_m=3.6, depth_m=0.5, span_m=0.9,
                           mach=20.0, reynolds_length=1.5e7, turbulent=True)
    row0 = min(r['alpha'], key=lambda x: abs(x['alpha_deg']))
    assert row0['C_L'] < 0.0
    assert r['trim']['alpha_star_deg'] > 0.0


# ── Direction: more friction (lower Re) raises α* and lowers (L/D)max ────────
def test_wedge_alpha_star_rises_as_reynolds_falls():
    star = [lifting_body_sweep("wedge", length_m=3.6, depth_m=0.5, span_m=0.9,
                               mach=20.0, reynolds_length=Re, turbulent=True)['trim']
            for Re in (1e8, 1e7, 1e6)]
    a = [s['alpha_star_deg'] for s in star]
    ld = [s['LD_max'] for s in star]
    assert a[0] < a[1] < a[2]                    # α* rises as Re falls
    assert ld[0] > ld[1] > ld[2]                 # L/D falls as friction grows


# ── Documented limitation: sweep-independent at trim (NOT Fetterman's trend) ─
def test_sharp_wedge_LD_is_nearly_sweep_independent_at_trim():
    """Honest codification of the fidelity limit: with the top facets shadowed
    at trim, the sharp flat-bottom wedge's (L/D)max barely depends on planform
    sweep (only through a small change in top-facet wetted area) — span sets the
    reference area and the picture, not this L/D.  Contrast Fetterman's STRONG
    sweep trend, a bluntness/tip effect above this fidelity that we do NOT
    claim."""
    def ld(b):
        return lifting_body_sweep("wedge", length_m=1.0, depth_m=0.025, span_m=b,
                                  mach=6.8, reynolds_length=1.5e6,
                                  turbulent=False, base_drag=False)['trim']['LD_max']
    assert ld(0.353) == pytest.approx(ld(0.73), rel=5e-3)   # 80° vs 70°: <0.5%


# ── Anchor 13: sharp wedge is an UPPER BOUND above the real HTV-2 ────────────
def test_wedge_is_an_upper_bound_over_real_htv2():
    """Sharp-Newtonian over-predicts the real (blunt, viscous) HTV-2: our L/D
    sits ABOVE Candler's CFD ≈2.5 yet BELOW the viscous-waverider ceiling, and
    α* is a plausible hypersonic trim.  We assert the bracket, not a fit."""
    r = lifting_body_sweep("wedge", length_m=3.6, depth_m=0.5, span_m=0.9,
                           mach=20.0, reynolds_length=1.5e7, turbulent=True)
    ld = r['trim']['LD_max']
    assert 2.5 < ld < 6.0                        # above real, below waverider
    assert 5.0 <= r['trim']['alpha_star_deg'] <= 18.0


# ── β / reference area: wedge is planform-referenced ────────────────────────
def test_wedge_beta_uses_planform_reference():
    r = lifting_body_sweep("wedge", length_m=3.6, depth_m=0.5, span_m=0.9,
                           mach=20.0, reynolds_length=1.5e7, turbulent=True,
                           mass_kg=900.0)
    assert r['conditions']['a_ref_kind'] == "planform"
    sp = wedge_planform_area(3.6, 0.9)
    assert r['conditions']['a_ref_m2'] == pytest.approx(sp)
    # β = m / (C_D0 · A_ref) at the near-zero-α row
    assert 'beta_zero_lift' in r['trim'] and r['trim']['beta_zero_lift'] > 0.0
    assert r['conditions']['sweep_deg'] == pytest.approx(
        math.degrees(math.atan2(2 * 3.6, 0.9)))
