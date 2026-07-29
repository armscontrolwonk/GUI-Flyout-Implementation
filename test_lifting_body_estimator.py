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
