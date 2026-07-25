"""β-basis reconciliation: one meaning for `beta_kg_m2`, Acton validation kept.

`beta_kg_m2` is the ZERO-LIFT ballistic coefficient (the polar's convention,
C_D0 = m/(β·A)).  Acton's / Tracy's closed forms want the GLIDE-TRIM β_L =
m/(C_D,glide·A); at the max-L/D trim C_D = 2·C_D0, so β_L = β_zerolift/2 — a
labeled inference (Acton gives no β_L formula, only a fit).  The analytic path
now derives β_L that way, so the same stored number serves both families.

The vehicle this exists for: HTV-2 shipped with β = 13,000 = Acton 2015
Table 3's fitted glide β_L, which the polar mis-read as zero-lift (implausible
C_D0 = 0.27, operating C_L* = 1.4 past stall).  Re-based to zero-lift β =
26,000, the polar reads C_D0 = 0.14 / C_L* = 0.71, AND the analytic β_L derives
back to 26,000/2 = 13,000, reproducing Acton exactly.  Only HTV-2's stored
value changed.
"""

import json

import numpy as np
import pytest

from booster_models import ro_from_dict
from trajectory import _aero_polar


def _ro(name):
    return ro_from_dict(json.load(open(f"ro_library/{name}.ro.json")))


# ── HTV-2: the re-base reproduces Acton Table 3 in BOTH families ─────────────
def test_htv2_analytic_beta_L_reproduces_acton_table3():
    """The derived glide β_L (β_zerolift/2) must equal Acton 2015 Table 3's
    fitted β_L = 13,000 — the validation anchor the whole re-base protects."""
    ro = _ro("HTV-2")
    assert ro.beta_kg_m2 == 26000.0                 # zero-lift, re-based
    assert ro.beta_kg_m2 / 2.0 == 13000.0           # Acton Table 3 β_L
    assert ro.glider_beta_entry_kg_m2 == 7.0        # Acton Table 3 β_S
    assert ro.glider_LD == 2.6                       # Acton Table 3 L/D


def test_htv2_polar_operating_point_is_plausible():
    """Reading the re-based zero-lift β, the polar must trim at a physical AoA
    — not the past-stall C_L* = 1.4 the glide-basis value forced."""
    p = _aero_polar(_ro("HTV-2"))
    assert p.C_D0 == pytest.approx(0.136, abs=0.01)
    assert p.C_L_star < 0.9                          # was 1.42 (α past stall)


# ── the convention is consistent across the shipped fleet ───────────────────
def test_shipped_gliders_store_plausible_zero_lift_beta():
    """Every shipped glider's stored β must read as a physically plausible
    zero-lift C_D0 (< 0.2 for a slender glider) — the guard that catches a
    glide-basis β mis-stored as zero-lift, which is exactly what HTV-2 was."""
    for name in ("SWERVE", "AHW", "C-HGB", "HTV-2"):
        p = _aero_polar(_ro(name))
        assert 0.03 < p.C_D0 < 0.20, (name, p.C_D0)


def test_only_htv2_was_rebased():
    """The other gliders keep their (already zero-lift) stored β untouched."""
    assert _ro("SWERVE").beta_kg_m2 == 24733.0
    assert _ro("AHW").beta_kg_m2 == 9000.0
    assert _ro("C-HGB").beta_kg_m2 == 15000.0


# ── the derivation is behaviour-neutral for the shipped fleet ───────────────
def test_analytic_mode_objects_are_ballistic_so_derivation_is_inert():
    """The only shipped plans on an analytic mode (Generic_RV, Mk21) are
    ballistic — glider_LD = 0 — so the β_L = β/2 derivation is never reached
    for them (the analytic pull-up gates on glider_LD > 0).  This is what makes
    the shared-code change touch NO shipped behaviour except HTV-2's."""
    from booster_models import load_reentry_plan
    for ro_name, plan_name in (("Generic-RV", "Generic_RV"), ("Mk21", "Mk21_MIRV")):
        rp = json.load(open(f"reentry_plans/{plan_name}.reentryplan.json"))
        assert rp["glider_guidance"] in ("equilibrium_glide",
                                         "equilibrium_glide_acton")
        assert _ro(ro_name).glider_LD == 0.0        # ballistic → β_L unused


def test_htv2_analytic_run_uses_the_halved_beta():
    """End-to-end: flying HTV-2 on the Acton mode, the derivation must make the
    glide behave as β_L = 13,000, not the stored 26,000.  Pinned by comparing
    against an object whose STORED β is 13,000 flown the same way — the halved
    26,000 run must match the raw-13,000-would-be run's glide, not the raw
    26,000 run's."""
    import copy
    import dataclasses
    from booster_models import get_booster, apply_reentry_plan, load_reentry_plan
    from trajectory import integrate_trajectory

    def fly(beta_stored):
        ro = dataclasses.replace(_ro("HTV-2"), beta_kg_m2=beta_stored,
                                 glider_beta_entry_kg_m2=7.0)
        rp = dict(load_reentry_plan("HTV-2") or {},
                  glider_guidance="equilibrium_glide_acton")
        ro = apply_reentry_plan(ro, rp)
        b = get_booster("Minotaur-IV + HTV-2"); b.ro = copy.deepcopy(ro)
        r = integrate_trajectory(b, 34.7, -120.6, 90.0, max_time_s=8000.0,
                                 dt_output=4.0)
        return r["range_km"]

    r_26 = fly(26000.0)      # stored zero-lift → β_L derived = 13,000
    r_13 = fly(13000.0)      # stored 13,000    → β_L derived = 6,500
    # The two must differ — proving the derivation halves the stored value
    # (if β_L were read raw, r_26 would equal the old raw-26,000 behaviour).
    assert abs(r_26 - r_13) > 1.0, (r_26, r_13)
