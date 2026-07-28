"""Interstage adapters and conical stages (Phase 1: geometry + mass).

The contract, held to identities not eyeballs:
  * an interstage is carried as mass from launch until its jettison event
    (default = the carrying stage's separation; else a manual time) and NOT
    counted in throw weight;
  * geometry is DERIVED, never invented — the schematic frustum spans this
    stage's top to the next stage's base;
  * Phase 1 is drag-neutral: declaring a taper or an interstage must not move
    the boost reference area;
  * every field defaults off, so a vehicle that opts into nothing is
    byte-identical in mass-vs-time to before the feature existed.
"""

import json

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

import pytest

from booster_models import (get_booster, booster_from_dict, booster_to_dict,
                            booster_mass, booster_area, BOOSTER_DB)
from booster_schematic import draw_booster


def _two_stage():
    return get_booster("Taepodong-I")          # 2-stage, distinct diameters


# ── serialization ───────────────────────────────────────────────────────────
def test_new_fields_round_trip():
    b = _two_stage()
    b.conical = True; b.top_diameter_m = 1.0
    b.has_interstage = True; b.interstage_length_m = 1.5
    b.interstage_mass_kg = 300.0; b.interstage_jettison_s = None
    b2 = booster_from_dict(booster_to_dict(b))
    assert b2.conical and b2.top_diameter_m == 1.0
    assert b2.has_interstage and b2.interstage_length_m == 1.5
    assert b2.interstage_mass_kg == 300.0
    assert b2.interstage_jettison_s is None
    # a manual jettison time survives too
    b.interstage_jettison_s = 42.0
    assert booster_from_dict(booster_to_dict(b)).interstage_jettison_s == 42.0


# ── mass accounting ─────────────────────────────────────────────────────────
def test_interstage_mass_carried_then_dropped_at_stage_separation():
    base = _two_stage()
    ist = _two_stage()
    ist.has_interstage = True; ist.interstage_mass_kg = 300.0
    ist.interstage_length_m = 1.2; ist.interstage_jettison_s = None   # = stage-1 sep
    sep = ist.booster_core_delay_s + ist.burn_time_s
    # carried through stage-1 flight
    assert booster_mass(ist, sep - 1.0) - booster_mass(base, sep - 1.0) == \
        pytest.approx(300.0)
    # gone the instant stage 1 separates
    assert booster_mass(ist, sep + 1.0) == pytest.approx(booster_mass(base, sep + 1.0))


def test_manual_jettison_time_overrides_the_default():
    base = _two_stage()
    ist = _two_stage()
    ist.has_interstage = True; ist.interstage_mass_kg = 250.0
    ist.interstage_length_m = 1.0; ist.interstage_jettison_s = 5.0    # very early
    assert booster_mass(ist, 4.0) - booster_mass(base, 4.0) == pytest.approx(250.0)
    assert booster_mass(ist, 6.0) == pytest.approx(booster_mass(base, 6.0))


def test_interstage_not_in_throw_weight():
    """The interstage is a booster-side adapter, not payload — it must never
    inflate payload_kg / the throw-weight tally."""
    b = _two_stage()
    pw = b.payload_kg
    b.has_interstage = True; b.interstage_mass_kg = 500.0
    assert b.payload_kg == pw


# ── Phase-1 drag neutrality ─────────────────────────────────────────────────
def test_conical_and_interstage_do_not_move_the_reference_area():
    """Phase 1 is geometry + mass only.  A taper or interstage must leave the
    boost reference area exactly as it was (drag coupling is Phase 2)."""
    base = _two_stage()
    a0 = booster_area(base, altitude_m=0.0, top_params=base)
    tap = _two_stage(); tap.conical = True; tap.top_diameter_m = 0.5
    tap.has_interstage = True; tap.interstage_mass_kg = 300.0; tap.interstage_length_m = 1.0
    a1 = booster_area(tap, altitude_m=0.0, top_params=tap)
    assert a1 == pytest.approx(a0)


# ── back-compat ─────────────────────────────────────────────────────────────
def test_every_library_vehicle_unchanged_when_nothing_opted_in():
    import numpy as np
    from booster_models import total_burn_time
    for name in BOOSTER_DB:
        p = get_booster(name)
        p2 = booster_from_dict(booster_to_dict(p))
        T = max(1.0, total_burn_time(p))
        for t in np.linspace(0.0, T * 1.3, 30):
            assert booster_mass(p, t) == pytest.approx(booster_mass(p2, t)), (name, t)


# ── schematic geometry ──────────────────────────────────────────────────────
def _ax():
    return Figure(figsize=(5, 8)).add_subplot(111)


def test_schematic_interstage_uses_derived_diameters_and_adds_height():
    """The interstage frustum spans this stage's top → the next stage's base,
    and its length adds to the stack height — nothing invented."""
    plain = _two_stage()
    h0 = draw_booster(_ax(), plain)["total_height_m"]
    ist = _two_stage()
    ist.has_interstage = True; ist.interstage_length_m = 1.3; ist.interstage_mass_kg = 300.0
    ax = _ax()
    h1 = draw_booster(ax, ist)["total_height_m"]
    assert h1 == pytest.approx(h0 + 1.3)
    # a frustum polygon exists spanning S1-top (1.32) to S2-base (0.84)
    d_bot, d_top = ist.diameter_m, ist.stage2.diameter_m
    widths = set()
    for patch in ax.patches:
        if isinstance(patch, Polygon):
            xs = patch.get_path().vertices[:, 0]
            widths.add(round(xs.max() - xs.min(), 4))
    assert round(d_bot, 4) in widths and round(d_top, 4) in widths


def test_schematic_conical_stage_is_a_trapezoid_not_a_rectangle():
    cyl = _two_stage()
    ax0 = _ax(); draw_booster(ax0, cyl)
    n_rect0 = sum(isinstance(p, Rectangle) for p in ax0.patches)
    con = _two_stage(); con.stage2.conical = True; con.stage2.top_diameter_m = 0.5
    ax1 = _ax(); draw_booster(ax1, con)
    n_rect1 = sum(isinstance(p, Rectangle) for p in ax1.patches)
    # the conical stage moved from a Rectangle to a Polygon
    assert n_rect1 == n_rect0 - 1
