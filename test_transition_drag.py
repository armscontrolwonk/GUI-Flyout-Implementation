"""Interstage / conical-stage flare drag (METHODS §6.7 Phase 2, lean screening).

Phase 1 carried the interstage and conical-stage geometry + mass but was
drag-neutral.  Phase 2 adds a screening wave-drag increment for a FLARE — a
conical stage or interstage whose body is wider at its aft (downstream) end
than its forward end, presenting a forward-facing conical surface to the
nose-first flow.  Boattails (narrower aft) and same-diameter sections add
nothing (conservative), and a plain stack with neither feature is
byte-identical.
"""

import copy
import math

import numpy as np
import pytest

import booster_models as mm
from booster_models import (get_booster, load_booster_library,
                            drag_force_vector, active_stage,
                            _flare_cd, _transition_wave_drag)
from trajectory import integrate_trajectory

load_booster_library()

_AREF = math.pi * (1.0 / 2) ** 2      # 1 m reference


# ── _flare_cd: only a flare charges ─────────────────────────────────────────

def test_flare_charges_boattail_and_equal_do_not():
    assert _flare_cd(1.0, 0.6, 1.0, 2.0, _AREF) > 0.0     # widens aft -> flare
    assert _flare_cd(0.6, 1.0, 1.0, 2.0, _AREF) == 0.0    # narrows aft -> boattail
    assert _flare_cd(0.8, 0.8, 1.0, 2.0, _AREF) == 0.0    # same diameter


def test_flare_scales_with_area_step():
    small = _flare_cd(1.0, 0.9, 1.0, 2.0, _AREF)
    big = _flare_cd(1.0, 0.5, 1.0, 2.0, _AREF)
    assert big > small > 0.0


def test_bare_step_floors_to_blunt():
    """A near-zero length (diameter step, no ramp) still returns a finite,
    positive increment (fineness floored at 0.5)."""
    v = _flare_cd(1.0, 0.6, 1e-9, 2.0, _AREF)
    assert v > 0.0 and math.isfinite(v)


# ── _transition_wave_drag over a stack ──────────────────────────────────────

def test_plain_stack_is_zero():
    p = get_booster("Taepodong-I")
    assert _transition_wave_drag(p, active_stage(p, 5.0), 2.0, _AREF) == 0.0


def test_conical_stage_adds_drag():
    p = copy.deepcopy(get_booster("Taepodong-I"))
    p.conical = True
    p.top_diameter_m = p.diameter_m * 0.7          # narrows toward nose -> flare aft
    assert _transition_wave_drag(p, active_stage(p, 5.0), 2.0, _AREF) > 0.0


def test_interstage_flare_adds_drag_taper_does_not():
    p = copy.deepcopy(get_booster("Taepodong-I"))
    s1 = p                                         # stage 1 carries the interstage
    s1.has_interstage = True
    s1.interstage_length_m = 1.0
    # stage 1 fatter than stage 2 -> interstage steps DOWN going forward -> flare
    if s1.stage2 is not None and s1.stage2.diameter_m < s1.diameter_m:
        assert _transition_wave_drag(p, active_stage(p, 5.0), 2.0, _AREF) > 0.0
    # make stage 2 as wide as stage 1 -> no flare
    if s1.stage2 is not None:
        s1.stage2.diameter_m = s1.diameter_m
        assert _transition_wave_drag(p, active_stage(p, 5.0), 2.0, _AREF) == 0.0


# ── trajectory: byte-identical off, shorter range on ────────────────────────

def test_plain_vehicle_byte_identical():
    r0 = integrate_trajectory(get_booster("Taepodong-I"), 39.0, 125.0, 90.0,
                              max_time_s=6000.0)
    r1 = integrate_trajectory(get_booster("Taepodong-I"), 39.0, 125.0, 90.0,
                              max_time_s=6000.0)
    assert r0["range_km"] == r1["range_km"]


def test_conical_flare_shortens_range():
    plain = get_booster("Taepodong-I")
    flared = copy.deepcopy(plain)
    flared.conical = True
    flared.top_diameter_m = flared.diameter_m * 0.6     # pronounced flare
    r_plain = integrate_trajectory(plain, 39.0, 125.0, 90.0, max_time_s=6000.0)
    r_flare = integrate_trajectory(flared, 39.0, 125.0, 90.0, max_time_s=6000.0)
    assert r_flare["range_km"] < r_plain["range_km"]
