"""Power-on base-drag correction (base bleed).

While a stage's engine fires, the exhaust plume fills the nozzle exit and
suppresses the base drag over that area, so the build-up must not charge the
FULL power-off base drag during the burn.  The correction scales the base-drag
term by ``base_bleed_ratio = 1 − A_exit/A_base`` (floored at 0) while powered,
and is a no-op (ratio 1.0) when the stage carries no nozzle area or is coasting
— so vehicles without nozzle data and all reentry evaluations are unchanged.

Only the decomposed nose-shape drag path (`_cd_nose_shape`) separates base
drag; the Forden mach-table path bakes it in and is untouched.
"""

import math
import types

import pytest

from booster_models import (get_booster, load_booster_library, ROParams,
                            compose_loadout, base_bleed_ratio,
                            _total_nozzle_exit_area, _cd_nose_shape)
import booster_models as mm
from trajectory import integrate_trajectory

load_booster_library()


def _stage(d, exit_tot=0.0, each=0.0, n=1):
    return types.SimpleNamespace(diameter_m=d, nozzle_exit_area_m2=exit_tot,
                                 nozzle_area_each_m2=each, n_nozzles=n)


# ── ratio math ──────────────────────────────────────────────────────────────

def test_ratio_is_one_minus_exit_over_base():
    d, A_exit = 0.88, 0.30
    A_base = math.pi * (d / 2.0) ** 2
    assert base_bleed_ratio(_stage(d, A_exit), d) == pytest.approx(
        1.0 - A_exit / A_base)


def test_ratio_is_one_without_nozzle_data():
    """No nozzle area → full power-off base drag (unchanged)."""
    assert base_bleed_ratio(_stage(0.88, 0.0), 0.88) == 1.0


def test_ratio_is_one_without_diameter():
    assert base_bleed_ratio(_stage(0.0, 0.30), 0.0) == 1.0


def test_ratio_floored_at_zero_for_oversize_exit():
    """A_exit ≥ A_base cannot make base drag negative."""
    d = 0.5
    A_base = math.pi * (d / 2.0) ** 2
    assert base_bleed_ratio(_stage(d, 2.0 * A_base), d) == 0.0


def test_total_area_prefers_authoritative_then_per_nozzle():
    assert _total_nozzle_exit_area(_stage(1.0, 0.4)) == pytest.approx(0.4)
    assert _total_nozzle_exit_area(_stage(1.0, 0.0, each=0.1, n=4)) == pytest.approx(0.4)


# ── Cd term: base bleed only touches the base component ─────────────────────

def test_cd_drops_by_base_bleed_and_only_by_that():
    """The power-on Cd is lower by exactly the base-drag reduction — wave and
    friction are untouched."""
    from booster_models import _cd_base
    M, ld = 1.2, 3.0
    ratio = base_bleed_ratio(_stage(0.88, 0.30), 0.88)
    full = _cd_nose_shape('von_karman', ld, M, base_area_ratio=1.0)
    bled = _cd_nose_shape('von_karman', ld, M, base_area_ratio=ratio)
    expected_drop = _cd_base(M, 1.0) - _cd_base(M, ratio)
    assert (full - bled) == pytest.approx(expected_drop, abs=1e-9)
    assert bled < full


# ── trajectory: isolated effect (thrust held constant) ─────────────────────

def _strypi_body(nozzle=True):
    p = get_booster("Scud-B (R-17)")
    p.body_reenters = True          # the booster owns the separation link
    p.diameter_m = 0.88
    p.length_m = 11.0
    if nozzle:
        p.nozzle_exit_area_m2 = 0.30
    ro = ROParams(name="B", mass_kg=500.0, beta_kg_m2=6000.0, shape="von_karman",
                  diameter_m=0.88, length_m=11.0, separation_mode="body",
                  body_nose_length_m=2.0)
    p2 = compose_loadout(p, ro, 1)
    p2.ro = ro
    return p2


def _range(p):
    return integrate_trajectory(p, 39.12, 125.67, 90.0, burnout_angle_deg=45.0,
                                max_time_s=3600.0)["range_km"]


def test_base_bleed_extends_range_thrust_held_constant(monkeypatch):
    """With nozzle area present in BOTH runs (so the thrust pressure-correction
    is identical), turning base bleed OFF vs ON isolates the drag effect: less
    boost drag → more burnout energy → longer range."""
    r_on = _range(_strypi_body(nozzle=True))
    monkeypatch.setattr(mm, "base_bleed_ratio", lambda *a, **k: 1.0)
    r_off = _range(_strypi_body(nozzle=True))
    assert r_on > r_off
    # meaningful (larger than a boattail's <1%), not runaway
    assert 0.005 < (r_on - r_off) / r_off < 0.20


def test_no_nozzle_data_is_unchanged(monkeypatch):
    """A body with no nozzle area flies identically whether or not the base-
    bleed code path exists (ratio is 1.0 either way)."""
    r_real = _range(_strypi_body(nozzle=False))
    monkeypatch.setattr(mm, "base_bleed_ratio", lambda *a, **k: 1.0)
    r_forced = _range(_strypi_body(nozzle=False))
    assert r_real == pytest.approx(r_forced, rel=1e-9)
