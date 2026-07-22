"""Screening-envelope thresholds (thresholds.py) — the user-adjustable
benchmark numbers behind the survivability screen.

Two disciplines are structural and are pinned here so they can't silently
rot: (1) the shipped defaults in REGISTRY MUST equal the live module
constants they mirror (a drift guard — otherwise the dialog would show one
number while the model used another), and (2) a user edit lives only in the
overlay and always restores, while any modified benchmark self-discloses.
"""

import json
import os

import heating
import survivability_report as sr
import thresholds


def _reset():
    thresholds.reset()
    thresholds.apply()


# ── Drift guard: REGISTRY defaults == the live module constants ──────────────
def test_registry_defaults_match_live_constants():
    d = thresholds.defaults()
    assert d["uhtc_dwell_floor_s"] == heating.TPS_MATERIALS["uhtc"]["oxidation_dwell_s"]
    assert d["acreage_flux_fraction"] == heating.BODY_FLUX_FRACTION
    assert (d["windward_alpha_lo"], d["windward_alpha_hi"]) == heating._WINDWARD_ALPHA_BAND
    assert d["graphite_load_floor_MJ_m2"] == \
        heating.TPS_MATERIALS["carbon_carbon"]["demonstrated_load_MJ_m2"]
    assert d["pica_load_floor_MJ_m2"] == \
        heating.TPS_MATERIALS["pica"]["demonstrated_load_MJ_m2"]
    assert d["marv_g_operational"] == sr._MARV_G_OPERATIONAL
    assert d["marv_g_demonstrated"] == sr._MARV_G_DEMONSTRATED


def test_every_row_is_documented():
    for e in thresholds.REGISTRY:
        for field in ("key", "group", "label", "units", "default", "source"):
            assert e.get(field) not in (None, ""), (e["key"], field)


# ── Overrides: set / clamp / clear-at-default ────────────────────────────────
def test_set_override_and_current():
    _reset()
    assert not thresholds.is_modified()
    thresholds.set_override("marv_g_demonstrated", 150.0)
    assert thresholds.current()["marv_g_demonstrated"] == 150.0
    assert thresholds.is_modified()
    _reset()


def test_setting_back_to_default_clears_override():
    _reset()
    e = thresholds._BY_KEY["acreage_flux_fraction"]
    thresholds.set_override("acreage_flux_fraction", 0.2)
    assert thresholds.is_modified()
    thresholds.set_override("acreage_flux_fraction", e["default"])
    assert not thresholds.is_modified()          # equals default => cleared
    _reset()


def test_override_clamps_to_bounds():
    _reset()
    e = thresholds._BY_KEY["marv_g_operational"]
    thresholds.set_override("marv_g_operational", 1e9)
    assert thresholds.current()["marv_g_operational"] == e["hi"]
    thresholds.set_override("marv_g_operational", -5.0)
    assert thresholds.current()["marv_g_operational"] == e["lo"]
    _reset()


# ── modified(): the self-disclosure payload ──────────────────────────────────
def test_modified_lists_changed_rows_with_provenance():
    _reset()
    assert thresholds.modified() == []
    thresholds.set_override("uhtc_dwell_floor_s", 600.0)
    mods = thresholds.modified()
    assert len(mods) == 1
    m = mods[0]
    assert m["key"] == "uhtc_dwell_floor_s"
    assert m["value"] == 600.0
    assert m["default"] == 300.0
    assert m["source"]          # the default's citation of record travels with it
    _reset()


# ── apply(): drives the live models, including the two kwarg-default paths ────
def test_apply_drives_live_modules():
    _reset()
    thresholds.set_override("acreage_flux_fraction", 0.20)
    thresholds.set_override("windward_alpha_hi", 25.0)
    thresholds.set_override("marv_g_demonstrated", 150.0)
    thresholds.apply()
    assert heating.BODY_FLUX_FRACTION == 0.20
    assert heating._WINDWARD_ALPHA_BAND == (5.0, 25.0)
    assert sr._MARV_G_DEMONSTRATED == 150.0
    _reset()
    assert heating.BODY_FLUX_FRACTION == 0.13   # reset+apply restores the default


def test_windward_kwarg_default_follows_module_attr():
    # The bug this guards: alpha_band_deg / body_flux_fraction were bound at
    # def-time, so apply() couldn't reach them.  They now resolve at call time.
    import numpy as np
    _reset()
    t = np.linspace(0, 10, 11)
    rho = np.full(11, 1e-3)
    V = np.full(11, 6000.0)
    alt = np.full(11, 40000.0)
    rng = np.linspace(0, 100, 11)
    base = heating.windward_flank_flux(
        t, rho, V, alt, rng, body_radius_m=0.1, flank_half_angle_deg=8.0,
        alpha_op_deg=12.0, body_material="silica_tile")
    thresholds.set_override("acreage_flux_fraction", 0.26)   # 2x default
    thresholds.apply()
    hotter = heating.windward_flank_flux(
        t, rho, V, alt, rng, body_radius_m=0.1, flank_half_angle_deg=8.0,
        alpha_op_deg=12.0, body_material="silica_tile")
    assert hotter["q_flank0_peak_MW_m2"] > base["q_flank0_peak_MW_m2"]
    _reset()


# ── Persistence: overlay round-trip; file deleted at defaults ────────────────
def test_save_load_round_trip(tmp_path):
    _reset()
    path = str(tmp_path / "ov.json")
    thresholds.set_override("pica_load_floor_MJ_m2", 300.0)
    thresholds.save(path)
    assert os.path.exists(path)
    with open(path) as f:
        assert json.load(f) == {"pica_load_floor_MJ_m2": 300.0}
    thresholds.reset()
    assert not thresholds.is_modified()
    thresholds.load(path)
    assert thresholds.current()["pica_load_floor_MJ_m2"] == 300.0
    _reset()


def test_save_at_defaults_removes_overlay(tmp_path):
    _reset()
    path = str(tmp_path / "ov.json")
    thresholds.set_override("pica_load_floor_MJ_m2", 300.0)
    thresholds.save(path)
    assert os.path.exists(path)
    thresholds.reset()
    thresholds.save(path)          # back at defaults => overlay deleted
    assert not os.path.exists(path)
    _reset()


def _fly_glider():
    """A C-HGB glide run — reenters and produces a Form B/C heating report."""
    import copy
    import json
    from booster_models import (get_booster, ro_from_dict,
                                 apply_reentry_plan, load_reentry_plan)
    from trajectory import integrate_trajectory
    chgb = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    _rp = load_reentry_plan("C-HGB")
    if _rp is not None:
        chgb = apply_reentry_plan(chgb, _rp)
    p = get_booster("AUR+HGB")
    p.ro = copy.deepcopy(chgb)
    return integrate_trajectory(p, 0.0, 0.0, 90.0, max_time_s=6000.0,
                                dt_output=2.0)


# ── Report disclosure: a modified benchmark stamps the headline + body ───────
def test_report_discloses_modified_benchmark():
    _reset()
    r = _fly_glider()
    clean = sr.build_report(r)
    assert not clean["headline"].rstrip().endswith("*")   # no asterisk at defaults
    assert "Modified benchmarks" not in clean["body"]

    thresholds.set_override("marv_g_demonstrated", 150.0)
    dirty = sr.build_report(r)
    assert dirty["headline"].rstrip().endswith("*")
    assert "Modified benchmarks" in dirty["body"]
    assert "150" in dirty["body"] and "default 100" in dirty["body"]
    _reset()
