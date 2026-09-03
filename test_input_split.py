"""The four inputs stay apart.

Thrusty has four inputs: two hardware files (booster, reentry object) and two
non-hardware files (flight plan, reentry plan).  The rule, held here as a
schema test over the shipped files and the serialisers:

  * a hardware file carries no plan key, and a plan file carries no hardware
    key -- nothing is stored twice;
  * timings (when something jettisons, deploys or ignites) are plan data, even
    when the thing that jettisons is hardware;
  * the ONLY link between a booster and a reentry object is the booster's
    ``body_reenters`` flag.  Neither the object file nor the reentry plan
    stores a separation choice; the run derives it from the booster.

Pure JSON plus the serialisers -- no GUI needed.
"""

import dataclasses as dc
import glob
import json

import pytest

import booster_models as mm
from booster_models import (BoosterParams, ROParams, booster_to_dict,
                            booster_from_dict, ro_to_dict, ro_from_dict,
                            apply_flight_plan, extract_flight_plan,
                            apply_reentry_plan, extract_reentry_plan,
                            compose_loadout, effective_ro,
                            run_separation_mode, bind_ro_separation,
                            get_booster)

FLIGHT_PLAN_KEYS = set(mm._FLIGHT_PLAN_TOP_KEYS) | set(mm._FLIGHT_PLAN_STAGE_KEYS)
REENTRY_PLAN_KEYS = set(mm._REENTRY_PLAN_KEYS)
LINK_KEY = 'body_reenters'
DERIVED = {'separation_mode'}
META = {'name', 'booster', 'base_plan', 'source', 'notes', 'stages', 'stage2'}

BOOSTER_HARDWARE = ({f.name for f in dc.fields(BoosterParams)}
                    - FLIGHT_PLAN_KEYS - META - {'ro'})
RO_HARDWARE = {f.name for f in dc.fields(ROParams)} - REENTRY_PLAN_KEYS - META - DERIVED

BOOSTER_FILES = sorted(glob.glob('booster_library/*.booster.json') + glob.glob('*.booster.json'))
RO_FILES = sorted(glob.glob('ro_library/*.ro.json') + glob.glob('*.ro.json'))
FLIGHT_PLAN_FILES = sorted(glob.glob('flight_plans/*.flightplan.json'))
REENTRY_PLAN_FILES = sorted(glob.glob('reentry_plans/*.reentryplan.json'))


def _stages(d):
    node = d
    while node is not None:
        yield node
        node = node.get('stage2')


# ── the key sets themselves ─────────────────────────────────────────────────

def test_timings_are_flight_plan_keys():
    """When an adapter drops and when the core lights are flight decisions."""
    assert 'interstage_jettison_s' in mm._FLIGHT_PLAN_STAGE_KEYS
    assert 'grid_fin_deploy_schedule' in mm._FLIGHT_PLAN_STAGE_KEYS
    assert 'booster_core_delay_s' in mm._FLIGHT_PLAN_TOP_KEYS
    assert 'booster_jettison_s' in mm._FLIGHT_PLAN_TOP_KEYS
    assert 'shroud_jettison_alt_km' in mm._FLIGHT_PLAN_TOP_KEYS


def test_separation_is_not_a_plan_key_and_link_is_booster_hardware():
    assert 'separation_mode' not in REENTRY_PLAN_KEYS
    assert LINK_KEY in BOOSTER_HARDWARE
    assert LINK_KEY not in RO_HARDWARE and LINK_KEY not in REENTRY_PLAN_KEYS


def test_no_key_is_both_hardware_and_plan():
    assert not (BOOSTER_HARDWARE & FLIGHT_PLAN_KEYS)
    assert not (RO_HARDWARE & REENTRY_PLAN_KEYS)


# ── the shipped files ───────────────────────────────────────────────────────

@pytest.mark.parametrize('path', BOOSTER_FILES)
def test_booster_file_is_hardware_only(path):
    d = json.load(open(path))
    for i, st in enumerate(_stages(d)):
        leaked = set(st) & (FLIGHT_PLAN_KEYS | DERIVED)
        assert not leaked, f"{path} stage {i + 1} stores plan/derived keys {sorted(leaked)}"


@pytest.mark.parametrize('path', RO_FILES)
def test_ro_file_is_hardware_only(path):
    d = json.load(open(path))
    leaked = set(d) & (REENTRY_PLAN_KEYS | DERIVED | {LINK_KEY})
    assert not leaked, f"{path} stores plan/derived/link keys {sorted(leaked)}"


@pytest.mark.parametrize('path', FLIGHT_PLAN_FILES)
def test_flight_plan_file_has_no_hardware(path):
    d = json.load(open(path))
    leaked = set(d) & (BOOSTER_HARDWARE | {LINK_KEY})
    assert not leaked, f"{path} stores hardware keys {sorted(leaked)}"
    for i, st in enumerate(d.get('stages', []) or []):
        leaked = set(st) & BOOSTER_HARDWARE
        assert not leaked, f"{path} stage {i + 1} stores hardware keys {sorted(leaked)}"


@pytest.mark.parametrize('path', REENTRY_PLAN_FILES)
def test_reentry_plan_file_has_no_hardware_or_separation(path):
    d = json.load(open(path))
    leaked = set(d) & (RO_HARDWARE | DERIVED | {LINK_KEY})
    assert not leaked, f"{path} stores hardware/derived/link keys {sorted(leaked)}"


# ── the serialisers hold the line for user-saved files too ──────────────────

def test_booster_serialiser_omits_plan_keys_at_every_level():
    p = get_booster("Scud-B (R-17)")
    p.interstage_jettison_s = 12.0
    p.booster_core_delay_s = 3.0
    p.grid_fin_deploy_schedule = [[3, 4]]
    d = booster_to_dict(p, include_flight_plan=False)
    for st in _stages(d):
        assert not (set(st) & FLIGHT_PLAN_KEYS)
    # ...and the full round-trip still carries them for in-memory use.
    q = booster_from_dict(booster_to_dict(p))
    assert q.interstage_jettison_s == 12.0 and q.booster_core_delay_s == 3.0


def test_ro_serialiser_never_writes_separation_mode():
    ro = ROParams(name="x", mass_kg=1.0, beta_kg_m2=1.0, shape="cone",
                  diameter_m=0.5, length_m=1.0, separation_mode="body")
    assert 'separation_mode' not in ro_to_dict(ro)
    assert 'separation_mode' not in ro_to_dict(ro, include_reentry_plan=False)
    assert not (set(ro_to_dict(ro, include_reentry_plan=False)) & REENTRY_PLAN_KEYS)


def test_reentry_plan_extract_omits_separation_and_apply_ignores_legacy():
    ro = ROParams(name="x", mass_kg=1.0, beta_kg_m2=1.0, shape="cone",
                  diameter_m=0.5, length_m=1.0)
    assert 'separation_mode' not in extract_reentry_plan(ro)
    q = apply_reentry_plan(ro, {'separation_mode': 'body'})   # legacy plan file
    assert q.separation_mode == 'separating_ro'


def test_flight_plan_carries_the_timings():
    p = get_booster("Scud-B (R-17)")
    fp = extract_flight_plan(p)
    assert 'booster_core_delay_s' in fp
    assert all('interstage_jettison_s' in st for st in fp['stages'])
    q = apply_flight_plan(p, {'booster_core_delay_s': 2.5,
                              'stages': [{'interstage_jettison_s': 7.0}]})
    assert q.booster_core_delay_s == 2.5 and q.interstage_jettison_s == 7.0
    assert p.booster_core_delay_s == 0.0          # source untouched


# ── the one link, and only that link ────────────────────────────────────────

def _pair(body_reenters, ro_mode):
    p = get_booster("Scud-B (R-17)")
    p.body_reenters = body_reenters
    ro = ROParams(name="w", mass_kg=800.0, beta_kg_m2=5000.0, shape="cone",
                  diameter_m=0.88, length_m=2.0, separation_mode=ro_mode)
    p = compose_loadout(p, ro, 1)
    p.ro = ro
    return p


@pytest.mark.parametrize('ro_mode', ['separating_ro', 'body'])
def test_booster_flag_decides_regardless_of_object(ro_mode):
    assert run_separation_mode(_pair(True, ro_mode)) == 'body'
    assert run_separation_mode(_pair(False, ro_mode)) == 'separating_ro'
    assert effective_ro(_pair(True, ro_mode)).separation_mode == 'body'
    assert effective_ro(_pair(False, ro_mode)).separation_mode == 'separating_ro'


def test_bind_stamps_the_object_without_mutating_the_caller():
    p = _pair(True, 'separating_ro')
    q = bind_ro_separation(p)
    assert q.ro.separation_mode == 'body'
    assert p.ro.separation_mode == 'separating_ro'      # caller untouched
    assert bind_ro_separation(q) is q                    # already bound: no copy


def test_body_reenters_forces_single_object_loadout():
    p = get_booster("Scud-B (R-17)")
    p.body_reenters = True
    ro = ROParams(name="w", mass_kg=800.0, beta_kg_m2=5000.0, shape="cone",
                  diameter_m=0.88, length_m=2.0)
    assert compose_loadout(p, ro, 3).num_ros == 1
