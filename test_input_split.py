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
RUN_SCRATCH = set(mm._RUN_LOADOUT_KEYS) | {'ro_mass_kg', 'body_payload_kg', 'ro_separates', 'rv_separates'}
META = {'name', 'booster', 'base_plan', 'source', 'notes', 'stages', 'stage2',
        'reentry_object'}          # a flight plan names the object it flies

BOOSTER_HARDWARE = ({f.name for f in dc.fields(BoosterParams)}
                    - FLIGHT_PLAN_KEYS - META - RUN_SCRATCH - {'ro'})
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
        leaked = set(st) & (FLIGHT_PLAN_KEYS | DERIVED | RUN_SCRATCH)
        assert not leaked, f"{path} stage {i + 1} stores plan/derived/loadout keys {sorted(leaked)}"


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


# ── g-limit is hardware; the plan commands at or below it ───────────────────

def test_beta_S_is_hardware_and_pullup_g_is_clamped_to_the_limit():
    assert 'glider_beta_entry_kg_m2' not in REENTRY_PLAN_KEYS
    assert 'glider_beta_entry_kg_m2' in RO_HARDWARE
    assert 'pullup_g_limit' in RO_HARDWARE and 'pullup_g_limit' not in REENTRY_PLAN_KEYS
    ro = ROParams(name="x", mass_kg=1.0, beta_kg_m2=1.0, shape="cone",
                  diameter_m=0.5, length_m=1.0, pullup_g_limit=8.0,
                  glider_beta_entry_kg_m2=7.0)
    assert apply_reentry_plan(ro, {'glider_pullup_g_max': 30.0}).glider_pullup_g_max == 8.0
    assert apply_reentry_plan(ro, {'glider_pullup_g_max': 5.0}).glider_pullup_g_max == 5.0
    assert apply_reentry_plan(ro, {}).glider_pullup_g_max <= 8.0
    # an UNSET limit (0, the default) is unlimited: the plan's command stands
    free = ROParams(name="y", mass_kg=1.0, beta_kg_m2=1.0, shape="cone",
                    diameter_m=0.5, length_m=1.0)
    assert free.pullup_g_limit == 0.0
    assert apply_reentry_plan(free, {'glider_pullup_g_max': 30.0}).glider_pullup_g_max == 30.0
    # a legacy plan carrying beta_S is ignored; the object's value stands
    assert apply_reentry_plan(ro, {'glider_beta_entry_kg_m2': 99.0}).glider_beta_entry_kg_m2 == 7.0
    # and both survive the hardware-only serialiser
    d = ro_to_dict(ro, include_reentry_plan=False)
    assert d['pullup_g_limit'] == 8.0 and d['glider_beta_entry_kg_m2'] == 7.0


# ── booster files are stack-only; the object owns its mass ──────────────────

def _legacy(payload, baked):
    """A pre-2026-09 booster dict: design payload inside every stage's launch
    mass and, when 'baked' (ro_separates False, Scud class), inside the last
    stage's burnout mass too."""
    d = booster_to_dict(get_booster("No-dong"))          # a separating stack
    node = d
    while node is not None:
        node['mass_initial'] += payload
        last = node
        node = node.get('stage2')
    if baked:
        last['mass_final'] += payload
    d['payload_kg'] = payload
    d['ro_separates'] = not baked
    return d


@pytest.mark.parametrize('baked', [False, True])
def test_legacy_payload_is_normalised_on_load_and_reproduced_by_composition(baked):
    clean = get_booster("No-dong")
    p = booster_from_dict(_legacy(1000.0, baked))
    # loaded chain is stack-only: same masses as the shipped (clean) file
    assert p.payload_kg == 0.0
    assert p.mass_initial == pytest.approx(clean.mass_initial)
    assert p.mass_final == pytest.approx(clean.mass_final)
    # composing the object that carries the mass reproduces the legacy launch mass
    ro = ROParams(name="w", mass_kg=1000.0, beta_kg_m2=5000.0, shape="cone",
                  diameter_m=1.32, length_m=1.0)
    c = compose_loadout(p, ro, 1)
    assert c.mass_initial == pytest.approx(clean.mass_initial + 1000.0)
    assert c.payload_kg == 1000.0


def test_hardware_only_booster_serialiser_omits_the_loadout_record():
    p = compose_loadout(get_booster("No-dong"),                # separating stack
                        ROParams(name="w", mass_kg=800.0, beta_kg_m2=5000.0,
                                 shape="cone", diameter_m=0.84, length_m=1.0), 2)
    assert p.payload_kg == 1600.0 and p.num_ros == 2
    d = booster_to_dict(p, include_flight_plan=False)
    assert not (set(d) & RUN_SCRATCH)
    # the full (internal) form still round-trips the record
    q = booster_from_dict(booster_to_dict(p))
    assert q.num_ros == 2


@pytest.mark.parametrize('path', FLIGHT_PLAN_FILES)
def test_plan_named_reentry_object_resolves(path):
    """A flight plan may name the object it flies; when it does, the object
    must ship (every default run that used to fly a stored payload now flies
    a real object)."""
    name = json.load(open(path)).get('reentry_object', '')
    if not name:
        return
    shipped = {json.load(open(f))['name'] for f in RO_FILES}
    assert name in shipped, f"{path} names '{name}', which is not in ro_library/"


def test_non_separating_shipped_boosters_carry_their_warhead_as_object_payload():
    """Scud-B and Al Hussein do not separate: the booster is body_reenters and
    the flight plan names a front-end object whose payload_kg is the warhead."""
    for bname in ("Scud-B (R-17)", "Al Hussein"):
        b = get_booster(bname)
        assert b.body_reenters is True
        oname = mm.load_flight_plan(bname)['reentry_object']
        ro = next(ro_from_dict(json.load(open(f))) for f in RO_FILES
                  if json.load(open(f))['name'] == oname)
        assert ro.payload_kg > 0 and ro.beta_kg_m2 == 0.0     # warhead rides; beta derived
        c = compose_loadout(b, ro, 1)
        assert c.mass_initial == pytest.approx(b.mass_initial + ro.payload_kg)
        assert c.mass_final == pytest.approx(b.mass_final + ro.payload_kg)


def test_headless_get_booster_attaches_the_plan_named_object_and_flies_its_mass():
    """A flight plan that names its object is honoured headless too: get_booster
    attaches it, and integrate_trajectory composes it onto the stack-only
    chain, so a 'bare' run carries the same front end the GUI default flies."""
    from trajectory import integrate_trajectory
    p = get_booster("No-dong")
    assert p.ro is not None and p.ro.name == "No-dong warhead"
    assert p.payload_kg == 0.0                       # stack-only until composed
    r = integrate_trajectory(p, 39.12, 125.67, 90.0, max_time_s=1200.0)
    assert r["range_km"] > 0
    # a booster whose plan names no object stays bare
    assert get_booster("Shahab-3").ro is None
