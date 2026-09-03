"""Old files keep loading, exactly as they did.

Thrusty has accepted several file vintages: Forden's `loft` guidance, the
`rv_*` field family that predates the reentry-object rename, nose length as a
diameter ratio, reentry hardware stored on the booster, a design payload baked
into the stage masses, retired glide-law names, and object files from before
the maneuvering capability was split from the plan's intent.

All of that now lives in ONE place per file kind -- `upgrade_booster_dict` and
`upgrade_ro_dict` -- so the constructors read the current schema and nothing
else.  This file is the contract that the move changed no behaviour:
`tests_data/legacy_load_golden.json` records what every shape in
`legacy_corpus.py` loaded to before the refactor, and each shape must still
load to exactly that.  Regenerate the golden file only when a load result is
*deliberately* changed, and say so in the commit.
"""

import glob
import json
import os

import pytest

import booster_models as mm
from booster_models import (booster_from_dict, ro_from_dict, booster_to_dict,
                            ro_to_dict, upgrade_booster_dict, upgrade_ro_dict)
from legacy_corpus import LEGACY_BOOSTERS, LEGACY_ROS

GOLDEN = json.load(open("tests_data/legacy_load_golden.json"))


def _json(obj):
    """Compare in serialised form: the golden file is JSON, so a tuple and the
    list it serialises to are the same stored value."""
    return json.loads(json.dumps(obj))


def _booster_snapshot(p):
    d = booster_to_dict(p)
    d['_ro'] = None if p.ro is None else ro_to_dict(p.ro)
    return _json(d)


def _fresh(raw):
    """A deep copy, so a conversion that mutated its input would be caught."""
    return json.loads(json.dumps(raw))


# ── the contract: every vintage still loads to the same object ──────────────

@pytest.mark.parametrize('key', sorted(LEGACY_BOOSTERS))
def test_legacy_booster_loads_unchanged(key):
    got = _booster_snapshot(booster_from_dict(_fresh(LEGACY_BOOSTERS[key])))
    assert got == GOLDEN['boosters'][key], f"load behaviour changed for {key}"


@pytest.mark.parametrize('key', sorted(LEGACY_ROS))
def test_legacy_ro_loads_unchanged(key):
    got = _json(ro_to_dict(ro_from_dict(_fresh(LEGACY_ROS[key]))))
    assert got == GOLDEN['ros'][key], f"load behaviour changed for {key}"


@pytest.mark.parametrize('path', sorted(glob.glob('booster_library/*.booster.json')
                                        + ['KN-23A.booster.json']))
def test_shipped_booster_file_loads_unchanged(path):
    got = _booster_snapshot(booster_from_dict(json.load(open(path))))
    assert got == GOLDEN['boosters']['file:' + os.path.basename(path)]


@pytest.mark.parametrize('path', sorted(glob.glob('ro_library/*.ro.json')
                                        + ['KN-23A_warhead.ro.json']))
def test_shipped_ro_file_loads_unchanged(path):
    got = _json(ro_to_dict(ro_from_dict(json.load(open(path)))))
    assert got == GOLDEN['ros']['file:' + os.path.basename(path)]


# ── properties the upgraders must hold ──────────────────────────────────────

@pytest.mark.parametrize('key', sorted(LEGACY_BOOSTERS))
def test_booster_upgrade_is_pure_and_idempotent(key):
    raw = _fresh(LEGACY_BOOSTERS[key])
    before = _fresh(raw)
    once = upgrade_booster_dict(raw)
    assert raw == before, "upgrade mutated the caller's dict"
    assert upgrade_booster_dict(once) == once, "upgrade is not idempotent"
    assert once['schema'] == mm.BOOSTER_SCHEMA


@pytest.mark.parametrize('key', sorted(LEGACY_ROS))
def test_ro_upgrade_is_pure_and_idempotent(key):
    raw = _fresh(LEGACY_ROS[key])
    before = _fresh(raw)
    once = upgrade_ro_dict(raw)
    assert raw == before, "upgrade mutated the caller's dict"
    assert upgrade_ro_dict(once) == once, "upgrade is not idempotent"
    assert once['schema'] == mm.RO_SCHEMA


LEGACY_KEYS = {
    'loft_angle_deg', 'num_rvs', 'rv_mass_kg', 'rv_separates', 'ro_separates',
    'nose_ld_ratio', 'shroud_nose_ld_ratio', 'payload_kg', 'rv',
    'ro_beta_kg_m2', 'rv_beta_kg_m2', 'ro_shape', 'rv_shape', 'ro_diameter_m',
    'rv_diameter_m', 'ro_length_m', 'rv_length_m', 'glider_LD',
    'glider_enabled', 'glider_guidance', 'glider_pullup_g_max',
    'glider_terminal_dive', 'glider_terminal_alt_km',
}


@pytest.mark.parametrize('key', sorted(LEGACY_BOOSTERS))
def test_upgraded_booster_carries_no_legacy_key(key):
    """The upgrader's output is current-schema: nothing downstream has to know
    what a legacy key looks like."""
    d = upgrade_booster_dict(_fresh(LEGACY_BOOSTERS[key]))
    node = d
    while node is not None:
        assert not (set(node) & LEGACY_KEYS), sorted(set(node) & LEGACY_KEYS)
        node = node.get('stage2')


def test_upgraded_booster_guidance_vocabulary_is_current():
    for key, raw in LEGACY_BOOSTERS.items():
        d = upgrade_booster_dict(_fresh(raw))
        node = d
        while node is not None:
            assert node.get('guidance', 'pitch_program') not in ('loft', 'gravity_turn')
            node = node.get('stage2')


def test_upgraded_ro_vocabulary_is_current():
    for key, raw in LEGACY_ROS.items():
        d = upgrade_ro_dict(_fresh(raw))
        assert d['separation_mode'] in ('separating_ro', 'body')
        assert d['glider_guidance'] not in ('constant_bank', 'azimuth_command',
                                            'skip_to_equilibrium')
        assert d['body_form'] in mm.BODY_FORMS
        assert isinstance(d['maneuvering'], bool)


def test_upgrade_rejects_a_non_dict():
    for fn in (upgrade_booster_dict, upgrade_ro_dict):
        with pytest.raises(TypeError):
            fn(["not", "a", "dict"])
