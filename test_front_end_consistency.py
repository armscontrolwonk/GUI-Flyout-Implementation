"""DRAWN ≡ FLOWN — the front-end oversight invariant (FRONT_END_DESIGN.md §3).

The schematic is how a human user oversees the model.  If what is drawn differs
from what is flown, the human cannot exercise authority over the code.  So the
geometry rendered by booster_schematic.draw_booster MUST equal the geometry the
physics consumes through effective_ro / _boost_front_geometry — same overall
length, same body diameter, same nose shape.  These tests pin that.

To make the invariant machine-checkable, draw_booster reports the front end it
actually drew in its summary dict under 'front_end':
    {'kind', 'shape', 'nose_length_m', 'body_diameter_m'}

The KN-23 case that motivated the redesign: a single ⌀1.1 × 6.7 m stage carrying
a non-separating ⌀1.1 × 2.0 m Von Kármán body.  Before the fix the schematic drew
an 8.46 m stack (6.7 m stage + a fabricated 1.6×⌀ cone, as a plain cone) while
the physics flew a ⌀1.1 × 6.7 m body — three different front ends.
"""

import glob
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from booster_models import (get_booster, load_booster_library, ROParams,
                            booster_from_dict, compose_loadout, effective_ro)
import booster_schematic as bsch

load_booster_library()


def _ax():
    return plt.figure().add_subplot(111)


def _airframe_len(p):
    """Sum of stage lengths + declared interstages — the physical airframe."""
    from booster_schematic import stage_chain, _stage_top_diameter  # noqa
    total = 0.0
    stages = stage_chain(p)
    for i, s in enumerate(stages):
        total += float(getattr(s, "length_m", 0.0) or 0.0)
        if getattr(s, "has_interstage", False) \
                and (getattr(s, "interstage_length_m", 0.0) or 0) > 0:
            total += float(s.interstage_length_m)
    return total


def _kn23():
    """A body-mode KN-23 fixture: one ⌀1.1×6.7 m stage, ⌀1.1×2.0 m Von Kármán
    non-separating body."""
    p = get_booster("Scud-B (R-17)")
    p.body_reenters = True
    p.diameter_m = 1.1
    p.length_m = 6.7
    if p.stage2 is None:                       # ensure single stage
        pass
    ro = ROParams(name="KN23 front end", mass_kg=500.0, beta_kg_m2=27395.0,
                  shape="karman", diameter_m=1.1, length_m=2.0,
                  nose_radius_m=0.05, separation_mode="body",
                  body_nose_length_m=2.0)
    p2 = compose_loadout(p, ro, 1)
    p2.ro = ro
    return p2


# ── the invariant ───────────────────────────────────────────────────────────

def test_body_mode_total_height_is_the_airframe_not_airframe_plus_nose():
    """A non-separating body's drawn height is the airframe length (the nose is
    carved from it), NOT the airframe plus a stacked cone."""
    p = _kn23()
    info = bsch.draw_booster(_ax(), p, title="KN-23")
    assert info["total_height_m"] == _airframe_len(p)          # 6.7, not 8.46


def test_body_mode_draws_the_declared_nose_shape():
    """The nose drawn is the RO's declared shape (Von Kármán), not a cone."""
    p = _kn23()
    info = bsch.draw_booster(_ax(), p)
    fe = info["front_end"]
    assert fe["shape"] == "karman"
    assert fe["kind"] == "body_nose"


def test_body_mode_diameter_matches_flown():
    """Drawn body diameter equals the flown (effective_ro) diameter."""
    p = _kn23()
    info = bsch.draw_booster(_ax(), p)
    assert abs(info["front_end"]["body_diameter_m"]
               - effective_ro(p).diameter_m) < 1e-9


def test_body_mode_nose_never_exceeds_the_body():
    """A subtractive nose can never be longer than the airframe it is carved
    from (the class of the fabricated-cone bug)."""
    p = _kn23()
    info = bsch.draw_booster(_ax(), p)
    assert info["front_end"]["nose_length_m"] <= _airframe_len(p) + 1e-9


def test_no_fabricated_nose_flag_when_a_reentry_object_is_present():
    """With a real RO to draw from, the schematic must not fall back to the
    1.6×⌀ 'nose length unset' fabrication."""
    p = _kn23()
    info = bsch.draw_booster(_ax(), p)
    assert not any("nose length unset" in f for f in info["flags"])


def test_no_phantom_fit_warning_for_a_body():
    """A non-separating body is not contained in anything, so there is no
    'payload does not fit' verdict to raise."""
    p = _kn23()
    info = bsch.draw_booster(_ax(), p)
    assert not any("does not fit" in f.lower() or "too long" in f.lower()
                   for f in info["flags"])


# ── every library booster still reports a coherent front end ────────────────

def test_library_boosters_report_front_end():
    for f in glob.glob("booster_library/*.booster.json"):
        b = booster_from_dict(json.load(open(f)))
        info = bsch.draw_booster(_ax(), b)
        assert "front_end" in info, f
        assert info["total_height_m"] > 0, f
        assert np.isfinite(info["total_height_m"]), f
