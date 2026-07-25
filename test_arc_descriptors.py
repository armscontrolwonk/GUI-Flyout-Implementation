"""Arc descriptors — the structural retirement of Form C.

The report used to key three "Forms" off two facts.  Form A/B split on the
one thing that genuinely changes the judgement model (ballistic accuracy
ladder vs glider stopwatch), which survives as classify().  Form C did not:
it was triggered by a commanded TERMINAL DIVE but labelled "maneuvering",
and it gated three unrelated blocks behind that one flag.  Two consequences,
both wrong, both pinned here:

  * SWERVE — a lifting body that pulled -10° AoA and was rated to 10 g — read
    as a plain glider, so it was denied the windward-flank heating block (the
    off-nose acreage heat a vehicle at AoA actually carries) AND the
    maneuver-load anchors, the two blocks most relevant to it.
  * A vehicle with a dive altitude and an EMPTY bank schedule was announced
    in the headline as "maneuvering (MaRV)" — an assertion about behaviour
    the plan did not carry.

Each block now hangs on its own trigger and each headline descriptor is
earned by its own fact.
"""

import copy
import json

import survivability_report as sr
from test_report_lead import _fly_ballistic
from test_thresholds import _fly_glider

_CACHE = {}


def _glider():
    if "g" not in _CACHE:
        _CACHE["g"] = _fly_glider()
    return _CACHE["g"]


def _swerve():
    """SWERVE on STARS-1 — the vehicle the retired Form C mis-sorted."""
    if "s" not in _CACHE:
        from booster_models import (get_booster, ro_from_dict,
                                    apply_reentry_plan, load_reentry_plan)
        from trajectory import integrate_trajectory
        ro = ro_from_dict(json.load(open("ro_library/SWERVE.ro.json")))
        rp = load_reentry_plan("SWERVE")
        if rp is not None:
            ro = apply_reentry_plan(ro, rp)
        p = get_booster("STARS-1")
        p.ro = copy.deepcopy(ro)
        _CACHE["s"] = integrate_trajectory(p, 0.0, 0.0, 90.0,
                                           max_time_s=6000.0, dt_output=2.0)
    return _CACHE["s"]


def _patched(result, **profile):
    """A copy of a flown result with the reentry-plan profile overridden."""
    r = copy.deepcopy(result)
    r["heating_arc"]["profile"].update(profile)
    return r


# ── classify(): the one structural fork that survives ───────────────────────
def test_classify_is_the_judgement_model_only():
    assert sr.classify(_fly_ballistic("Mk21", "Generic ICBM")) == "ballistic"
    assert sr.classify(_glider()) == "glide"
    # ...and it does NOT move when a dive is commanded — a diving glider is
    # still judged on the stopwatch, which is what classify() decides.
    assert sr.classify(_patched(_glider(), terminal_alt_km=15.0)) == "glide"
    assert sr.classify(_patched(_glider(), dive_target_radius_km=5.0)) == "glide"


def test_no_form_letters_in_the_headline():
    """The headline names what the plan does, not a letter."""
    for r in (_fly_ballistic("Mk21", "Generic ICBM"), _glider()):
        head = sr.build_report(r)["headline"]
        assert "Form " not in head
        assert "MaRV" not in head


# ── descriptors(): one fact, one descriptor ─────────────────────────────────
def test_each_descriptor_has_its_own_trigger():
    g = _glider()
    assert sr.descriptors(_fly_ballistic("Mk21", "Generic ICBM")) == ["ballistic RV"]
    assert sr.descriptors(g) == ["glide"]
    assert sr.descriptors(_patched(g, banking=True)) == ["glide", "banking"]
    assert sr.descriptors(_patched(g, terminal_alt_km=15.0)) == \
        ["glide", "terminal dive"]
    assert sr.descriptors(_patched(g, dive_target_radius_km=5.0)) == \
        ["glide", "dive-at-target"]
    assert sr.descriptors(_patched(g, banking=True, terminal_alt_km=12.0)) == \
        ["glide", "banking", "terminal dive"]


def test_glide_to_impact_is_not_a_dive():
    """The shipped C-HGB/AHW plans set glider_terminal_dive=True with
    terminal_alt_km=0, which trajectory.py reads as GLIDE TO IMPACT (the
    analytic glide runs to its 30 km validity floor, sans commanded dive).
    The descriptor must agree with the integrator, not with the flag name."""
    prof = (_glider().get("heating_arc") or {}).get("profile") or {}
    assert prof.get("terminal_alt_km") == 0.0
    assert "terminal dive" not in sr.descriptors(_glider())


# ── The blocks Form C used to withhold ──────────────────────────────────────
def test_swerve_gets_the_windward_and_maneuver_blocks():
    """The regression this file exists for.  SWERVE has no commanded dive, so
    the retired Form C trigger withheld both blocks from the one vehicle in
    the library whose flight record is an AoA maneuver demonstration."""
    r = _swerve()
    prof = (r.get("heating_arc") or {}).get("profile") or {}
    assert prof.get("terminal_alt_km") == 0.0        # would have been Form B
    assert float(prof.get("pullup_g_max") or 0.0) > 0.0
    body = sr.build_report(r)["body"]
    assert "Windward-flank heating" in body
    assert "Maneuver-load anchors" in body


def test_maneuver_anchors_follow_the_commanded_lift_cap():
    """Gated on the g cap, not on a dive: zero g → no block, at any dive."""
    g = _glider()
    assert "Maneuver-load anchors" in sr.build_report(g)["body"]
    off = sr.build_report(_patched(g, pullup_g_max=0.0, terminal_alt_km=15.0))
    assert "Maneuver-load anchors" not in off["body"]


def test_terminal_dive_block_follows_the_dive_knobs():
    """Gated on a dive actually being commanded — by either knob."""
    g = _glider()
    assert "Terminal-dive transient" not in sr.build_report(g)["body"]
    for patch in ({"terminal_alt_km": 15.0}, {"dive_target_radius_km": 5.0}):
        assert "Terminal-dive transient" in \
            sr.build_report(_patched(g, **patch))["body"]


def test_a_ballistic_rv_gets_none_of_the_glide_blocks():
    """The one fork that still gates: the ballistic judgement model carries
    no glide-phase blocks, whatever the plan's leftover glider fields say."""
    body = sr.build_report(_fly_ballistic("Mk21", "Generic ICBM"))["body"]
    for blk in ("Windward-flank heating", "Terminal-dive transient",
                "Maneuver-load anchors"):
        assert blk not in body


def test_adding_a_dive_does_not_move_the_verdict():
    """Structural safety net: the retired Form C changed which blocks printed,
    never the survival tier.  Commanding a dive must still be presentation-
    only — a tier that moves means a block leaked into the verdict."""
    g = _glider()
    base = sr.build_report(g)
    for patch in ({"terminal_alt_km": 15.0}, {"banking": True},
                  {"dive_target_radius_km": 5.0}):
        assert sr.build_report(_patched(g, **patch))["tier"] == base["tier"]
