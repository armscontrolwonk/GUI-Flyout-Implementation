"""Plain-language lead of the survivability report (inverted pyramid).

The report leads with three layers a policy reader needs — what was flown,
why the verdict is what it is, what would change it — then a "Full analysis"
divider, below which the full engineering text (citations intact) is
unchanged.  Also pins the tier-vocabulary fix: the coverage text names the
blue/yellow bands the plot actually shows (the legacy "RED (too long/too
hot)" labels described colours that no longer exist), and the NRC ladder's
nearest-rung marker says design-lineage context, not "anchor".
"""

import copy
import json

import survivability_report as sr
import tps_ladder
from test_thresholds import _fly_glider

DIVIDER = "═══ Full analysis"


def _fly_ballistic(ro_name, booster):
    from booster_models import get_booster, ro_from_dict
    from trajectory import integrate_trajectory
    ro = ro_from_dict(json.load(open(f"ro_library/{ro_name}.ro.json")))
    p = get_booster(booster)
    p.ro = copy.deepcopy(ro)
    return integrate_trajectory(p, 0.0, 0.0, 90.0, max_time_s=6000.0,
                                dt_output=2.0)


def _lead(body):
    return body.split(DIVIDER)[0]


def test_lead_and_divider_glider():
    rep = sr.build_report(_fly_glider())
    body = rep["body"]
    assert DIVIDER in body
    lead = _lead(body)
    # layer 1: what was flown
    assert "C-HGB" in lead and "reentry arc" in lead
    # layer 2: the binding mechanism in plain words, with the material label
    assert "UHTC" in lead and "active oxidation" in lead
    assert "the nose is the binding problem" in lead
    # layer 3: the fix levers
    assert "What would change the verdict" in lead
    # NRC lineage context sentence, mechanism-distinguishing
    assert "NRC 2008" in lead and "different failure mode" in lead
    # the full analysis below the divider still carries the citations
    full = body.split(DIVIDER)[1]
    assert "Monteverde" in full and "Heating budget" in full


def test_lead_ballistic_degraded_and_fail():
    deg = sr.build_report(_fly_ballistic("Mk21", "Generic ICBM"))
    lead = _lead(deg["body"])
    assert "accuracy degrades" in lead or "accuracy is heavily degraded" in lead
    assert "What would change the verdict" in lead
    # An Mk21-class RV surviving with flight-demonstrated recession is WITHIN
    # EXPERIENCE (green) with the consequence annotated — not 'beyond design
    # envelope' (the accuracy axis must not drag the survival tier down).
    assert deg["tier"] == "experience"
    assert "WITHIN EXPERIENCE" in deg["headline"]
    assert "(accuracy degraded)" in deg["headline"]

    fail = sr.build_report(_fly_ballistic("Hwasong-11", "Scud-B (R-17)"))
    lead = _lead(fail["body"])
    assert fail["tier"] == "fail"
    # names the true failure mode (steel nose melts — it is not an ablator)
    assert "TPS fails" in lead and "melt" in lead
    assert "ablator burns through" not in lead


def test_coverage_text_uses_tier_vocabulary():
    body = sr.build_report(_fly_glider())["body"]
    # legacy labels named a red that no longer exists on the plot
    assert "RED (too long)" not in body and "RED (too hot)" not in body
    # too-hot exit fires in this fixture → yellow 'beyond design' wording
    assert "yellow 'beyond design' band" in body


def test_ladder_marker_is_lineage_not_anchor():
    txt = tps_ladder.format_ladder(900.0)
    assert "nearest anchor" not in txt
    assert "design lineage, not a demonstration" in txt
    # the crossover line spells out the F.o.M. switch in plain words
    assert "F.o.M." not in txt
    assert "wall temperature" in txt
