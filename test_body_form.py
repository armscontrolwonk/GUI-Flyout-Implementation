"""Lifting-body forms (ROParams.body_form) — Phase 1: data model + depiction.

The contract under test is the project's derive-don't-invent discipline plus
one hard physics guarantee: body_form is DEPICTION AND DATA ONLY.  Two
ROParams that differ only in body_form must produce byte-identical drag
polars (the trajectory rides on β / L/D until the lifting-body trim
estimator lands), the schematic must draw the honest asymmetric silhouette
from the stored fields, and what is NOT modeled (a wedge's span) must be
flagged, never invented.
"""

import dataclasses
import json

import pytest

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.patches import Polygon

import booster_models as mm
from booster_models import (BODY_FORMS, ro_from_dict, ro_to_dict,
                            booster_from_dict)
from booster_schematic import draw_booster
from trajectory import _aero_polar


def _ax():
    return Figure(figsize=(4, 8)).add_subplot(111)


def _ro(**overrides):
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    return dataclasses.replace(ro, **overrides) if overrides else ro


def _with_ro(**overrides):
    p = booster_from_dict(
        json.load(open("booster_library/AUR.booster.json")))   # finless
    ro = _ro(**overrides)
    p = mm.compose_loadout(p, ro, 1)
    p.ro = ro
    return p, ro


def _ro_body_patch(ax, ro):
    """The RO silhouette: the base-at-y=0 polygon offset to the right of the
    stack (x > 0), excluding wing panels (which never reach the body's full
    side-view width)."""
    cands = []
    for q in ax.patches:
        if not isinstance(q, Polygon):
            continue
        v = q.get_path().vertices
        if v[:, 1].min() <= 1e-9 and v[:, 0].min() > 0 \
                and v[:, 1].max() >= 0.5 * ro.length_m:
            cands.append(q)
    assert cands, "reentry-object silhouette not found"
    return max(cands, key=lambda q: q.get_path().vertices[:, 0].max()
               - q.get_path().vertices[:, 0].min())


# ── data model ──────────────────────────────────────────────────────────────
def test_round_trip_default_and_normalisation():
    assert _ro().body_form == "axisymmetric"          # absent key → default
    for form in BODY_FORMS:
        d = ro_to_dict(_ro(body_form=form))
        assert d["body_form"] == form
        assert ro_from_dict(d).body_form == form
    # unknown / legacy strings normalise, never crash and never propagate
    assert ro_from_dict(dict(ro_to_dict(_ro()),
                             body_form="waverider")).body_form == "axisymmetric"


# ── the amended guarantee (Phase 3): cruise polar is form-blind; the pull
# ceiling is geometry ────────────────────────────────────────────────────────
def test_cruise_polar_is_form_identical_but_ceiling_is_geometry():
    """Phase 1 promised body_form was not physics; Phase 3 amends that
    DELIBERATELY and narrowly: the CRUISE polar (C_D0, k, A_ref, C_L*,
    e_pull, C_L0) still depends only on β / L-D — identical across forms
    when no estimator trim row is stored — but C_L_max is now SHAPE-DERIVED
    for a lifting form with sufficient geometry.  The half-cone derives from
    ⌀ + length alone; the wedge needs its span and keeps the body ceiling
    without it (flagged elsewhere, never invented)."""
    polars = {f: _aero_polar(_ro(body_form=f)) for f in BODY_FORMS}
    base = polars["axisymmetric"]
    for f, p in polars.items():
        assert (p.C_D0, p.k, p.A_ref, p.C_L_star, p.e_pull, p.C_L0) == \
            (base.C_D0, base.k, base.A_ref, base.C_L_star, base.e_pull,
             base.C_L0), f
    # C-HGB stores no body span → the wedge cannot state a planform: ceiling
    # unchanged.  The half-cone CAN derive from ⌀/L: ceiling is its own.
    assert polars["wedge"].C_L_max == base.C_L_max
    assert polars["half_cone"].C_L_max != base.C_L_max


# ── the silhouette is the stored geometry, asymmetric ───────────────────────
def test_wedge_tip_sits_over_the_flat_flank_not_the_centreline():
    """A wedge's nose tip lies on the flat surface — over one edge of the
    base, NOT centred like a cone.  Base width = the stored ⌀ (= base depth)."""
    p, ro = _with_ro(body_form="wedge")
    ax = _ax(); draw_booster(ax, p)
    v = _ro_body_patch(ax, ro).get_path().vertices
    assert v[:, 0].max() - v[:, 0].min() == pytest.approx(ro.diameter_m)
    tip_x = v[v[:, 1].argmax(), 0]
    assert tip_x == pytest.approx(v[:, 0].min())      # over the flat flank
    assert v[:, 1].max() == pytest.approx(ro.length_m)


def test_half_cone_side_view_is_half_the_diameter_deep():
    """Cut at the diametral plane, a half-cone's side elevation is ⌀/2 deep —
    drawing it a full ⌀ wide would claim a body of revolution."""
    p, ro = _with_ro(body_form="half_cone")
    ax = _ax(); draw_booster(ax, p)
    v = _ro_body_patch(ax, ro).get_path().vertices
    assert v[:, 0].max() - v[:, 0].min() == pytest.approx(ro.diameter_m / 2.0)
    tip_x = v[v[:, 1].argmax(), 0]
    assert tip_x == pytest.approx(v[:, 0].min())


def test_captions_name_the_form_and_flag_the_unmodeled_span():
    p, _ = _with_ro(body_form="wedge")
    ax = _ax(); draw_booster(ax, p)
    txt = "\n".join(t.get_text() for t in ax.texts)
    assert "wedge lifting body" in txt
    assert "span not modeled" in txt                  # the flag, not a guess
    p2, _ = _with_ro(body_form="half_cone")
    ax2 = _ax(); draw_booster(ax2, p2)
    txt2 = "\n".join(t.get_text() for t in ax2.texts)
    assert "half-cone lifting body" in txt2


def test_biconic_is_ignored_for_lifting_forms():
    """biconic is a body-of-revolution concept: on a wedge the stored biconic
    fields must neither reshape the silhouette nor appear in the caption."""
    p, ro = _with_ro(body_form="wedge", biconic=True,
                     fore_length_m=0.5, break_diameter_m=0.3)
    ax = _ax(); draw_booster(ax, p)
    v = _ro_body_patch(ax, ro).get_path().vertices
    tip_x = v[v[:, 1].argmax(), 0]
    assert tip_x == pytest.approx(v[:, 0].min())      # still the wedge triangle
    txt = "\n".join(t.get_text() for t in ax.texts)
    assert "biconic" not in txt


def test_body_span_round_trips_and_defaults_zero():
    assert _ro().body_span_m == 0.0
    d = ro_to_dict(_ro(body_form="wedge", body_span_m=0.9))
    assert d["body_span_m"] == 0.9
    assert ro_from_dict(d).body_span_m == 0.9


def test_body_span_never_leaks_into_wing_geometry():
    """The phantom-wing guard: body_span_m is BODY geometry (the wedge's own
    lifting surface), distinct from wing_span_exposed_m (a wing ON a body).
    wing_geometry() must be blind to it — with body_span set and no wing
    fields it returns nothing, and with both set the wing derivation is
    byte-identical to the same wing without body_span."""
    from booster_models import wing_geometry
    bare = _ro(body_form="wedge", body_span_m=0.9)
    assert wing_geometry(bare) == (0.0, 0.0, None)
    winged = _ro(wing_root_chord_m=0.6, wing_span_exposed_m=0.15)
    winged_plus_span = _ro(wing_root_chord_m=0.6, wing_span_exposed_m=0.15,
                           body_span_m=0.9)
    assert wing_geometry(winged) == wing_geometry(winged_plus_span)


def test_wedge_caption_shows_stored_span_and_flags_unset():
    """The schematic's 'span not modeled' flag retires exactly when the span
    is stored — and the stored span is reported, not drawn (out of the
    side-elevation plane)."""
    p, _ = _with_ro(body_form="wedge", body_span_m=0.9)
    ax = _ax(); draw_booster(ax, p)
    txt = "\n".join(t.get_text() for t in ax.texts)
    assert "span 0.9 m (not drawn)" in txt
    assert "span not modeled" not in txt
    p2, _ = _with_ro(body_form="wedge")          # span unset
    ax2 = _ax(); draw_booster(ax2, p2)
    assert "span not modeled" in "\n".join(t.get_text() for t in ax2.texts)


def test_trajectory_passes_body_form_to_windward_heating():
    """End-to-end wiring: a glide flown with body_form='wedge' must reach the
    windward screen with the flat-surface acreage fraction (0.018), and the
    same vehicle as a body of revolution with the cone fraction (0.13) —
    trajectory.py forwards body_form; heating selects (METHODS §13.8)."""
    import copy
    import heating
    from booster_models import (get_booster, apply_reentry_plan,
                                load_reentry_plan)
    from trajectory import integrate_trajectory

    def fly(form):
        ro = _ro()
        rp = load_reentry_plan("C-HGB")
        if rp is not None:
            ro = apply_reentry_plan(ro, rp)
        ro = dataclasses.replace(ro, body_form=form, glider_enabled=True)
        p = get_booster("Minotaur-IV + HTV-2")
        p.ro = copy.deepcopy(ro)
        r = integrate_trajectory(p, 0.0, 0.0, 90.0, max_time_s=8000.0,
                                 dt_output=2.0, cutoff_time_s=170.0)
        w = (r.get('heating_fom') or {}).get('windward') or {}
        return w

    w_wedge = fly("wedge")
    w_cone = fly("axisymmetric")
    assert w_wedge.get('acreage_fraction') == heating.BODY_FLUX_FRACTION_FLAT
    assert w_cone.get('acreage_fraction') == heating.BODY_FLUX_FRACTION
    # same flight, ~7x lower windward flux for the flat form
    assert w_wedge['q_windward_MW_m2']['hi'] < 0.5 * w_cone['q_windward_MW_m2']['hi']


def test_wings_on_a_lifting_body_are_reported_not_drawn():
    """Spanwise panels lie out of a lifting body's side-elevation plane, so
    wing data adds NO patches — the caption reports S/AR flagged 'not drawn'."""
    p0, _ = _with_ro(body_form="wedge")
    ax0 = _ax(); draw_booster(ax0, p0)
    p1, _ = _with_ro(body_form="wedge", wing_root_chord_m=0.6,
                     wing_span_exposed_m=0.15, wing_sweep_deg=60.0)
    ax1 = _ax(); draw_booster(ax1, p1)
    assert len(ax1.patches) == len(ax0.patches)
    txt = "\n".join(t.get_text() for t in ax1.texts)
    assert "wings" in txt and "not drawn" in txt and "derived" in txt
