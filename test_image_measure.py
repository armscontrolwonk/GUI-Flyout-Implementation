"""Pure measurement logic for the image dimensioning tool (image_measure.py).

Pins the risk-register behaviour (IMAGE_DIMENSION_TOOL_DESIGN.md) that must
hold with or without a GUI: scale + pixel quantum, the resolution floor (R4),
convention conversions (R5), the clocking correction (R1), anchor-free
reporting (R2), and the provenance stamp (decision 1).  No Tk, no display.
"""

import math

import pytest

import image_measure as im


def _scale(px_per_m=100.0, note="L=10 m claimed"):
    # two points 1000 px apart == 10 m → 0.01 m/px (1 px = 1 cm)
    return im.Scale((0, 0), (10.0 * px_per_m, 0), 10.0, anchor_note=note)


# ── scale & pixel quantum ───────────────────────────────────────────────────
def test_scale_metres_per_pixel_and_quantum():
    s = _scale()
    assert s.m_per_px == pytest.approx(0.01)
    assert s.pixel_quantum_m() == pytest.approx(0.01)
    assert "1 px = 1 cm" in s.quantum_str()
    v, span = s.measure((0, 0), (250, 0))
    assert v == pytest.approx(2.5) and span == pytest.approx(250)


def test_scale_rejects_degenerate_anchor():
    with pytest.raises(ValueError):
        im.Scale((5, 5), (5, 5), 10.0)          # coincident points
    with pytest.raises(ValueError):
        im.Scale((0, 0), (100, 0), 0.0)         # non-positive distance


# ── R4 resolution floor ─────────────────────────────────────────────────────
def test_measurement_below_floor_is_refused_and_flagged():
    s = _scale()
    click_m, span = s.measure((0, 0), (3, 0))    # 3 px < 5 px floor
    m = im.Measurement("_nose_var", click_m, span, scale=s, convention="ro_nose_r")
    assert m.refused is True
    assert any("resolution floor" in f for f in m.flags)


def test_measurement_above_floor_is_accepted():
    s = _scale()
    click_m, span = s.measure((0, 0), (60, 0))
    m = im.Measurement("_dia_var", click_m, span, scale=s, convention="ro_diameter")
    assert m.refused is False
    assert m.value_m == pytest.approx(0.60)
    assert "1 px = 1 cm" in m.quantum_str()


# ── R5 convention conversions (click → stored) ──────────────────────────────
def test_half_cone_depth_stores_twice_the_clicked_depth():
    s = _scale()
    click_m, span = s.measure((0, 0), (29, 0))   # 0.29 m clicked side-view depth
    m = im.Measurement("_dia_var", click_m, span, scale=s, convention="half_cone_depth")
    assert m.value_m == pytest.approx(0.58)      # stored ⌀ = 2 × depth


def test_nose_radius_is_half_the_clicked_tip_width():
    s = _scale()
    click_m, span = s.measure((0, 0), (8, 0))    # 8 cm tip width
    m = im.Measurement("_nose_var", click_m, span, scale=s, convention="ro_nose_r")
    assert m.value_m == pytest.approx(0.04)      # radius = width / 2


def test_plain_length_is_identity():
    s = _scale()
    click_m, span = s.measure((0, 0), (200, 0))
    m = im.Measurement("_len_var", click_m, span, scale=s, convention="ro_length")
    assert m.value_m == pytest.approx(2.0)


# ── R1 clocking correction ──────────────────────────────────────────────────
def test_x_roll_correction_and_flag():
    s = _scale()
    click_m, span = s.measure((0, 0), (100, 0))  # 1.0 m apparent span
    straight = im.Measurement("fin_span", click_m, span, scale=s, clocking="in_plane")
    rolled = im.Measurement("fin_span", click_m, span, scale=s, clocking="x_rolled")
    assert straight.value_m == pytest.approx(1.0)
    assert rolled.value_m == pytest.approx(1.0 / math.cos(math.radians(45.0)))
    assert any("cos45" in f for f in rolled.flags)


def test_unknown_clocking_does_not_correct_but_warns():
    s = _scale()
    click_m, span = s.measure((0, 0), (100, 0))
    m = im.Measurement("fin_span", click_m, span, scale=s, clocking="unknown")
    assert m.value_m == pytest.approx(1.0)       # never auto-corrected
    assert any("clocking unknown" in f for f in m.flags)


def test_clocking_correction_helper_direct():
    assert im.clocking_correction(1.0, "in_plane") == pytest.approx(1.0)
    assert im.clocking_correction(1.0, "x_rolled") == pytest.approx(math.sqrt(2.0))


# ── R2 anchor-free reporting ────────────────────────────────────────────────
def test_anchor_free_note_names_ratios_and_angles():
    note = im.anchor_free_note()
    assert "fineness ratio" in note and "half-angle" in note
    assert "inherit the scale anchor's error 1:1" in note


# ── provenance stamp (decision 1) ───────────────────────────────────────────
def test_provenance_stamp_is_a_dimensional_draft_with_fields_and_quantum():
    s = _scale(note="scale from claimed L=10 m")
    ms = [im.Measurement("_len_var", *s.measure((0, 0), (200, 0)), scale=s,
                         convention="ro_length"),
          im.Measurement("_dia_var", *s.measure((0, 0), (60, 0)), scale=s,
                         convention="ro_diameter")]
    stamp = im.provenance_stamp(ms, s, "2026-07-30")
    assert stamp.startswith("[2026-07-30] dimensional draft from image")
    assert "_len_var" in stamp and "_dia_var" in stamp
    assert "claimed L=10 m" in stamp and "1 px = 1 cm" in stamp


def test_provenance_stamp_excludes_refused_measurements():
    s = _scale()
    good = im.Measurement("_dia_var", *s.measure((0, 0), (60, 0)), scale=s,
                          convention="ro_diameter")
    bad = im.Measurement("_nose_var", *s.measure((0, 0), (2, 0)), scale=s,
                         convention="ro_nose_r")
    stamp = im.provenance_stamp([good, bad], s, "2026-07-30")
    assert "_dia_var" in stamp and "_nose_var" not in stamp   # refused omitted


# ── the prompt checklist by body form (R5 embedding) ────────────────────────
def test_ro_prompts_by_body_form():
    ax = im.ro_prompts("axisymmetric")
    assert [p["field"] for p in ax] == ["_len_var", "_dia_var", "_nose_var"]
    assert all(p["view"] == "side" for p in ax)

    wed = im.ro_prompts("wedge")
    fields = [p["field"] for p in wed]
    assert "_body_span_var" in fields
    span = next(p for p in wed if p["field"] == "_body_span_var")
    assert span["view"] == "plan"               # span needs a plan view
    depth = next(p for p in wed if p["field"] == "_dia_var")
    assert depth["convention"] == "wedge_depth"

    hc = im.ro_prompts("half_cone")
    depth = next(p for p in hc if p["field"] == "_dia_var")
    assert depth["convention"] == "half_cone_depth"   # stored ⌀ = 2×


def test_booster_prompts_from_topology():
    ps = im.booster_prompts(n_stages=2, has_fairing=True, has_fins=True,
                            n_fins=4, n_strapons=4)
    fields = [p["field"] for p in ps]
    assert fields[:4] == ["stage1_len", "stage1_dia", "stage2_len", "stage2_dia"]
    assert {"fairing_len", "fairing_dia", "fin_span", "fin_root", "fin_tip",
            "strapon_dia", "strapon_len"} <= set(fields)


def test_booster_prompts_state_the_count_assumption():
    """Measure-one-declare-count: a repeated feature's prompt says how many the
    single measurement will be replicated to (design 'measure one')."""
    ps = im.booster_prompts(has_fins=True, n_fins=4, n_strapons=3)
    fin = next(p for p in ps if p["field"] == "fin_span")
    strap = next(p for p in ps if p["field"] == "strapon_dia")
    assert "4" in fin["label"] and "assumed identical" in fin["label"]
    assert "3" in strap["label"] and "assumed identical" in strap["label"]


def test_booster_prompts_minimal_is_just_stage_one():
    assert [p["field"] for p in im.booster_prompts(n_stages=1)] == \
        ["stage1_len", "stage1_dia"]
