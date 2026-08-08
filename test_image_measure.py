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


def test_all_conventions_are_metre_to_metre():
    """UNITS CONTRACT: the tool works in metres end-to-end.  Every convention
    converter is a pure GEOMETRIC factor (identity, ×2 half-cone depth, ×½
    nose radius) — never a unit change (no cm→m, mm→m hiding here).  The
    editor fields the tool writes are all metre-labelled (*_m model
    attributes); non-metre fields (masses kg, sweeps °, motor web mm) are
    outside the prompt lists by design."""
    for key, (label, conv) in im.CONVENTIONS.items():
        assert conv(1.0) in (1.0, 2.0, 0.5), key
        assert conv(0.0) == 0.0, key


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


def test_hand_entry_is_stamped_as_entered_not_measured():
    """Type-value path: a known dimension can be ENTERED instead of clicked.
    The stamp lists it separately — the audit trail never claims a typed
    number came off the image."""
    s = _scale(note="claimed L=10 m")
    m = im.Measurement("_len_var", *s.measure((0, 0), (200, 0)), scale=s,
                       convention="ro_length")
    h = im.HandEntry("_dia_var", 0.58)
    assert h.refused is False and h.value_m == pytest.approx(0.58)
    assert any("entered by hand" in f for f in h.flags)
    stamp = im.provenance_stamp([m, h], s, "2026-08-01")
    assert "entered by hand (not measured): _dia_var" in stamp
    assert "_dia_var" not in stamp.split("entered by hand")[0]  # not in measured list


def test_provenance_stamp_excludes_refused_measurements():
    s = _scale()
    good = im.Measurement("_dia_var", *s.measure((0, 0), (60, 0)), scale=s,
                          convention="ro_diameter")
    bad = im.Measurement("_nose_var", *s.measure((0, 0), (2, 0)), scale=s,
                         convention="ro_nose_r")
    stamp = im.provenance_stamp([good, bad], s, "2026-07-30")
    assert "_dia_var" in stamp and "_nose_var" not in stamp   # refused omitted


# ── Phase B: cross-view consistency (two views, two scales, one truth) ─────
def test_view_consistency_needs_both_lengths():
    assert im.view_consistency({}) is None
    assert im.view_consistency({"_len_var": 3.0}) is None
    assert im.view_consistency({im.PLAN_LEN_CHECK_FIELD: 3.0}) is None


def test_view_consistency_reports_scale_disagreement():
    vc = im.view_consistency({"_len_var": 3.0, im.PLAN_LEN_CHECK_FIELD: 3.3})
    assert vc["rel"] == pytest.approx(0.10)
    note = im.view_consistency_note(vc)
    assert "3" in note and "+10" in note


def test_wedge_prompts_include_the_plan_length_check():
    """The plan-view length is check-only (never stored) — it exists to audit
    the two views' independent scale anchors against each other."""
    wed = im.ro_prompts("wedge")
    chk = next(p for p in wed if p["field"] == im.PLAN_LEN_CHECK_FIELD)
    assert chk["view"] == "plan"
    assert chk["convention"] == "ro_length"


def test_stamp_notes_multiple_views():
    s = _scale()
    m1 = im.Measurement("_len_var", *s.measure((0, 0), (200, 0)), scale=s,
                        convention="ro_length", view="side")
    m2 = im.Measurement("_body_span_var", *s.measure((0, 0), (90, 0)), scale=s,
                        convention="wedge_span", view="plan")
    stamp = im.provenance_stamp([m1, m2], s, "2026-08-01")
    assert "views: plan+side" in stamp
    single = im.provenance_stamp([m1], s, "2026-08-01")
    assert "views:" not in single


# ── the prompt checklist by body form (R5 embedding) ────────────────────────
def test_ro_prompts_by_body_form():
    ax = im.ro_prompts("axisymmetric")
    assert [p["field"] for p in ax] == ["_len_var", "_dia_var", "_nose_var",
                                        "_wing_root_var", "_wing_span_var",
                                        "_wing_tip_derive"]
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


def test_ro_prompts_biconic_adds_break_geometry_when_declared():
    """The biconic checkbox is declared topology: ticking it adds fore-cone
    length and break diameter to the checklist; unticked, they never appear
    (the fields are disabled in the editor too)."""
    plain = [p["field"] for p in im.ro_prompts("axisymmetric", biconic=False)]
    assert "_fore_len_var" not in plain and "_break_dia_var" not in plain
    bic = [p["field"] for p in im.ro_prompts("axisymmetric", biconic=True)]
    assert "_fore_len_var" in bic and "_break_dia_var" in bic


def test_ro_prompts_always_offer_the_planform_not_area():
    """Wing prompts are ALWAYS offered for forms that can carry a wing — the
    wings are visible in the image whether or not Maneuvering is ticked yet
    (apply enables the section when wing geometry lands; skip is structural).
    The tool measures the PLANFORM (root chord + exposed span) — S and AR
    stay derived by the editor (wing_geometry single source of truth)."""
    ps = im.ro_prompts("axisymmetric")
    fields = [p["field"] for p in ps]
    assert "_wing_root_var" in fields and "_wing_span_var" in fields
    assert not any("area" in f or "sweep" in f or "_wing_ar" in f
                   for f in fields)
    # half-cone + delta wing (the Fetterman configuration) keeps its wing rows
    hc = [p["field"] for p in im.ro_prompts("half_cone")]
    assert "_wing_span_var" in hc
    # the wedge's body IS the lifting surface — wing rows are disabled in the
    # editor and must never be prompted (they'd be zeroed on save anyway)
    wed = [p["field"] for p in im.ro_prompts("wedge")]
    assert not any(f.startswith("_wing") for f in wed)


def test_wedge_never_gets_wing_prompts():
    """The wedge's body IS its lifting surface — its wing rows are disabled
    by design, so neither planform nor sweep prompts exist for it."""
    assert not any(f.startswith("_wing") for f in
                   (p["field"] for p in im.ro_prompts("wedge")))
    assert im.ro_angle_prompts("wedge") == []


def test_booster_prompts_from_topology():
    ps = im.booster_prompts(n_stages=2, has_fairing=True, has_fins=True,
                            n_fins=4, n_strapons=4)
    fields = [p["field"] for p in ps]
    assert fields[:4] == ["stage1_len", "stage1_dia", "stage2_len", "stage2_dia"]
    assert {"fairing_len", "fairing_dia", "fin_span", "fin_root", "fin_tip",
            "strapon_dia", "strapon_len"} <= set(fields)
    assert fields[-1] == im.OVERALL_LEN_CHECK_FIELD    # closure check, always last


def test_booster_prompts_state_the_count_assumption():
    """Measure-one-declare-count: a repeated feature's prompt says how many the
    single measurement will be replicated to (design 'measure one')."""
    ps = im.booster_prompts(has_fins=True, n_fins=4, n_strapons=3)
    fin = next(p for p in ps if p["field"] == "fin_span")
    strap = next(p for p in ps if p["field"] == "strapon_dia")
    assert "4" in fin["label"] and "assumed identical" in fin["label"]
    assert "3" in strap["label"] and "assumed identical" in strap["label"]


def test_booster_prompts_minimal_is_stage_one_plus_check():
    """Every booster checklist ends with the check-only overall length — the
    one 'measurement' that never writes a field (the model derives the total
    from its parts); it exists to feed the length-closure warning."""
    assert [p["field"] for p in im.booster_prompts(n_stages=1)] == \
        ["stage1_len", "stage1_dia", im.OVERALL_LEN_CHECK_FIELD]


# ── length closure (warn-only, never normalizes) ────────────────────────────
def _closure_prompts():
    return im.booster_prompts(n_stages=2, has_fairing=True)


def test_length_closure_needs_a_total():
    assert im.length_closure({"stage1_len": 5.0}, _closure_prompts(), None) is None
    assert im.length_closure({"stage1_len": 5.0}, _closure_prompts(), 0.0) is None


def test_length_closure_not_applicable_without_segments():
    # the RO checklist has no stage/fairing segments — nothing to tile
    assert im.length_closure({"_len_var": 2.0},
                             im.ro_prompts("axisymmetric"), 2.0) is None


def test_length_closure_pending_lists_missing_segments():
    c = im.length_closure({"stage1_len": 5.0, "stage2_len": 2.6},
                          _closure_prompts(), 10.2)
    assert c["complete"] is False
    assert c["missing"] == ["fairing_len"]
    assert c["sum_m"] == pytest.approx(7.6)
    assert "pending" in im.closure_note(c) and "fairing_len" in im.closure_note(c)


def test_length_closure_complete_reports_signed_error():
    acc = {"stage1_len": 5.0, "stage2_len": 2.6, "fairing_len": 2.2}
    c = im.length_closure(acc, _closure_prompts(), 10.2)
    assert c["complete"] is True
    assert c["delta_m"] == pytest.approx(-0.4)
    assert c["rel"] == pytest.approx(-0.4 / 10.2)
    note = im.closure_note(c)
    assert "9.8" in note and "10.2" in note and "-3.9%" in note


def test_length_closure_diameters_do_not_count():
    """Only LENGTH segments tile the stack — a diameter accepted along the way
    must not pollute the sum."""
    acc = {"stage1_len": 5.0, "stage1_dia": 3.0, "stage2_len": 5.2}
    c = im.length_closure(acc, im.booster_prompts(n_stages=2), 10.2)
    assert c["complete"] is True
    assert c["sum_m"] == pytest.approx(10.2)
    assert c["rel"] == pytest.approx(0.0)


# ── R1 clocking wired to the prompts that need it ───────────────────────────
def test_only_exposed_spans_are_clocking_sensitive():
    """The ×-roll cos45 correction is offered ONLY where a span can foreshorten:
    a fin's or wing panel's exposed span seen side-on.  Chords, diameters,
    lengths and the (plan-view) wedge span are not clocking-sensitive — so the
    dialog never offers a correction that would silently inflate them."""
    ps = im.booster_prompts(n_stages=1, has_fairing=True, has_fins=True,
                            n_fins=4, n_strapons=2)
    sensitive = [p["field"] for p in ps if p.get("clocking_sensitive")]
    assert sensitive == ["fin_span"]
    ro = im.ro_prompts("axisymmetric", biconic=True)
    sensitive = [p["field"] for p in ro if p.get("clocking_sensitive")]
    assert sensitive == ["_wing_span_var"]
    wed = im.ro_prompts("wedge")
    assert not any(p.get("clocking_sensitive") for p in wed)   # plan view = true span


def test_clocking_options_default_to_no_correction():
    """First option must be the do-nothing choice: the correction is OFFERED,
    never the default (design R1 — never silently inflate a span)."""
    assert im.CLOCKING_OPTIONS[0][1] == "in_plane"
    keys = [k for _, k in im.CLOCKING_OPTIONS]
    assert keys == ["in_plane", "x_rolled", "unknown"]


# ── Angle measurement (3 clicks; anchor-free) and its cross-checks ──────────
def test_angle_between_deg_basics():
    assert im.angle_between_deg((0, 0), (10, 0), (0, 10)) == pytest.approx(90.0)
    assert im.angle_between_deg((0, 0), (10, 0), (10, 10)) == pytest.approx(45.0)
    with pytest.raises(ValueError):
        im.angle_between_deg((0, 0), (0, 0), (10, 0))     # degenerate ray


def test_angle_measurement_needs_no_scale_but_refuses_short_rays():
    """Angles are anchor-free (no Scale object anywhere), but a short ray is
    noisy — the guard is on RAY LENGTH, the angle analogue of R4."""
    m = im.AngleMeasurement("_wing_sweep_var", (0, 0), (200, 0), (140, 140))
    assert m.refused is False
    assert m.value_deg == pytest.approx(45.0)
    assert any("anchor-free" in f for f in m.flags)
    short = im.AngleMeasurement("_wing_sweep_var", (0, 0), (6, 0), (0, 200))
    assert short.refused is True


def test_angle_complement_stores_sweep_from_the_le_root_opening():
    """Wing sweep is measured between two REAL edges (LE and root chord) as a
    small opening, and stored as Λ = 90° − it.  A 13° LE↔root opening → 77°
    sweep; raw_deg keeps the clicked value for the read-out."""
    import math
    # a ~13° opening at the origin: root chord along +x, LE 13° off it
    a1 = (math.cos(math.radians(13.0)) * 200, math.sin(math.radians(13.0)) * 200)
    m = im.AngleMeasurement("_wing_sweep_var", (0, 0), a1, (200, 0),
                            complement=True)
    assert m.raw_deg == pytest.approx(13.0)
    assert m.value_deg == pytest.approx(77.0)
    assert any("→ sweep Λ" in f for f in m.flags)


def test_sweep_from_planform_identity():
    """tan Λ = (c_r − c_t)/s_e; delta wing (c_t = 0) → tan Λ = c_r/s_e."""
    assert im.sweep_from_planform(1.0, 1.0) == pytest.approx(45.0)
    assert im.sweep_from_planform(2.0, 1.0, tip_chord_m=1.0) == pytest.approx(45.0)
    assert im.sweep_from_planform(1.0, 0.0) is None


def test_cone_half_angle_identity():
    assert im.cone_half_angle_from_lengths(1.0, 0.5) == pytest.approx(45.0)
    assert im.cone_half_angle_from_lengths(1.0, 0.0) is None


def test_diagram_style_tokens_and_fillable_bodies():
    """The art direction is a single data dict — every key the renderer
    reads must exist — and every base (plus every shape-aware RO variant)
    has at least one CLOSED outline, so the filled-body style always has a
    body to fill.  closed_poly is the fill/stroke discriminator: closed →
    filled body art, open → detail stroke (cone break, stage joint)."""
    for key in ("bg", "outline", "outline_width", "fill", "highlight",
                "measure", "measure_width", "arrowshape", "dot_r",
                "label_offset", "arc_r"):
        assert key in im.DIAGRAM_STYLE
    assert im.closed_poly([(0, 0), (1, 0), (1, 1), (0, 0)])
    assert not im.closed_poly([(0, 0), (1, 0)])           # detail stroke
    assert not im.closed_poly([(0, 0), (1, 0), (1, 1)])   # unclosed triangle
    for name, items in im.DIAGRAM_BASES.items():
        assert any(im.closed_poly(it["pts"]) for it in items), name
    for shape in ("cone", "tangent_ogive", "lv_haack", "parabola",
                  "blunt_cylinder"):
        items = im.ro_side_base(shape)
        assert all(im.closed_poly(it["pts"]) for it in items), shape


def test_diagram_subject_names_a_drawn_element():
    """Every prompt's diagram declares WHICH vehicle element it measures
    (the renderer fills that one white, the rest grey) — and the subject is
    always a tag the base art actually draws, so the highlight can never
    silently miss.  The overall-length cross-check spans the whole vehicle:
    no subject, nothing singled out."""
    tags = {name: {it["tag"] for it in items}
            for name, items in im.DIAGRAM_BASES.items()}
    checks = [("stage1_len", "s1"), ("stage1_dia", "s1"),
              ("stage1_top_dia", "s1"), ("stage1_interstage_len", "inter1"),
              ("stage2_len", "s2"), ("stage2_interstage_len", "inter2"),
              ("fairing_len", "fairing"), ("fin_root", "fin"),
              ("strapon_len", "strapon"), ("_len_var", "body"),
              ("_nose_var", "nose"), ("_wing_root_var", "fin_r")]
    for field, want in checks:
        spec = im.diagram_spec(dict(field=field))
        assert spec["subject"] == want, field
        assert want in tags[spec["base"]], field
    assert im.diagram_spec(
        dict(field=im.OVERALL_LEN_CHECK_FIELD))["subject"] is None
    # the shape-aware RO bases draw the same tags the RO subjects use
    assert {"body", "fin_r"} <= {it["tag"] for it in im.ro_side_base("cone")}


def test_ro_side_layout_draws_loaded_proportions():
    """When dimensions are loaded, the art IS those proportions: uniform
    px-per-metre, so a 10:1 body draws 10:1.  Core gate: no length or no
    diameter → None (the caller keeps the representative default art).
    The measurement points follow the reshaped geometry — the diameter
    arrow lands exactly on the drawn base corners."""
    assert im.ro_side_layout(dict(length_m=0, diameter_m=1)) is None
    assert im.ro_side_layout(dict(diameter_m=1)) is None
    lay = im.ro_side_layout(dict(length_m=10.0, diameter_m=1.0), 232, 160)
    body = next(p for p in lay["polys"] if p["tag"] == "body")
    xs = [x * 232 for x, _ in body["pts"]]
    ys = [y * 160 for _, y in body["pts"]]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    assert w / h == pytest.approx(0.1, rel=0.15)      # slenderness literal
    def _on_body(pt):
        return any(abs(pt[0] - x) < 1e-9 and abs(pt[1] - y) < 1e-9
                   for x, y in body["pts"])
    p1, p2 = lay["pts"]["_dia_var"]
    assert _on_body(p1) and _on_body(p2)
    # biconic: the break line sits at the fore-length fraction
    lay = im.ro_side_layout(dict(length_m=2.0, diameter_m=1.0, biconic=True,
                                 fore_length_m=1.0, break_diameter_m=0.5))
    y_break = lay["pts"]["_break_dia_var"][0][1]
    assert y_break == pytest.approx(0.06 + 0.5 * 0.84, abs=0.02)
    # wings: tip chord derives from sweep, drawn with a visible tip edge
    lay = im.ro_side_layout(dict(length_m=2.0, diameter_m=1.0,
                                 wing_root_m=0.6, wing_span_m=0.3,
                                 wing_sweep_deg=45.0))
    assert "_wing_tip_derive" in lay["pts"]
    assert any(p["tag"] == "fin_r" for p in lay["polys"])
    assert lay["subjects"]["_wing_root_var"] == "fin_r"
    # shape-aware in the layout too: a curved profile has many vertices
    cone = im.ro_side_layout(dict(length_m=2.0, diameter_m=1.0))
    haack = im.ro_side_layout(dict(length_m=2.0, diameter_m=1.0,
                                   shape="lv_haack"))
    n_of = lambda l: len(next(p for p in l["polys"]
                              if p["tag"] == "body")["pts"])
    assert n_of(haack) > n_of(cone) + 4


def test_dims_merge_and_measured_zero_tip_draws_pointed():
    """Live diagrams: accepted values overlay the open-time dims, and a
    MEASURED tip chord of 0 is a real datum — the fin draws as a triangle
    (the sweep-derived path can never say 'pointed' as firmly).  The
    tip-chord prompt's own diagram keeps the visible edge (tip_floor)."""
    dims = dict(length_m=2.0, diameter_m=1.0, wing_root_m=0.6,
                wing_span_m=0.3, wing_sweep_deg=45.0)
    merged = im.ro_dims_with_accepted(dims, {"_wing_tip_derive": 0.0,
                                             "_len_var": 4.0})
    assert merged["wing_tip_m"] == 0.0 and merged["length_m"] == 4.0
    assert merged is not dims and dims.get("wing_tip_m") is None  # no mutate
    fin = next(p for p in im.ro_side_layout(merged)["polys"]
               if p["tag"] == "fin_r")
    assert len(fin["pts"]) == 4                      # pointed: triangle
    fin_f = next(p for p in im.ro_side_layout(merged, tip_floor=True)["polys"]
                 if p["tag"] == "fin_r")
    assert len(fin_f["pts"]) == 5                    # tip prompt: real edge
    # booster merge routes stage / fin / fairing fields into the structure
    bdims = dict(stages=[dict(length_m=8.0, diameter_m=1.0)],
                 fins=dict(root_m=2.0, span_m=1.0, tip_m=0.5),
                 fairing=dict(length_m=2.0, diameter_m=1.0))
    bm = im.booster_dims_with_accepted(
        bdims, {"stage1_len": 9.0, "fin_tip": 0.0, "fairing_len": 2.5})
    assert bm["stages"][0]["length_m"] == 9.0
    assert bm["fins"]["tip_m"] == 0.0
    assert bm["fairing"]["length_m"] == 2.5
    assert bdims["stages"][0]["length_m"] == 8.0     # original untouched
    fin = next(p for p in im.booster_side_layout(bm)["polys"]
               if p["tag"] == "fin")
    assert len(fin["pts"]) == 4                      # measured-0 fin: pointed


def test_booster_side_layout_stacks_loaded_stages():
    """Stage boxes tile the stack in proportion to their loaded lengths;
    a third stage gets its OWN box/points (the static art folds it into
    S1); interstage bands appear only when declared.  Core gate: stage 1
    length + diameter required."""
    assert im.booster_side_layout(dict(stages=[])) is None
    assert im.booster_side_layout(dict(stages=[dict(length_m=0,
                                                    diameter_m=1)])) is None
    lay = im.booster_side_layout(dict(stages=[
        dict(length_m=8.0, diameter_m=1.0),
        dict(length_m=4.0, diameter_m=1.0, interstage=True,
             interstage_len_m=0.5),
        dict(length_m=2.0, diameter_m=0.8)]))
    (_, t1), (_, b1) = lay["pts"]["stage1_len"]
    (_, t2), (_, b2) = lay["pts"]["stage2_len"]
    assert (b1 - t1) / (b2 - t2) == pytest.approx(2.0, rel=0.01)
    assert "stage3_len" in lay["pts"]                  # own box, not S1's
    assert lay["subjects"]["stage3_len"] == "s3"
    assert any(p["tag"] == "s3" for p in lay["polys"])
    assert any(p["tag"] == "inter2" for p in lay["polys"])   # declared
    assert not any(p["tag"] == "inter1" for p in lay["polys"])  # not declared
    assert "stage2_interstage_len" in lay["pts"]
    # optional elements draw from their dims when present
    lay = im.booster_side_layout(dict(
        stages=[dict(length_m=8.0, diameter_m=1.0)],
        fairing=dict(length_m=2.0, diameter_m=1.2, nose_len_m=1.0),
        fins=dict(root_m=2.0, span_m=1.0, tip_m=0.5),
        strapon=dict(length_m=5.0, diameter_m=0.6)))
    for f in ("fairing_len", "fairing_nose_len", "fin_tip", "strapon_len",
              im.OVERALL_LEN_CHECK_FIELD):
        assert f in lay["pts"], f
    assert lay["subjects"][im.OVERALL_LEN_CHECK_FIELD] is None


def test_nose_radius_tangency_identity():
    """Forward: a blunted cone with sphere R_N and half-angle θ shows a tip
    of width 2·R_N·cos(θ) (the tangency circle).  The inverse must recover
    the R_N the geometry started from — an identity round-trip, not a fit."""
    for r_n, th in ((0.05, 15.0), (0.30, 45.0), (0.10, 8.2)):
        visible_half = r_n * math.cos(math.radians(th))
        assert im.nose_radius_from_tip_width(visible_half, th) \
            == pytest.approx(r_n)
    # θ = 0 (blunt cylinder) and θ = None (no cone declared) are the plain
    # hemisphere convention — the half-width IS the radius, untouched.
    assert im.nose_radius_from_tip_width(0.05, 0.0) == pytest.approx(0.05)
    assert im.nose_radius_from_tip_width(0.05, None) == pytest.approx(0.05)
    # a near-flat "cone" (θ ≥ 85°) refuses the divide-by-~0 blow-up
    assert im.nose_radius_from_tip_width(0.05, 89.0) == pytest.approx(0.05)


def test_angle_check_note_diagnoses_disagreement():
    ok = im.angle_check_note(45.5, 45.0, "wing sweep")
    assert "agrees" in ok and "DISAGREES" not in ok
    bad = im.angle_check_note(52.0, 45.0, "wing sweep")
    assert "DISAGREES" in bad and "stretch" in bad


def test_symmetry_note_is_a_tilt_detector():
    ok = im.symmetry_note(10.1, 10.0)
    assert "no tilt" in ok
    bad = im.symmetry_note(12.0, 10.0)
    assert "ASYMMETRIC" in bad and "suspicion" in bad


def test_wing_sweep_derives_from_the_tip_chord_length():
    """Sweep is no longer a measured ANGLE — it derives from the planform:
    the RO checklist offers a TIP-CHORD length (0 for a pointed delta), and
    tan Λ = (root − tip)/span.  Delta (tip 0): atan(root/span).  Trapezoid
    (tip > 0): the taper reduces Λ.  A length RATIO → still anchor-free."""
    tip_prompt = next(p for p in im.ro_prompts("axisymmetric")
                      if p["field"] == "_wing_tip_derive")
    assert tip_prompt["convention"] == "wing_tip"
    assert tip_prompt["derives"] == "sweep"
    delta = im.sweep_from_planform(0.53, 0.12, 0.0)
    assert delta == pytest.approx(math.degrees(math.atan2(0.53, 0.12)))
    assert delta == pytest.approx(77.2, abs=0.3)          # the recurring value
    trap = im.sweep_from_planform(0.53, 0.12, 0.20)
    assert trap == pytest.approx(math.degrees(math.atan2(0.33, 0.12)))
    assert trap < delta                                    # tip taper lowers Λ


def test_no_angle_prompts_for_sweep_anywhere():
    """Neither editor asks for a sweep ANGLE any more (the recurring
    complement trap is gone); only the check-only cone flanks remain."""
    ro = [p["field"] for p in im.ro_angle_prompts("axisymmetric")]
    assert ro == [im.FLANK_UPPER_FIELD, im.FLANK_LOWER_FIELD]
    assert "_wing_sweep_var" not in ro
    assert im.ro_angle_prompts("half_cone") == []          # no flanks, no sweep
    assert im.ro_angle_prompts("wedge") == []
    assert im.booster_angle_prompts(has_fins=True) == []


def test_stamp_ignores_the_angle_pseudo_view():
    s = _scale()
    m = im.Measurement("_len_var", *s.measure((0, 0), (200, 0)), scale=s,
                       convention="ro_length", view="side")
    a = im.AngleMeasurement(im.FLANK_UPPER_FIELD, (0, 0), (200, 0), (140, 140))
    stamp = im.provenance_stamp([m, a], s, "2026-08-01")
    assert "views:" not in stamp          # side + angle ≠ two real views
    assert im.FLANK_UPPER_FIELD in stamp


# ── R8: the apply delta view (never silently overwrite) ─────────────────────
def test_apply_deltas_orders_findings_first_and_excludes_checks():
    """Biggest |Δ| first (findings on top), fields with no prior value last,
    and check-only fields (absent from `current`) never appear — they are
    never written."""
    accepted = {"_len_var": 3.3, "_dia_var": 0.58, "_nose_var": 0.02,
                im.OVERALL_LEN_CHECK_FIELD: 10.2}
    current = {"_len_var": 3.0, "_dia_var": 0.575, "_nose_var": None}
    rows = im.apply_deltas(accepted, current)
    assert [r["field"] for r in rows] == ["_len_var", "_dia_var", "_nose_var"]
    assert rows[0]["rel"] == pytest.approx(0.10)
    assert rows[1]["rel"] == pytest.approx(0.58 / 0.575 - 1.0)
    assert rows[2]["rel"] is None                 # blank field → "new"
    assert all(r["field"] != im.OVERALL_LEN_CHECK_FIELD for r in rows)


def test_apply_deltas_zero_current_counts_as_new():
    rows = im.apply_deltas({"_body_span_var": 1.4}, {"_body_span_var": 0.0})
    assert rows[0]["rel"] is None                 # 0 = unset, not a baseline


def test_delta_warn_threshold_is_five_percent():
    assert im.DELTA_WARN_REL == pytest.approx(0.05)


def test_angle_check_decodes_the_swapped_reference_mistake():
    """When measured + derived ≈ 90° (the reference edges were swapped), the
    check names it and states the complement to use — but never writes it
    (offered, not inferred)."""
    note = im.angle_check_note(12.7, 77.2, "wing sweep")
    assert "DISAGREES" in note and "ROOT CHORD" in note
    assert "77.3" in note                     # 90 − 12.7, the value to use
    # an ordinary disagreement (no 90° relationship) keeps the generic verdict
    plain = im.angle_check_note(52.0, 45.0, "wing sweep")
    assert "ROOT CHORD" not in plain and "stretch" in plain


# ── the "what to click" diagrams ────────────────────────────────────────────
def _all_prompts():
    ps = []
    for form in ("axisymmetric", "wedge", "half_cone"):
        ps += im.ro_prompts(form, biconic=(form == "axisymmetric"))
        ps += im.ro_angle_prompts(form)
    ps += im.booster_prompts(n_stages=3, has_fairing=True, has_fins=True,
                             n_fins=4, n_strapons=2)
    ps += im.booster_angle_prompts(has_fins=True)
    return ps


def test_every_prompt_has_a_diagram():
    """No prompt without its picture: every checklist entry maps to a spec
    with a known base, unit-square coordinates, and a click count matching
    its kind (2 for a length, 3 for an angle)."""
    for p in _all_prompts():
        spec = im.diagram_spec(p)
        assert spec is not None, p["field"]
        assert spec["base"] in im.DIAGRAM_BASES, p["field"]
        want = 3 if p.get("angle") else 2
        assert len(spec["pts"]) == want, p["field"]
        assert spec["kind"] == ("angle" if p.get("angle") else "line")
        for x, y in spec["pts"]:
            assert 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0, p["field"]


def test_tip_chord_diagram_points_at_the_drawn_fin_tip():
    """Sweep is derived, not measured; the tip-chord prompt (which drives the
    derivation) shows a segment at the fin tip in the same base art."""
    spec = im.diagram_spec(dict(field="_wing_tip_derive"))
    assert spec["kind"] == "line" and spec["base"] == "ro_side"
    fin_r = next(it for it in im.DIAGRAM_BASES["ro_side"]
                 if it["tag"] == "fin_r")
    assert spec["pts"][0] == fin_r["pts"][1]             # fin's tip vertex


def test_unknown_field_has_no_diagram_rather_than_a_wrong_one():
    assert im.diagram_spec(dict(field="no_such_field")) is None
    assert im.diagram_spec(None) is None


# ── interstage: length is the one image-measurable dimension ────────────────
def test_interstage_prompt_only_for_declared_stages():
    """The interstage adapter's LENGTH is offered only for stages whose
    interstage is declared (its ⌀ is inherited, its mass/jettison aren't
    dimensions).  It carries the stage_length convention so it tiles the
    closure sum."""
    none = [p["field"] for p in im.booster_prompts(n_stages=2)]
    assert not any("interstage" in f for f in none)
    ps = im.booster_prompts(n_stages=2, interstage_stages=(1,))
    fields = [p["field"] for p in ps]
    assert "stage1_interstage_len" in fields
    assert "stage2_interstage_len" not in fields
    inter = next(p for p in ps if p["field"] == "stage1_interstage_len")
    assert inter["convention"] == "stage_length"      # counts toward closure
    # order: the interstage follows its stage's len/dia
    assert fields.index("stage1_interstage_len") == fields.index("stage1_dia") + 1


def test_interstage_has_its_own_diagram_at_the_stage_top():
    spec = im.diagram_spec(dict(field="stage1_interstage_len"))
    assert spec is not None and spec["base"] == "booster" and spec["kind"] == "line"
    # a short segment straddling the S1/S2 junction (y≈0.44), distinct from
    # the stage-1 length segment (which spans the whole S1 box)
    assert spec["pts"] != im.diagram_spec(dict(field="stage1_len"))["pts"]
    s2 = im.diagram_spec(dict(field="stage2_interstage_len"))
    assert s2["pts"] != spec["pts"]                   # different stage, own spot


# ── nose/fairing shapes: measurements are shape-agnostic; art is shape-aware ─
def test_fairing_nose_segment_length_is_measurable_and_not_double_counted():
    """The fairing's nose-segment length (where the ogive/cone meets the
    cylinder) is its own prompt — the SHAPE is declared, but where the curve
    ends is measurable and shape-independent.  It is part of the total
    fairing length, so it must NOT be a separate closure segment."""
    ps = im.booster_prompts(n_stages=1, has_fairing=True)
    fields = [p["field"] for p in ps]
    assert "fairing_nose_len" in fields
    nose = next(p for p in ps if p["field"] == "fairing_nose_len")
    assert nose["convention"] == "fairing_nose_length"
    assert "fairing_nose_len" not in im.closure_segments(ps)   # not additive
    assert "fairing_len" in im.closure_segments(ps)            # the total is
    assert "fairing_nose_len" not in [p["field"]
                                      for p in im.booster_prompts(n_stages=1)]


@pytest.mark.parametrize("shape", ["tangent_ogive", "von_karman", "lv_haack",
                                   "parabola", "blunt_cylinder"])
def test_ro_side_base_follows_the_declared_profile(shape):
    """Every non-cone profile (incl. Sears-Haack = lv_haack) draws a CURVED
    nose outline — more vertices than the straight cone — so the diagram is
    honest about the shape.  The measured dimensions are unchanged; only the
    picture differs."""
    cone = im.ro_side_base("cone")[0]["pts"]
    curved = im.ro_side_base(shape)[0]["pts"]
    assert len(curved) > len(cone) + 4         # a real curve, not two edges
    assert curved != cone
    # a biconic keeps its own two-cone outline (its break-⌀ prompt owns that)
    assert im.ro_side_base(shape, biconic=True) is im.DIAGRAM_BASES["ro_side"]


def test_cone_and_ogive_share_the_same_measured_dimensions():
    """The validation point: nothing shape-specific is MEASURED — a cone and
    a Sears-Haack RO present the identical prompt list (length, ⌀, nose r,
    wings); the profile is the declared shape dropdown, not a click."""
    fields = lambda: [p["field"] for p in im.ro_prompts("axisymmetric")]
    assert fields() == ["_len_var", "_dia_var", "_nose_var",
                        "_wing_root_var", "_wing_span_var", "_wing_tip_derive"]


def test_conical_stage_adds_a_top_diameter_prompt():
    """A conical (tapered) stage is a frustum — its TOP diameter is a free
    parameter distinct from the base.  Offered only for declared conical
    stages; the stage's own dia prompt relabels to BASE.  Not a closure
    segment (diameters don't tile length)."""
    plain = [p["field"] for p in im.booster_prompts(n_stages=2)]
    assert not any("_top_dia" in f for f in plain)
    ps = im.booster_prompts(n_stages=2, conical_stages=(1,))
    fields = [p["field"] for p in ps]
    assert "stage1_top_dia" in fields and "stage2_top_dia" not in fields
    top = next(p for p in ps if p["field"] == "stage1_top_dia")
    assert top["convention"] == "stage_diameter"
    base = next(p for p in ps if p["field"] == "stage1_dia")
    assert "BASE" in base["label"]                # relabeled for the taper
    # order: top ⌀ follows the base ⌀
    assert fields.index("stage1_top_dia") == fields.index("stage1_dia") + 1
    assert "stage1_top_dia" not in im.closure_segments(ps)


def test_conical_top_diameter_has_its_own_diagram_at_the_stage_top():
    spec = im.diagram_spec(dict(field="stage1_top_dia"))
    assert spec["base"] == "booster" and spec["kind"] == "line"
    base = im.diagram_spec(dict(field="stage1_dia"))["pts"]
    assert spec["pts"] != base                    # top edge, not the base line
