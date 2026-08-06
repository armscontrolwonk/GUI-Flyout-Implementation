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
    fin_tip = im.DIAGRAM_BASES["ro_side"][2][1]   # the fin's tip vertex
    assert spec["pts"][0] == fin_tip


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
    cone = im.ro_side_base("cone")[0]
    curved = im.ro_side_base(shape)[0]
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
