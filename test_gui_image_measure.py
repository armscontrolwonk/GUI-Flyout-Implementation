"""Image dimensioning tool — GUI wiring (Phase A).

The measurement math is covered by test_image_measure.py; these tests cover
the editor-side wiring the pure core can't reach: the dialog builds, the
"Measure from image…" button exists, and the apply path writes accepted
values into the editor fields and stamps the notes (the tool's only durable
outputs).  The canvas click plumbing needs a real event loop and is verified
by hand, not here.

Skips cleanly where tkinter / a display is unavailable.
"""

import json

import pytest

pytest.importorskip("tkinter", reason="no Tk in this interpreter")

import matplotlib
matplotlib.use("Agg")
import tkinter as tk

import thrusty
import image_measure as im
from booster_models import ro_from_dict


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    r.withdraw()
    yield r
    r.destroy()


def _editor(root):
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    dlg = thrusty.ROEditorDialog(root, ro=ro)
    dlg.withdraw()
    return dlg


def _all(w, acc=None):
    acc = [] if acc is None else acc
    for c in w.winfo_children():
        acc.append(c); _all(c, acc)
    return acc


def test_measure_button_exists(root):
    dlg = _editor(root)
    labels = [b.cget("text") for b in _all(dlg)
              if isinstance(b, tk.ttk.Button)]
    assert any("Measure from image" in t for t in labels)


def test_apply_writes_fields_and_stamps_notes(root):
    """The tool's durable outputs: accepted values land in the editor fields,
    and a dimensional-draft stamp is appended to notes.  Refused measurements
    (below the resolution floor) never reach apply, so nothing about them is
    written."""
    pytest.importorskip("PIL")
    dlg = _editor(root)
    s = im.Scale((0, 0), (1000, 0), 10.0, anchor_note="claimed L=10 m")
    length = im.Measurement("_len_var", *s.measure((0, 0), (200, 0)), scale=s,
                            convention="ro_length")          # 2.0 m
    diam = im.Measurement("_dia_var", *s.measure((0, 0), (58, 0)), scale=s,
                          convention="ro_diameter")          # 0.58 m
    accepted = {"_len_var": length.value_m, "_dia_var": diam.value_m}
    dlg._apply_image_measurements(accepted, [length, diam], s)

    assert float(dlg._len_var.get()) == pytest.approx(2.0)
    assert float(dlg._dia_var.get()) == pytest.approx(0.58)
    notes = dlg._notes_text.get("1.0", "end-1c")
    assert "dimensional draft from image" in notes
    assert "claimed L=10 m" in notes and "1 px = 1 cm" in notes


def test_apply_maps_body_span_for_wedge(root):
    dlg = _editor(root)
    dlg._shape_var.set(dlg._BODY_FORM_LABELS["wedge"])
    dlg._update_body_form_state()
    s = im.Scale((0, 0), (1000, 0), 10.0)
    span = im.Measurement("_body_span_var", *s.measure((0, 0), (90, 0)),
                          scale=s, view="plan", convention="wedge_span")  # 0.9 m
    dlg._apply_image_measurements({"_body_span_var": span.value_m}, [span], s)
    assert float(dlg._body_span_var.get()) == pytest.approx(0.9)


def test_apply_maps_biconic_and_wing_planform(root):
    """The RO tool covers the editor's FULL dimensional field set: biconic
    break geometry and the wing planform land in their fields, and the wing
    S/AR derivation fires from the written planform (measure the planform,
    derive the area — never the other way)."""
    dlg = _editor(root)
    dlg._glider_var.set(True)
    dlg._biconic_var.set(True); dlg._update_biconic_state()
    acc = {"_fore_len_var": 1.1, "_break_dia_var": 0.3,
           "_wing_root_var": 0.7, "_wing_span_var": 0.45}
    dlg._apply_image_measurements(acc, [], None)
    assert float(dlg._fore_len_var.get()) == pytest.approx(1.1)
    assert float(dlg._break_dia_var.get()) == pytest.approx(0.3)
    assert float(dlg._wing_root_var.get()) == pytest.approx(0.7)
    assert float(dlg._wing_span_var.get()) == pytest.approx(0.45)
    assert float(dlg._wing_area_var.get()) > 0.0     # S derived from planform


def test_apply_with_nothing_accepted_is_a_noop(root):
    dlg = _editor(root)
    before_len = dlg._len_var.get()
    before_notes = dlg._notes_text.get("1.0", "end-1c")
    dlg._apply_image_measurements({}, [], None)
    assert dlg._len_var.get() == before_len
    assert dlg._notes_text.get("1.0", "end-1c") == before_notes


def test_booster_button_and_apply(root):
    """A2: the booster editor gets the same tool.  Its apply writes stage,
    fairing, one-fin and one-strap-on GEOMETRY into the existing fields while
    leaving the declared COUNTS untouched (measure-one-declare-count — the
    model replicates)."""
    d = thrusty.BoosterDialog(root, on_save=lambda *a, **k: None)
    d.withdraw()
    assert any("Measure from image" in b.cget("text")
               for b in _all(d) if isinstance(b, tk.ttk.Button))
    d._n_stages_var.set("2"); d._update_stage_frames()
    d._shroud_var.set(True); d._update_shroud_state()
    d._fins_var.set(True); d._update_fins_state(); d._fin_n_var.set("4")
    d._n_boosters_var.set("4")
    s = im.Scale((0, 0), (1000, 0), 10.0, anchor_note="L=10 m")
    acc = {"stage1_len": 9.0, "stage2_dia": 0.88, "fairing_len": 2.6,
           "fin_span": 0.5, "strapon_dia": 1.2}
    d._apply_image_measurements(acc, [], s)
    assert float(d._stage_frames[0]._length.get()) == pytest.approx(9.0)
    assert float(d._stage_frames[1]._dia.get()) == pytest.approx(0.88)
    assert float(d._shroud_length_var.get()) == pytest.approx(2.6)
    assert float(d._fin_span_var.get()) == pytest.approx(0.5)
    assert float(d._b_diam_var.get()) == pytest.approx(1.2)
    assert d._fin_n_var.get() == "4"          # declared count untouched
    d.destroy()


def test_booster_apply_ignores_the_check_only_total(root):
    """The overall-length cross-check measurement exists only to feed the
    closure warning: apply must not write it anywhere (there is no editor
    field for a derived total)."""
    d = thrusty.BoosterDialog(root, on_save=lambda *a, **k: None)
    d.withdraw()
    d._n_stages_var.set("1"); d._update_stage_frames()
    before = d._stage_frames[0]._length.get()
    d._apply_image_measurements({im.OVERALL_LEN_CHECK_FIELD: 10.2}, [], None)
    assert d._stage_frames[0]._length.get() == before
    d.destroy()


def test_clocking_control_present_for_fins(root):
    """R1: when the declared topology has fins, the shared dialog carries the
    clocking selector (the cos45 correction the pure core already applies) with
    the do-nothing option first.  The RO dialog, whose prompts have no span a
    ×-roll foreshortens, must not offer it."""
    pytest.importorskip("PIL")
    d = thrusty.BoosterDialog(root, on_save=lambda *a, **k: None)
    d.withdraw()
    d._fins_var.set(True); d._update_fins_state(); d._fin_n_var.set("4")
    opened = []
    orig = tk.Toplevel
    tk.Toplevel = lambda *a, **k: (lambda w: (opened.append(w), w)[1])(orig(*a, **k))
    try:
        d._measure_from_image()
    finally:
        tk.Toplevel = orig
    combos = [w for w in _all(opened[-1]) if isinstance(w, tk.ttk.Combobox)]
    clock = [c for c in combos
             if any("×-rolled" in v for v in c.cget("values"))]
    assert clock, "fin topology must expose the clocking selector"
    assert im.CLOCKING_OPTIONS[0][0] in clock[0].cget("values")
    opened[-1].destroy(); d.destroy()

    # RO editor with NO wings declared (Maneuvering off): no clocking-sensitive
    # prompt → the selector is not built at all.
    dlg = _editor(root)
    dlg._glider_var.set(False)
    opened2 = []
    tk.Toplevel = lambda *a, **k: (lambda w: (opened2.append(w), w)[1])(orig(*a, **k))
    try:
        dlg._measure_from_image()
    finally:
        tk.Toplevel = orig
    ro_combos = [w for w in _all(opened2[-1]) if isinstance(w, tk.ttk.Combobox)]
    assert not any("×-rolled" in v for c in ro_combos for v in c.cget("values"))
    opened2[-1].destroy()


def _open_measure_dialog(dlg):
    opened = []
    orig = tk.Toplevel
    tk.Toplevel = lambda *a, **k: (lambda w: (opened.append(w), w)[1])(orig(*a, **k))
    try:
        dlg._measure_from_image()
    finally:
        tk.Toplevel = orig
    return opened[-1]


def test_paste_and_new_image_resets_scale(root, tmp_path, monkeypatch):
    """Paste accepts both a raw clipboard image and a copied-file list, and
    loading ANY new image resets the scale — metres-per-pixel belongs to the
    image it was anchored on; carrying it to a different picture would be
    silently wrong."""
    pytest.importorskip("PIL")
    from PIL import Image
    d = _open_measure_dialog(_editor(root))
    side = d._im_views["side"]
    # ⌘V of a raw clipboard image (a screenshot)
    monkeypatch.setattr("PIL.ImageGrab.grabclipboard",
                        lambda: Image.new("RGB", (300, 200), "white"))
    d._im_paste()
    assert side["img"] is not None
    # anchor a scale, then load a NEW image → scale must clear
    side["scale"] = im.Scale((0, 0), (100, 0), 1.0)
    p = tmp_path / "v.png"
    Image.new("RGB", (400, 150), "gray").save(p)
    d._im_load_path(str(p))
    assert side["scale"] is None
    assert side["img"].size == (400, 150)
    # ⌘V of a copied FILE (Finder copy) → the file loads
    monkeypatch.setattr("PIL.ImageGrab.grabclipboard", lambda: [str(p)])
    side["img"] = None
    d._im_paste()
    assert side["img"] is not None
    d.destroy()


def test_paste_button_and_opportunistic_dnd(root):
    """The Paste button is always there; drag-and-drop is enabled exactly when
    the OPTIONAL tkinterdnd2 package is importable (no hard dependency)."""
    pytest.importorskip("PIL")
    d = _open_measure_dialog(_editor(root))
    btxt = [b.cget("text") for b in _all(d) if isinstance(b, tk.ttk.Button)]
    assert any("Paste image" in t for t in btxt)
    try:
        import tkinterdnd2                     # noqa: F401
        assert d._im_dnd is True
    except ImportError:
        assert d._im_dnd is False
    d.destroy()


def test_type_value_needs_no_image_or_scale(root, monkeypatch):
    """The checklist never forces a click: a known dimension can be TYPED for
    the selected prompt with no image loaded and no scale set (only Measure
    needs those).  The value lands in accepted, recorded as hand-entered so
    the stamp cannot claim it was measured."""
    pytest.importorskip("PIL")
    d = _open_measure_dialog(_editor(root))
    import tkinter.simpledialog as sd
    monkeypatch.setattr(sd, "askfloat", lambda *a, **k: 0.58)
    d._im_type_value()
    st = d._im_state
    assert d._im_views["side"]["img"] is None             # truly ungated:
    assert d._im_views["side"]["scale"] is None           # no image, no scale
    assert list(st["accepted"].values()) == [pytest.approx(0.58)]
    assert getattr(st["measurements"][0], "hand_entered", False) is True
    d.destroy()


def _wedge_editor(root):
    dlg = _editor(root)
    dlg._shape_var.set(dlg._BODY_FORM_LABELS["wedge"])
    dlg._update_body_form_state()
    return dlg


def test_multiview_gating_and_per_view_scales(root, tmp_path):
    """Phase B: the wedge checklist needs side + plan.  The dialog gets a view
    selector; each view carries its OWN image and scale; a plan-view prompt is
    HARD-GATED on the plan view being loaded and scaled (the old label-only
    warning let a span be clicked off a side elevation — pure garbage, the
    span runs into the page); and the side view's scale survives plan loads."""
    pytest.importorskip("PIL")
    from PIL import Image
    d = _open_measure_dialog(_wedge_editor(root))
    assert set(d._im_views) == {"side", "plan"}

    # side view: load + scale
    ps = tmp_path / "side.png"; Image.new("RGB", (400, 100), "gray").save(ps)
    d._im_load_path(str(ps))
    d._im_views["side"]["scale"] = im.Scale((0, 0), (400, 0), 8.0)

    # select the plan-only span prompt and try to measure with NO plan image:
    # must refuse to arm
    span_label = next(lab for lab, p in
                      [(f"{p['field']}  —  {p['label']}", p)
                       for p in im.ro_prompts("wedge")]
                      if "_body_span_var" in lab)
    d._im_prompt_var.set(span_label)
    d._im_begin_measure()
    assert d._im_state["mode"] == "idle"          # refused: no plan image
    assert d._im_state["cur"] == "plan"           # but auto-switched view

    # load the plan image: side's scale must be untouched, plan's is its own
    pp = tmp_path / "plan.png"; Image.new("RGB", (500, 300), "gray").save(pp)
    d._im_load_path(str(pp))                       # loads into CURRENT (plan)
    assert d._im_views["plan"]["img"].size == (500, 300)
    assert d._im_views["side"]["scale"] is not None   # untouched
    d._im_begin_measure()
    assert d._im_state["mode"] == "idle"          # still refused: no plan scale
    d._im_views["plan"]["scale"] = im.Scale((0, 0), (250, 0), 8.0)
    d._im_begin_measure()
    assert d._im_state["mode"] == "measure"       # armed at last
    d.destroy()


def test_single_view_dialog_has_no_view_selector(root):
    pytest.importorskip("PIL")
    d = _open_measure_dialog(_editor(root))       # axisymmetric: side only
    assert set(d._im_views) == {"side"}
    assert not any(isinstance(w, tk.ttk.Radiobutton) for w in _all(d))
    d.destroy()


def test_zoom_never_touches_measurements(root, tmp_path):
    """Zoom is display-only: clicks are stored in ORIGINAL-image pixels, so a
    measurement's value and quantum are identical at any zoom."""
    pytest.importorskip("PIL")
    from PIL import Image
    d = _open_measure_dialog(_editor(root))
    p = tmp_path / "v.png"; Image.new("RGB", (400, 150), "gray").save(p)
    d._im_load_path(str(p))
    side = d._im_views["side"]
    side["scale"] = im.Scale((0, 0), (400, 0), 10.0)
    d._im_state["clicks"] = [(0.0, 0.0), (200.0, 0.0)]
    before = list(d._im_state["clicks"])
    z0 = side["zoom"]
    d._im_zoom_at(2.0)
    assert side["zoom"] == pytest.approx(z0 * 2.0)
    assert d._im_state["clicks"] == before        # image-px clicks unmoved
    assert side["scale"].m_per_px == pytest.approx(0.025)   # 10 m / 400 px
    d._im_fit()
    assert side["zoom"] == pytest.approx(z0)
    d.destroy()


def test_accept_records_overlay_annotation(root, tmp_path):
    """The overlay audits what was clicked: accepting a measurement stores its
    clicked segment (view-tagged, original-image px) for drawing."""
    pytest.importorskip("PIL")
    from PIL import Image
    d = _open_measure_dialog(_editor(root))
    p = tmp_path / "v.png"; Image.new("RGB", (400, 150), "gray").save(p)
    d._im_load_path(str(p))
    d._im_views["side"]["scale"] = im.Scale((0, 0), (400, 0), 10.0)
    st = d._im_state
    st["prompt"] = im.ro_prompts("axisymmetric")[0]        # _len_var, side
    st["clicks"] = [(50.0, 40.0), (250.0, 40.0)]
    st["_finish_measure"]()                                 # proposes 5.0 m
    acc = [b for b in _all(d) if isinstance(b, tk.ttk.Button)
           and b.cget("text") == "Accept"][0]
    acc.invoke()
    assert st["accepted"]["_len_var"] == pytest.approx(5.0)
    view, p1, p2, label = st["annotations"]["_len_var"]
    assert view == "side" and p1 == (50.0, 40.0) and p2 == (250.0, 40.0)
    assert "5" in label
    d.destroy()


def test_dialog_builds_without_error(root):
    """Smoke: the Toplevel and all its widgets construct (catches layout/closure
    errors the apply-path test skips).  Pillow present → real dialog."""
    pytest.importorskip("PIL")
    dlg = _editor(root)
    opened = []
    orig = tk.Toplevel
    tk.Toplevel = lambda *a, **k: (lambda w: (opened.append(w), w)[1])(orig(*a, **k))
    try:
        dlg._measure_from_image()
    finally:
        tk.Toplevel = orig
    assert opened and "Measure from image" in opened[-1].title()
    # a canvas and the Load/Set-scale/Measure/Apply buttons are present
    ws = _all(opened[-1])
    assert any(isinstance(w, tk.Canvas) for w in ws)
    btxt = [b.cget("text") for b in ws if isinstance(b, tk.ttk.Button)]
    for want in ("Load image…", "Set scale…", "Measure", "Apply to editor"):
        assert any(want in t for t in btxt), want
    opened[-1].destroy()


def test_angle_measure_accept_and_apply(root, tmp_path):
    """Angles end-to-end: a winged RO's checklist carries the sweep angle
    prompt; the 3-click finish proposes degrees with NO scale set (anchor-
    free); Accept records it; Apply writes the DEGREES field.  The flank
    check fields are check-only and never reach the editor."""
    pytest.importorskip("PIL")
    from PIL import Image
    dlg = _editor(root)
    dlg._glider_var.set(True)
    d = _open_measure_dialog(dlg)
    st = d._im_state
    p = tmp_path / "v.png"; Image.new("RGB", (400, 200), "gray").save(p)
    d._im_load_path(str(p))
    assert d._im_views["side"]["scale"] is None            # no scale on purpose
    ang = [q for q in im.ro_angle_prompts("axisymmetric", winged=True)
           if q["field"] == "_wing_sweep_var"][0]
    st["prompt"] = ang
    st["clicks"] = [(0.0, 0.0), (200.0, 0.0), (140.0, 140.0)]   # 45°
    st["_finish_measure"]()
    acc = [b for b in _all(d) if isinstance(b, tk.ttk.Button)
           and b.cget("text") == "Accept"][0]
    acc.invoke()
    assert st["accepted"]["_wing_sweep_var"] == pytest.approx(45.0)
    dlg._apply_image_measurements(
        {"_wing_sweep_var": st["accepted"]["_wing_sweep_var"],
         im.FLANK_UPPER_FIELD: 10.0}, st["measurements"], None)
    assert float(dlg._wing_sweep_var.get()) == pytest.approx(45.0)
    assert not hasattr(dlg, im.FLANK_UPPER_FIELD)          # check-only: no var
    d.destroy()


def test_angle_check_line_flags_disagreement(root, tmp_path):
    """The identity twin fires once the lengths arrive: a measured sweep that
    contradicts the accepted planform turns the check line red-worded
    (DISAGREES) — warn-only, nothing is corrected."""
    pytest.importorskip("PIL")
    from PIL import Image
    dlg = _editor(root)
    dlg._glider_var.set(True)
    d = _open_measure_dialog(dlg)
    st = d._im_state
    p = tmp_path / "v.png"; Image.new("RGB", (400, 200), "gray").save(p)
    d._im_load_path(str(p))
    st["accepted"].update({"_wing_root_var": 1.0, "_wing_span_var": 1.0})
    ang = [q for q in im.ro_angle_prompts("axisymmetric", winged=True)
           if q["field"] == "_wing_sweep_var"][0]
    st["prompt"] = ang
    st["clicks"] = [(0.0, 0.0), (200.0, 0.0), (100.0, 173.2)]   # 60° ≠ 45°
    st["_finish_measure"]()
    acc = [b for b in _all(d) if isinstance(b, tk.ttk.Button)
           and b.cget("text") == "Accept"][0]
    acc.invoke()
    texts = [str(w.cget("text")) for w in _all(d)
             if isinstance(w, tk.ttk.Label)]
    joined = " ".join(texts)
    # the check line is a textvariable label; read it via the recorded vars
    st["_refresh_closure"]()
    lbls = [w for w in _all(d) if isinstance(w, tk.ttk.Label)]
    var_texts = []
    for w in lbls:
        tv = str(w.cget("textvariable"))
        if tv:
            try:
                var_texts.append(str(w.tk.globalgetvar(tv)))
            except Exception:
                pass
    assert any("DISAGREES" in t for t in var_texts + texts)
    d.destroy()


def test_apply_shows_delta_preview_and_writes_only_on_confirm(root):
    """R8 end-to-end: Apply opens the old-vs-new preview instead of writing;
    Back leaves every field untouched; Write commits.  The check-only total
    is counted as audit-only, never listed as a write."""
    pytest.importorskip("PIL")
    dlg = _editor(root)
    dlg._len_var.set("3.0")
    d = _open_measure_dialog(dlg)
    d._im_state["accepted"] = {"_len_var": 3.3}
    apply_btn = [b for b in _all(d) if isinstance(b, tk.ttk.Button)
                 and "Apply to editor" in b.cget("text")][0]

    opened = []
    orig = tk.Toplevel
    tk.Toplevel = lambda *a, **k: (lambda w: (opened.append(w), w)[1])(orig(*a, **k))
    try:
        apply_btn.invoke()
    finally:
        tk.Toplevel = orig
    pv = opened[-1]
    assert "Apply preview" in pv.title()
    assert dlg._len_var.get() == "3.0"            # nothing written yet
    texts = [str(w.cget("text")) for w in _all(pv)
             if isinstance(w, tk.ttk.Label)]
    assert any("+10.0%" in t for t in texts)      # the delta is shown
    # Back: no write
    [b for b in _all(pv) if isinstance(b, tk.ttk.Button)
     and b.cget("text") == "Back"][0].invoke()
    assert dlg._len_var.get() == "3.0"
    # Apply again and confirm: written
    tk.Toplevel = lambda *a, **k: (lambda w: (opened.append(w), w)[1])(orig(*a, **k))
    try:
        apply_btn.invoke()
    finally:
        tk.Toplevel = orig
    pv2 = opened[-1]
    [b for b in _all(pv2) if isinstance(b, tk.ttk.Button)
     and "Write" in b.cget("text")][0].invoke()
    assert float(dlg._len_var.get()) == pytest.approx(3.3)
