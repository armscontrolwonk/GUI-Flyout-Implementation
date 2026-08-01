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
