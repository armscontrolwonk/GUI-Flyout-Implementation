"""β-estimator dialog routing by body form (Phase 2c GUI wiring).

The estimator dialog is presentation over booster_models.lifting_body_sweep()
and cd_cone_hypersonic(); these tests check the WIRING the pure-function tests
can't reach: that a lifting body opens the α-sweep estimator, an axisymmetric
body opens the cone build-up, and that "Use β and L/D" writes both the β and
the glider-L/D fields from one consistent trim row.

Skips cleanly where tkinter / a display is unavailable.
"""

import json

import pytest

pytest.importorskip("tkinter", reason="no Tk in this interpreter")

import matplotlib
matplotlib.use("Agg")
import tkinter as tk

import thrusty
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


def _editor(root, form):
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    dlg = thrusty.ROEditorDialog(root, ro=ro)
    dlg.withdraw()
    # merged Shape selector: a lifting form is its own entry; 'axisymmetric'
    # is any nose profile (Cone here).
    label = (dlg._BODY_FORM_LABELS[form] if form in ("wedge", "half_cone")
             else thrusty.NOSE_SHAPE_LABELS["cone"])
    dlg._shape_var.set(label)
    dlg._update_body_form_state()
    dlg._mass_var.set("900"); dlg._len_var.set("3.6"); dlg._dia_var.set("0.5")
    return dlg


def _capture_dialog(dlg):
    """Invoke _calc_beta and return the Toplevel it creates."""
    opened = []
    orig = tk.Toplevel
    tk.Toplevel = lambda *a, **k: (lambda w: (opened.append(w), w)[1])(orig(*a, **k))
    try:
        dlg._calc_beta()
    finally:
        tk.Toplevel = orig
    return opened[-1]


def _all_widgets(w, acc=None):
    acc = [] if acc is None else acc
    for c in w.winfo_children():
        acc.append(c); _all_widgets(c, acc)
    return acc


def test_axisymmetric_opens_the_cone_builder(root):
    dlg = _editor(root, "axisymmetric")
    sub = _capture_dialog(dlg)
    assert sub.title() == "Estimate Object β"


@pytest.mark.parametrize("form,frag", [("wedge", "wedge"),
                                       ("half_cone", "half-cone")])
def test_lifting_forms_open_the_alpha_sweep_estimator(root, form, frag):
    dlg = _editor(root, form)
    sub = _capture_dialog(dlg)
    assert "L/D" in sub.title() and frag in sub.title()


def test_use_writes_both_beta_and_ld_from_one_trim_row(root):
    dlg = _editor(root, "wedge")
    sub = _capture_dialog(dlg)
    ents = [w for w in _all_widgets(sub) if isinstance(w, tk.ttk.Entry)]
    # rows: mass, length, depth, span, mach, Re, T_w  → span is index 3
    ents[3].insert(0, "0.9")                 # REQUIRED planform span
    sub.update_idletasks()
    use = next(w for w in _all_widgets(sub)
               if isinstance(w, tk.ttk.Button) and w.cget("text").startswith("Use"))
    use.invoke()
    beta = float(dlg._beta_var.get())
    ld = float(dlg._LD_var.get())
    assert beta > 0.0                        # zero-lift β written
    assert 2.0 < ld < 6.0                    # (L/D)max written, sharp-body band


def test_merged_shape_selector_sets_both_shape_and_body_form(root):
    """One selector, two data fields: an axisymmetric nose profile implies a
    body of revolution; a lifting-body entry sets body_form and keeps a
    (moot) cone nose."""
    import dataclasses
    dlg = _editor(root, "axisymmetric")
    dlg._shape_var.set(thrusty.NOSE_SHAPE_LABELS["tangent_ogive"])
    dlg._update_body_form_state()
    ro = dlg._build_ro()
    assert ro.shape == "tangent_ogive" and ro.body_form == "axisymmetric"
    dlg._shape_var.set(dlg._BODY_FORM_LABELS["wedge"])
    dlg._update_body_form_state()
    ro2 = dlg._build_ro()
    assert ro2.body_form == "wedge" and ro2.shape == "cone"


def test_lifting_form_wins_the_label_on_reopen_and_disables_biconic(root):
    """Reopening a stored (shape, body_form): a lifting body_form shows its own
    label regardless of the stored nose shape, and disables the biconic tick;
    an axisymmetric body shows its nose profile."""
    import dataclasses
    dlg = _editor(root, "wedge")
    wedge_ro = dataclasses.replace(dlg._build_ro(), shape="tangent_ogive")
    reopened = thrusty.ROEditorDialog(root, ro=wedge_ro); reopened.withdraw()
    assert reopened._body_form_key() == "wedge"
    assert reopened._shape_var.get() == reopened._BODY_FORM_LABELS["wedge"]
    assert str(reopened._biconic_chk.cget("state")) == "disabled"
    # an ogive-nosed round body shows the ogive and re-enables biconic
    og = dataclasses.replace(wedge_ro, body_form="axisymmetric", shape="tangent_ogive")
    r2 = thrusty.ROEditorDialog(root, ro=og); r2.withdraw()
    assert r2._shape_key() == "tangent_ogive" and r2._body_form_key() == "axisymmetric"
    assert str(r2._biconic_chk.cget("state")) == "normal"


def test_missing_required_span_yields_no_result(root):
    """Span is required for the wedge; without it the dialog must refuse to
    produce a β (the derive-don't-invent rule — no fabricated planform)."""
    dlg = _editor(root, "wedge")
    before = dlg._beta_var.get()
    sub = _capture_dialog(dlg)               # span left at default 0
    use = next(w for w in _all_widgets(sub)
               if isinstance(w, tk.ttk.Button) and w.cget("text").startswith("Use"))
    use.invoke()
    assert dlg._beta_var.get() == before     # unchanged — nothing written
