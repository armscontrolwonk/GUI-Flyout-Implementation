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
    dlg._body_form_var.set(dlg._BODY_FORM_LABELS[form])
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
