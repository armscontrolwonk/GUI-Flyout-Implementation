"""Flight Plan dialog wiring for the α envelope + induced-drag toggle.

The dogleg lives in the Flight Plan dialog (the sidebar yaw controls were
removed), so the α limit and induced-drag toggle must be editable THERE and
travel with the plan.  These tests pin that wiring, which the pure-trajectory
tests (test_alpha_loads.py) can't reach:

  * the dialog SAVES alpha_limit_deg / alpha_induced_drag into the plan;
  * reopening the dialog on that plan REPOPULATES the fields;
  * a blank α limit saves as None (no limit);
  * the run-path readers (_alpha_limit_value / _alpha_induced_value) honor
    the surfaced values WITHOUT gating on the removed sidebar yaw checkbox.

Skips cleanly where tkinter / a display is unavailable.
"""

import pytest

pytest.importorskip("tkinter", reason="no Tk in this interpreter")

import matplotlib
matplotlib.use("Agg")
import tkinter as tk

import thrusty
from booster_models import get_booster, load_booster_library

load_booster_library()


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display: {e}")
    r.withdraw()
    yield r
    r.destroy()


def _plan():
    return {'guidance': 'pitch_program', 'launch_elevation_deg': 90.0,
            'yaw_maneuvers': [[55.0, 56.0, 200.0]], 'stages': [{}, {}]}


def test_dialog_defaults_alpha_limit_to_10(root):
    """A plan that never set alpha_limit_deg opens with the 10° default; a
    plan that stored None (user cleared it) stays blank."""
    b = get_booster("AUR")
    d1 = thrusty.FlightPlanDialog(root, "AUR",
                                  {'guidance': 'pitch_program', 'stages': [{}, {}]}, b)
    assert d1._alpha_limit_var.get() == "10"
    d1.destroy()
    d2 = thrusty.FlightPlanDialog(root, "AUR",
                                  {'guidance': 'pitch_program', 'stages': [{}, {}],
                                   'alpha_limit_deg': None}, b)
    assert d2._alpha_limit_var.get() == ""
    d2.destroy()


def test_dialog_saves_alpha_fields(root):
    b = get_booster("AUR")
    dlg = thrusty.FlightPlanDialog(root, "AUR", _plan(), b)
    dlg._alpha_limit_var.set("8")
    dlg._alpha_induced_var.set(True)
    dlg._save()
    out = dlg.result
    assert out['alpha_limit_deg'] == 8.0
    assert out['alpha_induced_drag'] is True
    # The dogleg it governs is preserved alongside.
    assert out['yaw_maneuvers'] == [[55.0, 56.0, 200.0]]


def test_dialog_reopen_repopulates_fields(root):
    b = get_booster("AUR")
    dlg = thrusty.FlightPlanDialog(root, "AUR", _plan(), b)
    dlg._alpha_limit_var.set("6.5")
    dlg._alpha_induced_var.set(True)
    dlg._save()
    saved = dlg.result

    dlg2 = thrusty.FlightPlanDialog(root, "AUR", saved, b)
    assert float(dlg2._alpha_limit_var.get()) == 6.5
    assert dlg2._alpha_induced_var.get() is True
    dlg2.destroy()


def test_dialog_blank_limit_saves_none(root):
    b = get_booster("AUR")
    dlg = thrusty.FlightPlanDialog(root, "AUR", _plan(), b)
    dlg._alpha_limit_var.set("")            # blank = no limit
    dlg._alpha_induced_var.set(False)
    dlg._save()
    out = dlg.result
    assert out['alpha_limit_deg'] is None
    assert out['alpha_induced_drag'] is False


def test_run_readers_not_gated_on_removed_yaw_checkbox(root):
    """The run path must read the surfaced α values even though the sidebar
    yaw checkbox (its old gate) is gone / False."""
    app = thrusty.BoosterFlyoutApp()
    try:
        app.withdraw()
        # Transfer-bus vars as _on_booster_changed would set them from a
        # dialog-saved plan; the removed checkbox stays False.
        app._adv_yaw_var.set(False)
        app._alpha_limit_var.set("8")
        app._alpha_induced_var.set(True)
        assert app._alpha_limit_value() == 8.0
        assert app._alpha_induced_value() is True
        app._alpha_limit_var.set("")        # blank clears the limit
        assert app._alpha_limit_value() is None
    finally:
        app.destroy()
