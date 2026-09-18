"""Test isolation — a run depends only on what is committed.

`thrusty.py` points `booster_models` at the user's own library on import
(`~/Documents/Thrusty/{flight_plans,reentry_plans,ro_library}`, thrusty.py
§"user library paths").  Those directories take precedence over the shipped
ones, so as soon as ANY test imports the GUI, every later `get_booster()` in
that pytest session can silently fly a personal saved plan instead of the
packaged one — and pytest imports all test modules up front, so a single GUI
test contaminates files that never mention the GUI.

That is not hypothetical.  `test_dem_terminates_on_terrain` passed on its own
and failed in a whole-suite run, because a user-saved `No-dong.flightplan.json`
carrying `burnout_angle_deg` 43.0 replaced the shipped 45.0 and moved the
impact point 19 km.  The suite was measuring the developer's desktop.

This matters beyond a flaky assertion: reference trajectories dumped from this
code are the specification the Rust core is being written against.  Dumping
them on a machine with saved plans would bake someone's personal flight plan
into the port.

So: blank the user-library paths for the duration of every test.  A test that
deliberately exercises the user library sets these globals itself, inside the
test, and this fixture restores them afterwards.
"""

import pytest

import booster_models as mm


def pytest_configure(config):
    """Let the GUI point the paths wherever it likes — once, up front.

    The fixture below runs before every test, so whoever writes those globals
    LAST wins.  Importing the GUI here means that is always the fixture, even
    in a selection where no module imports `thrusty` at collection time and
    some test imports it halfway through its own body.  Guarded: on a machine
    without Tk (or without a display) the import simply fails and the fixture
    still does its job.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import thrusty                                          # noqa: F401
    except Exception:
        pass


@pytest.fixture(autouse=True)
def shipped_libraries_only():
    saved = (list(mm.USER_FLIGHT_PLAN_DIRS),
             list(mm.USER_REENTRY_PLAN_DIRS),
             list(mm.USER_RO_DIRS),
             dict(mm.ACTIVE_FLIGHT_PLANS),
             dict(mm.ACTIVE_REENTRY_PLANS))
    mm.USER_FLIGHT_PLAN_DIRS = []
    mm.USER_REENTRY_PLAN_DIRS = []
    mm.USER_RO_DIRS = []
    mm.ACTIVE_FLIGHT_PLANS.clear()
    mm.ACTIVE_REENTRY_PLANS.clear()
    try:
        yield
    finally:
        (mm.USER_FLIGHT_PLAN_DIRS,
         mm.USER_REENTRY_PLAN_DIRS,
         mm.USER_RO_DIRS,
         _active_fp,
         _active_rp) = saved
        mm.ACTIVE_FLIGHT_PLANS.clear()
        mm.ACTIVE_FLIGHT_PLANS.update(_active_fp)
        mm.ACTIVE_REENTRY_PLANS.clear()
        mm.ACTIVE_REENTRY_PLANS.update(_active_rp)
