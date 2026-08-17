"""Thrusty.app — the macOS wrapper bundle that names the Dock tile.

The bundle can't be exercised here (no macOS), so the tests pin its
integrity instead: the plist parses with consistent names, every file it
references exists, the launcher is executable and targets thrusty.py, and
the machine-local interpreter copy it creates stays out of git."""

import os
import plistlib
import stat

_APP = os.path.join(os.path.dirname(__file__), "Thrusty.app", "Contents")


def test_bundle_plist_is_consistent():
    with open(os.path.join(_APP, "Info.plist"), "rb") as f:
        d = plistlib.load(f)
    assert d["CFBundleName"] == "Thrusty"
    assert d["CFBundleDisplayName"] == "Thrusty"
    assert d["CFBundlePackageType"] == "APPL"
    # the executable and icon the plist names must exist in the bundle
    exe = os.path.join(_APP, "MacOS", d["CFBundleExecutable"])
    assert os.path.exists(exe)
    assert os.path.exists(os.path.join(_APP, "Resources",
                                       d["CFBundleIconFile"]))


def test_launcher_is_executable_and_targets_thrusty():
    exe = os.path.join(_APP, "MacOS", "Thrusty")
    assert os.stat(exe).st_mode & stat.S_IXUSR      # exec bit committed
    src = open(exe).read()
    assert src.startswith("#!/bin/bash")
    assert 'thrusty.py' in src
    # the machine-local interpreter copy must never be committed
    ignores = open(os.path.join(os.path.dirname(__file__),
                                ".gitignore")).read()
    assert "Thrusty-python" in ignores


def test_bundle_icns_matches_the_app_icon():
    """The .icns is generated from assets/thrusty.png and shipped twice
    (assets/ source + bundle Resources) — both copies identical."""
    a = open(os.path.join(os.path.dirname(__file__), "assets",
                          "thrusty.icns"), "rb").read()
    b = open(os.path.join(_APP, "Resources", "thrusty.icns"), "rb").read()
    assert a == b and len(a) > 10000
