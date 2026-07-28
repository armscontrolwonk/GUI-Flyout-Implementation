"""Schematic renderer (booster_schematic.draw_booster).

The panel exists to make bad geometry visible, so the tests hold it to the
same data-honesty it enforces: heights are the arithmetic of stored fields
(identity, not eyeball), unset fields must surface as flags rather than be
silently invented, and the nose must point forward (the sin-vs-cos inverted
cone was a real prototype bug).  Nothing here pins a "pretty" outcome.
"""

import glob
import json
import math

import pytest

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure

from booster_models import booster_from_dict
from booster_schematic import draw_booster, stage_chain, fin_polygon


def _ax():
    return Figure(figsize=(4, 8)).add_subplot(111)


def _load(name):
    return booster_from_dict(
        json.load(open(f"booster_library/{name}.booster.json")))


# ── every library booster draws without error ───────────────────────────────
def test_every_library_booster_renders():
    for f in glob.glob("booster_library/*.booster.json"):
        b = booster_from_dict(json.load(open(f)))
        info = draw_booster(_ax(), b)
        assert info["total_height_m"] > 0, f
        assert math.isfinite(info["total_height_m"]), f


# ── height is the arithmetic of the stored fields ───────────────────────────
def test_total_height_is_sum_of_stage_lengths_plus_front_end():
    """AUR: 5.0 (S1) + 2.6 (S2) + 2.67 (full-length conical fairing) with no
    interstage frustum (constant diameter).  The identity that caught the
    original 10.2 m all-up-round-in-the-S1-field error."""
    info = draw_booster(_ax(), _load("AUR"))
    assert info["total_height_m"] == (5.0 + 2.6 + 2.67)


def test_corrected_aur_carries_no_fallback_flags():
    """AUR's geometry is fully specified (shape=cone, nose length set), so
    the drawing must not need — or report — any invented values."""
    info = draw_booster(_ax(), _load("AUR"))
    assert info["flags"] == []


def test_unset_fields_are_flagged_not_silently_defaulted():
    """Minotaur's shroud has no nose shape/length in the file; Strypi's
    strap-ons have no length.  Fallbacks are drawn but must be declared."""
    mi = draw_booster(_ax(), _load("Minotaur-IV_-_HTV-2"))
    assert any("shape unset" in f for f in mi["flags"])
    st = draw_booster(_ax(), _load("Strypi_VIII_R"))
    assert any("strap-on" in f and "unset" in f for f in st["flags"])


# ── the nose points forward ─────────────────────────────────────────────────
def test_nose_apex_is_the_topmost_point():
    """The stack's maximum y must lie on the vehicle centreline (the nose
    tip), not at a shoulder — a taper drawn the wrong way round (the
    inverted-cone bug) puts the widest section at the very top instead."""
    ax = _ax()
    info = draw_booster(ax, _load("AUR"))
    top_y = info["total_height_m"]
    for patch in ax.patches:
        for (x, y) in patch.get_path().vertices:
            if abs(y - top_y) < 1e-9:
                assert abs(x) < 1e-9, (x, y)


# ── feature presence tracks the data ────────────────────────────────────────
def test_features_draw_iff_the_data_declares_them():
    """Strypi (fins + strap-ons) must produce more patches than a plain
    single-stack of the same stage count would; a booster with no fins,
    shroud, or straps must not sprout any."""
    plain = draw_booster(_ax(), _load("No-dong"))
    assert plain["flags"] == [] or all("strap" not in f for f in plain["flags"])
    ax_st = _ax()
    draw_booster(ax_st, _load("Strypi_VIII_R"))
    ax_nd = _ax()
    draw_booster(ax_nd, _load("No-dong"))
    assert len(ax_st.patches) > len(ax_nd.patches)


# ── tail fins point the right way ────────────────────────────────────────────
def test_fin_trailing_edge_sits_at_the_base_not_floating_high():
    """The reversed-fin bug: anchoring the tip to the leading edge left a
    clipped fin (tip < root) floating above the base with a forward-swept
    trailing edge.  A correct tail fin has its trailing edge AT the base (or
    below, once swept), never hanging above it."""
    yb = 0.0
    for sweep in (0.0, 30.0, 45.0):
        pts = fin_polygon(+1, R=0.66, yb=yb, span=1.0, root=2.0, tip=1.0,
                          sweep_deg=sweep)
        ys = [p[1] for p in pts]
        # the fin's lowest point (a tip trailing corner) is at or below the base
        assert min(ys) <= yb + 1e-9, (sweep, ys)
        # and the highest point is the root leading edge, at yb + root
        assert max(ys) == yb + 2.0


def test_leading_edge_is_swept_back_not_forward():
    """Going outboard, the leading (forward, upper) edge must move AFT (down):
    the tip leading edge is below the root leading edge.  Forward sweep (tip
    leading above root leading) is the reversed rendering."""
    root = 1.5
    root_leading_y = 0.0 + root
    for sweep in (0.0, 20.0, 40.0):
        _, _, tip_leading, _ = fin_polygon(+1, R=0.5, yb=0.0, span=1.0,
                                           root=root, tip=0.6, sweep_deg=sweep)
        assert tip_leading[1] < root_leading_y, (sweep, tip_leading)


def test_equal_aspect_is_enforced():
    """Proportion honesty is the whole point: the axes must be metre-true in
    both directions, not stretched to fit the panel — including after the
    scale-figure imshow, which must not flip the axes to aspect='auto'."""
    ax = _ax()
    draw_booster(ax, _load("AUR"))
    assert ax.get_aspect() == 1.0


# ── the 2 m Thrusty scale reference ─────────────────────────────────────────
def test_scale_figure_is_two_metres_tall():
    """The mascot silhouette is placed exactly 2 m tall (feet on y=0), so it
    reads as a literal human-scale ruler beside the stack."""
    import booster_schematic as bs
    if bs._scale_image() is None:
        pytest.skip("scale asset not present in this checkout")
    ax = _ax()
    draw_booster(ax, _load("Scud-B_-R-17-"))
    imgs = ax.get_images()
    assert len(imgs) == 1
    x0, x1, y0, y1 = imgs[0].get_extent()
    assert (y0, y1) == (0.0, bs._SCALE_FIGURE_M)          # feet at 0, head at 2 m
    assert y1 == 2.0
    # width preserves the art's aspect ratio (never stretched)
    h_px, w_px = imgs[0].get_array().shape[:2]
    assert (x1 - x0) == pytest.approx(2.0 * w_px / h_px)


def test_scale_reference_falls_back_without_the_asset(monkeypatch):
    """A stripped checkout (no assets/) must still render — the reference
    degrades to a plain 2 m bar, never a crash."""
    import booster_schematic as bs
    monkeypatch.setattr(bs, "_scale_img_cache", None)     # simulate missing asset
    ax = _ax()
    info = draw_booster(ax, _load("Scud-B_-R-17-"))
    assert info["total_height_m"] > 0
    assert ax.get_images() == []                          # no silhouette drawn
    assert ax.get_aspect() == 1.0
    assert any("2 m" in t.get_text() for t in ax.texts)   # bar still labelled
