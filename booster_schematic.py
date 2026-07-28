"""To-scale side elevation of a booster stack — the Schematic tab's renderer.

Draws the vehicle purely from stored geometry fields on a BoosterParams chain:
per-stage diameter/length, nose shape and length, shroud (fairing) dimensions
and nose shape, fin planform (span/chords/sweep), grid fins, and strap-on
boosters.  Equal-aspect metres on both axes, so proportions are literal — the
panel exists to make a mis-entered length or an oversized fairing obvious at
a glance (it caught the AUR's all-up-round length sitting in the stage-1
field, and its ogive-vs-cone fairing, on its first outing).

Data honesty: nothing is invented silently.  Where a field the drawing needs
is unset (nose shape, nose length, strap-on length), a conservative fallback
is drawn and the label is flagged "(… unset)" — missing geometry should be
visible, not papered over.  The same fields feed the boost-phase drag build-up
(booster_models._cd_nose_shape), so what you see is what the physics flies.

Pure matplotlib (no Tk import) — embeddable in the GUI via FigureCanvasTkAgg
and testable headless under Agg.
"""

import math
import os

import matplotlib.image as mpimg
from matplotlib.patches import Polygon, Rectangle

# Thrusty mascot silhouette: a human-relatable figure standing beside the
# vehicle for a felt sense of scale, with the quantitative reference carried by
# a dimensioned metre bar next to it.  Loaded once and cached; a missing asset
# falls back to the bar alone, so headless renders and stripped checkouts work.
_SCALE_IMG_PATH = os.path.join(os.path.dirname(__file__),
                               "assets", "thrusty_scale.png")
_SCALE_FIGURE_M = 1.8                      # the mascot stands ~1.8 m tall (human)
_SCALE_BAR_M    = 5.0                      # the quantitative reference bar
_UNLOADED = object()                       # sentinel (image may be an ndarray)
_scale_img_cache = _UNLOADED


def _scale_image():
    global _scale_img_cache
    if _scale_img_cache is _UNLOADED:
        try:
            _scale_img_cache = mpimg.imread(_SCALE_IMG_PATH)
        except (FileNotFoundError, OSError):
            _scale_img_cache = None
    return _scale_img_cache

# Muted, print-friendly greys; the fairing is the one tinted element because
# it is the piece most worth eyeballing.
BODY, BODY_E     = "#c9ccd1", "#5a5e66"
SHROUD, SHROUD_E = "#b7c7d8", "#4a6076"
FIN, FIN_E       = "#9aa0a8", "#4a4e56"
STRAP            = "#bfc4cb"
NOSE             = "#d7dae0"
LABEL, LABEL_MUT = "#333333", "#555555"


def stage_chain(p):
    """The stage list, bottom (stage 1) first, walking the .stage2 chain."""
    out, node = [], p
    while node is not None:
        out.append(node)
        node = getattr(node, "stage2", None)
    return out


def fin_polygon(sgn, R, yb, span, root, tip, sweep_deg):
    """Side-elevation outline of one tail fin, vehicle nose-up (forward = +y).

    The trailing (aft) edge sits at the stage base yb; the leading edge sweeps
    back, so the tip is forward of the root and, for sweep_deg > 0, shifted aft
    (down).  Anchoring the tip to the TRAILING edge is what keeps a clipped fin
    (tip < root) reading as swept-back, not the reversed forward-swept look.
    Returns four (x, y) points: root-trailing, root-leading, tip-leading,
    tip-trailing.
    """
    off = span * math.tan(math.radians(sweep_deg))
    return [(sgn * R, yb),
            (sgn * R, yb + root),
            (sgn * (R + span), yb + tip - off),
            (sgn * (R + span), yb - off)]


def _body_patch(ax, x0, y0, d_bottom, d_top, length, color, edge):
    """A stage/interstage body from y0 up to y0+length, centred on x0.

    A cylinder when d_bottom == d_top, otherwise a frustum (trapezoid) tapering
    from d_bottom at the base to d_top at the top.
    """
    Rb, Rt = d_bottom / 2.0, d_top / 2.0
    if abs(d_bottom - d_top) < 1e-9:
        ax.add_patch(Rectangle((x0 - Rb, y0), d_bottom, length,
                               fc=color, ec=edge, lw=1.3, zorder=2))
    else:
        ax.add_patch(Polygon([(x0 - Rb, y0), (x0 + Rb, y0),
                              (x0 + Rt, y0 + length), (x0 - Rt, y0 + length)],
                             closed=True, fc=color, ec=edge, lw=1.3, zorder=2))


def _stage_top_diameter(s):
    """The diameter at the top of stage `s` — its top_diameter_m when the stage
    is conical (and set), else its base diameter."""
    d = float(getattr(s, "diameter_m", 0.0) or 0.6)
    if getattr(s, "conical", False):
        dt = float(getattr(s, "top_diameter_m", 0.0) or 0.0)
        if dt > 0:
            return dt
    return d


def _nose_patch(ax, x0, y0, diam, length, color, edge, shape):
    """A nose from y0 up to y0+length, base width diam, centred on x0.

    'cone' is a straight taper; anything containing 'ogive' gets a rounded
    tangent-ogive-ish curve; blunt shapes get a capped dome.
    """
    R = diam / 2.0
    s = (shape or "").lower()
    if "ogive" in s or "haack" in s or "karman" in s or "parabola" in s:
        n = 24
        left = [(x0 - R * math.cos(math.pi / 2 * i / n), y0 + length * i / n)
                for i in range(n + 1)]
        pts = left + [(x, y) for (x, y) in
                      ((2 * x0 - xx, yy) for (xx, yy) in reversed(left))]
    elif "blunt" in s:
        n = 16
        left = [(x0 - R * math.cos(math.pi / 2 * i / n),
                 y0 + length * math.sin(math.pi / 2 * i / n)) for i in range(n + 1)]
        pts = left + [(2 * x0 - xx, yy) for (xx, yy) in reversed(left)]
    else:                                   # straight cone
        pts = [(x0 - R, y0), (x0, y0 + length), (x0 + R, y0)]
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=edge, lw=1.2, zorder=3))


def draw_booster(ax, p, title=None):
    """Draw the stack on `ax` (cleared first).  Returns a summary dict:
    {'total_height_m': float, 'flags': [str, ...]} — flags list every place a
    fallback stood in for unset data."""
    ax.clear()
    stages = stage_chain(p)
    flags = []
    x0 = 0.0
    finned = grid_finned = strap = None

    shroud_stage = next(
        (s for s in stages
         if (getattr(s, "shroud_length_m", 0.0) or 0.0) > 0.0), None)

    y = 0.0
    for i, s in enumerate(stages):
        d = float(getattr(s, "diameter_m", 0.0) or 0.6)
        L = float(getattr(s, "length_m", 0.0) or 1.0)
        R = d / 2.0
        d_top = _stage_top_diameter(s)                 # equals d unless conical
        _body_patch(ax, x0, y, d, d_top, L, BODY, BODY_E)
        _lbl = (f"S{i+1}: ⌀{d:g}→{d_top:g}×{L:g} m" if d_top != d
                else f"S{i+1}: ⌀{d:g}×{L:g} m")
        ax.text(x0 + R + 0.15, y + L / 2, _lbl,
                va="center", ha="left", fontsize=8, color=LABEL)
        if getattr(s, "has_fins", False) and (getattr(s, "fin_span_m", 0.0) or 0) > 0:
            finned = (s, y)
        if getattr(s, "has_grid_fins", False) and (getattr(s, "n_grid_fins", 0) or 0) > 0:
            grid_finned = (s, y, L)
        if (getattr(s, "n_boosters", 0) or 0) > 0:
            strap = (s, y, L)
        # Stages butt directly together — a diameter change shows as an honest
        # step, never a smoothing frustum (inventing one would hide an
        # unspecified transition, which this panel exists to surface).  A real
        # adapter is drawn ONLY when the stage declares an interstage; its
        # diameters are DERIVED (this stage's top -> the next stage's base) so
        # nothing about the transition is fabricated.
        y = y + L
        if getattr(s, "has_interstage", False) \
                and (getattr(s, "interstage_length_m", 0.0) or 0) > 0:
            il = float(s.interstage_length_m)
            d_is_bot = d_top                                    # this stage's top
            nxt = stages[i + 1] if i + 1 < len(stages) else None
            d_is_top = float(getattr(nxt, "diameter_m", 0.0) or d_top) if nxt \
                else d_top                                       # next base, or hold
            _body_patch(ax, x0, y, d_is_bot, d_is_top, il, SHROUD, BODY_E)
            _im = getattr(s, "interstage_mass_kg", 0.0) or 0.0
            _jt = getattr(s, "interstage_jettison_s", None)
            _jtxt = f"{_jt:g} s" if _jt is not None else "with stage"
            ax.text(x0 - max(d_is_bot, d_is_top) / 2 - 0.15, y + il / 2,
                    f"interstage {il:g} m, {_im:g} kg\njett {_jtxt}",
                    va="center", ha="right", fontsize=7.5, color=SHROUD_E)
            y += il

    top = stages[-1]
    top_surface_d = _stage_top_diameter(top)       # nose/fairing sits on this
    if shroud_stage is not None:
        sd = float(getattr(shroud_stage, "shroud_diameter_m", 0.0)
                   or top_surface_d)
        sl = float(getattr(shroud_stage, "shroud_length_m", 0.0) or 2 * sd)
        shape = getattr(shroud_stage, "shroud_nose_shape", "") or ""
        nose = float(getattr(shroud_stage, "shroud_nose_length_m", 0.0) or 0.0)
        R = sd / 2.0
        flag = ""
        if not (0.0 < nose <= sl):
            nose = 0.45 * sl
            flag = " (nose length unset)"
        if not shape:
            flag += " (shape unset — cone shown)"
        cyl = sl - nose
        if cyl > 0:
            ax.add_patch(Rectangle((x0 - R, y), sd, cyl,
                                   fc=SHROUD, ec=SHROUD_E, lw=1.3, zorder=3))
        _nose_patch(ax, x0, y + cyl, sd, nose, SHROUD, SHROUD_E, shape or "cone")
        ax.text(x0 - R - 0.15, y + sl * 0.5, f"fairing ⌀{sd:g}×{sl:g} m{flag}",
                va="center", ha="right", fontsize=8, color=SHROUD_E)
        if flag:
            flags.append("fairing" + flag)
        y += sl
    else:
        nd = top_surface_d or 1.0
        shape = getattr(top, "nose_shape", "") or ""
        nl = float(getattr(top, "nose_length_m", 0.0) or 0.0)
        flag = ""
        if nl <= 0.0:
            nl = 1.6 * nd
            flag = " (nose length unset)"
        if not shape:
            flag += " (shape unset — cone shown)"
        _nose_patch(ax, x0, y, nd, nl, NOSE, BODY_E, shape or "cone")
        ax.text(x0 - nd / 2 - 0.15, y + 0.5 * nl, f"payload / RV{flag}",
                va="center", ha="right", fontsize=8, color=LABEL_MUT)
        if flag:
            flags.append("nose" + flag)
        y += nl

    if finned:
        s, yb = finned
        d = float(s.diameter_m); R = d / 2.0
        span = float(s.fin_span_m)
        root = float(getattr(s, "fin_root_chord_m", 0.0) or 0.8 * span)
        tip = float(getattr(s, "fin_tip_chord_m", 0.0) or 0.4 * root)
        sweep_deg = float(getattr(s, "fin_sweep_deg", 0.0) or 0.0)
        for sgn in (+1, -1):
            pts = fin_polygon(sgn, R, yb, span, root, tip, sweep_deg)
            ax.add_patch(Polygon(pts, closed=True, fc=FIN, ec=FIN_E,
                                 lw=1.1, zorder=1))
        ax.text(0, yb - 0.5, f"{int(s.n_fins or 4)} fins  span {span:g} m",
                va="top", ha="center", fontsize=7.5, color=LABEL_MUT)

    if grid_finned:
        s, yb, Lc = grid_finned
        d = float(s.diameter_m); R = d / 2.0
        gh = float(getattr(s, "grid_fin_height_m", 0.0) or 0.15 * d)
        gc = float(getattr(s, "grid_fin_chord_m", 0.0) or gh)
        # Grid fins sit at the stage BASE (aft), like tail fins — the panel's
        # bottom edge just above the base line, not partway up the body.
        yg = yb + 0.2
        for sgn in (+1, -1):
            ax.add_patch(Rectangle((sgn * R if sgn > 0 else -R - gh, yg),
                                   gh, gc, fc=FIN, ec=FIN_E, lw=1.0,
                                   zorder=1, hatch="++"))
        ax.text(0, yb - 0.4,
                f"{int(s.n_grid_fins)} grid fins {gh:g}×{gc:g} m",
                va="top", ha="center", fontsize=7.5, color=LABEL_MUT)

    if strap:
        s, yb, Lc = strap
        n = int(s.n_boosters)
        bd = float(getattr(s, "booster_diam_m", 0.0) or 0.3)
        bL = float(getattr(s, "booster_length_m", 0.0) or 0.0)
        flag = ""
        if bL <= 0:
            bL = min(0.45 * Lc, 18 * bd)
            flag = " (length unset — nominal)"
            flags.append("strap-on" + flag)
        cR = float(s.diameter_m) / 2.0
        R = bd / 2.0
        for sgn in (+1, -1):
            cx = sgn * (cR + R + 0.05)
            ax.add_patch(Rectangle((cx - R, yb), bd, bL,
                                   fc=STRAP, ec=BODY_E, lw=1.1, zorder=1))
            _nose_patch(ax, cx, yb + bL, bd, 1.4 * bd, STRAP, BODY_E, "cone")
        ax.text(0, yb + bL + 1.4 * bd + 0.2,
                f"{n}× strap-on ⌀{bd:g}×{bL:.1f} m{flag}",
                va="bottom", ha="center", fontsize=7.5, color=LABEL_MUT)

    ax.set_aspect("equal")
    ax.relim(); ax.autoscale_view()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    _draw_scale_reference(ax, xl, yl)
    if title:
        ax.set_title(title, fontsize=11, weight="bold")
    ax.axis("off")
    return {"total_height_m": y, "flags": flags}


def _draw_scale_reference(ax, xl, yl):
    """Anchor the reference group (~1.8 m Thrusty · thin 5 m bar · "5 m") in the
    LOWER-LEFT corner of the panel, feet/base on the y = 0 ground line.

    A tall, thin stack makes the equal-aspect axes box tall and narrow, which
    matplotlib centres in the panel — so a group pinned just left of the stack
    rides toward the middle and crowds it (all the empty space is panel padding
    outside the box).  We instead widen the data x-range to fill the axes box
    exactly, which removes the centring padding, then place the group at the
    far left.  The mascot's metre height is unchanged (equal aspect), only its
    position moves.  Falls back to the bar alone when the asset is absent."""
    img = _scale_image()
    fig_w = 0.0
    if img is not None:
        h_px, w_px = img.shape[0], img.shape[1]
        fig_w = _SCALE_FIGURE_M * (w_px / h_px)         # preserve the art's aspect
    bar_gap, label_w = 0.35, 0.8
    group_w = fig_w + bar_gap + 0.2 + label_w           # silhouette · bar · "5 m"

    # Data width that fills the axes box at equal aspect: box_aspect × height.
    figr = ax.figure
    pos = ax.get_position(original=True)   # panel rect BEFORE equal-aspect shrink
    fw_in, fh_in = figr.get_size_inches()
    box_aspect = (pos.width * fw_in) / max(pos.height * fh_in, 1e-6)
    H = max(yl[1] - yl[0], 1e-6)
    want_w = max(box_aspect * H, group_w + 1.0)
    # Keep the stack centred (its side labels stay clear of the panel edges,
    # since text is not counted in autoscale) and fill the box; the extra width
    # opens as empty margin the group drops into at the lower-left.
    cx = 0.5 * (xl[0] + xl[1])
    new_left = cx - want_w / 2.0
    view_right = cx + want_w / 2.0

    x = new_left + 0.3                                   # left margin
    if img is not None:
        left, right = x, x + fig_w
        # origin='upper' puts image row 0 (the head) at the top of the extent
        ax.imshow(img, extent=(left, right, 0.0, _SCALE_FIGURE_M),
                  aspect="equal", zorder=3, interpolation="antialiased")
        bar_x = right + bar_gap
    else:
        bar_x = x
    # Thin 5 m reference bar with end ticks — reads as a rule, not a beam
    ax.plot([bar_x, bar_x], [0, _SCALE_BAR_M], color="k", lw=1.0, zorder=4)
    for yy in (0.0, _SCALE_BAR_M):
        ax.plot([bar_x - 0.12, bar_x + 0.12], [yy, yy],
                color="k", lw=1.0, zorder=4)
    ax.text(bar_x + 0.18, _SCALE_BAR_M / 2, "5 m",
            va="center", ha="left", fontsize=9, weight="bold")

    # imshow re-tightens the view; restore the full extent (group at far left,
    # stack toward the right) so the box fills the panel and nothing is centred.
    ax.set_xlim(new_left, view_right)
    ax.set_ylim(yl)
