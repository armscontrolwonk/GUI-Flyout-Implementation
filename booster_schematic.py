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

from matplotlib.patches import Polygon, Rectangle

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
        ax.add_patch(Rectangle((x0 - R, y), d, L,
                               fc=BODY, ec=BODY_E, lw=1.3, zorder=2))
        ax.text(x0 + R + 0.15, y + L / 2, f"S{i+1}: ⌀{d:g}×{L:g} m",
                va="center", ha="left", fontsize=8, color=LABEL)
        if getattr(s, "has_fins", False) and (getattr(s, "fin_span_m", 0.0) or 0) > 0:
            finned = (s, y)
        if getattr(s, "has_grid_fins", False) and (getattr(s, "n_grid_fins", 0) or 0) > 0:
            grid_finned = (s, y, L)
        if (getattr(s, "n_boosters", 0) or 0) > 0:
            strap = (s, y, L)
        y_top = y + L
        if i < len(stages) - 1:            # frustum where the diameter changes
            du = float(getattr(stages[i + 1], "diameter_m", 0.0) or d)
            if abs(du - d) > 1e-3:
                Ru = du / 2.0
                fl = 0.4 * abs(d - du) + 0.2
                ax.add_patch(Polygon([(x0 - R, y_top), (x0 + R, y_top),
                                      (x0 + Ru, y_top + fl), (x0 - Ru, y_top + fl)],
                                     closed=True, fc=BODY, ec=BODY_E,
                                     lw=1.2, zorder=2))
                y_top += fl
        y = y_top

    top = stages[-1]
    if shroud_stage is not None:
        sd = float(getattr(shroud_stage, "shroud_diameter_m", 0.0)
                   or getattr(top, "diameter_m", 1.0))
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
        nd = float(getattr(top, "diameter_m", 0.0) or 1.0)
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
        sweep = math.radians(float(getattr(s, "fin_sweep_deg", 0.0) or 0.0))
        off = span * math.tan(sweep)
        for sgn in (+1, -1):
            pts = [(sgn * R, yb), (sgn * R, yb + root),
                   (sgn * (R + span), yb + root - off),
                   (sgn * (R + span), yb + root - off - tip)]
            ax.add_patch(Polygon(pts, closed=True, fc=FIN, ec=FIN_E,
                                 lw=1.1, zorder=1))
        ax.text(0, yb - 0.5, f"{int(s.n_fins or 4)} fins  span {span:g} m",
                va="top", ha="center", fontsize=7.5, color=LABEL_MUT)

    if grid_finned:
        s, yb, Lc = grid_finned
        d = float(s.diameter_m); R = d / 2.0
        gh = float(getattr(s, "grid_fin_height_m", 0.0) or 0.15 * d)
        gc = float(getattr(s, "grid_fin_chord_m", 0.0) or gh)
        yg = yb + 0.15 * Lc
        for sgn in (+1, -1):
            ax.add_patch(Rectangle((sgn * R if sgn > 0 else -R - gh, yg),
                                   gh, gc, fc=FIN, ec=FIN_E, lw=1.0,
                                   zorder=1, hatch="++"))
        ax.text(0, yg - 0.3,
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
    xl = ax.get_xlim()
    barx = xl[0] - 0.6
    ax.plot([barx, barx], [0, 5], color="k", lw=2)
    for yy in (0, 5):
        ax.plot([barx - 0.1, barx + 0.1], [yy, yy], color="k", lw=2)
    ax.text(barx - 0.2, 2.5, "5 m", va="center", ha="right", fontsize=8)
    ax.set_xlim(barx - 1.0, xl[1])
    if title:
        ax.set_title(title, fontsize=11, weight="bold")
    ax.axis("off")
    return {"total_height_m": y, "flags": flags}
