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


def _reentry_shape(ax, x0, y0, diam, length, nose_r, color, edge):
    """A reentry vehicle drawn nose-up: a cone of base `diam` and height
    `length` from the base at y0, with the tip blunted to radius `nose_r`
    (0 = sharp).  Its own true geometry — no fabrication."""
    R = diam / 2.0
    rn = min(max(nose_r, 0.0), 0.9 * R)
    if rn <= 1e-6 or length <= rn:
        pts = [(x0 - R, y0), (x0 + R, y0), (x0, y0 + length)]        # sharp cone
    else:
        capc = y0 + length - rn                                      # cap centre
        n = 14
        arc = [(x0 + rn * math.cos(math.pi * i / n),
                capc + rn * math.sin(math.pi * i / n)) for i in range(n + 1)]
        pts = [(x0 - R, y0), (x0 + R, y0)] + arc                     # blunted cone
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=edge, lw=1.3, zorder=3))


def _biconic_shape(ax, x0, y0, diam, length, break_d, fore_len, nose_r,
                   color, edge):
    """A biconic RV nose-up: an aft frustum from base `diam` up to `break_d` at
    the junction, then a forward cone to a tip blunted to radius `nose_r`."""
    r_b, r_1 = diam / 2.0, break_d / 2.0
    La = length - fore_len                                # aft-frustum height
    rn = min(max(nose_r, 0.0), 0.9 * r_1)
    if rn <= 1e-6 or fore_len <= rn:
        fore = [(x0, y0 + length)]                        # sharp apex
    else:
        capc = y0 + length - rn
        n = 12
        fore = [(x0 + rn * math.cos(math.pi * i / n),
                 capc + rn * math.sin(math.pi * i / n)) for i in range(n + 1)]
    pts = ([(x0 - r_b, y0), (x0 + r_b, y0), (x0 + r_1, y0 + La)]
           + fore + [(x0 - r_1, y0 + La)])
    ax.add_patch(Polygon(pts, closed=True, fc=color, ec=edge, lw=1.3, zorder=3))


def _draw_reentry_object(ax, ro, view_right, yl, veh_right=0.0):
    """Draw the reentry object to scale in the lower-RIGHT corner, base on the
    y = 0 ground line, from its own stored geometry, with its text to the right.

    Wings: S and AR alone cannot define a planform on a conical body (they give
    area and slenderness, not position, root chord, or shape).  So the wings
    are drawn FAITHFULLY only when the optional planform fields are entered
    (wing_root_chord_m + wing_span_exposed_m, with wing_sweep_deg): a panel
    whose root follows the body flank, trailing edge on the base line, leading
    edge swept back from the spanwise axis.  With only a wing AREA stored, a
    small fixed-proportion delta tab is drawn at the aft flank and the label
    says "(schematic)" — an honest marker, never fake dimensions.  A length of
    0 falls back to a nominal cone and is flagged, like the nose."""
    D = float(getattr(ro, "diameter_m", 0.0) or 0.0)
    if D <= 0:
        return
    L = float(getattr(ro, "length_m", 0.0) or 0.0)
    flag = ""
    if L <= 0:
        L = 1.6 * D
        flag = " (length unset)"
    rn = float(getattr(ro, "nose_radius_m", 0.0) or 0.0)
    R = D / 2.0

    # Biconic body when declared and geometrically valid; else a plain cone.
    bic = None
    if getattr(ro, "biconic", False):
        Lf = float(getattr(ro, "fore_length_m", 0.0) or 0.0)
        Dbrk = float(getattr(ro, "break_diameter_m", 0.0) or 0.0)
        if 0 < Lf < L and 0 < Dbrk < D:
            bic = (Lf, Dbrk)

    def r_local(y):
        """Body flank radius at height y — cone or biconic piecewise."""
        if bic is not None:
            Lf, Dbrk = bic
            La, R1 = L - Lf, Dbrk / 2.0
            if y <= La:
                return R - y * (R - R1) / La
            return max(0.0, R1 * (1.0 - (y - La) / Lf))
        return max(0.0, R * (1.0 - y / L))

    # Wing depiction mode.  Faithful when the planform fields are set; a
    # flagged schematic tab when only the area is; nothing otherwise.
    S = float(getattr(ro, "wing_area_m2", 0.0) or 0.0)
    w_rc = float(getattr(ro, "wing_root_chord_m", 0.0) or 0.0)
    w_ss = float(getattr(ro, "wing_span_exposed_m", 0.0) or 0.0)
    w_sw = float(getattr(ro, "wing_sweep_deg", 0.0) or 0.0)
    planform = (w_rc > 0.0 and w_ss > 0.0)
    wing_flag = ""
    if S > 0 and float(getattr(ro, "wing_aspect_ratio", 0.0) or 0.0) <= 0:
        wing_flag = " · AR def."                      # polar fail-safe, flagged
    if planform:
        wing_ext, glyph = w_ss, False
    elif S > 0:
        wing_ext, glyph = 0.35 * R, True              # fixed-proportion tab
    else:
        wing_ext, glyph = 0.0, False

    # Reserve room for the right-hand label from its pixel size (text is fixed
    # points, so its width in metres scales with the view).
    pos = ax.get_position(original=True)
    _fw_in, fh_in = ax.figure.get_size_inches()
    H = max(yl[1] - yl[0], 1e-6)
    m_per_in = H / max(pos.height * fh_in, 1e-6)
    label_w = 1.7 * m_per_in           # room for the longest label line
    x0 = view_right - 0.25 - label_w - (R + wing_ext)
    # Never drift left into (or past) the vehicle on small views: the RO stays
    # a clear margin right of the stack even if the label then runs tight.
    x0 = max(x0, veh_right + 0.4 + wing_ext + R)

    if bic is not None:
        _biconic_shape(ax, x0, 0.0, D, L, bic[1], bic[0], rn, NOSE, BODY_E)
    else:
        _reentry_shape(ax, x0, 0.0, D, L, rn, NOSE, BODY_E)

    if planform:
        # Faithful panel: root follows the flank from the base to the root
        # chord; trailing edge straight on the base line; the leading edge
        # sweeps back (from the spanwise axis) so the tip chord shrinks —
        # collapsing to a delta when the sweep consumes the whole chord.
        y_tip_le = min(max(w_rc - w_ss * math.tan(math.radians(w_sw)), 0.0), w_rc)
        for sgn in (+1, -1):
            pts = [(x0 + sgn * r_local(0.0), 0.0),
                   (x0 + sgn * r_local(w_rc), w_rc),
                   (x0 + sgn * (r_local(0.0) + w_ss), y_tip_le),
                   (x0 + sgn * (r_local(0.0) + w_ss), 0.0)]
            ax.add_patch(Polygon(pts, closed=True, fc=FIN, ec=FIN_E,
                                 lw=1.0, zorder=2))
    elif glyph:
        # Schematic tab: a small delta hugging the aft flank, deliberately
        # fixed-proportion (0.35·R out, 0.22·L up) — a marker, not a claim.
        rc_g = 0.22 * L
        for sgn in (+1, -1):
            pts = [(x0 + sgn * r_local(0.0), 0.0),
                   (x0 + sgn * r_local(rc_g), rc_g),
                   (x0 + sgn * (r_local(0.0) + wing_ext), 0.0)]
            ax.add_patch(Polygon(pts, closed=True, fc=FIN, ec=FIN_E,
                                 lw=1.0, zorder=2))

    name = getattr(ro, "name", "") or "RV"
    beta = float(getattr(ro, "beta_kg_m2", 0.0) or 0.0)
    ld = float(getattr(ro, "glider_LD", 0.0) or 0.0)
    lines = [f"reentry object: {name}", f"⌀{D:g}×{L:g} m{flag}"]
    tail = []
    if beta > 0:
        tail.append(f"β {beta:,.0f}")
    if ld > 0:
        tail.append(f"L/D {ld:g}")
    if tail:
        lines.append(" · ".join(tail))
    if bic is not None:
        Lf, Dbrk = bic
        th1 = math.degrees(math.atan2(Dbrk / 2.0, Lf))
        th2 = math.degrees(math.atan2((D - Dbrk) / 2.0, L - Lf))
        lines.append(f"biconic {th1:.1f}°/{th2:.1f}°")
    if planform:
        lines.append(f"wings S={S:g} m²{wing_flag}" if S > 0
                     else f"wings {w_rc:g}×{w_ss:g} m (no area set)")
    elif glyph:
        lines.append(f"wings S={S:g} m² (schematic{wing_flag})")
    # text to the RIGHT of the body/wings
    label_x = x0 + R + wing_ext + 0.2
    ax.text(label_x, L / 2.0, "\n".join(lines),
            va="center", ha="left", fontsize=7.5, color=LABEL_MUT)
    # widen the view rightward if the caption would run past the edge (keeps
    # the RO clear of the stack AND the text on-panel on narrow views)
    lbl_right = label_x + label_w
    if lbl_right > view_right:
        ax.set_xlim(ax.get_xlim()[0], lbl_right + 0.1)


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
        ax.text(x0 - R - 0.15, y + sl * 0.5,
                f"fairing ⌀{sd:g}×{sl:g} m{flag}".replace(" (", "\n("),
                va="center", ha="right", fontsize=8, color=SHROUD_E)
        if flag:
            flags.append("fairing" + flag)
        nose_base_d = sd
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
        ax.text(x0 - nd / 2 - 0.15, y + 0.5 * nl,
                f"payload / RV{flag}".replace(" (", "\n("),
                va="center", ha="right", fontsize=8, color=LABEL_MUT)
        if flag:
            flags.append("nose" + flag)
        nose_base_d = nd
        y += nl

    # Aerospike — a forward drag-reduction probe from the nose apex.  Length and
    # aerodisk diameter come straight from the stored ratios (× the forebody
    # diameter): aerospike_LD = spike length / D, aerospike_dD = disk ⌀ / D
    # (0 = pointed).  A top-level booster property, so read from the root node.
    a_LD = float(getattr(p, "aerospike_LD", 0.0) or 0.0)
    if a_LD > 0:
        a_dD = float(getattr(p, "aerospike_dD", 0.0) or 0.0)
        L_spike = a_LD * nose_base_d
        tip_y = y + L_spike
        ax.plot([x0, x0], [y, tip_y], color=BODY_E, lw=2.0, zorder=4)   # stalk
        if a_dD > 0:                                        # aerodisk at the tip
            disk_d = a_dD * nose_base_d
            ax.add_patch(Rectangle((x0 - disk_d / 2, tip_y - 0.02 * nose_base_d),
                                   disk_d, 0.04 * nose_base_d,
                                   fc=BODY, ec=BODY_E, lw=1.0, zorder=4))
            _lbl = f"aerospike {L_spike:.2g} m · disk ⌀{disk_d:.2g} m"
        else:                                               # pointed spike
            _lbl = f"aerospike {L_spike:.2g} m (pointed)"
        ax.text(x0 + nose_base_d / 2 + 0.15, y + L_spike / 2, _lbl,
                va="center", ha="left", fontsize=7.5, color=LABEL_MUT)
        y = tip_y

    if finned:
        s, yb = finned
        d = float(s.diameter_m); R = d / 2.0
        span = float(s.fin_span_m)
        root = float(getattr(s, "fin_root_chord_m", 0.0) or 0.8 * span)
        tip = float(getattr(s, "fin_tip_chord_m", 0.0) or 0.4 * root)
        sweep_deg = float(getattr(s, "fin_sweep_deg", 0.0) or 0.0)
        off = span * math.tan(math.radians(sweep_deg))
        for sgn in (+1, -1):
            pts = fin_polygon(sgn, R, yb, span, root, tip, sweep_deg)
            ax.add_patch(Polygon(pts, closed=True, fc=FIN, ec=FIN_E,
                                 lw=1.1, zorder=1))
        # label BELOW the fins (clear of the planform), never across them
        ax.text(0, yb - max(0.0, off) - 0.4,
                f"{int(s.n_fins or 4)} fins  span {span:g} m",
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
            flag = " (nom.)"
            flags.append("strap-on (length unset — nominal)")
        cR = float(s.diameter_m) / 2.0
        R = bd / 2.0
        for sgn in (+1, -1):
            cx = sgn * (cR + R + 0.05)
            ax.add_patch(Rectangle((cx - R, yb), bd, bL,
                                   fc=STRAP, ec=BODY_E, lw=1.1, zorder=1))
            _nose_patch(ax, cx, yb + bL, bd, 1.4 * bd, STRAP, BODY_E, "cone")
        # label in the tall clear LEFT margin at the strap top, wrapped so it
        # never runs over the core body (nor off the panel edge)
        ax.text(-(cR + 2 * R + 0.35), yb + bL,
                f"{n}× strap-on\n⌀{bd:g}×{bL:.1f} m{flag}",
                va="center", ha="right", fontsize=7.5, color=LABEL_MUT)

    ax.set_aspect("equal")
    ax.relim(); ax.autoscale_view()
    xl, yl = ax.get_xlim(), ax.get_ylim()
    _new_left, _view_right = _draw_scale_reference(ax, xl, yl)
    # A to-scale reentry object in the lower-right corner (when a loadout object
    # is composed onto the stack); drawn from its own geometry.
    _ro = getattr(p, "ro", None)
    if _ro is not None and float(getattr(_ro, "diameter_m", 0.0) or 0.0) > 0:
        _draw_reentry_object(ax, _ro, _view_right, yl, veh_right=xl[1])
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
    return new_left, view_right
