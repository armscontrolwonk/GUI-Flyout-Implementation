"""Antialiased renderer for the image-measure prompt diagrams.

The little what-to-click diagrams were drawn as raw Tk canvas items, which
have no antialiasing — every diagonal edge a pixel staircase.  This module
renders the SAME pure data (image_measure.DIAGRAM_BASES / diagram_spec /
DIAGRAM_STYLE) through matplotlib's Agg canvas instead — the architecture
the Schematic tab already uses (booster_schematic.py) — at 2× and
downscales, so edges are print-quality at any size.  Matplotlib is already
an app dependency; this costs nothing new.

Pure matplotlib object API (no pyplot, no Tk, no global backend change —
the app's own figures keep their TkAgg canvases untouched).  Testable
headless: render_prompt_diagram returns a PIL image whose pixels tests can
sample.

Same conventions as the Tk renderer it replaces: unit square, y DOWN;
closed outlines are filled body art (the prompt's SUBJECT element fills
white, the rest grey); open polylines are detail strokes; a length prompt
is a single-headed red 1→2 arrow (direction = click order); an angle
prompt is vertex + two rays + the minor arc; numbered red discs mark the
clicks.  Style-token pixel values are treated as pixels at 1× scale.
"""
import math

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon
from PIL import Image

import image_measure as im


def _pt(px):
    """Style-token pixels (at 1× = 100 dpi) → matplotlib points."""
    return px * 72.0 / 100.0


def render_prompt_diagram(spec, base_polys, w_px, h_px, supersample=2):
    """Render one prompt's diagram to a PIL RGB image of (w_px, h_px).
    `spec` is image_measure.diagram_spec(...); `base_polys` the tagged
    outline list (DIAGRAM_BASES[...] or ro_side_base(...))."""
    st = im.DIAGRAM_STYLE
    fig = Figure(figsize=(w_px / 100.0, h_px / 100.0),
                 dpi=100 * supersample)
    FigureCanvasAgg(fig)
    fig.patch.set_facecolor(st["bg"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)              # unit square, y DOWN (data convention)
    ax.axis("off")

    subject = spec.get("subject")
    for item in base_polys:
        poly = item["pts"]
        if im.closed_poly(poly):       # body art: filled + outlined
            ax.add_patch(Polygon(
                poly, closed=True,
                facecolor=(st["highlight"] if item["tag"] == subject
                           else st["fill"]),
                edgecolor=st["outline"],
                linewidth=_pt(st["outline_width"]), joinstyle="round"))
        else:                          # detail stroke (break/joint line)
            ax.plot([q[0] for q in poly], [q[1] for q in poly],
                    color=st["outline"], linewidth=_pt(st["outline_width"]),
                    solid_capstyle="round")

    pts = spec["pts"]
    if spec["kind"] == "angle":
        (vx, vy), (a2x, a2y), (b2x, b2y) = pts
        for x2, y2 in ((a2x, a2y), (b2x, b2y)):
            ax.plot([vx, x2], [vy, y2], color=st["measure"],
                    linewidth=_pt(st["measure_width"] - 1),
                    solid_capstyle="round")
        # minor arc, circular in PIXELS (the anisotropic data square would
        # squash a data-space circle): angles from the pixel-space rays,
        # radius mapped back per-axis.
        a1 = -math.degrees(math.atan2((a2y - vy) * h_px, (a2x - vx) * w_px))
        a2_ = -math.degrees(math.atan2((b2y - vy) * h_px, (b2x - vx) * w_px))
        ext = (a2_ - a1) % 360.0
        if ext > 180.0:
            a1, ext = a2_, 360.0 - ext
        r = st["arc_r"]
        ts = [math.radians(a1 + ext * i / 24.0) for i in range(25)]
        ax.plot([vx + r / w_px * math.cos(t) for t in ts],
                [vy - r / h_px * math.sin(t) for t in ts],
                color=st["measure"], linewidth=_pt(2))
    else:
        (x1, y1), (x2, y2) = pts
        # single-headed 1→2 arrow: the direction IS the click order
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>",
            mutation_scale=_pt(st["arrowshape"][1]),
            color=st["measure"], linewidth=_pt(st["measure_width"]),
            shrinkA=0.0, shrinkB=0.0, capstyle="round"))

    # Click markers: a SMALL dot at the exact point, with the number set
    # off to the side (radially outward from the click cluster, so it sits
    # beyond the measured span, not on it) — big numbered discs covered
    # the very feature being pointed at.  A label that would leave the
    # canvas falls back to the perpendicular side toward the interior.
    r = st["dot_r"]
    off = st["label_offset"]
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    mx, my = 8.0 / w_px, 8.0 / h_px          # keep labels fully on-canvas
    for n, (x, y) in enumerate(pts, start=1):
        ax.add_patch(Ellipse((x, y), width=2 * r / w_px,
                             height=2 * r / h_px,
                             facecolor=st["measure"], edgecolor="none",
                             zorder=5))
        ux, uy = (x - cx) * w_px, (y - cy) * h_px
        norm = math.hypot(ux, uy)
        ux, uy = (ux / norm, uy / norm) if norm > 1e-9 else (1.0, 0.0)
        lx, ly = x + ux * off / w_px, y + uy * off / h_px
        if not (mx <= lx <= 1 - mx and my <= ly <= 1 - my):
            px_, py_ = -uy, ux               # perpendicular, toward interior
            if px_ * (0.5 - x) * w_px + py_ * (0.5 - y) * h_px < 0:
                px_, py_ = -px_, -py_
            lx, ly = x + px_ * off / w_px, y + py_ * off / h_px
        lx = min(max(lx, mx), 1 - mx)
        ly = min(max(ly, my), 1 - my)
        ax.text(lx, ly, str(n), color=st["measure"], fontsize=8,
                fontweight="bold", ha="center", va="center", zorder=6)

    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    # NOTE: fig.bbox.width can be 463.999…, and int() would truncate — a
    # 1-px stride mismatch that shears the whole image.  Ask the canvas.
    size = fig.canvas.get_width_height(physical=True)
    img = Image.frombuffer("RGBA", size, buf, "raw", "RGBA", 0, 1)
    return img.convert("RGB").resize((w_px, h_px), Image.LANCZOS)
