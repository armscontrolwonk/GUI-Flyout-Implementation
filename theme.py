"""theme.py — plot styling extracted from the design mockup (slice 1).

Source of truth: design/thrusty-mockup.html (Thrusty.dc.html), the approved
visual target.  This module styles PLOTS ONLY — widgets are native (the V1
clam restyle was tried on screen and rolled back; UI_DESIGN_SPEC.md records
that history).  Recipe measured from the mockup's own SVGs and styles:

  * plot area: full light frame (1px #d8d8d8, all four spines) over a faint
    #f3f3f3 interior grid — NOT the hide-top/right treatment the spec text
    described; the mockup as approved uses the framed look;
  * primary series: #1a1a1a solid 2px, no fill under curves;
  * secondary (twin-axis) series: #cf5a2e dashed (5,4) 1.8px; its tick
    labels + axis title in the same color (the frame stays light);
  * tick numbers in a monospace face; left-axis label ink, x-label muted;
  * legend: white @88%, 1px #ececec hairline.

Colors are assigned EXPLICITLY per series at the call sites — no global
prop_cycle (a cycle leaked surprise colors into multi-series plots once;
never again).  Survival-tier colors remain owned by survivability_report.py
(semantic, METHODS §13.5) and are not defined here.
"""

from __future__ import annotations

# ── Tokens (mockup :root variables) ─────────────────────────────────────────
INK     = "#1a1a1a"
SUB     = "#9a9a9a"
ACCENT  = "#334155"
ACCENT2 = "#cf5a2e"   # secondary (twin) axis + its dashed series — only use
LINE    = "#ececec"   # hairlines (legend border)
FRAME   = "#d8d8d8"   # plot-area frame (--line2)
GRID    = "#f3f3f3"   # interior gridlines
BG      = "#ffffff"
RED     = "#9a3535"
GREEN   = "#3f6b4f"

DASH = (0, (5, 4))    # the mockup's secondary-series dash pattern

_MONO_STACK = ["IBM Plex Mono", "Menlo", "Consolas", "DejaVu Sans Mono"]
_SANS_STACK = ["IBM Plex Sans", "Helvetica Neue", "Helvetica", "Arial",
               "DejaVu Sans"]


def apply_matplotlib():
    """Mockup plot chrome via rcParams: framed axes, faint grid, white
    surfaces.  No prop_cycle override and no font-size overrides — the 6-plot
    grid keeps its deliberately small type, and series colors are explicit."""
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.sans-serif": _SANS_STACK,
        "font.monospace": _MONO_STACK,
        "axes.edgecolor": FRAME,
        "axes.linewidth": 1.0,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.titlecolor": INK,
        "axes.titleweight": 600,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "xtick.color": SUB,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.framealpha": 0.88,
        "legend.edgecolor": LINE,
    })


def style_twin(ax2):
    """Secondary (right) axis: tick labels + title in ACCENT2; the frame
    itself stays light like the mockup."""
    ax2.tick_params(axis="y", colors=ACCENT2)
    ax2.yaxis.label.set_color(ACCENT2)
    return ax2


def mono_ticks(fig):
    """Tick numbers in the monospace face (mockup: numbers are always mono).
    Called after a redraw populates the tick labels."""
    for ax in fig.axes:
        for t in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            t.set_fontfamily("monospace")


# ── Slice 2: status strip (mockup recipe) ───────────────────────────────────
# Row: white, bottom hairline; each metric = UPPERCASE muted sans label +
# mono ink value (label 10px, value 12px, 16px between metrics, 5px inside).
# Right: state pill — #f1f3f5, 1px #dcdcdc, radius 20, dot + 11px label.
PILL_BG     = "#f1f3f5"
PILL_BORDER = "#dcdcdc"
PILL_TEXT   = ACCENT

_UI_FONTS = {}


def _ui_family(widget, kind):
    """Resolve (and cache) the installed sans/mono family for Tk widgets."""
    if kind not in _UI_FONTS:
        try:
            import tkinter.font as tkfont
            fams = set(tkfont.families(widget))
        except Exception:
            fams = set()
        stack = _SANS_STACK if kind == "sans" else _MONO_STACK
        _UI_FONTS[kind] = next((f for f in stack if f in fams),
                               "Helvetica" if kind == "sans" else "Menlo")
    return _UI_FONTS[kind]


def ui_sans(widget, size, bold=False):
    fam = _ui_family(widget, "sans")
    return (fam, size, "bold") if bold else (fam, size)


def ui_mono(widget, size, bold=False):
    fam = _ui_family(widget, "mono")
    return (fam, size, "bold") if bold else (fam, size)


def _round_rect(canvas, x0, y0, x1, y1, r, **kw):
    pts = [x0+r,y0, x1-r,y0, x1,y0, x1,y0+r, x1,y1-r, x1,y1, x1-r,y1,
           x0+r,y1, x0,y1, x0,y1-r, x0,y0+r, x0,y0, x0+r,y0]
    return canvas.create_polygon(pts, smooth=True, **kw)


class StatusPill:
    """The mockup's state pill: rounded chip, colored dot, short label."""

    def __init__(self, parent):
        import tkinter as tk
        self._c = tk.Canvas(parent, height=24, highlightthickness=0,
                            bd=0, bg=BG)
        self._text = "Ready — press Run Flyout"
        self._dot = SUB
        self._c.bind("<Configure>", lambda _e: self._paint())
        self.set(self._text, self._dot)

    def widget(self):
        return self._c

    def set(self, text, dot_color):
        self._text = text
        self._dot = dot_color
        self._paint()

    def _paint(self):
        c = self._c
        c.delete("all")
        f = ui_sans(c, 10)
        w = 30 + c.tk.call("font", "measure", f, self._text)
        c.configure(width=w + 2)
        _round_rect(c, 1, 1, w, 23, 11, fill=PILL_BG, outline=PILL_BORDER)
        c.create_oval(10, 9, 17, 16, fill=self._dot, outline=self._dot)
        c.create_text(23, 12, text=self._text, anchor="w",
                      font=f, fill=PILL_TEXT)
