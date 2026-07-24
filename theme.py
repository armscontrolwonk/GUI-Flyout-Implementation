"""theme.py — plot styling extracted from the design mockup (slice 1).

Source of truth: design/thrusty-mockup.html (Thrusty.dc.html), the approved
visual target.  This module styles PLOTS ONLY — widgets are native (the V1
clam restyle was tried on screen and rolled back; UI_DESIGN_SPEC.md records
that history).  Recipe measured from the mockup's own SVGs and styles:

  * plot area: full light frame (1px #d8d8d8, all four spines) over a faint
    #f3f3f3 interior grid — NOT the hide-top/right treatment the spec text
    described; the mockup as approved uses the framed look;
  * primary series: #1a1a1a solid 2px, no fill under curves;
  * secondary (twin-axis) series: dashed (5,4) 1.8px in ACCENT2; its tick
    labels + axis title in the same color (the frame stays light).  ACCENT2
    was the mockup's red-orange #cf5a2e; unified 2026-07-24 (user decision)
    to the mockup's own muted brick red #9a3535 so the right axes, the map's
    impact/debris markers, and warning text share ONE warm accent;
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
ACCENT2 = "#9a3535"   # THE warm accent: secondary axes/series, map markers
LINE    = "#ececec"   # hairlines (legend border)
FRAME   = "#d8d8d8"   # plot-area frame (--line2)
GRID    = "#f3f3f3"   # interior gridlines
BG      = "#ffffff"
RED     = ACCENT2     # unified 2026-07-24: one brick/cardinal accent
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
        f = ui_sans(c, FS["pill"])
        w = 30 + c.tk.call("font", "measure", f, self._text)
        c.configure(width=w + 2)
        _round_rect(c, 1, 1, w, 23, 11, fill=PILL_BG, outline=PILL_BORDER)
        c.create_oval(10, 9, 17, 16, fill=self._dot, outline=self._dot)
        c.create_text(23, 12, text=self._text, anchor="w",
                      font=f, fill=PILL_TEXT)


# ── Slice 3: rail card components (mockup recipe) ───────────────────────────
# Card: white group, 1px hairline, UPPERCASE muted section label.
# Underline field: borderless entry over a 1px #cfcfcf rule, mono digits.
# Link button: underlined accent text, hover→ink, disabled #c8c8c8 no rule.
UNDERLINE = "#cfcfcf"
DISABLED  = "#c8c8c8"

# Font sizes (pt) — tuned to match the original app's density (user call:
# the first card render was too large; CSS px ≈ 0.75 pt).
FS = {"section": 9, "label": 10, "field": 10, "unit": 9, "link": 10,
      "drop": 10, "strip_label": 8, "strip_value": 10, "pill": 9}


def card(parent, title):
    """Rail group card.  Returns (outer, body): pack `outer`; build fields
    into `body`."""
    import tkinter as tk
    outer = tk.Frame(parent, bg=BG, bd=0, highlightthickness=1,
                     highlightbackground=LINE, highlightcolor=LINE)
    body = tk.Frame(outer, bg=BG)
    body.pack(fill="both", expand=True, padx=10, pady=(8, 10))
    tk.Label(body, text=title.upper(), bg=BG, fg=SUB,
             font=ui_sans(parent, FS["section"], bold=True)).pack(anchor="w", pady=(0, 6))
    return outer, body


def underline_entry(parent, textvariable, width=10, mono=True):
    """Borderless entry + 1px underline (the mockup's field).  Returns
    (wrap, entry)."""
    import tkinter as tk
    wrap = tk.Frame(parent, bg=BG)
    e = tk.Entry(wrap, textvariable=textvariable, bd=0, relief="flat",
                 bg=BG, fg=INK, insertbackground=INK, highlightthickness=0,
                 width=width,
                 font=ui_mono(parent, FS["field"]) if mono
                      else ui_sans(parent, FS["field"]))
    e.pack(fill="x", ipady=3)
    tk.Frame(wrap, bg=UNDERLINE, height=1).pack(fill="x")
    return wrap, e


class LinkButton:
    """Underlined text link standing in for a button.  Supports
    .config(state=tk.NORMAL/tk.DISABLED) like the ttk.Buttons it replaces."""

    def __init__(self, parent, text, command=None, size=None):
        size = FS["link"] if size is None else size
        import tkinter as tk
        self._tk = tk
        self._f  = ui_sans(parent, size)
        self._fu = self._f + ("underline",)
        self._command = command
        self._enabled = True
        self._lbl = tk.Label(parent, text=text, bg=BG, fg=ACCENT,
                             font=self._fu, cursor="hand2")
        self._lbl.bind("<Button-1>", self._click)
        self._lbl.bind("<Enter>", lambda _e: self._enabled
                       and self._lbl.config(fg=INK))
        self._lbl.bind("<Leave>", lambda _e: self._enabled
                       and self._lbl.config(fg=ACCENT))

    # pack/grid passthroughs so it drops in where a Button was
    def pack(self, **kw):  self._lbl.pack(**kw);  return self
    def grid(self, **kw):  self._lbl.grid(**kw);  return self

    def config(self, **kw):
        state = kw.pop("state", None)
        if state is not None:
            self._enabled = str(state) != "disabled"
            if self._enabled:
                self._lbl.config(fg=ACCENT, font=self._fu, cursor="hand2")
            else:
                self._lbl.config(fg=DISABLED, font=self._f, cursor="arrow")
        if kw:
            self._lbl.config(**kw)
    configure = config

    def _click(self, _e):
        if self._enabled and self._command:
            self._command()



class Dropdown:
    """The mockup's underline dropdown: current value + accent chevron over a
    1px rule; click opens a borderless popup list (scroll + type-to-jump).
    Drop-in for readonly ttk.Combobox call sites: .config(values=...) works,
    `command` fires on selection (in place of <<ComboboxSelected>>), and the
    shared textvariable carries the value.  Separator rows (starting '─')
    are shown but not selectable."""

    def __init__(self, parent, textvariable, values=(), command=None,
                 size=None):
        import tkinter as tk
        self._tkmod = tk
        self._var = textvariable
        self._values = list(values)
        self._command = command
        self._size = FS["drop"] if size is None else size
        self._outer = tk.Frame(parent, bg=BG)
        row = tk.Frame(self._outer, bg=BG)
        row.pack(fill="x")
        self._val = tk.Label(row, textvariable=self._var, anchor="w", bg=BG,
                             fg=INK, font=ui_sans(parent, self._size),
                             cursor="hand2")
        self._val.pack(side="left", fill="x", expand=True, ipady=3, padx=(2, 0))
        self._chev = tk.Label(row, text="▾", bg=BG, fg=ACCENT,
                              font=ui_sans(parent, self._size), cursor="hand2")
        self._chev.pack(side="right", padx=(0, 3))
        tk.Frame(self._outer, bg=UNDERLINE, height=1).pack(fill="x")
        for w in (self._val, self._chev):
            w.bind("<Button-1>", lambda _e: self._open())
        self._pop = None

    def pack(self, **kw):  self._outer.pack(**kw);  return self
    def grid(self, **kw):  self._outer.grid(**kw);  return self

    def config(self, values=None, state=None, **_kw):
        if values is not None:
            self._values = list(values)
    configure = config

    def __setitem__(self, key, val):
        if key == "values":
            self._values = list(val)

    def set(self, value):  self._var.set(value)
    def get(self):         return self._var.get()

    def _close(self):
        if self._pop is not None:
            try:
                self._pop.destroy()
            except Exception:
                pass
            self._pop = None

    def _open(self):
        tk = self._tkmod
        if self._pop is not None:
            self._close()
            return
        x = self._outer.winfo_rootx()
        y = self._outer.winfo_rooty() + self._outer.winfo_height()
        w = max(self._outer.winfo_width(), 120)
        pop = tk.Toplevel(self._outer)
        pop.overrideredirect(True)
        pop.geometry(f"+{x}+{y}")
        frame = tk.Frame(pop, bg=BG, highlightthickness=1,
                         highlightbackground=UNDERLINE)
        frame.pack()
        n = len(self._values)
        lb = tk.Listbox(frame, height=min(max(n, 1), 16), activestyle="none",
                        font=ui_sans(self._outer, self._size), bg=BG, fg=INK,
                        bd=0, highlightthickness=0, relief="flat",
                        selectbackground="#e7ecf3", selectforeground=INK,
                        width=max(w // max(self._size - 3, 6), 24))
        for v in self._values:
            lb.insert("end", v)
        if n > 16:
            sb = tk.Scrollbar(frame, orient="vertical", command=lb.yview)
            lb.configure(yscrollcommand=sb.set)
            sb.pack(side="right", fill="y")
        lb.pack(side="left", fill="both", expand=True)
        cur = self._var.get()
        if cur in self._values:
            i = self._values.index(cur)
            lb.selection_set(i)
            lb.see(i)

        def _choose(_e=None):
            sel = lb.curselection()
            if sel:
                v = lb.get(sel[0])
                if not str(v).startswith("─"):
                    self._var.set(v)
                    self._close()
                    if self._command:
                        self._command()
                    return
            self._close()

        _buf = {"s": "", "after": None}

        def _type(e):
            if e.keysym in ("Return", "KP_Enter"):
                _choose(); return
            if e.keysym == "Escape":
                self._close(); return
            ch = e.char
            if not ch or not ch.isprintable():
                return
            if _buf["after"]:
                lb.after_cancel(_buf["after"])
            _buf["s"] += ch.lower()
            _buf["after"] = lb.after(800, lambda: _buf.update(s="", after=None))
            for i, v in enumerate(self._values):
                if str(v).lower().startswith(_buf["s"]):
                    lb.selection_clear(0, "end")
                    lb.selection_set(i)
                    lb.see(i)
                    break

        lb.bind("<ButtonRelease-1>", _choose)
        lb.bind("<Key>", _type)
        pop.bind("<FocusOut>", lambda _e: self._close())
        pop.bind("<Escape>", lambda _e: self._close())
        self._pop = pop
        lb.focus_set()
