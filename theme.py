"""theme.py — Thrusty "Journal / Minimal" visual system (V1).

Single source of truth for the UI's colors, fonts, ttk styles and Matplotlib
styling (UI_DESIGN_SPEC.md).  SOLELY visual: nothing here changes behavior,
numbers, or verdict logic.

V1 scope (the "global coat"):
  * color tokens + two-font rule (sans for text, mono for numbers),
    applied app-wide by reconfiguring Tk's NAMED fonts — every existing
    widget inherits without call-site changes;
  * `clam` ttk theme restyled flat/white/hairline;
  * Matplotlib rcParams + style helpers (ink primary, red-orange secondary
    axis) — color/spine/grid only, per-plot font SIZES untouched (the 6-plot
    grid uses deliberately small type);
  * brand stripe helper.

SEMANTIC COLORS — EXEMPT FROM THE PALETTE RULES.  The four survival-tier
colors (survivability_report.SURVIVAL_TIERS: green=experience, blue=design,
yellow=beyond, red=fail) are the evidence ladder (METHODS §13.5), not
decoration.  They are owned by survivability_report.py and are deliberately
NOT retoned here; tier-blue is a verdict, not an accent.
"""

from __future__ import annotations

# ── Color tokens ────────────────────────────────────────────────────────────
INK       = "#1a1a1a"   # primary text, primary plot series + left axis
SUB       = "#9a9a9a"   # labels, captions, x-axis ticks, muted text
ACCENT    = "#334155"   # interactive: buttons, active tab, chevrons, links
ACCENT2   = "#cf5a2e"   # SECONDARY PLOT AXIS ONLY (right axis + dashed series)
LINE      = "#ececec"   # hairline separators, faint borders
UNDERLINE = "#cfcfcf"   # control underlines (V2)
RULE      = "#1a1a1a"   # header underline, plot axis lines
GRID      = "#f3f3f3"   # plot gridlines
BG        = "#ffffff"   # all surfaces
RAIL_BG   = "#fbfcfe"   # left control rail
RED       = "#9a3535"   # negative deltas, error/warning text
GREEN     = "#3f6b4f"   # positive / OK status (muted)
TINT_KEY    = "#eef4ff" # timeline key-event rows (Ignition/Apogee/Impact)
TINT_DEBRIS = "#fff7e8" # timeline debris / empty-impact rows
TINT_ZEBRA  = "#fafafa" # odd-row zebra stripe

# Grayscale-forward line cycle for multi-series plots.  ACCENT2 is reserved
# for twin-axis series and is assigned explicitly at the call sites.
PLOT_CYCLE = [INK, ACCENT, SUB, GREEN, RED]

# ── Fonts ───────────────────────────────────────────────────────────────────
# Preferred faces; fall back to system faces when IBM Plex isn't installed.
_SANS_PREF = ("IBM Plex Sans", "Helvetica Neue", "Helvetica", "Segoe UI", "Arial")
_MONO_PREF = ("IBM Plex Mono", "Menlo", "Consolas", "Courier New")
SANS = _SANS_PREF[1]    # resolved by apply_theme() against installed families
MONO = _MONO_PREF[1]


def sans(sz, w="normal"):
    return (SANS, sz, w) if w != "normal" else (SANS, sz)


def num(sz, w="normal"):
    return (MONO, sz, w) if w != "normal" else (MONO, sz)


def _resolve_fonts(root):
    """Pick the first installed face from each preference list."""
    global SANS, MONO
    import tkinter.font as tkfont
    try:
        fams = set(tkfont.families(root))
    except Exception:
        return
    for f in _SANS_PREF:
        if f in fams:
            SANS = f
            break
    for f in _MONO_PREF:
        if f in fams:
            MONO = f
            break


# ── Tk application theme ────────────────────────────────────────────────────
def apply_theme(root):
    """Apply the visual system to a Tk root.  Call once, early in __init__
    (before widgets are built).  Returns the ttk.Style."""
    import tkinter as tk
    import tkinter.font as tkfont
    from tkinter import ttk

    _resolve_fonts(root)

    # Named fonts: every widget that never asked for a font inherits these —
    # the whole app's typography in a few lines, no call-site churn.
    # Families only; sizes are left at platform defaults so nothing reflows.
    for name, family in (("TkDefaultFont", SANS), ("TkTextFont", SANS),
                         ("TkMenuFont", SANS), ("TkHeadingFont", SANS),
                         ("TkTooltipFont", SANS), ("TkCaptionFont", SANS),
                         ("TkFixedFont", MONO)):
        try:
            tkfont.nametofont(name).configure(family=family)
        except tk.TclError:
            pass

    root.configure(bg=BG)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")     # the only theme that recolors freely
    except tk.TclError:
        return style

    style.configure(".", background=BG, foreground=INK,
                    bordercolor=LINE, lightcolor=BG, darkcolor=BG,
                    troughcolor=LINE, focuscolor=ACCENT,
                    selectbackground=ACCENT, selectforeground="#ffffff")
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=INK)

    # Group frames: hairline card, muted small-caps-ish title.
    style.configure("TLabelframe", background=BG, bordercolor=LINE,
                    relief="solid", borderwidth=1)
    style.configure("TLabelframe.Label", background=BG, foreground=SUB)

    # Buttons: quiet outlined; hover = faint fill.
    style.configure("TButton", background=BG, foreground=ACCENT,
                    bordercolor="#c9cfd8", borderwidth=1, relief="solid",
                    focusthickness=0, padding=(10, 4))
    style.map("TButton",
              background=[("disabled", BG), ("active", "#f2f4f6")],
              foreground=[("disabled", "#c8c8c8")],
              bordercolor=[("disabled", LINE)])
    style.configure("Primary.TButton", background=ACCENT, foreground="#ffffff",
                    bordercolor=ACCENT)
    style.map("Primary.TButton",
              background=[("disabled", "#b5bcc7"), ("active", "#26303f")],
              foreground=[("disabled", "#ffffff")])

    # Entries / spinboxes / combobox: flat white field, hairline border.
    style.configure("TEntry", fieldbackground=BG, bordercolor=UNDERLINE,
                    lightcolor=BG, darkcolor=BG, insertcolor=INK, padding=2)
    style.configure("TSpinbox", fieldbackground=BG, bordercolor=UNDERLINE,
                    arrowcolor=ACCENT, padding=2)
    style.configure("TCombobox", fieldbackground=BG, background=BG,
                    foreground=INK, arrowcolor=ACCENT,
                    bordercolor=UNDERLINE, lightcolor=BG, darkcolor=BG,
                    padding=(4, 2))
    style.map("TCombobox",
              fieldbackground=[("readonly", BG)],
              foreground=[("disabled", SUB)])
    try:
        root.option_add("*TCombobox*Listbox.background", BG)
        root.option_add("*TCombobox*Listbox.foreground", INK)
        root.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
    except tk.TclError:
        pass

    style.configure("TCheckbutton", background=BG, foreground=INK)
    style.configure("TRadiobutton", background=BG, foreground=INK)
    style.configure("TSeparator", background=LINE)

    # Notebook: underline-style tabs (color carries the state; clam can't do
    # a per-tab bottom border, so active = ink-on-white, inactive = muted).
    style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(8, 6, 8, 0))
    style.configure("TNotebook.Tab", background=BG, foreground="#a5a5a5",
                    borderwidth=0, padding=(12, 6))
    style.map("TNotebook.Tab",
              foreground=[("selected", INK)],
              background=[("selected", BG)],
              expand=[("selected", (0, 0, 0, 0))])

    # Tables: white field, muted flat headings, roomier rows.
    style.configure("Treeview", background=BG, fieldbackground=BG,
                    foreground=INK, rowheight=24, borderwidth=0)
    style.configure("Treeview.Heading", background=BG, foreground=SUB,
                    relief="flat", borderwidth=0, padding=(4, 4))
    style.map("Treeview.Heading", background=[("active", BG)])
    style.map("Treeview",
              background=[("selected", "#e7ecf3")],
              foreground=[("selected", INK)])

    # Scrollbars: slim, flat, no arrows-color shouting.
    for orient in ("Vertical", "Horizontal"):
        style.configure(f"{orient}.TScrollbar", background="#e3e5e8",
                        troughcolor=BG, bordercolor=BG,
                        arrowcolor=SUB, relief="flat")

    style.configure("TPanedwindow", background=LINE)

    return style


def brand_stripe(parent):
    """The 3px accent→accent2 gradient stripe (top of the window)."""
    import tkinter as tk
    c = tk.Canvas(parent, height=3, highlightthickness=0, bd=0, bg=ACCENT)

    def _paint(event=None):
        c.delete("all")
        w = max(c.winfo_width(), 1)
        a = tuple(int(ACCENT[i:i + 2], 16) for i in (1, 3, 5))
        b = tuple(int(ACCENT2[i:i + 2], 16) for i in (1, 3, 5))
        steps = 60
        seg = max(w // steps, 1)
        for i in range(steps + 1):
            f = i / steps
            col = "#%02x%02x%02x" % tuple(int(a[j] + (b[j] - a[j]) * f)
                                          for j in range(3))
            c.create_rectangle(i * seg, 0, (i + 1) * seg, 3,
                               fill=col, outline=col)
    c.bind("<Configure>", _paint)
    return c


# ── Matplotlib ──────────────────────────────────────────────────────────────
def apply_matplotlib():
    """Journal rcParams: colors/spines/grid only — no size overrides, so the
    dense 6-plot grid keeps its deliberately small type."""
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["IBM Plex Sans", "Helvetica Neue", "Helvetica", "Arial"],
        "axes.edgecolor": INK,
        "axes.linewidth": 1.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlecolor": INK,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1.0,
        "xtick.color": SUB,
        "ytick.color": INK,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.prop_cycle": cycler(color=PLOT_CYCLE),
        "legend.framealpha": 0.88,
        "legend.edgecolor": LINE,
    })


def style_secondary(ax2):
    """Recolor an existing twin (right) axis to the reserved secondary color.
    The right spine is re-shown in ACCENT2 (rcParams hide it by default)."""
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(ACCENT2)
    ax2.tick_params(axis="y", colors=ACCENT2)
    ax2.yaxis.label.set_color(ACCENT2)
    return ax2
