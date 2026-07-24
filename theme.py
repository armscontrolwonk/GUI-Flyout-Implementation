"""theme.py — Thrusty visual tokens + Matplotlib "journal" plot style.

OUTCOME OF THE V1 OVERHAUL (2026-07-24): the ttk/`clam` widget restyle and
named-font retypography were tried and ROLLED BACK — replacing native macOS
aqua widgets with clam's flat approximations read as worse, not better, and
no amount of later polish closes that gap on Tk.  What SURVIVES is the part
of the design the toolkit can fully deliver:

  * color tokens (single source of truth for any styling that remains);
  * the Matplotlib journal style — ink primary curves on the left axis,
    ONE reserved red-orange for every secondary (twin) axis and its dashed
    series, hidden top/right spines, faint grid behind the data;
  * timeline row tints.

Widgets are native.  UI_DESIGN_SPEC.md records the full design and the
rollback decision.

SEMANTIC COLORS — EXEMPT.  The four survival-tier colors
(survivability_report.SURVIVAL_TIERS: green=experience, blue=design,
yellow=beyond, red=fail) are the evidence ladder (METHODS §13.5), owned by
survivability_report.py, and deliberately not defined or retoned here.
"""

from __future__ import annotations

# ── Color tokens ────────────────────────────────────────────────────────────
INK       = "#1a1a1a"   # primary text, primary plot series + left axis
SUB       = "#9a9a9a"   # muted labels, x-axis ticks
ACCENT    = "#334155"   # slate — interactive emphasis (sparingly)
ACCENT2   = "#cf5a2e"   # SECONDARY PLOT AXIS ONLY (right axis + dashed series)
LINE      = "#ececec"   # hairlines
GRID      = "#f3f3f3"   # plot gridlines
BG        = "#ffffff"
RED       = "#9a3535"   # negative deltas, error/warning text
GREEN     = "#3f6b4f"   # positive / OK status (muted)
TINT_KEY    = "#eef4ff" # timeline key-event rows (Ignition/Apogee/Impact)
TINT_DEBRIS = "#fff7e8" # timeline debris / empty-impact rows
TINT_ZEBRA  = "#fafafa" # odd-row zebra stripe

# Grayscale-forward line cycle for multi-series plots.  ACCENT2 is reserved
# for twin-axis series and is assigned explicitly at the call sites.
PLOT_CYCLE = [INK, ACCENT, SUB, GREEN, RED]


# ── Matplotlib ──────────────────────────────────────────────────────────────
def apply_matplotlib():
    """Journal rcParams: colors/spines/grid only — no size overrides, so the
    dense 6-plot grid keeps its deliberately small type."""
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
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
