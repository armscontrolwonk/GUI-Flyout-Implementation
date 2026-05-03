"""
Thrusty — Python/tkinter port of Forden's MATLAB GUIDE application.

Layout mirrors the original MATLAB GUIDE application:
  Left panel  : missile type, units, launch site (decimal °), target (decimal °),
                cutoff time, run buttons, range/apogee results
  Right panel : 4-up matplotlib plots (altitude, speed, trajectory, ground track)
  Bottom bar  : status line
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import tkinter.font as tkfont
import copy
import threading
import numpy as np
import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

import matplotlib.ticker

from missile_models import (MISSILE_DB, get_missile,
                           missile_to_dict, missile_from_dict,
                           total_burn_time, tumbling_cylinder_beta,
                           NOSE_SHAPES, NOSE_SHAPE_LABELS,
                           GRAIN_LABELS, grain_fill_factor, _GRAIN_FILL_RANGE)
from trajectory import (integrate_trajectory, maximize_range, aim_missile,
                        plan_orbital_insertion)
from coordinates import range_between
from slv_performance import schilling_performance

# ---------------------------------------------------------------------------
# Country border map data (Natural Earth 110m, bundled GeoJSON)
# ---------------------------------------------------------------------------

_BORDERS_CACHE = None   # loaded once on first draw, then reused

def _open_file(path: str) -> None:
    """Open a file with the default viewer, cross-platform."""
    import subprocess
    try:
        if sys.platform == 'darwin':
            subprocess.Popen(['open', path])
        elif sys.platform == 'win32':
            os.startfile(path)
        else:
            subprocess.Popen(['xdg-open', path])
    except Exception:
        pass


def _load_borders():
    """Return the Natural Earth 110m country GeoJSON (lazy, cached)."""
    global _BORDERS_CACHE
    if _BORDERS_CACHE is None:
        p = Path(__file__).parent / "data" / "ne_110m_countries.geojson"
        if p.exists():
            _BORDERS_CACHE = json.loads(p.read_text())
    return _BORDERS_CACHE


def _draw_borders(ax, center_lon):
    """Overlay country borders on *ax*, re-centred on *center_lon* degrees."""
    data = _load_borders()
    if data is None:
        return

    def _plot_ring(ring):
        # Shift every vertex into [−180, +180] relative to center_lon
        xs = [((pt[0] - center_lon + 180.0) % 360.0) - 180.0 for pt in ring]
        ys = [pt[1] for pt in ring]
        # Split the ring wherever it still crosses ±180° in centred space
        seg_x, seg_y = [[]], [[]]
        for i in range(len(xs)):
            if i > 0 and abs(xs[i] - xs[i - 1]) > 180:
                seg_x.append([])
                seg_y.append([])
            seg_x[-1].append(xs[i])
            seg_y[-1].append(ys[i])
        for sx, sy in zip(seg_x, seg_y):
            if len(sx) > 1:
                ax.plot(sx, sy, color='#777777', linewidth=0.4,
                        solid_capstyle='round', zorder=1)

    for feature in data.get('features', []):
        geom  = feature.get('geometry') or {}
        gtype = geom.get('type', '')
        coords = geom.get('coordinates', [])
        if gtype == 'Polygon':
            for ring in coords:
                _plot_ring(ring)
        elif gtype == 'MultiPolygon':
            for polygon in coords:
                for ring in polygon:
                    _plot_ring(ring)


# ---------------------------------------------------------------------------
# Custom missile persistence
# ---------------------------------------------------------------------------

# Sentinel string inserted into the missile combobox between non-Forden and
# Forden entries.  It is never a valid missile name.

# ---------------------------------------------------------------------------
# Newtonian blunted-cone Cd table — Ref (4) Ch. 5, hypersonic / zero AoA.
# Rows: half-angle 10°, 20°, 30°, 40°.
# Cols: nose-radius ratio ε = r_N/r_b  0.0, 0.2, 0.4, 0.6, 0.8, 1.0.
# ---------------------------------------------------------------------------
_BCON_THETA = [10.0, 20.0, 30.0, 40.0]
_BCON_EPS   = [0.0,  0.2,  0.4,  0.6,  0.8,  1.0]
_BCON_TABLE = [
    [0.0603, 0.063, 0.068, 0.080, 0.200, 1.00],
    [0.2340, 0.238, 0.250, 0.310, 0.540, 1.00],
    [0.5000, 0.507, 0.530, 0.600, 0.750, 1.00],
    [0.8264, 0.835, 0.860, 0.900, 0.965, 1.00],
]


def _cd_blunted_cone_newtonian(theta_deg: float, eps: float) -> float:
    """
    Cd (based on base area) for a spherically-blunted cone at zero angle of
    attack in hypersonic (Newtonian) flow.

    theta_deg : cone half-angle (degrees)
    eps       : nose-radius ratio r_N/r_b  (0 = sharp tip, 1 = hemisphere)

    For eps = 0 the exact Newtonian formula 2·sin²θ is returned.
    For other values bilinear interpolation is used on the chart table;
    the bluntness excess is scaled by the actual Cd_sharp so that angles
    outside the 10°–40° table range are handled smoothly.
    """
    import math
    th        = math.radians(max(1.0, min(float(theta_deg), 89.0)))
    cd_sharp  = 2.0 * math.sin(th) ** 2
    eps       = max(0.0, min(float(eps), 1.0))
    if eps == 0.0:
        return cd_sharp

    theta_c = max(_BCON_THETA[0], min(float(theta_deg), _BCON_THETA[-1]))
    i_th = next((i for i in range(len(_BCON_THETA) - 1)
                 if _BCON_THETA[i + 1] >= theta_c), len(_BCON_THETA) - 2)
    i_ep = next((i for i in range(len(_BCON_EPS) - 1)
                 if _BCON_EPS[i + 1] >= eps), len(_BCON_EPS) - 2)

    t_th = (theta_c - _BCON_THETA[i_th]) / (_BCON_THETA[i_th + 1] - _BCON_THETA[i_th])
    t_ep = (eps     - _BCON_EPS[i_ep])   / (_BCON_EPS[i_ep + 1]   - _BCON_EPS[i_ep])

    c = _BCON_TABLE
    cd_tbl = (c[i_th    ][i_ep    ] * (1 - t_th) * (1 - t_ep) +
              c[i_th + 1][i_ep    ] * t_th        * (1 - t_ep) +
              c[i_th    ][i_ep + 1] * (1 - t_th)  * t_ep       +
              c[i_th + 1][i_ep + 1] * t_th         * t_ep)

    # Bluntness excess at the (clamped) table half-angle
    cd_sharp_tbl = c[i_th][0] * (1 - t_th) + c[i_th + 1][0] * t_th
    bluntness    = cd_tbl - cd_sharp_tbl
    return cd_sharp + bluntness


# Names that ship with the program and cannot be deleted
_PACKAGED_NAMES: set[str] = set(MISSILE_DB.keys())
# Packaged missiles the user has overridden with custom edits
_OVERRIDDEN_PACKAGED: set[str] = set()
# Where user-created missiles are saved
_CUSTOM_PATH      = Path.home() / ".gui_missile_flyout" / "custom_missiles.json"
_TRAJ_PATH        = Path.home() / ".gui_missile_flyout" / "trajectory_profiles.json"
_EXPORT_TRAJ_DIR  = Path.home() / ".gui_missile_flyout" / "exports" / "trajectories"
_EXPORT_MISS_DIR  = Path.home() / ".gui_missile_flyout" / "exports" / "missiles"
_EXPORT_SITE_DIR  = Path.home() / ".gui_missile_flyout" / "exports" / "sites"


def _load_traj_profiles() -> dict:
    """Return saved trajectory profiles keyed by missile name."""
    if not _TRAJ_PATH.exists():
        return {}
    try:
        return json.loads(_TRAJ_PATH.read_text())
    except Exception:
        return {}


def _save_traj_profiles(profiles: dict) -> None:
    _TRAJ_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TRAJ_PATH.write_text(json.dumps(profiles, indent=2))


def _load_custom_missiles():
    """Read custom_missiles.json and register any saved missiles in MISSILE_DB."""
    if not _CUSTOM_PATH.exists():
        return
    try:
        data = json.loads(_CUSTOM_PATH.read_text())
        for name, d in data.items():
            p = missile_from_dict(d)
            MISSILE_DB[name] = lambda _p=p: _p
            if name in _PACKAGED_NAMES:
                _OVERRIDDEN_PACKAGED.add(name)
    except Exception as exc:
        print(f"Warning: could not load custom missiles: {exc}")


def _save_custom_missiles():
    """Write all non-packaged and overridden-packaged missiles to custom_missiles.json."""
    _CUSTOM_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    for name in MISSILE_DB:
        if name not in _PACKAGED_NAMES or name in _OVERRIDDEN_PACKAGED:
            data[name] = missile_to_dict(MISSILE_DB[name]())
    _CUSTOM_PATH.write_text(json.dumps(data, indent=2))


_SITE_SEPARATOR = "──────────────────────────────"


def _bind_typeahead(cb):
    """
    Prefix-typeahead via a Toplevel autocomplete popup.

    As the user types, a popup appears directly below the combobox listing
    every item whose name begins with the typed prefix (case-insensitive).
    Separator entries (starting with '─') are excluded.  Works on macOS,
    Linux, and Windows without relying on the native dropdown widget.

    Commit paths
    ────────────
    • Click an item in the popup  → select + fire <<ComboboxSelected>>
    • Enter / Tab                 → best-prefix match + fire the event
    • ↓ arrow                     → move keyboard focus into the popup list
    • Escape                      → dismiss popup, leave field unchanged
    • FocusOut (click elsewhere)  → silently snap to best match
    """
    _all   = list(cb['values'])
    _popup = [None]   # Toplevel reference (reused, not recreated)
    _lb    = [None]   # Listbox inside the popup

    cb.config(state='normal')

    def _is_sep(v): return v.startswith('─')

    def _best(prefix):
        p = prefix.lower()
        return next((v for v in _all if not _is_sep(v) and v.lower().startswith(p)), None)

    def _matches(prefix):
        p = prefix.lower()
        return [v for v in _all if not _is_sep(v) and v.lower().startswith(p)]

    # ── popup lifecycle ───────────────────────────────────────────────────

    def _dismiss(event=None):
        if _popup[0] and _popup[0].winfo_exists():
            _popup[0].withdraw()

    def _show(items):
        if not _popup[0] or not _popup[0].winfo_exists():
            pop = tk.Toplevel(cb)
            pop.wm_overrideredirect(True)
            pop.attributes('-topmost', True)
            lb = tk.Listbox(pop, selectmode=tk.SINGLE,
                            exportselection=False, activestyle='dotbox')
            sb = ttk.Scrollbar(pop, orient=tk.VERTICAL, command=lb.yview)
            lb.config(yscrollcommand=sb.set)
            lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            lb.bind('<ButtonRelease-1>', _pick)
            lb.bind('<Return>', _pick)
            lb.bind('<Escape>', lambda e: (_dismiss(), cb.focus_set()))
            _popup[0] = pop
            _lb[0]    = lb

        lb = _lb[0]
        lb.delete(0, tk.END)
        for item in items:
            lb.insert(tk.END, item)

        n = min(10, len(items))
        lb.config(height=n)
        x = cb.winfo_rootx()
        y = cb.winfo_rooty() + cb.winfo_height()
        w = max(cb.winfo_width(), 180)
        _popup[0].geometry(f'{w}x{n * 20}+{x}+{y}')
        _popup[0].deiconify()
        _popup[0].lift()

    # ── selection ─────────────────────────────────────────────────────────

    def _pick(event=None):
        lb  = _lb[0]
        sel = lb.curselection()
        if not sel and event:
            sel = (lb.nearest(event.y),)
        if sel:
            value = lb.get(sel[0])
            _dismiss()
            cb.focus_set()
            cb.set(value)
            cb['values'] = _all
            cb.event_generate('<<ComboboxSelected>>')

    def _commit_fire(event=None):
        _dismiss()
        cb['values'] = _all
        m = _best(cb.get())
        if m:
            cb.set(m)
            cb.event_generate('<<ComboboxSelected>>')

    def _commit_silent_later(event=None):
        # Delay so a listbox click can register before we snap.
        cb.after(150, _do_commit_silent)

    def _do_commit_silent():
        try:
            focused = cb.focus_get()
        except Exception:
            focused = None
        if focused is _lb[0]:
            return   # user is navigating the popup — let _pick handle it
        _dismiss()
        cb['values'] = _all
        m = _best(cb.get())
        if m:
            cb.set(m)

    def _on_selected(event=None):
        cb['values'] = _all
        _dismiss()

    # ── key handler ───────────────────────────────────────────────────────

    def _on_key(event=None):
        keysym = event.keysym if event else ''
        if keysym == 'Escape':
            _dismiss()
            return
        if keysym in ('Return', 'KP_Enter', 'Tab'):
            _commit_fire()
            return
        if keysym == 'Down':
            if _lb[0] and _popup[0] and _popup[0].winfo_exists():
                _lb[0].focus_set()
                if not _lb[0].curselection():
                    _lb[0].selection_set(0)
            return
        typed = cb.get()
        if not typed:
            _dismiss()
            return
        hits = _matches(typed)
        if hits:
            _show(hits)
        else:
            _dismiss()

    cb.bind('<KeyRelease>', _on_key)
    cb.bind('<FocusOut>',   _commit_silent_later)
    cb.bind('<<ComboboxSelected>>', _on_selected, add='+')


# Bundled sites (read-only) come from launch_sites.json in the source tree.
# User-added sites are stored separately so the bundled file stays clean.
_USER_SITES_PATH = Path.home() / ".gui_missile_flyout" / "user_sites.json"
_BUNDLED_SITE_NAMES: set = set()   # populated by _load_launch_sites()


def _load_user_sites() -> list:
    """Return list of user-defined site dicts, or [] on error/missing."""
    if not _USER_SITES_PATH.exists():
        return []
    try:
        return json.loads(_USER_SITES_PATH.read_text())
    except Exception as exc:
        print(f"Warning: could not load user_sites.json: {exc}")
        return []


def _save_user_sites(sites: list) -> None:
    _USER_SITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _USER_SITES_PATH.write_text(json.dumps(sites, indent=2))


def _load_launch_sites():
    """Return (combobox_values, name→site_dict) from bundled + user sites."""
    global _BUNDLED_SITE_NAMES
    path = Path(__file__).parent / "launch_sites.json"
    bundled = []
    if path.exists():
        try:
            bundled = json.loads(path.read_text())
        except Exception as exc:
            print(f"Warning: could not load launch_sites.json: {exc}")
    _BUNDLED_SITE_NAMES = {s["name"] for s in bundled}
    all_sites = bundled + _load_user_sites()
    by_country = {}
    for s in all_sites:
        by_country.setdefault(s["country"], []).append(s)
    values, site_map = [], {}
    for country in sorted(by_country):
        values.append(f"── {country} ──")
        for s in sorted(by_country[country], key=lambda x: x["name"]):
            values.append(s["name"])
            site_map[s["name"]] = s
    return values, site_map


# ---------------------------------------------------------------------------
# Reusable labelled entry helper
# ---------------------------------------------------------------------------

def _entry_row(parent, label, row, default, unit="", width=10):
    """Grid a Label + Entry + unit-label; return the StringVar."""
    ttk.Label(parent, text=label).grid(row=row, column=0,
                                       sticky=tk.W, padx=(6, 2), pady=2)
    var = tk.StringVar(value=default)
    inner = ttk.Frame(parent)
    inner.grid(row=row, column=1, sticky=tk.W, padx=(0, 6), pady=2)
    ttk.Entry(inner, textvariable=var, width=width).pack(side=tk.LEFT)
    if unit:
        ttk.Label(inner, text=unit).pack(side=tk.LEFT, padx=(2, 0))
    return var


# ---------------------------------------------------------------------------
# Stage sub-frame used inside MissileDialog
# ---------------------------------------------------------------------------

class _StageFrame(ttk.LabelFrame):
    """Entry widgets for one rocket stage."""

    # Default thrust derived from default prop/isp/burn:
    # T = 230 × 9.80665 × (5000−1500) / 70 ≈ 112.9 kN
    _DEFAULTS = dict(fueled="5000", dry="1500", dia="0.88",
                     length="12.0", thrust_kn="112.9", isp="230",
                     nozzle_area="0", coast="0")

    _G0 = 9.80665  # m/s²

    def __init__(self, parent, label, stage_num=1, defaults=None):
        super().__init__(parent, text=label)
        self._stage_num = stage_num
        d = {**self._DEFAULTS, **(defaults or {})}
        self._fueled      = _entry_row(self, "Fueled mass (kg):",    0, d["fueled"],      "kg")
        self._dry         = _entry_row(self, "Dry mass (kg):",       1, d["dry"],         "kg")
        self._dia         = _entry_row(self, "Diameter (m):",        2, d["dia"],         "m")
        self._length      = _entry_row(self, "Length (m):",          3, d["length"],      "m")
        # Thrust row (row 4) with Suggest button
        self._thrust_lbl = ttk.Label(self, text="Thrust (kN):")
        self._thrust_lbl.grid(row=4, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._thrust_kn = tk.StringVar(value=d["thrust_kn"])
        _thr_inner = ttk.Frame(self)
        _thr_inner.grid(row=4, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        ttk.Entry(_thr_inner, textvariable=self._thrust_kn, width=10).pack(side=tk.LEFT)
        ttk.Label(_thr_inner, text="kN").pack(side=tk.LEFT, padx=(2, 6))
        if self._stage_num == 1:
            ttk.Button(_thr_inner, text="Estimate…",
                       command=self._suggest_thrust).pack(side=tk.LEFT)
        self._isp         = _entry_row(self, "Isp (vacuum, s):",      5, d["isp"],         "s")
        # Nozzle exit area — entry + Suggest button (row 6)
        ttk.Label(self, text="Nozzle exit area (m²):").grid(
            row=6, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._nozzle_area = tk.StringVar(value=d["nozzle_area"])
        _noz_inner = ttk.Frame(self)
        _noz_inner.grid(row=6, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        ttk.Entry(_noz_inner, textvariable=self._nozzle_area, width=10).pack(side=tk.LEFT)
        ttk.Label(_noz_inner, text="m²").pack(side=tk.LEFT, padx=(2, 6))
        if self._stage_num == 1:
            ttk.Button(_noz_inner, text="Estimate…",
                       command=self._suggest_nozzle_area).pack(side=tk.LEFT)

        # Burn time (row 7) — readonly/computed for liquid; user-entered for solid.
        ttk.Label(self, text="Burn time (s):").grid(
            row=7, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._burn_var = tk.StringVar()
        _burn_inner = ttk.Frame(self)
        _burn_inner.grid(row=7, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._burn_entry = ttk.Entry(_burn_inner, textvariable=self._burn_var,
                                     width=10, state="readonly")
        self._burn_entry.pack(side=tk.LEFT)
        self._burn_hint_lbl = ttk.Label(_burn_inner, text="s  (computed)",
                                        foreground="gray50")
        self._burn_hint_lbl.pack(side=tk.LEFT, padx=(2, 0))

        # Solid motor checkbox (row 9) — coast time moved to advanced pitch panel
        self._solid_motor_var = tk.BooleanVar(value=False)
        self._solid_motor_check = ttk.Checkbutton(
            self, text="Solid rocket motor (cannot be shut off)",
            variable=self._solid_motor_var,
            command=self._on_solid_toggled)
        self._solid_motor_check.grid(
            row=9, column=0, columnspan=2, sticky=tk.W, padx=(6, 2), pady=(2, 4))

        # ── Solid grain profile block (row 10) — hidden until solid is checked ──
        _GRAIN_KEYS   = list(GRAIN_LABELS.keys())
        _GRAIN_LABELS = [GRAIN_LABELS[k] for k in _GRAIN_KEYS]
        self._grain_keys = _GRAIN_KEYS

        self._solid_frame = ttk.LabelFrame(self, text="Grain profile")
        self._solid_frame.grid(row=10, column=0, columnspan=2,
                               sticky=tk.EW, padx=4, pady=(0, 4))
        self._solid_frame.columnconfigure(1, weight=1)
        self._solid_frame.grid_remove()

        # Row 0: grain type selector
        ttk.Label(self._solid_frame, text="Grain type:").grid(
            row=0, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._grain_var = tk.StringVar(value=GRAIN_LABELS["star"])
        self._grain_cb = ttk.Combobox(self._solid_frame, textvariable=self._grain_var,
                                      values=_GRAIN_LABELS, state="readonly", width=28)
        self._grain_cb.current(3)   # star
        self._grain_cb.grid(row=0, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._grain_cb.bind("<<ComboboxSelected>>", self._on_grain_changed)

        # Row 1: thrust specification toggle (peak vs average)
        ttk.Label(self._solid_frame, text="Specify:").grid(
            row=1, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        _tmode_f = ttk.Frame(self._solid_frame)
        _tmode_f.grid(row=1, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._thrust_mode_var = tk.StringVar(value="average")
        ttk.Radiobutton(_tmode_f, text="Peak thrust",
                        variable=self._thrust_mode_var, value="peak",
                        command=self._on_thrust_mode_changed).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(_tmode_f, text="Average thrust",
                        variable=self._thrust_mode_var, value="average",
                        command=self._on_thrust_mode_changed).pack(side=tk.LEFT)

        # Row 2: computed alternate thrust
        ttk.Label(self._solid_frame, text="Computed:").grid(
            row=2, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._alt_thrust_lbl = ttk.Label(self._solid_frame, text="—",
                                         foreground="navy")
        self._alt_thrust_lbl.grid(row=2, column=1, sticky=tk.W, padx=(0, 6), pady=2)

        # Row 3: boost-phase fraction (two-phase grains only)
        self._boost_frac_lbl = ttk.Label(self._solid_frame, text="Boost phase:")
        self._boost_frac_lbl.grid(row=3, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        _boost_f = ttk.Frame(self._solid_frame)
        _boost_f.grid(row=3, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._boost_frac_var = tk.StringVar(value="35")
        ttk.Entry(_boost_f, textvariable=self._boost_frac_var,
                  width=6).pack(side=tk.LEFT)
        ttk.Label(_boost_f, text="% of burn time").pack(side=tk.LEFT, padx=(4, 0))
        self._boost_frac_inner = _boost_f
        self._boost_frac_lbl.grid_remove()
        self._boost_frac_inner.grid_remove()

        # Row 4: custom CSV profile
        ttk.Label(self._solid_frame, text="Custom profile:").grid(
            row=4, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        _csv_f = ttk.Frame(self._solid_frame)
        _csv_f.grid(row=4, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._profile_path_var = tk.StringVar(value="")
        ttk.Label(_csv_f, textvariable=self._profile_path_var,
                  foreground="gray50", width=22,
                  anchor=tk.W).pack(side=tk.LEFT)
        ttk.Button(_csv_f, text="Browse…",
                   command=self._browse_profile).pack(side=tk.LEFT, padx=(6, 0))

        # Row 5: fill-factor warning
        self._fill_warn_lbl = ttk.Label(self._solid_frame, text="",
                                        foreground="darkorange", wraplength=340)
        self._fill_warn_lbl.grid(row=5, column=0, columnspan=2,
                                 sticky=tk.W, padx=(6, 2), pady=(2, 4))

        # Recompute burn whenever any of the four driving fields change
        for _v in (self._fueled, self._dry, self._thrust_kn, self._isp):
            _v.trace_add("write", self._recompute_burn)
        self._recompute_burn()

    def _recompute_burn(self, *_):
        """Liquid: compute burn = Isp×g₀×prop/thrust.  Solid: update alt-thrust display."""
        if getattr(self, '_solid_motor_var', None) and self._solid_motor_var.get():
            # Solid motor: burn time is user-entered; update computed alternate thrust.
            self._recompute_solid()
            return
        try:
            prop     = float(self._fueled.get()) - float(self._dry.get())
            thrust_n = float(self._thrust_kn.get()) * 1000.0
            isp      = float(self._isp.get())
            if thrust_n <= 0 or isp <= 0 or prop <= 0:
                raise ValueError
            self._burn_var.set(f"{isp * self._G0 * prop / thrust_n:.1f}")
        except (ValueError, ZeroDivisionError):
            self._burn_var.set("—")

    def _on_solid_toggled(self):
        """Show/hide grain frame; switch burn field between computed and user-entered."""
        is_solid = self._solid_motor_var.get()
        if is_solid:
            self._solid_frame.grid()
            # Burn time becomes user-entered
            self._burn_entry.config(state="normal")
            self._burn_hint_lbl.config(text="s  (enter value)")
            self._thrust_lbl.config(text=self._thrust_label_text())
        else:
            self._solid_frame.grid_remove()
            self._burn_entry.config(state="readonly")
            self._burn_hint_lbl.config(text="s  (computed)")
            self._thrust_lbl.config(text="Thrust (kN):")
            self._recompute_burn()

    def _thrust_label_text(self):
        mode = getattr(self, '_thrust_mode_var', None)
        if mode and mode.get() == "average":
            return "Avg thrust (kN):"
        return "Peak thrust (kN):"

    def _on_thrust_mode_changed(self):
        self._thrust_lbl.config(text=self._thrust_label_text())
        self._recompute_solid()

    def _on_grain_changed(self, *_):
        key = self._get_grain_key()
        two_phase = key in ("multi_fin", "dual_composition")
        if two_phase:
            self._boost_frac_lbl.grid()
            self._boost_frac_inner.grid()
        else:
            self._boost_frac_lbl.grid_remove()
            self._boost_frac_inner.grid_remove()
        self._recompute_solid()

    def _get_grain_key(self):
        label = self._grain_var.get()
        for k, v in GRAIN_LABELS.items():
            if v == label:
                return k
        return ""

    def _recompute_solid(self, *_):
        """Compute the alternate thrust and update the fill-factor warning."""
        if not (hasattr(self, '_grain_cb') and self._solid_motor_var.get()):
            return
        key = self._get_grain_key()
        fill = grain_fill_factor(key) if key else 1.0
        try:
            thrust_entered = float(self._thrust_kn.get())
            mode = self._thrust_mode_var.get()
            if mode == "peak":
                alt_kn = thrust_entered * fill
                self._alt_thrust_lbl.config(
                    text=f"{alt_kn:.1f} kN  (average, fill factor {fill:.3f})")
            else:
                if fill > 0:
                    alt_kn = thrust_entered / fill
                    self._alt_thrust_lbl.config(
                        text=f"{alt_kn:.1f} kN  (peak, fill factor {fill:.3f})")
                else:
                    self._alt_thrust_lbl.config(text="—")
        except (ValueError, ZeroDivisionError):
            self._alt_thrust_lbl.config(text="—")
        # Fill-factor warning
        if key and key in _GRAIN_FILL_RANGE:
            lo, hi = _GRAIN_FILL_RANGE[key]
            if not (lo <= fill <= hi):
                self._fill_warn_lbl.config(
                    text=f"\u26a0 Fill factor {fill:.3f} outside typical range "
                         f"{lo}–{hi} for {GRAIN_LABELS.get(key, key)}")
            else:
                self._fill_warn_lbl.config(text="")
        else:
            self._fill_warn_lbl.config(text="")

    def _browse_profile(self):
        """Let user pick a CSV thrust-profile file; show preview plot."""
        import tkinter.filedialog as fd
        import csv as _csv
        path = fd.askopenfilename(
            parent=self,
            title="Select thrust profile CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        # Parse two-column CSV: t_frac, F_frac
        pairs = []
        try:
            with open(path, newline='') as f:
                reader = _csv.reader(f)
                for row in reader:
                    row = [c.strip() for c in row if c.strip()]
                    if len(row) >= 2:
                        try:
                            pairs.append((float(row[0]), float(row[1])))
                        except ValueError:
                            pass   # skip header rows
        except OSError as exc:
            tk.messagebox.showerror("Cannot open file", str(exc), parent=self)
            return
        if len(pairs) < 2:
            tk.messagebox.showerror(
                "Invalid profile",
                "File must contain at least two rows with columns: t_frac, F_frac",
                parent=self)
            return
        import os
        self._profile_path_var.set(os.path.basename(path))
        self._profile_data = pairs   # stored for get()
        self._show_profile_preview(pairs, path)

    def _show_profile_preview(self, pairs, path):
        """Pop up a small Matplotlib preview of the thrust profile."""
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as _plt
        except ImportError:
            tk.messagebox.showinfo(
                "No preview",
                "matplotlib is not installed; profile loaded without preview.",
                parent=self)
            return
        fig, ax = _plt.subplots(figsize=(5, 3), tight_layout=True)
        ts = [p[0] for p in pairs]
        fs = [p[1] for p in pairs]
        ax.step(ts, fs, where='post', color='steelblue', linewidth=1.5)
        ax.set_xlabel("t / burn time")
        ax.set_ylabel("F / F_peak")
        ax.set_title(f"Thrust profile: {path}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid(True, alpha=0.3)
        _plt.show()

    @staticmethod
    def _iter_entries(widget):
        """Yield all ttk.Entry descendants of widget."""
        for child in widget.winfo_children():
            if isinstance(child, ttk.Entry):
                yield child
            else:
                yield from _StageFrame._iter_entries(child)

    def _suggest_nozzle_area(self):
        """Estimate Ae = (g₀ / p₀) × ṁ × Isp_vac × performance_factor"""
        try:
            isp_vac = float(self._isp.get())
            prop    = float(self._fueled.get()) - float(self._dry.get())
            burn    = float(self._burn_var.get())
            if prop <= 0 or burn <= 0 or isp_vac <= 0:
                raise ValueError
        except (ValueError, TypeError):
            tk.messagebox.showerror(
                "Cannot estimate",
                "Please enter valid fueled mass, dry mass, Isp, and thrust first.",
                parent=self)
            return

        mdot = prop / burn

        dlg = tk.Toplevel(self)
        dlg.title("Estimate Nozzle Exit Area")
        dlg.resizable(False, False)
        dlg.grab_set()

        ttk.Label(dlg, text="Isp_vac (s):").grid(
            row=0, column=0, sticky=tk.W, padx=(10, 4), pady=(10, 2))
        isp_var = tk.StringVar(value=str(isp_vac))
        ttk.Entry(dlg, textvariable=isp_var, width=10).grid(
            row=0, column=1, sticky=tk.W, padx=(0, 10), pady=(10, 2))

        ttk.Label(dlg, text="Performance factor:").grid(
            row=1, column=0, sticky=tk.W, padx=(10, 4), pady=2)
        pf_var = tk.StringVar(value="0.10")
        ttk.Entry(dlg, textvariable=pf_var, width=10).grid(
            row=1, column=1, sticky=tk.W, padx=(0, 10), pady=2)

        result_var = tk.StringVar(value="")
        ttk.Label(dlg, textvariable=result_var, foreground="navy").grid(
            row=2, column=0, columnspan=2, padx=10, pady=(6, 4))

        def _compute(*_):
            try:
                isp  = float(isp_var.get())
                pf   = float(pf_var.get())
                if isp <= 0 or pf <= 0:
                    raise ValueError
                ae = (self._G0 / 101325.0) * mdot * isp * pf
                result_var.set(f"Ae ≈ {ae:.4f} m²")
                return ae
            except (ValueError, TypeError):
                result_var.set("Enter valid Isp and performance factor.")
                return None

        isp_var.trace_add("write", lambda *_: _compute())
        pf_var .trace_add("write", lambda *_: _compute())

        btn_row = ttk.Frame(dlg)
        btn_row.grid(row=3, column=0, columnspan=2, pady=(4, 10))

        def _accept():
            ae = _compute()
            if ae is not None:
                self._nozzle_area.set(f"{ae:.4f}")
                dlg.destroy()

        ttk.Button(btn_row, text="Accept", command=_accept).pack(
            side=tk.LEFT, padx=6)
        ttk.Button(btn_row, text="Cancel",
                   command=dlg.destroy).pack(side=tk.LEFT, padx=6)

        # Centre over parent
        dlg.update_idletasks()
        px = self.winfo_rootx() + (self.winfo_width()  - dlg.winfo_reqwidth())  // 2
        py = self.winfo_rooty() + (self.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{px}+{py}")

    def _suggest_thrust(self):
        """Estimate thrust from observed rocket acceleration during boost."""
        import math
        G0 = 9.80665

        dlg = tk.Toplevel(self)
        dlg.title("Estimate Thrust")
        dlg.resizable(False, False)
        dlg.grab_set()

        # ── Input fields ──────────────────────────────────────────────
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.X)
        frm.columnconfigure(1, weight=1)

        def _lbl(row, text):
            ttk.Label(frm, text=text).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3)

        _lbl(0, "Fueled mass (kg):")
        mass_var = tk.StringVar(value=self._fueled.get())
        ttk.Entry(frm, textvariable=mass_var, width=10).grid(
            row=0, column=1, sticky=tk.W)

        _lbl(1, "Vertical acceleration (m/s², upward +):")
        av_var = tk.StringVar(value="")
        ttk.Entry(frm, textvariable=av_var, width=10).grid(
            row=1, column=1, sticky=tk.W)

        _lbl(2, "Horizontal acceleration (m/s²):")
        ah_inner = ttk.Frame(frm)
        ah_inner.grid(row=2, column=1, sticky=tk.W)
        ah_var = tk.StringVar(value="0")
        ttk.Entry(ah_inner, textvariable=ah_var, width=10).pack(side=tk.LEFT)
        ttk.Label(ah_inner, text="  (0 for vertical flight)",
                  foreground="gray50").pack(side=tk.LEFT)

        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)

        # ── Result display ────────────────────────────────────────────
        res_frm = ttk.Frame(dlg, padding=(12, 8))
        res_frm.pack(fill=tk.X)
        res_frm.columnconfigure(1, weight=1)
        ttk.Label(res_frm, text="Estimated thrust:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=2)
        thrust_lbl = ttk.Label(res_frm, text="—",
                               font=("", 11, "bold"), foreground="navy")
        thrust_lbl.grid(row=0, column=1, sticky=tk.W)
        note_lbl = ttk.Label(res_frm, text="", foreground="gray50", wraplength=360)
        note_lbl.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 0))

        _thrust_result = [None]

        def _compute(*_):
            try:
                mass = float(mass_var.get())
                if mass <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                thrust_lbl.config(text="—")
                note_lbl.config(text="invalid mass")
                _thrust_result[0] = None
                return
            try:
                av  = float(av_var.get())
                ah  = float(ah_var.get())
                f_n = mass * math.sqrt(ah**2 + (av + G0)**2)
                if f_n <= 0:
                    raise ValueError
            except (ValueError, TypeError):
                thrust_lbl.config(text="—")
                note_lbl.config(text="")
                _thrust_result[0] = None
                return
            note = (f"T = m·√(a_h²+(a_v+g)²)  =  {mass:.0f}·"
                    f"√({ah:.2f}²+{av+G0:.3f}²)  =  {f_n/1000:.2f} kN")
            thrust_lbl.config(text=f"{f_n/1000:,.1f} kN")
            note_lbl.config(text=note)
            _thrust_result[0] = f_n / 1000.0

        for _v in (mass_var, av_var, ah_var):
            _v.trace_add("write", _compute)

        # ── Buttons ───────────────────────────────────────────────────
        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        btn_frm = ttk.Frame(dlg, padding=(12, 8))
        btn_frm.pack(fill=tk.X)

        def _use():
            if _thrust_result[0] is not None:
                self._thrust_kn.set(f"{_thrust_result[0]:.1f}")
            dlg.destroy()

        ttk.Button(btn_frm, text="Use this value", command=_use).pack(side=tk.LEFT)
        ttk.Button(btn_frm, text="Cancel",
                   command=dlg.destroy).pack(side=tk.LEFT, padx=6)

    def set_readonly(self, readonly: bool):
        """Set all editable entry fields to readonly (for Forden reference missiles)."""
        state = "readonly" if readonly else "normal"
        cb_state = "disabled" if readonly else "normal"
        self._solid_motor_check.config(state=cb_state)
        self._grain_cb.config(state="disabled" if readonly else "readonly")
        for entry in self._iter_entries(self):
            # Burn entry: readonly unless solid motor is enabled (managed by _on_solid_toggled)
            if entry is self._burn_entry:
                if not readonly and self._solid_motor_var.get():
                    entry.config(state="normal")
                else:
                    entry.config(state="readonly")
            else:
                entry.config(state=state)

    def get(self):
        burn_str = self._burn_var.get()
        if burn_str == "—":
            raise ValueError("Burn time could not be computed — check thrust, Isp, and masses.")
        _LABELS = {
            "fueled": "Fueled Mass", "dry": "Dry Mass", "dia": "Diameter",
            "length": "Length", "thrust_kn": "Thrust", "isp": "Isp",
            "nozzle_area": "Nozzle Exit Area",
        }
        result = {}
        for k, v in [
            ("fueled",      self._fueled),      ("dry",         self._dry),
            ("dia",         self._dia),         ("length",      self._length),
            ("thrust_kn",   self._thrust_kn),   ("isp",         self._isp),
            ("nozzle_area", self._nozzle_area),
        ]:
            try:
                result[k] = float(v.get())
            except ValueError:
                raise ValueError(
                    f"{_LABELS.get(k, k)}: expected a number, got {v.get()!r:.40s}"
                )
        try:
            result["burn"] = float(burn_str)
        except ValueError:
            raise ValueError(f"Burn time: expected a number, got {burn_str!r:.40s}")
        result["coast"] = 0.0   # coast is now set in the advanced pitch panel
        result["solid_motor"] = bool(self._solid_motor_var.get())

        # Solid-motor grain fields
        result["grain_type"]    = ""
        result["thrust_peak_N"] = 0.0
        result["thrust_profile"] = []
        if result["solid_motor"]:
            grain_key = self._get_grain_key()
            result["grain_type"] = grain_key
            fill = grain_fill_factor(grain_key) if grain_key else 1.0
            try:
                thrust_entered_n = float(self._thrust_kn.get()) * 1000.0
                if self._thrust_mode_var.get() == "peak":
                    result["thrust_peak_N"] = thrust_entered_n
                    # thrust_kn already holds peak; override to avg for prop-mass consistency
                    result["thrust_kn"] = thrust_entered_n * fill / 1000.0
                else:
                    result["thrust_peak_N"] = (thrust_entered_n / fill
                                               if fill > 0 else thrust_entered_n)
                    # thrust_kn holds avg — leave as is
            except ValueError:
                pass
            result["thrust_profile"] = getattr(self, '_profile_data', [])
        return result

    def populate(self, d):
        # Back-calculate thrust_kn from stored burn/isp/prop so the round-trip
        # is exact: T = Isp × g₀ × prop / burn
        prop = d["fueled"] - d["dry"]
        burn = d["burn"]
        thrust_kn = (d["isp"] * self._G0 * prop / burn / 1000.0
                     if burn > 0 and prop > 0 else 0.0)

        self._fueled      .set(str(d["fueled"]))
        self._dry         .set(str(d["dry"]))
        self._dia         .set(str(d["dia"]))
        self._length      .set(str(d["length"]))
        self._thrust_kn   .set(f"{thrust_kn:.1f}")
        self._isp         .set(str(d["isp"]))
        self._nozzle_area .set(str(d.get("nozzle_area", 0)))
        # _burn_var is updated automatically by the trace (liquid) or set directly (solid)
        self._solid_motor_var.set(bool(d.get("solid_motor", False)))

        # Grain profile fields
        grain_key = d.get("grain_type", "")
        if grain_key and grain_key in GRAIN_LABELS:
            self._grain_var.set(GRAIN_LABELS[grain_key])
        else:
            self._grain_var.set(GRAIN_LABELS.get("star", ""))

        thrust_peak_N = float(d.get("thrust_peak_N", 0.0))
        if thrust_peak_N > 0.0:
            self._thrust_mode_var.set("peak")
            self._thrust_kn.set(f"{thrust_peak_N / 1000.0:.1f}")
        else:
            self._thrust_mode_var.set("average")
            # thrust_kn is already set above from Isp/prop/burn (average thrust)

        profile = d.get("thrust_profile", [])
        self._profile_data = [tuple(p) for p in profile] if profile else []
        if self._profile_data:
            self._profile_path_var.set(f"<{len(self._profile_data)} pts>")
        else:
            self._profile_path_var.set("")

        # Trigger UI state update
        if self._solid_motor_var.get():
            self._on_solid_toggled()
            self._on_grain_changed()
            self._on_thrust_mode_changed()
            # For solid, burn time was saved as-is — restore directly
            self._burn_var.set(f"{d['burn']:.1f}")


# ---------------------------------------------------------------------------
# New / Edit missile dialog
# ---------------------------------------------------------------------------

class MissileDialog(tk.Toplevel):
    """Modal dialog for creating or editing a custom missile."""

    def __init__(self, parent, on_save, existing_name=None):
        super().__init__(parent)
        self._on_save = on_save
        self._existing_name = existing_name
        self._readonly_mode = (existing_name is not None
                               and existing_name.endswith(" (Forden)"))
        if self._readonly_mode:
            self.title("View Missile — Forden Reference")
        elif existing_name:
            self.title("Edit Missile")
        else:
            self.title("New Missile")
        self.resizable(False, True)
        self.grab_set()               # modal
        self._build(existing_name)
        # Centre over parent; cap height to 90 % of screen so dialog is scrollable
        self.update_idletasks()
        max_h = int(parent.winfo_screenheight() * 0.90)
        nat_h = self.winfo_reqheight()
        dlg_h = min(nat_h, max_h)
        dlg_w = self.winfo_reqwidth()
        px = parent.winfo_rootx() + (parent.winfo_width()  - dlg_w) // 2
        py = parent.winfo_rooty() + (parent.winfo_height() - dlg_h) // 2
        self.geometry(f"{dlg_w}x{dlg_h}+{px}+{py}")

    # ------------------------------------------------------------------
    def _build(self, existing_name):
        pad = dict(padx=8, pady=4)

        # Name row
        nf = ttk.Frame(self)
        nf.pack(fill=tk.X, **pad)
        ttk.Label(nf, text="Missile name:").pack(side=tk.LEFT)
        self._name_var = tk.StringVar(value=existing_name or "My Missile")
        self._name_entry = ttk.Entry(nf, textvariable=self._name_var, width=24)
        self._name_entry.pack(side=tk.LEFT, padx=(6, 16))
        ttk.Label(nf, text="Stages:").pack(side=tk.LEFT)
        self._n_stages_var = tk.StringVar(value="1")
        self._stages_cb = ttk.Combobox(nf, textvariable=self._n_stages_var,
                                       values=["1", "2", "3", "4"],
                                       state="readonly", width=3)
        self._stages_cb.pack(side=tk.LEFT, padx=(4, 0))
        self._stages_cb.bind("<<ComboboxSelected>>",
                             lambda _: self._update_stage_frames())
        ttk.Label(nf, text="  Boosters:").pack(side=tk.LEFT)
        self._n_boosters_var = tk.StringVar(value="0")
        self._n_boosters_spin = ttk.Spinbox(
            nf, textvariable=self._n_boosters_var, from_=0, to=9,
            width=2, command=self._update_booster_frame)
        self._n_boosters_spin.pack(side=tk.LEFT, padx=(4, 0))
        self._n_boosters_var.trace_add("write",
                                       lambda *_: self._update_booster_frame())

        # Scrollable body: canvas + scrollbar sandwiched between the name row
        # and the Save/Cancel buttons so buttons are always visible.
        scroll_outer = ttk.Frame(self)
        scroll_outer.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(
            scroll_outer, borderwidth=0, highlightthickness=0, height=500)
        vsb = ttk.Scrollbar(scroll_outer, orient="vertical",
                            command=self._scroll_canvas.yview)
        self._scroll_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Inner frame — all middle content lives here
        body = ttk.Frame(self._scroll_canvas)
        body_id = self._scroll_canvas.create_window(
            (0, 0), window=body, anchor="nw")

        def _on_body_configure(_event):
            self._scroll_canvas.configure(
                scrollregion=self._scroll_canvas.bbox("all"))

        def _on_canvas_configure(event):
            self._scroll_canvas.itemconfig(body_id, width=event.width)

        body.bind("<Configure>", _on_body_configure)
        self._scroll_canvas.bind("<Configure>", _on_canvas_configure)

        # Mousewheel scrolling (Mac/Windows uses <MouseWheel>; Linux Button-4/5)
        def _on_mousewheel(event):
            self._scroll_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units")

        def _on_mousewheel_linux_up(_event):
            self._scroll_canvas.yview_scroll(-1, "units")

        def _on_mousewheel_linux_down(_event):
            self._scroll_canvas.yview_scroll(1, "units")

        self._scroll_canvas.bind("<MouseWheel>", _on_mousewheel)
        self._scroll_canvas.bind("<Button-4>",   _on_mousewheel_linux_up)
        self._scroll_canvas.bind("<Button-5>",   _on_mousewheel_linux_down)
        body.bind("<MouseWheel>", _on_mousewheel)
        body.bind("<Button-4>",   _on_mousewheel_linux_up)
        body.bind("<Button-5>",   _on_mousewheel_linux_down)

        # ── Front End panel ──────────────────────────────────────────────────
        pl = ttk.LabelFrame(body, text="Front End")
        pl.pack(fill=tk.X, padx=8, pady=4)
        pl.columnconfigure(1, weight=1)

        _ns_labels = list(NOSE_SHAPE_LABELS.values())

        def _fe_entry(parent, label, row, default, unit, pady=2):
            """Helper: label in col 0, entry+unit label in col 1."""
            ttk.Label(parent, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(6, 2), pady=pady)
            _inner = ttk.Frame(parent)
            _inner.grid(row=row, column=1, sticky=tk.W, padx=(0, 6), pady=pady)
            _var = tk.StringVar(value=default)
            _ent = ttk.Entry(_inner, textvariable=_var, width=10)
            _ent.pack(side=tk.LEFT)
            ttk.Label(_inner, text=unit).pack(side=tk.LEFT, padx=(2, 0))
            return _var, _ent

        # ── Row 0: Throw weight ──────────────────────────────────────────────
        ttk.Label(pl, text="Throw weight (kg):").grid(
            row=0, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._throw_weight_var = tk.StringVar(value="1000")
        _tw_inner = ttk.Frame(pl)
        _tw_inner.grid(row=0, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._throw_weight_entry = ttk.Entry(
            _tw_inner, textvariable=self._throw_weight_var, width=10)
        self._throw_weight_entry.pack(side=tk.LEFT)
        ttk.Label(_tw_inner, text="kg").pack(side=tk.LEFT, padx=(2, 0))

        # ── Rows 1-3: Payload nose shape/size (hidden when RV separates) ────────
        self._payload_shape_frame = ttk.Frame(pl)
        self._payload_shape_frame.grid(row=1, column=0, columnspan=2,
                                       sticky=tk.EW, pady=0)
        self._payload_shape_frame.columnconfigure(1, weight=1)

        ttk.Label(self._payload_shape_frame, text="Payload shape:").grid(
            row=0, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._payload_shape_var = tk.StringVar(value=NOSE_SHAPE_LABELS["cone"])
        self._payload_shape_cb = ttk.Combobox(
            self._payload_shape_frame, textvariable=self._payload_shape_var,
            values=_ns_labels, state="readonly", width=18)
        self._payload_shape_cb.grid(row=0, column=1, sticky=tk.W, padx=(0, 6), pady=2)

        self._payload_diameter_var, self._payload_diameter_entry = _fe_entry(
            self._payload_shape_frame, "Payload diameter (m):", 1, "0", "m")
        self._payload_length_var, self._payload_length_entry = _fe_entry(
            self._payload_shape_frame, "Payload length (m):", 2, "0", "m", pady=(2, 4))

        # Keep legacy aliases so _calc_rv_beta pre-fill still resolves
        self._nose_shape_var    = self._payload_shape_var
        self._nose_length_var   = self._payload_length_var
        self._nose_shape_cb     = self._payload_shape_cb
        self._nose_length_entry = self._payload_length_entry

        # ── Row 2: RV separates toggle (row numbering continues in pl) ────────
        self._rv_separates_var = tk.BooleanVar(value=False)
        self._rv_separates_check = ttk.Checkbutton(
            pl, text="RV separates",
            variable=self._rv_separates_var,
            command=self._update_rv_separates_state)
        self._rv_separates_check.grid(
            row=2, column=0, columnspan=2, sticky=tk.W, padx=(6, 2), pady=(4, 0))

        # ── Row 3: RV section (hidden until checkbox ticked) ─────────────────
        self._rv_section = ttk.Frame(pl)
        self._rv_section.grid(row=3, column=0, columnspan=2,
                              sticky=tk.EW, padx=(16, 0))
        self._rv_section.columnconfigure(1, weight=1)
        self._rv_section.grid_remove()

        # Per-RV mass
        ttk.Label(self._rv_section, text="Per-RV mass (kg):").grid(
            row=0, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._rv_mass_var = tk.StringVar(value="1000")
        _rvm_inner = ttk.Frame(self._rv_section)
        _rvm_inner.grid(row=0, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._rv_mass_entry = ttk.Entry(_rvm_inner, textvariable=self._rv_mass_var, width=10)
        self._rv_mass_entry.pack(side=tk.LEFT)
        ttk.Label(_rvm_inner, text="kg").pack(side=tk.LEFT, padx=(2, 0))

        # No. of RVs
        ttk.Label(self._rv_section, text="No. of RVs:").grid(
            row=1, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._num_rvs_var = tk.StringVar(value="1")
        _rvn_inner = ttk.Frame(self._rv_section)
        _rvn_inner.grid(row=1, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._num_rvs_spinbox = ttk.Spinbox(
            _rvn_inner, textvariable=self._num_rvs_var, from_=1, to=24, width=4)
        self._num_rvs_spinbox.pack(side=tk.LEFT)

        # RV shape
        ttk.Label(self._rv_section, text="RV shape:").grid(
            row=2, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._rv_shape_var = tk.StringVar(value=NOSE_SHAPE_LABELS["cone"])
        self._rv_shape_cb = ttk.Combobox(
            self._rv_section, textvariable=self._rv_shape_var,
            values=_ns_labels, state="readonly", width=18)
        self._rv_shape_cb.grid(row=2, column=1, sticky=tk.W, padx=(0, 6), pady=2)

        # RV diameter / length
        self._rv_diameter_var, self._rv_diameter_entry = _fe_entry(
            self._rv_section, "RV diameter (m):", 3, "0", "m")
        self._rv_length_var, self._rv_length_entry = _fe_entry(
            self._rv_section, "RV length (m):", 4, "0", "m")

        # Has PBV toggle
        self._has_pbv_var = tk.BooleanVar(value=False)
        self._has_pbv_check = ttk.Checkbutton(
            self._rv_section, text="Has PBV (post-boost vehicle)",
            variable=self._has_pbv_var,
            command=self._update_pbv_state)
        self._has_pbv_check.grid(
            row=5, column=0, columnspan=2, sticky=tk.W, padx=(6, 2), pady=(4, 0))

        # PBV sub-section
        self._pbv_section = ttk.Frame(self._rv_section)
        self._pbv_section.grid(row=6, column=0, columnspan=2,
                               sticky=tk.EW, padx=(16, 0))
        self._pbv_section.columnconfigure(1, weight=1)
        self._pbv_section.grid_remove()

        self._pbv_mass_var, self._pbv_mass_entry = _fe_entry(
            self._pbv_section, "PBV mass (kg):", 0, "0", "kg")
        self._pbv_diameter_var, self._pbv_diameter_entry = _fe_entry(
            self._pbv_section, "PBV diameter (m):", 1, "0", "m")
        self._pbv_length_var, self._pbv_length_entry = _fe_entry(
            self._pbv_section, "PBV length (m):", 2, "0", "m")

        # Legacy alias: bus_var → pbv_mass_var
        self._bus_var = self._pbv_mass_var

        # RV β + Estimate button
        ttk.Label(self._rv_section, text="RV β (kg/m²):").grid(
            row=7, column=0, sticky=tk.W, padx=(6, 2), pady=(4, 2))
        self._rv_beta_var = tk.StringVar(value="0")
        _beta_inner = ttk.Frame(self._rv_section)
        _beta_inner.grid(row=7, column=1, sticky=tk.W, padx=(0, 6), pady=(4, 2))
        self._rv_beta_entry = ttk.Entry(_beta_inner, textvariable=self._rv_beta_var, width=10)
        self._rv_beta_entry.pack(side=tk.LEFT)
        ttk.Label(_beta_inner, text="kg/m²").pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(_beta_inner, text="Estimate…",
                   command=self._calc_rv_beta).pack(side=tk.LEFT)

        # ── Row 6: Has Shroud toggle ─────────────────────────────────────────
        self._shroud_var = tk.BooleanVar(value=False)
        self._shroud_check = ttk.Checkbutton(
            pl, text="Has Shroud",
            variable=self._shroud_var,
            command=self._update_shroud_state)
        self._shroud_check.grid(
            row=6, column=0, columnspan=2, sticky=tk.W, padx=(6, 2), pady=(4, 0))

        # ── Row 7: Shroud section (hidden until checkbox ticked) ─────────────
        self._shroud_section = ttk.Frame(pl)
        self._shroud_section.grid(row=7, column=0, columnspan=2,
                                  sticky=tk.EW, padx=(16, 0), pady=(0, 4))
        self._shroud_section.columnconfigure(1, weight=1)
        self._shroud_section.grid_remove()

        self._shroud_mass_var, self._shroud_mass_entry = _fe_entry(
            self._shroud_section, "Mass (kg):", 0, "0", "kg")
        ttk.Label(self._shroud_section, text="Shape:").grid(
            row=1, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._shroud_nose_shape_var = tk.StringVar(value=NOSE_SHAPE_LABELS["cone"])
        self._shroud_nose_shape_cb = ttk.Combobox(
            self._shroud_section, textvariable=self._shroud_nose_shape_var,
            values=_ns_labels, state="readonly", width=18)
        self._shroud_nose_shape_cb.grid(row=1, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._shroud_diameter_var, self._shroud_diameter_entry = _fe_entry(
            self._shroud_section, "Diameter (m):", 2, "0", "m")
        self._shroud_length_var, self._shroud_length_entry = _fe_entry(
            self._shroud_section, "Total shroud length (m):", 3, "0", "m")
        self._shroud_nose_length_var, self._shroud_nose_length_entry = _fe_entry(
            self._shroud_section, "Nose segment length (m):", 4, "0", "m")
        self._shroud_alt_var, self._shroud_alt_entry = _fe_entry(
            self._shroud_section, "Jettison alt (km):", 5, "80", "km", pady=(2, 4))

        # Live throw-weight update when RV fields change
        for _v in (self._rv_mass_var, self._num_rvs_var, self._pbv_mass_var):
            _v.trace_add("write", self._update_throw_weight)

        # ── Guidance mode ────────────────────────────────────────────────
        gf = ttk.LabelFrame(body, text="Guidance Mode")
        gf.pack(fill=tk.X, padx=8, pady=4)
        self._guidance_var = tk.StringVar(value="gravity_turn")
        ttk.Radiobutton(gf, text="Gravity Turn (SRBM / MRBM / IRBM / ICBM)",
                        variable=self._guidance_var, value="gravity_turn").pack(
            anchor=tk.W, padx=8, pady=(4, 2))
        ttk.Radiobutton(gf, text="Orbital Insertion",
                        variable=self._guidance_var, value="orbital_insertion").pack(
            anchor=tk.W, padx=8, pady=(0, 4))

        # ── Strap-on Boosters panel ─────────────────────────────────────────
        self._booster_frame = ttk.LabelFrame(body, text="Strap-on Boosters")
        self._booster_frame.columnconfigure(1, weight=1)
        # (packed/unpacked dynamically by _update_booster_frame)

        def _be_entry(row, label, default, unit, pady=2):
            ttk.Label(self._booster_frame, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(6, 2), pady=pady)
            _inner = ttk.Frame(self._booster_frame)
            _inner.grid(row=row, column=1, sticky=tk.W, padx=(0, 6), pady=pady)
            _var = tk.StringVar(value=default)
            ttk.Entry(_inner, textvariable=_var, width=10).pack(side=tk.LEFT)
            ttk.Label(_inner, text=unit).pack(side=tk.LEFT, padx=(2, 0))
            return _var

        self._b_thrust_var      = _be_entry(0, "Thrust per booster (kN):", "500",  "kN")
        self._b_burn_var        = _be_entry(1, "Burn time (s):",            "60",   "s")
        self._b_core_delay_var  = _be_entry(2, "Core ignition delay (s):", "0",    "s")
        self._b_inert_var       = _be_entry(3, "Inert mass per booster (kg):", "2000", "kg")
        self._b_prop_var        = _be_entry(4, "Propellant per booster (kg):", "10000","kg")
        self._b_isp_var         = _be_entry(5, "Isp (vacuum, s):",          "270",  "s")
        self._b_nozzle_var      = _be_entry(6, "Nozzle exit area (m²):",    "0",    "m²")
        self._b_diam_var        = _be_entry(7, "Diameter (m):",              "1.2",  "m")
        self._b_length_var      = _be_entry(8, "Length (m):",               "0",    "m  (0 = 2×dia)")
        self._b_cd_var          = _be_entry(9, "Cd (drag coeff):",          "0.20", "",
                                            pady=(2, 6))
        ttk.Label(self._booster_frame, text="Cd guide: 0.10 ogive · 0.20 cone · 0.40 hemi · 1.0 flat",
                  foreground="gray50").grid(
            row=10, column=0, columnspan=2, sticky=tk.W, padx=(6, 6), pady=(0, 4))

        # Stage frames (1 always visible; 2-4 toggled).
        # A dedicated container ensures dynamically-packed stages always appear
        # between the payload row and the buttons (not after the buttons).
        self._stages_container = ttk.Frame(body)
        self._stages_container.pack(fill=tk.X)
        self._stage_frames = [_StageFrame(self._stages_container, f"Stage {i+1}",
                                           stage_num=i+1)
                               for i in range(4)]
        self._stage_frames[0].pack(fill=tk.X, **pad)  # Stage 1 always shown

        # Buttons — outside the scroll area so always visible
        bf = ttk.Frame(self)
        bf.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Button(bf, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=(4, 0))
        self._save_btn = ttk.Button(bf, text="Save Missile", command=self._save)
        self._save_btn.pack(side=tk.RIGHT)
        self._save_as_btn = ttk.Button(bf, text="Save as New Missile",
                                       command=self._save_as_new)
        self._save_as_btn.pack(side=tk.RIGHT, padx=(0, 8))

        # Pre-fill if editing an existing missile
        if existing_name:
            self._prefill(existing_name)
            if self._readonly_mode:
                self._apply_readonly()

    # ------------------------------------------------------------------
    def _apply_readonly(self):
        """Lock all input fields for Forden reference missiles."""
        for sf in self._stage_frames:
            sf.set_readonly(True)
        self._name_entry.config(state="readonly")
        self._stages_cb.config(state="disabled")
        # Payload / throw weight
        self._throw_weight_entry.config(state="disabled")
        self._payload_shape_cb.config(state="disabled")
        self._payload_diameter_entry.config(state="disabled")
        self._payload_length_entry.config(state="disabled")
        # RV section
        self._rv_separates_check.config(state="disabled")
        self._rv_mass_entry.config(state="disabled")
        self._num_rvs_spinbox.config(state="disabled")
        self._rv_shape_cb.config(state="disabled")
        self._rv_diameter_entry.config(state="disabled")
        self._rv_length_entry.config(state="disabled")
        self._has_pbv_check.config(state="disabled")
        self._pbv_mass_entry.config(state="disabled")
        self._pbv_diameter_entry.config(state="disabled")
        self._pbv_length_entry.config(state="disabled")
        self._rv_beta_entry.config(state="disabled")
        # Shroud section
        self._shroud_check.config(state="disabled")
        self._shroud_mass_entry.config(state="disabled")
        self._shroud_nose_shape_cb.config(state="disabled")
        self._shroud_diameter_entry.config(state="disabled")
        self._shroud_length_entry.config(state="disabled")
        self._shroud_nose_length_entry.config(state="disabled")
        self._shroud_alt_entry.config(state="disabled")
        # Booster section
        self._n_boosters_spin.config(state="disabled")
        for _bv in (self._b_thrust_var, self._b_burn_var, self._b_core_delay_var,
                    self._b_inert_var, self._b_prop_var, self._b_isp_var,
                    self._b_nozzle_var, self._b_diam_var, self._b_length_var,
                    self._b_cd_var):
            for _w in self._booster_frame.winfo_children():
                if isinstance(_w, ttk.Frame):
                    for _c in _w.winfo_children():
                        if isinstance(_c, ttk.Entry):
                            try:
                                _c.config(state="disabled")
                            except tk.TclError:
                                pass
        self._save_btn.pack_forget()
        self._save_as_btn.pack_forget()

    # ------------------------------------------------------------------
    def _update_throw_weight(self, *_):
        """Recompute throw weight from RV fields when RV separates is active."""
        if not self._rv_separates_var.get():
            return
        try:
            n   = max(1, int(self._num_rvs_var.get()))
            rv  = float(self._rv_mass_var.get())
            bus = float(self._pbv_mass_var.get()) if self._has_pbv_var.get() else 0.0
            total = n * rv + bus
            self._throw_weight_entry.config(state="normal")
            self._throw_weight_var.set(f"{total:.0f}")
            self._throw_weight_entry.config(state="readonly")
        except (ValueError, tk.TclError):
            pass

    def _update_rv_separates_state(self):
        """Show/hide the RV sub-section; toggle payload-shape rows and throw weight."""
        if self._rv_separates_var.get():
            self._rv_section.grid()
            self._payload_shape_frame.grid_remove()   # RV geometry takes over
            self._throw_weight_entry.config(state="readonly")
            self._update_throw_weight()
        else:
            self._rv_section.grid_remove()
            self._payload_shape_frame.grid()
            self._throw_weight_entry.config(state="normal")

    def _update_pbv_state(self):
        """Show/hide PBV sub-section; refresh throw weight."""
        if self._has_pbv_var.get():
            self._pbv_section.grid()
        else:
            self._pbv_section.grid_remove()
        self._update_throw_weight()

    def _update_shroud_state(self):
        """Show/hide the shroud sub-section."""
        if self._shroud_var.get():
            self._shroud_section.grid()
        else:
            self._shroud_section.grid_remove()

    # ------------------------------------------------------------------
    def _update_booster_frame(self, *_):
        """Show or hide the booster parameter panel based on booster count."""
        try:
            n = int(self._n_boosters_var.get())
        except (ValueError, tk.TclError):
            n = 0
        if n > 0:
            self._booster_frame.pack(fill=tk.X, padx=8, pady=4,
                                     before=self._stages_container)
        else:
            self._booster_frame.pack_forget()

    # ------------------------------------------------------------------
    def _update_stage_frames(self):
        """Show the right number of stage frames and coast-time rows."""
        n = int(self._n_stages_var.get())
        pad = dict(padx=8, pady=4)
        for i, sf in enumerate(self._stage_frames):
            if i < n:
                sf.pack(fill=tk.X, **pad)
            else:
                sf.pack_forget()

    # ------------------------------------------------------------------
    def _prefill(self, name):
        """Populate all fields from an existing missile (custom or packaged)."""
        p = MISSILE_DB[name]()

        payload      = p.payload_kg
        shroud_mass  = p.shroud_mass_kg

        # Walk the linked list to collect per-stage data.
        # mass_initial is cumulative (includes all upper stages); we recover
        # per-stage fueled mass by differencing adjacent mass_initial values
        # and stripping payload/shroud from the appropriate stages.
        # dry is always fueled - mass_propellant: mass_propellant is per-stage
        # and reliable even in missiles loaded from older JSON files, so this
        # avoids any dependency on the (potentially corrupt) mass_final field.
        stage_data = []
        node = p
        stage_idx = 0
        while node is not None:
            nxt      = node.stage2
            is_first = (stage_idx == 0)
            is_last  = (nxt is None)
            if is_last and is_first:
                # Single-stage missile
                fueled = node.mass_initial - payload - shroud_mass
            elif is_last:
                # Last of multiple stages: no shroud here (lives on stage 1)
                fueled = node.mass_initial - payload
            elif is_first:
                # First of multiple stages: subtract shroud and upper stack
                fueled = node.mass_initial - shroud_mass - nxt.mass_initial
            else:
                # Middle stage
                fueled = node.mass_initial - nxt.mass_initial
            # Per-stage dry mass is always: fueled - own propellant.
            # Do NOT use node.mass_final here — it may be a cumulative burnout
            # mass (includes upper stages) if the model was loaded from an
            # older serialised file.
            dry = fueled - node.mass_propellant
            stage_data.append({
                "fueled":        fueled,                   "dry":          dry,
                "dia":           node.diameter_m,          "length":       node.length_m,
                "burn":          node.burn_time_s,         "isp":          node.isp_s,
                "nozzle_area":   node.nozzle_exit_area_m2, "coast":        node.coast_time_s,
                "solid_motor":   getattr(node, 'solid_motor', False),
                "grain_type":    getattr(node, 'grain_type', ''),
                "thrust_peak_N": getattr(node, 'thrust_peak_N', 0.0),
                "thrust_profile": list(getattr(node, 'thrust_profile', [])),
            })
            node = nxt
            stage_idx += 1

        n = len(stage_data)
        self._n_stages_var.set(str(n))
        self._update_stage_frames()
        for i, sd in enumerate(stage_data):
            self._stage_frames[i].populate(sd)

        # Throw weight and RV decomposition
        self._rv_separates_var.set(p.rv_separates)
        self._throw_weight_var.set(f"{payload:.0f}")
        if p.rv_separates and p.rv_mass_kg > 0:
            self._rv_mass_var.set(f"{p.rv_mass_kg:.0f}")
            self._num_rvs_var.set(str(p.num_rvs))
            has_pbv = p.bus_mass_kg > 0
            self._has_pbv_var.set(has_pbv)
            self._pbv_mass_var.set(f"{p.bus_mass_kg:.0f}")
            self._pbv_diameter_var.set(f"{getattr(p, 'pbv_diameter_m', 0.0):.2f}")
            self._pbv_length_var.set(f"{getattr(p, 'pbv_length_m', 0.0):.2f}")
        else:
            self._rv_mass_var.set(f"{payload:.0f}")
            self._num_rvs_var.set("1")
            self._has_pbv_var.set(False)
            self._pbv_mass_var.set("0")
            self._pbv_diameter_var.set("0")
            self._pbv_length_var.set("0")
        self._rv_beta_var.set(f"{p.rv_beta_kg_m2:.0f}")
        self._rv_shape_var.set(
            NOSE_SHAPE_LABELS.get(getattr(p, 'rv_shape', ''),
                                  NOSE_SHAPE_LABELS["cone"]))
        self._rv_diameter_var.set(f"{getattr(p, 'rv_diameter_m', 0.0):.2f}")
        self._rv_length_var.set(f"{getattr(p, 'rv_length_m', 0.0):.2f}")

        # Payload shape / diameter / length
        self._payload_shape_var.set(
            NOSE_SHAPE_LABELS.get(p.nose_shape, NOSE_SHAPE_LABELS["cone"]))
        self._payload_diameter_var.set(
            f"{getattr(p, 'payload_diameter_m', 0.0):.2f}")
        self._payload_length_var.set(f"{p.nose_length_m:.2f}")

        # Shroud
        has_shroud = shroud_mass > 0
        self._shroud_var.set(has_shroud)
        self._shroud_mass_var.set(f"{shroud_mass:.0f}")
        self._shroud_alt_var.set(f"{p.shroud_jettison_alt_km:.0f}")
        self._shroud_length_var.set(f"{p.shroud_length_m:.1f}")
        self._shroud_diameter_var.set(f"{p.shroud_diameter_m:.2f}")
        self._shroud_nose_shape_var.set(
            NOSE_SHAPE_LABELS.get(p.shroud_nose_shape, NOSE_SHAPE_LABELS["cone"]))
        self._shroud_nose_length_var.set(f"{p.shroud_nose_length_m:.2f}")

        # Strap-on boosters
        nb = getattr(p, 'n_boosters', 0)
        self._n_boosters_var.set(str(nb))
        if nb > 0:
            G0 = 9.80665
            b_prop = getattr(p, 'booster_prop_kg', 0.0)
            b_burn = getattr(p, 'booster_burn_time_s', 0.0)
            b_isp  = getattr(p, 'booster_isp_s', 0.0)
            b_thrust_kn = (b_isp * G0 * b_prop / b_burn / 1000.0
                           if b_burn > 0 and b_prop > 0 and b_isp > 0
                           else getattr(p, 'booster_thrust_n', 0.0) / 1000.0)
            self._b_thrust_var.set(f"{b_thrust_kn:.1f}")
            self._b_burn_var.set(f"{b_burn:.1f}")
            self._b_inert_var.set(f"{getattr(p, 'booster_inert_kg', 0.0):.0f}")
            self._b_prop_var.set(f"{b_prop:.0f}")
            self._b_isp_var.set(f"{b_isp:.1f}")
            self._b_nozzle_var.set(f"{getattr(p, 'booster_nozzle_area_m2', 0.0):.4f}")
            self._b_diam_var.set(f"{getattr(p, 'booster_diam_m', 0.0):.2f}")
            self._b_length_var.set(f"{getattr(p, 'booster_length_m', 0.0):.2f}")
            self._b_cd_var.set(f"{getattr(p, 'booster_cd', 0.20):.2f}")
            self._b_core_delay_var.set(f"{getattr(p, 'booster_core_delay_s', 0.0):.1f}")
        self._update_booster_frame()

        # Apply show/hide state for all sections
        self._update_rv_separates_state()
        self._update_pbv_state()
        self._update_shroud_state()

        self._name_var.set(name)
        self._guidance_var.set(p.guidance)

    # ------------------------------------------------------------------
    def _collect(self) -> 'MissileParams':
        """Read and validate all fields; return a MissileParams linked list."""
        from missile_models import MissileParams, _FORDEN_MACH, _FORDEN_CD

        name = self._name_var.get().strip()
        if not name:
            raise ValueError("Missile name cannot be blank.")

        n = int(self._n_stages_var.get())

        # Throw weight / payload decomposition
        rv_separates = self._rv_separates_var.get()
        if rv_separates:
            try:
                num_rvs  = max(1, int(self._num_rvs_var.get()))
                rv_mass  = float(self._rv_mass_var.get())
                bus_mass = float(self._pbv_mass_var.get()) if self._has_pbv_var.get() else 0.0
            except ValueError:
                raise ValueError("Per-RV mass and No. of RVs must be numbers.")
            payload = num_rvs * rv_mass + bus_mass
        else:
            try:
                payload = float(self._throw_weight_var.get())
            except ValueError:
                raise ValueError("Throw weight must be a number.")
            num_rvs = 1
            rv_mass = payload
            bus_mass = 0.0
        try:
            rv_beta = float(self._rv_beta_var.get()) if rv_separates else 0.0
        except ValueError:
            rv_beta = 0.0

        # Payload shape / diameter / length
        _ps_label = self._payload_shape_var.get()
        nose_shape = next((k for k, v in NOSE_SHAPE_LABELS.items() if v == _ps_label),
                          "")
        try:
            payload_diameter_m = float(self._payload_diameter_var.get())
            nose_length_m      = float(self._payload_length_var.get())
        except ValueError:
            raise ValueError("Payload diameter and length must be numbers.")

        # RV geometry (round-trips the Estimate β dialog)
        _rv_shape_lbl = self._rv_shape_var.get()
        rv_shape = next((k for k, v in NOSE_SHAPE_LABELS.items() if v == _rv_shape_lbl),
                        "")
        try:
            rv_diameter_m = float(self._rv_diameter_var.get()) if rv_separates else 0.0
            rv_length_m   = float(self._rv_length_var.get())   if rv_separates else 0.0
        except ValueError:
            raise ValueError("RV diameter and length must be numbers.")

        # PBV geometry
        try:
            pbv_diameter_m = (float(self._pbv_diameter_var.get())
                              if (rv_separates and self._has_pbv_var.get()) else 0.0)
            pbv_length_m   = (float(self._pbv_length_var.get())
                              if (rv_separates and self._has_pbv_var.get()) else 0.0)
        except ValueError:
            raise ValueError("PBV diameter and length must be numbers.")

        # Shroud
        shroud_mass          = 0.0
        shroud_alt_km        = 80.0
        shroud_length_m      = 0.0
        shroud_diameter_m    = 0.0
        shroud_nose_shape    = ""
        shroud_nose_length_m = 0.0
        if self._shroud_var.get():
            try:
                shroud_mass          = float(self._shroud_mass_var.get())
                shroud_alt_km        = float(self._shroud_alt_var.get())
                shroud_length_m      = float(self._shroud_length_var.get())
                shroud_diameter_m    = float(self._shroud_diameter_var.get())
                _snl = self._shroud_nose_length_var.get().strip()
                shroud_nose_length_m = float(_snl) if _snl and float(_snl) > 0 \
                                       else shroud_length_m
            except ValueError:
                raise ValueError("Shroud fields must be numbers.")
            _slabel = self._shroud_nose_shape_var.get()
            shroud_nose_shape = next(
                (k for k, v in NOSE_SHAPE_LABELS.items() if v == _slabel), "")

        # Read and validate all active stage frames
        stages = []
        for i in range(n):
            sd = self._stage_frames[i].get()
            if sd["fueled"] <= sd["dry"]:
                raise ValueError(
                    f"Stage {i+1}: fueled mass must exceed dry mass.")
            stages.append(sd)

        # Build the linked list from the last stage back to the first.
        # Shroud lives on the first (bottom) stage; payload is part of the
        # last (top) stage's mass until final burnout.
        node = None
        upper_mass = 0.0
        for idx, sd in enumerate(reversed(stages)):
            stage_num = n - idx
            is_last  = (idx == 0)        # last stage of missile (first in reversed loop)
            is_first = (idx == n - 1)    # first stage of missile (last in reversed loop)
            prop = sd["fueled"] - sd["dry"]
            if is_last and is_first:
                # Single-stage missile
                m0     = sd["fueled"] + payload + shroud_mass
                mfinal = sd["dry"] if rv_separates else sd["dry"] + payload
            elif is_last:
                # Last of multiple stages: payload present, shroud is on stage 1
                m0     = sd["fueled"] + payload
                mfinal = sd["dry"] if rv_separates else sd["dry"] + payload
            elif is_first:
                # First of multiple stages: add shroud here
                m0     = sd["fueled"] + shroud_mass + upper_mass
                mfinal = sd["dry"]
            else:
                # Middle stage
                m0     = sd["fueled"] + upper_mass
                mfinal = sd["dry"]
            upper_mass = m0
            node = MissileParams(
                name=f"{name} Stage {stage_num}",
                mass_initial=m0,
                mass_propellant=prop,
                mass_final=mfinal,
                diameter_m=sd["dia"],  length_m=sd["length"],
                thrust_N=round(sd["thrust_kn"] * 1000.0),
                burn_time_s=sd["burn"], isp_s=sd["isp"],
                coast_time_s=sd["coast"] if not is_last else 0.0,
                nozzle_exit_area_m2=sd["nozzle_area"],
                mach_table=list(_FORDEN_MACH), cd_table=list(_FORDEN_CD),
                stage2=node,
                solid_motor=bool(sd.get("solid_motor", False)),
                grain_type=sd.get("grain_type", ""),
                thrust_peak_N=float(sd.get("thrust_peak_N", 0.0)),
                thrust_profile=list(sd.get("thrust_profile", [])),
            )

        node.name                   = name
        node.guidance               = self._guidance_var.get()
        node.payload_kg             = payload
        node.rv_beta_kg_m2          = rv_beta
        node.bus_mass_kg            = bus_mass
        node.num_rvs                = num_rvs
        node.rv_mass_kg             = rv_mass
        node.rv_separates           = rv_separates
        node.nose_shape             = nose_shape
        node.nose_length_m          = nose_length_m
        node.payload_diameter_m     = payload_diameter_m
        node.rv_shape               = rv_shape
        node.rv_diameter_m          = rv_diameter_m
        node.rv_length_m            = rv_length_m
        node.pbv_diameter_m         = pbv_diameter_m
        node.pbv_length_m           = pbv_length_m
        node.shroud_mass_kg         = shroud_mass
        node.shroud_jettison_alt_km = shroud_alt_km
        node.shroud_length_m        = shroud_length_m
        node.shroud_diameter_m      = shroud_diameter_m
        node.shroud_nose_shape      = shroud_nose_shape
        node.shroud_nose_length_m   = shroud_nose_length_m

        # Strap-on boosters
        try:
            _n_b = max(0, min(9, int(self._n_boosters_var.get())))
        except (ValueError, tk.TclError):
            _n_b = 0
        if _n_b > 0:
            try:
                _b_thrust_kn   = float(self._b_thrust_var.get())
                _b_burn        = float(self._b_burn_var.get())
                _b_core_delay  = float(self._b_core_delay_var.get())
                _b_inert       = float(self._b_inert_var.get())
                _b_prop        = float(self._b_prop_var.get())
                _b_isp         = float(self._b_isp_var.get())
                _b_nozzle      = float(self._b_nozzle_var.get())
                _b_diam        = float(self._b_diam_var.get())
                _b_length      = float(self._b_length_var.get())
                _b_cd          = float(self._b_cd_var.get())
            except ValueError as exc:
                raise ValueError(f"Booster field: {exc}") from exc
            if _b_burn <= 0:
                raise ValueError("Booster burn time must be > 0.")
            if _b_diam <= 0:
                raise ValueError("Booster diameter must be > 0.")
            node.n_boosters             = _n_b
            node.booster_thrust_n       = _b_thrust_kn * 1000.0
            node.booster_burn_time_s    = _b_burn
            node.booster_core_delay_s   = max(0.0, _b_core_delay)
            node.booster_inert_kg       = _b_inert
            node.booster_prop_kg        = _b_prop
            node.booster_isp_s          = _b_isp
            node.booster_nozzle_area_m2 = _b_nozzle
            node.booster_diam_m         = _b_diam
            node.booster_length_m       = _b_length
            node.booster_cd             = _b_cd

        return node

    # ------------------------------------------------------------------
    def _save(self):
        try:
            p = self._collect()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self)
            return
        self._on_save(p)
        self.destroy()

    def _save_as_new(self):
        try:
            p = self._collect()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e), parent=self)
            return
        new_name = simpledialog.askstring(
            "Save as New Missile",
            "Enter a name for the new missile:",
            initialvalue=p.name,
            parent=self)
        if not new_name or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name in MISSILE_DB:
            if not messagebox.askyesno(
                    "Overwrite?",
                    f"A missile named '{new_name}' already exists. Overwrite it?",
                    parent=self):
                return
        p.name = new_name
        self._on_save(p)
        self.destroy()

    # ------------------------------------------------------------------
    def _calc_rv_beta(self):
        """Open the Calculate β dialog (Newtonian hypersonic model)."""
        import math
        dlg = tk.Toplevel(self)
        dlg.title("Calculate RV β")
        dlg.resizable(False, False)
        dlg.grab_set()

        # ── Pre-fill from RV geometry fields; fall back to Stage 1 body ─
        try:
            _rv_d = float(self._rv_diameter_var.get())
            _rv_l = float(self._rv_length_var.get())
            if _rv_d > 0 and _rv_l > 0:
                _ld0    = _rv_l / _rv_d
                _theta0 = f"{math.degrees(math.atan(1.0 / (2.0 * _ld0))):.1f}"
                _dia_default = self._rv_diameter_var.get()
            else:
                raise ValueError
        except Exception:
            try:
                _nose_len = float(self._payload_length_var.get())
                _dia0     = float(self._stage_frames[0]._dia.get())
                if _nose_len > 0 and _dia0 > 0:
                    _ld0    = _nose_len / _dia0
                    _theta0 = f"{math.degrees(math.atan(1.0 / (2.0 * _ld0))):.1f}"
                else:
                    _theta0 = "10.0"
            except Exception:
                _theta0 = "10.0"
            try:
                _dia_default = self._stage_frames[0]._dia.get()
            except Exception:
                _dia_default = "0"

        # ── Input fields ──────────────────────────────────────────────
        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill=tk.X)
        frm.columnconfigure(1, weight=1)

        def _lbl(row, text):
            ttk.Label(frm, text=text).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 8), pady=3)

        _lbl(0, "RV mass (kg):")
        mass_var = tk.StringVar(value=self._rv_mass_var.get())
        ttk.Entry(frm, textvariable=mass_var, width=10).grid(
            row=0, column=1, sticky=tk.W)

        _lbl(1, "RV base diameter (m):")
        dia_var = tk.StringVar(value=_dia_default)
        ttk.Entry(frm, textvariable=dia_var, width=10).grid(
            row=1, column=1, sticky=tk.W)

        _lbl(2, "Cone half-angle (°):")
        theta_var = tk.StringVar(value=_theta0)
        ttk.Entry(frm, textvariable=theta_var, width=10).grid(
            row=2, column=1, sticky=tk.W)

        _lbl(3, "Nose radius / base radius:")
        eps_var = tk.StringVar(value="0.0")
        eps_inner = ttk.Frame(frm)
        eps_inner.grid(row=3, column=1, sticky=tk.W)
        ttk.Entry(eps_inner, textvariable=eps_var, width=10).pack(side=tk.LEFT)
        ttk.Label(eps_inner,
                  text="  (0 = sharp tip,  1 = hemisphere)",
                  foreground="gray50").pack(side=tk.LEFT)

        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)

        # ── Result display ────────────────────────────────────────────
        res_frm = ttk.Frame(dlg, padding=(12, 8))
        res_frm.pack(fill=tk.X)
        res_frm.columnconfigure(1, weight=1)

        def _res_row(row, label):
            ttk.Label(res_frm, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(0, 8), pady=2)
            lbl = ttk.Label(res_frm, text="—", foreground="gray40")
            lbl.grid(row=row, column=1, sticky=tk.W)
            return lbl

        cd_lbl   = _res_row(0, "Cd (Newtonian):")
        area_lbl = _res_row(1, "Reference area (m²):")
        beta_lbl = ttk.Label(res_frm, text="—",
                             font=("", 11, "bold"), foreground="navy")
        ttk.Label(res_frm, text="β = m / (Cd · A):").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 8), pady=2)
        beta_lbl.grid(row=2, column=1, sticky=tk.W)
        ttk.Label(res_frm,
                  text="Hypersonic Newtonian flow (Mach > 8).  Chart: Ref (4) Ch. 5.",
                  foreground="gray50").grid(
            row=3, column=0, columnspan=2, sticky=tk.W, pady=(4, 0))

        _beta_result = [None]

        def _compute(*_):
            try:
                mass  = float(mass_var.get())
                dia   = float(dia_var.get())
                theta = float(theta_var.get())
                eps   = float(eps_var.get())
                if dia <= 0 or mass <= 0 or theta <= 0:
                    raise ValueError
            except ValueError:
                cd_lbl.config(text="—")
                area_lbl.config(text="—")
                beta_lbl.config(text="invalid input")
                _beta_result[0] = None
                return

            cd   = _cd_blunted_cone_newtonian(theta, eps)
            area = math.pi * (dia / 2.0) ** 2
            beta = mass / (cd * area) if cd > 0 else float('inf')

            cd_lbl.config(  text=f"{cd:.4f}")
            area_lbl.config(text=f"{area:.4f} m²")
            beta_lbl.config(text=f"{beta:,.0f} kg/m²")
            _beta_result[0] = beta

        for _v in (mass_var, dia_var, theta_var, eps_var):
            _v.trace_add("write", _compute)
        _compute()

        # ── Buttons ───────────────────────────────────────────────────
        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)
        btn_frm = ttk.Frame(dlg, padding=(12, 8))
        btn_frm.pack(fill=tk.X)

        def _use():
            if _beta_result[0] is not None and _beta_result[0] != float('inf'):
                self._rv_beta_var.set(f"{_beta_result[0]:.0f}")
            dlg.destroy()

        ttk.Button(btn_frm, text="Use this value",
                   command=_use).pack(side=tk.LEFT)
        ttk.Button(btn_frm, text="Cancel",
                   command=dlg.destroy).pack(side=tk.LEFT, padx=6)


# ---------------------------------------------------------------------------
# Range-ring dialog
# ---------------------------------------------------------------------------

class RangeRingDialog(tk.Toplevel):
    """Compute and export a maximum-range ring for the current missile.

    Sweeps 72 azimuths (every 5°) using maximize_range(), collects the
    impact point for each direction, then renders the closed polygon on a
    Cartopy map using the shared projection picker.
    """

    _N_AZ = 72   # number of azimuths → 5° spacing

    def __init__(self, app):
        super().__init__(app)
        self._app   = app
        self._ring  = None   # list of (lon, lat) impact points once computed
        self._stop  = threading.Event()

        self.title("Range Ring")
        self.resizable(False, False)
        self.grab_set()

        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.BOTH)
        frm.columnconfigure(1, weight=1)

        # Missile label (informational)
        ttk.Label(frm, text="Missile:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._missile_lbl = ttk.Label(frm, text=app._missile_var.get(),
                                      foreground="navy")
        self._missile_lbl.grid(row=0, column=1, sticky=tk.W)

        # Launch lat
        ttk.Label(frm, text="Launch lat (°N):").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._lat_var = tk.StringVar(value=app._launch_lat.get())
        ttk.Entry(frm, textvariable=self._lat_var, width=12).grid(
            row=1, column=1, sticky=tk.W)

        # Launch lon
        ttk.Label(frm, text="Launch lon (°E):").grid(
            row=2, column=0, sticky=tk.W, padx=(0, 8), pady=3)
        self._lon_var = tk.StringVar(value=app._launch_lon.get())
        ttk.Entry(frm, textvariable=self._lon_var, width=12).grid(
            row=2, column=1, sticky=tk.W)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)

        # Progress bar + status label
        prog_frm = ttk.Frame(self, padding=(12, 6))
        prog_frm.pack(fill=tk.X)
        self._prog_var = tk.StringVar(value="Press Compute to start.")
        ttk.Label(prog_frm, textvariable=self._prog_var).pack(anchor=tk.W)
        self._pbar = ttk.Progressbar(prog_frm, maximum=self._N_AZ,
                                     mode="determinate")
        self._pbar.pack(fill=tk.X, pady=(4, 0))

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12)

        # Buttons
        btn_frm = ttk.Frame(self, padding=(12, 8))
        btn_frm.pack(fill=tk.X)
        self._compute_btn = ttk.Button(btn_frm, text="Compute Ring",
                                       command=self._compute)
        self._compute_btn.pack(side=tk.LEFT)
        self._cancel_btn = ttk.Button(btn_frm, text="Stop",
                                      command=self._cancel,
                                      state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, padx=6)
        self._export_btn = ttk.Button(btn_frm, text="Export Map…",
                                      command=self._export,
                                      state=tk.DISABLED)
        self._export_btn.pack(side=tk.LEFT, padx=(24, 0))
        ttk.Button(btn_frm, text="Close",
                   command=self.destroy).pack(side=tk.RIGHT)

        app._center_dialog(self)

    # ------------------------------------------------------------------
    def _compute(self):
        try:
            lat = float(self._lat_var.get())
            lon = float(self._lon_var.get())
        except ValueError:
            messagebox.showerror("Input error",
                                 "Enter valid launch lat/lon.", parent=self)
            return

        try:
            (missile, guidance, _la, _lo, _az, cutoff, la,
             gt_start_s, gt_stop_s, _orb,
             _yaw_maneuvers, launch_elevation_deg) = self._app._get_inputs()
        except Exception as e:
            messagebox.showerror("Input error", str(e), parent=self)
            return

        self._ring = None
        self._stop.clear()
        self._pbar["value"] = 0
        self._prog_var.set(f"Computing 0 / {self._N_AZ}…")
        self._compute_btn.config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._export_btn.config(state=tk.DISABLED)

        threading.Thread(
            target=self._worker,
            args=(missile, guidance, lat, lon, la,
                  gt_start_s, gt_stop_s, launch_elevation_deg),
            daemon=True,
        ).start()

    def _cancel(self):
        self._stop.set()

    def _worker(self, missile, guidance, lat, lon, la,
                gt_start_s, gt_stop_s, launch_elevation_deg):
        azimuths = np.linspace(0.0, 360.0, self._N_AZ, endpoint=False)
        points   = []   # (az, impact_lon, impact_lat)

        for i, az in enumerate(azimuths):
            if self._stop.is_set():
                self.after(0, self._on_cancelled)
                return
            try:
                result = maximize_range(
                    missile, lat, lon, az,
                    guidance=guidance,
                    burnout_angle_deg=la,
                    gt_turn_start_s=gt_start_s,
                    gt_turn_stop_s=gt_stop_s,
                )
                ms_list = result.get('milestones', [])
                impact  = next(
                    (m for m in ms_list
                     if 'impact' in m.get('event', '').lower()
                     and not m.get('is_debris', False)),
                    None)
                if impact:
                    t_arr  = np.asarray(result['t'])
                    la_arr = np.asarray(result['lat'])
                    lo_arr = np.asarray(result['lon'])
                    imp_lat = float(np.interp(impact['t_s'], t_arr, la_arr))
                    imp_lon = float(np.interp(impact['t_s'], t_arr, lo_arr))
                    points.append((az, imp_lon, imp_lat))
            except Exception:
                pass   # skip failed azimuths silently

            self.after(0, self._on_progress, i + 1, len(points))

        self.after(0, self._on_done, points, lat, lon)

    def _on_progress(self, done, n_ok):
        if not self.winfo_exists():
            return
        self._pbar["value"] = done
        self._prog_var.set(
            f"Computing {done} / {self._N_AZ}… ({n_ok} points OK)")

    def _on_cancelled(self):
        if not self.winfo_exists():
            return
        self._prog_var.set("Cancelled.")
        self._compute_btn.config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)

    def _on_done(self, points, launch_lat, launch_lon):
        if not self.winfo_exists():
            return
        self._cancel_btn.config(state=tk.DISABLED)
        self._compute_btn.config(state=tk.NORMAL)
        if len(points) < 3:
            self._prog_var.set(
                f"Too few valid azimuths ({len(points)}). Check missile params.")
            return
        avg_range = float(np.mean([
            np.sqrt((p[2] - launch_lat)**2 + (p[1] - launch_lon)**2)
            for p in points]))
        self._ring         = points
        self._launch_lat   = launch_lat
        self._launch_lon   = launch_lon
        self._prog_var.set(
            f"Done — {len(points)} / {self._N_AZ} azimuths succeeded.")
        self._export_btn.config(state=tk.NORMAL)

    # ------------------------------------------------------------------
    def _export(self):
        if not self._ring:
            return
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import matplotlib.patheffects as pe
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ImportError as _e:
            messagebox.showerror("Missing package",
                                 f"Cartopy not installed.\n{_e}", parent=self)
            return

        # Reuse the full export-options dialog (projection + map extent).
        mid_lon = float(np.mean([p[1] for p in self._ring]))
        mid_lat = float(np.mean([p[2] for p in self._ring]))
        proj, extent_spec = self._app._pick_cartopy_export_options(mid_lon, mid_lat)
        if proj is None:
            return

        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image",    "*.png"), ("PDF document", "*.pdf"),
                       ("SVG image",    "*.svg"), ("All files",    "*.*")],
            title="Save range-ring map",
            parent=self,
        )
        if not path:
            return

        geo    = ccrs.Geodetic()
        fig    = Figure(figsize=(10, 8), dpi=300)
        canvas = FigureCanvasAgg(fig)
        ax     = fig.add_subplot(1, 1, 1, projection=proj)

        # ── Map extent ────────────────────────────────────────────────
        if extent_spec is None:
            ax.set_global()
        elif extent_spec[0] == 'auto':
            pad_frac = extent_spec[1] / 100.0
            ring_lats = [p[2] for p in self._ring] + [self._launch_lat]
            ring_lons = [p[1] for p in self._ring] + [self._launch_lon]
            lat_span = max(float(max(ring_lats) - min(ring_lats)), 2.0)
            lon_span = max(float(max(ring_lons) - min(ring_lons)), 2.0)
            ax.set_extent([
                max(-180.0, min(ring_lons) - lon_span * pad_frac),
                min(+180.0, max(ring_lons) + lon_span * pad_frac),
                max( -90.0, min(ring_lats) - lat_span * pad_frac),
                min( +90.0, max(ring_lats) + lat_span * pad_frac),
            ], crs=ccrs.PlateCarree())
        else:
            ax.set_extent(list(extent_spec), crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.OCEAN,     facecolor="#d6e8f5", zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor="#e8e4d8", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555555",
                       zorder=2)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#888888",
                       linestyle=":", zorder=2)
        ax.add_feature(cfeature.LAKES,     facecolor="#d6e8f5", linewidth=0.3,
                       edgecolor="#555555", zorder=2)
        ax.gridlines(color="white", linewidth=0.4, linestyle="--", alpha=0.6,
                     zorder=3)

        # Ring polygon — close it by repeating the first point.
        ring_lons = [p[1] for p in self._ring] + [self._ring[0][1]]
        ring_lats = [p[2] for p in self._ring] + [self._ring[0][2]]
        ax.fill(ring_lons, ring_lats,
                color="crimson", alpha=0.12, transform=geo, zorder=4)
        ax.plot(ring_lons, ring_lats,
                color="crimson", linewidth=1.6,
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
                transform=geo, zorder=5)

        # Launch point
        ax.plot(self._launch_lon, self._launch_lat,
                marker="^", markersize=8, color="black",
                markeredgecolor="white", markeredgewidth=1.0,
                transform=geo, zorder=6)

        # Estimate average max range for the title.
        ranges_km = []
        for _, imp_lon, imp_lat in self._ring:
            try:
                km = range_between(
                    np.radians(self._launch_lat), np.radians(self._launch_lon),
                    np.radians(imp_lat), np.radians(imp_lon)) / 1000.0
                ranges_km.append(km)
            except Exception:
                pass
        rng_str = (f"~{np.mean(ranges_km):.0f} km max range"
                   if ranges_km else "max range")
        missile_name = self._app._missile_var.get()
        ax.set_title(f"{missile_name}  ·  {rng_str}", fontsize=11, pad=8)

        fig.tight_layout()
        canvas.print_figure(path, bbox_inches="tight")
        self._app._status_var.set(f"Range ring saved: {path}")
        _open_file(path)


# ---------------------------------------------------------------------------
# Parametric sweep / sensitivity-analysis dialog
# ---------------------------------------------------------------------------

class ParametricSweepDialog(tk.Toplevel):
    """Non-modal dialog for 1-D parametric trajectory sweep.

    Reproduces the analyses Forden performs in all three worked examples:
      • Table 2  — Range vs azimuth (vary azimuth, fixed loft / cutoff)
      • Figure 7 — Range vs loft angle (vary burnout_angle_deg)
      • Ad hoc   — Range vs cutoff time (vary engine cutoff)

    The user picks which parameter to vary plus a start/stop/step range;
    the remaining parameters are taken from the main window at the moment
    "Run Sweep" is clicked.  Results appear incrementally in a live plot
    and a scrollable table.  An "Overplot trajectories" option shows all
    altitude-vs-range profiles on one axes (≤ 20 curves).
    """

    _PARAM_INFO = {
        "Azimuth":     dict(key="azimuth",    lo=0.0,  hi=360.0, step=5.0,  unit="°"),
        "Loft Angle":  dict(key="loft_angle", lo=10.0, hi=80.0,  step=5.0,  unit="°"),
        "Cutoff Time": dict(key="cutoff",     lo=None, hi=None,  step=5.0,  unit="s"),
        "Turn Stop":   dict(key="turn_stop",  lo=None, hi=None,  step=10.0, unit="s"),
    }

    def __init__(self, parent_app):
        super().__init__(parent_app)
        self.title("Parametric Sweep / Sensitivity Analysis")
        self.geometry("820x680")
        self.resizable(True, True)
        self._app        = parent_app
        self._stop_evt   = threading.Event()
        self._results    = []          # list of (param_val, range_km, apogee_km)
        self._traj_store = []          # list of (param_val, result_dict), for overplot
        self._build()

    # ------------------------------------------------------------------
    def _build(self):
        pad = dict(padx=8, pady=4)

        # ── Sweep configuration ────────────────────────────────────────
        cf = ttk.LabelFrame(self, text="Sweep Configuration")
        cf.pack(fill=tk.X, **pad)

        row0 = ttk.Frame(cf)
        row0.pack(fill=tk.X, padx=6, pady=(4, 2))

        ttk.Label(row0, text="Vary:").pack(side=tk.LEFT)
        self._param_var = tk.StringVar(value="Azimuth")
        pcb = ttk.Combobox(row0, textvariable=self._param_var,
                           values=list(self._PARAM_INFO.keys()),
                           state="readonly", width=14)
        pcb.pack(side=tk.LEFT, padx=(4, 12))
        pcb.bind("<<ComboboxSelected>>", self._on_param_changed)

        self._lo_var   = tk.StringVar(value="0.0")
        self._hi_var   = tk.StringVar(value="360.0")
        self._step_var = tk.StringVar(value="5.0")
        for lbl, var in [("From:", self._lo_var), ("To:", self._hi_var),
                          ("Step:", self._step_var)]:
            ttk.Label(row0, text=lbl).pack(side=tk.LEFT, padx=(4, 1))
            ttk.Entry(row0, textvariable=var, width=7).pack(side=tk.LEFT)

        opts = ttk.Frame(cf)
        opts.pack(fill=tk.X, padx=6, pady=(2, 6))
        ttk.Label(opts, text="Show:").pack(side=tk.LEFT)
        self._show_range  = tk.BooleanVar(value=True)
        self._show_apogee = tk.BooleanVar(value=True)
        self._overplot    = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Range",  variable=self._show_range ).pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(opts, text="Apogee", variable=self._show_apogee).pack(side=tk.LEFT, padx=4)
        ttk.Separator(opts, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=2)
        ttk.Checkbutton(opts, text="Overplot trajectory profiles (≤ 20 pts)",
                        variable=self._overplot).pack(side=tk.LEFT, padx=4)

        # ── Buttons + progress ─────────────────────────────────────────
        bf = ttk.Frame(self)
        bf.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._run_btn = ttk.Button(bf, text="▶  Run Sweep", command=self._run)
        self._run_btn.pack(side=tk.LEFT, padx=(0, 4))
        self._cancel_btn = ttk.Button(bf, text="■  Cancel",
                                      command=self._cancel, state=tk.DISABLED)
        self._cancel_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(bf, text="Export CSV…", command=self._export).pack(side=tk.LEFT, padx=4)
        self._prog_lbl = tk.StringVar(value="")
        ttk.Label(bf, textvariable=self._prog_lbl).pack(side=tk.LEFT, padx=8)
        self._progressbar = ttk.Progressbar(bf, mode="determinate", length=180)
        self._progressbar.pack(side=tk.RIGHT, padx=(4, 0))

        # ── Embedded matplotlib figure ─────────────────────────────────
        self._fig     = Figure(figsize=(8, 3.2), dpi=96)
        self._canvas  = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8)
        NavigationToolbar2Tk(self._canvas, self).update()
        self._init_plot()

        # ── Results table ──────────────────────────────────────────────
        tf = ttk.LabelFrame(self, text="Results Table")
        tf.pack(fill=tk.X, padx=8, pady=(4, 8))
        self._tree = ttk.Treeview(tf,
                                  columns=("param", "range", "apogee"),
                                  show="headings", height=6)
        self._tree.heading("param",  text="Parameter")
        self._tree.heading("range",  text="Range (km)")
        self._tree.heading("apogee", text="Apogee (km)")
        for col in ("param", "range", "apogee"):
            self._tree.column(col, width=120, anchor=tk.CENTER)
        vsb = ttk.Scrollbar(tf, orient=tk.VERTICAL, command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.X, padx=4, pady=4)

    # ------------------------------------------------------------------
    def _init_plot(self):
        self._fig.clf()
        ax = self._fig.add_subplot(111)
        ax.set_title("Run a sweep to see results", fontsize=9)
        ax.grid(True, alpha=0.35)
        self._fig.tight_layout(pad=2.5)
        self._canvas.draw()

    # ------------------------------------------------------------------
    def _on_param_changed(self, _event=None):
        info = self._PARAM_INFO[self._param_var.get()]
        lo   = info["lo"]
        hi   = info["hi"]
        if lo is None or hi is None:
            # Cutoff: derive from selected missile
            try:
                missile, *_ = self._app._get_inputs()
                lo, hi = 5.0, float(int(total_burn_time(missile)))
            except Exception:
                lo, hi = 5.0, 100.0
        self._lo_var  .set(str(lo))
        self._hi_var  .set(str(hi))
        self._step_var.set(str(info["step"]))
        # Update table column header
        self._tree.heading("param", text=f"{self._param_var.get()} ({info['unit']})")

    # ------------------------------------------------------------------
    def _make_points(self):
        lo   = float(self._lo_var.get())
        hi   = float(self._hi_var.get())
        step = float(self._step_var.get())
        if step <= 0:
            raise ValueError("Step must be > 0.")
        n = max(2, int(round((hi - lo) / step)) + 1)
        return np.linspace(lo, hi, n)

    # ------------------------------------------------------------------
    def _run(self):
        self._stop_evt.clear()
        try:
            (missile, guidance, lat, lon, az, cutoff, la,
             gt_start_s, gt_stop_s, _orb,
             _yaw_maneuvers, launch_elevation_deg) = self._app._get_inputs()
        except Exception as e:
            messagebox.showerror("Input error", str(e), parent=self)
            return
        try:
            points = self._make_points()
        except Exception as e:
            messagebox.showerror("Sweep range error", str(e), parent=self)
            return

        if cutoff is None:
            cutoff = total_burn_time(missile)

        overplot = self._overplot.get()
        if overplot and len(points) > 20:
            messagebox.showwarning(
                "Too many points for overplot",
                f"Overplot is limited to 20 trajectory profiles.\n"
                f"Your sweep has {len(points)} points — overplot will be skipped.\n"
                "Increase the step size or disable 'Overplot trajectory profiles'.",
                parent=self)
            overplot = False

        param_key = self._PARAM_INFO[self._param_var.get()]["key"]
        self._results    = []
        self._traj_store = []
        self._tree.delete(*self._tree.get_children())
        self._progressbar["maximum"] = len(points)
        self._progressbar["value"]   = 0
        self._prog_lbl.set(f"0 / {len(points)}")
        self._run_btn   .config(state=tk.DISABLED)
        self._cancel_btn.config(state=tk.NORMAL)
        self._init_plot()

        threading.Thread(
            target=self._sweep_worker,
            args=(missile, guidance, lat, lon, az, la, cutoff,
                  param_key, points, overplot, gt_start_s, gt_stop_s,
                  launch_elevation_deg),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    def _sweep_worker(self, missile, guidance, lat, lon, az, la, cutoff,
                      param_key, points, store_trajs, gt_start_s=5.0, gt_stop_s=None,
                      launch_elevation_deg=90.0):
        for i, val in enumerate(points):
            if self._stop_evt.is_set():
                break
            run_az      = val if param_key == "azimuth"    else az
            run_la      = val if param_key == "loft_angle" else la
            run_cut     = val if param_key == "cutoff"     else cutoff
            run_gt_stop = val if param_key == "turn_stop"  else gt_stop_s
            try:
                r = integrate_trajectory(
                    missile, lat, lon, run_az,
                    guidance=guidance,
                    burnout_angle_deg=run_la,
                    cutoff_time_s=run_cut,
                    gt_turn_start_s=gt_start_s,
                    gt_turn_stop_s=run_gt_stop,
                    launch_elevation_deg=launch_elevation_deg,
                )
                row  = (val, r["range_km"] if r["range_km"] is not None else float("nan"),
                        r["apogee_km"])
                traj = (val, r) if store_trajs else None
            except Exception:
                row  = (val, float("nan"), float("nan"))
                traj = None
            self.after(0, self._add_point, row, traj, i + 1, len(points))
        self.after(0, self._sweep_done)

    # ------------------------------------------------------------------
    def _add_point(self, row, traj, done, total):
        self._results.append(row)
        if traj is not None:
            self._traj_store.append(traj)
        val, rng, apo = row
        self._tree.insert("", tk.END, values=(
            f"{val:.2f}",
            f"{rng:.1f}"  if np.isfinite(rng) else "—",
            f"{apo:.1f}"  if np.isfinite(apo) else "—",
        ))
        self._tree.yview_moveto(1.0)
        self._progressbar["value"] = done
        self._prog_lbl.set(f"{done} / {total}")
        self._redraw()

    # ------------------------------------------------------------------
    def _sweep_done(self):
        self._run_btn   .config(state=tk.NORMAL)
        self._cancel_btn.config(state=tk.DISABLED)
        n = len(self._results)
        cancelled = self._stop_evt.is_set()
        self._prog_lbl.set(
            f"{'Cancelled after' if cancelled else 'Done —'} {n} point{'s' if n != 1 else ''}.")
        self._redraw()

    # ------------------------------------------------------------------
    def _cancel(self):
        self._stop_evt.set()

    # ------------------------------------------------------------------
    def _redraw(self):
        if not self._results:
            return

        info   = self._PARAM_INFO[self._param_var.get()]
        xlabel = f"{self._param_var.get()} ({info['unit']})"

        xs          = [r[0] for r in self._results]
        ys_range    = [r[1] for r in self._results]
        ys_apogee   = [r[2] for r in self._results]

        self._fig.clf()

        if self._traj_store:
            # ── Overplot trajectory profiles ──────────────────────────
            ax   = self._fig.add_subplot(111)
            cmap = matplotlib.cm.viridis
            vals = [t[0] for t in self._traj_store]
            vmin, vmax = min(vals), max(vals)
            span = max(vmax - vmin, 1e-9)
            for pval, r in self._traj_store:
                color = cmap((pval - vmin) / span)
                ax.plot(r["range"] / 1000.0, r["alt"] / 1000.0,
                        color=color, linewidth=1.0, alpha=0.85,
                        label=f"{pval:.1f}")
            ax.set_xlabel("Downrange (km)", fontsize=8)
            ax.set_ylabel("Altitude (km)",  fontsize=8)
            ax.set_title(f"Trajectory Profiles  —  {self._param_var.get()} sweep",
                         fontsize=9)
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=7)
            if len(self._traj_store) <= 12:
                ax.legend(title=info["unit"], fontsize=6, title_fontsize=6,
                          loc="upper right")
            sm = matplotlib.cm.ScalarMappable(
                cmap=cmap,
                norm=matplotlib.colors.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cb = self._fig.colorbar(sm, ax=ax, pad=0.02)
            cb.set_label(f"{self._param_var.get()} ({info['unit']})", fontsize=7)
            cb.ax.tick_params(labelsize=6)
        else:
            # ── Range / apogee vs parameter ───────────────────────────
            sr = self._show_range.get()
            sa = self._show_apogee.get()

            if sr and sa:
                ax  = self._fig.add_subplot(111)
                ax2 = ax.twinx()
                ax .plot(xs, ys_range,  "b-o",  markersize=3, linewidth=1.5, label="Range")
                ax2.plot(xs, ys_apogee, "r--s", markersize=3, linewidth=1.2, label="Apogee")
                ax .set_ylabel("Range (km)",  color="royalblue",  fontsize=8)
                ax2.set_ylabel("Apogee (km)", color="firebrick", fontsize=8)
                ax .tick_params(axis="y", labelcolor="royalblue",  labelsize=7)
                ax2.tick_params(axis="y", labelcolor="firebrick", labelsize=7)
                lines  = ax.get_lines() + ax2.get_lines()
                ax.legend(lines, [l.get_label() for l in lines], fontsize=7)
            elif sr:
                ax = self._fig.add_subplot(111)
                ax.plot(xs, ys_range, "b-o", markersize=3, linewidth=1.5)
                ax.set_ylabel("Range (km)", fontsize=8)
            else:
                ax = self._fig.add_subplot(111)
                ax.plot(xs, ys_apogee, "r-s", markersize=3, linewidth=1.5)
                ax.set_ylabel("Apogee (km)", fontsize=8)

            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_title(f"{self._param_var.get()} Sweep", fontsize=9)
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=7)

        self._fig.tight_layout(pad=2.5)
        self._canvas.draw_idle()

    # ------------------------------------------------------------------
    def _export(self):
        if not self._results:
            messagebox.showinfo("No data", "Run a sweep first.", parent=self)
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export sweep results",
            parent=self,
        )
        if not path:
            return
        info   = self._PARAM_INFO[self._param_var.get()]
        header = f"{self._param_var.get()}_{info['unit']},Range_km,Apogee_km"
        np.savetxt(path, np.array(self._results),
                   delimiter=",", header=header, comments="")
        self._app._status_var.set(f"Sweep exported: {path}")


# ---------------------------------------------------------------------------
# Helper: labelled decimal-degree row in a grid parent
# ---------------------------------------------------------------------------
def _dd_row(parent, label, row, default="0.0"):
    """Pack a label + decimal-degree Entry into a grid row; return the StringVar."""
    ttk.Label(parent, text=label).grid(row=row, column=0,
                                       sticky=tk.W, padx=(8, 2), pady=2)
    var = tk.StringVar(value=default)
    inner = ttk.Frame(parent)
    inner.grid(row=row, column=1, sticky=tk.W, padx=(0, 8), pady=2)
    ttk.Entry(inner, textvariable=var, width=10).pack(side=tk.LEFT)
    ttk.Label(inner, text="°").pack(side=tk.LEFT, padx=(2, 0))
    return var


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class MissileFlyoutApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Thrusty")
        self.minsize(900, 700)
        # Size to 92 % of the available screen, capped at 1600 × 1050.
        self.update_idletasks()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        w = min(1600, int(sw * 0.92))
        h = min(1050, int(sh * 0.92))
        x = (sw - w) // 2
        y = max(0, (sh - h) // 2 - 24)   # shift up slightly for macOS menu bar
        self.geometry(f"{w}x{h}+{x}+{y}")

        self._result         = None
        self._running        = False
        self._notam_overlay  = None   # list of GeoJSON-style polygon rings, or None

        _load_custom_missiles()      # restore any user-saved missiles

        self._build_menu()
        self._build_ui()
        self._on_missile_changed()   # populate params tab with default missile

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _center_dialog(self, dlg):
        """Centre a Toplevel dialog over the main window."""
        dlg.update_idletasks()
        px = self.winfo_rootx() + (self.winfo_width()  - dlg.winfo_reqwidth())  // 2
        py = self.winfo_rooty() + (self.winfo_height() - dlg.winfo_reqheight()) // 2
        dlg.geometry(f"+{max(0, px)}+{max(0, py)}")

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        # ── Modeling inputs (load/save) ───────────────────────────────
        file_menu.add_command(label="Load Missile…",            command=self._load_missile)
        file_menu.add_command(label="Save Missile…",            command=self._export_missile)
        file_menu.add_command(label="Load Missile from XLSX…",  command=self._import_missile_xlsx)
        file_menu.add_command(label="Save Missile to XLSX…",    command=self._export_missile_xlsx)
        file_menu.add_command(label="New Missile XLSX Template…", command=self._new_missile_template)
        file_menu.add_separator()
        file_menu.add_command(label="Load Guidance…",           command=self._import_guidance)
        file_menu.add_command(label="Save Guidance…",           command=self._export_guidance)
        file_menu.add_separator()
        file_menu.add_command(label="Load Launch Site…",        command=self._load_site)
        file_menu.add_command(label="Save Launch Site…",        command=self._export_site)
        file_menu.add_separator()
        # ── Trajectory outcomes (export only) ─────────────────────────
        file_menu.add_command(label="Export Trajectory CSV…",   command=self._save_trajectory)
        file_menu.add_command(label="Export Trajectory XLSX…",  command=self._export_trajectory_xlsx)
        file_menu.add_command(label="Export Trajectory KML…",   command=self._export_kml)
        file_menu.add_separator()
        # ── Flight events (export only) ───────────────────────────────
        file_menu.add_command(label="Export Flight Events CSV…",  command=self._export_timeline)
        file_menu.add_command(label="Export Flight Events XLSX…", command=self._export_timeline_xlsx)
        file_menu.add_separator()
        # ── Cartographic (export only) ────────────────────────────────
        file_menu.add_command(label="Open Folium Map…",       command=self._export_folium)
        file_menu.add_command(label="Export Cartopy Map…",    command=self._export_cartopy)
        file_menu.add_command(label="Export Figures…",        command=self._export_figures)
        file_menu.add_separator()
        file_menu.add_command(label="Load NOTAM overlay…",    command=self._load_notam_overlay)
        file_menu.add_command(label="Clear NOTAM overlay",    command=self._clear_notam_overlay)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Parametric Sweep…",        command=self._open_sweep)
        analysis_menu.add_command(label="Range Ring (Cartopy)…",    command=self._open_range_ring)
        analysis_menu.add_command(label="Aim at Target (liquid)…",  command=self._aim_at_target)
        menubar.add_cascade(label="Analysis", menu=analysis_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About…", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.config(menu=menubar)

    # ------------------------------------------------------------------
    # Top-level layout
    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # Left control panel — fixed width, vertically scrollable
        LEFT_W = 490
        left_outer = ttk.Frame(top, width=LEFT_W)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left_outer.pack_propagate(False)

        left_canvas = tk.Canvas(left_outer, highlightthickness=0)
        left_vsb = ttk.Scrollbar(left_outer, orient=tk.VERTICAL,
                                  command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_vsb.set)
        left_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        left = ttk.Frame(left_canvas)
        _left_win = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def _left_on_frame(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        def _left_on_canvas(event):
            left_canvas.itemconfig(_left_win, width=event.width)
        left.bind("<Configure>", _left_on_frame)
        left_canvas.bind("<Configure>", _left_on_canvas)

        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            left_canvas.bind(seq,
                lambda e, c=left_canvas: c.yview_scroll(
                    -1 if e.num == 4 else (1 if e.num == 5
                    else -1 * (e.delta // 120)), "units"))

        self._build_control_panel(left)

        # Right panel — tabbed notebook (Plots | Flight Timeline)
        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Pinned results strip — always visible above the notebook tabs
        self._results_strip_var = tk.StringVar(value="")
        results_strip = ttk.Frame(right, relief=tk.GROOVE, borderwidth=1)
        results_strip.pack(fill=tk.X, padx=2, pady=(0, 3))
        ttk.Label(results_strip, textvariable=self._results_strip_var,
                  anchor=tk.W).pack(
            fill=tk.X, padx=8, pady=3)

        self._right_nb = ttk.Notebook(right)
        self._right_nb.pack(fill=tk.BOTH, expand=True)

        plots_tab    = ttk.Frame(self._right_nb)
        timeline_tab = ttk.Frame(self._right_nb)
        params_tab   = ttk.Frame(self._right_nb)
        slv_tab      = ttk.Frame(self._right_nb)
        self._right_nb.add(plots_tab,    text="  Plots  ")
        self._right_nb.add(timeline_tab, text="  Flight Timeline  ")
        self._right_nb.add(params_tab,   text="  Missile Parameters  ")
        self._right_nb.add(slv_tab,      text="  SLV Performance  ")

        self._build_plot_panel(plots_tab)
        self._build_timeline_panel(timeline_tab)
        self._build_params_tab(params_tab)
        self._build_slv_tab(slv_tab)

        # Status bar
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

    # ------------------------------------------------------------------
    # Control panel  (mirrors Forden's left-side panel)
    # ------------------------------------------------------------------
    def _build_control_panel(self, parent):
        # ── Title ──────────────────────────────────────────────────────
        ttk.Label(parent, text="Thrusty",
                  font=("", 11, "bold")).pack(pady=(6, 2))
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── Missile type ───────────────────────────────────────────────
        mf = ttk.LabelFrame(parent, text="Missile Type")
        mf.pack(fill=tk.X, padx=6, pady=3)
        _cb_values   = list(MISSILE_DB.keys())
        _first_valid = _cb_values[0] if _cb_values else ""
        self._last_valid_missile = _first_valid
        self._missile_var = tk.StringVar(value=_first_valid)
        self._missile_cb = ttk.Combobox(mf, textvariable=self._missile_var,
                                        values=_cb_values,
                                        state="readonly", width=24)
        self._missile_cb.pack(padx=6, pady=(4, 2))
        self._missile_cb.bind("<<ComboboxSelected>>", self._on_missile_changed)
        _bind_typeahead(self._missile_cb)

        mb = ttk.Frame(mf)
        mb.pack(padx=6, pady=(0, 4))
        ttk.Button(mb, text="New",    width=7,
                   command=self._new_missile).pack(side=tk.LEFT, padx=2)
        ttk.Button(mb, text="Edit…",  width=7,
                   command=self._edit_missile).pack(side=tk.LEFT, padx=2)
        self._del_btn = ttk.Button(mb, text="Delete", width=7,
                                   command=self._delete_missile,
                                   state=tk.DISABLED)
        self._del_btn.pack(side=tk.LEFT, padx=2)

        # ── Units ──────────────────────────────────────────────────────
        uf = ttk.LabelFrame(parent, text="Display Units for Plots")
        uf.pack(fill=tk.X, padx=6, pady=3)
        self._units_var = tk.StringVar(value="km")
        uf_inner = ttk.Frame(uf)
        uf_inner.pack(pady=3)
        for val, lbl in [("km", "km"), ("nm", "nmi"), ("mi", "miles")]:
            ttk.Radiobutton(uf_inner, text=lbl, variable=self._units_var,
                            value=val).pack(side=tk.LEFT, padx=8)

        # ── Launch site ────────────────────────────────────────────────
        lf = ttk.LabelFrame(parent, text="Launch Site")
        lf.pack(fill=tk.X, padx=6, pady=3)

        _site_values, self._site_map = _load_launch_sites()
        self._site_var = tk.StringVar(value="")
        self._site_cb = ttk.Combobox(lf, textvariable=self._site_var,
                                     values=_site_values, state="readonly", width=26)
        self._site_cb.pack(padx=6, pady=(4, 2))
        self._site_cb.bind("<<ComboboxSelected>>", self._on_site_selected)
        _bind_typeahead(self._site_cb)

        sb = ttk.Frame(lf)
        sb.pack(padx=6, pady=(0, 4))
        ttk.Button(sb, text="New",    width=7,
                   command=self._new_site).pack(side=tk.LEFT, padx=2)
        ttk.Button(sb, text="Edit…",  width=7,
                   command=self._edit_site).pack(side=tk.LEFT, padx=2)
        self._site_del_btn = ttk.Button(sb, text="Delete", width=7,
                                        command=self._delete_site,
                                        state=tk.DISABLED)
        self._site_del_btn.pack(side=tk.LEFT, padx=2)

        lf_grid = ttk.Frame(lf)
        lf_grid.pack(fill=tk.X)
        self._launch_lat = _dd_row(lf_grid, "Latitude:",  row=0, default="0.0")
        self._launch_lon = _dd_row(lf_grid, "Longitude:", row=1, default="0.0")

        ttk.Label(lf_grid, text="Azimuth:").grid(row=2, column=0,
                                                  sticky=tk.W, padx=(8, 2), pady=2)
        az_frame = ttk.Frame(lf_grid)
        az_frame.grid(row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._azimuth_var = tk.StringVar(value="0.0")
        ttk.Entry(az_frame, textvariable=self._azimuth_var, width=8).pack(side=tk.LEFT)
        ttk.Label(az_frame, text="°  (from N)").pack(side=tk.LEFT, padx=2)

        # ── Guidance — mode radio + loft angle / pitch-over ───────────
        gf = ttk.LabelFrame(parent, text="Guidance")
        gf.pack(fill=tk.X, padx=6, pady=3)
        self._guidance_frame = gf          # saved for dynamic grid management
        gf.columnconfigure(1, weight=1)    # column 1 fills available width

        self._guidance_var = tk.StringVar(value="gravity_turn")
        gmode_frame = ttk.Frame(gf)
        gmode_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W,
                         padx=6, pady=(4, 2))
        ttk.Radiobutton(gmode_frame, text="Gravity Turn",
                        variable=self._guidance_var, value="gravity_turn",
                        command=self._on_guidance_changed).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(gmode_frame, text="Orbital Insertion",
                        variable=self._guidance_var, value="orbital_insertion",
                        command=self._on_guidance_changed).pack(side=tk.LEFT, padx=4)

        ttk.Label(gf, text="Launch elev.:").grid(
            row=1, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        le_frame = ttk.Frame(gf)
        le_frame.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._launch_el_var = tk.StringVar(value="90.0")
        ttk.Entry(le_frame, textvariable=self._launch_el_var, width=8).pack(side=tk.LEFT)
        ttk.Label(le_frame, text="°  (90 = vertical)").pack(side=tk.LEFT, padx=2)

        self._loft_angle_lbl = ttk.Label(gf, text="Loft Angle:")
        self._loft_angle_lbl.grid(row=3, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        la_frame = ttk.Frame(gf)
        la_frame.grid(row=3, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._loft_angle_frame = la_frame
        self._loft_angle_var = tk.StringVar(value="45.0")
        ttk.Entry(la_frame, textvariable=self._loft_angle_var, width=8).pack(side=tk.LEFT)
        self._loft_angle_unit_lbl = ttk.Label(la_frame, text="°  (final elev.)")
        self._loft_angle_unit_lbl.pack(side=tk.LEFT, padx=2)

        self._gt_turn_start_lbl = ttk.Label(gf, text="Turn Start:")
        self._gt_turn_start_lbl.grid(row=5, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        gt_ts_frame = ttk.Frame(gf)
        gt_ts_frame.grid(row=5, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._gt_turn_start_frame = gt_ts_frame
        self._gt_turn_start_var = tk.StringVar(value="5.0")
        ttk.Entry(gt_ts_frame, textvariable=self._gt_turn_start_var, width=8).pack(side=tk.LEFT)
        ttk.Label(gt_ts_frame, text="s").pack(side=tk.LEFT, padx=2)

        self._gt_turn_stop_lbl = ttk.Label(gf, text="Turn Stop:")
        self._gt_turn_stop_lbl.grid(row=6, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        gt_te_frame = ttk.Frame(gf)
        gt_te_frame.grid(row=6, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._gt_turn_stop_frame = gt_te_frame
        self._gt_turn_stop_var = tk.StringVar(value="")
        ttk.Entry(gt_te_frame, textvariable=self._gt_turn_stop_var, width=8).pack(side=tk.LEFT)
        ttk.Label(gt_te_frame, text="s  (blank = full burn)").pack(side=tk.LEFT, padx=2)

        self._orbit_alt_lbl = ttk.Label(gf, text="Target orbit alt:")
        self._orbit_alt_lbl.grid(row=7, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        orb_frame = ttk.Frame(gf)
        orb_frame.grid(row=7, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._orbit_alt_frame = orb_frame
        self._orbit_alt_var = tk.StringVar(value="400")
        ttk.Entry(orb_frame, textvariable=self._orbit_alt_var, width=8).pack(side=tk.LEFT)
        ttk.Label(orb_frame, text="km").pack(side=tk.LEFT, padx=2)

        # Row 8: Plan Orbit button — placed directly in gf (no wrapper Frame)
        # so grid_forget/grid reliably shows and hides it.
        self._plan_orbit_btn = ttk.Button(gf, text="Plan Orbit",
                                          command=self._plan_orbit)
        self._plan_orbit_btn.grid(row=8, column=0, columnspan=2,
                                  sticky=tk.EW, padx=8, pady=(4, 6), ipadx=2, ipady=4)

        # Row 9: Advanced pitch program toggle (gravity_turn / orbital_insertion only)
        self._adv_pitch_var = tk.BooleanVar(value=False)
        self._adv_pitch_chk = ttk.Checkbutton(
            gf, text="Advanced pitch program (per-stage)",
            variable=self._adv_pitch_var,
            command=self._on_adv_pitch_toggled)
        self._adv_pitch_chk.grid(row=9, column=0, columnspan=2,
                                  sticky=tk.W, padx=8, pady=(0, 2))

        # Row 10: Per-stage inline rows — rebuilt whenever missile changes
        self._adv_pitch_frame = ttk.Frame(gf)
        self._stage_rows = []   # list of dicts with StringVars per stage

        # Row 11: Yaw / dogleg program toggle (gravity_turn / orbital_insertion only)
        self._adv_yaw_var = tk.BooleanVar(value=False)
        self._adv_yaw_chk = ttk.Checkbutton(
            gf, text="Yaw / dogleg program",
            variable=self._adv_yaw_var,
            command=self._on_adv_yaw_toggled)

        # Row 10: Global yaw fields — shown when checkbox enabled
        # Three maneuvers laid out as a grid: rows=field, cols=maneuver
        yf = ttk.Frame(gf)
        self._adv_yaw_frame = yf
        self._yaw_vars = [
            {'start': tk.StringVar(value=""),
             'stop':  tk.StringVar(value=""),
             'final_az': tk.StringVar(value="")}
            for _ in range(3)
        ]
        for _mc, _hdr in enumerate(["#1", "#2", "#3"], start=1):
            ttk.Label(yf, text=_hdr, foreground="#555555").grid(
                row=0, column=_mc, padx=4, pady=(4, 1))
        for _yr, _lbl, _key, _unit in [
                (1, "Yaw start:", "start",    "s"),
                (2, "Yaw end:",   "stop",     "s"),
                (3, "Final az:",  "final_az", "°")]:
            ttk.Label(yf, text=_lbl).grid(
                row=_yr, column=0, sticky=tk.W, padx=(8, 2), pady=1)
            for _mc, _yvars in enumerate(self._yaw_vars, start=1):
                ttk.Entry(yf, textvariable=_yvars[_key], width=6).grid(
                    row=_yr, column=_mc, padx=3, pady=1)
            ttk.Label(yf, text=_unit).grid(
                row=_yr, column=4, sticky=tk.W, padx=(2, 8), pady=1)

        # Row 11: Reset trajectory button — always visible
        self._reset_traj_btn = ttk.Button(
            gf, text="Reset trajectory to defaults",
            command=self._reset_traj_profile)
        self._reset_traj_btn.grid(row=11, column=0, columnspan=2,
                                  sticky=tk.EW, padx=8, pady=(4, 2))

        # Row 12: Export / Import guidance program
        _gp_frame = ttk.Frame(gf)
        _gp_frame.grid(row=12, column=0, columnspan=2,
                       sticky=tk.EW, padx=8, pady=(0, 6))
        _gp_frame.columnconfigure(0, weight=1)
        _gp_frame.columnconfigure(1, weight=1)
        ttk.Button(_gp_frame, text="Export guidance…",
                   command=self._export_guidance).grid(
            row=0, column=0, sticky=tk.EW, padx=(0, 2))
        ttk.Button(_gp_frame, text="Import guidance…",
                   command=self._import_guidance).grid(
            row=0, column=1, sticky=tk.EW, padx=(2, 0))

        # Initialise guidance-specific row visibility for the default mode.
        self._orbit_alt_lbl.grid_forget()
        self._orbit_alt_frame.grid_forget()
        self._plan_orbit_btn.grid_forget()
        self._adv_pitch_chk.grid_forget()
        self._adv_yaw_chk.grid_forget()
        self._update_guidance_labels("gravity_turn")

        # ── Engine cutoff ─────────────────────────────────────────────
        cf = ttk.LabelFrame(parent, text="Engine Cutoff")
        cf.pack(fill=tk.X, padx=6, pady=3)
        cf_inner = ttk.Frame(cf)
        cf_inner.pack(padx=6, pady=4)
        ttk.Label(cf_inner, text="Cutoff time:").pack(side=tk.LEFT)
        self._cutoff_var = tk.StringVar(value="")
        ttk.Entry(cf_inner, textvariable=self._cutoff_var, width=8).pack(
            side=tk.LEFT, padx=4)
        ttk.Label(cf_inner, text="s  (blank = full burn)").pack(side=tk.LEFT)

        # ── Re-entry query altitude ────────────────────────────────────
        rq = ttk.LabelFrame(parent, text="Re-entry Query")
        rq.pack(fill=tk.X, padx=6, pady=3)
        rq_inner = ttk.Frame(rq)
        rq_inner.pack(padx=6, pady=4)
        self._query_alt_enable = tk.BooleanVar(value=False)
        self._query_alt_km_var = tk.StringVar(value="50")
        ttk.Checkbutton(rq_inner, text="Report state at:",
                        variable=self._query_alt_enable,
                        command=self._toggle_query_alt).pack(side=tk.LEFT)
        self._query_alt_entry = ttk.Entry(
            rq_inner, textvariable=self._query_alt_km_var, width=6, state="disabled")
        self._query_alt_entry.pack(side=tk.LEFT, padx=4)
        ttk.Label(rq_inner, text="km (descent)").pack(side=tk.LEFT)

        # ── Run buttons ───────────────────────────────────────────────
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=6, pady=6)
        ttk.Button(btn_frame, text="Run Flyout",
                   command=self._run_flyout).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 2), ipady=4)
        ttk.Button(btn_frame, text="Max Range",
                   command=self._maximize_range).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=2, ipady=4)
        ttk.Button(btn_frame, text="Sweep…",
                   command=self._open_sweep).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(2, 0), ipady=4)

        # (Missile Parameters moved to right-panel notebook tab)

    # ------------------------------------------------------------------
    # SLV Performance tab  (Schilling / Townsend algebraic analysis)
    # ------------------------------------------------------------------
    def _build_slv_tab(self, parent):
        # ── Target orbit input ────────────────────────────────────────
        of = ttk.LabelFrame(parent, text="Target Orbit")
        of.pack(fill=tk.X, padx=8, pady=(8, 4))

        of_grid = ttk.Frame(of)
        of_grid.pack(padx=8, pady=6, anchor=tk.W)

        ttk.Label(of_grid, text="Perigee:").grid(
            row=0, column=0, sticky=tk.E, padx=(0, 4))
        self._slv_alt_var = tk.StringVar(value="500")
        ttk.Entry(of_grid, textvariable=self._slv_alt_var,
                  width=8).grid(row=0, column=1)
        ttk.Label(of_grid, text="km").grid(
            row=0, column=2, sticky=tk.W, padx=(4, 0))

        ttk.Label(of_grid, text="Apogee:").grid(
            row=1, column=0, sticky=tk.E, padx=(0, 4), pady=(4, 0))
        self._slv_apo_var = tk.StringVar(value="")
        ttk.Entry(of_grid, textvariable=self._slv_apo_var,
                  width=8).grid(row=1, column=1, pady=(4, 0))
        ttk.Label(of_grid, text="km  (blank = circular)").grid(
            row=1, column=2, sticky=tk.W, padx=(4, 0), pady=(4, 0))

        ttk.Button(of, text="Analyze SLV Performance",
                   command=self._run_slv_analysis).pack(pady=(0, 6))

        # ── Results ───────────────────────────────────────────────────
        rf = ttk.LabelFrame(parent, text="Results  (Schilling / Townsend method)")
        rf.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self._slv_text = tk.Text(
            rf, state=tk.DISABLED, font=("TkFixedFont", 9),
            wrap=tk.NONE, relief=tk.FLAT, background="#f8f8f8",
            foreground="#222222", selectbackground="#c0d8f0")
        _fam = tkfont.nametofont("TkFixedFont").actual()["family"]
        _tab = tkfont.Font(family=_fam, size=9).measure("x" * 32)
        self._slv_text.configure(tabs=(_tab,))
        vsb = ttk.Scrollbar(rf, orient=tk.VERTICAL,
                            command=self._slv_text.yview)
        self._slv_text.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._slv_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Logo — overlaid on the results panel, bottom-left
        import os as _os
        _logo_path = _os.path.join(_os.path.dirname(__file__), "data", "Thrusty3.png")
        try:
            from PIL import Image, ImageTk as _ITk
            _img = Image.open(_logo_path)
            _h = 195
            _w = int(_img.width * _h / _img.height)
            _img = _img.resize((_w, _h), Image.LANCZOS)
            self._slv_logo_photo = _ITk.PhotoImage(_img)
            tk.Label(self._slv_text, image=self._slv_logo_photo,
                     borderwidth=0, highlightthickness=0, bg="#f8f8f8"
                     ).place(relx=1.0, rely=1.0, anchor="se", x=-6, y=-4)
        except Exception:
            pass

        # Tag for the headline verdict line
        self._slv_text.tag_configure("yes", foreground="#006600",
                                     font=("TkFixedFont", 9, "bold"))
        self._slv_text.tag_configure("no",  foreground="#aa0000",
                                     font=("TkFixedFont", 9, "bold"))
        self._slv_text.tag_configure("hdr", font=("TkFixedFont", 9, "bold"))

        self._slv_set_text(
            "Select a missile, set the launch site and azimuth in the left\n"
            "panel, enter a target orbit above, then click Analyze.\n\n"
            "For a circular orbit enter only the perigee altitude (apogee\n"
            "blank or equal to perigee).  For a Hohmann transfer or GTO set\n"
            "apogee higher than perigee; the rocket burns to the perigee\n"
            "injection speed and coasts to apogee.\n\n"
            "The launch azimuth determines the orbital inclination and the\n"
            "Earth-rotation benefit (maximum for a due-east launch).\n\n"
            "Accuracy: ~260 m/s RMS in total mission ΔV; typically < 10 %\n"
            "error in payload capacity  (Schilling 2009).",
            verdict=None)

    def _slv_set_text(self, body: str, verdict=None):
        """Replace the SLV results text widget contents."""
        self._slv_text.configure(state=tk.NORMAL)
        self._slv_text.delete("1.0", tk.END)
        if verdict is not None:
            tag = "yes" if verdict else "no"
            mark = "✓  CAN reach orbit" if verdict else "✗  CANNOT reach orbit"
            self._slv_text.insert(tk.END, mark + "\n", tag)
            self._slv_text.insert(tk.END, "\n")
        self._slv_text.insert(tk.END, body)
        self._slv_text.configure(state=tk.DISABLED)

    def _run_slv_analysis(self):
        # ── Parse inputs ──────────────────────────────────────────────
        try:
            perigee_km = float(self._slv_alt_var.get())
        except ValueError:
            messagebox.showerror("Input error",
                                 "Perigee altitude must be a number (km).",
                                 parent=self)
            return
        if perigee_km <= 0:
            messagebox.showerror("Input error",
                                 "Perigee altitude must be positive.", parent=self)
            return

        apo_str = self._slv_apo_var.get().strip()
        if apo_str == "" or apo_str == str(perigee_km):
            apogee_km = None           # circular
        else:
            try:
                apogee_km = float(apo_str)
            except ValueError:
                messagebox.showerror("Input error",
                                     "Apogee altitude must be a number or blank.",
                                     parent=self)
                return
            if apogee_km < perigee_km:
                messagebox.showerror("Input error",
                                     "Apogee must be ≥ perigee.", parent=self)
                return

        try:
            lat = float(self._launch_lat.get())
            az  = float(self._azimuth_var.get())
        except ValueError:
            messagebox.showerror("Input error",
                                 "Check launch latitude and azimuth.", parent=self)
            return

        missile = get_missile(self._missile_var.get())

        try:
            r = schilling_performance(missile, perigee_km, lat, az,
                                      target_apogee_km=apogee_km)
        except Exception as exc:
            messagebox.showerror("Analysis error", str(exc), parent=self)
            return

        # ── Format results ────────────────────────────────────────────
        from slv_performance import stage_delta_v as _sdv

        n_stages = 0
        s = missile
        while s:
            n_stages += 1
            s = s.stage2

        stage_lines = []
        s, i = missile, 1
        while s:
            stage_lines.append(
                f"    Stage {i} ({s.isp_s:.0f} s Isp):\t{_sdv(s):8.0f} m/s")
            s = s.stage2
            i += 1

        is_circular = (apogee_km is None or
                       apogee_km == perigee_km)
        if is_circular:
            orbit_desc = f"{perigee_km:.0f} km circular"
            inj_label  = "Circular orbit speed:"
        else:
            orbit_desc = (f"{perigee_km:.0f} × {r['orbit_apogee_km']:.0f} km  "
                          f"(e = {r['orbit_eccentricity']:.4f})")
            inj_label  = "Injection speed (perigee):"

        margin_sign = "+" if r['dv_margin_ms'] >= 0 else ""

        payload_line = ""
        if missile.payload_kg > 0:
            pm   = r['payload_margin_kg'] or 0.0
            sign = "+" if pm >= 0 else ""
            payload_line = (
                f"  Claimed payload:\t{missile.payload_kg:8.0f} kg\n"
                f"  Payload margin:\t{sign}{pm:7.0f} kg\n"
            )

        body = (
            f"Vehicle:  {missile.name}  ({n_stages}-stage)\n"
            f"Target:   {orbit_desc}\n"
            f"Launch:   {lat:.2f}° lat,  azimuth {az:.1f}°\n"
            "\n"
            "─── Delta-V Budget ─────────────────────────────────────────\n"
            "  Available ΔV (rocket eq.):\n"
            + "\n".join(stage_lines) + "\n"
            f"  Total available:\t{r['dv_available_ms']:8.0f} m/s\n"
            "\n"
            "  Required to reach orbit:\n"
            f"    {inj_label}\t{r['v_injection_ms']:8.0f} m/s\n"
            f"    Loss penalty (eq. 5):\t{r['dv_penalty_ms']:8.0f} m/s\n"
            f"    Earth rotation benefit:\t{-r['v_rotation_ms']:8.0f} m/s\n"
            f"  Total required:\t{r['dv_required_ms']:8.0f} m/s\n"
            "\n"
            f"  Margin:\t{margin_sign}{r['dv_margin_ms']:7.0f} m/s\n"
            "\n"
            "─── Payload Capacity ───────────────────────────────────────\n"
            f"  Maximum payload:\t{r['max_payload_kg']:8.0f} kg\n"
            + payload_line +
            "\n"
            "─── Schilling Timing Parameters ────────────────────────────\n"
            f"  Actual burn time  (Tₐ):\t{r['t_actual_s']:8.1f} s\n"
            f"  3-stage equiv.    (T₃ₛ):\t{r['t_3stage_s']:8.1f} s\n"
            f"  Blended time    (T_mix):\t{r['t_mix_s']:8.1f} s\n"
            f"  Initial accel.    (A₀):\t{r['a0_ms2']:8.2f} m/s²"
            f"  ({r['a0_ms2'] / 9.80665:.2f} g)\n"
            "\n"
            "Method accuracy: ±260 m/s RMS in mission ΔV; < 10 % in payload.\n"
            "Ref: Schilling (2009), Townsend / Martin-Marietta (1962)."
        )

        self._slv_set_text(body, verdict=r['can_reach_orbit'])
        self._right_nb.select(3)

    # ------------------------------------------------------------------
    # Plot panel  (6-subplot grid; slot [2,1] reserved for future use)
    # ------------------------------------------------------------------
    def _build_plot_panel(self, parent):
        self._fig = Figure(figsize=(8, 8.5), dpi=96)
        gs = self._fig.add_gridspec(3, 2, hspace=0.52, wspace=0.38,
                                    left=0.10, right=0.95,
                                    top=0.95, bottom=0.06)
        self._ax_alt  = self._fig.add_subplot(gs[0, 0])  # alt vs time
        self._ax_spd  = self._fig.add_subplot(gs[0, 1])  # speed vs time
        self._ax_traj = self._fig.add_subplot(gs[1, 0])  # alt vs range
        self._ax_trk  = self._fig.add_subplot(gs[1, 1])  # ground track
        self._ax_guid      = self._fig.add_subplot(gs[2, 0])  # pitch / azimuth
        self._ax_guid_twin = self._ax_guid.twinx()            # azimuth axis (created once)
        self._ax_qmach      = self._fig.add_subplot(gs[2, 1]) # q + Mach (burn period)
        self._ax_qmach_twin = self._ax_qmach.twinx()          # Mach axis

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self._canvas, parent)
        toolbar.update()

        # Initialise axes with placeholder labels
        self._init_axes()
        self._canvas.draw()

    def _init_axes(self):
        for ax, title, xl, yl in [
            (self._ax_alt,   "Altitude vs Time",       "Time (s)",       "Altitude (km)"),
            (self._ax_spd,   "Speed vs Time",           "Time (s)",       "Speed (km/s)"),
            (self._ax_traj,  "Altitude vs Range",       "Downrange (km)", "Altitude (km)"),
            (self._ax_trk,   "Ground Track",            "Longitude (°E)", "Latitude (°N)"),
            (self._ax_guid,  "Pitch, Azimuth vs. Time",          "Time (s)", "Elevation (°)"),
            (self._ax_qmach, "Dyn. Pressure, Mach vs. Time",     "Time (s)", "q  (kPa)"),
        ]:
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=7)
        self._ax_guid_twin.set_ylabel('Azimuth (°)', fontsize=7, color='darkorange')
        self._ax_guid_twin.tick_params(labelsize=7, colors='darkorange')
        self._ax_qmach_twin.set_ylabel('Mach', fontsize=7, color='darkorange')
        self._ax_qmach_twin.tick_params(labelsize=7, colors='darkorange')

    # ------------------------------------------------------------------
    # Flight Timeline panel
    # ------------------------------------------------------------------
    _TL_COLS = [
        ("event",       "Event",             180, tk.W),
        ("t_s",         "Time (s)",           72, tk.E),
        ("alt_km",      "Alt (km)",            72, tk.E),
        ("range_km",    "Range (km)",          80, tk.E),
        ("gnd_speed",   "Gnd Spd (km/s)",      90, tk.E),
        ("inrtl_speed", "Inrtl Spd (km/s)",    95, tk.E),
        ("accel",       "Accel (m/s²)",         80, tk.E),
        ("mass",        "Mass (t)",             72, tk.E),
    ]

    def _build_timeline_panel(self, parent):
        # Summary block (mirrors left-panel results, visible without switching back)
        sf = ttk.LabelFrame(parent, text="Summary")
        sf.pack(fill=tk.X, padx=6, pady=(6, 2))
        self._tl_summary_var = tk.StringVar(
            value="Run a simulation to populate the flight timeline.")
        ttk.Label(sf, textvariable=self._tl_summary_var,
                  justify=tk.LEFT, anchor=tk.W).pack(
            fill=tk.X, padx=8, pady=4)

        # Timeline table
        tf = ttk.LabelFrame(parent, text="Flight Event Timeline")
        tf.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        col_ids = [c[0] for c in self._TL_COLS]
        self._tl_tree = ttk.Treeview(tf, columns=col_ids, show="headings",
                                     height=14)
        for col_id, heading, width, anchor in self._TL_COLS:
            self._tl_tree.heading(col_id, text=heading)
            self._tl_tree.column(col_id, width=width, anchor=anchor,
                                 stretch=(col_id == "event"))

        vsb = ttk.Scrollbar(tf, orient=tk.VERTICAL,
                            command=self._tl_tree.yview)
        self._tl_tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tl_tree.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Alternating row colours; no explicit font — inherits system default
        self._tl_tree.tag_configure("odd",    background="#f5f5f5")
        self._tl_tree.tag_configure("even",   background="#ffffff")
        self._tl_tree.tag_configure("key",    background="#ddeeff", font="bold")
        self._tl_tree.tag_configure("debris", background="#fff3cd")

        # Logo — overlaid on the treeview using place() so it sits inside
        # the white area, bottom-left, regardless of platform theme.
        import os as _os
        _logo_path = _os.path.join(_os.path.dirname(__file__), "data", "Thrusty3.png")
        try:
            from PIL import Image, ImageTk as _ITk
            _img = Image.open(_logo_path)
            _h = 195
            _w = int(_img.width * _h / _img.height)
            _img = _img.resize((_w, _h), Image.LANCZOS)
            self._tl_logo_photo = _ITk.PhotoImage(_img)
            tk.Label(self._tl_tree, image=self._tl_logo_photo,
                     borderwidth=0, highlightthickness=0, bg="white"
                     ).place(relx=0.0, rely=1.0, anchor="sw", x=6, y=-4)
        except Exception:
            pass   # logo absent or Pillow unavailable — silent skip

    # ------------------------------------------------------------------
    # Missile Parameters tab
    # ------------------------------------------------------------------
    def _build_params_tab(self, parent):
        """Scrollable structured display — rebuilt on each missile change."""
        self._params_canvas = tk.Canvas(
            parent, borderwidth=0, highlightthickness=0)
        vsb = ttk.Scrollbar(parent, orient="vertical",
                            command=self._params_canvas.yview)
        self._params_canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._params_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._params_inner = ttk.Frame(self._params_canvas)
        self._params_win_id = self._params_canvas.create_window(
            (0, 0), window=self._params_inner, anchor="nw")

        self._params_inner.bind(
            "<Configure>",
            lambda _e: self._params_canvas.configure(
                scrollregion=self._params_canvas.bbox("all")))
        self._params_canvas.bind(
            "<Configure>",
            lambda e: self._params_canvas.itemconfig(
                self._params_win_id, width=e.width))

        def _mw(event):
            self._params_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units")

        self._params_canvas.bind("<Enter>",
            lambda _e: self._params_canvas.bind_all("<MouseWheel>", _mw))
        self._params_canvas.bind("<Leave>",
            lambda _e: self._params_canvas.unbind_all("<MouseWheel>"))

    # ------------------------------------------------------------------
    # Missile selection
    # ------------------------------------------------------------------
    def _on_missile_changed(self, _event=None):
        name = self._missile_var.get()
        if name not in MISSILE_DB:
            return
        self._last_valid_missile = name
        p = get_missile(name)

        prof = _load_traj_profiles().get(name)
        if prof:
            self._guidance_var.set(prof.get('guidance', p.guidance))
            self._loft_angle_var.set(str(prof.get('burnout_angle_deg', p.burnout_angle_deg)))
            self._launch_el_var.set(str(prof.get('launch_elevation_deg', p.launch_elevation_deg)))
            gt_start = prof.get('gt_turn_start_s', 5.0)
            self._gt_turn_start_var.set(str(gt_start) if gt_start else "5.0")
            gt_stop = prof.get('gt_turn_stop_s')
            self._gt_turn_stop_var.set(str(gt_stop) if gt_stop is not None else "")
            cutoff = prof.get('cutoff_time_s')
            self._cutoff_var.set(str(int(cutoff)) if cutoff is not None
                                 else str(int(total_burn_time(p))))
        else:
            self._cutoff_var.set(str(int(total_burn_time(p))))
            self._loft_angle_var.set(f"{p.burnout_angle_deg:.4f}")
            self._launch_el_var.set(f"{p.launch_elevation_deg:.1f}")
            self._guidance_var.set(p.guidance)

        self._update_guidance_labels(self._guidance_var.get())
        self._update_params_display(p)
        self._del_btn.config(state=tk.NORMAL)
        if self._adv_pitch_var.get():
            self._rebuild_stage_rows()

    # ------------------------------------------------------------------
    # Advanced per-stage pitch program
    # ------------------------------------------------------------------
    def _on_adv_pitch_toggled(self):
        """Show or hide the per-stage pitch rows."""
        if self._adv_pitch_var.get():
            self._rebuild_stage_rows()
            self._adv_pitch_frame.grid(row=8, column=0, columnspan=2,
                                        sticky=tk.EW, padx=0, pady=(0, 4))
        else:
            self._adv_pitch_frame.grid_forget()

    def _on_adv_yaw_toggled(self):
        """Show or hide the global yaw fields."""
        if self._adv_yaw_var.get():
            self._adv_yaw_frame.grid(row=10, column=0, columnspan=2,
                                      sticky=tk.EW, padx=0, pady=(0, 4))
        else:
            self._adv_yaw_frame.grid_forget()

    def _rebuild_stage_rows(self):
        """Rebuild inline per-stage pitch rows from the current missile."""
        for w in self._adv_pitch_frame.winfo_children():
            w.destroy()
        self._stage_rows = []

        p = get_missile(self._missile_var.get())
        if p is None:
            return

        # Walk stage chain, record absolute ignition / burnout times
        stages, node, t_ign = [], p, 0.0
        while node is not None:
            t_burn = t_ign + node.burn_time_s
            stages.append({'node': node, 't_ign': t_ign, 't_burn': t_burn})
            t_ign = t_burn + node.coast_time_s
            node = node.stage2

        # Defaults from simple-mode fields
        try:
            g_angle = float(self._loft_angle_var.get())
        except ValueError:
            g_angle = 45.0
        try:
            g_start = float(self._gt_turn_start_var.get())
        except ValueError:
            g_start = 5.0
        try:
            g_stop = float(self._gt_turn_stop_var.get())
        except (ValueError, AttributeError):
            g_stop = None

        _gui_guidance = self._guidance_var.get()
        n_stg = len(stages)

        # Column header
        af = self._adv_pitch_frame
        af.columnconfigure(0, minsize=55)
        _headers = [
            (0, "Stage"),
            (1, "Turn start (s)"),
            (2, "Turn stop (s)"),
            (3, "Angle (°)"),
            (4, "Coast (s)"),
            (5, "Burn window"),
        ]
        for col, hdr in _headers:
            ttk.Label(af, text=hdr, foreground="#555555").grid(
                row=0, column=col, padx=(8 if col == 0 else 4, 4),
                pady=(4, 1), sticky=tk.W)

        for i, s in enumerate(stages):
            node = s['node']
            t_i, t_b = s['t_ign'], s['t_burn']
            is_last = (i == n_stg - 1)

            # Seed per-stage values: use stored overrides if present,
            # otherwise derive sensible defaults from simple-mode params.
            if node.stage_turn_start_s is not None:
                def_start = node.stage_turn_start_s
            elif _gui_guidance == "orbital_insertion":
                # All stages share a single global turn-start so the
                # continuous two-phase pitch is reproduced exactly.
                def_start = g_start
            else:
                def_start = g_start if i == 0 else t_i

            if node.stage_turn_stop_s is not None:
                def_stop = node.stage_turn_stop_s
            elif is_last and _gui_guidance == "orbital_insertion":
                # Final stage burns horizontally — stop pitch just before ignition.
                def_stop = max(0.0, t_i - 1.0)
            elif _gui_guidance == "orbital_insertion" and g_stop is not None:
                # Pre-final orbital stages use Plan Orbit's global turn_stop.
                def_stop = g_stop
            else:
                def_stop = max(t_i, t_b - 5.0)

            if node.stage_burnout_angle_deg is not None:
                def_angle = node.stage_burnout_angle_deg
            elif is_last and _gui_guidance == "orbital_insertion":
                def_angle = 0.0   # final stage burns horizontally
            else:
                def_angle = g_angle

            sv_start = tk.StringVar(value=f"{def_start:.1f}")
            sv_stop  = tk.StringVar(value=f"{def_stop:.1f}")
            sv_angle = tk.StringVar(value=f"{def_angle:.1f}")

            # Coast — pre-populate from missile definition; blank for last stage
            sv_coast = tk.StringVar(
                value=f"{node.coast_time_s:.1f}" if not is_last else "")

            row = i + 1
            ttk.Label(af, text=f"Stage {i+1}:").grid(
                row=row, column=0, sticky=tk.W, padx=(8, 4), pady=1)
            ttk.Entry(af, textvariable=sv_start, width=5).grid(
                row=row, column=1, padx=3, pady=1)
            ttk.Entry(af, textvariable=sv_stop,  width=5).grid(
                row=row, column=2, padx=3, pady=1)
            ttk.Entry(af, textvariable=sv_angle, width=5).grid(
                row=row, column=3, padx=3, pady=1)
            coast_e = ttk.Entry(af, textvariable=sv_coast, width=5)
            coast_e.grid(row=row, column=4, padx=3, pady=1)
            if is_last:
                coast_e.config(state="disabled")
            ttk.Label(af, text=f"({t_i:.0f}–{t_b:.0f} s)",
                      foreground="#888888").grid(
                row=row, column=5, sticky=tk.W, padx=(4, 8), pady=1)

            self._stage_rows.append(
                {'start': sv_start, 'stop': sv_stop, 'angle': sv_angle,
                 'coast': sv_coast, 'node': node})

    # ------------------------------------------------------------------
    def _on_site_selected(self, _event=None):
        name = self._site_var.get()
        site = self._site_map.get(name)
        if site is None:          # country-header row clicked — revert
            self._site_var.set("")
            return
        self._launch_lat.set(f"{site['lat']:.4f}")
        self._launch_lon.set(f"{site['lon']:.4f}")
        is_user = name not in _BUNDLED_SITE_NAMES
        self._site_del_btn.config(state=tk.NORMAL if is_user else tk.DISABLED)

    def _new_site(self):
        """Clear the site selector and lat/lon fields for fresh entry."""
        self._site_var.set("")
        self._launch_lat.set("")
        self._launch_lon.set("")
        self._site_del_btn.config(state=tk.DISABLED)

    def _edit_site(self):
        """Save current lat/lon as a named user site."""
        lat_str = self._launch_lat.get().strip()
        lon_str = self._launch_lon.get().strip()
        try:
            lat = float(lat_str)
            lon = float(lon_str)
        except ValueError:
            messagebox.showerror("Invalid coordinates",
                                 "Enter valid lat/lon before saving.", parent=self)
            return

        dlg = tk.Toplevel(self)
        dlg.title("Save Site")
        dlg.resizable(False, False)
        dlg.grab_set()
        ttk.Label(dlg, text="Name:").grid(   row=0, column=0, sticky=tk.W, padx=(10,4), pady=(10,2))
        ttk.Label(dlg, text="Country:").grid(row=1, column=0, sticky=tk.W, padx=(10,4), pady=2)
        name_var    = tk.StringVar(value=self._site_var.get()
                                   if self._site_var.get() in self._site_map else "")
        country_var = tk.StringVar(value=self._site_map.get(
                                   self._site_var.get(), {}).get("country", ""))
        ttk.Entry(dlg, textvariable=name_var,    width=28).grid(row=0, column=1, padx=(0,10), pady=(10,2))
        ttk.Entry(dlg, textvariable=country_var, width=28).grid(row=1, column=1, padx=(0,10), pady=2)

        def _do_save():
            name    = name_var.get().strip()
            country = country_var.get().strip()
            if not name or not country:
                messagebox.showerror("Missing fields",
                                     "Name and country are required.", parent=dlg)
                return
            user_sites = _load_user_sites()
            # Update in place if name already exists in user list
            for s in user_sites:
                if s["name"] == name:
                    s.update({"country": country, "lat": lat, "lon": lon})
                    break
            else:
                user_sites.append({"name": name, "country": country,
                                   "lat": lat, "lon": lon})
            _save_user_sites(user_sites)
            self._site_map, cb_values = {}, []
            new_values, new_map = _load_launch_sites()
            self._site_map = new_map
            self._site_cb.config(values=new_values)
            self._site_var.set(name)
            is_user = name not in _BUNDLED_SITE_NAMES
            self._site_del_btn.config(state=tk.NORMAL if is_user else tk.DISABLED)
            self._status_var.set(f"Site '{name}' saved.")
            dlg.destroy()

        bf = ttk.Frame(dlg)
        bf.grid(row=2, column=0, columnspan=2, pady=(6, 10))
        ttk.Button(bf, text="Save",   command=_do_save).pack(side=tk.LEFT, padx=6)
        ttk.Button(bf, text="Cancel", command=dlg.destroy).pack(side=tk.LEFT, padx=6)
        dlg.bind("<Return>", lambda _e: _do_save())

    def _delete_site(self):
        name = self._site_var.get()
        if name in _BUNDLED_SITE_NAMES:
            return
        if not messagebox.askyesno("Delete site",
                                   f"Permanently delete '{name}'?", parent=self):
            return
        user_sites = [s for s in _load_user_sites() if s["name"] != name]
        _save_user_sites(user_sites)
        new_values, new_map = _load_launch_sites()
        self._site_map = new_map
        self._site_cb.config(values=new_values)
        self._site_var.set("")
        self._site_del_btn.config(state=tk.DISABLED)
        self._status_var.set(f"Site '{name}' deleted.")

    def _toggle_query_alt(self):
        state = "normal" if self._query_alt_enable.get() else "disabled"
        self._query_alt_entry.config(state=state)

    def _on_guidance_changed(self):
        """Called when the user clicks a guidance radio button."""
        self._update_guidance_labels(self._guidance_var.get())

    # ------------------------------------------------------------------
    def _update_guidance_labels(self, guidance: str):
        """Relabel the main-panel guidance fields to match the active mode."""
        if guidance in ("gravity_turn", "orbital_insertion"):
            self._loft_angle_lbl.config(text="Burnout Angle:")
            self._loft_angle_unit_lbl.config(text="°  (Wheelon ε*)")
            self._loft_angle_lbl.grid(
                row=2, column=0, sticky=tk.W, padx=(8, 2), pady=2)
            self._loft_angle_frame.grid(
                row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2)
            self._gt_turn_start_lbl.grid(
                row=3, column=0, sticky=tk.W, padx=(8, 2), pady=2)
            self._gt_turn_start_frame.grid(
                row=3, column=1, sticky=tk.W, padx=(0, 8), pady=2)
            self._gt_turn_stop_lbl.grid(
                row=4, column=0, sticky=tk.W, padx=(8, 2), pady=2)
            self._gt_turn_stop_frame.grid(
                row=4, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        if guidance == "orbital_insertion":
            self._orbit_alt_lbl.grid(
                row=5, column=0, sticky=tk.W, padx=(8, 2), pady=2)
            self._orbit_alt_frame.grid(
                row=5, column=1, sticky=tk.W, padx=(0, 8), pady=2)
            self._plan_orbit_btn.grid(
                row=6, column=0, columnspan=2,
                sticky=tk.EW, padx=8, pady=(4, 6), ipadx=2, ipady=4)
        else:
            self._orbit_alt_lbl.grid_forget()
            self._orbit_alt_frame.grid_forget()
            self._plan_orbit_btn.grid_forget()

        # Advanced pitch checkbox — only meaningful for turn-based modes
        if guidance in ("gravity_turn", "orbital_insertion"):
            self._adv_pitch_chk.grid(row=7, column=0, columnspan=2,
                                      sticky=tk.W, padx=8, pady=(0, 2))
            if self._adv_pitch_var.get():
                self._adv_pitch_frame.grid(row=8, column=0, columnspan=2,
                                            sticky=tk.EW, padx=0, pady=(0, 4))
            # Yaw checkbox — also only for turn-based modes
            self._adv_yaw_chk.grid(row=9, column=0, columnspan=2,
                                    sticky=tk.W, padx=8, pady=(0, 2))
            if self._adv_yaw_var.get():
                self._adv_yaw_frame.grid(row=10, column=0, columnspan=2,
                                          sticky=tk.EW, padx=0, pady=(0, 4))
        else:
            self._adv_pitch_chk.grid_forget()
            self._adv_pitch_frame.grid_forget()
            self._adv_yaw_chk.grid_forget()
            self._adv_yaw_frame.grid_forget()
            self._adv_yaw_var.set(False)

    # ------------------------------------------------------------------
    # Custom missile management
    # ------------------------------------------------------------------
    def _refresh_missile_list(self, select_name=None):
        """Rebuild the combobox values from the current MISSILE_DB."""
        names = list(MISSILE_DB.keys())
        self._missile_cb.configure(values=names)
        target = select_name or self._missile_var.get()
        if target not in MISSILE_DB:
            target = names[0] if names else ""
        self._missile_var.set(target)
        self._del_btn.config(state=tk.NORMAL if target else tk.DISABLED)
        if target:
            self._on_missile_changed()

    def _on_missile_saved(self, p):
        """Callback invoked by MissileDialog when the user clicks Save."""
        name = p.name
        MISSILE_DB[name] = lambda _p=p: _p
        _save_custom_missiles()
        # Snapshot trajectory panel so saving the missile doesn't reset it.
        self._snapshot_traj_profile(name)
        self._refresh_missile_list(select_name=name)
        self._status_var.set(f"Missile '{name}' saved.")

    def _new_missile(self):
        MissileDialog(self, on_save=self._on_missile_saved)

    def _edit_missile(self):
        name = self._missile_var.get()
        MissileDialog(self, on_save=self._on_missile_saved, existing_name=name)

    def _delete_missile(self):
        name = self._missile_var.get()
        if not name or name not in MISSILE_DB:
            return
        if not messagebox.askyesno("Delete missile",
                                   f"Permanently delete '{name}'?",
                                   parent=self):
            return
        del MISSILE_DB[name]
        _save_custom_missiles()
        profiles = _load_traj_profiles()
        if name in profiles:
            del profiles[name]
            _save_traj_profiles(profiles)
        self._refresh_missile_list()
        self._status_var.set(f"Missile '{name}' deleted.")

    def _snapshot_traj_profile(self, missile_name: str) -> None:
        """Save current trajectory panel fields as a profile for missile_name."""
        cutoff_str   = self._cutoff_var.get().strip()
        gt_start_str = self._gt_turn_start_var.get().strip()
        gt_stop_str  = self._gt_turn_stop_var.get().strip()
        try:
            loft_angle = float(self._loft_angle_var.get())
        except ValueError:
            loft_angle = 45.0
        try:
            launch_el = float(self._launch_el_var.get())
        except (ValueError, AttributeError):
            launch_el = 90.0
        prof = {
            'guidance':             self._guidance_var.get(),
            'burnout_angle_deg':       loft_angle,
            'launch_elevation_deg': launch_el,
            'gt_turn_start_s':       float(gt_start_str) if gt_start_str else 5.0,
            'gt_turn_stop_s':        float(gt_stop_str)  if gt_stop_str  else None,
            'cutoff_time_s':         float(cutoff_str)   if cutoff_str   else None,
        }
        profiles = _load_traj_profiles()
        profiles[missile_name] = prof
        _save_traj_profiles(profiles)

    def _reset_traj_profile(self) -> None:
        """Delete the saved trajectory profile for the current missile and restore defaults."""
        name = self._missile_var.get()
        if not name:
            return
        profiles = _load_traj_profiles()
        if name in profiles:
            del profiles[name]
            _save_traj_profiles(profiles)
        p = get_missile(name)
        self._cutoff_var.set(str(int(total_burn_time(p))))
        self._loft_angle_var.set(f"{p.burnout_angle_deg:.4f}")
        self._launch_el_var.set(f"{p.launch_elevation_deg:.1f}")
        self._guidance_var.set(p.guidance)
        self._gt_turn_start_var.set("5.0")
        self._gt_turn_stop_var.set("")
        self._update_guidance_labels(p.guidance)
        self._status_var.set(f"Trajectory reset to '{name}' defaults.")

    # Guidance-program fields that belong in an export file.
    _GUIDANCE_KEYS = frozenset({
        'guidance', 'burnout_angle_deg', 'launch_elevation_deg',
        'gt_turn_start_s', 'gt_turn_stop_s', 'cutoff_s',
        'adv_pitch', 'stage_overrides', 'adv_yaw', 'yaw_maneuvers',
        'azimuth_deg', 'launch_lat', 'launch_lon',
    })

    def _export_guidance(self):
        """Save the current guidance program to a JSON file."""
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(
            title="Export guidance program",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        meta = self._trajectory_metadata()
        data = {k: v for k, v in meta.items() if k in self._GUIDANCE_KEYS}
        data['_type'] = 'guidance_program'
        try:
            with open(path, 'w') as fh:
                json.dump(data, fh, indent=2)
            self._status_var.set(
                f"Guidance exported: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc), parent=self)

    def _import_guidance(self):
        """Load a guidance program from a JSON file."""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Import guidance program",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            with open(path) as fh:
                data = json.load(fh)
            self._apply_trajectory_metadata(data)
            self._status_var.set(
                f"Guidance imported: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Import error", str(exc), parent=self)

    def _update_params_display(self, p=None):
        """Rebuild the Missile Parameters tab with structured label rows."""
        if p is None:
            p = get_missile(self._missile_var.get())

        _G0 = 9.80665
        pad = dict(padx=8, pady=4)

        # Clear previous content
        for w in self._params_inner.winfo_children():
            w.destroy()

        def _row(frame, row, label, value):
            ttk.Label(frame, text=label).grid(
                row=row, column=0, sticky=tk.W, padx=(6, 2), pady=2)
            ttk.Label(frame, text=value).grid(
                row=row, column=1, sticky=tk.W, padx=(0, 6), pady=2)

        def _row2(frame, row, lab1, val1, lab2='', val2=''):
            """Two label:value pairs on a single row (4-column layout)."""
            ttk.Label(frame, text=lab1).grid(
                row=row, column=0, sticky=tk.W, padx=(6, 2), pady=2)
            ttk.Label(frame, text=val1).grid(
                row=row, column=1, sticky=tk.W, padx=(0, 10), pady=2)
            if lab2:
                ttk.Label(frame, text=lab2).grid(
                    row=row, column=2, sticky=tk.W, padx=(8, 2), pady=2)
                ttk.Label(frame, text=val2).grid(
                    row=row, column=3, sticky=tk.W, padx=(0, 6), pady=2)

        # ── Compute totals used in summary ────────────────────────────
        total_prop = p.mass_propellant
        node = p.stage2
        while node is not None:
            total_prop += node.mass_propellant
            node = node.stage2

        _node_l, _sn_l, _stage_lengths = p, 1, []
        while _node_l is not None:
            if _node_l.length_m > 0:
                _stage_lengths.append((_sn_l, _node_l.length_m))
            _node_l = _node_l.stage2
            _sn_l += 1
        _total_len = sum(l for _, l in _stage_lengths)
        if p.shroud_length_m > 0:
            _total_len += p.shroud_length_m

        liftoff_tw = (p.thrust_N / (p.mass_initial * _G0)
                      if p.mass_initial > 0 else 0.0)

        # ── Summary (4-col, 2 pairs per row) ──────────────────────────
        sf = ttk.LabelFrame(self._params_inner, text="Summary")
        sf.pack(fill=tk.X, **pad)
        sf.columnconfigure(1, weight=1)
        sf.columnconfigure(3, weight=1)

        r = 0
        ttk.Label(sf, text="Name:").grid(
            row=r, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        ttk.Label(sf, text=p.name).grid(
            row=r, column=1, columnspan=3, sticky=tk.W, padx=(0, 6), pady=2)
        r += 1
        _len_str = f"{_total_len:.1f} m" if _total_len > 0 else "—"
        _row2(sf, r, "Launch mass:", f"{p.mass_initial:,.0f} kg",
              "Total length:", _len_str); r += 1
        _row2(sf, r, "Total propellant:", f"{total_prop:,.0f} kg",
              "Liftoff T/W:", f"{liftoff_tw:.2f}"); r += 1
        if p.payload_kg > 0:
            _tw_lbl = "Throw weight:" if p.rv_separates else "Payload:"
            _row2(sf, r, _tw_lbl, f"{p.payload_kg:,.0f} kg"); r += 1

        # ── Per-stage blocks ──────────────────────────────────────────
        sn = 1
        node = p
        while node is not None:
            is_last = node.stage2 is None
            lf = ttk.LabelFrame(self._params_inner,
                                text=f"Stage {sn}" if sn > 1 else "Stage 1")
            lf.pack(fill=tk.X, **pad)

            prop = node.mass_propellant
            tw   = (node.thrust_N / (node.mass_initial * _G0)
                    if node.mass_initial > 0 else 0.0)

            # Recover this stage's own fueled mass (payload excluded).
            # For non-last stages mass_final is the jettisoned dry mass only,
            # so stage_fueled = mass_final + propellant — no stack arithmetic needed.
            # For the last stage (and single-stage) we strip payload (and shroud
            # if single-stage) from mass_initial, which requires payload_kg to be
            # set correctly on the top-level node.
            is_first = (sn == 1)
            if is_last and is_first:
                # Single-stage: mass_initial = fueled + payload + shroud
                stage_fueled = node.mass_initial - p.payload_kg - p.shroud_mass_kg
            elif is_last:
                # Last of multi: mass_initial = fueled + payload
                stage_fueled = node.mass_initial - p.payload_kg
            else:
                # Non-last: mass_final = jettisoned dry mass only
                stage_fueled = node.mass_final + prop
            stage_dry = stage_fueled - prop
            dry_pct   = stage_dry / stage_fueled * 100 if stage_fueled > 0 else 0.0

            # Two-column layout inside the stage LabelFrame.
            # Left: Dimensions & Masses   Right: Engine Performance
            lf.columnconfigure(0, weight=1)
            lf.columnconfigure(2, weight=1)
            left  = ttk.Frame(lf)
            left.grid( row=0, column=0, sticky="nsew", padx=(4, 0), pady=4)
            ttk.Separator(lf, orient="vertical").grid(
                row=0, column=1, sticky="ns", padx=4)
            right = ttk.Frame(lf)
            right.grid(row=0, column=2, sticky="nsew", padx=(0, 4), pady=4)

            # ── Left: Dimensions & Masses ─────────────────────────────
            r = 0
            _row(left, r, "Diameter (m):",      f"{node.diameter_m:.2f}");       r += 1
            _row(left, r, "Length (m):",         f"{node.length_m:.2f}");         r += 1
            _row(left, r, "Fueled mass (kg):",   f"{stage_fueled:,.0f}");         r += 1
            _row(left, r, "Propellant (kg):",    f"{prop:,.0f}  (computed)");     r += 1
            _row(left, r, "Dry mass (kg):",      f"{stage_dry:,.0f}");            r += 1
            _row(left, r, "Dry mass %:",         f"{dry_pct:.1f}%");              r += 1
            if not is_last:
                _row(left, r, "Coast (s):",      f"{node.coast_time_s:.0f}");     r += 1
            # Debris β for jettisoned stage bodies.
            _body_jettisoned = (not is_last) or p.rv_separates
            if _body_jettisoned:
                beta = tumbling_cylinder_beta(node.mass_final,
                                              node.diameter_m, node.length_m)
                if beta > 0:
                    _row(left, r, "Empty β (kg/m²):", f"{beta:,.0f}");            r += 1

            # ── Right: Engine Performance ─────────────────────────────
            mdot = (node.thrust_N / (node.isp_s * _G0)
                    if node.isp_s > 0 else 0.0)
            r = 0
            _row(right, r, "Thrust (kN):",       f"{node.thrust_N/1000:,.0f}");  r += 1
            _row(right, r, "ISP (s):",            f"{node.isp_s:.0f}");           r += 1
            _row(right, r, "Nozzle area (m²):",  f"{node.nozzle_exit_area_m2:.4f}"); r += 1
            _row(right, r, "Burntime (s):",       f"{node.burn_time_s:.1f}  (computed)"); r += 1
            _row(right, r, "Mass flow (kg/s):",   f"{mdot:.1f}  (computed)");     r += 1
            _row(right, r, "T/W ratio:",          f"{tw:.2f}  (computed)");       r += 1

            sn  += 1
            node = node.stage2

        # ── Front End ─────────────────────────────────────────────────
        af = ttk.LabelFrame(self._params_inner, text="Front End")
        af.pack(fill=tk.X, **pad)
        af.columnconfigure(1, weight=1)
        af.columnconfigure(3, weight=1)

        _pd_m     = getattr(p, 'payload_diameter_m', 0.0)
        _pl_m     = p.nose_length_m
        _fe_shape = NOSE_SHAPE_LABELS.get(p.nose_shape, p.nose_shape)
        r = 0
        _row2(af, r, "Payload shape:", _fe_shape,
              "Payload diameter:", f"{_pd_m:.2f} m" if _pd_m > 0 else "—"); r += 1
        if _pl_m > 0:
            _ref_d = _pd_m if _pd_m > 0 else p.diameter_m
            _ld_str = f"  (L/D {_pl_m / _ref_d:.2f})" if _ref_d > 0 else ""
            _row2(af, r, "Payload length:", f"{_pl_m:.2f} m{_ld_str}"); r += 1

        if p.rv_separates:
            _row2(af, r, "No. of RVs:", str(p.num_rvs),
                  "Per-RV mass:", f"{p.rv_mass_kg:,.0f} kg"); r += 1
            _rv_beta = p.rv_beta_kg_m2
            _pbv_m   = p.bus_mass_kg
            if _pbv_m > 0:
                _row2(af, r, "PBV mass:", f"{_pbv_m:,.0f} kg",
                      "RV β:", f"{_rv_beta:,.0f} kg/m²" if _rv_beta > 0 else "—"); r += 1
            elif _rv_beta > 0:
                _row2(af, r, "RV β:", f"{_rv_beta:,.0f} kg/m²"); r += 1
            _rv_shape_s = NOSE_SHAPE_LABELS.get(
                getattr(p, 'rv_shape', ''), NOSE_SHAPE_LABELS['cone'])
            _rv_d = getattr(p, 'rv_diameter_m', 0.0)
            _rv_l = getattr(p, 'rv_length_m', 0.0)
            _row2(af, r, "RV shape:", _rv_shape_s,
                  "RV diameter:", f"{_rv_d:.2f} m" if _rv_d > 0 else "—"); r += 1
            if _rv_l > 0:
                _row2(af, r, "RV length:", f"{_rv_l:.2f} m"); r += 1

        # ── Shroud ────────────────────────────────────────────────────
        if p.shroud_mass_kg > 0:
            ff = ttk.LabelFrame(self._params_inner, text="Shroud")
            ff.pack(fill=tk.X, **pad)
            r = 0
            _row(ff, r, "Mass (kg):",          f"{p.shroud_mass_kg:,.0f}"); r += 1
            _row(ff, r, "Jettison alt (km):",  f"{p.shroud_jettison_alt_km:.0f}"); r += 1
            if p.shroud_diameter_m > 0:
                _sd = p.shroud_diameter_m
                _row(ff, r, "Diameter (m):",   f"{_sd:.2f}"); r += 1
                _area_ratio = (_sd / p.diameter_m) ** 2
                _row(ff, r, "Area vs body:",   f"{_area_ratio:.2f}×  (drag pre-jettison)"); r += 1
            else:
                _sd = p.diameter_m
            if p.shroud_nose_shape not in ('', 'forden'):
                _row(ff, r, "Nose shape:",
                     NOSE_SHAPE_LABELS.get(p.shroud_nose_shape, p.shroud_nose_shape)); r += 1
                if p.shroud_nose_length_m > 0 and _sd > 0:
                    _sld = p.shroud_nose_length_m / _sd
                    _row(ff, r, "Nose length (m):",
                         f"{p.shroud_nose_length_m:.2f}  (L/D = {_sld:.2f})"); r += 1
            if p.shroud_length_m > 0:
                _row(ff, r, "Length (m):",     f"{p.shroud_length_m:.2f}"); r += 1
                beta = tumbling_cylinder_beta(p.shroud_mass_kg, _sd, p.shroud_length_m)
                if beta > 0:
                    _row(ff, r, "Shroud β (kg/m²):", f"{beta:,.0f}"); r += 1


    # ------------------------------------------------------------------
    # Aim at target
    # ------------------------------------------------------------------
    def _aim_at_target(self):
        """
        Prompt for target lat/lon, then compute great-circle azimuth and
        bisect cutoff time to hit the target range.
        """
        dlg = tk.Toplevel(self)
        dlg.title("Aim at Target (liquid)")
        dlg.resizable(False, False)
        dlg.grab_set()

        frm = ttk.Frame(dlg, padding=10)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Target Latitude (°):").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        lat_var = tk.StringVar(value="0.0")
        ttk.Entry(frm, textvariable=lat_var, width=12).grid(
            row=0, column=1, sticky=tk.W, pady=4)

        ttk.Label(frm, text="Target Longitude (°):").grid(
            row=1, column=0, sticky=tk.W, padx=(0, 6), pady=4)
        lon_var = tk.StringVar(value="0.0")
        ttk.Entry(frm, textvariable=lon_var, width=12).grid(
            row=1, column=1, sticky=tk.W, pady=4)

        result = {}

        def _ok():
            try:
                result['lat'] = float(lat_var.get())
                result['lon'] = float(lon_var.get())
            except ValueError:
                messagebox.showerror("Input error",
                                     "Latitude and longitude must be numbers.",
                                     parent=dlg)
                return
            dlg.destroy()

        def _cancel():
            dlg.destroy()

        btn_frm = ttk.Frame(frm)
        btn_frm.grid(row=2, column=0, columnspan=2, pady=(8, 0))
        ttk.Button(btn_frm, text="OK",     width=8, command=_ok    ).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frm, text="Cancel", width=8, command=_cancel).pack(side=tk.LEFT, padx=4)

        dlg.bind("<Return>", lambda e: _ok())
        dlg.bind("<Escape>", lambda e: _cancel())
        self.wait_window(dlg)

        if 'lat' not in result:
            return

        try:
            lat1_dd = float(self._launch_lat.get())
            lon1_dd = float(self._launch_lon.get())
            lat2_dd = result['lat']
            lon2_dd = result['lon']

            lat1 = np.radians(lat1_dd)
            lon1 = np.radians(lon1_dd)
            lat2 = np.radians(lat2_dd)
            lon2 = np.radians(lon2_dd)

            dlon = lon2 - lon1
            x = np.sin(dlon) * np.cos(lat2)
            y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
            az = np.degrees(np.arctan2(x, y)) % 360
            self._azimuth_var.set(f"{az:.2f}")

            rng_km = range_between(lat1, lon1, lat2, lon2) / 1000.0
            self._status_var.set(
                f"Target: {rng_km:.1f} km  |  Azimuth: {az:.1f}°  —  "
                "Computing cutoff time…")

            missile  = get_missile(self._missile_var.get())
            guidance = self._guidance_var.get()
            la           = float(self._loft_angle_var.get())
            gt_start_str = self._gt_turn_start_var.get().strip()
            gt_stop_str  = self._gt_turn_stop_var.get().strip()
            gt_start_s   = float(gt_start_str) if gt_start_str else 5.0
            gt_stop_s    = float(gt_stop_str)  if gt_stop_str  else None
            try:
                missile.launch_elevation_deg = float(self._launch_el_var.get())
            except (ValueError, AttributeError):
                pass
            threading.Thread(
                target=self._aim_thread,
                args=(missile, guidance, lat1_dd, lon1_dd, az, rng_km, la,
                      gt_start_s, gt_stop_s),
                daemon=True,
            ).start()

        except Exception as e:
            messagebox.showerror("Aim error", str(e))

    def _aim_thread(self, missile, guidance, lat, lon, az, rng_km, la,
                    gt_start_s=5.0, gt_stop_s=None):
        try:
            cutoff = aim_missile(missile, lat, lon, az, rng_km,
                                 guidance=guidance,
                                 burnout_angle_deg=la,
                                 gt_turn_start_s=gt_start_s,
                                 gt_turn_stop_s=gt_stop_s)
            self.after(0, lambda: self._cutoff_var.set(f"{cutoff:.1f}"))
            self.after(0, lambda: self._status_var.set(
                f"Target: {rng_km:.1f} km  |  Azimuth: {az:.1f}°  |  "
                f"Cutoff: {cutoff:.1f} s"))
        except Exception as e:
            self.after(0, lambda: self._status_var.set(f"Aim failed: {e}"))

    # ------------------------------------------------------------------
    # Run buttons
    # ------------------------------------------------------------------
    def _get_inputs(self):
        missile  = get_missile(self._missile_var.get())
        guidance = self._guidance_var.get()
        lat      = float(self._launch_lat.get())
        lon      = float(self._launch_lon.get())
        az       = float(self._azimuth_var.get())
        cutoff_str = self._cutoff_var.get().strip()
        cutoff   = float(cutoff_str) if cutoff_str else None
        la           = float(self._loft_angle_var.get())
        gt_start_str = self._gt_turn_start_var.get().strip()
        gt_stop_str  = self._gt_turn_stop_var.get().strip()
        gt_start_s   = float(gt_start_str) if gt_start_str else 5.0
        gt_stop_s    = float(gt_stop_str)  if gt_stop_str  else None
        orb_alt_str  = self._orbit_alt_var.get().strip()
        target_orbit_km = float(orb_alt_str) if (guidance == "orbital_insertion"
                                                   and orb_alt_str) else None

        # Advanced per-stage pitch: deep-copy the missile and stamp each
        # stage object with the values from the inline rows.
        if (self._adv_pitch_var.get()
                and guidance in ("gravity_turn", "orbital_insertion")
                and self._stage_rows):
            missile = copy.deepcopy(missile)
            node = missile
            for row in self._stage_rows:
                if node is None:
                    break
                try:
                    node.stage_turn_start_s      = float(row['start'].get())
                    node.stage_turn_stop_s        = float(row['stop'].get())
                    node.stage_burnout_angle_deg  = float(row['angle'].get())
                except ValueError:
                    pass  # leave existing/None values if field is blank
                coast_s = row.get('coast', tk.StringVar()).get().strip()
                if coast_s:
                    try:
                        node.coast_time_s = float(coast_s)
                    except ValueError:
                        pass
                node = node.stage2

        # Global yaw program (checkbox + three fields)
        def _fon(sv):
            try:
                s = sv.get().strip()
                return float(s) if s else None
            except Exception:
                return None
        _yaw_chk = getattr(self, '_adv_yaw_var', None)
        yaw_enabled = (bool(_yaw_chk and _yaw_chk.get())
                       and guidance in ("gravity_turn", "orbital_insertion"))
        yaw_maneuvers = []
        if yaw_enabled:
            for _yvars in self._yaw_vars:
                fa = _fon(_yvars['final_az'])
                if fa is not None:
                    yaw_maneuvers.append((_fon(_yvars['start']),
                                          _fon(_yvars['stop']),
                                          fa))

        try:
            launch_elevation_deg = float(self._launch_el_var.get())
        except (ValueError, AttributeError):
            launch_elevation_deg = 90.0
        missile.launch_elevation_deg = launch_elevation_deg

        return (missile, guidance, lat, lon, az, cutoff, la,
                gt_start_s, gt_stop_s, target_orbit_km,
                yaw_maneuvers, launch_elevation_deg)

    def _open_sweep(self):
        ParametricSweepDialog(self)

    def _open_range_ring(self):
        RangeRingDialog(self)

    def _run_flyout(self):
        if self._running:
            return
        try:
            (missile, guidance, lat, lon, az, cutoff, la,
             gt_start_s, gt_stop_s, target_orbit_km,
             yaw_maneuvers, launch_elevation_deg) = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self._running = True
        self._status_var.set("Running simulation…")
        threading.Thread(
            target=self._run_thread,
            args=(missile, guidance, lat, lon, az, cutoff, la,
                  gt_start_s, gt_stop_s, target_orbit_km,
                  yaw_maneuvers, launch_elevation_deg, False),
            daemon=True,
        ).start()

    def _maximize_range(self):
        if self._running:
            return
        try:
            (missile, guidance, lat, lon, az, cutoff, la,
             gt_start_s, gt_stop_s, target_orbit_km,
             yaw_maneuvers, launch_elevation_deg) = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self._running = True
        self._status_var.set("Optimising for maximum range…")
        threading.Thread(
            target=self._run_thread,
            args=(missile, guidance, lat, lon, az, cutoff, la,
                  gt_start_s, None, target_orbit_km,
                  yaw_maneuvers, launch_elevation_deg, True),
            daemon=True,
        ).start()

    def _plan_orbit(self):
        """Handler for the Plan Orbit button."""
        if self._running:
            return
        try:
            (missile, guidance, lat, lon, az, cutoff, la,
             gt_start_s, gt_stop_s, target_orbit_km,
             _yaw_maneuvers, _launch_el) = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        if target_orbit_km is None:
            messagebox.showerror("Input error",
                                 "Enter a target orbit altitude (km) first.")
            return
        self._running = True
        self._status_var.set(
            f"Planning orbital trajectory to {target_orbit_km:.0f} km…")
        threading.Thread(
            target=self._plan_orbit_thread,
            args=(missile, lat, lon, az, target_orbit_km, gt_start_s),
            daemon=True,
        ).start()

    def _plan_orbit_thread(self, missile, lat, lon, az,
                           target_orbit_km, gt_start_s):
        """Worker: runs plan_orbital_insertion then fires the full simulation."""
        try:
            plan = plan_orbital_insertion(
                missile, lat, lon, az, target_orbit_km,
                gt_turn_start_s=gt_start_s)
        except Exception as e:
            self._running = False
            self.after(0, lambda: messagebox.showerror(
                "Planner error", str(e)))
            return

        if not plan['success']:
            self._running = False
            self.after(0, lambda: messagebox.showerror(
                "No solution", plan['message']))
            return

        boost_angle = plan['boost_angle_deg']
        turn_stop   = plan['turn_stop_s']

        # Update GUI fields on the main thread, then run the simulation.
        def _apply_and_run():
            self._loft_angle_var.set(f"{boost_angle:.1f}")
            self._gt_turn_stop_var.set(f"{turn_stop:.1f}")
            self._status_var.set(
                f"Plan found: boost {boost_angle:.0f}°  →  "
                f"{plan['perigee_km']:.0f}×{plan['apogee_km']:.0f} km  "
                f"— running simulation…")

            # If advanced pitch mode is active, populate per-stage rows to
            # exactly replicate the two-phase orbital insertion program that
            # Plan Orbit found:
            #   • Pre-final stages: pitch from 90° to boost_angle over
            #     [gt_start_s, turn_stop], then hold boost_angle.
            #   • Final stage: horizontal (0°) from ignition onwards.
            if self._adv_pitch_var.get() and self._stage_rows:
                p = get_missile(self._missile_var.get())
                # Walk stage chain to find ignition/burnout times
                times, node, t_ign = [], p, 0.0
                while node is not None:
                    t_burn = t_ign + node.burn_time_s
                    times.append({'t_ign': t_ign, 't_burn': t_burn})
                    t_ign = t_burn + node.coast_time_s
                    node = node.stage2
                n_stages = len(self._stage_rows)
                for i, row in enumerate(self._stage_rows):
                    is_last = (i == n_stages - 1)
                    t_i = times[i]['t_ign'] if i < len(times) else 0.0
                    if is_last:
                        # Final stage burns horizontally — stop pitch just before
                        # ignition so the stage is already at 0° when it lights.
                        row['start'].set(f"{max(0.0, t_i - 5.0):.1f}")
                        row['stop'].set(f"{max(0.0, t_i - 1.0):.1f}")
                        row['angle'].set("0.0")
                    else:
                        # Pre-final stages: use the same global pitch program
                        # Plan Orbit found (start → turn_stop → boost_angle).
                        row['start'].set(f"{gt_start_s:.1f}")
                        row['stop'].set(f"{turn_stop:.1f}")
                        row['angle'].set(f"{boost_angle:.1f}")

            # Use _get_inputs() so per-stage overrides (when advanced is active)
            # are applied to the missile before running the simulation.
            try:
                (m_run, guidance_run, lat_run, lon_run, az_run,
                 cutoff_run, la_run,
                 gts_run, gtstp_run, orb_run,
                 yaw_maneuvers_run, el_run) = self._get_inputs()
            except ValueError:
                # Fallback: use the original plan parameters without per-stage overrides.
                m_run, guidance_run = missile, "orbital_insertion"
                lat_run, lon_run, az_run = lat, lon, az
                cutoff_run, la_run = None, boost_angle
                gts_run, gtstp_run, orb_run = gt_start_s, turn_stop, target_orbit_km
                yaw_maneuvers_run = []
                el_run = 90.0

            # _run_thread checks self._running; it's still True from _plan_orbit
            threading.Thread(
                target=self._run_thread,
                args=(m_run, guidance_run, lat_run, lon_run, az_run,
                      cutoff_run, la_run,
                      gts_run, gtstp_run, orb_run,
                      yaw_maneuvers_run, el_run, False),
                daemon=True,
            ).start()

        self.after(0, _apply_and_run)

    def _run_thread(self, missile, guidance, lat, lon, az, cutoff, la,
                    gt_start_s, gt_stop_s, target_orbit_km,
                    yaw_maneuvers, launch_elevation_deg, maximise):
        q_str = self._query_alt_km_var.get().strip()
        q_alt = float(q_str) if (self._query_alt_enable.get() and q_str) else None
        try:
            if maximise:
                result = maximize_range(missile, lat, lon, az, guidance=guidance,
                                        cutoff_time_s=cutoff,
                                        gt_turn_start_s=gt_start_s,
                                        gt_turn_stop_s=gt_stop_s,
                                        reentry_query_alt_km=q_alt)
            else:
                # Orbital insertion trajectories can have very long flight
                # times: a highly elliptical transfer orbit peaks at thousands
                # of km and takes 90+ minutes to come back down.  Use 3 hours
                # so the integrator always reaches the ground.
                _max_t = 10800.0 if guidance == "orbital_insertion" else 3600.0
                result = integrate_trajectory(
                    missile, lat, lon, az,
                    guidance=guidance,
                    burnout_angle_deg=la,
                    cutoff_time_s=cutoff,
                    gt_turn_start_s=gt_start_s,
                    gt_turn_stop_s=gt_stop_s,
                    reentry_query_alt_km=q_alt,
                    target_orbit_alt_km=target_orbit_km,
                    yaw_maneuvers=yaw_maneuvers,
                    launch_elevation_deg=launch_elevation_deg,
                    max_time_s=_max_t)
            self._result = result
            self.after(0, self._on_result_ready)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Simulation error", str(e)))
        finally:
            self._running = False

    # ------------------------------------------------------------------
    # Display results
    # ------------------------------------------------------------------
    def _on_result_ready(self):
        r = self._result

        # If this was a Max Range run, update guidance fields now — all GUI
        # mutations happen here in one batch so nothing fires between the field
        # updates and the canvas redraw.
        if r.get('optimal_burnout_angle_deg') is not None:
            self._loft_angle_var.set(f"{r['optimal_burnout_angle_deg']:.4f}")
        if r.get('optimal_gt_turn_stop_s') is not None and self._guidance_var.get() == "gravity_turn":
            self._gt_turn_stop_var.set(f"{r['optimal_gt_turn_stop_s']:.1f}")

        orbital   = r.get('orbital', False)
        rng_km    = r['range_km']
        rng_nm    = rng_km / 1.852   if rng_km    is not None else None
        rng_mi    = rng_km / 1.60934 if rng_km    is not None else None
        apogee_km = r['apogee_km']

        tof_s       = r['time_of_flight_s']
        imp_spd_kms = r['impact_speed_ms'] / 1000.0 if r['impact_speed_ms'] is not None else None
        apo_lat     = r.get('apogee_lat_deg')
        apo_lon     = r.get('apogee_lon_deg')

        units = self._units_var.get()
        scale_map = {"km": (1.0, "km"), "nm": (1/1.852, "nmi"), "mi": (1/1.60934, "mi")}
        scale, ulbl = scale_map[units]

        oe = r.get('orbital_elements')
        _oe_str = ""
        if oe:
            _oe_str = (f"  |  {oe['perigee_km']:.0f}×{oe['apogee_km']:.0f} km"
                       f"  i={oe['inclination_deg']:.1f}°"
                       f"  e={oe['eccentricity']:.4f}"
                       f"  T={oe['period_min']:.1f} min")

        if orbital and r.get('max_range_km') is None:
            _strip = (f"No sub-orbital solution — exceeds orbital velocity.  "
                      f"Apogee: {apogee_km*scale:.1f} {ulbl}")
            self._status_var.set("Max Range: " + _strip)
        elif orbital:
            _strip = (f"In orbit.  Apogee: {apogee_km*scale:.1f} {ulbl}" + _oe_str)
            self._status_var.set(_strip)
        else:
            _spd_str = f"{imp_spd_kms:.2f} km/s" if imp_spd_kms is not None else "—"
            _strip = (f"Range: {rng_km*scale:.1f} {ulbl}  |  "
                      f"Apogee: {apogee_km*scale:.1f} {ulbl}  |  "
                      f"ToF: {tof_s:.0f} s  |  "
                      f"Impact: {r['impact_lat']:.2f}°N, {r['impact_lon']:.2f}°E  |  "
                      f"Impact spd: {_spd_str}")
            self._status_var.set("Done.  " + _strip)
        self._results_strip_var.set(_strip)
        self._plot_results(r, scale, ulbl)
        self._populate_timeline(r)

    # ------------------------------------------------------------------
    def _populate_timeline(self, r):
        """Fill the Flight Timeline tab from the milestones list."""
        # Clear existing rows
        self._tl_tree.delete(*self._tl_tree.get_children())

        rng_km    = r['range_km']
        apogee_km = r['apogee_km']
        tof_s     = r['time_of_flight_s']
        _orbital  = r.get('orbital', False)
        if rng_km is not None:
            rng_nm = rng_km / 1.852
            rng_mi = rng_km / 1.60934
        imp_spd   = r['impact_speed_ms'] / 1000.0 if r['impact_speed_ms'] is not None else None

        if _orbital:
            oe = r.get('orbital_elements')
            _oe_line = ""
            if oe:
                _oe_line = (f"\nOrbit: {oe['perigee_km']:.0f}×{oe['apogee_km']:.0f} km"
                            f"   i={oe['inclination_deg']:.1f}°"
                            f"   e={oe['eccentricity']:.4f}"
                            f"   T={oe['period_min']:.1f} min")
            _apo_loc = (f"{r['apogee_lat_deg']:.2f}°N  {r['apogee_lon_deg']:.2f}°E"
                        if r.get('apogee_lat_deg') is not None else "—")
            self._tl_summary_var.set(
                f"In orbit — no ground impact within integration window\n"
                f"Apogee: {apogee_km:.1f} km   "
                f"Apogee loc: {_apo_loc}"
                + _oe_line
            )
        else:
            _apo_loc = (f"{r['apogee_lat_deg']:.2f}°N  {r['apogee_lon_deg']:.2f}°E"
                        if r.get('apogee_lat_deg') is not None else "—")
            self._tl_summary_var.set(
                f"Range: {rng_km:.1f} km  /  {rng_nm:.1f} nmi  /  {rng_mi:.1f} mi\n"
                f"Apogee: {apogee_km:.1f} km   "
                f"Apogee loc: {_apo_loc}\n"
                f"Impact: {r['impact_lat']:.2f}°N  {r['impact_lon']:.2f}°E   "
                f"Flight time: {tof_s:.0f} s   "
                f"Impact speed: {f'{imp_spd:.2f} km/s' if imp_spd is not None else '—'}"
            )

        # Key events highlighted differently; debris impact rows get their own tag
        _key_prefixes = ("Ignition", "Apogee", "Perigee", "Impact", "Orbital insertion")

        for idx, m in enumerate(r.get('milestones', [])):
            if m.get('is_debris'):
                tag = "debris"
            elif m['event'].startswith(_key_prefixes):
                tag = "key"
            else:
                tag = "odd" if idx % 2 else "even"
            # Acceleration at Impact is dominated by drag spike — show as blank
            accel_str = (f"{m['accel_ms2']:+.1f}"
                         if not m['event'].startswith("Impact") else "—")
            self._tl_tree.insert("", tk.END, tags=(tag,), values=(
                m['event'],
                f"{m['t_s']:.1f}",
                f"{m['alt_km']:.1f}",
                f"{m['range_km']:.1f}",
                f"{m['speed_kms']:.3f}",
                f"{m['inertial_speed_kms']:.3f}",
                accel_str,
                f"{m['mass_t']:.3f}",
            ))

    def _plot_results(self, r, scale, ulbl):
        t   = np.asarray(r['t'])
        alt = np.asarray(r['alt']) / 1000.0 * scale
        spd = np.asarray(r['speed']) / 1000.0   # always km/s
        rng = np.asarray(r['range']) / 1000.0 * scale
        lat_arr = np.asarray(r['lat'])
        lon_arr = np.asarray(r['lon'])
        orbital = r.get('orbital', False)

        for ax in (self._ax_alt, self._ax_spd, self._ax_traj, self._ax_trk,
                   self._ax_guid, self._ax_guid_twin,
                   self._ax_qmach, self._ax_qmach_twin):
            ax.cla()
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=7)

        # ── Find key event times for orbital trajectories ────────────
        _ins_t = _apo_t = _peri_t = None
        if orbital:
            for ms in r.get('milestones', []):
                ev = ms.get('event', '').lower()
                if 'orbital insertion' in ev and _ins_t is None:
                    _ins_t = ms['t_s']
                elif ev.startswith('apogee') and _apo_t is None:
                    _apo_t = ms['t_s']
                elif ev.startswith('perigee') and _peri_t is None:
                    _peri_t = ms['t_s']
            # Circular-orbit fallback: no perigee found → show one full period
            if _peri_t is None and _ins_t is not None:
                oe = r.get('orbital_elements')
                if oe:
                    _peri_t = _ins_t + oe['period_min'] * 60.0

        # Array indices for truncation
        _ins_idx  = (int(np.searchsorted(t, _ins_t))
                     if _ins_t is not None else len(t) - 1)
        _peri_idx = (int(np.searchsorted(t, _peri_t))
                     if _peri_t is not None else len(t) - 1)
        # Clamp to valid range
        _ins_idx  = min(_ins_idx,  len(t) - 1)
        _peri_idx = min(_peri_idx, len(t) - 1)

        # ── Altitude vs Time (truncate at insertion for orbital) ──────
        _sl = slice(0, _ins_idx + 1) if orbital else slice(None)
        self._ax_alt.plot(t[_sl], alt[_sl], color='royalblue', linewidth=1.5)
        self._ax_alt.set_xlabel("Time (s)", fontsize=8)
        self._ax_alt.set_ylabel(f"Altitude ({ulbl})", fontsize=8)
        self._ax_alt.set_title("Altitude vs Time", fontsize=9)
        self._ax_alt.fill_between(t[_sl], 0, alt[_sl],
                                  alpha=0.12, color='royalblue')

        # ── Speed vs Time ─────────────────────────────────────────────
        self._ax_spd.plot(t, spd, color='firebrick', linewidth=1.5)
        self._ax_spd.set_xlabel("Time (s)", fontsize=8)
        self._ax_spd.set_ylabel("Speed (km/s)", fontsize=8)
        self._ax_spd.set_title("Speed vs Time", fontsize=9)

        # ── Altitude vs Range (truncate at insertion for orbital) ─────
        self._ax_traj.plot(rng[_sl], alt[_sl], color='seagreen', linewidth=1.5)
        self._ax_traj.set_xlabel(f"Downrange ({ulbl})", fontsize=8)
        self._ax_traj.set_ylabel(f"Altitude ({ulbl})", fontsize=8)
        self._ax_traj.set_title("Altitude vs Range", fontsize=9)
        self._ax_traj.fill_between(rng[_sl], 0, alt[_sl],
                                   alpha=0.12, color='seagreen')

        # ── Ground Track (truncate at perigee / one orbit for orbital) ─
        center_lon = float(lon_arr[0])          # launch meridian as origin

        # Truncate ground-track arrays
        _trk_sl = slice(0, _peri_idx + 1) if orbital else slice(None)
        lon_trk = lon_arr[_trk_sl]
        lat_trk = lat_arr[_trk_sl]
        lon_c   = ((lon_trk - center_lon + 180.0) % 360.0) - 180.0

        # NaN-break any residual jumps > 180° (multi-hemisphere trajectories)
        lon_c = list(lon_c)
        lat_c = list(lat_trk)
        i = 1
        while i < len(lon_c):
            if abs(lon_c[i] - lon_c[i - 1]) > 180:
                lon_c.insert(i, np.nan)
                lat_c.insert(i, np.nan)
                i += 2
            else:
                i += 1

        self._ax_trk.plot(lon_c, lat_c, color='black', linewidth=1.2, zorder=2)
        self._ax_trk.plot(0.0, lat_arr[0], 'go', markersize=7,
                          label="Launch", zorder=5)

        if not orbital:
            impact_lon_c = ((lon_arr[-1] - center_lon + 180.0) % 360.0) - 180.0
            self._ax_trk.plot(impact_lon_c, lat_arr[-1], 'r*', markersize=9,
                              label="Impact", zorder=5)

        # Orbital event markers (insertion ◆, apogee ▲, perigee ▼)
        if orbital:
            for _t_ev, mkr, col, lbl in [
                (_ins_t,  'D', '#003580', 'Insertion'),
                (_apo_t,  '^', '#6600bb', 'Apogee'),
                (_peri_t, 'v', '#006655', 'Perigee'),
            ]:
                if _t_ev is None or _t_ev > t[-1]:
                    continue
                _ev_lat = float(np.interp(_t_ev, t, lat_arr))
                _ev_lon = float(np.interp(_t_ev, t, lon_arr))
                _ev_lon_c = ((_ev_lon - center_lon + 180.0) % 360.0) - 180.0
                self._ax_trk.plot(_ev_lon_c, _ev_lat, mkr, color=col,
                                  markersize=7, label=lbl, zorder=6)

        # Debris impact locations — red crosses, one per shed stage / fairing.
        _debris_plotted = False
        for m in r.get('milestones', []):
            if not m.get('is_debris'):
                continue
            d_lat = m.get('impact_lat')
            d_lon = m.get('impact_lon')
            if d_lat is None or d_lon is None:
                continue
            d_lon_c = ((d_lon - center_lon + 180.0) % 360.0) - 180.0
            self._ax_trk.plot(d_lon_c, d_lat, 'rx', markersize=8,
                              markeredgewidth=1.8,
                              label="Debris" if not _debris_plotted else "_nolegend_",
                              zorder=5)
            _debris_plotted = True

        # Capture the trajectory-fitted limits, draw borders, then restore so
        # the world-spanning border lines cannot expand the view.
        # Add 20% padding on each side beyond matplotlib's default 5% margin.
        self._ax_trk.autoscale()
        xlo, xhi = self._ax_trk.get_xlim()
        ylo, yhi = self._ax_trk.get_ylim()
        xpad = (xhi - xlo) * 0.20
        ypad = (yhi - ylo) * 0.20
        _draw_borders(self._ax_trk, center_lon)
        self._ax_trk.set_xlim(xlo - xpad, xhi + xpad)
        self._ax_trk.set_ylim(ylo - ypad, yhi + ypad)

        # Tick labels show absolute longitudes (convert back from centred frame)
        self._ax_trk.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{((v + center_lon + 180) % 360) - 180:.0f}°"))
        self._ax_trk.set_xlabel("Longitude (°E)", fontsize=8)
        self._ax_trk.set_ylabel("Latitude (°N)", fontsize=8)
        self._ax_trk.set_title("Ground Track", fontsize=9)
        self._ax_trk.legend(fontsize=7)

        # ── Pitch, Azimuth vs. Time ───────────────────────────────────
        ax_g  = self._ax_guid
        ax_g2 = self._ax_guid_twin   # reuse pre-created twin (avoids stacking)
        t_plot = np.asarray(r.get('t', []))
        pc     = np.asarray(r.get('pitch_cmd_deg', []))
        ac     = np.asarray(r.get('az_cmd_deg', []))

        if len(t_plot) > 0 and len(pc) == len(t_plot):
            ax_g.plot(t_plot, pc, color='royalblue', lw=1.4, label='Pitch (°)')
            if len(ac) == len(t_plot):
                ax_g2.plot(t_plot, ac, color='darkorange', lw=1.4,
                           ls='--', label='Azimuth (°)')
                # 5° tick steps on the azimuth axis
                ax_g2.yaxis.set_major_locator(
                    matplotlib.ticker.MultipleLocator(5))
                ax_g2.yaxis.set_label_position('right')
                ax_g2.yaxis.set_ticks_position('right')
                ax_g2.set_ylabel('Azimuth (°)', fontsize=7, color='darkorange')
                ax_g2.tick_params(labelsize=7, colors='darkorange')
                # Combined legend on the primary axis
                _l1, _lb1 = ax_g.get_legend_handles_labels()
                _l2, _lb2 = ax_g2.get_legend_handles_labels()
                ax_g.legend(_l1 + _l2, _lb1 + _lb2,
                             fontsize=7, loc='upper right')
            # Stage separation and yaw event lines
            for ms in r.get('milestones', []):
                _ev = ms.get('event', '').lower()
                _t  = ms.get('t_s', None)
                if _t is None:
                    continue
                if 'burnout' in _ev or 'ignition' in _ev:
                    ax_g.axvline(_t, color='#aaaaaa', lw=0.8, ls=':')
        ax_g.set_xlabel('Time (s)', fontsize=7)
        ax_g.set_ylabel('Elevation (°)', fontsize=7, color='royalblue')
        ax_g.tick_params(labelsize=7, colors='royalblue')
        ax_g.set_title('Pitch, Azimuth vs. Time', fontsize=8)
        ax_g.grid(True, alpha=0.35)

        # ── Dyn. Pressure & Mach (burn period only) ──────────────────
        from atmosphere import atmosphere as _atm
        _alt_m  = np.asarray(r.get('alt', []))
        _vel_ec = np.asarray(r.get('vel_ecef', []))
        _t_aero = np.asarray(r['t'])

        if len(_alt_m) > 1 and _vel_ec.ndim == 2 and len(_vel_ec) == len(_alt_m):
            _spd_ms = np.asarray(r['speed'])
            _rho    = np.empty(len(_alt_m))
            _sound  = np.empty(len(_alt_m))
            for _i, _h in enumerate(_alt_m):
                _, _, _rho[_i], _sound[_i] = _atm(float(_h))
            _q_kpa = 0.5 * _rho * _spd_ms**2 / 1e3
            _mach  = _spd_ms / np.where(_sound > 0, _sound, 1.0)

            # Restrict to burn period (t ≤ last burnout milestone)
            _ms = r.get('milestones', [])
            _bo_times = [float(m['t_s']) for m in _ms
                         if any(k in m.get('event', '').lower()
                                for k in ('burnout', 'cutoff', 'burn out'))]
            _t_cutoff = max(_bo_times) if _bo_times else float(_t_aero[-1])
            _mask = _t_aero <= _t_cutoff
            _tb   = _t_aero[_mask]
            _qb   = _q_kpa[_mask]
            _mb   = _mach[_mask]

            ax_qm  = self._ax_qmach
            ax_mch = self._ax_qmach_twin
            ax_qm.fill_between(_tb, _qb, alpha=0.18, color='steelblue')
            ax_qm.plot(_tb, _qb, color='steelblue', lw=1.3, label='q (kPa)')
            ax_mch.plot(_tb, _mb, color='darkorange', lw=1.2, ls='--', label='Mach')
            # Annotate max-q
            _qmax_i = int(np.argmax(_qb))
            ax_qm.axvline(_tb[_qmax_i], color='steelblue', lw=0.8, ls=':', alpha=0.7)
            ax_qm.annotate(
                f"max-q\n{_qb[_qmax_i]:.1f} kPa\nM {_mb[_qmax_i]:.1f}",
                xy=(_tb[_qmax_i], _qb[_qmax_i]),
                xytext=(6, -4), textcoords='offset points',
                fontsize=6, color='steelblue', va='top')
            _l1, _lb1 = ax_qm.get_legend_handles_labels()
            _l2, _lb2 = ax_mch.get_legend_handles_labels()
            ax_qm.legend(_l1 + _l2, _lb1 + _lb2, fontsize=6, loc='upper right')
            ax_qm.set_xlabel('Time (s)', fontsize=7)
            ax_qm.set_ylabel('q  (kPa)', fontsize=7, color='steelblue')
            ax_qm.tick_params(labelsize=7, colors='steelblue')
            ax_qm.set_title('Dyn. Pressure, Mach vs. Time', fontsize=8)
            ax_qm.grid(True, alpha=0.35)
            ax_mch.set_ylabel('Mach', fontsize=7, color='darkorange')
            ax_mch.tick_params(labelsize=7, colors='darkorange')
            ax_mch.yaxis.set_label_position('right')
            ax_mch.yaxis.set_ticks_position('right')

        self._canvas.draw()

    # ------------------------------------------------------------------
    # File / Help actions
    # ------------------------------------------------------------------
    def _export_figures(self):
        """Save the trajectory plots to PNG, PDF, or SVG."""
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[
                ("PNG image",    "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG vector",   "*.svg"),
                ("All files",    "*.*"),
            ],
            title="Export figures",
        )
        if not path:
            return
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
        self._status_var.set(f"Figures exported: {path}")

    # ------------------------------------------------------------------
    def _trajectory_metadata(self):
        """Return a dict of all guidance/launch settings for CSV header embedding."""
        meta = {
            'missile':              self._missile_var.get(),
            'launch_lat':           self._launch_lat.get(),
            'launch_lon':           self._launch_lon.get(),
            'azimuth_deg':          self._azimuth_var.get(),
            'guidance':        self._guidance_var.get(),
            'burnout_angle_deg': self._loft_angle_var.get(),
            'gt_turn_start_s': self._gt_turn_start_var.get(),
            'gt_turn_stop_s':       self._gt_turn_stop_var.get(),
            'cutoff_s':             self._cutoff_var.get(),
            'launch_elevation_deg': getattr(self, '_launch_el_var',
                                            tk.StringVar(value='90')).get(),
            'adv_pitch':            self._adv_pitch_var.get(),
            'adv_yaw':              self._adv_yaw_var.get(),
            'yaw_maneuvers': [
                {'start':    v['start'].get(),
                 'stop':     v['stop'].get(),
                 'final_az': v['final_az'].get()}
                for v in self._yaw_vars
            ],
        }
        # Per-stage pitch / yaw overrides
        if self._adv_pitch_var.get() and self._stage_rows:
            meta['stage_overrides'] = [
                {
                    'start': row['start'].get(),
                    'stop':  row['stop'].get(),
                    'angle': row['angle'].get(),
                    'coast': row.get('coast', tk.StringVar()).get(),
                }
                for row in self._stage_rows
            ]
        return meta

    def _apply_trajectory_metadata(self, meta):
        """Restore GUI fields from a metadata dict loaded from a CSV header."""
        name = meta.get('missile', '')
        if name in MISSILE_DB or name in [m for m in MISSILE_DB]:
            self._missile_var.set(name)
            self._on_missile_changed()
        self._launch_lat.set(meta.get('launch_lat', ''))
        self._launch_lon.set(meta.get('launch_lon', ''))
        self._azimuth_var.set(meta.get('azimuth_deg', '0.0'))
        guidance = meta.get('guidance', 'gravity_turn')
        self._guidance_var.set(guidance)
        self._update_guidance_labels(guidance)
        self._loft_angle_var.set(meta.get('burnout_angle_deg', '45.0'))
        self._gt_turn_start_var.set(meta.get('gt_turn_start_s', '5.0'))
        self._gt_turn_stop_var.set(meta.get('gt_turn_stop_s', ''))
        self._cutoff_var.set(meta.get('cutoff_s', ''))
        if hasattr(self, '_launch_el_var'):
            self._launch_el_var.set(meta.get('launch_elevation_deg', '90.0'))
        self._adv_yaw_var.set(bool(meta.get('adv_yaw', False)))
        saved_yaw = meta.get('yaw_maneuvers', [])
        # Back-compat: old single-maneuver keys
        if not saved_yaw and meta.get('yaw_final_az_deg', ''):
            saved_yaw = [{'start': meta.get('yaw_start_s', ''),
                          'stop':  meta.get('yaw_stop_s', ''),
                          'final_az': meta.get('yaw_final_az_deg', '')}]
        for _i, _yvars in enumerate(self._yaw_vars):
            _d = saved_yaw[_i] if _i < len(saved_yaw) else {}
            _yvars['start'].set(_d.get('start', ''))
            _yvars['stop'].set(_d.get('stop', ''))
            _yvars['final_az'].set(_d.get('final_az', ''))
        # Per-stage overrides — expand the panel then fill row by row
        adv = bool(meta.get('adv_pitch', False))
        self._adv_pitch_var.set(adv)
        self._on_adv_pitch_toggled()
        overrides = meta.get('stage_overrides', [])
        if adv and overrides and self._stage_rows:
            for row, ov in zip(self._stage_rows, overrides):
                row['start'].set(ov.get('start', ''))
                row['stop'].set(ov.get('stop', ''))
                row['angle'].set(ov.get('angle', ''))
                if 'coast' in row:
                    row['coast'].set(ov.get('coast', ''))

    def _save_trajectory(self):
        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        import re as _re, datetime as _dt
        from tkinter.filedialog import asksaveasfilename
        _EXPORT_TRAJ_DIR.mkdir(parents=True, exist_ok=True)
        ts      = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        missile = _re.sub(r'[^\w\-]', '_', self._missile_var.get())[:32]
        rng_km  = self._result.get('range_km')
        rng_sfx = f"_{rng_km:.0f}km" if rng_km is not None else ""
        path = asksaveasfilename(
            defaultextension=".csv",
            initialfile=f"{ts}_{missile}{rng_sfx}.traj.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export Trajectory",
        )
        if not path:
            return
        r = self._result
        rows = ["piece,time_s,lat_deg,lon_deg,alt_m,speed_ms,range_km"]
        for i, ti in enumerate(r['t']):
            rows.append(f"vehicle,{ti:.3f},{r['lat'][i]:.6f},{r['lon'][i]:.6f},"
                        f"{r['alt'][i]:.1f},{r['speed'][i]:.2f},{r['range'][i]/1000.0:.3f}")
        for d in r.get('debris_trajectories', []):
            label = d['label'].replace(',', ' ')
            for i, ti in enumerate(d['t']):
                rows.append(f"{label},{ti:.3f},{d['lat'][i]:.6f},{d['lon'][i]:.6f},"
                            f"{d['alt'][i]:.1f},,")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(rows) + "\n")
        self._status_var.set(f"Trajectory CSV exported: {path}")

    def _export_trajectory_xlsx(self):
        """Export the trajectory time-series to an XLSX workbook."""
        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            messagebox.showerror("Missing dependency",
                                 f"openpyxl is required:\n{exc}")
            return
        import re as _re, datetime as _dt
        from tkinter.filedialog import asksaveasfilename
        ts      = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        missile = _re.sub(r'[^\w\-]', '_', self._missile_var.get())[:32]
        rng_km  = self._result.get('range_km')
        rng_sfx = f"_{rng_km:.0f}km" if rng_km is not None else ""
        path = asksaveasfilename(
            defaultextension=".xlsx",
            initialfile=f"{ts}_{missile}{rng_sfx}.traj.xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
            title="Export Trajectory XLSX",
        )
        if not path:
            return
        r = self._result
        wb = Workbook()
        ws = wb.active
        ws.title = "Trajectory"
        ws.append(["piece", "time_s", "lat_deg", "lon_deg",
                   "alt_m", "speed_ms", "range_km"])
        for i, ti in enumerate(r['t']):
            ws.append(["vehicle", float(ti),
                       float(r['lat'][i]), float(r['lon'][i]),
                       float(r['alt'][i]), float(r['speed'][i]),
                       float(r['range'][i]) / 1000.0])
        for d in r.get('debris_trajectories', []):
            label = d['label']
            for i, ti in enumerate(d['t']):
                ws.append([label, float(ti),
                           float(d['lat'][i]), float(d['lon'][i]),
                           float(d['alt'][i]), None, None])
        wb.save(path)
        self._status_var.set(f"Trajectory XLSX exported: {path}")

    def _export_kml(self):
        """Export the ground track and 3-D trajectory path as a KML file."""
        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".kml",
            filetypes=[("KML files", "*.kml"), ("All files", "*.*")],
            title="Export trajectory KML",
        )
        if not path:
            return

        r   = self._result
        lat = np.asarray(r['lat'])
        lon = np.asarray(r['lon'])
        alt = np.asarray(r['alt'])

        # 3-D trajectory (absolute altitude)
        coords_3d = " ".join(
            f"{lo:.6f},{la:.6f},{a:.1f}"
            for lo, la, a in zip(lon, lat, alt)
        )
        # Ground track (clamped to ground)
        coords_gnd = " ".join(
            f"{lo:.6f},{la:.6f},0"
            for lo, la in zip(lon, lat)
        )

        missile_name = self._missile_var.get()

        # Build debris Placemarks
        debris_placemarks = []
        for d in r.get('debris_trajectories', []):
            d_lat = np.asarray(d['lat'])
            d_lon = np.asarray(d['lon'])
            d_alt = np.asarray(d['alt'])
            label = d['label']
            c3d = " ".join(f"{lo:.6f},{la:.6f},{a:.1f}"
                           for lo, la, a in zip(d_lon, d_lat, d_alt))
            cgnd = " ".join(f"{lo:.6f},{la:.6f},0"
                            for lo, la in zip(d_lon, d_lat))
            debris_placemarks.append(f"""
    <Placemark>
      <name>{label} (3-D)</name>
      <styleUrl>#debrisTraj</styleUrl>
      <LineString>
        <altitudeMode>absolute</altitudeMode>
        <tessellate>0</tessellate>
        <coordinates>{c3d}</coordinates>
      </LineString>
    </Placemark>

    <Placemark>
      <name>{label} ground track</name>
      <styleUrl>#debrisGnd</styleUrl>
      <LineString>
        <altitudeMode>clampToGround</altitudeMode>
        <tessellate>1</tessellate>
        <coordinates>{cgnd}</coordinates>
      </LineString>
    </Placemark>

    <Placemark>
      <name>{label} impact</name>
      <Point>
        <altitudeMode>clampToGround</altitudeMode>
        <coordinates>{d_lon[-1]:.6f},{d_lat[-1]:.6f},0</coordinates>
      </Point>
    </Placemark>""")

        debris_xml = "".join(debris_placemarks)

        kml = f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>{missile_name} Trajectory</name>

    <Style id="traj3d">
      <LineStyle><color>ffff0000</color><width>2</width></LineStyle>
    </Style>
    <Style id="trajGnd">
      <LineStyle><color>880000ff</color><width>1</width></LineStyle>
    </Style>
    <Style id="debrisTraj">
      <LineStyle><color>ff00aaff</color><width>1</width></LineStyle>
    </Style>
    <Style id="debrisGnd">
      <LineStyle><color>8800aaff</color><width>1</width></LineStyle>
    </Style>

    <Placemark>
      <name>Launch</name>
      <Point>
        <altitudeMode>clampToGround</altitudeMode>
        <coordinates>{lon[0]:.6f},{lat[0]:.6f},0</coordinates>
      </Point>
    </Placemark>

    <Placemark>
      <name>Impact</name>
      <Point>
        <altitudeMode>clampToGround</altitudeMode>
        <coordinates>{lon[-1]:.6f},{lat[-1]:.6f},0</coordinates>
      </Point>
    </Placemark>

    <Placemark>
      <name>Trajectory (3-D)</name>
      <styleUrl>#traj3d</styleUrl>
      <LineString>
        <altitudeMode>absolute</altitudeMode>
        <tessellate>0</tessellate>
        <coordinates>{coords_3d}</coordinates>
      </LineString>
    </Placemark>

    <Placemark>
      <name>Ground Track</name>
      <styleUrl>#trajGnd</styleUrl>
      <LineString>
        <altitudeMode>clampToGround</altitudeMode>
        <tessellate>1</tessellate>
        <coordinates>{coords_gnd}</coordinates>
      </LineString>
    </Placemark>
{debris_xml}
  </Document>
</kml>"""

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(kml)
        self._status_var.set(f"KML exported: {path}")

    # ------------------------------------------------------------------
    # NOTAM overlay load / clear
    # ------------------------------------------------------------------

    def _load_notam_overlay(self):
        """Parse a KML or KMZ file and store polygon rings for Folium rendering."""
        from tkinter.filedialog import askopenfilename
        import xml.etree.ElementTree as ET
        import zipfile, io

        path = askopenfilename(
            title="Load NOTAM overlay",
            filetypes=[("KML / KMZ files", "*.kml *.kmz"), ("All files", "*.*")],
        )
        if not path:
            return

        # KMZ is a ZIP containing a .kml file.
        if path.lower().endswith(".kmz"):
            with zipfile.ZipFile(path) as zf:
                kml_names = [n for n in zf.namelist() if n.lower().endswith(".kml")]
                if not kml_names:
                    messagebox.showerror("NOTAM overlay",
                                         "No .kml file found inside the .kmz archive.")
                    return
                kml_text = zf.read(kml_names[0])
        else:
            with open(path, "rb") as fh:
                kml_text = fh.read()

        # KML uses a namespace; strip it so tag names are plain.
        kml_text = kml_text.replace(b'xmlns="http://www.opengis.net/kml/2.2"', b"")
        kml_text = kml_text.replace(b'xmlns="http://earth.google.com/kml/2.1"', b"")
        kml_text = kml_text.replace(b'xmlns="http://earth.google.com/kml/2.0"', b"")

        try:
            root = ET.fromstring(kml_text)
        except ET.ParseError as exc:
            messagebox.showerror("NOTAM overlay", f"KML parse error:\n{exc}")
            return

        polygons = []
        for poly_el in root.iter("Polygon"):
            outer = poly_el.find(".//outerBoundaryIs/LinearRing/coordinates")
            if outer is None or not outer.text:
                continue
            coords = []
            for token in outer.text.split():
                parts = token.split(",")
                if len(parts) >= 2:
                    try:
                        lon, lat = float(parts[0]), float(parts[1])
                        coords.append([lon, lat])
                    except ValueError:
                        pass
            if len(coords) >= 3:
                # GeoJSON polygon rings must be closed
                if coords[0] != coords[-1]:
                    coords.append(coords[0])
                polygons.append(coords)

        if not polygons:
            messagebox.showwarning("NOTAM overlay",
                                    "No polygon features found in the file.")
            return

        self._notam_overlay = polygons
        n = len(polygons)
        self._status_var.set(
            f"NOTAM overlay loaded: {n} polygon{'s' if n != 1 else ''} "
            f"from {Path(path).name}"
        )

    def _clear_notam_overlay(self):
        self._notam_overlay = None
        self._status_var.set("NOTAM overlay cleared.")

    # Projection catalogue used by the Cartopy export dialog.
    # Each entry: (display label, factory callable(mid_lon, mid_lat) → CRS)
    _CARTOPY_PROJECTIONS = [
        ("Orthographic (globe)",
         lambda lo, la: __import__('cartopy.crs', fromlist=['Orthographic'])
                        .Orthographic(central_longitude=lo, central_latitude=la)),
        ("Azimuthal Equidistant (true distances from centre)",
         lambda lo, la: __import__('cartopy.crs', fromlist=['AzimuthalEquidistant'])
                        .AzimuthalEquidistant(central_longitude=lo, central_latitude=la)),
        ("Lambert Conformal Conic (mid-latitude)",
         lambda lo, la: __import__('cartopy.crs', fromlist=['LambertConformal'])
                        .LambertConformal(central_longitude=lo, central_latitude=la)),
        ("Plate Carrée (equirectangular)",
         lambda lo, la: __import__('cartopy.crs', fromlist=['PlateCarree'])
                        .PlateCarree()),
        ("Mercator",
         lambda lo, la: __import__('cartopy.crs', fromlist=['Mercator'])
                        .Mercator()),
        ("Robinson (global overview)",
         lambda lo, la: __import__('cartopy.crs', fromlist=['Robinson'])
                        .Robinson(central_longitude=lo)),
        ("Equal Earth",
         lambda lo, la: __import__('cartopy.crs', fromlist=['EqualEarth'])
                        .EqualEarth(central_longitude=lo)),
        ("North Polar Stereographic",
         lambda lo, la: __import__('cartopy.crs', fromlist=['NorthPolarStereo'])
                        .NorthPolarStereo(central_longitude=lo)),
        ("South Polar Stereographic",
         lambda lo, la: __import__('cartopy.crs', fromlist=['SouthPolarStereo'])
                        .SouthPolarStereo(central_longitude=lo)),
    ]

    def _pick_cartopy_projection(self, mid_lon, mid_lat):
        """Modal dialog to choose a Cartopy projection. Returns a CRS or None."""
        dlg = tk.Toplevel(self)
        dlg.title("Choose Projection")
        dlg.resizable(False, False)
        dlg.grab_set()

        ttk.Label(dlg, text="Projection:", padding=(12, 10, 12, 4)).pack(anchor=tk.W)

        lb_frame = ttk.Frame(dlg)
        lb_frame.pack(fill=tk.BOTH, padx=12)
        vsb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        lb  = tk.Listbox(lb_frame, yscrollcommand=vsb.set, activestyle="dotbox",
                         width=52, height=len(self._CARTOPY_PROJECTIONS),
                         selectmode=tk.SINGLE)
        vsb.config(command=lb.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.BOTH)

        for label, _ in self._CARTOPY_PROJECTIONS:
            lb.insert(tk.END, label)
        lb.selection_set(0)   # default: Orthographic

        result = [None]

        def _ok(*_):
            sel = lb.curselection()
            if sel:
                _, factory = self._CARTOPY_PROJECTIONS[sel[0]]
                result[0] = factory(mid_lon, mid_lat)
            dlg.destroy()

        lb.bind("<Double-Button-1>", _ok)

        btn_frm = ttk.Frame(dlg, padding=(12, 8))
        btn_frm.pack(fill=tk.X)
        ttk.Button(btn_frm, text="OK",     command=_ok).pack(side=tk.LEFT)
        ttk.Button(btn_frm, text="Cancel", command=dlg.destroy).pack(
            side=tk.LEFT, padx=6)

        self._center_dialog(dlg)
        self.wait_window(dlg)
        return result[0]

    def _pick_cartopy_export_options(self, mid_lon, mid_lat):
        """Combined projection + map-extent dialog.

        Returns (proj, extent_spec) on OK, or (None, None) on cancel.
        extent_spec is one of:
          None                       → global (ax.set_global)
          ('auto', pad_pct)          → auto-fit with % padding
          (lon_min, lon_max, lat_min, lat_max) → explicit bounds
        """
        dlg = tk.Toplevel(self)
        dlg.title("Cartopy Export Options")
        dlg.resizable(False, False)
        dlg.grab_set()

        # ── Projection list ───────────────────────────────────────────
        ttk.Label(dlg, text="Projection:", padding=(12, 10, 12, 4)).pack(anchor=tk.W)
        lb_frame = ttk.Frame(dlg)
        lb_frame.pack(fill=tk.BOTH, padx=12)
        vsb = ttk.Scrollbar(lb_frame, orient=tk.VERTICAL)
        lb  = tk.Listbox(lb_frame, yscrollcommand=vsb.set, activestyle="dotbox",
                         width=52, height=len(self._CARTOPY_PROJECTIONS),
                         selectmode=tk.SINGLE)
        vsb.config(command=lb.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.BOTH)
        for _lbl, _ in self._CARTOPY_PROJECTIONS:
            lb.insert(tk.END, _lbl)
        lb.selection_set(0)

        ttk.Separator(dlg, orient=tk.HORIZONTAL).pack(fill=tk.X, padx=12, pady=(8, 4))

        # ── Map extent ────────────────────────────────────────────────
        ttk.Label(dlg, text="Map extent:", padding=(12, 0, 12, 4)).pack(anchor=tk.W)
        extent_var = tk.StringVar(value="auto")
        ef = ttk.Frame(dlg, padding=(12, 0, 12, 4))
        ef.pack(fill=tk.X)

        ttk.Radiobutton(ef, text="Global (full world)", variable=extent_var,
                        value="global").grid(row=0, column=0, columnspan=6,
                                            sticky=tk.W, pady=2)

        # Auto-fit row
        af = ttk.Frame(ef)
        af.grid(row=1, column=0, columnspan=6, sticky=tk.W, pady=2)
        ttk.Radiobutton(af, text="Auto-fit to trajectory  —  padding:",
                        variable=extent_var, value="auto").pack(side=tk.LEFT)
        pad_var = tk.StringVar(value="25")
        ttk.Entry(af, textvariable=pad_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(af, text="%").pack(side=tk.LEFT)

        ttk.Radiobutton(ef, text="Custom bounds:", variable=extent_var,
                        value="custom").grid(row=2, column=0, columnspan=6,
                                            sticky=tk.W, pady=(6, 2))

        # Custom bounds sub-grid
        cf = ttk.Frame(ef)
        cf.grid(row=3, column=0, columnspan=6, sticky=tk.W, padx=20, pady=(0, 4))
        ttk.Label(cf, text="N:").grid(row=0, column=0, padx=(0, 2))
        n_var = tk.StringVar(value="")
        ttk.Entry(cf, textvariable=n_var, width=7).grid(row=0, column=1, padx=2)
        ttk.Label(cf, text="°").grid(row=0, column=2)
        ttk.Label(cf, text="S:").grid(row=0, column=3, padx=(8, 2))
        s_var = tk.StringVar(value="")
        ttk.Entry(cf, textvariable=s_var, width=7).grid(row=0, column=4, padx=2)
        ttk.Label(cf, text="°").grid(row=0, column=5)
        ttk.Label(cf, text="W:").grid(row=1, column=0, padx=(0, 2), pady=2)
        w_var = tk.StringVar(value="")
        ttk.Entry(cf, textvariable=w_var, width=7).grid(row=1, column=1, padx=2)
        ttk.Label(cf, text="°").grid(row=1, column=2)
        ttk.Label(cf, text="E:").grid(row=1, column=3, padx=(8, 2))
        e_var = tk.StringVar(value="")
        ttk.Entry(cf, textvariable=e_var, width=7).grid(row=1, column=4, padx=2)
        ttk.Label(cf, text="°").grid(row=1, column=5)

        result = [None, None]

        def _ok(*_):
            sel = lb.curselection()
            if not sel:
                dlg.destroy()
                return
            _, factory = self._CARTOPY_PROJECTIONS[sel[0]]
            result[0] = factory(mid_lon, mid_lat)
            mode = extent_var.get()
            if mode == "global":
                result[1] = None
            elif mode == "auto":
                try:
                    pad = max(0.0, float(pad_var.get()))
                except ValueError:
                    pad = 25.0
                result[1] = ('auto', pad)
            else:
                try:
                    n = float(n_var.get())
                    s = float(s_var.get())
                    w = float(w_var.get())
                    e = float(e_var.get())
                    if s >= n or w >= e:
                        raise ValueError("degenerate bounds")
                    result[1] = (w, e, s, n)
                except ValueError:
                    messagebox.showerror(
                        "Invalid bounds",
                        "Enter numeric values where N > S and E > W.",
                        parent=dlg,
                    )
                    return
            dlg.destroy()

        lb.bind("<Double-Button-1>", _ok)
        btn_frm = ttk.Frame(dlg, padding=(12, 8))
        btn_frm.pack(fill=tk.X)
        ttk.Button(btn_frm, text="OK",     command=_ok).pack(side=tk.LEFT)
        ttk.Button(btn_frm, text="Cancel", command=dlg.destroy).pack(
            side=tk.LEFT, padx=6)

        self._center_dialog(dlg)
        self.wait_window(dlg)
        return result[0], result[1]

    def _export_cartopy(self):
        """Export a static Cartopy map of the current trajectory."""
        try:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature
            import matplotlib.patheffects as pe
            from matplotlib.backends.backend_agg import FigureCanvasAgg
        except ImportError as _e:
            messagebox.showerror(
                "Missing package",
                f"Cartopy is not installed.\n\n{_e}\n\nRun:  pip install cartopy",
            )
            return

        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return

        r   = self._result
        lat = np.asarray(r['lat'], dtype=float)
        lon = np.asarray(r['lon'], dtype=float)
        t   = np.asarray(r['t'],   dtype=float)

        mid_lat = float(np.mean(lat))
        mid_lon = float(np.mean(lon))

        proj, extent_spec = self._pick_cartopy_export_options(mid_lon, mid_lat)
        if proj is None:
            return   # user cancelled

        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("PDF document", "*.pdf"),
                       ("SVG image", "*.svg"), ("All files", "*.*")],
            title="Save Cartopy map",
        )
        if not path:
            return

        geo = ccrs.Geodetic()

        fig    = Figure(figsize=(10, 8), dpi=300)
        canvas = FigureCanvasAgg(fig)
        ax     = fig.add_subplot(1, 1, 1, projection=proj)

        # ── Map extent ────────────────────────────────────────────────
        if extent_spec is None:
            ax.set_global()
        elif extent_spec[0] == 'auto':
            pad_frac = extent_spec[1] / 100.0
            # Include debris track points in bounding box
            _all_lat = [lat]
            _all_lon = [lon]
            for _d in r.get('debris_trajectories', []):
                _all_lat.append(np.asarray(_d['lat'], dtype=float))
                _all_lon.append(np.asarray(_d['lon'], dtype=float))
            _flat = np.concatenate(_all_lat)
            _flon = np.concatenate(_all_lon)
            lat_span = max(float(np.max(_flat) - np.min(_flat)), 2.0)
            lon_span = max(float(np.max(_flon) - np.min(_flon)), 2.0)
            ax.set_extent([
                max(-180.0, float(np.min(_flon)) - lon_span * pad_frac),
                min(+180.0, float(np.max(_flon)) + lon_span * pad_frac),
                max( -90.0, float(np.min(_flat)) - lat_span * pad_frac),
                min( +90.0, float(np.max(_flat)) + lat_span * pad_frac),
            ], crs=ccrs.PlateCarree())
        else:
            ax.set_extent(list(extent_spec), crs=ccrs.PlateCarree())

        # ── Background features ───────────────────────────────────────
        ax.add_feature(cfeature.OCEAN,     facecolor="#d6e8f5", zorder=0)
        ax.add_feature(cfeature.LAND,      facecolor="#e8e4d8", zorder=1)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor="#555555",
                       zorder=2)
        ax.add_feature(cfeature.BORDERS,   linewidth=0.3, edgecolor="#888888",
                       linestyle=":", zorder=2)
        ax.add_feature(cfeature.LAKES,     facecolor="#d6e8f5", linewidth=0.3,
                       edgecolor="#555555", zorder=2)
        ax.gridlines(color="white", linewidth=0.4, linestyle="--", alpha=0.6,
                     zorder=3)

        # ── NOTAM overlay ─────────────────────────────────────────────
        if self._notam_overlay:
            try:
                from shapely.geometry import Polygon as _ShapelyPoly
                from cartopy.feature import ShapelyFeature as _ShapelyFeat
                _polys = [_ShapelyPoly(ring) for ring in self._notam_overlay]
                ax.add_feature(_ShapelyFeat(
                    _polys, ccrs.PlateCarree(),
                    facecolor="#f5f5f5", edgecolor="#c0392b",
                    linewidth=1.5, alpha=0.7, zorder=3,
                ))
            except Exception:
                pass

        _OUTLINE = [pe.withStroke(linewidth=2.5, foreground="white")]

        # ── Main ground track ─────────────────────────────────────────
        _ins_t = next(
            (ms['t_s'] for ms in r.get('milestones', [])
             if 'orbital insertion' in ms.get('event', '').lower()),
            None)
        if _ins_t is not None:
            _sp = int(np.searchsorted(t, _ins_t))
            ax.plot(lon[:_sp + 1], lat[:_sp + 1], color="black",
                    linewidth=1.8, transform=geo, zorder=4,
                    path_effects=_OUTLINE)
            ax.plot(lon[_sp:], lat[_sp:], color="#555555",
                    linewidth=1.2, linestyle="--", transform=geo, zorder=4)
        else:
            ax.plot(lon, lat, color="black", linewidth=1.8,
                    transform=geo, zorder=4, path_effects=_OUTLINE)

        # ── Debris arcs ───────────────────────────────────────────────
        for d in r.get('debris_trajectories', []):
            ax.plot(np.asarray(d['lon'], dtype=float),
                    np.asarray(d['lat'], dtype=float),
                    color="black", linewidth=1.0, alpha=0.5,
                    transform=geo, zorder=4)

        # ── Milestone markers and tick marks ──────────────────────────

        def _show_labeled(e, is_debris, ms):
            if is_debris:
                return (('empty impact' in e or 'shroud impact' in e)
                        and 'impact_lat' in ms)
            return (('ignition' in e and 'stage' not in e) or
                    ('impact'   in e and 'empty' not in e
                                     and 'shroud' not in e))

        def _show_tick(e, is_debris):
            if is_debris:
                return False
            return ('apogee' in e or 're-entry' in e or 'burnout' in e or
                    ('ignition' in e and 'stage' in e) or 'jettison' in e)

        def _mk_pos(ms):
            if ms.get('is_debris') and 'impact_lat' in ms:
                return ms['impact_lat'], ms['impact_lon']
            return (float(np.interp(ms['t_s'], t, lat)),
                    float(np.interp(ms['t_s'], t, lon)))

        # Tick half-length: ~1.5 % of the map's latitude span so ticks scale
        # with the map extent (same visual weight at all ranges).
        try:
            _ext = ax.get_extent(crs=ccrs.PlateCarree())
            _lat_span = max(1.0, _ext[3] - _ext[2])
        except Exception:
            _lat_span = max(1.0, float(np.ptp(lat)))
        _tick_half = max(0.25, _lat_span * 0.015)

        for ms in r.get('milestones', []):
            is_debris = ms.get('is_debris', False)
            e         = ms['event'].lower()
            if _show_labeled(e, is_debris, ms):
                mk_lat, mk_lon = _mk_pos(ms)
                is_impact = 'impact' in e and not is_debris
                ax.plot(mk_lon, mk_lat, marker="o",
                        markersize=7 if is_impact else 5,
                        color="crimson" if is_impact else "white",
                        markeredgecolor="black", markeredgewidth=0.8,
                        transform=geo, zorder=6)
            elif _show_tick(e, is_debris):
                mk_lat, mk_lon = _mk_pos(ms)
                # Find nearest trajectory index to get the local tangent.
                _i = int(np.argmin(np.abs(t - ms['t_s'])))
                _i = int(np.clip(_i, 1, len(t) - 2))
                _dlat = float(lat[_i + 1] - lat[_i - 1])
                _dlon = float(lon[_i + 1] - lon[_i - 1])
                _cos  = np.cos(np.radians(mk_lat)) or 1e-9
                # Magnitude in (north, east) space
                _mag  = np.hypot(_dlat, _dlon * _cos) or 1e-9
                # Perpendicular unit vector (CW rotation): (−E, N) / mag
                _pn = -_dlon * _cos / _mag   # northward component
                _pe =  _dlat       / _mag    # eastward component
                # Offset in geographic degrees
                _dlat_t = _tick_half * _pn
                _dlon_t = _tick_half * _pe / _cos
                ax.plot([mk_lon - _dlon_t, mk_lon + _dlon_t],
                        [mk_lat - _dlat_t, mk_lat + _dlat_t],
                        color="#333333", linewidth=1.5,
                        transform=geo, zorder=6)

        # ── Title ─────────────────────────────────────────────────────
        parts = [self._missile_var.get()]
        rng = r.get('range_km')
        apo = r.get('apogee_km')
        if rng is not None: parts.append(f"Range {rng:.0f} km")
        if apo is not None: parts.append(f"Apogee {apo:.0f} km")
        ax.set_title("  ·  ".join(parts), fontsize=11, pad=8)

        fig.tight_layout()
        canvas.print_figure(path, bbox_inches="tight")
        self._status_var.set(f"Cartopy map saved: {path}")
        _open_file(path)

    def _export_folium(self):
        """Generate an interactive Folium HTML map and open it in the browser."""
        try:
            import folium
        except ImportError:
            messagebox.showerror(
                "Missing package",
                "folium is not installed.\n\nRun:  pip install folium",
            )
            return

        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return

        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
            title="Save Folium map",
        )
        if not path:
            return

        r   = self._result
        lat = np.asarray(r['lat'])
        lon = np.asarray(r['lon'])
        t   = np.asarray(r['t'])
        alt = np.asarray(r.get('alt', []))   # metres; used for prefer_above

        # Unwrap longitude so the polyline never jumps across the antimeridian.
        # Values may exceed ±180°; Leaflet renders them on the correct world copy.
        _diffs = np.diff(lon)
        _diffs = (_diffs + 180.0) % 360.0 - 180.0
        lon_uw = np.empty_like(lon, dtype=float)
        lon_uw[0] = lon[0]
        lon_uw[1:] = lon[0] + np.cumsum(_diffs)

        mid_lat  = float(np.mean(lat))
        mid_lon  = float(np.mean(lon_uw))
        lon_uw_min = float(lon_uw.min())
        lon_uw_max = float(lon_uw.max())

        fmap = folium.Map(location=[mid_lat, mid_lon], zoom_start=4,
                          tiles="CartoDB positron")

        # ── NOTAM overlay (loaded via File → Load NOTAM overlay…) ─────
        if self._notam_overlay:
            _notam_geojson = {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": [ring]},
                        "properties": {},
                    }
                    for ring in self._notam_overlay
                ],
            }
            folium.GeoJson(
                _notam_geojson,
                style_function=lambda _: {
                    "color":       "#c0392b",
                    "weight":      1.5,
                    "opacity":     0.8,
                    "fillColor":   "#f5f5f5",
                    "fillOpacity": 0.30,
                },
            ).add_to(fmap)

        # ── Ground track ──────────────────────────────────────────────
        # For orbital insertions split the track: boost phase in black,
        # orbital phase in dark grey, sharing one point at the junction.
        _ins_t = next(
            (ms['t_s'] for ms in r.get('milestones', [])
             if 'orbital insertion' in ms.get('event', '').lower()),
            None)
        if _ins_t is not None:
            _sp = int(np.searchsorted(t, _ins_t))
            folium.PolyLine(
                list(zip(lat[:_sp + 1].tolist(), lon_uw[:_sp + 1].tolist())),
                color="black", weight=2.0, opacity=0.8,
                tooltip="Boost phase",
            ).add_to(fmap)
            folium.PolyLine(
                list(zip(lat[_sp:].tolist(), lon_uw[_sp:].tolist())),
                color="#555555", weight=1.5, opacity=0.7,
                tooltip="Orbital phase",
            ).add_to(fmap)
        else:
            folium.PolyLine(
                list(zip(lat.tolist(), lon_uw.tolist())),
                color="black", weight=2.0, opacity=0.8,
                tooltip="Ground track",
            ).add_to(fmap)

        # ── Debris ground tracks ──────────────────────────────────────
        for d in r.get('debris_trajectories', []):
            _d_lon = np.asarray(d['lon'], dtype=float)
            _d_diffs = np.diff(_d_lon)
            _d_diffs = (_d_diffs + 180.0) % 360.0 - 180.0
            _d_lon_uw = np.empty_like(_d_lon)
            _d_lon_uw[0] = _d_lon[0]
            if len(_d_diffs):
                _d_lon_uw[1:] = _d_lon[0] + np.cumsum(_d_diffs)
            folium.PolyLine(
                list(zip(d['lat'].tolist(), _d_lon_uw.tolist())),
                color="black", weight=1.5, opacity=0.5,
                tooltip=d['label'],
            ).add_to(fmap)

        # ── Merge simultaneous milestones (coast_time_s == 0) ────────
        raw_milestones = r.get('milestones', [])
        merged = []
        i = 0
        while i < len(raw_milestones):
            ms = raw_milestones[i]
            group = [ms]
            if not ms.get('is_debris', False):
                j = i + 1
                while j < len(raw_milestones):
                    nxt = raw_milestones[j]
                    if (not nxt.get('is_debris', False) and
                            abs(nxt['t_s'] - ms['t_s']) < 0.1):
                        group.append(nxt)
                        j += 1
                    else:
                        break
            merged.append(group)
            i += len(group)

        def _is_rv_impact(group):
            return any('impact' in g['event'].lower() and
                       not g.get('is_debris', False) for g in group)

        merged.sort(key=lambda g: (1 if _is_rv_impact(g) else 0, g[0]['t_s']))

        # ── Circle markers + label data collection ────────────────────
        # Labeled events (filled circle + label + popup):
        #   Launch, stage empty impacts, fairing impact, warhead impact.
        # Tick-mark events (SVG perpendicular line, no circle, no label):
        #   All other non-debris flight events (apogee, re-entry, burnouts…).
        def _show_labeled(e, is_debris, ms):
            if is_debris:
                return (('empty impact' in e or 'shroud impact' in e)
                        and 'impact_lat' in ms)
            return (('ignition' in e and 'stage' not in e) or
                    ('impact'   in e and 'empty' not in e
                                     and 'shroud' not in e))

        def _show_tick(e, is_debris):
            if is_debris:
                return False
            return ('apogee'   in e or
                    're-entry' in e or
                    'burnout'  in e or
                    ('ignition' in e and 'stage' in e) or
                    'jettison'  in e)

        import re as _re_ev, json as _json

        def _name_only(raw):
            """Strip time/data parentheticals; rename Ignition → Launch."""
            name = _re_ev.sub(r'\s*\(\d[^)]*\)\s*$', '', raw).strip()
            if name.lower() == 'ignition':
                return 'Launch'
            return name

        _label_data = []   # [{lat, lon, text, t_s, prefer_above}] for JS labels
        _tick_data  = []   # [{lat, lon}] for JS tick marks

        def _prefer_above(ms, e, is_debris, mk_lat, mk_lon):
            """
            Hint for initial vertical label placement.
            Debris impacts: place the label on the SAME SIDE of the main
            trajectory as the dot, so no trajectory sits between dot and label.
            Other impact events: always above (dot at ground, label toward arc).
            Other events: above while ascending, below while descending.
            """
            if is_debris and 'impact' in e and 'impact_lat' in ms:
                # Find nearest main-trajectory point and compare latitudes.
                dlat2 = (lat - mk_lat) ** 2
                dlon2 = (lon_uw - mk_lon) ** 2
                ni    = int(np.argmin(dlat2 + dlon2))
                diff  = mk_lat - float(lat[ni])
                if abs(diff) < 1e-5:
                    return True   # essentially on the main track → default above
                return bool(diff > 0)   # north of main track → above in screen
            if 'impact' in e:
                return True
            if len(alt) < 2:
                return True
            ti  = float(ms['t_s'])
            ic  = int(np.searchsorted(t, ti))
            i0  = max(0, ic - 5)
            i1  = min(len(t) - 1, ic + 5)
            return bool(alt[i1] >= alt[i0])

        for group in merged:
            ms        = group[0]
            is_debris = ms.get('is_debris', False)
            label     = " / ".join(g['event'] for g in group)
            e         = label.lower()

            if is_debris and 'impact_lat' in ms:
                mk_lat = ms['impact_lat']
                mk_lon = ms['impact_lon']
            else:
                mk_lat = float(np.interp(ms['t_s'], t, lat))
                mk_lon = float(np.interp(ms['t_s'], t, lon_uw))

            if _show_labeled(e, is_debris, ms):
                display_name = _name_only(label)
                popup_html = (
                    f"<b>{display_name}</b><br>"
                    f"t = {ms['t_s']:.1f} s<br>"
                    f"Alt: {ms['alt_km']:.1f} km<br>"
                    f"Range: {ms['range_km']:.1f} km<br>"
                    f"Speed: {ms['speed_kms']:.2f} km/s"
                )
                popup = folium.Popup(popup_html, max_width=220)
                folium.CircleMarker(
                    [mk_lat, mk_lon], radius=5,
                    color="black", weight=1,
                    fill=True, fill_color="black", fill_opacity=1.0,
                    popup=popup, tooltip=display_name,
                ).add_to(fmap)
                _label_data.append({'lat':  mk_lat, 'lon':  mk_lon,
                                    'text': display_name, 't_s': ms['t_s'],
                                    'prefer_above': _prefer_above(
                                        ms, e, is_debris, mk_lat, mk_lon)})
            elif _show_tick(e, is_debris):
                display_name = _name_only(label)
                popup_html = (
                    f"<b>{display_name}</b><br>"
                    f"t = {ms['t_s']:.1f} s<br>"
                    f"Alt: {ms['alt_km']:.1f} km<br>"
                    f"Range: {ms['range_km']:.1f} km<br>"
                    f"Speed: {ms['speed_kms']:.2f} km/s"
                )
                folium.CircleMarker(
                    [mk_lat, mk_lon], radius=8,
                    color="black", weight=0,
                    fill=True, fill_color="black", fill_opacity=0.0,
                    opacity=0.0,
                    popup=folium.Popup(popup_html, max_width=220),
                    tooltip=display_name,
                ).add_to(fmap)
                _tick_data.append({'lat': mk_lat, 'lon': mk_lon})

        # ── Trajectory skeleton for tick-mark perpendicular computation ──
        _n_traj   = min(200, len(lat))
        _idx_traj = np.round(np.linspace(0, len(lat) - 1, _n_traj)).astype(int)
        _traj_pts = [{'lat': float(lat[i]), 'lon': float(lon_uw[i])}
                     for i in _idx_traj]
        traj_json = _json.dumps(_traj_pts)
        tick_json = _json.dumps(_tick_data)

        # All trajectory polylines (main + debris arcs) for collision detection.
        # Passed to JS so that labels are not separated from their dots by ANY arc.
        _all_traj_polys = [_traj_pts]
        for _d in r.get('debris_trajectories', []):
            _dl  = np.asarray(_d['lat'])
            _dlo = np.asarray(_d['lon'])
            _dd  = np.diff(_dlo)
            _dd  = (_dd + 180.0) % 360.0 - 180.0
            _dlu = np.empty_like(_dlo)
            _dlu[0] = _dlo[0]
            if len(_dd):
                _dlu[1:] = _dlo[0] + np.cumsum(_dd)
            _nd  = min(100, len(_dl))
            _ixd = np.round(np.linspace(0, len(_dl) - 1, _nd)).astype(int)
            _all_traj_polys.append([{'lat': float(_dl[i]), 'lon': float(_dlu[i])}
                                    for i in _ixd])
        all_traj_json = _json.dumps(_all_traj_polys)

        # ── Leader-line labels + tick marks (pure JS, update on zoom+pan) ──
        # Labels are name-only; full detail is in the click popup.
        # Tick marks are drawn perpendicular to the trajectory skeleton.
        map_var    = fmap.get_name()
        label_json = _json.dumps(_label_data)
        leader_js  = f"""
        <script>
        (function() {{
            var LABELS    = {label_json};
            var TICKS     = {tick_json};
            var TRAJ      = {traj_json};
            var ALL_TRAJ  = {all_traj_json};
            var H_GAP     = 10;   // px right of the dot centre
            var V_ABOVE   = 4;    // px between dot and nearest edge of label
            var STACK_GAP = 3;    // px between stacked labels
            var PAD       = 2;    // extra padding around each label box
            var TICK_HALF = 8;    // px: half-length of tick mark

            var _svg = null, _con = null, _divs = [], _labelsLayer = null;

            function _init(map) {{
                var mc = map.getContainer();
                _svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
                _svg.style.cssText = 'position:absolute;top:0;left:0;' +
                    'width:100%;height:100%;pointer-events:none;z-index:450;' +
                    'overflow:visible;';
                mc.appendChild(_svg);
                _con = document.createElement('div');
                _con.style.cssText = 'position:absolute;top:0;left:0;' +
                    'width:0;height:0;pointer-events:none;z-index:500;';
                mc.appendChild(_con);
                LABELS.forEach(function(lb) {{
                    var d = document.createElement('div');
                    d.style.cssText = 'position:absolute;font-size:11px;' +
                        'font-family:sans-serif;font-weight:bold;' +
                        'white-space:nowrap;padding:1px 4px;display:none;';
                    d.textContent = lb.text;
                    _con.appendChild(d);
                    _divs.push(d);
                }});

                // Dummy LayerGroup used purely as a toggle handle for the
                // Labels overlay (tick marks + leader-line labels).
                // Added to the map so it starts checked in the layer control.
                _labelsLayer = L.layerGroup().addTo(map);
                L.control.layers(
                    {{}},
                    {{'Labels': _labelsLayer}},
                    {{collapsed: false}}
                ).addTo(map);
            }}

            function _update(map) {{
                if (!_con) return;
                _svg.innerHTML = '';
                // If the Labels overlay is unchecked, hide all label divs
                // (tick SVG was already cleared above) and stop.
                if (_labelsLayer && !map.hasLayer(_labelsLayer)) {{
                    _divs.forEach(function(d) {{ d.style.display = 'none'; }});
                    return;
                }}

                var mc   = map.getContainer();
                var mapW = mc.offsetWidth  || 800;
                var mapH = mc.offsetHeight || 600;
                var EDGE = 40;

                // ── Convert trajectory skeleton to container points ───────
                var tPts = TRAJ.map(function(tp) {{
                    return map.latLngToContainerPoint([tp.lat, tp.lon]);
                }});
                // All polylines (main + debris arcs) projected for collision checks.
                var allPts = ALL_TRAJ.map(function(poly) {{
                    return poly.map(function(tp) {{
                        return map.latLngToContainerPoint([tp.lat, tp.lon]);
                    }});
                }});

                // ── Draw tick marks ───────────────────────────────────────
                TICKS.forEach(function(tk) {{
                    var tp = map.latLngToContainerPoint([tk.lat, tk.lon]);
                    // Find nearest TRAJ point by geographic distance so the
                    // result is zoom-independent (screen distances compress at
                    // low zoom, causing the wrong segment to be selected).
                    var bestD2 = Infinity, bestI = 0;
                    for (var i = 0; i < TRAJ.length; i++) {{
                        var dlat = tk.lat - TRAJ[i].lat;
                        var dlon = tk.lon - TRAJ[i].lon;
                        var d2   = dlat*dlat + dlon*dlon;
                        if (d2 < bestD2) {{ bestD2 = d2; bestI = i; }}
                    }}
                    // Project the two neighbouring geographic points to screen
                    // to get the tangent in screen space.
                    var i0  = Math.min(bestI, TRAJ.length - 2);
                    var p0  = map.latLngToContainerPoint([TRAJ[i0].lat,   TRAJ[i0].lon]);
                    var p1  = map.latLngToContainerPoint([TRAJ[i0+1].lat, TRAJ[i0+1].lon]);
                    var tdx = p1.x - p0.x;
                    var tdy = p1.y - p0.y;
                    var tlen = Math.sqrt(tdx*tdx + tdy*tdy) || 1;
                    // Perpendicular unit vector (rotated 90°).
                    var px = -tdy / tlen, py = tdx / tlen;
                    var tl = document.createElementNS(
                        'http://www.w3.org/2000/svg', 'line');
                    tl.setAttribute('x1', tp.x - px * TICK_HALF);
                    tl.setAttribute('y1', tp.y - py * TICK_HALF);
                    tl.setAttribute('x2', tp.x + px * TICK_HALF);
                    tl.setAttribute('y2', tp.y + py * TICK_HALF);
                    tl.setAttribute('stroke', '#333');
                    tl.setAttribute('stroke-width', '1.5');
                    tl.setAttribute('opacity', '0.75');
                    _svg.appendChild(tl);
                }});

                // ── Label markers ─────────────────────────────────────────
                var pts = LABELS.map(function(lb) {{
                    return map.latLngToContainerPoint([lb.lat, lb.lon]);
                }});

                // ── Collision-detection helpers ───────────────────────────
                // Liang-Barsky segment / axis-aligned rect intersection.
                function segHitsRect(x1,y1,x2,y2,rx,ry,rw,rh) {{
                    var dx=x2-x1,dy=y2-y1;
                    var p=[-dx,dx,-dy,dy],q=[x1-rx,rx+rw-x1,y1-ry,ry+rh-y1];
                    var u0=0,u1=1;
                    for(var k=0;k<4;k++){{
                        if(p[k]===0){{if(q[k]<0)return false;}}
                        else{{var u=q[k]/p[k];if(p[k]<0)u0=Math.max(u0,u);else u1=Math.min(u1,u);}}
                        if(u0>u1)return false;
                    }}
                    return true;
                }}
                // Check rect against ALL trajectory polylines (main + debris arcs).
                function labelHitsTraj(lx,ly,lw,lh) {{
                    for(var p=0;p<allPts.length;p++){{
                        var ap=allPts[p];
                        for(var i=0;i<ap.length-1;i++){{
                            if(segHitsRect(ap[i].x,ap[i].y,ap[i+1].x,ap[i+1].y,
                                           lx,ly,lw,lh)) return true;
                        }}
                    }}
                    return false;
                }}
                // Check the GAP corridor between the dot and the label rectangle.
                // A trajectory can pass through the gap without entering the label
                // rect; this catches that case.  The immediate dot vicinity (DOT_R)
                // is excluded so the trajectory passing through the dot itself does
                // not generate a false positive.
                var DOT_R=8;
                function corridorHitsTraj(pt,lx,ly,lw,lh,lRight,above){{
                    var cx,cy,cw,ch;
                    if(above){{cy=ly;ch=pt.y-DOT_R-ly;}}
                    else{{cy=pt.y+DOT_R;ch=ly+lh-(pt.y+DOT_R);}}
                    if(lRight){{cx=pt.x+DOT_R;cw=lx+lw-(pt.x+DOT_R);}}
                    else{{cx=lx;cw=pt.x-DOT_R-lx;}}
                    if(cw<=0||ch<=0)return false;
                    return labelHitsTraj(cx,cy,cw,ch);
                }}
                function rectsOverlap(ax,ay,aw,ah,bx,by,bw,bh){{
                    return ax<bx+bw&&ax+aw>bx&&ay<by+bh&&ay+ah>by;
                }}

                // Vertical side: label above dot when more trajectory points
                // are below it (more traj below → label above), and vice-versa.
                function goAbovePt(pt){{
                    var bC=0,aC=0;
                    for(var i=0;i<tPts.length;i++){{
                        var dy=tPts[i].y-pt.y;
                        if(dy>8)bC++;else if(dy<-8)aC++;
                    }}
                    return bC>=aC;
                }}

                // Local trajectory tangent at the nearest skeleton point.
                function trajTan(pt){{
                    var best=Infinity,bi=0;
                    for(var i=0;i<tPts.length;i++){{
                        var dx=tPts[i].x-pt.x,dy=tPts[i].y-pt.y;
                        var d=dx*dx+dy*dy;
                        if(d<best){{best=d;bi=i;}}
                    }}
                    var i0=Math.max(0,bi-1),i1=Math.min(tPts.length-1,bi+1);
                    return {{dx:tPts[i1].x-tPts[i0].x,dy:tPts[i1].y-tPts[i0].y}};
                }}

                // ── Label placement ───────────────────────────────────────
                _divs.forEach(function(d){{d.style.display='none';}});

                var order=pts.map(function(_,i){{return i;}}).filter(function(i){{
                    var p=pts[i];
                    return p.x>=-EDGE&&p.x<=mapW+EDGE&&p.y>=-EDGE&&p.y<=mapH+EDGE;
                }});

                // Latest event placed first → sits nearest its dot.
                // Earlier events are pushed outward by the collision loop.
                // Reading order: chronological top-to-bottom (above) or
                //                bottom-to-top (below).
                order.sort(function(a,b){{return LABELS[b].t_s-LABELS[a].t_s;}});

                var topY={{}},lxMap={{}},goLeft={{}},goAbove={{}},lwCache={{}};
                var placed=[];   // already-positioned label rects

                // Horizontal side — proven safe from trajectory crossing:
                //   Label ABOVE dot: go RIGHT when slope ≥ 0, LEFT when < 0.
                //   Label BELOW dot: opposite.
                function sideFor(pt,above){{
                    var tan=trajTan(pt);
                    var slopePos=(tan.dx*tan.dy)>=0;
                    var lRight=above?slopePos:!slopePos;
                    var lx=lRight?pt.x+H_GAP:pt.x-H_GAP-lwCache[0]||80;
                    return {{lRight:lRight}};
                }}

                // Try to place a label on one vertical side (above=true/false).
                // Returns {{lx,ly,ok}} after up to 40 outward pushes.
                function tryPlace(pt,lw,lh,above){{
                    var tan=trajTan(pt);
                    var slopePos=(tan.dx*tan.dy)>=0;
                    var lRight=above?slopePos:!slopePos;
                    var lx=lRight?pt.x+H_GAP:pt.x-H_GAP-lw;
                    var ly=above?pt.y-lh-V_ABOVE:pt.y+V_ABOVE;
                    var dir=above?-1:1;
                    var step=lh+STACK_GAP;
                    for(var attempt=0;attempt<40;attempt++){{
                        // Check the label rect AND the gap corridor between dot
                        // and label — any trajectory in either region is a violation.
                        var bad=labelHitsTraj(lx,ly,lw,lh)||
                                corridorHitsTraj(pt,lx,ly,lw,lh,lRight,above);
                        for(var j=0;j<placed.length&&!bad;j++){{
                            bad=rectsOverlap(lx,ly,lw,lh,
                                placed[j].lx,placed[j].ly,
                                placed[j].lw,placed[j].lh);
                        }}
                        if(!bad)return{{lx:lx,ly:ly,lRight:lRight,ok:true}};
                        ly+=dir*step;
                    }}
                    return{{lx:lx,ly:ly,lRight:lRight,ok:false}};
                }}

                order.forEach(function(idx){{
                    var pt=pts[idx];
                    _divs[idx].style.display='block';
                    var lh=(_divs[idx].offsetHeight||14)+PAD*2;
                    var lw=(_divs[idx].offsetWidth ||80)+PAD*2;
                    lwCache[idx]=lw;

                    // prefer_above from Python (ascending/impact → true, descending → false).
                    // Falls back to counting trajectory points if not set.
                    var prefAbove=(LABELS[idx].prefer_above!==undefined)
                        ?LABELS[idx].prefer_above
                        :goAbovePt(pt);

                    // Try preferred side first; if exhausted, try the other side.
                    // This naturally splits two nearby debris impacts: the second
                    // one finds its preferred side blocked and flips, landing on
                    // the opposite edge of the dot cluster.
                    var r1=tryPlace(pt,lw,lh,prefAbove);
                    var best=r1;
                    if(!r1.ok){{
                        var r2=tryPlace(pt,lw,lh,!prefAbove);
                        if(r2.ok)best=r2;
                        // If both exhausted, keep preferred side's final position.
                    }}

                    topY[idx]=best.ly;
                    lxMap[idx]=best.lx;
                    goAbove[idx]=prefAbove;
                    goLeft[idx]=!best.lRight;
                    placed.push({{lx:best.lx,ly:best.ly,lw:lw,lh:lh}});
                }});

                // Render.
                order.forEach(function(idx){{
                    _divs[idx].style.left=lxMap[idx]+'px';
                    _divs[idx].style.top =topY[idx]+'px';
                }});
            }}

            var _poll = setInterval(function() {{
                var map = window["{map_var}"];
                if (map && map.getZoom) {{
                    clearInterval(_poll);
                    _init(map);
                    // Snap back only when the centre drifts well outside the
                    // trajectory extent (±90° margin).  This lets the user pan
                    // freely across the full track without a sudden jump.
                    var SNAP_MIN = {lon_uw_min:.3f} - 90;
                    var SNAP_MAX = {lon_uw_max:.3f} + 90;
                    map.on('moveend', function() {{
                        var c = this.getCenter(), lng = c.lng;
                        if (lng < SNAP_MIN || lng > SNAP_MAX) {{
                            this.setView(
                                [c.lat, ((lng % 360) + 540) % 360 - 180],
                                this.getZoom(), {{animate: false}});
                        }}
                    }}, map);
                    map.on('moveend zoomend', function() {{ _update(map); }});
                    _update(map);
                }}
            }}, 50);
        }})();
        </script>"""
        fmap.get_root().html.add_child(folium.Element(leader_js))

        # ── Logo overlay (lower-left, ~1/8–1/7 of viewport height) ──────
        import base64 as _b64, os as _os
        _logo_path = _os.path.join(_os.path.dirname(__file__), "data", "Thrusty.png")
        if _os.path.exists(_logo_path):
            with open(_logo_path, "rb") as _lf:
                _logo_b64 = _b64.b64encode(_lf.read()).decode()
            _logo_html = (
                '<img src="data:image/png;base64,' + _logo_b64 + '" '
                'style="position:fixed;bottom:12px;left:12px;'
                'height:35vh;width:auto;z-index:1000;pointer-events:none;" />'
            )
            fmap.get_root().html.add_child(folium.Element(_logo_html))

        fmap.save(path)
        import webbrowser
        webbrowser.open(f"file://{path}")
        self._status_var.set(f"Folium map saved and opened: {path}")

    def _export_timeline(self):
        """Export the flight event timeline to CSV."""
        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        milestones = self._result.get("milestones", [])
        if not milestones:
            messagebox.showinfo("No data", "No timeline events in last result.")
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export flight timeline",
        )
        if not path:
            return
        header = "event,time_s,alt_km,range_km,gnd_speed_kms,inrtl_speed_kms,accel_ms2,mass_t"
        rows = []
        for m in milestones:
            rows.append(",".join([
                f'"{m.get("event","")}"',
                f'{m.get("t_s", ""):g}',
                f'{m.get("alt_km", ""):g}',
                f'{m.get("range_km", ""):g}',
                f'{m.get("speed_kms", ""):g}',
                f'{m.get("inertial_speed_kms", ""):g}',
                f'{m.get("accel_ms2", ""):g}',
                f'{m.get("mass_t", ""):g}',
            ]))
        Path(path).write_text(header + "\n" + "\n".join(rows))
        self._status_var.set(f"Timeline exported: {path}")

    def _export_timeline_xlsx(self):
        """Export the flight event timeline to an XLSX workbook."""
        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        milestones = self._result.get("milestones", [])
        if not milestones:
            messagebox.showinfo("No data", "No timeline events in last result.")
            return
        try:
            from openpyxl import Workbook
        except ImportError as exc:
            messagebox.showerror("Missing dependency",
                                 f"openpyxl is required:\n{exc}")
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
            title="Export flight timeline XLSX",
        )
        if not path:
            return
        wb = Workbook()
        ws = wb.active
        ws.title = "Timeline"
        ws.append(["event", "time_s", "alt_km", "range_km",
                   "gnd_speed_kms", "inrtl_speed_kms",
                   "accel_ms2", "mass_t"])
        def _num(v):
            return float(v) if isinstance(v, (int, float)) else None
        for m in milestones:
            ws.append([
                m.get("event", ""),
                _num(m.get("t_s")),
                _num(m.get("alt_km")),
                _num(m.get("range_km")),
                _num(m.get("speed_kms")),
                _num(m.get("inertial_speed_kms")),
                _num(m.get("accel_ms2")),
                _num(m.get("mass_t")),
            ])
        wb.save(path)
        self._status_var.set(f"Timeline XLSX exported: {path}")

    def _export_missile(self):
        """Export the current missile definition to a .missile.json file."""
        name = self._missile_var.get()
        if not name or name not in MISSILE_DB:
            messagebox.showinfo("No missile", "Select a missile first.")
            return
        from tkinter.filedialog import asksaveasfilename
        _EXPORT_MISS_DIR.mkdir(parents=True, exist_ok=True)
        safe = name.replace(" ", "_").replace("/", "-")
        path = asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{safe}.missile.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Missile",
        )
        if not path:
            return
        data = missile_to_dict(MISSILE_DB[name]())
        Path(path).write_text(json.dumps(data, indent=2))
        self._status_var.set(f"Missile exported: {path}")

    def _export_missile_xlsx(self):
        """Export current missile to a filled-in XLSX template."""
        name = self._missile_var.get()
        if not name or name not in MISSILE_DB:
            messagebox.showinfo("No missile", "Select a missile first.", parent=self)
            return
        try:
            from missile_xlsx import export_missile_xlsx
        except ImportError as exc:
            messagebox.showerror("Missing dependency", str(exc), parent=self)
            return
        from tkinter.filedialog import asksaveasfilename
        safe = name.replace(" ", "_").replace("/", "-")
        path = asksaveasfilename(
            title="Export Missile to XLSX",
            defaultextension=".xlsx",
            initialfile=f"{safe}.xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            export_missile_xlsx(path, MISSILE_DB[name]())
            self._status_var.set(f"Missile exported: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Export error", str(exc), parent=self)

    def _import_missile_xlsx(self):
        """Import a missile from a filled XLSX template."""
        try:
            from missile_xlsx import import_missile_xlsx
        except ImportError as exc:
            messagebox.showerror("Missing dependency", str(exc), parent=self)
            return
        from tkinter.filedialog import askopenfilename
        path = askopenfilename(
            title="Import Missile from XLSX",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            params = import_missile_xlsx(path)
        except Exception as exc:
            messagebox.showerror("Import error", str(exc), parent=self)
            return
        if not params.name:
            messagebox.showwarning("Import warning",
                                   "Missile name is blank — please fill in "
                                   "the Name field in the XLSX and re-import.",
                                   parent=self)
            return
        if params.name in MISSILE_DB and not messagebox.askyesno(
                "Overwrite?", f"'{params.name}' already exists. Overwrite?",
                parent=self):
            return
        MISSILE_DB[params.name] = lambda p=params: p
        _save_custom_missiles()
        self._refresh_missile_list(select_name=params.name)
        self._status_var.set(f"Missile imported: {params.name}")

    def _new_missile_template(self):
        """Save a blank XLSX template the user fills in from scratch."""
        try:
            from missile_xlsx import make_blank_template
        except ImportError as exc:
            messagebox.showerror("Missing dependency", str(exc), parent=self)
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            title="Save Blank Missile Template",
            defaultextension=".xlsx",
            initialfile="missile_template.xlsx",
            filetypes=[("Excel workbook", "*.xlsx"), ("All files", "*.*")],
            parent=self,
        )
        if not path:
            return
        try:
            make_blank_template(path)
            self._status_var.set(f"Template saved: {os.path.basename(path)}")
        except Exception as exc:
            messagebox.showerror("Template error", str(exc), parent=self)

    def _load_missile(self):
        """Import a .missile.json file into the custom missile library."""
        from tkinter.filedialog import askopenfilename
        path = askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Missile",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            p    = missile_from_dict(data)
        except Exception as e:
            messagebox.showerror("Load error", f"Could not parse missile file:\n{e}")
            return
        name = data.get('name') or Path(path).stem.replace('.missile', '')
        if not name:
            messagebox.showerror("Load error", "Missile file has no name field.")
            return
        if name in MISSILE_DB and not messagebox.askyesno(
                "Overwrite?", f"'{name}' already exists. Overwrite?"):
            return
        MISSILE_DB[name] = lambda p=p: p
        _save_custom_missiles()
        self._refresh_missile_list(select_name=name)
        self._status_var.set(f"Missile '{name}' loaded from {Path(path).name}")

    def _export_site(self):
        """Export the current launch site to a .site.json file."""
        name = self._site_var.get()
        lat_s = self._launch_lat.get().strip()
        lon_s = self._launch_lon.get().strip()
        if not lat_s or not lon_s:
            messagebox.showinfo("No site", "Enter a launch site location first.")
            return
        from tkinter.filedialog import asksaveasfilename
        _EXPORT_SITE_DIR.mkdir(parents=True, exist_ok=True)
        safe = (name or "site").replace(" ", "_").replace("/", "-")
        path = asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{safe}.site.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export Launch Site",
        )
        if not path:
            return
        data = {"name": name, "lat": float(lat_s), "lon": float(lon_s)}
        Path(path).write_text(json.dumps(data, indent=2))
        self._status_var.set(f"Site exported: {path}")

    def _load_site(self):
        """Import a .site.json file into the custom launch-site library."""
        from tkinter.filedialog import askopenfilename
        path = askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Load Launch Site",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
            name = data['name']
            lat  = float(data['lat'])
            lon  = float(data['lon'])
        except Exception as e:
            messagebox.showerror("Load error", f"Could not parse site file:\n{e}")
            return
        user_sites = _load_user_sites()
        if any(s['name'] == name for s in user_sites):
            if not messagebox.askyesno("Overwrite?",
                                       f"'{name}' already exists. Overwrite?"):
                return
            user_sites = [s for s in user_sites if s['name'] != name]
        user_sites.append({"name": name, "lat": lat, "lon": lon})
        _save_user_sites(user_sites)
        new_values, new_map = _load_launch_sites()
        self._site_map = new_map
        self._site_cb.config(values=new_values)
        self._site_var.set(name)
        self._launch_lat.set(f"{lat:.4f}")
        self._launch_lon.set(f"{lon:.4f}")
        self._status_var.set(f"Site '{name}' loaded from {Path(path).name}")

    def _show_about(self):
        messagebox.showinfo(
            "About Thrusty",
            "Thrusty\n\n"
            "Based on the MATLAB application by Geoffrey Forden\n"
            "G. Forden, Science & Global Security 15 (2007)\n\n"
            "3-DOF trajectory integration:\n"
            "  • COESA 1976 standard atmosphere\n"
            "  • WGS-84 J2 gravity (ECEF)\n"
            "  • Coriolis & centrifugal corrections\n"
            "  • Gravity-turn guidance with per-stage pitch profiles\n"
            "  • Up to 4 stages with inter-stage coast\n\n"
            "Packaged missiles (Forden Table 1 + extension):\n"
            "  Scud-B, Al Hussein, No-dong,\n"
            "  Taepodong-I, Taepodong-II (3-stage),\n"
            "  Shahab-3, Generic ICBM\n"
        )


# ---------------------------------------------------------------------------
def main():
    app = MissileFlyoutApp()
    app.mainloop()


if __name__ == "__main__":
    main()
