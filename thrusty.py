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
                           NOSE_SHAPES, NOSE_SHAPE_LABELS)
from trajectory import (integrate_trajectory, maximize_range, aim_missile,
                        plan_orbital_insertion)
from coordinates import range_between
from slv_performance import schilling_performance

# ---------------------------------------------------------------------------
# Country border map data (Natural Earth 110m, bundled GeoJSON)
# ---------------------------------------------------------------------------

_BORDERS_CACHE = None   # loaded once on first draw, then reused

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
_CUSTOM_PATH = Path.home() / ".gui_missile_flyout" / "custom_missiles.json"
# Where per-missile trajectory profiles are saved
_TRAJ_PATH = Path.home() / ".gui_missile_flyout" / "trajectory_profiles.json"


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
        ttk.Label(self, text="Thrust (kN):").grid(
            row=4, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._thrust_kn = tk.StringVar(value=d["thrust_kn"])
        _thr_inner = ttk.Frame(self)
        _thr_inner.grid(row=4, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        ttk.Entry(_thr_inner, textvariable=self._thrust_kn, width=10).pack(side=tk.LEFT)
        ttk.Label(_thr_inner, text="kN").pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(_thr_inner, text="Suggest…",
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

        # Burn time — read-only computed field (row 7)
        ttk.Label(self, text="Burn time (s):").grid(
            row=7, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._burn_var = tk.StringVar()
        _burn_inner = ttk.Frame(self)
        _burn_inner.grid(row=7, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._burn_entry = ttk.Entry(_burn_inner, textvariable=self._burn_var,
                                     width=10, state="readonly")
        self._burn_entry.pack(side=tk.LEFT)
        ttk.Label(_burn_inner, text="s  (computed)").pack(side=tk.LEFT, padx=(2, 0))

        # Coast-time row (row 8) — shown only for non-last stages
        self._coast_var = tk.StringVar(value=d["coast"])
        self._coast_lbl = ttk.Label(self, text="Coast after (s):")
        self._coast_lbl.grid(row=8, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        coast_inner = ttk.Frame(self)
        coast_inner.grid(row=8, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        ttk.Entry(coast_inner, textvariable=self._coast_var, width=10).pack(side=tk.LEFT)
        ttk.Label(coast_inner, text="s  (0 = instant ignition)").pack(
            side=tk.LEFT, padx=(2, 0))
        self._coast_inner = coast_inner
        # Hidden by default; _update_stage_frames() reveals it for non-last stages
        self._coast_lbl.grid_remove()
        self._coast_inner.grid_remove()

        # Solid motor checkbox (row 9)
        self._solid_motor_var = tk.BooleanVar(value=False)
        self._solid_motor_check = ttk.Checkbutton(
            self, text="Solid rocket motor (cannot be shut off)",
            variable=self._solid_motor_var)
        self._solid_motor_check.grid(
            row=9, column=0, columnspan=2, sticky=tk.W, padx=(6, 2), pady=(2, 4))

        # Recompute burn whenever any of the four driving fields change
        for _v in (self._fueled, self._dry, self._thrust_kn, self._isp):
            _v.trace_add("write", self._recompute_burn)
        self._recompute_burn()

    def _recompute_burn(self, *_):
        """Compute burn time = Isp × g₀ × prop / thrust and update the display."""
        try:
            prop     = float(self._fueled.get()) - float(self._dry.get())
            thrust_n = float(self._thrust_kn.get()) * 1000.0
            isp      = float(self._isp.get())
            if thrust_n <= 0 or isp <= 0 or prop <= 0:
                raise ValueError
            self._burn_var.set(f"{isp * self._G0 * prop / thrust_n:.1f}")
        except (ValueError, ZeroDivisionError):
            self._burn_var.set("—")

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
        for entry in self._iter_entries(self):
            if entry is not self._burn_entry:   # burn time is always readonly
                entry.config(state=state)

    def set_coast_visible(self, visible: bool):
        """Show or hide the inter-stage coast-time row."""
        if visible:
            self._coast_lbl.grid()
            self._coast_inner.grid()
        else:
            self._coast_lbl.grid_remove()
            self._coast_inner.grid_remove()

    def get(self):
        burn_str = self._burn_var.get()
        if burn_str == "—":
            raise ValueError("Burn time could not be computed — check thrust, Isp, and masses.")
        _LABELS = {
            "fueled": "Fueled Mass", "dry": "Dry Mass", "dia": "Diameter",
            "length": "Length", "thrust_kn": "Thrust", "isp": "Isp",
            "nozzle_area": "Nozzle Exit Area", "coast": "Coast Time",
        }
        result = {}
        for k, v in [
            ("fueled",      self._fueled),      ("dry",         self._dry),
            ("dia",         self._dia),         ("length",      self._length),
            ("thrust_kn",   self._thrust_kn),   ("isp",         self._isp),
            ("nozzle_area", self._nozzle_area), ("coast",       self._coast_var),
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
        result["solid_motor"] = bool(self._solid_motor_var.get())
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
        # _burn_var is updated automatically by the trace
        self._coast_var   .set(str(d.get("coast", 0)))
        self._solid_motor_var.set(bool(d.get("solid_motor", False)))


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

        # ── Payload panel — grid layout matching stage entries ───────────────
        pl = ttk.LabelFrame(body, text="Front End")
        pl.pack(fill=tk.X, padx=8, pady=4)

        # Track payload input widgets so _apply_readonly can disable them.
        self._payload_inputs = []

        # Row 0: PBV mass
        self._bus_var = _entry_row(pl, "PBV mass (kg):", 0, "0", "kg")
        self._payload_inputs.append(pl.winfo_children()[-1].winfo_children()[0])

        # Row 1: Number of RVs (spinbox)
        ttk.Label(pl, text="No. of RVs:").grid(
            row=1, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._num_rvs_var = tk.StringVar(value="1")
        _rvn_inner = ttk.Frame(pl)
        _rvn_inner.grid(row=1, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._num_rvs_spinbox = ttk.Spinbox(_rvn_inner, textvariable=self._num_rvs_var,
                                            from_=1, to=24, width=4)
        self._num_rvs_spinbox.pack(side=tk.LEFT)
        self._payload_inputs.append(self._num_rvs_spinbox)

        # Row 2: Per-RV mass + computed total (in the unit label position)
        ttk.Label(pl, text="Per-RV mass (kg):").grid(
            row=2, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._rv_mass_var = tk.StringVar(value="1000")
        _rvm_inner = ttk.Frame(pl)
        _rvm_inner.grid(row=2, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        _rv_mass_entry = ttk.Entry(_rvm_inner, textvariable=self._rv_mass_var, width=10)
        _rv_mass_entry.pack(side=tk.LEFT)
        self._payload_inputs.append(_rv_mass_entry)
        self._total_payload_lbl = ttk.Label(_rvm_inner, text="kg  = 1000 total",
                                            foreground="gray40")
        self._total_payload_lbl.pack(side=tk.LEFT, padx=(2, 0))

        # Row 3: RV ballistic coefficient + Calculate button
        ttk.Label(pl, text="RV β (kg/m²):").grid(
            row=3, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._rv_beta_var = tk.StringVar(value="0")
        _beta_inner = ttk.Frame(pl)
        _beta_inner.grid(row=3, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        _beta_entry = ttk.Entry(_beta_inner, textvariable=self._rv_beta_var, width=10)
        _beta_entry.pack(side=tk.LEFT)
        ttk.Label(_beta_inner, text="kg/m²").pack(side=tk.LEFT, padx=(2, 6))
        ttk.Button(_beta_inner, text="Calculate…",
                   command=self._calc_rv_beta).pack(side=tk.LEFT)
        self._payload_inputs.append(_beta_entry)

        # Row 4: Warhead / RV separates checkbox
        self._rv_separates_var = tk.BooleanVar(value=False)
        self._rv_separates_check = ttk.Checkbutton(
            pl, text="Warhead / RV separates from stage at burnout",
            variable=self._rv_separates_var)
        self._rv_separates_check.grid(
            row=4, column=0, columnspan=2, sticky=tk.W, padx=(6, 2), pady=2)

        # Row 5: Payload nose shape — shape of the exposed tip after shroud jettison
        # (or throughout flight if no shroud is fitted)
        _ns_labels = list(NOSE_SHAPE_LABELS.values())
        ttk.Label(pl, text="Payload nose shape:").grid(
            row=5, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._nose_shape_var = tk.StringVar(value=NOSE_SHAPE_LABELS["forden"])
        self._nose_shape_cb = ttk.Combobox(
            pl, textvariable=self._nose_shape_var,
            values=_ns_labels, state="readonly", width=18)
        self._nose_shape_cb.grid(row=5, column=1, sticky=tk.W, padx=(0, 6), pady=2)

        # Row 6: Payload nose length
        ttk.Label(pl, text="Payload nose length (m):").grid(
            row=6, column=0, sticky=tk.W, padx=(6, 2), pady=(2, 6))
        self._nose_length_var = tk.StringVar(value="0")
        _nld_inner = ttk.Frame(pl)
        _nld_inner.grid(row=6, column=1, sticky=tk.W, padx=(0, 6), pady=(2, 6))
        self._nose_length_entry = ttk.Entry(
            _nld_inner, textvariable=self._nose_length_var, width=10)
        self._nose_length_entry.pack(side=tk.LEFT)
        ttk.Label(_nld_inner, text="m").pack(side=tk.LEFT, padx=(2, 0))

        # Row 7: Shroud — checkbutton acts as label; entry starts disabled
        self._shroud_var = tk.BooleanVar(value=False)
        self._shroud_check = ttk.Checkbutton(pl, text="Shroud mass (kg):",
                                             variable=self._shroud_var,
                                             command=self._update_shroud_state)
        self._shroud_check.grid(row=7, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._shroud_mass_var = tk.StringVar(value="0")
        _sm_inner = ttk.Frame(pl)
        _sm_inner.grid(row=7, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._shroud_mass_entry = ttk.Entry(
            _sm_inner, textvariable=self._shroud_mass_var, width=10, state="disabled")
        self._shroud_mass_entry.pack(side=tk.LEFT)
        ttk.Label(_sm_inner, text="kg").pack(side=tk.LEFT, padx=(2, 0))

        # Row 8: Jettison altitude (indented label to signal dependence on shroud)
        ttk.Label(pl, text="  Jettison alt (km):").grid(
            row=8, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._shroud_alt_var = tk.StringVar(value="80")
        _sa_inner = ttk.Frame(pl)
        _sa_inner.grid(row=8, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._shroud_alt_entry = ttk.Entry(
            _sa_inner, textvariable=self._shroud_alt_var, width=10, state="disabled")
        self._shroud_alt_entry.pack(side=tk.LEFT)
        ttk.Label(_sa_inner, text="km").pack(side=tk.LEFT, padx=(2, 0))

        # Row 9: Shroud length — used to compute tumbling-cylinder debris β
        ttk.Label(pl, text="  Shroud length (m):").grid(
            row=9, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._shroud_length_var = tk.StringVar(value="0")
        _sl_inner = ttk.Frame(pl)
        _sl_inner.grid(row=9, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._shroud_length_entry = ttk.Entry(
            _sl_inner, textvariable=self._shroud_length_var, width=10, state="disabled")
        self._shroud_length_entry.pack(side=tk.LEFT)
        ttk.Label(_sl_inner, text="m").pack(side=tk.LEFT, padx=(2, 0))

        # Row 10: Shroud diameter — for reference area correction pre-jettison
        ttk.Label(pl, text="  Shroud diameter (m):").grid(
            row=10, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._shroud_diameter_var = tk.StringVar(value="0")
        _sd_inner = ttk.Frame(pl)
        _sd_inner.grid(row=10, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        self._shroud_diameter_entry = ttk.Entry(
            _sd_inner, textvariable=self._shroud_diameter_var, width=10, state="disabled")
        self._shroud_diameter_entry.pack(side=tk.LEFT)
        ttk.Label(_sd_inner, text="m").pack(side=tk.LEFT, padx=(2, 0))

        # Row 11: Shroud nose shape dropdown
        ttk.Label(pl, text="  Shroud nose shape:").grid(
            row=11, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._shroud_nose_shape_var = tk.StringVar(value=NOSE_SHAPE_LABELS["forden"])
        self._shroud_nose_shape_cb = ttk.Combobox(
            pl, textvariable=self._shroud_nose_shape_var,
            values=_ns_labels, state="disabled", width=18)
        self._shroud_nose_shape_cb.grid(row=11, column=1, sticky=tk.W, padx=(0, 6), pady=2)

        # Row 12: Shroud nose length (L/D computed internally)
        ttk.Label(pl, text="  Shroud nose length (m):").grid(
            row=12, column=0, sticky=tk.W, padx=(6, 2), pady=(2, 6))
        self._shroud_nose_length_var = tk.StringVar(value="0")
        _snld_inner = ttk.Frame(pl)
        _snld_inner.grid(row=12, column=1, sticky=tk.W, padx=(0, 6), pady=(2, 6))
        self._shroud_nose_length_entry = ttk.Entry(
            _snld_inner, textvariable=self._shroud_nose_length_var, width=10, state="disabled")
        self._shroud_nose_length_entry.pack(side=tk.LEFT)
        ttk.Label(_snld_inner, text="m").pack(side=tk.LEFT, padx=(2, 0))

        # Live total-payload label update
        for _v in (self._bus_var, self._num_rvs_var, self._rv_mass_var):
            _v.trace_add("write", self._update_total_payload)

        # ── Guidance mode ────────────────────────────────────────────────
        gf = ttk.LabelFrame(body, text="Guidance Mode")
        gf.pack(fill=tk.X, padx=8, pady=4)
        self._guidance_var = tk.StringVar(value="loft")
        ttk.Radiobutton(gf, text="Forden Loft (SRBM / MRBM)",
                        variable=self._guidance_var, value="loft").pack(
            anchor=tk.W, padx=8, pady=(4, 0))
        ttk.Radiobutton(gf, text="Gravity Turn (IRBM / ICBM)",
                        variable=self._guidance_var, value="gravity_turn").pack(
            anchor=tk.W, padx=8, pady=(0, 2))
        ttk.Radiobutton(gf, text="Orbital Insertion",
                        variable=self._guidance_var, value="orbital_insertion").pack(
            anchor=tk.W, padx=8, pady=(0, 4))

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
        for w in self._payload_inputs:
            w.config(state="readonly")
        self._shroud_check.config(state="disabled")
        self._shroud_mass_entry.config(state="disabled")
        self._shroud_alt_entry.config(state="disabled")
        self._shroud_length_entry.config(state="disabled")
        self._shroud_diameter_entry.config(state="disabled")
        self._shroud_nose_shape_cb.config(state="disabled")
        self._shroud_nose_length_entry.config(state="disabled")
        self._nose_shape_cb.config(state="disabled")
        self._nose_length_entry.config(state="disabled")
        self._rv_separates_check.config(state="disabled")
        self._save_btn.pack_forget()
        self._save_as_btn.pack_forget()

    # ------------------------------------------------------------------
    def _update_total_payload(self, *_):
        """Recompute and display the total payload label."""
        try:
            bus = float(self._bus_var.get())
            n   = max(1, int(self._num_rvs_var.get()))
            rv  = float(self._rv_mass_var.get())
            self._total_payload_lbl.config(text=f"kg  = {bus + n * rv:.0f} total")
        except (ValueError, tk.TclError):
            self._total_payload_lbl.config(text="kg  = ? total")

    def _update_shroud_state(self):
        """Enable/disable shroud mass, altitude, length, diameter and nose widgets."""
        state = "normal" if self._shroud_var.get() else "disabled"
        cb_state = "readonly" if self._shroud_var.get() else "disabled"
        self._shroud_mass_entry.config(state=state)
        self._shroud_alt_entry.config(state=state)
        self._shroud_length_entry.config(state=state)
        self._shroud_diameter_entry.config(state=state)
        self._shroud_nose_shape_cb.config(state=cb_state)
        self._shroud_nose_length_entry.config(state=state)

    # ------------------------------------------------------------------
    def _update_stage_frames(self):
        """Show the right number of stage frames and coast-time rows."""
        n = int(self._n_stages_var.get())
        pad = dict(padx=8, pady=4)
        for i, sf in enumerate(self._stage_frames):
            if i < n:
                sf.pack(fill=tk.X, **pad)
                # Coast row visible only for non-last stages
                sf.set_coast_visible(i < n - 1)
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
                "fueled":      fueled,                   "dry":         dry,
                "dia":         node.diameter_m,          "length":      node.length_m,
                "burn":        node.burn_time_s,         "isp":         node.isp_s,
                "nozzle_area": node.nozzle_exit_area_m2, "coast":       node.coast_time_s,
                "solid_motor": getattr(node, 'solid_motor', False),
            })
            node = nxt
            stage_idx += 1

        n = len(stage_data)
        self._n_stages_var.set(str(n))
        self._update_stage_frames()
        for i, sd in enumerate(stage_data):
            self._stage_frames[i].populate(sd)

        # Payload decomposition
        if p.rv_mass_kg > 0:
            self._bus_var.set(f"{p.bus_mass_kg:.0f}")
            self._num_rvs_var.set(str(p.num_rvs))
            self._rv_mass_var.set(f"{p.rv_mass_kg:.0f}")
        else:
            # Old-style missile: treat entire payload as a single RV
            self._bus_var.set("0")
            self._num_rvs_var.set("1")
            self._rv_mass_var.set(f"{payload:.0f}")
        self._rv_beta_var.set(f"{p.rv_beta_kg_m2:.0f}")

        # Shroud
        has_shroud = shroud_mass > 0
        self._shroud_var.set(has_shroud)
        self._shroud_mass_var.set(f"{shroud_mass:.0f}")
        self._shroud_alt_var.set(f"{p.shroud_jettison_alt_km:.0f}")
        self._shroud_length_var.set(f"{p.shroud_length_m:.1f}")
        self._shroud_diameter_var.set(f"{p.shroud_diameter_m:.2f}")
        self._shroud_nose_shape_var.set(
            NOSE_SHAPE_LABELS.get(p.shroud_nose_shape, NOSE_SHAPE_LABELS["forden"]))
        self._shroud_nose_length_var.set(f"{p.shroud_nose_length_m:.2f}")
        self._update_shroud_state()

        # Body nose shape
        self._nose_shape_var.set(
            NOSE_SHAPE_LABELS.get(p.nose_shape, NOSE_SHAPE_LABELS["forden"]))
        self._nose_length_var.set(f"{p.nose_length_m:.2f}")

        # RV / warhead separates flag
        self._rv_separates_var.set(p.rv_separates)

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

        # Payload decomposition
        try:
            bus_mass = float(self._bus_var.get())
            num_rvs  = max(1, int(self._num_rvs_var.get()))
            rv_mass  = float(self._rv_mass_var.get())
        except ValueError:
            raise ValueError("PBV and per-RV mass must be numbers.")
        payload = bus_mass + num_rvs * rv_mass
        rv_beta      = float(self._rv_beta_var.get())
        rv_separates = self._rv_separates_var.get()

        # Shroud
        shroud_mass          = 0.0
        shroud_alt_km        = 80.0
        shroud_length_m      = 0.0
        shroud_diameter_m    = 0.0
        shroud_nose_shape    = "forden"
        shroud_nose_length_m = 0.0
        if self._shroud_var.get():
            try:
                shroud_mass       = float(self._shroud_mass_var.get())
                shroud_alt_km     = float(self._shroud_alt_var.get())
                shroud_length_m   = float(self._shroud_length_var.get())
                shroud_diameter_m = float(self._shroud_diameter_var.get())
                shroud_nose_length_m = float(self._shroud_nose_length_var.get())
            except ValueError:
                raise ValueError("Shroud mass, jettison altitude, length, diameter, and nose length must be numbers.")
            _label = self._shroud_nose_shape_var.get()
            shroud_nose_shape = next(
                (k for k, v in NOSE_SHAPE_LABELS.items() if v == _label), "forden")

        # Body nose shape
        _body_ns_label = self._nose_shape_var.get()
        nose_shape    = next(
            (k for k, v in NOSE_SHAPE_LABELS.items() if v == _body_ns_label), "forden")
        try:
            nose_length_m = float(self._nose_length_var.get())
        except ValueError:
            raise ValueError("Nose length must be a number.")

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
            )

        node.name              = name
        node.guidance          = self._guidance_var.get()
        node.payload_kg        = payload
        node.rv_beta_kg_m2     = rv_beta
        node.bus_mass_kg       = bus_mass
        node.num_rvs           = num_rvs
        node.rv_mass_kg        = rv_mass
        node.rv_separates           = rv_separates
        node.shroud_mass_kg         = shroud_mass
        node.shroud_jettison_alt_km = shroud_alt_km
        node.shroud_length_m        = shroud_length_m
        node.shroud_diameter_m      = shroud_diameter_m
        node.shroud_nose_shape      = shroud_nose_shape
        node.shroud_nose_length_m   = shroud_nose_length_m
        node.nose_shape             = nose_shape
        node.nose_length_m          = nose_length_m
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

        # ── Pre-fill half-angle from payload nose length / diameter ───
        try:
            _nose_len = float(self._nose_length_var.get())
            _dia0     = float(self._stage_frames[0]._dia.get())
            if _nose_len > 0 and _dia0 > 0:
                _ld0 = _nose_len / _dia0
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
# Parametric sweep / sensitivity-analysis dialog
# ---------------------------------------------------------------------------

class ParametricSweepDialog(tk.Toplevel):
    """Non-modal dialog for 1-D parametric trajectory sweep.

    Reproduces the analyses Forden performs in all three worked examples:
      • Table 2  — Range vs azimuth (vary azimuth, fixed loft / cutoff)
      • Figure 7 — Range vs loft angle (vary loft_angle_deg)
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
            (missile, guidance, lat, lon, az, cutoff, la, lar,
             gt_start_s, gt_stop_s, *_) = self._app._get_inputs()
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
            args=(missile, guidance, lat, lon, az, la, lar, cutoff,
                  param_key, points, overplot, gt_start_s, gt_stop_s),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    def _sweep_worker(self, missile, guidance, lat, lon, az, la, lar, cutoff,
                      param_key, points, store_trajs, gt_start_s=5.0, gt_stop_s=None):
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
                    loft_angle_deg=run_la,
                    loft_angle_rate_deg_s=lar,
                    cutoff_time_s=run_cut,
                    gt_turn_start_s=gt_start_s,
                    gt_turn_stop_s=run_gt_stop,
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
        self.geometry("1340x820")
        self.minsize(900, 620)

        self._result         = None
        self._running        = False
        self._notam_overlay  = None   # list of GeoJSON-style polygon rings, or None

        _load_custom_missiles()      # restore any user-saved missiles

        self._build_menu()
        self._build_ui()
        self._on_missile_changed()   # populate params tab with default missile

    # ------------------------------------------------------------------
    # Menu
    # ------------------------------------------------------------------
    def _build_menu(self):
        menubar = tk.Menu(self)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Export trajectory CSV…", command=self._save_trajectory)
        file_menu.add_command(label="Export trajectory KML…", command=self._export_kml)
        file_menu.add_command(label="Open Folium map…",       command=self._export_folium)
        file_menu.add_command(label="Export timeline CSV…",   command=self._export_timeline)
        file_menu.add_command(label="Export figures…",        command=self._export_figures)
        file_menu.add_command(label="Export missile JSON…",   command=self._export_missile)
        file_menu.add_separator()
        file_menu.add_command(label="Load NOTAM overlay…",    command=self._load_notam_overlay)
        file_menu.add_command(label="Clear NOTAM overlay",    command=self._clear_notam_overlay)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Parametric Sweep…",        command=self._open_sweep)
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
        LEFT_W = 350
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

        self._guidance_var = tk.StringVar(value="loft")
        gmode_frame = ttk.Frame(gf)
        gmode_frame.grid(row=0, column=0, columnspan=2, sticky=tk.W,
                         padx=6, pady=(4, 2))
        ttk.Radiobutton(gmode_frame, text="Forden Loft",
                        variable=self._guidance_var, value="loft",
                        command=self._on_guidance_changed).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(gmode_frame, text="Gravity Turn",
                        variable=self._guidance_var, value="gravity_turn",
                        command=self._on_guidance_changed).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(gmode_frame, text="Orbital Insertion",
                        variable=self._guidance_var, value="orbital_insertion",
                        command=self._on_guidance_changed).pack(side=tk.LEFT, padx=4)

        self._loft_angle_lbl = ttk.Label(gf, text="Loft Angle:")
        self._loft_angle_lbl.grid(row=1, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        la_frame = ttk.Frame(gf)
        la_frame.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._loft_angle_var = tk.StringVar(value="45.0")
        ttk.Entry(la_frame, textvariable=self._loft_angle_var, width=8).pack(side=tk.LEFT)
        self._loft_angle_unit_lbl = ttk.Label(la_frame, text="°  (final elev.)")
        self._loft_angle_unit_lbl.pack(side=tk.LEFT, padx=2)

        self._loft_rate_lbl = ttk.Label(gf, text="Loft Rate:")
        self._loft_rate_lbl.grid(row=2, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        lr_frame = ttk.Frame(gf)
        lr_frame.grid(row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._loft_rate_frame = lr_frame
        self._loft_rate_var = tk.StringVar(value="2.0")
        self._loft_rate_entry = ttk.Entry(lr_frame, textvariable=self._loft_rate_var, width=8)
        self._loft_rate_entry.pack(side=tk.LEFT)
        self._loft_rate_unit_lbl = ttk.Label(lr_frame, text="°/s")
        self._loft_rate_unit_lbl.pack(side=tk.LEFT, padx=2)

        self._gt_turn_start_lbl = ttk.Label(gf, text="Turn Start:")
        self._gt_turn_start_lbl.grid(row=3, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        gt_ts_frame = ttk.Frame(gf)
        gt_ts_frame.grid(row=3, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._gt_turn_start_frame = gt_ts_frame
        self._gt_turn_start_var = tk.StringVar(value="5.0")
        ttk.Entry(gt_ts_frame, textvariable=self._gt_turn_start_var, width=8).pack(side=tk.LEFT)
        ttk.Label(gt_ts_frame, text="s").pack(side=tk.LEFT, padx=2)

        self._gt_turn_stop_lbl = ttk.Label(gf, text="Turn Stop:")
        self._gt_turn_stop_lbl.grid(row=4, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        gt_te_frame = ttk.Frame(gf)
        gt_te_frame.grid(row=4, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._gt_turn_stop_frame = gt_te_frame
        self._gt_turn_stop_var = tk.StringVar(value="")
        ttk.Entry(gt_te_frame, textvariable=self._gt_turn_stop_var, width=8).pack(side=tk.LEFT)
        ttk.Label(gt_te_frame, text="s  (blank = full burn)").pack(side=tk.LEFT, padx=2)

        self._orbit_alt_lbl = ttk.Label(gf, text="Target orbit alt:")
        self._orbit_alt_lbl.grid(row=5, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        orb_frame = ttk.Frame(gf)
        orb_frame.grid(row=5, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._orbit_alt_frame = orb_frame
        self._orbit_alt_var = tk.StringVar(value="400")
        ttk.Entry(orb_frame, textvariable=self._orbit_alt_var, width=8).pack(side=tk.LEFT)
        ttk.Label(orb_frame, text="km").pack(side=tk.LEFT, padx=2)

        # Row 6: Plan Orbit button — placed directly in gf (no wrapper Frame)
        # so grid_forget/grid reliably shows and hides it.
        self._plan_orbit_btn = ttk.Button(gf, text="Plan Orbit",
                                          command=self._plan_orbit)
        self._plan_orbit_btn.grid(row=6, column=0, columnspan=2,
                                  sticky=tk.EW, padx=8, pady=(4, 6), ipadx=2, ipady=4)

        # Row 7: Advanced pitch program toggle (gravity_turn / orbital_insertion only)
        self._adv_pitch_var = tk.BooleanVar(value=False)
        self._adv_pitch_chk = ttk.Checkbutton(
            gf, text="Advanced pitch program (per-stage)",
            variable=self._adv_pitch_var,
            command=self._on_adv_pitch_toggled)
        self._adv_pitch_chk.grid(row=7, column=0, columnspan=2,
                                  sticky=tk.W, padx=8, pady=(0, 2))

        # Row 8: Per-stage inline rows — rebuilt whenever missile changes
        self._adv_pitch_frame = ttk.Frame(gf)
        self._stage_rows = []   # list of dicts with StringVars per stage

        # Row 9: Yaw / dogleg program toggle (gravity_turn / orbital_insertion only)
        self._adv_yaw_var = tk.BooleanVar(value=False)
        self._adv_yaw_chk = ttk.Checkbutton(
            gf, text="Yaw / dogleg program",
            variable=self._adv_yaw_var,
            command=self._on_adv_yaw_toggled)

        # Row 10: Global yaw fields — shown when checkbox enabled
        yf = ttk.Frame(gf)
        self._adv_yaw_frame = yf
        self._yaw_start_var    = tk.StringVar(value="")
        self._yaw_stop_var     = tk.StringVar(value="")
        self._yaw_final_az_var = tk.StringVar(value="")
        yf.columnconfigure(1, weight=1)
        for _yr, _lbl, _sv, _unit in [
                (0, "Yaw start:",  self._yaw_start_var,    "s"),
                (1, "Yaw end:",    self._yaw_stop_var,     "s"),
                (2, "Final az:",   self._yaw_final_az_var, "°")]:
            ttk.Label(yf, text=_lbl).grid(
                row=_yr, column=0, sticky=tk.W, padx=(8, 2), pady=1)
            ttk.Entry(yf, textvariable=_sv, width=8).grid(
                row=_yr, column=1, sticky=tk.W, padx=2, pady=1)
            ttk.Label(yf, text=_unit).grid(
                row=_yr, column=2, sticky=tk.W, padx=(2, 8), pady=1)

        # Row 11: Reset trajectory button — always visible
        self._reset_traj_btn = ttk.Button(
            gf, text="Reset trajectory to defaults",
            command=self._reset_traj_profile)
        self._reset_traj_btn.grid(row=11, column=0, columnspan=2,
                                  sticky=tk.EW, padx=8, pady=(4, 6))

        # Hide turn-start/stop, orbit-alt, plan-orbit, adv-pitch, and yaw rows by
        # default (loft mode is active at startup)
        self._gt_turn_start_lbl.grid_forget()
        self._gt_turn_start_frame.grid_forget()
        self._gt_turn_stop_lbl.grid_forget()
        self._gt_turn_stop_frame.grid_forget()
        self._orbit_alt_lbl.grid_forget()
        self._orbit_alt_frame.grid_forget()
        self._plan_orbit_btn.grid_forget()
        self._adv_pitch_chk.grid_forget()
        self._adv_yaw_chk.grid_forget()

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
            self._loft_angle_var.set(str(prof.get('loft_angle_deg', p.loft_angle_deg)))
            self._loft_rate_var.set(str(prof.get('loft_angle_rate_deg_s', p.loft_angle_rate_deg_s)))
            gt_start = prof.get('gt_turn_start_s', 5.0)
            self._gt_turn_start_var.set(str(gt_start) if gt_start else "5.0")
            gt_stop = prof.get('gt_turn_stop_s')
            self._gt_turn_stop_var.set(str(gt_stop) if gt_stop is not None else "")
            cutoff = prof.get('cutoff_time_s')
            self._cutoff_var.set(str(int(cutoff)) if cutoff is not None
                                 else str(int(total_burn_time(p))))
        else:
            self._cutoff_var.set(str(int(total_burn_time(p))))
            self._loft_angle_var.set(f"{p.loft_angle_deg:.4f}")
            self._loft_rate_var.set(f"{p.loft_angle_rate_deg_s:.3f}")
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
            (4, ""),                  # separator column
            (5, "Yaw start (s)"),
            (6, "Yaw end (s)"),
            (7, "Final az (°)"),
            (8, "Burn window"),
        ]
        for col, hdr in _headers:
            ttk.Label(af, text=hdr, foreground="#555555").grid(
                row=0, column=col, padx=(8 if col == 0 else 4, 4),
                pady=(4, 1), sticky=tk.W)
        # Thin visual separator between pitch and yaw columns
        ttk.Separator(af, orient=tk.VERTICAL).grid(
            row=0, column=4, rowspan=len(stages) + 1,
            sticky=tk.NS, padx=4, pady=2)

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

            # Yaw fields — empty by default; populate from node if set
            sv_yaw_start  = tk.StringVar(
                value=f"{node.stage_yaw_start_s:.1f}"
                      if node.stage_yaw_start_s  is not None else "")
            sv_yaw_stop   = tk.StringVar(
                value=f"{node.stage_yaw_stop_s:.1f}"
                      if node.stage_yaw_stop_s   is not None else "")
            sv_yaw_final  = tk.StringVar(
                value=f"{node.stage_yaw_final_az_deg:.1f}"
                      if node.stage_yaw_final_az_deg is not None else "")

            row = i + 1
            ttk.Label(af, text=f"Stage {i+1}:").grid(
                row=row, column=0, sticky=tk.W, padx=(8, 4), pady=1)
            ttk.Entry(af, textvariable=sv_start, width=7).grid(
                row=row, column=1, padx=4, pady=1)
            ttk.Entry(af, textvariable=sv_stop,  width=7).grid(
                row=row, column=2, padx=4, pady=1)
            ttk.Entry(af, textvariable=sv_angle, width=7).grid(
                row=row, column=3, padx=4, pady=1)
            ttk.Entry(af, textvariable=sv_yaw_start, width=7).grid(
                row=row, column=5, padx=4, pady=1)
            ttk.Entry(af, textvariable=sv_yaw_stop,  width=7).grid(
                row=row, column=6, padx=4, pady=1)
            ttk.Entry(af, textvariable=sv_yaw_final, width=7).grid(
                row=row, column=7, padx=4, pady=1)
            ttk.Label(af, text=f"({t_i:.0f}–{t_b:.0f} s)",
                      foreground="#888888").grid(
                row=row, column=8, sticky=tk.W, padx=(4, 8), pady=1)

            self._stage_rows.append(
                {'start': sv_start, 'stop': sv_stop, 'angle': sv_angle,
                 'yaw_start': sv_yaw_start, 'yaw_stop': sv_yaw_stop,
                 'yaw_final_az': sv_yaw_final, 'node': node})

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
            self._loft_rate_lbl.grid_forget()
            self._loft_rate_frame.grid_forget()
            self._gt_turn_start_lbl.grid(
                row=3, column=0, sticky=tk.W, padx=(8, 2), pady=2)
            self._gt_turn_start_frame.grid(
                row=3, column=1, sticky=tk.W, padx=(0, 8), pady=2)
            self._gt_turn_stop_lbl.grid(
                row=4, column=0, sticky=tk.W, padx=(8, 2), pady=2)
            self._gt_turn_stop_frame.grid(
                row=4, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        else:
            self._loft_angle_lbl.config(text="Loft Angle:")
            self._loft_angle_unit_lbl.config(text="°  (final elev.)")
            self._loft_rate_lbl.grid(row=2, column=0, sticky=tk.W, padx=(8, 2), pady=2)
            self._loft_rate_frame.grid(row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2)
            self._loft_rate_unit_lbl.config(text="°/s")
            self._loft_rate_entry.config(state="normal")
            self._gt_turn_start_lbl.grid_forget()
            self._gt_turn_start_frame.grid_forget()
            self._gt_turn_stop_lbl.grid_forget()
            self._gt_turn_stop_frame.grid_forget()
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
            loft_rate = float(self._loft_rate_var.get())
        except ValueError:
            loft_rate = 2.0
        prof = {
            'guidance':              self._guidance_var.get(),
            'loft_angle_deg':        loft_angle,
            'loft_angle_rate_deg_s': loft_rate,
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
        self._loft_angle_var.set(f"{p.loft_angle_deg:.4f}")
        self._loft_rate_var.set(f"{p.loft_angle_rate_deg_s:.3f}")
        self._guidance_var.set(p.guidance)
        self._gt_turn_start_var.set("5.0")
        self._gt_turn_stop_var.set("")
        self._update_guidance_labels(p.guidance)
        self._status_var.set(f"Trajectory reset to '{name}' defaults.")

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

        # ── Summary ───────────────────────────────────────────────────
        sf = ttk.LabelFrame(self._params_inner, text="Summary")
        sf.pack(fill=tk.X, **pad)

        total_prop = p.mass_propellant
        node = p.stage2
        while node is not None:
            total_prop += node.mass_propellant
            node = node.stage2

        r = 0
        _row(sf, r, "Name:", p.name); r += 1
        _row(sf, r, "Launch mass:", f"{p.mass_initial:,.0f} kg"); r += 1
        _row(sf, r, "Total propellant:", f"{total_prop:,.0f} kg"); r += 1
        if p.payload_kg > 0:
            _row(sf, r, "Payload:", f"{p.payload_kg:,.0f} kg"); r += 1
        if p.rv_mass_kg > 0:
            _row(sf, r, "RV mass:", f"{p.rv_mass_kg:,.0f} kg"); r += 1
        liftoff_tw = p.thrust_N / (p.mass_initial * _G0)
        _row(sf, r, "Liftoff T/W:", f"{liftoff_tw:.2f}"); r += 1

        # ── Per-stage blocks ──────────────────────────────────────────
        sn = 1
        node = p
        while node is not None:
            is_last = node.stage2 is None
            lf = ttk.LabelFrame(self._params_inner,
                                text=f"Stage {sn}" if sn > 1 else "Stage 1")
            lf.pack(fill=tk.X, **pad)

            prop = node.mass_propellant
            tw   = node.thrust_N / (node.mass_initial * _G0)

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
            mdot = node.thrust_N / (node.isp_s * _G0)
            r = 0
            _row(right, r, "Thrust (kN):",       f"{node.thrust_N/1000:,.0f}");  r += 1
            _row(right, r, "ISP (s):",            f"{node.isp_s:.0f}");           r += 1
            _row(right, r, "Nozzle area (m²):",  f"{node.nozzle_exit_area_m2:.4f}"); r += 1
            _row(right, r, "Burntime (s):",       f"{node.burn_time_s:.1f}  (computed)"); r += 1
            _row(right, r, "Mass flow (kg/s):",   f"{mdot:.1f}  (computed)");     r += 1
            _row(right, r, "T/W ratio:",          f"{tw:.2f}  (computed)");       r += 1

            sn  += 1
            node = node.stage2

        # ── Payload nose (shown in Front End section of params display) ──
        if p.nose_shape not in ('', 'forden'):
            af = ttk.LabelFrame(self._params_inner, text="Front End — Payload Nose")
            af.pack(fill=tk.X, **pad)
            _row(af, 0, "Payload nose shape:",
                 NOSE_SHAPE_LABELS.get(p.nose_shape, p.nose_shape))
            if p.nose_length_m > 0 and p.diameter_m > 0:
                _ld = p.nose_length_m / p.diameter_m
                _row(af, 1, "Payload nose length (m):",
                     f"{p.nose_length_m:.2f}  (L/D = {_ld:.2f})")

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

        # ── Total length ──────────────────────────────────────────────
        _stage_lengths = []
        _node = p
        _sn = 1
        while _node is not None:
            if _node.length_m > 0:
                _stage_lengths.append((_sn, _node.length_m))
            _node = _node.stage2
            _sn += 1
        _total_len = sum(l for _, l in _stage_lengths)
        if p.shroud_length_m > 0:
            _total_len += p.shroud_length_m
        if _total_len > 0:
            lf_len = ttk.LabelFrame(self._params_inner, text="Total Length")
            lf_len.pack(fill=tk.X, **pad)
            _parts = [f"S{n}: {l:.1f} m" for n, l in _stage_lengths]
            if p.shroud_length_m > 0:
                _parts.append(f"Shroud: {p.shroud_length_m:.1f} m")
            _row(lf_len, 0, "Total (m):", f"{_total_len:.2f}")
            _row(lf_len, 1, "Breakdown:", "  +  ".join(_parts))

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
            la  = float(self._loft_angle_var.get())
            lar = float(self._loft_rate_var.get())
            gt_start_str = self._gt_turn_start_var.get().strip()
            gt_stop_str  = self._gt_turn_stop_var.get().strip()
            gt_start_s   = float(gt_start_str) if gt_start_str else 5.0
            gt_stop_s    = float(gt_stop_str)  if gt_stop_str  else None
            threading.Thread(
                target=self._aim_thread,
                args=(missile, guidance, lat1_dd, lon1_dd, az, rng_km, la, lar,
                      gt_start_s, gt_stop_s),
                daemon=True,
            ).start()

        except Exception as e:
            messagebox.showerror("Aim error", str(e))

    def _aim_thread(self, missile, guidance, lat, lon, az, rng_km, la, lar,
                    gt_start_s=5.0, gt_stop_s=None):
        try:
            cutoff = aim_missile(missile, lat, lon, az, rng_km,
                                 guidance=guidance,
                                 loft_angle_deg=la,
                                 loft_angle_rate_deg_s=lar,
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
        la       = float(self._loft_angle_var.get())
        lar      = float(self._loft_rate_var.get())
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
                # Per-stage yaw overrides (blank = no yaw on this stage)
                _ys = row.get('yaw_start', tk.StringVar()).get().strip()
                _ye = row.get('yaw_stop',  tk.StringVar()).get().strip()
                _ya = row.get('yaw_final_az', tk.StringVar()).get().strip()
                if _ys: node.stage_yaw_start_s      = float(_ys)
                if _ye: node.stage_yaw_stop_s        = float(_ye)
                if _ya: node.stage_yaw_final_az_deg  = float(_ya)
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
        if yaw_enabled:
            yaw_start_s  = _fon(self._yaw_start_var)
            yaw_stop_s   = _fon(self._yaw_stop_var)
            yaw_final_az = _fon(self._yaw_final_az_var)
            yaw_enabled  = (yaw_final_az is not None)
        else:
            yaw_start_s = yaw_stop_s = yaw_final_az = None

        return (missile, guidance, lat, lon, az, cutoff, la, lar,
                gt_start_s, gt_stop_s, target_orbit_km,
                yaw_enabled, yaw_start_s, yaw_stop_s, yaw_final_az)

    def _open_sweep(self):
        ParametricSweepDialog(self)

    def _run_flyout(self):
        if self._running:
            return
        try:
            (missile, guidance, lat, lon, az, cutoff, la, lar,
             gt_start_s, gt_stop_s, target_orbit_km,
             yaw_enabled, yaw_start_s, yaw_stop_s, yaw_final_az) = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self._running = True
        self._status_var.set("Running simulation…")
        threading.Thread(
            target=self._run_thread,
            args=(missile, guidance, lat, lon, az, cutoff, la, lar,
                  gt_start_s, gt_stop_s, target_orbit_km,
                  yaw_enabled, yaw_start_s, yaw_stop_s, yaw_final_az, False),
            daemon=True,
        ).start()

    def _maximize_range(self):
        if self._running:
            return
        try:
            (missile, guidance, lat, lon, az, cutoff, la, lar,
             gt_start_s, gt_stop_s, target_orbit_km,
             yaw_enabled, yaw_start_s, yaw_stop_s, yaw_final_az) = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self._running = True
        self._status_var.set("Optimising for maximum range…")
        threading.Thread(
            target=self._run_thread,
            args=(missile, guidance, lat, lon, az, cutoff, la, lar,
                  gt_start_s, None, target_orbit_km,
                  yaw_enabled, yaw_start_s, yaw_stop_s, yaw_final_az, True),
            daemon=True,
        ).start()

    def _plan_orbit(self):
        """Handler for the Plan Orbit button."""
        if self._running:
            return
        try:
            (missile, guidance, lat, lon, az, cutoff, la, lar,
             gt_start_s, gt_stop_s, target_orbit_km,
             _yaw_en, _yaw_st, _yaw_sp, _yaw_fa) = self._get_inputs()
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
                 cutoff_run, la_run, lar_run,
                 gts_run, gtstp_run, orb_run,
                 yaw_en_run, yaw_st_run, yaw_sp_run, yaw_fa_run) = self._get_inputs()
            except ValueError:
                # Fallback: use the original plan parameters without per-stage overrides.
                m_run, guidance_run = missile, "orbital_insertion"
                lat_run, lon_run, az_run = lat, lon, az
                cutoff_run, la_run, lar_run = None, boost_angle, 0.0
                gts_run, gtstp_run, orb_run = gt_start_s, turn_stop, target_orbit_km
                yaw_en_run, yaw_st_run, yaw_sp_run, yaw_fa_run = False, None, None, None

            # _run_thread checks self._running; it's still True from _plan_orbit
            threading.Thread(
                target=self._run_thread,
                args=(m_run, guidance_run, lat_run, lon_run, az_run,
                      cutoff_run, la_run, lar_run,
                      gts_run, gtstp_run, orb_run,
                      yaw_en_run, yaw_st_run, yaw_sp_run, yaw_fa_run, False),
                daemon=True,
            ).start()

        self.after(0, _apply_and_run)

    def _run_thread(self, missile, guidance, lat, lon, az, cutoff, la, lar,
                    gt_start_s, gt_stop_s, target_orbit_km,
                    yaw_enabled, yaw_start_s, yaw_stop_s, yaw_final_az, maximise):
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
                    loft_angle_deg=la,
                    loft_angle_rate_deg_s=lar,
                    cutoff_time_s=cutoff,
                    gt_turn_start_s=gt_start_s,
                    gt_turn_stop_s=gt_stop_s,
                    reentry_query_alt_km=q_alt,
                    target_orbit_alt_km=target_orbit_km,
                    yaw_enabled=yaw_enabled,
                    yaw_start_s=yaw_start_s,
                    yaw_stop_s=yaw_stop_s,
                    yaw_final_az_deg=yaw_final_az,
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
        if r.get('optimal_loft_angle_deg') is not None:
            self._loft_angle_var.set(f"{r['optimal_loft_angle_deg']:.4f}")
            self._loft_rate_var .set(f"{r['optimal_loft_rate_deg_s']:.3f}")
        if r.get('optimal_gt_turn_stop_s') is not None and self._guidance_var.get() == "gravity_turn":
            self._gt_turn_stop_var.set(f"{r['optimal_gt_turn_stop_s']:.1f}")

        orbital   = r.get('orbital', False)
        rng_km    = r['range_km']
        rng_nm    = rng_km / 1.852   if rng_km    is not None else None
        rng_mi    = rng_km / 1.60934 if rng_km    is not None else None
        apogee_km = r['apogee_km']

        tof_s       = r['time_of_flight_s']
        imp_spd_kms = r['impact_speed_ms'] / 1000.0 if r['impact_speed_ms'] is not None else None
        apo_lat     = r['apogee_lat_deg']
        apo_lon     = r['apogee_lon_deg']

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
            self._status_var.set(
                "Max Range: no sub-orbital solution found — "
                "vehicle exceeds orbital velocity at all tested burnout angles.  "
                f"Apogee: {apogee_km*scale:.1f} {ulbl}"
            )
        elif orbital:
            self._status_var.set(
                f"In orbit (no ground impact within integration window).  "
                f"Apogee: {apogee_km*scale:.1f} {ulbl}"
                + _oe_str
            )
        else:
            self._status_var.set(
                f"Done.  Range: {rng_km*scale:.1f} {ulbl}  |  "
                f"Apogee: {apogee_km*scale:.1f} {ulbl}  |  "
                f"ToF: {tof_s:.0f} s  |  "
                f"Impact: {r['impact_lat']:.2f}°N, {r['impact_lon']:.2f}°E  |  "
                f"Impact spd: {imp_spd_kms:.2f} km/s"
            )
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
            self._tl_summary_var.set(
                f"In orbit — no ground impact within integration window\n"
                f"Apogee: {apogee_km:.1f} km   "
                f"Apogee loc: {r['apogee_lat_deg']:.2f}°N  {r['apogee_lon_deg']:.2f}°E"
                + _oe_line
            )
        else:
            self._tl_summary_var.set(
                f"Range: {rng_km:.1f} km  /  {rng_nm:.1f} nmi  /  {rng_mi:.1f} mi\n"
                f"Apogee: {apogee_km:.1f} km   "
                f"Apogee loc: {r['apogee_lat_deg']:.2f}°N  {r['apogee_lon_deg']:.2f}°E\n"
                f"Impact: {r['impact_lat']:.2f}°N  {r['impact_lon']:.2f}°E   "
                f"Flight time: {tof_s:.0f} s   "
                f"Impact speed: {imp_spd:.2f} km/s"
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

        self._canvas.draw_idle()
        self._canvas.flush_events()

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
    def _save_trajectory(self):
        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export trajectory CSV",
        )
        if not path:
            return
        r = self._result
        rows = []
        # Main vehicle trajectory
        for i, ti in enumerate(r['t']):
            rows.append(f"vehicle,{ti:.3f},{r['lat'][i]:.6f},{r['lon'][i]:.6f},"
                        f"{r['alt'][i]:.1f},{r['speed'][i]:.2f},{r['range'][i]/1000.0:.3f}")
        # Debris / stage trajectories
        for d in r.get('debris_trajectories', []):
            label = d['label'].replace(',', ' ')
            for i, ti in enumerate(d['t']):
                rows.append(f"{label},{ti:.3f},{d['lat'][i]:.6f},{d['lon'][i]:.6f},"
                            f"{d['alt'][i]:.1f},,")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("piece,time_s,lat_deg,lon_deg,alt_m,speed_ms,range_km\n")
            fh.write("\n".join(rows))
            fh.write("\n")
        self._status_var.set(f"Trajectory CSV exported: {path}")

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

        _label_data = []   # [{lat, lon, text, t_s}] for JS labels
        _tick_data  = []   # [{lat, lon}] for JS tick marks

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
                                    'text': display_name, 't_s': ms['t_s']})
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
            var H_GAP     = 10;   // px right of the dot centre
            var V_ABOVE   = 4;    // px between dot and nearest edge of label
            var STACK_GAP = 3;    // px between stacked labels
            var PAD       = 2;    // extra padding around each label box
            var TICK_HALF = 8;    // px: half-length of tick mark

            var _svg = null, _con = null, _divs = [];

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
            }}

            function _update(map) {{
                if (!_con) return;
                _svg.innerHTML = '';

                var mc   = map.getContainer();
                var mapW = mc.offsetWidth  || 800;
                var mapH = mc.offsetHeight || 600;
                var EDGE = 40;

                // ── Convert trajectory skeleton to container points ───────
                var tPts = TRAJ.map(function(tp) {{
                    return map.latLngToContainerPoint([tp.lat, tp.lon]);
                }});

                // ── Draw tick marks ───────────────────────────────────────
                TICKS.forEach(function(tk) {{
                    var tp = map.latLngToContainerPoint([tk.lat, tk.lon]);
                    // Find nearest TRAJ point.
                    var bestD2 = Infinity, bestI = 0;
                    for (var i = 0; i < tPts.length; i++) {{
                        var dx = tp.x - tPts[i].x, dy = tp.y - tPts[i].y;
                        var d2 = dx*dx + dy*dy;
                        if (d2 < bestD2) {{ bestD2 = d2; bestI = i; }}
                    }}
                    // Local tangent from the nearest segment.
                    var i0  = Math.min(bestI, tPts.length - 2);
                    var tdx = tPts[i0 + 1].x - tPts[i0].x;
                    var tdy = tPts[i0 + 1].y - tPts[i0].y;
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

                // Returns screen-space dy of the trajectory at the point
                // nearest to pt.  dy < 0 → ascending (going up-screen);
                // dy ≥ 0 → descending / flat.
                function tangentDy(pt) {{
                    var best = Infinity, bestIdx = 0;
                    for (var i = 0; i < tPts.length; i++) {{
                        var ddx = tPts[i].x - pt.x, ddy = tPts[i].y - pt.y;
                        var d2  = ddx*ddx + ddy*ddy;
                        if (d2 < best) {{ best = d2; bestIdx = i; }}
                    }}
                    var i0 = Math.max(0, bestIdx - 1);
                    var i1 = Math.min(tPts.length - 1, bestIdx + 1);
                    return tPts[i1].y - tPts[i0].y;
                }}

                // Hide all labels first.
                _divs.forEach(function(d) {{ d.style.display = 'none'; }});

                // Visible markers only, sorted right-to-left so stacking is
                // chronological and labels don't leap over one another.
                var order = pts.map(function(_, i) {{ return i; }}).filter(function(i) {{
                    var p = pts[i];
                    return p.x >= -EDGE && p.x <= mapW + EDGE &&
                           p.y >= -EDGE && p.y <= mapH + EDGE;
                }});
                // Chronological order: earliest event nearest its dot,
                // later events stacked further in the trajectory direction.
                order.sort(function(a, b) {{ return LABELS[a].t_s - LABELS[b].t_s; }});

                var CLUSTER_R = 60;
                var topY  = {{}}, below = {{}};

                var prevTop = null, prevBottom = null, prevPt = null;
                order.forEach(function(idx) {{
                    var pt = pts[idx];
                    _divs[idx].style.display = 'block';
                    var lh = (_divs[idx].offsetHeight || 14) + PAD * 2;

                    // Reset stacking group when this dot is far from the last.
                    if (prevPt) {{
                        var dx = pt.x - prevPt.x, dy = pt.y - prevPt.y;
                        if (Math.sqrt(dx*dx + dy*dy) > CLUSTER_R) {{
                            prevTop = null; prevBottom = null;
                        }}
                    }}

                    // Ascending trajectory at this dot (screen dy < 0) →
                    // label goes below the dot (opposite side of line).
                    // Descending / flat (dy ≥ 0) → label goes above.
                    var goBelow = tangentDy(pt) < 0;
                    var candidate;
                    if (goBelow) {{
                        var idealBottom = pt.y + V_ABOVE;
                        candidate = (prevBottom === null)
                            ? idealBottom
                            : Math.max(idealBottom, prevBottom + lh + STACK_GAP);
                        if (candidate + lh > mapH - EDGE) {{
                            goBelow = false;   // no room below — try above
                        }} else {{
                            prevBottom = candidate;
                        }}
                    }}
                    if (!goBelow) {{
                        var idealTop = pt.y - lh - V_ABOVE;
                        candidate = (prevTop === null)
                            ? idealTop
                            : Math.min(idealTop, prevTop - lh - STACK_GAP);
                        if (candidate < EDGE) {{
                            candidate  = pt.y + V_ABOVE;   // force below
                            goBelow    = true;
                            prevTop    = null;
                        }} else {{
                            prevTop = candidate;
                        }}
                    }}
                    topY[idx]  = candidate;
                    below[idx] = goBelow;
                    prevPt     = pt;
                }});

                // Render: position label divs only (no leader lines).
                order.forEach(function(idx) {{
                    var pt  = pts[idx];
                    var lh  = (_divs[idx].offsetHeight || 14) + PAD * 2;
                    var lx  = pt.x + H_GAP;
                    var ly  = topY[idx];
                    _divs[idx].style.left = lx + 'px';
                    _divs[idx].style.top  = ly + 'px';
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

    def _export_missile(self):
        """Export the current missile definition to a JSON file."""
        name = self._missile_var.get()
        if not name or name not in MISSILE_DB:
            messagebox.showinfo("No missile", "Select a missile first.")
            return
        from tkinter.filedialog import asksaveasfilename
        safe = name.replace(" ", "_").replace("/", "-")
        path = asksaveasfilename(
            defaultextension=".json",
            initialfile=f"{safe}.json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Export missile definition",
        )
        if not path:
            return
        data = missile_to_dict(MISSILE_DB[name]())
        Path(path).write_text(json.dumps(data, indent=2))
        self._status_var.set(f"Missile exported: {path}")

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
            "  • Per-stage loft-angle guidance (Forden Eq. 8)\n"
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
