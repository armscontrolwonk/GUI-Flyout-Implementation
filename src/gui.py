"""
GUI Missile Flyout V1.1 — Python/tkinter port of Forden's MATLAB GUIDE application.

Layout mirrors the original MATLAB GUIDE application:
  Left panel  : missile type, units, launch site (decimal °), target (decimal °),
                cutoff time, run buttons, range/apogee results
  Right panel : 4-up matplotlib plots (altitude, speed, trajectory, ground track)
  Bottom bar  : status line
"""

import tkinter as tk
from tkinter import ttk, messagebox
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

from missile_models import (MISSILE_DB, get_missile,
                           missile_to_dict, missile_from_dict,
                           total_burn_time)
from trajectory import integrate_trajectory, maximize_range, aim_missile
from coordinates import range_between

# ---------------------------------------------------------------------------
# Custom missile persistence
# ---------------------------------------------------------------------------

# Names that ship with the program and cannot be deleted
_PACKAGED_NAMES    = set(MISSILE_DB.keys())
_PACKAGED_ORIGINALS = dict(MISSILE_DB)   # original factory lambdas for reset
_OVERRIDDEN_PACKAGED: set = set()        # packaged missiles the user has edited

# Where user-created missiles are saved
_CUSTOM_PATH = Path.home() / ".gui_missile_flyout" / "custom_missiles.json"


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
                     loft_a="45.0", loft_r="1.5", coast="0")

    _G0 = 9.80665  # m/s²

    def __init__(self, parent, label, defaults=None):
        super().__init__(parent, text=label)
        d = {**self._DEFAULTS, **(defaults or {})}
        self._fueled    = _entry_row(self, "Fueled wt (kg):", 0, d["fueled"],    "kg")
        self._dry       = _entry_row(self, "Dry wt (kg):",    1, d["dry"],       "kg")
        self._dia       = _entry_row(self, "Diameter (m):",   2, d["dia"],       "m")
        self._length    = _entry_row(self, "Length (m):",     3, d["length"],    "m")
        self._thrust_kn = _entry_row(self, "Thrust (kN):",    4, d["thrust_kn"], "kN")
        self._isp       = _entry_row(self, "Isp (s):",        5, d["isp"],       "s")

        # Burn time — read-only computed field
        ttk.Label(self, text="Burn time (s):").grid(
            row=6, column=0, sticky=tk.W, padx=(6, 2), pady=2)
        self._burn_var = tk.StringVar()
        _burn_inner = ttk.Frame(self)
        _burn_inner.grid(row=6, column=1, sticky=tk.W, padx=(0, 6), pady=2)
        ttk.Entry(_burn_inner, textvariable=self._burn_var, width=10,
                  state="readonly").pack(side=tk.LEFT)
        ttk.Label(_burn_inner, text="s  (computed)").pack(side=tk.LEFT, padx=(2, 0))

        self._loft_a = _entry_row(self, "Loft angle (°):", 7, d["loft_a"],
                                  "° (final elev.)")
        self._loft_r = _entry_row(self, "Loft rate (°/s):", 8, d["loft_r"], "°/s")

        # Recompute burn whenever any of the four driving fields change
        for _v in (self._fueled, self._dry, self._thrust_kn, self._isp):
            _v.trace_add("write", self._recompute_burn)
        self._recompute_burn()

    def _recompute_burn(self, *_):
        """Compute burn time = Isp × g₀ × prop / thrust and update the display."""
        try:
            prop      = float(self._fueled.get()) - float(self._dry.get())
            thrust_n  = float(self._thrust_kn.get()) * 1000.0
            isp       = float(self._isp.get())
            if thrust_n <= 0 or isp <= 0 or prop <= 0:
                raise ValueError
            burn = isp * self._G0 * prop / thrust_n
            self._burn_var.set(f"{burn:.1f}")
        except (ValueError, ZeroDivisionError):
            self._burn_var.set("—")

        # Coast-time row — shown only when this stage is not the last stage.
        # Stored as grid row 8; hidden/shown via set_coast_visible().
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
            raise ValueError("Burn time could not be computed — check thrust, Isp, and weights.")
        return {k: float(v.get()) for k, v in [
            ("fueled",    self._fueled),    ("dry",    self._dry),
            ("dia",       self._dia),       ("length", self._length),
            ("thrust_kn", self._thrust_kn), ("isp",    self._isp),
            ("loft_a",    self._loft_a),    ("loft_r", self._loft_r),
            ("coast",     self._coast_var),
        ]} | {"burn": float(burn_str)}

    def populate(self, d):
        # Back-calculate thrust_kn from stored burn/isp/prop so the round-trip
        # is exact: T = Isp × g₀ × prop / burn
        prop = d["fueled"] - d["dry"]
        burn = d["burn"]
        thrust_kn = (d["isp"] * self._G0 * prop / burn / 1000.0
                     if burn > 0 and prop > 0 else 0.0)

        self._fueled    .set(str(d["fueled"]))
        self._dry       .set(str(d["dry"]))
        self._dia       .set(str(d["dia"]))
        self._length    .set(str(d["length"]))
        self._thrust_kn .set(f"{thrust_kn:.1f}")
        self._isp       .set(str(d["isp"]))
        # _burn_var is updated automatically by the trace
        self._loft_a    .set(str(d.get("loft_a", 45.0)))
        self._loft_r    .set(str(d.get("loft_r", 1.5)))
        self._coast_var .set(str(d.get("coast", 0)))


# ---------------------------------------------------------------------------
# New / Edit missile dialog
# ---------------------------------------------------------------------------

class MissileDialog(tk.Toplevel):
    """Modal dialog for creating or editing a custom missile."""

    def __init__(self, parent, on_save, existing_name=None):
        super().__init__(parent)
        self.title("Edit Missile" if existing_name else "New Missile")
        self.resizable(False, True)
        self.grab_set()               # modal
        self._on_save = on_save
        self._existing_name = existing_name
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
        ttk.Entry(nf, textvariable=self._name_var, width=24).pack(
            side=tk.LEFT, padx=(6, 16))
        ttk.Label(nf, text="Stages:").pack(side=tk.LEFT)
        self._n_stages_var = tk.StringVar(value="1")
        stages_cb = ttk.Combobox(nf, textvariable=self._n_stages_var,
                                 values=["1", "2", "3", "4"],
                                 state="readonly", width=3)
        stages_cb.pack(side=tk.LEFT, padx=(4, 0))
        stages_cb.bind("<<ComboboxSelected>>",
                       lambda _: self._update_stage_frames())

        # Scrollable body: canvas + scrollbar sandwiched between the name row
        # and the Save/Cancel buttons so buttons are always visible.
        scroll_outer = ttk.Frame(self)
        scroll_outer.pack(fill=tk.BOTH, expand=True)

        self._scroll_canvas = tk.Canvas(
            scroll_outer, borderwidth=0, highlightthickness=0)
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

        # Payload + RV beta (inside scrollable body)
        pf = ttk.Frame(body)
        pf.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(pf, text="Payload (kg):").pack(side=tk.LEFT)
        self._payload_var = tk.StringVar(value="1000")
        ttk.Entry(pf, textvariable=self._payload_var, width=8).pack(
            side=tk.LEFT, padx=(4, 12))
        ttk.Label(pf, text="RV β (kg/m²):").pack(side=tk.LEFT)
        self._rv_beta_var = tk.StringVar(value="0")
        ttk.Entry(pf, textvariable=self._rv_beta_var, width=8).pack(
            side=tk.LEFT, padx=(4, 8))
        ttk.Label(pf, text="(0 = use stage body aero)").pack(side=tk.LEFT)

        # Stage frames (1 always visible; 2-4 toggled).
        # A dedicated container ensures dynamically-packed stages always appear
        # between the payload row and the buttons (not after the buttons).
        self._stages_container = ttk.Frame(body)
        self._stages_container.pack(fill=tk.X)
        self._stage_frames = [_StageFrame(self._stages_container, f"Stage {i+1}")
                               for i in range(4)]
        self._stage_frames[0].pack(fill=tk.X, **pad)  # Stage 1 always shown

        # Buttons — outside the scroll area so always visible
        bf = ttk.Frame(self)
        bf.pack(fill=tk.X, padx=8, pady=(4, 8))
        ttk.Button(bf, text="Cancel", command=self.destroy).pack(
            side=tk.RIGHT, padx=(4, 0))
        ttk.Button(bf, text="Save Missile", command=self._save).pack(
            side=tk.RIGHT)

        # Pre-fill if editing an existing missile
        if existing_name:
            self._prefill(existing_name)

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

        # Walk the linked list to collect per-stage data and payload.
        # payload_kg is stored on the top-level node by _collect at save time.
        payload = p.payload_kg
        stage_data = []
        node = p
        while node is not None:
            nxt = node.stage2
            if nxt is None:
                # Last stage: payload is in mass_initial but NOT mass_final
                # (payload separates at burnout), so dry = mass_final directly.
                fueled = node.mass_initial - payload
                dry    = node.mass_final
            else:
                fueled = node.mass_initial - nxt.mass_initial
                dry    = node.mass_final
            stage_data.append({
                "fueled": fueled, "dry": dry,
                "dia":    node.diameter_m, "length": node.length_m,
                "burn":   node.burn_time_s, "isp":   node.isp_s,
                "loft_a": node.loft_angle_deg,
                "loft_r": node.loft_angle_rate_deg_s,
                "coast":  node.coast_time_s,
            })
            node = nxt

        n = len(stage_data)
        self._n_stages_var.set(str(n))
        self._update_stage_frames()
        for i, sd in enumerate(stage_data):
            self._stage_frames[i].populate(sd)
        self._payload_var.set(f"{payload:.0f}")
        self._rv_beta_var.set(f"{p.rv_beta_kg_m2:.0f}")
        self._name_var.set(name)

    # ------------------------------------------------------------------
    def _collect(self) -> 'MissileParams':
        """Read and validate all fields; return a MissileParams linked list."""
        from missile_models import MissileParams, _FORDEN_MACH, _FORDEN_CD

        name = self._name_var.get().strip()
        if not name:
            raise ValueError("Missile name cannot be blank.")

        n       = int(self._n_stages_var.get())
        payload = float(self._payload_var.get())
        rv_beta = float(self._rv_beta_var.get())

        # Read and validate all active stage frames
        stages = []
        for i in range(n):
            sd = self._stage_frames[i].get()
            if sd["fueled"] <= sd["dry"]:
                raise ValueError(
                    f"Stage {i+1}: fueled weight must exceed dry weight.")
            stages.append(sd)

        # Build the linked list from the last stage back to the first.
        # Last stage carries payload; upper stages are jettisoned at burnout.
        node = None
        upper_mass = 0.0   # cumulative mass of stages above the current one
        for idx, sd in enumerate(reversed(stages)):
            stage_num = n - idx           # stage number (1-based), decreasing
            is_last   = (node is None)    # first iteration = topmost (last) stage
            prop = sd["fueled"] - sd["dry"]
            if is_last:
                m0     = sd["fueled"] + payload
                mfinal = sd["dry"]              # payload separates at burnout
                upper_mass = m0
            else:
                m0     = sd["fueled"] + upper_mass
                mfinal = sd["dry"]    # jettisoned structure only
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
                loft_angle_deg=sd["loft_a"],
                loft_angle_rate_deg_s=sd["loft_r"],
                mach_table=list(_FORDEN_MACH), cd_table=list(_FORDEN_CD),
                stage2=node,
            )

        node.name = name            # top-level gets the missile's proper name
        node.payload_kg  = payload  # store so _prefill can round-trip correctly
        node.rv_beta_kg_m2 = rv_beta
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
            missile, lat, lon, az, cutoff, la, lar = self._app._get_inputs()
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
            args=(missile, lat, lon, az, la, lar, cutoff,
                  param_key, points, overplot),
            daemon=True,
        ).start()

    # ------------------------------------------------------------------
    def _sweep_worker(self, missile, lat, lon, az, la, lar, cutoff,
                      param_key, points, store_trajs):
        for i, val in enumerate(points):
            if self._stop_evt.is_set():
                break
            run_az  = val if param_key == "azimuth"    else az
            run_la  = val if param_key == "loft_angle" else la
            run_cut = val if param_key == "cutoff"     else cutoff
            try:
                r = integrate_trajectory(
                    missile, lat, lon, run_az,
                    loft_angle_deg=run_la,
                    loft_angle_rate_deg_s=lar,
                    cutoff_time_s=run_cut,
                )
                row  = (val, r["range_km"], r["apogee_km"])
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
        self.title("GUI Missile Flyout V1.1  —  Python port of Forden (2001)")
        self.geometry("1280x780")
        self.resizable(True, True)

        self._result  = None
        self._running = False

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
        file_menu.add_command(label="Save trajectory…", command=self._save_trajectory)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=file_menu)

        analysis_menu = tk.Menu(menubar, tearoff=0)
        analysis_menu.add_command(label="Parametric Sweep…", command=self._open_sweep)
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

        # Left control panel (fixed width, non-expanding)
        left = ttk.Frame(top, width=310)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 6))
        left.pack_propagate(False)
        self._build_control_panel(left)

        # Right plot panel
        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._build_plot_panel(right)

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
        ttk.Label(parent, text="GUI Missile Flyout V1.1",
                  font=("", 11, "bold")).pack(pady=(6, 2))
        ttk.Label(parent, text="Python port of Forden (2001)",
                  font=("", 8, "italic"), foreground="grey").pack()
        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=6)

        # ── Missile type ───────────────────────────────────────────────
        mf = ttk.LabelFrame(parent, text="Missile Type")
        mf.pack(fill=tk.X, padx=6, pady=3)
        self._missile_var = tk.StringVar(value=list(MISSILE_DB.keys())[0])
        self._missile_cb = ttk.Combobox(mf, textvariable=self._missile_var,
                                        values=list(MISSILE_DB.keys()),
                                        state="readonly", width=24)
        self._missile_cb.pack(padx=6, pady=(4, 2))
        self._missile_cb.bind("<<ComboboxSelected>>", self._on_missile_changed)

        mb = ttk.Frame(mf)
        mb.pack(padx=6, pady=(0, 4))
        ttk.Button(mb, text="New…",    width=7,
                   command=self._new_missile).pack(side=tk.LEFT, padx=2)
        ttk.Button(mb, text="Edit…",   width=7,
                   command=self._edit_missile).pack(side=tk.LEFT, padx=2)
        self._del_btn = ttk.Button(mb, text="Delete", width=7,
                                   command=self._delete_missile)
        self._del_btn.pack(side=tk.LEFT, padx=2)

        # ── Units ──────────────────────────────────────────────────────
        uf = ttk.LabelFrame(parent, text="Display Units")
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

        self._launch_lat = _dd_row(lf, "Latitude:",  row=0, default="0.0")
        self._launch_lon = _dd_row(lf, "Longitude:", row=1, default="0.0")

        ttk.Label(lf, text="Azimuth:").grid(row=2, column=0,
                                             sticky=tk.W, padx=(8, 2), pady=2)
        az_frame = ttk.Frame(lf)
        az_frame.grid(row=2, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._azimuth_var = tk.StringVar(value="0.0")
        ttk.Entry(az_frame, textvariable=self._azimuth_var, width=8).pack(side=tk.LEFT)
        ttk.Label(az_frame, text="°  (from N)").pack(side=tk.LEFT, padx=2)

        # ── Target ────────────────────────────────────────────────────
        tf = ttk.LabelFrame(parent, text="Target")
        tf.pack(fill=tk.X, padx=6, pady=3)

        self._target_lat = _dd_row(tf, "Latitude:",  row=0, default="0.0")
        self._target_lon = _dd_row(tf, "Longitude:", row=1, default="0.0")

        ttk.Button(tf, text="Aim at Target", command=self._aim_at_target,
                   width=18).grid(row=2, column=0, columnspan=2, pady=(4, 6))

        # ── Guidance — loft angle / pitch-over (Forden Eq. 8) ─────────
        gf = ttk.LabelFrame(parent, text="Guidance")
        gf.pack(fill=tk.X, padx=6, pady=3)

        ttk.Label(gf, text="Loft Angle:").grid(
            row=0, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        la_frame = ttk.Frame(gf)
        la_frame.grid(row=0, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._loft_angle_var = tk.StringVar(value="45.0")
        ttk.Entry(la_frame, textvariable=self._loft_angle_var, width=8).pack(side=tk.LEFT)
        ttk.Label(la_frame, text="°  (final elev.)").pack(side=tk.LEFT, padx=2)

        ttk.Label(gf, text="Loft Rate:").grid(
            row=1, column=0, sticky=tk.W, padx=(8, 2), pady=2)
        lr_frame = ttk.Frame(gf)
        lr_frame.grid(row=1, column=1, sticky=tk.W, padx=(0, 8), pady=2)
        self._loft_rate_var = tk.StringVar(value="2.0")
        ttk.Entry(lr_frame, textvariable=self._loft_rate_var, width=8).pack(side=tk.LEFT)
        ttk.Label(lr_frame, text="°/s").pack(side=tk.LEFT, padx=2)

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

        # ── Results display ───────────────────────────────────────────
        rf = ttk.LabelFrame(parent, text="Results")
        rf.pack(fill=tk.X, padx=6, pady=3)

        self._res_range_km  = tk.StringVar(value="Range (km):    —")
        self._res_range_nm  = tk.StringVar(value="Range (nmi):   —")
        self._res_range_mi  = tk.StringVar(value="Range (miles): —")
        self._res_apogee    = tk.StringVar(value="Apogee (km):   —")
        self._res_apogee_ll = tk.StringVar(value="Apogee loc:    —")
        self._res_impact    = tk.StringVar(value="Impact:        —")
        self._res_tof       = tk.StringVar(value="Flight time:   —")
        self._res_imp_spd   = tk.StringVar(value="Impact speed:  —")

        for var in (self._res_range_km, self._res_range_nm, self._res_range_mi,
                    self._res_apogee, self._res_apogee_ll,
                    self._res_impact, self._res_tof, self._res_imp_spd):
            ttk.Label(rf, textvariable=var,
                      font=("Courier", 9), anchor=tk.W).pack(
                fill=tk.X, padx=8, pady=1)

        # ── Missile parameters ────────────────────────────────────────
        pf = ttk.LabelFrame(parent, text="Missile Parameters")
        pf.pack(fill=tk.BOTH, expand=True, padx=6, pady=3)
        self._params_text = tk.Text(pf, width=36, height=8,
                                    font=("Courier", 8), state=tk.DISABLED,
                                    relief=tk.FLAT, bg=self.cget("bg"))
        sb = ttk.Scrollbar(pf, command=self._params_text.yview)
        self._params_text.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._params_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

    # ------------------------------------------------------------------
    # Plot panel  (4-subplot figure in a single tab + navigation toolbar)
    # ------------------------------------------------------------------
    def _build_plot_panel(self, parent):
        self._fig = Figure(figsize=(8, 6), dpi=96)
        self._ax_alt  = self._fig.add_subplot(221)   # top-left:  alt vs time
        self._ax_spd  = self._fig.add_subplot(222)   # top-right: speed vs time
        self._ax_traj = self._fig.add_subplot(223)   # bot-left:  alt vs range
        self._ax_trk  = self._fig.add_subplot(224)   # bot-right: ground track
        self._fig.tight_layout(pad=2.8)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self._canvas, parent)
        toolbar.update()

        # Initialise axes with placeholder labels
        self._init_axes()
        self._canvas.draw()

    def _init_axes(self):
        for ax, title, xl, yl in [
            (self._ax_alt,  "Altitude vs Time",       "Time (s)",         "Altitude (km)"),
            (self._ax_spd,  "Speed vs Time",           "Time (s)",         "Speed (km/s)"),
            (self._ax_traj, "Altitude vs Range",       "Downrange (km)",   "Altitude (km)"),
            (self._ax_trk,  "Ground Track",            "Longitude (°E)",   "Latitude (°N)"),
        ]:
            ax.set_title(title, fontsize=9)
            ax.set_xlabel(xl, fontsize=8)
            ax.set_ylabel(yl, fontsize=8)
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=7)

    # ------------------------------------------------------------------
    # Missile selection
    # ------------------------------------------------------------------
    def _on_missile_changed(self, _event=None):
        p = get_missile(self._missile_var.get())
        self._cutoff_var.set(str(int(total_burn_time(p))))
        self._loft_angle_var.set(f"{p.loft_angle_deg:.4f}")
        self._loft_rate_var.set(f"{p.loft_angle_rate_deg_s:.3f}")
        self._update_params_text(p)
        # Allow deleting custom missiles and resetting overridden packaged ones
        name = self._missile_var.get()
        is_custom = name not in _PACKAGED_NAMES
        is_overridden = name in _OVERRIDDEN_PACKAGED
        if is_custom:
            self._del_btn.config(state=tk.NORMAL, text="Delete")
        elif is_overridden:
            self._del_btn.config(state=tk.NORMAL, text="Reset")
        else:
            self._del_btn.config(state=tk.DISABLED, text="Delete")

    # ------------------------------------------------------------------
    # Custom missile management
    # ------------------------------------------------------------------
    def _refresh_missile_list(self, select_name=None):
        """Rebuild the combobox values from the current MISSILE_DB."""
        names = list(MISSILE_DB.keys())
        self._missile_cb.configure(values=names)
        target = select_name or self._missile_var.get()
        if target not in names:
            target = names[0]
        self._missile_var.set(target)
        self._on_missile_changed()

    def _on_missile_saved(self, p):
        """Callback invoked by MissileDialog when the user clicks Save."""
        name = p.name
        MISSILE_DB[name] = lambda _p=p: _p
        if name in _PACKAGED_NAMES:
            _OVERRIDDEN_PACKAGED.add(name)
        _save_custom_missiles()
        self._refresh_missile_list(select_name=name)
        self._status_var.set(f"Missile '{name}' saved.")

    def _new_missile(self):
        MissileDialog(self, on_save=self._on_missile_saved)

    def _edit_missile(self):
        name = self._missile_var.get()
        MissileDialog(self, on_save=self._on_missile_saved, existing_name=name)

    def _delete_missile(self):
        name = self._missile_var.get()
        if name in _OVERRIDDEN_PACKAGED:
            # Reset overridden built-in missile back to its packaged defaults
            if not messagebox.askyesno(
                    "Reset missile",
                    f"Reset '{name}' to its built-in defaults?",
                    parent=self):
                return
            MISSILE_DB[name] = _PACKAGED_ORIGINALS[name]
            _OVERRIDDEN_PACKAGED.discard(name)
            _save_custom_missiles()
            self._refresh_missile_list(select_name=name)
            self._status_var.set(f"Missile '{name}' reset to defaults.")
        elif name not in _PACKAGED_NAMES:
            if not messagebox.askyesno(
                    "Delete missile",
                    f"Permanently delete '{name}'?",
                    parent=self):
                return
            del MISSILE_DB[name]
            _save_custom_missiles()
            self._refresh_missile_list()
            self._status_var.set(f"Missile '{name}' deleted.")

    def _update_params_text(self, p=None):
        if p is None:
            p = get_missile(self._missile_var.get())

        lines = [
            f"{'Name:':<18}{p.name}",
            f"{'Mass (launch):':<18}{p.mass_initial:,.0f} kg",
            f"{'Propellant:':<18}{p.mass_propellant:,.0f} kg",
            f"{'Mass (burnout):':<18}{p.mass_final:,.0f} kg",
            f"{'Diameter:':<18}{p.diameter_m:.2f} m",
            f"{'Length:':<18}{p.length_m:.2f} m",
            f"{'Thrust (vac):':<18}{p.thrust_N/1000:,.0f} kN",
            f"{'Burn time:':<18}{p.burn_time_s:.0f} s",
            f"{'Isp:':<18}{p.isp_s:.0f} s",
            f"{'T/W ratio:':<18}{p.thrust_N/(p.mass_initial*9.81):.2f}",
        ]
        if p.coast_time_s > 0:
            lines.append(f"{'Coast after S1:':<18}{p.coast_time_s:.0f} s")
        sn, node = 2, p.stage2
        while node is not None:
            lines += [
                "─" * 28,
                f"Stage {sn}:",
                f"{'  Mass:':<18}{node.mass_initial:,.0f} kg",
                f"{'  Propellant:':<18}{node.mass_propellant:,.0f} kg",
                f"{'  Thrust (vac):':<18}{node.thrust_N/1000:,.0f} kN",
                f"{'  Burn time:':<18}{node.burn_time_s:.0f} s",
                f"{'  Isp:':<18}{node.isp_s:.0f} s",
            ]
            if node.stage2 is not None and node.coast_time_s > 0:
                lines.append(
                    f"{'  Coast after:':<18}{node.coast_time_s:.0f} s")
            sn  += 1
            node = node.stage2

        txt = "\n".join(lines)
        self._params_text.config(state=tk.NORMAL)
        self._params_text.delete("1.0", tk.END)
        self._params_text.insert(tk.END, txt)
        self._params_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    # Aim at target
    # ------------------------------------------------------------------
    def _aim_at_target(self):
        """
        Compute great-circle azimuth from launch to target and set the
        cutoff time to hit the target range (using aim_missile bisection).
        """
        try:
            lat1_dd = float(self._launch_lat.get())
            lon1_dd = float(self._launch_lon.get())
            lat2_dd = float(self._target_lat.get())
            lon2_dd = float(self._target_lon.get())

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

            missile = get_missile(self._missile_var.get())
            la  = float(self._loft_angle_var.get())
            lar = float(self._loft_rate_var.get())
            threading.Thread(
                target=self._aim_thread,
                args=(missile, lat1_dd, lon1_dd, az, rng_km, la, lar),
                daemon=True,
            ).start()

        except Exception as e:
            messagebox.showerror("Aim error", str(e))

    def _aim_thread(self, missile, lat, lon, az, rng_km, la, lar):
        try:
            cutoff = aim_missile(missile, lat, lon, az, rng_km,
                                 loft_angle_deg=la,
                                 loft_angle_rate_deg_s=lar)
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
        missile = get_missile(self._missile_var.get())
        lat     = float(self._launch_lat.get())
        lon     = float(self._launch_lon.get())
        az      = float(self._azimuth_var.get())
        cutoff_str = self._cutoff_var.get().strip()
        cutoff  = float(cutoff_str) if cutoff_str else None
        la      = float(self._loft_angle_var.get())
        lar     = float(self._loft_rate_var.get())
        return missile, lat, lon, az, cutoff, la, lar

    def _open_sweep(self):
        ParametricSweepDialog(self)

    def _run_flyout(self):
        if self._running:
            return
        try:
            missile, lat, lon, az, cutoff, la, lar = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self._running = True
        self._status_var.set("Running simulation…")
        threading.Thread(
            target=self._run_thread,
            args=(missile, lat, lon, az, cutoff, la, lar, False),
            daemon=True,
        ).start()

    def _maximize_range(self):
        if self._running:
            return
        try:
            missile, lat, lon, az, _, la, lar = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self._running = True
        self._status_var.set("Optimising for maximum range…")
        threading.Thread(
            target=self._run_thread,
            args=(missile, lat, lon, az, None, la, lar, True),
            daemon=True,
        ).start()

    def _run_thread(self, missile, lat, lon, az, cutoff, la, lar, maximise):
        try:
            if maximise:
                result = maximize_range(missile, lat, lon, az)
            else:
                result = integrate_trajectory(
                    missile, lat, lon, az,
                    loft_angle_deg=la,
                    loft_angle_rate_deg_s=lar,
                    cutoff_time_s=cutoff)
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
        if 'optimal_loft_angle_deg' in r:
            self._loft_angle_var.set(f"{r['optimal_loft_angle_deg']:.4f}")
            self._loft_rate_var .set(f"{r['optimal_loft_rate_deg_s']:.3f}")

        rng_km    = r['range_km']
        rng_nm    = rng_km / 1.852
        rng_mi    = rng_km / 1.60934
        apogee_km = r['apogee_km']

        tof_s       = r['time_of_flight_s']
        imp_spd_kms = r['impact_speed_ms'] / 1000.0
        apo_lat     = r['apogee_lat_deg']
        apo_lon     = r['apogee_lon_deg']

        self._res_range_km .set(f"Range (km):    {rng_km:>8.1f}")
        self._res_range_nm .set(f"Range (nmi):   {rng_nm:>8.1f}")
        self._res_range_mi .set(f"Range (miles): {rng_mi:>8.1f}")
        self._res_apogee   .set(f"Apogee (km):   {apogee_km:>8.1f}")
        self._res_apogee_ll.set(
            f"Apogee loc:  {apo_lat:.2f}°N  {apo_lon:.2f}°E")
        self._res_impact   .set(
            f"Impact:      {r['impact_lat']:.2f}°N  {r['impact_lon']:.2f}°E")
        self._res_tof      .set(f"Flight time:   {tof_s:>7.0f} s")
        self._res_imp_spd  .set(f"Impact speed: {imp_spd_kms:>7.2f} km/s")

        units = self._units_var.get()
        scale_map = {"km": (1.0, "km"), "nm": (1/1.852, "nmi"), "mi": (1/1.60934, "mi")}
        scale, ulbl = scale_map[units]

        self._status_var.set(
            f"Done.  Range: {rng_km*scale:.1f} {ulbl}  |  "
            f"Apogee: {apogee_km*scale:.1f} {ulbl}  |  "
            f"ToF: {tof_s:.0f} s  |  "
            f"Impact: {r['impact_lat']:.2f}°N, {r['impact_lon']:.2f}°E  |  "
            f"Impact spd: {imp_spd_kms:.2f} km/s"
        )
        self._plot_results(r, scale, ulbl)

    def _plot_results(self, r, scale, ulbl):
        t   = r['t']
        alt = r['alt'] / 1000.0 * scale
        spd = r['speed'] / 1000.0            # always km/s for speed axis
        rng = r['range'] / 1000.0 * scale
        lat = r['lat']
        lon = r['lon']

        for ax in (self._ax_alt, self._ax_spd, self._ax_traj, self._ax_trk):
            ax.cla()
            ax.grid(True, alpha=0.35)
            ax.tick_params(labelsize=7)

        # Altitude vs time
        self._ax_alt.plot(t, alt, color='royalblue', linewidth=1.5)
        self._ax_alt.set_xlabel("Time (s)", fontsize=8)
        self._ax_alt.set_ylabel(f"Altitude ({ulbl})", fontsize=8)
        self._ax_alt.set_title("Altitude vs Time", fontsize=9)
        self._ax_alt.fill_between(t, 0, alt, alpha=0.12, color='royalblue')

        # Speed vs time
        self._ax_spd.plot(t, spd, color='firebrick', linewidth=1.5)
        self._ax_spd.set_xlabel("Time (s)", fontsize=8)
        self._ax_spd.set_ylabel("Speed (km/s)", fontsize=8)
        self._ax_spd.set_title("Speed vs Time", fontsize=9)

        # Altitude vs range (trajectory profile)
        self._ax_traj.plot(rng, alt, color='seagreen', linewidth=1.5)
        self._ax_traj.set_xlabel(f"Downrange ({ulbl})", fontsize=8)
        self._ax_traj.set_ylabel(f"Altitude ({ulbl})", fontsize=8)
        self._ax_traj.set_title("Altitude vs Range", fontsize=9)
        self._ax_traj.fill_between(rng, 0, alt, alpha=0.12, color='seagreen')

        # Ground track
        self._ax_trk.plot(lon, lat, color='black', linewidth=1.2)
        self._ax_trk.plot(lon[0],  lat[0],  'go', markersize=7,
                          label="Launch", zorder=5)
        self._ax_trk.plot(lon[-1], lat[-1], 'r*', markersize=9,
                          label="Impact", zorder=5)
        self._ax_trk.set_xlabel("Longitude (°E)", fontsize=8)
        self._ax_trk.set_ylabel("Latitude (°N)", fontsize=8)
        self._ax_trk.set_title("Ground Track", fontsize=9)
        self._ax_trk.legend(fontsize=7)

        self._fig.tight_layout(pad=2.8)
        self._canvas.draw_idle()
        self._canvas.flush_events()

    # ------------------------------------------------------------------
    # File / Help actions
    # ------------------------------------------------------------------
    def _save_trajectory(self):
        if self._result is None:
            messagebox.showinfo("No data", "Run a simulation first.")
            return
        from tkinter.filedialog import asksaveasfilename
        path = asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save trajectory",
        )
        if not path:
            return
        r = self._result
        header = "time_s,lat_deg,lon_deg,alt_m,speed_ms,range_km"
        data = np.column_stack([r['t'], r['lat'], r['lon'],
                                 r['alt'], r['speed'], r['range'] / 1000.0])
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        self._status_var.set(f"Saved: {path}")

    def _show_about(self):
        messagebox.showinfo(
            "About GUI Missile Flyout V1.1",
            "GUI Missile Flyout V1.1 — Python port\n\n"
            "Original MATLAB application by Geoffrey Forden\n"
            "Published: Sci. & Global Security 15 (2007)\n\n"
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
