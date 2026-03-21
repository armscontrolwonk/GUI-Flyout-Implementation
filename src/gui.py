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

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from missile_models import MISSILE_DB, get_missile
from trajectory import integrate_trajectory, maximize_range, aim_missile
from coordinates import range_between


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
        cb = ttk.Combobox(mf, textvariable=self._missile_var,
                          values=list(MISSILE_DB.keys()),
                          state="readonly", width=24)
        cb.pack(padx=6, pady=4)
        cb.bind("<<ComboboxSelected>>", self._on_missile_changed)

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
            side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 3), ipady=4)
        ttk.Button(btn_frame, text="Maximize Range",
                   command=self._maximize_range).pack(
            side=tk.LEFT, expand=True, fill=tk.X, padx=(3, 0), ipady=4)

        # ── Results display ───────────────────────────────────────────
        rf = ttk.LabelFrame(parent, text="Results")
        rf.pack(fill=tk.X, padx=6, pady=3)

        self._res_range_km  = tk.StringVar(value="Range (km):    —")
        self._res_range_nm  = tk.StringVar(value="Range (nmi):   —")
        self._res_range_mi  = tk.StringVar(value="Range (miles): —")
        self._res_apogee    = tk.StringVar(value="Apogee (km):   —")
        self._res_impact    = tk.StringVar(value="Impact:        —")

        for var in (self._res_range_km, self._res_range_nm,
                    self._res_range_mi, self._res_apogee, self._res_impact):
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
        total = p.burn_time_s + (p.stage2.burn_time_s if p.stage2 else 0)
        self._cutoff_var.set(str(int(total)))
        self._update_params_text(p)

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
        if p.stage2:
            p2 = p.stage2
            lines += [
                "─" * 28,
                "Stage 2:",
                f"{'  Mass:':<18}{p2.mass_initial:,.0f} kg",
                f"{'  Propellant:':<18}{p2.mass_propellant:,.0f} kg",
                f"{'  Thrust (vac):':<18}{p2.thrust_N/1000:,.0f} kN",
                f"{'  Burn time:':<18}{p2.burn_time_s:.0f} s",
                f"{'  Isp:':<18}{p2.isp_s:.0f} s",
            ]

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

            # Compute cutoff in a background thread
            missile = get_missile(self._missile_var.get())
            threading.Thread(
                target=self._aim_thread,
                args=(missile, lat1_dd, lon1_dd, az, rng_km),
                daemon=True,
            ).start()

        except Exception as e:
            messagebox.showerror("Aim error", str(e))

    def _aim_thread(self, missile, lat, lon, az, rng_km):
        try:
            cutoff = aim_missile(missile, lat, lon, az, rng_km)
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
        return missile, lat, lon, az, cutoff

    def _run_flyout(self):
        if self._running:
            return
        try:
            missile, lat, lon, az, cutoff = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self._running = True
        self._status_var.set("Running simulation…")
        threading.Thread(
            target=self._run_thread,
            args=(missile, lat, lon, az, cutoff, False),
            daemon=True,
        ).start()

    def _maximize_range(self):
        if self._running:
            return
        try:
            missile, lat, lon, az, _ = self._get_inputs()
        except ValueError as e:
            messagebox.showerror("Input error", str(e))
            return
        self._running = True
        self._status_var.set("Optimising for maximum range…")
        threading.Thread(
            target=self._run_thread,
            args=(missile, lat, lon, az, None, True),
            daemon=True,
        ).start()

    def _run_thread(self, missile, lat, lon, az, cutoff, maximise):
        try:
            if maximise:
                result = maximize_range(missile, lat, lon, az)
                self.after(0, lambda: self._cutoff_var.set(
                    f"{result['optimal_cutoff_s']:.1f}"))
            else:
                result = integrate_trajectory(
                    missile, lat, lon, az, cutoff_time_s=cutoff)
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
        rng_km    = r['range_km']
        rng_nm    = rng_km / 1.852
        rng_mi    = rng_km / 1.60934
        apogee_km = r['apogee_km']

        self._res_range_km.set( f"Range (km):    {rng_km:>8.1f}")
        self._res_range_nm.set( f"Range (nmi):   {rng_nm:>8.1f}")
        self._res_range_mi.set( f"Range (miles): {rng_mi:>8.1f}")
        self._res_apogee.set(   f"Apogee (km):   {apogee_km:>8.1f}")
        self._res_impact.set(
            f"Impact: {r['impact_lat']:.2f}°N  {r['impact_lon']:.2f}°E")

        units = self._units_var.get()
        scale_map = {"km": (1.0, "km"), "nm": (1/1.852, "nmi"), "mi": (1/1.60934, "mi")}
        scale, ulbl = scale_map[units]

        self._status_var.set(
            f"Done.  Range: {rng_km*scale:.1f} {ulbl}  |  "
            f"Apogee: {apogee_km*scale:.1f} {ulbl}  |  "
            f"Impact: {r['impact_lat']:.2f}°N, {r['impact_lon']:.2f}°E"
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
        self._canvas.draw()

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
            "Published: Sci. & Global Security 9 (2001)\n\n"
            "3-DOF trajectory integration:\n"
            "  • COESA 1976 standard atmosphere\n"
            "  • WGS-84 J2 gravity (ECEF)\n"
            "  • Coriolis & centrifugal corrections\n"
            "  • Fixed-pitch guidance (Forden note 7)\n"
            "  • 2-stage missile support\n"
        )


# ---------------------------------------------------------------------------
def main():
    app = MissileFlyoutApp()
    app.mainloop()


if __name__ == "__main__":
    main()
