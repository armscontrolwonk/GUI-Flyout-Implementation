"""
GUI Missile Flyout — Python/tkinter port of Forden's MATLAB GUIDE application.

Layout mirrors the original:
  Left panel  : inputs (missile type, launch/target coordinates, cutoff time)
  Right panel : tabbed plots (trajectory, altitude, speed, range footprint)
  Bottom bar  : status / range output
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from missile_models import MISSILE_DB, get_missile
from trajectory import integrate_trajectory, maximize_range
from coordinates import range_between


# ---------------------------------------------------------------------------
# Helper: DMS entry widget
# ---------------------------------------------------------------------------
class DMSEntry(ttk.Frame):
    """Degrees / Minutes / Seconds entry that returns decimal degrees."""

    def __init__(self, master, label, **kw):
        super().__init__(master, **kw)
        ttk.Label(self, text=label, width=12).pack(side=tk.LEFT)
        self._deg = tk.StringVar(value="0")
        self._min = tk.StringVar(value="0")
        self._sec = tk.StringVar(value="0.0")
        for var, lbl in [(self._deg, "°"), (self._min, "'"), (self._sec, '"')]:
            ttk.Entry(self, textvariable=var, width=6).pack(side=tk.LEFT, padx=1)
            ttk.Label(self, text=lbl).pack(side=tk.LEFT)

    def get_decimal(self):
        d = float(self._deg.get())
        m = float(self._min.get())
        s = float(self._sec.get())
        sign = -1 if d < 0 else 1
        return sign * (abs(d) + m/60 + s/3600)

    def set_decimal(self, val):
        sign = -1 if val < 0 else 1
        val = abs(val)
        d = int(val)
        m = int((val - d) * 60)
        s = (val - d - m/60) * 3600
        self._deg.set(str(sign * d))
        self._min.set(str(m))
        self._sec.set(f"{s:.2f}")


# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------
class MissileFlyoutApp(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("GUI Missile Flyout  (Python port of Forden v1.1)")
        self.geometry("1200x750")
        self.resizable(True, True)

        self._result = None          # last trajectory result
        self._running = False

        self._build_menu()
        self._build_ui()

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
    def _build_ui(self):
        # ── top frame ──────────────────────────────────────────────────
        top = ttk.Frame(self)
        top.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        # ── left panel: inputs ─────────────────────────────────────────
        left = ttk.LabelFrame(top, text="Inputs", width=320)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        left.pack_propagate(False)

        self._build_input_panel(left)

        # ── right panel: plots ─────────────────────────────────────────
        right = ttk.Frame(top)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._build_plot_panel(right)

        # ── bottom status bar ──────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready.")
        ttk.Label(self, textvariable=self._status_var,
                  relief=tk.SUNKEN, anchor=tk.W).pack(
            side=tk.BOTTOM, fill=tk.X, padx=4, pady=2)

    # ------------------------------------------------------------------
    def _build_input_panel(self, parent):
        pad = dict(padx=6, pady=3, sticky=tk.W)

        # Missile type
        ttk.Label(parent, text="Missile type:").grid(row=0, column=0, **pad)
        self._missile_var = tk.StringVar(value=list(MISSILE_DB.keys())[0])
        cb = ttk.Combobox(parent, textvariable=self._missile_var,
                          values=list(MISSILE_DB.keys()), state="readonly", width=18)
        cb.grid(row=0, column=1, **pad)
        cb.bind("<<ComboboxSelected>>", self._on_missile_changed)

        # Units
        ttk.Label(parent, text="Units:").grid(row=1, column=0, **pad)
        self._units_var = tk.StringVar(value="km")
        for col, (val, lbl) in enumerate([("km", "km"), ("nm", "nmi"), ("mi", "mi")]):
            ttk.Radiobutton(parent, text=lbl, variable=self._units_var,
                            value=val).grid(row=1, column=col+1, padx=2)

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=2, columnspan=3, sticky=tk.EW, pady=4)

        # Launch coordinates
        ttk.Label(parent, text="Launch site", font=("", 9, "bold")).grid(
            row=3, columnspan=2, **pad)

        self._launch_lat = self._coord_row(parent, "Latitude:",  row=4)
        self._launch_lon = self._coord_row(parent, "Longitude:", row=5)

        ttk.Label(parent, text="Azimuth (°):").grid(row=6, column=0, **pad)
        self._azimuth_var = tk.StringVar(value="0.0")
        ttk.Entry(parent, textvariable=self._azimuth_var, width=10).grid(
            row=6, column=1, **pad)

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=7, columnspan=3, sticky=tk.EW, pady=4)

        # Target coordinates
        ttk.Label(parent, text="Target", font=("", 9, "bold")).grid(
            row=8, columnspan=2, **pad)

        self._target_lat = self._coord_row(parent, "Latitude:",  row=9)
        self._target_lon = self._coord_row(parent, "Longitude:", row=10)

        ttk.Button(parent, text="Aim at target",
                   command=self._aim_at_target).grid(row=11, columnspan=2, pady=4)

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=12, columnspan=3, sticky=tk.EW, pady=4)

        # Engine cutoff
        ttk.Label(parent, text="Cutoff time (s):").grid(row=13, column=0, **pad)
        self._cutoff_var = tk.StringVar(value="")
        ttk.Entry(parent, textvariable=self._cutoff_var, width=10).grid(
            row=13, column=1, **pad)
        ttk.Label(parent, text="(blank = full burn)").grid(row=14, column=1,
                                                            sticky=tk.W, padx=6)

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=15, columnspan=3, sticky=tk.EW, pady=4)

        # Buttons
        ttk.Button(parent, text="Run Flyout",
                   command=self._run_flyout).grid(row=16, columnspan=2, pady=4,
                                                  ipadx=10)
        ttk.Button(parent, text="Maximize Range",
                   command=self._maximize_range).grid(row=17, columnspan=2,
                                                      pady=2, ipadx=4)

        ttk.Separator(parent, orient=tk.HORIZONTAL).grid(
            row=18, columnspan=3, sticky=tk.EW, pady=4)

        # Results display
        self._range_var  = tk.StringVar(value="Range:   —")
        self._apogee_var = tk.StringVar(value="Apogee:  —")
        ttk.Label(parent, textvariable=self._range_var,
                  font=("Courier", 10)).grid(row=19, columnspan=2, **pad)
        ttk.Label(parent, textvariable=self._apogee_var,
                  font=("Courier", 10)).grid(row=20, columnspan=2, **pad)

    def _coord_row(self, parent, label, row):
        """Return a StringVar for decimal-degree entry with DMS label."""
        ttk.Label(parent, text=label).grid(row=row, column=0, padx=6,
                                            pady=2, sticky=tk.W)
        var = tk.StringVar(value="0.0")
        ttk.Entry(parent, textvariable=var, width=14).grid(
            row=row, column=1, padx=6, pady=2, sticky=tk.W)
        return var

    # ------------------------------------------------------------------
    def _build_plot_panel(self, parent):
        self._notebook = ttk.Notebook(parent)
        self._notebook.pack(fill=tk.BOTH, expand=True)

        self._fig = Figure(figsize=(8, 5), dpi=96)
        self._axes = {
            "Altitude":   self._fig.add_subplot(221),
            "Speed":      self._fig.add_subplot(222),
            "Trajectory": self._fig.add_subplot(223),
            "Footprint":  self._fig.add_subplot(224),
        }
        self._fig.tight_layout(pad=2.5)

        canvas_frame = ttk.Frame(self._notebook)
        self._notebook.add(canvas_frame, text="Plots")

        self._canvas = FigureCanvasTkAgg(self._fig, master=canvas_frame)
        self._canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self._canvas, canvas_frame)
        toolbar.update()

        # Parameters tab
        params_frame = ttk.Frame(self._notebook)
        self._notebook.add(params_frame, text="Missile Parameters")
        self._params_text = tk.Text(params_frame, width=60, height=20,
                                    font=("Courier", 10))
        self._params_text.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._update_params_tab()

    # ------------------------------------------------------------------
    def _on_missile_changed(self, _event=None):
        self._update_params_tab()
        p = get_missile(self._missile_var.get())
        total = p.burn_time_s + (p.stage2.burn_time_s if p.stage2 else 0)
        self._cutoff_var.set(str(int(total)))

    def _update_params_tab(self):
        p = get_missile(self._missile_var.get())
        txt = (
            f"Missile:          {p.name}\n"
            f"Launch mass:      {p.mass_initial:,.0f} kg\n"
            f"Propellant mass:  {p.mass_propellant:,.0f} kg\n"
            f"Burnout mass:     {p.mass_final:,.0f} kg\n"
            f"Diameter:         {p.diameter_m:.2f} m\n"
            f"Length:           {p.length_m:.2f} m\n"
            f"Thrust (vac):     {p.thrust_N:,.0f} N\n"
            f"Stage-1 burn:     {p.burn_time_s:.1f} s\n"
            f"Isp (stage 1):    {p.isp_s:.0f} s\n"
        )
        if p.stage2:
            p2 = p.stage2
            txt += (
                f"\n--- Stage 2 ---\n"
                f"Mass:             {p2.mass_initial:,.0f} kg\n"
                f"Propellant:       {p2.mass_propellant:,.0f} kg\n"
                f"Thrust (vac):     {p2.thrust_N:,.0f} N\n"
                f"Stage-2 burn:     {p2.burn_time_s:.1f} s\n"
                f"Isp (stage 2):    {p2.isp_s:.0f} s\n"
            )
        self._params_text.config(state=tk.NORMAL)
        self._params_text.delete("1.0", tk.END)
        self._params_text.insert(tk.END, txt)
        self._params_text.config(state=tk.DISABLED)

    # ------------------------------------------------------------------
    def _aim_at_target(self):
        """Set azimuth from launch to target (great-circle bearing)."""
        try:
            lat1 = np.radians(float(self._launch_lat.get()))
            lon1 = np.radians(float(self._launch_lon.get()))
            lat2 = np.radians(float(self._target_lat.get()))
            lon2 = np.radians(float(self._target_lon.get()))

            dlon = lon2 - lon1
            x = np.sin(dlon) * np.cos(lat2)
            y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
            az = np.degrees(np.arctan2(x, y)) % 360
            self._azimuth_var.set(f"{az:.2f}")
            rng = range_between(lat1, lon1, lat2, lon2) / 1000
            self._status_var.set(f"Target range: {rng:.1f} km  |  Azimuth: {az:.1f}°")
        except ValueError as e:
            messagebox.showerror("Input error", str(e))

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
                self._cutoff_var.set(f"{result['optimal_cutoff_s']:.1f}")
            else:
                result = integrate_trajectory(
                    missile, lat, lon, az, cutoff_time_s=cutoff)
            self._result = result
            self.after(0, self._on_result_ready)
        except Exception as e:
            self.after(0, lambda: messagebox.showerror("Simulation error", str(e)))
        finally:
            self._running = False

    def _on_result_ready(self):
        r = self._result
        units = self._units_var.get()
        scale = {"km": 1.0, "nm": 1/1.852, "mi": 1/1.60934}[units]

        rng_disp    = r['range_km'] * scale
        apogee_disp = r['apogee_km'] * scale

        self._range_var.set(f"Range:   {rng_disp:,.1f} {units}")
        self._apogee_var.set(f"Apogee:  {apogee_disp:,.1f} {units}")
        self._status_var.set(
            f"Done.  Range: {rng_disp:.1f} {units}  |  "
            f"Apogee: {apogee_disp:.1f} {units}  |  "
            f"Impact: {r['impact_lat']:.2f}°N, {r['impact_lon']:.2f}°E"
        )
        self._plot_results(r, units, scale)

    # ------------------------------------------------------------------
    def _plot_results(self, r, units, scale):
        t   = r['t']
        alt = r['alt'] / 1000 * scale   # km → display units
        spd = r['speed'] / 1000         # m/s → km/s
        rng = r['range'] / 1000 * scale
        lat = r['lat']
        lon = r['lon']

        for ax in self._axes.values():
            ax.cla()

        ax_alt = self._axes["Altitude"]
        ax_alt.plot(t, alt, 'b-', linewidth=1.5)
        ax_alt.set_xlabel("Time (s)")
        ax_alt.set_ylabel(f"Altitude ({units})")
        ax_alt.set_title("Altitude vs Time")
        ax_alt.grid(True, alpha=0.4)

        ax_spd = self._axes["Speed"]
        ax_spd.plot(t, spd, 'r-', linewidth=1.5)
        ax_spd.set_xlabel("Time (s)")
        ax_spd.set_ylabel("Speed (km/s)")
        ax_spd.set_title("Speed vs Time")
        ax_spd.grid(True, alpha=0.4)

        ax_traj = self._axes["Trajectory"]
        ax_traj.plot(rng, alt, 'g-', linewidth=1.5)
        ax_traj.set_xlabel(f"Downrange ({units})")
        ax_traj.set_ylabel(f"Altitude ({units})")
        ax_traj.set_title("Trajectory")
        ax_traj.grid(True, alpha=0.4)

        ax_fp = self._axes["Footprint"]
        ax_fp.plot(lon, lat, 'k-', linewidth=1.5)
        ax_fp.plot(lon[0], lat[0], 'go', markersize=8, label="Launch")
        ax_fp.plot(lon[-1], lat[-1], 'r*', markersize=10, label="Impact")
        ax_fp.set_xlabel("Longitude (°)")
        ax_fp.set_ylabel("Latitude (°)")
        ax_fp.set_title("Ground Track")
        ax_fp.legend(fontsize=8)
        ax_fp.grid(True, alpha=0.4)

        self._fig.tight_layout(pad=2.5)
        self._canvas.draw()

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
                                 r['alt'], r['speed'], r['range']/1000])
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        self._status_var.set(f"Trajectory saved to {path}")

    def _show_about(self):
        messagebox.showinfo(
            "About",
            "GUI Missile Flyout — Python port\n\n"
            "Original MATLAB tool by Gerald Forden\n"
            "Python implementation based on open physics\n\n"
            "3-DOF trajectory integration with:\n"
            "  • COESA 1976 standard atmosphere\n"
            "  • WGS-84 J2 gravity\n"
            "  • Coriolis & centrifugal corrections\n"
            "  • Gravity-turn guidance\n"
        )


# ---------------------------------------------------------------------------
def main():
    app = MissileFlyoutApp()
    app.mainloop()


if __name__ == "__main__":
    main()
