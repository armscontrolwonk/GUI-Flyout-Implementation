#!/usr/bin/env python3
"""
One-off smoke test: AEOLUS/SWERVE-derivative Mars entry vs Murbach 1997.
Entry: v=7 km/s, h=125 km, gamma=-15 deg  (paper p.9)
Vehicle: m=178 kg, 5-deg half-cone + wings, L/D>1  (paper p.5)
Reference trajectory: Fig 12 (alt vs range, pull-up at 20 km ~600-700 km)
                      Fig 13 (alt vs time, cruise at 20 km, ~4000 s total)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Mars constants ──────────────────────────────────────────────────────────
GM_M   = 4.282837e13        # m³/s²
R_M    = 3_389_500.0        # m  (mean radius)
OMEGA  = 7.0882e-5          # rad/s  (Mars sidereal rotation)

def mars_rho(h):
    """Mars Global Reference Atmosphere, exponential fit (equatorial)."""
    # Two-layer exponential: better fit to MGRAM than single scale height.
    if np.isscalar(h):
        if   h < 7_000:  return 0.0157 * np.exp(-h / 9_300)
        elif h < 50_000: return 0.0200 * np.exp(-h / 11_100)
        else:            return 0.0200 * np.exp(-50_000/11_100) * np.exp(-(h-50_000)/7_200)
    h = np.asarray(h, float)
    rho = np.where(h < 7_000,
                   0.0157 * np.exp(-h / 9_300),
                   np.where(h < 50_000,
                            0.0200 * np.exp(-h / 11_100),
                            0.0200 * np.exp(-50_000/11_100) * np.exp(-(h-50_000)/7_200)))
    return rho

def mars_g(h):
    return GM_M / (R_M + h)**2

# ── Vehicle ─────────────────────────────────────────────────────────────────
MASS = 178.0        # kg
# L/D from SWERVE PNS tables (alpha=10, M=12):
#   CN=0.476, estimating CA~0.15 for hypersonic cone-wing:
#   L/D = (CN cos a - CA sin a)/(CN sin a + CA cos a) ≈ 1.9
# Paper says "L/D > 1"; we'll bracket L/D = 1.5, 2.0, 2.5
# beta = m/(CD*Aref): calibrate to pull-up at ~20 km, ~600 km downrange.
# With Aref estimated from 2.75m 5-deg cone base:
#   A_ref = pi*(2.75*tan(5 deg))^2 ≈ 0.18 m²
# CD ~ 0.23 at alpha=10 → beta ~ 178/(0.23*0.18) ≈ 4300 kg/m²
# Adjust to 4000 kg/m² (gives pull-up depth ≈ 20 km in test runs).
BETA = 4_000.0      # kg/m²   — tuned to match Fig 12 pull-up depth
A_D  = MASS / BETA  # CD * Aref  (m²)

# ── 2-D spherical-planet EOM ─────────────────────────────────────────────────
# State: [h (m), s (m), v (m/s), gamma (rad)]
# h     = altitude above surface
# s     = downrange distance along surface arc
# v     = airspeed (assume thin-atmosphere → ≈ inertial speed at Mars low-omega)
# gamma = flight-path angle (+ve climbing)
# Glider lift direction: perpendicular to velocity, opposing sinking.

def eom(t, state, LD):
    h, s, v, gam = state
    if h < 0 or v < 50:
        return [0, 0, 0, 0]
    rho = mars_rho(h)
    g   = mars_g(h)
    r   = R_M + h
    q   = 0.5 * rho * v**2
    D   = q * A_D
    L   = D * LD

    dv   = -(D / MASS) - g * np.sin(gam)
    dgam = (L / MASS - (g - v**2 / r) * np.cos(gam)) / max(v, 50)
    dh   = v * np.sin(gam)
    ds   = v * np.cos(gam) * R_M / r
    return [dh, ds, dv, dgam]

def hit_ground(t, state, LD):
    return state[0]
hit_ground.terminal  = True
hit_ground.direction = -1

# ── Entry conditions ─────────────────────────────────────────────────────────
V0    = 7_000.0
H0    = 125_000.0
GAM0  = np.radians(-15.0)
y0    = [H0, 0.0, V0, GAM0]

# ── Run for three L/D values ─────────────────────────────────────────────────
LD_cases = [1.5, 2.0, 2.5]
colors    = ['#1f77b4', '#d62728', '#2ca02c']
results   = {}

print(f"{'L/D':>5}  {'Range km':>10}  {'Time s':>8}  {'Vimpact km/s':>13}  {'Min alt km':>10}")
print("-" * 55)

for LD in LD_cases:
    sol = solve_ivp(eom, [0, 5000], y0,
                    args=(LD,),
                    events=hit_ground,
                    max_step=2.0, rtol=1e-7, atol=1e-8)
    t    = sol.t
    h_km = sol.y[0] / 1e3
    s_km = sol.y[1] / 1e3
    v    = sol.y[2]
    results[LD] = (t, h_km, s_km, v)
    print(f"{LD:>5.1f}  {s_km[-1]:>10.0f}  {t[-1]:>8.0f}  {v[-1]/1000:>13.2f}  {h_km.min():>10.1f}")

# ── Figures matching Murbach 1997 ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("AEOLUS Mars entry — Thrusty EOM vs Murbach 1997\n"
             f"m={MASS} kg, β={BETA:.0f} kg/m², entry: {V0/1000:.0f} km/s / "
             f"{H0/1000:.0f} km / γ={np.degrees(GAM0):.0f}°",
             fontsize=11)

# ── Fig 12 analog: alt vs downrange ─────────────────────────────────────────
ax1 = axes[0]
for LD, col in zip(LD_cases, colors):
    t, h_km, s_km, v = results[LD]
    mask = s_km <= 800
    ax1.plot(s_km[mask], h_km[mask], color=col, lw=2, label=f'L/D = {LD}')
ax1.axhline(20, ls='--', color='gray', lw=0.8, alpha=0.7, label='20 km (pull-up)')
ax1.set_xlabel('Downrange distance, km')
ax1.set_ylabel('Altitude, km')
ax1.set_title('Fig 12 analog — Flight Profile Prior to Pull-up')
ax1.set_xlim(0, 800)
ax1.set_ylim(0, 140)
ax1.grid(True, ls=':', alpha=0.5)
ax1.legend(fontsize=9)

# ── Fig 13 analog: alt vs time ───────────────────────────────────────────────
ax2 = axes[1]
for LD, col in zip(LD_cases, colors):
    t, h_km, s_km, v = results[LD]
    ax2.plot(t, h_km, color=col, lw=2, label=f'L/D = {LD}')
ax2.axhline(20, ls='--', color='gray', lw=0.8, alpha=0.7, label='20 km cruise')
ax2.set_xlabel('Flight time, s')
ax2.set_ylabel('Altitude, km')
ax2.set_title('Fig 13 analog — Full Trajectory')
ax2.set_xlim(0, 5000)
ax2.set_ylim(0, 140)
ax2.grid(True, ls=':', alpha=0.5)
ax2.legend(fontsize=9)

plt.tight_layout()
out = '/home/user/GUI-Flyout-Implementation/mars_smoke_test.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nPlot saved → {out}")

# ── Equilibrium cruise speed at 20 km check ──────────────────────────────────
h_cruise = 20_000.0
rho_c    = mars_rho(h_cruise)
g_c      = mars_g(h_cruise)
for LD in LD_cases:
    # Level cruise: L = W → 0.5*rho*v²*CL_A = m*g
    CL_A  = A_D * LD
    v_eq  = np.sqrt(2 * MASS * g_c / (rho_c * CL_A))
    print(f"  L/D={LD}: equilibrium cruise speed at 20 km ≈ {v_eq/1000:.2f} km/s")
