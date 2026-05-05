#!/usr/bin/env python3
"""
Smoke test 2: compare skip-glide vs Acton equilibrium-glide on Mars AEOLUS.
Same entry conditions and vehicle as mars_smoke_test.py.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Mars constants ────────────────────────────────────────────────────────────
GM_M = 4.282837e13
R_M  = 3_389_500.0

def mars_rho(h):
    if np.isscalar(h):
        if   h < 7_000:  return 0.0157 * np.exp(-h / 9_300)
        elif h < 50_000: return 0.0200 * np.exp(-h / 11_100)
        else:            return 0.0200*np.exp(-50_000/11_100)*np.exp(-(h-50_000)/7_200)
    h = np.asarray(h, float)
    return np.where(h < 7_000,
                    0.0157*np.exp(-h/9_300),
                    np.where(h < 50_000,
                             0.0200*np.exp(-h/11_100),
                             0.0200*np.exp(-50_000/11_100)*np.exp(-(h-50_000)/7_200)))

def mars_g(h):
    return GM_M / (R_M + h)**2

# ── Vehicle ───────────────────────────────────────────────────────────────────
MASS = 178.0
BETA = 4_000.0
A_D  = MASS / BETA        # CD * Aref

# ── EOM factory ──────────────────────────────────────────────────────────────
# mode: 'skip' = full lift always
#       'eq'   = Acton equilibrium glide (cap lift once γ≥0)

def make_eom(LD, mode='skip', pullup_g_max=15.0):
    CL_A = A_D * LD
    def eom(t, state):
        h, s, v, gam = state
        if h < 0 or v < 50:
            return [0, 0, 0, 0]
        rho = mars_rho(h)
        g   = mars_g(h)
        r   = R_M + h
        q   = 0.5 * rho * v**2
        D   = q * A_D
        L   = q * CL_A

        if mode == 'eq' and gam >= 0.0:
            # Acton eq-glide: once climbing/level, cap lift so dγ/dt → 0
            eq_lift = MASS * max(0.0, g - v**2 / r)
            L = min(L, eq_lift)

        # g-cap (same as glider_pullup_g_max in Thrusty)
        L = min(L, pullup_g_max * g * MASS)

        dv   = -(D / MASS) - g * np.sin(gam)
        dgam = (L / MASS - (g - v**2/r) * np.cos(gam)) / max(v, 50)
        dh   = v * np.sin(gam)
        ds   = v * np.cos(gam) * R_M / r
        return [dh, ds, dv, dgam]
    return eom

def hit_ground(t, state):
    return state[0]
hit_ground.terminal  = True
hit_ground.direction = -1

# ── Entry conditions ──────────────────────────────────────────────────────────
y0 = [125_000.0, 0.0, 7_000.0, np.radians(-15.0)]

# Run skip and eq-glide for L/D = 2.0 (best Fig-12 match from test 1)
cases = [
    ('Skip-glide  L/D=2.0', make_eom(2.0, 'skip'), '#1f77b4', '--'),
    ('Eq-glide    L/D=2.0', make_eom(2.0, 'eq'),   '#d62728', '-'),
    ('Eq-glide    L/D=1.5', make_eom(1.5, 'eq'),   '#ff7f0e', '-'),
    ('Eq-glide    L/D=2.5', make_eom(2.5, 'eq'),   '#2ca02c', '-'),
]

print(f"{'Case':<22}  {'Range km':>10}  {'Time s':>8}  {'Min alt km':>10}")
print('-' * 58)
results = {}
for label, eom, col, ls in cases:
    sol = solve_ivp(eom, [0, 5000], y0,
                    events=hit_ground,
                    max_step=2.0, rtol=1e-7, atol=1e-8)
    t    = sol.t
    h_km = sol.y[0] / 1e3
    s_km = sol.y[1] / 1e3
    v    = sol.y[2]
    results[label] = (t, h_km, s_km, v, col, ls)
    print(f"{label:<22}  {s_km[-1]:>10.0f}  {t[-1]:>8.0f}  {h_km.min():>10.1f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("AEOLUS Mars entry — Skip-glide vs Acton Equilibrium-glide\n"
             "m=178 kg, β=4000 kg/m², 7 km/s / 125 km / γ=−15°  |  "
             "Murbach ref: pull-up @20 km ~600 km range, cruise ~3600 s",
             fontsize=10)

for ax, xlim, title, xlab in [
        (axes[0], (0, 800),  'Fig 12 — alt vs downrange (first 800 km)', 'Downrange, km'),
        (axes[1], (0, 5000), 'Fig 13 — alt vs time',                     'Time, s')]:
    for label, (t, h_km, s_km, v, col, ls) in results.items():
        xdata = s_km if ax is axes[0] else t
        mask  = xdata <= xlim[1]
        ax.plot(xdata[mask], h_km[mask], color=col, ls=ls, lw=2, label=label)
    ax.axhline(20, ls=':', color='gray', lw=0.8, label='Murbach cruise alt (20 km)')
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 140)
    ax.set_xlabel(xlab)
    ax.set_ylabel('Altitude, km')
    ax.set_title(title)
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend(fontsize=8)

plt.tight_layout()
out = '/home/user/GUI-Flyout-Implementation/mars_smoke_test2.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nPlot saved → {out}')

# ── Extended run: eq-glide L/D=2.0 to natural impact ─────────────────────────
print("\nExtended run (eq-glide L/D=2.0, t_max=12000 s):")
sol_ext = solve_ivp(make_eom(2.0, 'eq'), [0, 12000], y0,
                    events=hit_ground,
                    max_step=2.0, rtol=1e-7, atol=1e-8)
t_e    = sol_ext.t
h_e    = sol_ext.y[0] / 1e3
s_e    = sol_ext.y[1] / 1e3
v_e    = sol_ext.y[2]
print(f"  Impact at t={t_e[-1]:.0f} s,  range={s_e[-1]:.0f} km,  "
      f"v={v_e[-1]/1000:.2f} km/s,  min alt={h_e.min():.1f} km")
print(f"  (Murbach ref: >4000 s, ~12000 km)")

# Plot extended cruise vs Murbach
fig2, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(t_e, h_e, 'r-', lw=2, label='Eq-glide L/D=2.0 (extended)')
ax.axhline(20, ls=':', color='gray', lw=0.8, label='Murbach 20 km cruise')
ax.axvline(4000, ls='--', color='green', lw=0.8, alpha=0.7, label='Murbach cruise end (~4000 s)')
ax.set_xlim(0, max(t_e[-1], 5000))
ax.set_ylim(0, 130)
ax.set_xlabel('Flight time, s')
ax.set_ylabel('Altitude, km')
ax.set_title('Acton eq-glide — full trajectory to impact\n'
             'AEOLUS entry: 7 km/s / 125 km / γ=−15°,  m=178 kg, β=4000 kg/m²')
ax.grid(True, ls=':', alpha=0.5)
ax.legend(fontsize=9)
plt.tight_layout()
out2 = '/home/user/GUI-Flyout-Implementation/mars_smoke_test3.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Plot saved → {out2}")
