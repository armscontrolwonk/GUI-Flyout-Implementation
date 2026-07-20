"""Form C verification case — Regan 1984, Tables 6.7/6.8 (pp. 133-134).

Regan's published worked maneuvering trajectory: fixed L/D = 1.5,
β = 10,000 (his weight form, Pa — ÷ g for the repo's mass form, see
HEATING_MODEL_CROSSCHECK.md), V0 = 6,000 m/s at z0 = 100 km, γ0 = 10°
below horizontal, transverse-acceleration limit 4 g.  Book result: the
fixed-L/D maneuver HITS THE 4-g TRANSVERSE LIMIT AT ≈45 km, below which
L/D must be feathered.

This harness integrates the planar point-mass EOM with Thrusty's own
atmosphere() and the same force law as trajectory.py's constant_LD glide
(a_drag = q̄/β_m; a_lift = min(L/D · a_drag, n_max·g)) and reports where
the 4-g cap first binds.  It checks the atmosphere + force-model +
β-convention chain against a published table, not the 3-D integrator
itself (which embeds the same law).  Run: python3 regan1984_marv_check.py
"""
import numpy as np
from atmosphere import atmosphere

MU = 3.986004418e14        # m³/s²  (same point-mass gravity as trajectory.py)
RE = 6371.0e3              # spherical Earth for the planar case
G0 = 9.80665

LD = 1.5
BETA_REGAN_PA = 1.0e4      # Regan's β (weight form, Pa)
N_MAX = 4.0                # transverse-acceleration limit (g)
V0, Z0, GAMMA0 = 6000.0, 100.0e3, np.radians(10.0)


def fly(beta_kg_m2, dt=0.05):
    """Planar entry; returns (z_at_first_cap_bind_km, trace)."""
    r = RE + Z0
    v = V0
    gamma = -GAMMA0                       # below horizontal
    theta = 0.0
    z_bind = None
    trace = []
    for _ in range(int(4000.0 / dt)):
        z = r - RE
        if z <= 0.0:
            break
        rho = atmosphere(z)[2]
        q_bar = 0.5 * rho * v * v
        g = MU / (r * r)
        a_drag = q_bar / beta_kg_m2
        a_lift_nom = LD * a_drag          # fixed-L/D lift demand
        a_lift = min(a_lift_nom, N_MAX * G0)
        if z_bind is None and a_lift_nom >= N_MAX * G0:
            z_bind = z
        # planar point-mass EOM (spherical, non-rotating)
        v_dot = -a_drag - g * np.sin(gamma)
        gamma_dot = (a_lift / v) + (v / r - g / v) * np.cos(gamma)
        r_dot = v * np.sin(gamma)
        theta_dot = v * np.cos(gamma) / r
        v += v_dot * dt
        gamma += gamma_dot * dt
        r += r_dot * dt
        theta += theta_dot * dt
        trace.append((z, v, np.degrees(gamma), a_lift_nom / G0))
        if v < 100.0:
            break
    return z_bind, trace


if __name__ == "__main__":
    print("Regan 1984 Tables 6.7/6.8 — fixed L/D=1.5, 4-g transverse limit;")
    print("book result: the limit binds at ≈45 km.\n")
    for label, beta in (
            ("Regan β=10⁴ Pa ÷ g  → 1019.7 kg/m² (CORRECT convention)",
             BETA_REGAN_PA / G0),
            ("Regan β=10⁴ taken as kg/m²      (WRONG convention)",
             BETA_REGAN_PA)):
        z_bind, trace = fly(beta)
        if z_bind is None:
            print(f"  {label}:  4-g cap never binds")
            continue
        # state at bind
        zb, vb, gb, nb = min(trace, key=lambda s: abs(s[0] - z_bind))
        print(f"  {label}:")
        print(f"     4-g cap first binds at z = {z_bind/1000.0:6.1f} km "
              f"(V = {vb:5.0f} m/s, γ = {gb:5.1f}°)")
    print("\nPASS criterion: correct-convention bind altitude within ±5 km "
          "of Regan's ≈45 km.")
