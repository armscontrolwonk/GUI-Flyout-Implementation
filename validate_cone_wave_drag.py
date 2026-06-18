"""
Validate _cd_wave_cone against the EXACT Taylor-Maccoll conical-flow solution.

The supersonic nose wave drag is the dominant body-drag term, and it is the one
term we could not cleanly cross-check against Missile DATCOM (its chart is read
through a 2-D INTERX interpolator) or PDAS WAVEDRAG (far-field area rule for
whole aircraft).  Taylor-Maccoll gives the exact inviscid flow over a sharp cone
at zero angle of attack; for such a cone the surface pressure coefficient is
constant and the wave-drag coefficient referenced to base area equals that Cp.

The oblique-shock and isentropic relations used here are the standard NACA-1135
forms (the same set implemented in PDAS VUCALC).

Run:  python validate_cone_wave_drag.py
Needs scipy (already a project dependency via trajectory.py).

Result summary (gamma=1.4): our table matches the exact solution to within a
few thousandths of CD at M1.5-3 for slender-to-moderate cones (the boost/glide
regime), with a modest +10-16% high bias at M4-5 and for blunt (low-L/D) cones,
inherited from the slender-body (3/ld)^2 fineness scaling.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from missile_models import _cd_wave_cone

GAMMA = 1.4


def _deflection(M1, beta):
    """Flow deflection (rad) behind an oblique shock — theta-beta-M relation."""
    Mn1 = M1 * np.sin(beta)
    return np.arctan(2 / np.tan(beta) * (Mn1**2 - 1) /
                     (M1**2 * (GAMMA + np.cos(2 * beta)) + 2))


def _p0_ratio_shock(Mn1):
    """Stagnation-pressure ratio p0_2/p0_1 across a shock of normal Mach Mn1."""
    g = GAMMA
    return (((g + 1) * Mn1**2 / ((g - 1) * Mn1**2 + 2))**(g / (g - 1)) *
            ((g + 1) / (2 * g * Mn1**2 - (g - 1)))**(1 / (g - 1)))


def _post_shock_vr_vt(M1, beta, delta):
    """Radial/angular velocity (normalised by V_max) just behind the shock."""
    g = GAMMA
    Mn1 = M1 * np.sin(beta)
    M2n = np.sqrt(((g - 1) * Mn1**2 + 2) / (2 * g * Mn1**2 - (g - 1)))
    M2 = M2n / np.sin(beta - delta)
    Vp = 1.0 / np.sqrt(1 + 2 / ((g - 1) * M2**2))      # V / V_max
    return Vp * np.cos(beta - delta), -Vp * np.sin(beta - delta)


def _tm_rhs(theta, y):
    """Taylor-Maccoll ODE; y = [V_r, V_theta], velocities / V_max."""
    Vr, Vt = y
    a2 = (GAMMA - 1) / 2 * (1 - Vr**2 - Vt**2)          # (a / V_max)^2
    return [Vt, (Vr * Vt**2 - a2 * (2 * Vr + Vt / np.tan(theta))) / (a2 - Vt**2)]


def _cone_from_shock(M1, beta):
    """Integrate from the shock to the cone surface (V_theta = 0).

    Returns (cone half-angle sigma [rad], surface Cp) or None if no solution.
    """
    delta = _deflection(M1, beta)
    if delta <= 0:
        return None
    y0 = _post_shock_vr_vt(M1, beta, delta)
    ev = lambda t, y: y[1]
    ev.terminal = True
    ev.direction = 1                                   # V_theta crosses 0 upward
    sol = solve_ivp(_tm_rhs, [beta, np.radians(0.5)], y0, events=ev,
                    rtol=1e-9, atol=1e-12, max_step=np.radians(0.25))
    if not sol.t_events[0].size:
        return None
    sigma = sol.t_events[0][0]
    Vr = sol.y_events[0][0][0]
    Mc = np.sqrt(Vr**2 / ((GAMMA - 1) / 2 * (1 - Vr**2)))
    p0_2_pinf = (_p0_ratio_shock(M1 * np.sin(beta)) *
                 (1 + (GAMMA - 1) / 2 * M1**2)**(GAMMA / (GAMMA - 1)))
    pc_pinf = p0_2_pinf * (1 + (GAMMA - 1) / 2 * Mc**2)**(-GAMMA / (GAMMA - 1))
    Cp = (pc_pinf - 1) / (0.5 * GAMMA * M1**2)
    return sigma, Cp


def cone_wave_cd_exact(M1, sigma_deg):
    """Exact cone wave-drag coefficient (ref. base area) at Mach M1."""
    target = np.radians(sigma_deg)
    f = lambda b: (_cone_from_shock(M1, b) or (1e9, 0.0))[0] - target
    beta = brentq(f, np.arcsin(1 / M1) + 1e-4, np.radians(60), xtol=1e-8)
    return _cone_from_shock(M1, beta)[1]


def main():
    print("Cone wave drag CD (ref base area): Taylor-Maccoll exact vs _cd_wave_cone")
    worst = 0.0
    for ld in (2.0, 3.0, 5.0):
        sigma = np.degrees(np.arctan(1 / (2 * ld)))
        print(f"\n  L/D = {ld}  (half-angle {sigma:.2f} deg)")
        print(f"    {'Mach':>5} {'exact':>8} {'ours':>8} {'diff':>8} {'%':>7}")
        for M in (1.5, 2.0, 3.0, 4.0, 5.0):
            ex = cone_wave_cd_exact(M, sigma)
            ours = _cd_wave_cone(ld, M)
            worst = max(worst, abs(ours - ex))
            print(f"    {M:5.1f} {ex:8.4f} {ours:8.4f} "
                  f"{ours - ex:+8.4f} {100 * (ours - ex) / ex:+6.1f}")
    print(f"\n  worst absolute CD difference: {worst:.4f}")


if __name__ == "__main__":
    main()
