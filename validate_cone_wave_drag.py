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
    """Exact cone wave-drag coefficient (= surface Cp, ref. base area) at M1.

    The cone half-angle increases monotonically with shock angle up to
    detachment, so we scan from just above the Mach angle to find the bracket
    [lo, hi] straddling the target half-angle, then root-find within it.  This
    is robust for slender cones (narrow root near the Mach angle) and blunt
    cones at low Mach (shock angle well above 60 deg)."""
    target = np.radians(sigma_deg)
    # f(beta) = cone half-angle - target.  At the Mach angle the shock is
    # infinitely weak (half-angle -> 0), so treat the weak-shock limit (no
    # attached solution returned) as half-angle 0; f is then negative there and
    # rises monotonically with beta until the target is reached.
    f = lambda b: (_cone_from_shock(M1, b) or (0.0, 0.0))[0] - target
    mach_angle = np.arcsin(1.0 / M1)
    hi = None
    for b in np.radians(np.linspace(np.degrees(mach_angle) + 0.05, 89.0, 1000)):
        if f(b) > 0:
            hi = b
            break
    if hi is None:
        raise RuntimeError(f"no attached cone solution: M={M1}, sigma={sigma_deg} deg")
    beta = brentq(f, mach_angle + 1e-9, hi, xtol=1e-10)
    return _cone_from_shock(M1, beta)[1]


# Sims, NASA SP-3004, Table 9 — surface pressure coefficient Cp (= cone wave-drag
# CD ref. base area).  A representative span transcribed from the report, used as
# an authoritative third-party anchor for the exact solver above.
#   theta_c [deg] -> { M_inf : Cp }
_SIMS_TABLE9 = {
    2.5:  {1.5: .0123382, 2.0: .0107862, 3.0: .00913541, 4.0: .00814929,
           5.0: .00746199, 10.0: .00573592},
    5.0:  {1.5: .0396615, 2.0: .0339457, 3.0: .02824024, 5.0: .02304657,
           10.0: .01871810},
    10.0: {1.5: .1237983, 2.0: .1044583, 3.0: .08747518, 5.0: .07475691,
           10.0: .06670277},
    15.0: {2.0: .2022339, 3.0: .17310135, 5.0: .15423167},
    20.0: {2.0: .3255312, 2.5: .29915906},
}


def _sims_check():
    """Compare the exact solver against Sims SP-3004 Table 9 anchors."""
    errs = []
    for thc in sorted(_SIMS_TABLE9):
        for M in sorted(_SIMS_TABLE9[thc]):
            cp_s = _SIMS_TABLE9[thc][M]
            cp_m = cone_wave_cd_exact(M, thc)
            errs.append(abs(100 * (cp_m - cp_s) / cp_s))
    print(f"Sims SP-3004 Table 9 check ({len(errs)} anchors, theta_c 2.5-20 deg, "
          f"M 1.5-10): mean |%diff|={np.mean(errs):.3f}%, max={np.max(errs):.3f}%  "
          f"({'OK' if np.max(errs) < 0.5 else 'CHECK'})")


def _naca1135_check():
    """Verify the solver's oblique-shock relations against the canonical
    NACA Report 1135 forms (Eq 138 theta-beta-M deflection, Eq 132 downstream
    Mach).  These are the building blocks feeding the Taylor-Maccoll ODE; the
    NACA-1135 equations are also implemented in PDAS VUCALC (naca1135.pas).
    NACA 1135 itself has no conical-flow tables, so it anchors the inputs, not
    the cone result."""
    g = GAMMA
    def eq138_delta(M1, th):
        cot = np.tan(th) * ((g + 1) * M1**2 /
                            (2 * (M1**2 * np.sin(th)**2 - 1)) - 1)
        return np.arctan(1.0 / cot)
    def eq132_M2(M1, th):
        s = np.sin(th)**2
        num = (g + 1)**2 * M1**4 * s - 4 * (M1**2 * s - 1) * (g * M1**2 * s + 1)
        den = (2 * g * M1**2 * s - (g - 1)) * ((g - 1) * M1**2 * s + 2)
        return np.sqrt(num / den)
    worst_d = worst_m = 0.0
    for M1 in (2.0, 3.0, 5.0):
        for beta in np.radians((25.0, 35.0, 45.0)):
            delta = _deflection(M1, beta)
            Vr, Vt = _post_shock_vr_vt(M1, beta, delta)
            Vp = np.hypot(Vr, Vt)
            M2 = np.sqrt(Vp**2 / ((g - 1) / 2 * (1 - Vp**2)))
            worst_d = max(worst_d, abs(delta - eq138_delta(M1, beta)))
            worst_m = max(worst_m, abs(M2 - eq132_M2(M1, beta)))
    print(f"NACA-1135 building-block check: max |delta-Eq138|={worst_d:.1e} rad, "
          f"max |M2-Eq132|={worst_m:.1e}  "
          f"({'OK' if worst_d < 1e-9 and worst_m < 1e-9 else 'DIFFER'})")


def main():
    _naca1135_check()
    _sims_check()
    print("\nCone wave drag CD (ref base area): Taylor-Maccoll exact vs _cd_wave_cone")
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
