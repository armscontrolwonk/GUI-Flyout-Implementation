"""Lift-authority / damping ceiling for the C-HGB on the AUR (depressed),
swept over L/D.

Damping authority is set by the lift HEADROOM at the glide,
    headroom = a_L,max - g_eff,   a_L,max = min(q*(L/D)/beta, n_max*g),
which near the lift-margin (Lambda = a_L,max/g_eff ~ 1) is very sensitive to
L/D.  zeta_max ~ headroom / (2*omega_p*V*|gamma|),  omega_p = sqrt(g_eff/H_rho),
gamma ~ equilibrium glide angle 2*(L/D)*H_rho*g/V^2.
"""
import copy, json
import numpy as np
from missile_models import get_missile, rv_from_dict
from trajectory import integrate_trajectory
from atmosphere import atmosphere

CHGB = rv_from_dict(json.load(open("rv_library/C-HGB.rv.json")))
KICK = 17.0
G0, RE, HR = 9.80665, 6.378137e6, 7000.0
BETA = CHGB.beta_kg_m2
NMAX = getattr(CHGB, "glider_pullup_g_max", 10.0)


def fly(ld, zeta=0.5):
    p = get_missile("AUR+HGB")
    p.burnout_angle_deg = KICK; p.stage_burnout_angle_deg = KICK
    if p.stage2: p.stage2.stage_burnout_angle_deg = KICK
    rv = copy.deepcopy(CHGB)
    rv.glider_LD = ld
    rv.glider_guidance = "damped_glide"; rv.glider_aero_model = "constant_LD"
    rv.glider_damping_zeta = zeta
    p.rv = rv
    r = integrate_trajectory(p, 0.0, 0.0, 90.0, burnout_angle_deg=KICK,
                             dt_output=1.0, max_time_s=6000.0)
    return (np.asarray(r['alt']).ravel(), np.asarray(r['speed']).ravel(),
            np.asarray(r['range']).ravel())


def authority(ld):
    alt, spd, rng = fly(ld)
    iap = int(np.argmax(alt))
    a, v = alt[iap:], spd[iap:]
    band = (a/1000.0 >= 28) & (a/1000.0 <= 42) & (v > 100)   # the glide corridor
    if not band.any():
        return None
    lam_list, zmax_list = [], []
    for h, V in zip(a[band], v[band]):
        rho = atmosphere(h)[2]
        q = 0.5 * rho * V * V
        a_L = min(q * ld / BETA, NMAX * G0)
        g_eff = G0 - V * V / (RE + h)
        if g_eff <= 0:
            continue
        lam_list.append(a_L / g_eff)
        head = a_L - g_eff
        if head > 0:
            omega = np.sqrt(g_eff / HR)
            gamma = 2 * ld * HR * G0 / (V * V)        # equilibrium glide angle
            zmax_list.append(head / (2 * omega * V * gamma))
        else:
            zmax_list.append(0.0)
    glide_med = float(np.median(a[a/1000.0 < 55]) / 1000.0)
    return dict(ld=ld, glide=glide_med, rng=float(rng[-1]/1000.0),
                lam=float(np.median(lam_list)), zmax=float(np.median(zmax_list)))


print(f"C-HGB (beta={BETA:.0f}, n_max={NMAX}g) on AUR depressed; "
      f"effect of L/D on glide & damping authority:\n")
print(f"{'L/D':>5} {'glide km':>9} {'range km':>9} {'Lambda(glide)':>14} {'zeta_max':>9}")
for ld in (2.0, 2.2, 2.4):
    d = authority(ld)
    if d is None:
        print(f"{ld:>5.1f}  no glide corridor"); continue
    print(f"{ld:>5.1f} {d['glide']:>8.1f} {d['rng']:>9.0f} {d['lam']:>13.2f} {d['zmax']:>9.2f}")
