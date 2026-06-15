"""Lift-authority / damping ceiling for the C-HGB on the AUR (depressed).

'How hard can this airframe damp?' is set by how much lift it can pull versus
what gravity demands:  Lambda = a_L,max / g_eff,
    a_L,max = min( q*(L/D)/beta , n_max*g ),   q = 0.5*rho*V^2,
    g_eff   = g - V^2/r.
Lambda < 1  -> lift-starved, cannot even hold the glide (free-falls);
Lambda >> 1 -> lots of headroom, can damp the phugoid hard (high zeta OK).
We evaluate Lambda along the real trajectory, and separately confirm the
commanded-zeta sweep does not saturate (no fixed ceiling at the glide).
"""
import copy, json
import numpy as np
from missile_models import get_missile, rv_from_dict
from trajectory import integrate_trajectory
from atmosphere import atmosphere

CHGB = rv_from_dict(json.load(open("rv_library/C-HGB.rv.json")))
KICK = 17.0
G0, RE = 9.80665, 6.378137e6
BETA = CHGB.beta_kg_m2
LD   = CHGB.glider_LD
NMAX = getattr(CHGB, "glider_pullup_g_max", 10.0)
print(f"C-HGB: beta={BETA:.0f} kg/m^2, L/D={LD}, n_max={NMAX} g\n")


def fly(zeta):
    p = get_missile("AUR+HGB")
    p.burnout_angle_deg = KICK; p.stage_burnout_angle_deg = KICK
    if p.stage2: p.stage2.stage_burnout_angle_deg = KICK
    rv = copy.deepcopy(CHGB)
    rv.glider_guidance = "damped_glide"; rv.glider_aero_model = "constant_LD"
    rv.glider_damping_zeta = zeta
    p.rv = rv
    r = integrate_trajectory(p, 0.0, 0.0, 90.0, burnout_angle_deg=KICK,
                             dt_output=1.0, max_time_s=6000.0)
    return (np.asarray(r['alt']).ravel(), np.asarray(r['speed']).ravel(),
            np.asarray(r['range']).ravel())


def lam(h_m, V):
    rho = atmosphere(h_m)[2]
    q = 0.5 * rho * V * V
    a_L = min(q * LD / BETA, NMAX * G0)
    g_eff = G0 - V * V / (RE + h_m)
    return a_L / g_eff if g_eff > 0 else np.inf, a_L, g_eff


alt, spd, rng = fly(0.5)                      # representative run
iap = int(np.argmax(alt))
ad, vd = alt[iap:], spd[iap:]
print("Lift authority Lambda = a_L,max / g_eff along the descent (zeta=0.5):")
print(f"{'alt km':>7} {'V km/s':>7} {'a_L m/s2':>9} {'g_eff':>7} {'Lambda':>7}")
for h_km in (80, 70, 60, 50, 45, 40, 35, 30):
    j = int(np.argmin(np.abs(ad/1000.0 - h_km)))
    if ad[j]/1000.0 > 1.0:
        L, aL, ge = lam(ad[j], vd[j])
        flag = "  lift-starved (<1)" if L < 1 else ("  ample" if L > 1.8 else "")
        print(f"{ad[j]/1000:>7.1f} {vd[j]/1000:>7.2f} {aL:>9.1f} {ge:>7.2f} {L:>7.2f}{flag}")

print("\nCommanded-zeta sweep (glide altitude & range) -- does it saturate?")
print(f"{'zeta':>5} {'glide med km':>13} {'range km':>9}")
for z in (0.4, 0.7, 1.0, 2.0, 5.0):
    a, s, rg = fly(z)
    ia = int(np.argmax(a)); g = a[ia:]; g = g[g/1000.0 < 55]
    print(f"{z:>5.1f} {np.median(g)/1000:>12.1f} {rg[-1]/1000:>9.0f}")
