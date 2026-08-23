"""How much does an angle-of-attack C_L margin (winglets/elevons) raise the
C-HGB damping ceiling?

The constant-L/D model rigidly ties lift to drag (no AoA margin), so the only
damping headroom is 'dip into denser air' -> zeta_max ~ 0.2.  A control-surface
C_L margin M = C_L,max / C_L,trim lets the vehicle pull transient lift up to
M x its trim value:
    a_L,max(M) = M * a_L,trim,   a_L,trim = q*(L/D)/beta,
    headroom(M) = a_L,max(M) - g_eff,
    zeta_max(M) ~ headroom / (2*omega_p*V*gamma),  omega_p=sqrt(g_eff/H_rho),
    gamma ~ 2*(L/D)*H_rho*g/V^2.
Because the glider sits right at the lift margin (headroom is tiny at M=1),
zeta_max is highly leveraged in M.
"""
import copy, json
import numpy as np
from booster_models import get_booster, ro_from_dict
from trajectory import integrate_trajectory
from atmosphere import atmosphere

CHGB = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
# Reentry-object files are hardware-only; apply the shipped reentry plan.
from booster_models import apply_reentry_plan as _arp, load_reentry_plan as _lrp
_rp = _lrp("C-HGB")
if _rp is not None:
    CHGB = _arp(CHGB, _rp)
KICK = 17.0
G0, RE, HR = 9.80665, 6.378137e6, 7000.0
BETA, LD = CHGB.beta_kg_m2, CHGB.glider_LD


def glide_states():
    p = get_booster("AUR")
    p.burnout_angle_deg = KICK; p.stage_burnout_angle_deg = KICK
    if p.stage2: p.stage2.stage_burnout_angle_deg = KICK
    ro = copy.deepcopy(CHGB)
    ro.glider_guidance = "damped_glide"; ro.glider_aero_model = "constant_LD"
    ro.glider_damping_zeta = 0.5
    p.ro = ro
    r = integrate_trajectory(p, 0.0, 0.0, 90.0, burnout_angle_deg=KICK,
                             dt_output=1.0, max_time_s=6000.0)
    alt = np.asarray(r['alt']).ravel(); spd = np.asarray(r['speed']).ravel()
    iap = int(np.argmax(alt)); a, v = alt[iap:], spd[iap:]
    band = (a/1000 >= 28) & (a/1000 <= 42) & (v > 100)
    return a[band], v[band]


def zeta_max_for_margin(a, v, M):
    z = []
    for h, V in zip(a, v):
        rho = atmosphere(h)[2]
        q = 0.5 * rho * V * V
        a_L_trim = q * LD / BETA
        g_eff = G0 - V * V / (RE + h)
        if g_eff <= 0:
            continue
        head = M * a_L_trim - g_eff
        if head <= 0:
            z.append(0.0); continue
        omega = np.sqrt(g_eff / HR)
        gamma = 2 * LD * HR * G0 / (V * V)
        z.append(head / (2 * omega * V * gamma))
    return float(np.median(z))


a, v = glide_states()
print(f"C-HGB (beta={BETA:.0f}, L/D={LD}) -- damping ceiling vs C_L margin\n")
print("C_L margin M   meaning                         zeta_max")
labels = {1.0: "no margin (= constant-L/D, bare body)",
          1.1: "+10% C_L (tiny elevon deflection)",
          1.25: "+25% C_L (modest control surfaces)",
          1.5: "+50% C_L",
          2.0: "2x trim C_L (large AoA margin)"}
for M in (1.0, 1.1, 1.25, 1.5, 2.0):
    print(f"   {M:>4.2f}        {labels[M]:<34} {zeta_max_for_margin(a, v, M):>6.2f}")

# margin needed to reach target zeta
Ms = np.linspace(1.0, 2.0, 201)
zs = np.array([zeta_max_for_margin(a, v, M) for M in Ms])
for tgt in (0.4, 0.7):
    i = np.argmax(zs >= tgt)
    if zs[i] >= tgt:
        print(f"\n  -> zeta_max = {tgt}: needs C_L margin M ~ {Ms[i]:.2f} "
              f"(+{(Ms[i]-1)*100:.0f}% lift over trim)")
