"""Objective check of the CBO 'majority of glide between 30-40 km' claim.

AUR booster + C-HGB glide body on a depressed trajectory; damped_glide at
several zeta.  Step 1 finds the burnout (kick) angle that gives a CBO-like
~120 km apogee; step 2 reports, at that angle, the apogee, the first pull-up
dip, and the fraction of the in-atmosphere glide in the 30-40 km (CBO) and
30-55 km (~100-180 kft, Mike White) bands, by time and by downrange.
"""
import copy, json
import numpy as np
from missile_models import get_missile, ro_from_dict
from trajectory import integrate_trajectory

CHGB = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))


def fly(zeta, burnout_deg, mode="damped_glide", aero="constant_LD"):
    p = get_missile("AUR+HGB")
    p.burnout_angle_deg = burnout_deg
    p.stage_burnout_angle_deg = burnout_deg
    if p.stage2:
        p.stage2.stage_burnout_angle_deg = burnout_deg
    rv = copy.deepcopy(CHGB)
    rv.glider_guidance = mode
    rv.glider_aero_model = aero
    if zeta is not None:
        rv.glider_damping_zeta = zeta
    p.rv = rv
    r = integrate_trajectory(p, 0.0, 0.0, 90.0, burnout_angle_deg=burnout_deg,
                             dt_output=1.0, max_time_s=6000.0)
    alt = np.asarray(r['alt']).ravel() / 1000.0
    rng = np.asarray(r['range']).ravel() / 1000.0
    t   = np.asarray(r['t']).ravel()
    return alt, rng, t


def glide_window(alt, rng, t):
    """Return (alt,rng,t) restricted to the in-atmosphere glide:
    from the first post-apogee descent below 55 km, to impact."""
    iap = int(np.argmax(alt))
    a, r_, tt = alt[iap:], rng[iap:], t[iap:]
    below = np.where(a < 55.0)[0]
    if len(below) == 0:
        return None
    g = below[0]
    return a[g:], r_[g:], tt[g:]


def first_dip(ga):
    """Altitude of the first local minimum (the pull-up bottom)."""
    for i in range(1, len(ga) - 1):
        if ga[i] <= ga[i-1] and ga[i] < ga[i+1]:
            return float(ga[i])
    return float(ga.min())


def metrics(zeta, burnout_deg):
    alt, rng, t = fly(zeta, burnout_deg)
    apogee = float(alt.max())
    w = glide_window(alt, rng, t)
    if w is None:
        return dict(zeta=zeta, apogee=apogee, note="no glide corridor")
    ga, gr, gt = w
    dt = np.diff(gt, prepend=gt[0]); dt[0] = 0.0
    dr = np.diff(gr, prepend=gr[0]); dr[0] = 0.0
    def frac(arr, w_, lo, hi):
        m = (arr >= lo) & (arr <= hi)
        return float(w_[m].sum() / w_.sum()) if w_.sum() > 0 else 0.0
    return dict(
        zeta=zeta, apogee=apogee, dip=first_dip(ga),
        med=float(np.median(ga)),
        t3040=frac(ga, dt, 30, 40), t3055=frac(ga, dt, 30, 55),
        r3040=frac(ga, dr, 30, 40), r3055=frac(ga, dr, 30, 55),
        rng=float(rng[-1]), tof=float(t[-1]),
    )


# --- Step 1: find the kick angle for a CBO-like ~120 km apogee (use zeta=0.5)
print("Step 1 — apogee vs depressed kick angle (AUR + C-HGB, zeta=0.5):")
best, best_err = None, 1e9
for ang in (15, 16, 17, 17.5, 18, 18.5, 19, 20):
    a, r_, t = fly(0.5, ang)
    ap = float(a.max())
    print(f"   kick {ang:>2} deg  ->  apogee {ap:6.0f} km   range {r_[-1]:5.0f} km")
    if abs(ap - 120) < best_err:
        best, best_err = ang, abs(ap - 120)
print(f"   -> using kick = {best} deg (apogee closest to 120 km)\n")

# --- Step 2: band fractions at that angle, across zeta
print(f"Step 2 — glide-band fractions at kick = {best} deg "
      f"(CBO: 'majority' 30-40 km):")
hdr = ("zeta", "apogee", "1st dip", "glide med", "%t 30-40", "%t 30-55",
       "%dr 30-40", "range", "TOF")
print("{:>5} {:>7} {:>8} {:>9} {:>8} {:>8} {:>9} {:>7} {:>6}".format(*hdr))
for z in (0.3, 0.4, 0.5, 0.7):
    d = metrics(z, best)
    if "note" in d:
        print(f"{z:>5.1f} {d['apogee']:>6.0f}k  {d['note']}"); continue
    print("{zeta:>5.1f} {apogee:>6.0f}k {dip:>7.1f}k {med:>8.1f}k "
          "{t3040:>7.0%} {t3055:>7.0%} {r3040:>8.0%} {rng:>6.0f}k {tof:>5.0f}s".format(**d))
