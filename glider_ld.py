"""
Whole-missile (no-separation body) lift-to-drag estimate at angle of attack.

For a vehicle whose RV does NOT separate, the gliding/maneuvering body is the
ENTIRE airframe (nose + body + fins) flying at a trim angle of attack.  Its L/D
is an emergent geometric property, not a designed value, so it is DERIVED from
geometry here -- in contrast to a separating RV, whose L/D is the RV's own
designed property and is supplied directly as rv.glider_LD.

Method: the standard semi-empirical component build-up for slender body+fin
configurations at angle of attack (the same theoretical core Missile DATCOM
uses), assembled from primary sources, all in data/:
  * Allen & Perkins, NACA Rep. 1048 (1951): the two-term body normal force
    (slender-body potential + viscous crossflow); origin of the crossflow term.
  * Jorgensen, NASA TR R-474 (1977): generalises it to all AoA (Eq 2.12), the
    body+wing assembly (Eq 5.3) with the sin(2a)/(2a) high-AoA correction, the
    axial force (Eq 2.18), and C_L/C_D (Eqs 2.16/2.17).
  * Pitts, Nielsen & Kaattari, NACA Rep. 1307 (1957): the wing-body
    interference factors K_W(B), K_B(W) (slender-body Eqs 14/21), whose sum is
    the exact identity  K_W(B) + K_B(W) = (1 + r/s)^2.

Assembly (referenced to body base area A_r = pi*(d/2)^2):
    C_Na_pot = 2*(A_b/A_r)                         # body slender-body potential
             + (1 + r/s)^2 * (C_La)_W * (S_W/A_r)  # wing + body carryover (N-K-P)
    C_N(a)   = C_Na_pot * sin(2a)/2                # high-AoA correction (Jorgensen 5.3)
             + eta * C_dn * (A_p/A_r) * sin^2(a)   # viscous crossflow (Allen-Perkins)
    C_A(a)   = C_A0 * cos^2(a)                      # Jorgensen 2.18
    C_L = C_N*cos(a) - C_A*sin(a) ;  C_D = C_N*sin(a) + C_A*cos(a)
    L/D maximised over a.

Constants: eta = 1.0 (supersonic/hypersonic, Jorgensen p.26), C_dn = 1.2
(modified-Newtonian crossflow drag of a cylinder, Jorgensen p.47).  A_p is the
total (body + lifting-fin) planform.  Validity follows the sources: best at
supersonic Mach, extended into hypersonic by the modified-Newtonian crossflow
and to high AoA by the sin(2a)/(2a) term; linear-theory wing slope assumes
unbanked fins without swept-forward LE / swept-back TE (NACA 1307).

CAVEATS: a slender whole missile is a POOR lifting shape, so the resulting L/D
is modest; this is a preliminary-design estimate, not a substitute for
wind-tunnel/CFD.  The body planform is the cone/slender ~0.5*L*d approximation.
"""

from __future__ import annotations
import math
from missile_models import MissileParams, _cd_nose_shape, drag_coefficient, _SHAPE_ALIAS

_ETA = 1.0       # crossflow drag proportionality factor.  Jorgensen (NASA TN
                 # D-7228, 1973) states eta = 1 for supersonic/hypersonic
                 # free-stream Mach; the eta(L/d) chart (Gowen-Perkins TN 2960
                 # Fig. 8) applies only at subsonic free-stream.

# Crossflow drag coefficient C_dn of a circular cylinder (section normal to the
# stream) vs the crossflow Mach number M_n = M*sin(alpha).  This is the
# "state-of-knowledge" curve Jorgensen's method (TN D-7228 Eq. 1) feeds into the
# viscous-crossflow term.  Values read from Gowen & Perkins, NACA TN 2960 (1953)
# Fig. 7 (subcritical-Reynolds branch): ~1.2 at low M, a transonic rise to a
# sharp peak ~2.1 at M_n=1, decaying through the supersonic range to ~1.34 at
# M_n=2.9.  Held flat outside the measured range (Re/drag-crisis effects at low
# subsonic M_n with supercritical crossflow Reynolds are not modelled — the
# subcritical curve is what Jorgensen's method uses).
_CDN_VS_MCROSS = [
    (0.0, 1.20), (0.4, 1.20), (0.5, 1.25), (0.6, 1.32), (0.7, 1.42),
    (0.8, 1.55), (0.9, 1.75), (1.0, 2.10), (1.1, 2.00), (1.2, 1.80),
    (1.4, 1.65), (1.6, 1.55), (1.8, 1.52), (2.0, 1.49), (2.4, 1.42),
    (2.9, 1.34), (4.0, 1.30),
]


def crossflow_cd(m_cross: float) -> float:
    """Circular-cylinder crossflow drag coefficient C_dn at crossflow Mach
    M_n = M*sin(alpha), piecewise-linear in the Gowen-Perkins (NACA TN 2960
    Fig. 7) data, clamped flat beyond the measured range."""
    tbl = _CDN_VS_MCROSS
    if m_cross <= tbl[0][0]:
        return tbl[0][1]
    if m_cross >= tbl[-1][0]:
        return tbl[-1][1]
    for (m0, c0), (m1, c1) in zip(tbl, tbl[1:]):
        if m_cross <= m1:
            return c0 + (c1 - c0) * (m_cross - m0) / (m1 - m0)
    return tbl[-1][1]


# Nose planform "fill" factor: side-projected nose area / (L_nose * d).  A cone
# projects to a triangle (0.5); rounded noses are fuller (a tangent ogive is
# ~0.67 by exact integration).  Used to build the Allen-Perkins crossflow
# planform area for a nose+cylinder body.
_NOSE_PLANFORM_FILL = {
    'cone': 0.5, 'tangent_ogive': 0.667, 'parabola': 0.667,
    'von_karman': 0.667, 'lv_haack': 0.667, 'blunt_cylinder': 0.85,
}

# Representative glide Mach at which a no-sep body's L/D is derived for the
# trajectory (the build-up's L/D-max is only weakly Mach-sensitive across the
# supersonic-hypersonic glide range, so a single reference is adequate).
GLIDE_MACH_REF = 5.0


def nkp_interference(r: float, s: float):
    """NACA 1307 slender-body wing-body interference factors (K_W(B), K_B(W)).

    r = body radius, s = wing semispan from body AXIS to tip (= r + exposed
    semispan).  Eqs 14 and 21; their sum is the exact identity (1 + r/s)^2.
    Returns (K_WB, K_BW)."""
    if s <= 0 or r < 0 or r >= s:
        return 1.0, 0.0
    lam = min(r / s, 0.999)                        # r/s in (0,1); clamp off the
    inv = 1.0 / lam                                # (1-lam)^2 singularity at lam=1
    brace = ((1.0 + lam**4) * (0.5 * math.atan(0.5 * (inv - lam)) + math.pi / 4.0)
             - lam**2 * ((inv - lam) + 2.0 * math.atan(lam)))
    denom = (1.0 - lam) ** 2
    k_wb = (2.0 / math.pi) * brace / denom
    k_bw = ((1.0 - lam**2) ** 2 - (2.0 / math.pi) * brace) / denom
    return k_wb, k_bw


def wing_alone_cla(exposed_semispan: float, c_root: float, c_tip: float,
                   mach: float, sweep_deg: float = 0.0) -> float:
    """Wing-alone lift-curve slope /rad (N-K-P 'wing alone' = the two exposed
    panels joined at the centreline), referenced to the wing-alone area
    S_W = exposed_semispan*(c_root+c_tip).  Low-aspect-ratio linear-theory form
    (Barrowman/Diederich single-surface), referenced to S_W."""
    s_e = exposed_semispan
    if s_e <= 0 or c_root <= 0:
        return 0.0
    c_tip = max(c_tip, 0.0)
    S_W = s_e * (c_root + c_tip)                  # two joined panels
    ar = (2.0 * s_e) ** 2 / S_W                   # joined-wing aspect ratio
    beta = math.sqrt(abs(mach * mach - 1.0))
    tan_gc = math.tan(math.radians(sweep_deg)) + (c_tip - c_root) / (2.0 * s_e)
    cos_gc = 1.0 / math.sqrt(1.0 + tan_gc * tan_gc)
    return 2.0 * math.pi * ar / (2.0 + math.sqrt(4.0 + (beta * ar / cos_gc) ** 2))


def _last_stage(params: MissileParams) -> MissileParams:
    last = params
    while last.stage2 is not None:
        last = last.stage2
    return last


def _body_cd0(last: MissileParams, mach: float) -> float:
    """Body zero-lift drag coefficient (referenced to base area)."""
    nose = _SHAPE_ALIAS.get(last.nose_shape or '', last.nose_shape or '')
    d = last.diameter_m
    if nose and nose not in ('', 'forden') and d > 0:
        ld_nose = last.nose_length_m / d if last.nose_length_m > 0 else 3.0
        ld_body = last.length_m / d if last.length_m > 0 else None
        return _cd_nose_shape(nose, ld_nose, mach, ld_body=ld_body,
                              aerospike_LD=float(last.aerospike_LD or 0.0),
                              aerospike_dD=float(last.aerospike_dD or 0.0))
    return drag_coefficient(last, mach)


def whole_missile_LD(params: MissileParams, mach: float = 3.0,
                     return_curve: bool = False) -> dict:
    """Maximum L/D (and the angle of attack) of the whole no-sep airframe at the
    given Mach, by the Jorgensen + Allen-Perkins + N-K-P build-up above.

    Returns a dict: ld_max, alpha_deg, c_na_pot, k_sum (=(1+r/s)^2), cla_wing,
    cd0, plus the body/fin geometry used.  Set return_curve=True to also get the
    (alpha, L/D) sweep."""
    last = _last_stage(params)
    d = float(last.diameter_m)
    if d <= 0:
        return dict(ld_max=0.0, alpha_deg=0.0, error="body diameter not set")
    A_ref = math.pi * (d / 2.0) ** 2
    L_body = float(last.length_m) if last.length_m > 0 else 5.0 * d
    A_b = A_ref                                   # base area = reference
    # Planform (side-projected) area for the Allen-Perkins viscous-crossflow
    # term — NACA 1048 uses the body's true planform S_plan.  The pointed-body
    # triangle 0.5*L*d is only correct for a cone; for a body with a long
    # cylindrical afterbody it underestimates S_plan badly (validated against
    # Digital DATCOM: the triangle drove L/D ~20-30% low, growing with Mach).
    # Build it as nose (shape fill factor x L_nose x d) + cylindrical afterbody.
    L_nose = float(getattr(last, 'nose_length_m', 0.0) or 0.0)
    if 0.0 < L_nose < L_body:
        nose = _SHAPE_ALIAS.get(last.nose_shape or '', last.nose_shape or '')
        fill = _NOSE_PLANFORM_FILL.get(nose, 0.667)   # cone 0.5, ogive ~0.67
        A_p_body = fill * L_nose * d + (L_body - L_nose) * d
    else:
        A_p_body = 0.5 * L_body * d                # pointed-body fallback

    cd0 = _body_cd0(last, mach)

    # Fins (the pitch-plane lifting pair); body+fin carryover via N-K-P.
    k_sum = 0.0
    cla_w = 0.0
    S_W = 0.0
    A_p_fin = 0.0
    if getattr(last, 'has_fins', False) and last.fin_span_m > 0 \
            and last.fin_root_chord_m > 0:
        s_e = float(last.fin_span_m)              # exposed semispan
        cr = float(last.fin_root_chord_m)
        ct = float(last.fin_tip_chord_m)
        sw = float(last.fin_sweep_deg)
        r = d / 2.0
        s = r + s_e                               # semispan from body axis
        k_wb, k_bw = nkp_interference(r, s)
        k_sum = k_wb + k_bw                        # == (1 + r/s)^2
        cla_w = wing_alone_cla(s_e, cr, ct, mach, sw)
        S_W = s_e * (cr + ct)                      # joined exposed-panel area
        A_p_fin = S_W                              # fin planform for crossflow

    # Potential normal-force slope (per rad), referenced to A_ref, split into
    # the body (slender-body) part and the fin+interference (N-K-P) part.
    c_na_body = 2.0 * (A_b / A_ref)
    c_na_fin = k_sum * cla_w * (S_W / A_ref)
    c_na_pot = c_na_body + c_na_fin
    A_p = A_p_body + A_p_fin

    best_ld, best_a, curve = 0.0, 0.0, []
    for i in range(1, 60):                         # alpha = 1..59 deg
        a = math.radians(float(i))
        sn, cs = math.sin(a), math.cos(a)
        c_dn = crossflow_cd(mach * sn)             # C_dn at crossflow Mach M*sin(a)
        c_n = c_na_pot * math.sin(2.0 * a) / 2.0 + _ETA * c_dn * (A_p / A_ref) * sn * sn
        c_a = cd0 * cs * cs
        c_l = c_n * cs - c_a * sn
        c_d = c_n * sn + c_a * cs
        ld = c_l / c_d if c_d > 0 else 0.0
        if return_curve:
            curve.append((float(i), ld))
        if ld > best_ld:
            best_ld, best_a = ld, float(i)

    out = dict(ld_max=best_ld, alpha_deg=best_a, c_na_pot=c_na_pot,
               c_na_body=c_na_body, c_na_fin=c_na_fin,
               k_sum=k_sum, cla_wing=cla_w, cd0=cd0, mach=mach,
               diameter_m=d, body_planform_m2=A_p_body, fin_planform_m2=A_p_fin)
    if return_curve:
        out["curve"] = curve
    return out


def derive_glider_LD(params: MissileParams, mach: float = GLIDE_MACH_REF) -> float:
    """Geometry-derived max L/D of a no-separation airframe (the value to use as
    glider_LD for a body glider).  Thin wrapper over whole_missile_LD; returns
    0.0 if it cannot be computed."""
    try:
        return float(whole_missile_LD(params, mach=mach).get("ld_max", 0.0))
    except Exception:
        return 0.0


if __name__ == "__main__":
    from missile_models import get_missile, MissileParams as MP
    # 1) N-K-P identity check
    for rs in (0.0, 0.25, 0.5, 0.75, 1.0):
        r, s = rs, 1.0
        kwb, kbw = nkp_interference(r, s) if rs > 0 else (1.0, 0.0)
        print(f"r/s={rs:.2f}: K_W(B)={kwb:.3f} K_B(W)={kbw:.3f} sum={kwb+kbw:.3f} "
              f"(1+r/s)^2={(1+rs)**2:.3f}")
    print()
    # 2) a slender finless body, and a finned no-sep body, at a few Mach
    body = MP(name="slender body", diameter_m=0.5, length_m=4.0,
              nose_shape="tangent_ogive", nose_length_m=1.5,
              mass_initial=500, mass_propellant=0, mass_final=500, burn_time_s=1,
              isp_s=1, thrust_N=1)
    for M in (2.0, 3.0, 5.0):
        r = whole_missile_LD(body, mach=M)
        print(f"finless slender body  M{M}: L/D_max={r['ld_max']:.2f} at "
              f"{r['alpha_deg']:.0f}deg  (Cd0={r['cd0']:.3f})")
    finned = MP(name="no-sep finned body", diameter_m=0.5, length_m=4.0,
                nose_shape="tangent_ogive", nose_length_m=1.5, has_fins=True,
                n_fins=4, fin_span_m=0.3, fin_root_chord_m=0.4, fin_tip_chord_m=0.2,
                fin_sweep_deg=20.0, fin_thickness_m=0.02,
                mass_initial=500, mass_propellant=0, mass_final=500, burn_time_s=1,
                isp_s=1, thrust_N=1)
    for M in (2.0, 3.0, 5.0):
        r = whole_missile_LD(finned, mach=M)
        print(f"no-sep finned body    M{M}: L/D_max={r['ld_max']:.2f} at "
              f"{r['alpha_deg']:.0f}deg  (k_sum={r['k_sum']:.2f}, "
              f"cla_W={r['cla_wing']:.2f})")
