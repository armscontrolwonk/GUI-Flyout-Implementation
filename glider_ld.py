"""
Whole-booster (no-separation body) lift-to-drag estimate at angle of attack.

For a vehicle whose RV does NOT separate, the gliding/maneuvering body is the
ENTIRE airframe (nose + body + fins) flying at a trim angle of attack.  Its L/D
is an emergent geometric property, not a designed value, so it is DERIVED from
geometry here -- in contrast to a separating RV, whose L/D is the RV's own
designed property and is supplied directly as ro.glider_LD.

Method: the standard semi-empirical component build-up for slender body+fin
configurations at angle of attack (the same theoretical core Booster DATCOM
uses), assembled from these primary sources.  NOTE none of the three is in the
repo: data/ holds no PDFs (papers live in Drive per data/REFERENCES.md, which
does not list them either), so the citations below are the record, not a file
you can open.  The one cross-check with committed primary material is Digital
DATCOM, in validation/datcom/:
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

CAVEATS: a slender whole booster is a POOR lifting shape, so the resulting L/D
is modest; this is a preliminary-design estimate, not a substitute for
wind-tunnel/CFD.  The body planform is built as nose + cylindrical afterbody
(falling back to the ~0.5*L*d pointed-body triangle when no nose length is
known), and its centroid is returned alongside the area for the trim gate's
moment balance.  The fin path (c_na_fin and the N-K-P carryover) is NOT
validated against any external reference here: the committed DATCOM case is
deliberately finless.
"""

from __future__ import annotations
import math
from booster_models import BoosterParams, _cd_nose_shape, drag_coefficient, _SHAPE_ALIAS

_ETA = 1.0       # crossflow drag proportionality factor.  Jorgensen (NASA TN
                 # D-7228, 1973) states eta = 1 for supersonic/hypersonic
                 # free-stream Mach; the eta(L/d) chart (Gowen-Perkins TN 2960
                 # Fig. 8) applies only at subsonic free-stream.  NOT a free
                 # calibration knob: at eta = 1 the whole-body L/D_max sits
                 # within 5/9/10% (M2/3/5), conservative, of Digital DATCOM for
                 # the finless slender reference body (validation/datcom/,
                 # METHODS §"Whole-missile L/D").  De-rating it to chase a lower
                 # number breaks that validation — the low free-flight L/D of a
                 # fin-stabilized body is the TRIMMED value at its (low) trim
                 # alpha, set by cg via trim_gate, not a smaller L/D ceiling.

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

# The angle-of-attack sweep the L/D curve and the trim solve BOTH run on.
# Exported so trim_gate.py brackets its moment balance on exactly this interval
# instead of hard-coding a copy that would silently desynchronise if the sweep
# ever changed.
ALPHA_SWEEP_MIN_DEG = 1
ALPHA_SWEEP_MAX_DEG = 59


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


def _last_stage(params: BoosterParams) -> BoosterParams:
    last = params
    while last.stage2 is not None:
        last = last.stage2
    return last


def _front_nose_aero(params: BoosterParams, last: BoosterParams):
    """(nose_shape, nose_length_m) of the AS-FLOWN front end for the aero
    build-up.  For a non-separating body the nose shape lives on the reentry
    object (the stage carries none), so it is read from there and its taper is
    body_nose_length_m (flagged default when unset), consistent with the
    schematic and grid_fin_sizing; otherwise the stage's own nose fields."""
    from booster_models import effective_ro
    d = float(last.diameter_m)
    ro = effective_ro(params)
    if (ro is not None
            and getattr(ro, 'separation_mode', 'separating_ro') == 'body'
            and getattr(ro, 'shape', '')):
        _bn = float(getattr(ro, 'body_nose_length_m', 0.0) or 0.0)
        if _bn <= 0.0 and d > 0:
            _bn = min(3.0 * d, 0.5 * (float(last.length_m) or (2.0 * d)))
        return (ro.shape, _bn)
    return (getattr(last, 'nose_shape', '') or '',
            float(getattr(last, 'nose_length_m', 0.0) or 0.0))


def _body_cd0(last: BoosterParams, mach: float,
              nose_shape: str = None, nose_len: float = None) -> float:
    """Body zero-lift drag coefficient (referenced to base area).

    nose_shape / nose_len override the stage's own nose fields with the
    as-flown front end (the RO nose for a non-separating body — see
    _front_nose_aero); when None the stage fields are used (legacy behaviour)."""
    _ns = last.nose_shape if nose_shape is None else nose_shape
    _nl = last.nose_length_m if nose_len is None else nose_len
    nose = _SHAPE_ALIAS.get(_ns or '', _ns or '')
    d = last.diameter_m
    if nose and nose not in ('', 'forden') and d > 0:
        ld_nose = _nl / d if _nl and _nl > 0 else 3.0
        ld_body = last.length_m / d if last.length_m > 0 else None
        return _cd_nose_shape(nose, ld_nose, mach, ld_body=ld_body,
                              aerospike_LD=float(last.aerospike_LD or 0.0),
                              aerospike_dD=float(last.aerospike_dD or 0.0))
    return drag_coefficient(last, mach)


def body_cd0(params: BoosterParams, mach: float) -> float:
    """As-flown body Cd0: resolves the correct front-end nose (the reentry
    object's for a non-separating body) and evaluates _body_cd0.

    A DECLARED BICONIC flies the two-cone build-up (cd0_biconic_body): fore
    cone + aft frustum + cylindrical afterbody, so the flown β / L-D see the
    real shape, not a single cone (FRONT_END_DESIGN.md).  Falls through to the
    single-nose path when the biconic geometry is absent or invalid."""
    from booster_models import biconic_nose_geometry, cd0_biconic_body
    last = _last_stage(params)
    _bic = biconic_nose_geometry(params)
    if _bic is not None:
        d = float(last.diameter_m)
        ld_body = (last.length_m / d) if (d > 0 and last.length_m > 0) else 5.0
        return cd0_biconic_body(_bic, ld_body, mach)
    _ns, _nl = _front_nose_aero(params, last)
    return _body_cd0(last, mach, _ns, _nl)


def whole_booster_LD(params: BoosterParams, mach: float = 3.0,
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
    # As-flown front-end nose (the RO's for a non-separating body — the stage
    # itself carries none), feeding BOTH the planform fill and the Cd0 so the
    # whole build-up sees the real nose, not a generic fallback.
    # A declared biconic is two cones: its planform is the fore triangle + the
    # aft trapezoid (not one fill fraction), and its Cd0 is the two-cone
    # build-up — so the L/D denominator and the crossflow planform see the real
    # shape.  Falls through to the single-nose path when not a valid biconic.
    from booster_models import biconic_nose_geometry, cd0_biconic_body
    _bic = biconic_nose_geometry(params)
    _nose_shape_eff, L_nose = _front_nose_aero(params, last)
    # The crossflow planform CENTROID (station aft of the nose tip) is built from
    # the same decomposition as the area.  The viscous-crossflow normal force acts
    # on the planform, so its line of action is the planform centroid — needed by
    # the trim gate's moment balance (trim_gate.py).  It is derived here, next to
    # the area, so the two can never drift apart.  Each piece contributes
    # area x its own centroid: a triangle apex-forward at 2/3 of its length, a
    # trapezoid at its area-weighted centroid, a rectangle at its mid-length.
    if _bic is not None:
        _Lf, _La = _bic['fore_len_m'], _bic['aft_len_m']
        _bd = _bic['break_diameter_m']
        _Ln = _bic['nose_len_m']
        # fore triangle (tip→break) + aft trapezoid (break→base) + afterbody
        _a_fore = 0.5 * _Lf * _bd
        _a_aft = 0.5 * (_bd + d) * _La
        _a_body = max(0.0, L_body - _Ln) * d
        A_p_body = _a_fore + _a_aft + _a_body
        # trapezoid centroid from its forward edge: (La/3)*(2d + bd)/(d + bd)
        _x_aft = (_Lf + (_La / 3.0) * (2.0 * d + _bd) / (d + _bd)
                  if (d + _bd) > 0 else _Lf + 0.5 * _La)
        _x_body_piece = _Ln + 0.5 * max(0.0, L_body - _Ln)
        x_p_body = ((_a_fore * (2.0 / 3.0) * _Lf + _a_aft * _x_aft
                     + _a_body * _x_body_piece) / A_p_body
                    if A_p_body > 0 else 0.5 * L_body)
        cd0 = cd0_biconic_body(_bic, (L_body / d) if d > 0 else 5.0, mach)
    else:
        if 0.0 < L_nose < L_body:
            nose = _SHAPE_ALIAS.get(_nose_shape_eff or '', _nose_shape_eff or '')
            fill = _NOSE_PLANFORM_FILL.get(nose, 0.667)   # cone 0.5, ogive ~0.67
            _a_nose = fill * L_nose * d
            _a_aft = (L_body - L_nose) * d
            A_p_body = _a_nose + _a_aft
            # A cone's side projection is a triangle (centroid 2/3 back); a fuller
            # (ogive) nose is nearer its mid-length.  Interpolate on the same fill
            # factor that sets the area, so shape enters both consistently.
            _x_nose = L_nose * (2.0 / 3.0 if fill <= 0.5 else 0.5 + (0.667 - fill))
            x_p_body = ((_a_nose * _x_nose
                         + _a_aft * (L_nose + 0.5 * (L_body - L_nose))) / A_p_body
                        if A_p_body > 0 else 0.5 * L_body)
        else:
            A_p_body = 0.5 * L_body * d                # pointed-body fallback
            x_p_body = (2.0 / 3.0) * L_body            # triangle, apex at the tip
        cd0 = _body_cd0(last, mach, _nose_shape_eff, L_nose)

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

    # Chordwise centroid of one exposed fin panel, measured AFT OF THE ROOT
    # LEADING EDGE.  The panel's VISCOUS (crossflow) normal force acts here,
    # not at the fin aerodynamic centre where the potential part acts -- Simon &
    # Blake, "Missile Datcom: High Angle of Attack Capabilities", AIAA 99-4258
    # (AFRL), which states the split as C_m = (x_ac - x_cg)*C_N,p +
    # (x_c - x_cg)*C_N,v with "the viscous normal force is assumed to act at the
    # panel centroid".  trim_gate.py places it from the fin station.
    #
    # Closed form for a straight-tapered panel, taper ratio lam = c_tip/c_root,
    # leading-edge sweep Lam, exposed semispan s_e.  Integrating the local chord
    # midpoint x_LE(y) + c(y)/2 weighted by chord c(y) over the span:
    #     x_c = [ s_e*tan(Lam)*(1+2*lam)/6 + (c_root/2)*(lam^2+lam+1)/3 ]
    #           / ((1+lam)/2)
    # Reduces to c_root/2 for an unswept rectangular panel, as it must.
    fin_centroid_aft_le = 0.0
    if S_W > 0 and last.fin_root_chord_m > 0:
        _cr = float(last.fin_root_chord_m)
        _lam = max(0.0, float(last.fin_tip_chord_m) / _cr)
        _tanL = math.tan(math.radians(float(last.fin_sweep_deg)))
        fin_centroid_aft_le = (
            (float(last.fin_span_m) * _tanL * (1.0 + 2.0 * _lam) / 6.0
             + (_cr / 2.0) * (_lam * _lam + _lam + 1.0) / 3.0)
            / ((1.0 + _lam) / 2.0))

    best_ld, best_a, curve = 0.0, 0.0, []
    for i in range(ALPHA_SWEEP_MIN_DEG, ALPHA_SWEEP_MAX_DEG + 1):   # alpha = 1..59 deg
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
               diameter_m=d, body_planform_m2=A_p_body, fin_planform_m2=A_p_fin,
               body_planform_centroid_m=x_p_body, ref_area_m2=A_ref,
               body_length_m=L_body,
               fin_centroid_aft_le_m=fin_centroid_aft_le,
               fin_root_chord_m=float(getattr(last, 'fin_root_chord_m', 0.0) or 0.0))
    if return_curve:
        out["curve"] = curve
    return out


def cn_components(aero: dict, alpha_rad: float) -> dict:
    """Split the build-up's normal force at one angle of attack into the three
    contributions that act at DIFFERENT stations, so a caller can take moments.

    ``aero`` is a whole_booster_LD() result.  The split is exactly the C_N the
    sweep in whole_booster_LD forms, term for term:

        C_N(a) = c_na_body * sin(2a)/2        <- slender-body potential, at the
                                                 body's Barrowman c.p.
               + c_na_fin  * sin(2a)/2        <- fin + N-K-P carryover, at the
                                                 fin station
               + eta * C_dn(M sin a) * (A_p/A_ref) * sin^2 a
                                              <- Allen-Perkins viscous crossflow,
                                                 at the planform centroid

    The planform-centroid station is not a modelling guess: Simon & Blake,
    "Missile Datcom: High Angle of Attack Capabilities", AIAA 99-4258 (AFRL,
    1999), describing Missile DATCOM's own implementation of this same
    Allen-Perkins / Jorgensen build-up, states that "the center of pressure of
    the body at large angles of attack is effectively at the planform centroid",
    and gives the moment as the two-station sum
    C_m = (x_ac - x_cg)*C_N,potential + (x_c - x_cg)*C_N,viscous (their Eq. 6),
    with the fin's viscous part at the PANEL centroid.  trim_gate.py takes
    moments in exactly that form.

    Summing the three reproduces whole_booster_LD's C_N identically; only the
    grouping is new.  The crossflow term is split between body and fin planform
    in proportion to their areas, because the two act at different stations.
    Returns each term (referenced to A_ref, per the same convention).
    """
    a = float(alpha_rad)
    sn = math.sin(a)
    s2 = math.sin(2.0 * a) / 2.0
    A_ref = float(aero.get('ref_area_m2', 0.0) or 0.0)
    A_pb = float(aero.get('body_planform_m2', 0.0) or 0.0)
    A_pf = float(aero.get('fin_planform_m2', 0.0) or 0.0)
    c_dn = crossflow_cd(float(aero.get('mach', 0.0)) * abs(sn))
    # sn*|sn| keeps the crossflow force odd in alpha, so the moment balance has
    # the right sign for a negative (nose-down) angle of attack too.
    _q = _ETA * c_dn * (sn * abs(sn)) / A_ref if A_ref > 0 else 0.0
    return dict(
        body_potential=float(aero.get('c_na_body', 0.0)) * s2,
        fin_potential=float(aero.get('c_na_fin', 0.0)) * s2,
        crossflow_body=_q * A_pb,
        crossflow_fin=_q * A_pf,
    )


def derive_glider_LD(params: BoosterParams, mach: float = GLIDE_MACH_REF) -> float:
    """Geometry-derived max L/D of a no-separation airframe (the value to use as
    glider_LD for a body glider).  Thin wrapper over whole_booster_LD; returns
    0.0 if it cannot be computed."""
    try:
        return float(whole_booster_LD(params, mach=mach).get("ld_max", 0.0))
    except Exception:
        return 0.0


if __name__ == "__main__":
    from booster_models import get_booster, BoosterParams as MP
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
        r = whole_booster_LD(body, mach=M)
        print(f"finless slender body  M{M}: L/D_max={r['ld_max']:.2f} at "
              f"{r['alpha_deg']:.0f}deg  (Cd0={r['cd0']:.3f})")
    finned = MP(name="no-sep finned body", diameter_m=0.5, length_m=4.0,
                nose_shape="tangent_ogive", nose_length_m=1.5, has_fins=True,
                n_fins=4, fin_span_m=0.3, fin_root_chord_m=0.4, fin_tip_chord_m=0.2,
                fin_sweep_deg=20.0, fin_thickness_m=0.02,
                mass_initial=500, mass_propellant=0, mass_final=500, burn_time_s=1,
                isp_s=1, thrust_N=1)
    for M in (2.0, 3.0, 5.0):
        r = whole_booster_LD(finned, mach=M)
        print(f"no-sep finned body    M{M}: L/D_max={r['ld_max']:.2f} at "
              f"{r['alpha_deg']:.0f}deg  (k_sum={r['k_sum']:.2f}, "
              f"cla_W={r['cla_wing']:.2f})")
