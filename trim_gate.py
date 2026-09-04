"""
Trim / control gate for a no-separation glider.

A no-sep airframe only *achieves* an L/D if it can actually trim (and hold) the
angle of attack where that L/D occurs.  This gate couples three things we now
compute from first principles:

  * the L/D-vs-AoA curve and its peak AoA  (glider_ld.py: Jorgensen + Allen-
    Perkins + N-K-P build-up),
  * the static margin  SM = (x_CP - x_CG)/d  (grid_fin_sizing.py: Barrowman
    body normal force incl. diameter transitions + the fin normal force, with a
    mass-stack CG estimate),
  * the control authority of the fins.

Pitching moment about the CG, NONLINEAR.  The build-up's normal force is not
linear in α, and its three terms act at three different stations, so the moment
is summed term by term (glider_ld.cn_components) rather than from a single slope:

    C_m(α,δ)·d = -[ C_N,body(α)·(x_body - x_CG)        slender-body potential
                  + C_N,cross(α)·(x_planform - x_CG)   Allen-Perkins crossflow
                  + C_N,fin(α)·(x_fin - x_CG)          fin + N-K-P carryover
                  + control_eff·C_Nα,fin·δ·(x_fin - x_CG) ]   commanded control

α_trim(δ) is the root of C_m = 0, found by bisection on the same α sweep the L/D
curve uses (1–59°), so nothing is extrapolated.  α_trim,max = α_trim(δ_max).

This replaced a linearised relation,
    α_trim,max = (C_Nδ/C_Nα,total)·(x_fin - x_CG)/(x_CP - x_CG)·δ_max ,
which assumed a constant normal-force slope and a FIXED centre of pressure.  Both
fail here: the crossflow term grows as sin²α on the body PLANFORM, well aft of
the nose c.p., so the c.p. migrates aft and the airframe stiffens with incidence.
With a fixed c.p. the denominator is a small difference of large numbers, and
near-neutral stability sent it to nonsense — a Scud-B returned α_trim = 144°,
which the gate read as "control reaches best glide" and so handed back the
unconstrained aerodynamic peak.  A gate whose failure mode is to grant full L/D
is not a gate.

Control authority is READ FROM THE VEHICLE (control_authority(), below), not
assumed: the reentry object's `glider_control_surfaces` descriptor sets the
usable deflection, within the Kumar & Stollery separation band this repo already
uses for the damping estimator.  A body with fixed surfaces ('none') has no
commanded deflection, trims at zero incidence, and does not glide — the branch a
fin-stabilised ballistic missile body belongs in, which the gate could not
previously express because it assumed a 25° all-moving control on every finned
airframe.

Control normal-force slope (all-moving-fin model, ref body base area):
    C_Nδ = control_eff · C_Nα,fin
where control_eff is the N-K-P deflection-vs-AoA effectiveness ratio
k_W(B)/K_W(B), DERIVED from fin geometry (glider_ld.control_effectiveness), not
assumed.  It replaced a hard-coded 0.85 that had no backing document, and it is
a FUNCTION of body-radius/semispan -- about 1.0 for a vanishing body down to
about 0.52 for a fin nearly buried in one -- so no constant could have been right
in form.  A Scud-B's fins give 0.66, i.e. the old constant had been overstating
control authority.

Why that ratio is the whole answer.  NACA Rep. 1307 (Drive; linked from
data/REFERENCES.md) defines all four factors
on a COMMON normalisation -- its Eqs. (4),(5) for K_B(W),K_W(B) and (7),(8) for
k_B(W),k_W(B), every one of them divided by the same wing-alone (C_La)_W.  Moore,
McInville & Hymer (JSR 33(3), 1996) give the fin construction as
[k_W(B) + k_B(W)] / [K_W(B) + K_B(W)], which is what c_na_fin's carryover needs.
NACA 1307 Eq. (34) then collapses it:

    k_B(W) ~= k_W(B) * K_B(W) / K_W(B)

(their k'_B(W); the report states this differs from the exact slender-body
k_B(W) of its Eq. (33) "by no more than 0.01").  Substituting,

    [k_W(B) + k_W(B)*K_B(W)/K_W(B)] / [K_W(B) + K_B(W)]
        = k_W(B)*[K_W(B) + K_B(W)] / K_W(B) / [K_W(B) + K_B(W)]
        = k_W(B) / K_W(B)

so the carryover cancels EXACTLY, leaving the simple ratio.  Both halves are
implemented: K_W(B) by glider_ld.nkp_interference, and k_W(B) by
glider_ld.nkp_deflection_factor -- the closed form of NACA 1307 **Eq. (19)** in
tau = s/r.  The Eq. (19) transcription is checked against its two theoretical
limits and against **Chart 1** of the same report (a shallow minimum near 0.93
at r/s ~ 0.4 between endpoints of 1.0, reproduced in both depth and location),
pinned by tests in test_ld_calibration.py.

OUTCOMES:
  * SM <= 0  -> statically unstable -> tumbles -> reenters BALLISTICALLY (L/D≈0).
  * δ_max = 0 ('none') -> nothing commands an incidence -> trims at zero lift
    -> ballistic reentry, however good the airframe's aerodynamic peak is.
  * SM > 0, α_trim,max >= α_LDmax -> control reaches the best-glide AoA;
    achievable L/D = the L/D-max from glider_ld.
  * SM > 0, α_trim,max < α_LDmax -> control-limited; achievable L/D is the best
    value of the L/D curve over the reachable band (0, α_trim,max].  Very stiff
    (over-stable) vehicles fall here with a small α_trim,max and a weak glide.
  * no root at δ_max -> the control moment beats the restoring moment everywhere
    on the sweep; reported as "not a glide", NOT as unlimited authority.

Achievable L/D is the BEST L/D over the reachable band, not the value at its
endpoint: deflection is commandable, so a vehicle that can reach α_trim,max can
also hold anything below it, and L/D peaks at best glide and falls away beyond.

This is a preliminary-design gate, not a 6-DOF trim solution: the moment balance
is static and longitudinal only, with a single control set and the
constant-area-reference build-up of glider_ld.  The normal force is now solved
nonlinearly, but the stations it acts through are fixed geometric estimates (the
Barrowman body c.p., the planform centroid, and an aft fin station), not
alpha-resolved centres of pressure.  CG and fin station are estimates
(overridable).  The fin term is linear wing theory with no stall, which is why a
trim solution past _ALPHA_FIN_LINEAR_DEG is reported but marked indicative.

Known limitation (cone noses): the Barrowman cone-nose c.p. fraction (0.666 of
nose length) is the slender-body value; the flight-substantiated sharp-cone
result is X_cp/l = 2/(3cos^2 delta), Mach-independent and constant to alpha
<= 90 deg (Eastman, Boeing D2-36139-1 / DTIC AD0376942, Eq. 3.2 + Fig. 3.6).
0.666 under-places the c.p. by ~3% at delta = 10 deg, ~13% at 20 deg, ~33% at
30 deg -> the gate slightly understates static margin for fat cones.  Fine at
screening tier for slender (RV-class) noses; see BENCHMARKING.md.
"""

from __future__ import annotations
import math
import glider_ld
import grid_fin_sizing as gfs

# --- Control authority ------------------------------------------------------
# Usable control deflection is NOT a free parameter: past the incipient-separation
# angle the boundary layer separates ahead of the surface and effectiveness
# collapses.  The tiers below map onto that physics.
#
# SOURCE, NOW VERIFIED AND PARTLY CONTRADICTING THESE NUMBERS (2026-09-04).
# The anchor is Kumar, D. & Stollery, J. L., "Hypersonic Control Flap
# Effectiveness", ICAS-94-4.4.3, 19th ICAS Congress, 1994, pp. 1194-1204
# (Cranfield).  It was read against the paper on 2026-09-04, and two things it
# had been cited for are wrong:
#   * It is an ICAS 1994 congress paper, not Aeronautical Journal 100(996) 1996.
#   * It is at M 8.2, not "M ~ 10", and it states NO "usable deflection 5-15 deg"
#     band and no "critical deflection ~ 15 deg".
# What it DOES report: laminar incipient separation at flap angle 7.8 deg at
# alpha = 5 deg and 8.4 deg at alpha = 10 deg; flow still attached at beta = 10
# deg when alpha = 10 deg (incidence delays separation); and the flap boundary
# layer laminar at beta = 5, transitional at 15, turbulent at 25.  Bluntness
# "causes significant loss of control effectiveness".
#
# So the paper anchors separation onset near 8 deg, not 15.  The 15 deg upper
# tier below is therefore NOT sourced -- it survives from the superseded reading,
# and damping_estimate.DELTA_MAX_DEG rests on the same one.  Re-anchoring both is
# a physics decision with flown consequences (it would cut achievable glide for
# every control-rich body), so it is FLAGGED here and in TODO.md rather than
# changed silently.  See docs/cl_margin_references.md for the verified entry.
#
# The qualitative tiers map onto that band:
#
#   none        -- fixed surfaces: no commanded deflection at all.  The body
#                  trims where its own aerodynamics put it (alpha ~ 0 for a
#                  statically stable airframe), so it does not glide.
#   small       -- 5 deg, laminar and attached in the paper's tests.
#   substantial -- 15 deg.  NOT sourced; see the note above.
#   unknown     -- 10 deg, reported as an ASSUMPTION.  Set the reentry object's
#                  glider_control_surfaces to replace it.  Notably this is the
#                  one tier the paper does support: attached flow was observed
#                  at beta = 10 deg with incidence.
#
# Laying tier names onto deflection angles is a modelling choice either way; no
# document in this repo grades those three words.
_DELTA_MAX_BY_CONTROL = {'none': 0.0, 'small': 5.0, 'substantial': 15.0,
                         'unknown': 10.0}
_DELTA_MAX_DEFAULT_DEG = 10.0

# Newtonian control predictions are optimistic once real-gas effects matter.
# Maus, Griffith, Szema & Best, J. Spacecraft & Rockets 21(2), 1984 (and the
# STS-1 trim anomaly): the real-gas gamma reduction shifted the centre of
# pressure and the Shuttle body flap needed ~16 deg against ~11 deg predicted.
# docs/cl_margin_references.md turns that into a derate above M ~ 5-7;
# damping_estimate.py applies it as REAL_GAS_DERATE.  Applied here to the
# control term for the same reason, so the two estimators agree.
_REAL_GAS_DERATE = 0.85
_REAL_GAS_MACH = 7.0

# Above this angle of attack the FIN term stops being trustworthy: c_na_fin comes
# from linear wing theory (wing_alone_cla), which has no stall and no separation.
# The body terms are Jorgensen's and stay valid to high alpha, but a "glide"
# resolved by an extrapolated fin slope is not a screening result we should
# report as one.  A trim solution beyond this is reported, and flagged, as a
# high-incidence (broadside) attitude rather than a glide.
_ALPHA_FIN_LINEAR_DEG = 25.0
# The moment balance is bracketed on the build-up's OWN alpha sweep, imported
# rather than copied so the two cannot drift apart.
_ALPHA_SWEEP_MAX_DEG = float(glider_ld.ALPHA_SWEEP_MAX_DEG)


def control_authority(ro, mach: float = 0.0,
                      fin_body_radius_m: float = 0.0,
                      fin_semispan_m: float = 0.0) -> dict:
    """Commanded control authority for a no-separation body, from the reentry
    object's ``glider_control_surfaces`` descriptor.

    fin_body_radius_m / fin_semispan_m give the fin geometry the effectiveness
    ratio is derived from (body radius, and semispan measured from the body
    AXIS).  Omit them and the ratio falls back to the wing-alone value 1.0.

    Returns delta_max_deg (usable one-sided deflection), control_eff (the N-K-P
    deflection-vs-AoA effectiveness ratio, real-gas derated when the reference
    Mach warrants it), the tier name, and whether the value is an assumption.
    """
    tier = str(getattr(ro, 'glider_control_surfaces', 'unknown') or 'unknown').lower()
    assumed = tier not in _DELTA_MAX_BY_CONTROL or tier == 'unknown'
    delta = _DELTA_MAX_BY_CONTROL.get(tier, _DELTA_MAX_DEFAULT_DEG)
    # An explicit deflection on the object overrides the tier, still capped by the
    # separation limit (the same precedence damping_estimate.py uses).
    explicit = float(getattr(ro, 'glider_flap_deflection_deg', 0.0) or 0.0)
    if explicit > 0.0:
        delta = min(explicit, _DELTA_MAX_BY_CONTROL['substantial'])
        assumed = False
    # control_eff is the N-K-P deflection-vs-AoA effectiveness ratio k_W(B)/K_W(B),
    # now DERIVED from fin geometry rather than assumed.  Both factors are NACA
    # 1307 slender-body results on a common normalisation, so the ratio is exact
    # within that theory; see glider_ld.control_effectiveness for the derivation
    # and for why the body carryover cancels instead of being neglected.
    #
    # This replaces a hard-coded 0.85 that had no backing document.  The derived
    # value is a FUNCTION of body-radius/semispan, running ~1.0 for a vanishing
    # body down to ~0.52 for a fin nearly buried in one, so no constant could
    # have been right in form.  For typical tail-fin geometries it lands well
    # BELOW 0.85 (a Scud-B's fins give 0.66), meaning the old constant
    # overstated control authority -- the non-conservative direction.
    eff = (glider_ld.control_effectiveness(fin_body_radius_m, fin_semispan_m)
           if (fin_body_radius_m > 0.0 and fin_semispan_m > fin_body_radius_m)
           else 1.0)
    if mach and float(mach) >= _REAL_GAS_MACH:
        eff *= _REAL_GAS_DERATE
    return dict(delta_max_deg=float(delta), control_eff=float(eff),
                tier=tier if tier in _DELTA_MAX_BY_CONTROL else 'unknown',
                assumed=bool(assumed))


def cg_targets(params, mach: float = 3.0, sm_target_cal: float = 0.5,
               find_glide_cg: bool = True) -> dict:
    """Where the reentry CG has to be, in metres aft of the nose.

    "Set the CG forward" is not actionable on its own -- forward of WHAT?  This
    answers that.  Everything here follows from the gate's own static margin
    definition, SM = (x_cp - x_cg)/d, so a target margin inverts directly:

        x_cg(SM) = x_cp - SM * d

    Returns, all in metres aft of the nose tip unless noted:
      neutral_point_m   x_cp.  The CG must be FORWARD of this or the body is
                        statically unstable and tumbles.  This is the hard limit.
      x_cg_m            what the CG estimate currently gives.
      static_margin_cal current margin, in calibers.
      cg_for_sm_m       CG that would give sm_target_cal.
      cg_for_glide_m    (find_glide_cg only -- it costs a few hundred gate
                        evaluations, so callers driving a live preview can skip
                        it; the other fields are closed-form and free)
                        CG at which the body first reaches its best-glide angle
                        of attack, i.e. the full aerodynamic ceiling, or None if
                        no CG in the stable range gets there at the vehicle's
                        control authority.  Found by scanning the stable range,
                        because trim alpha depends on the CG through the whole
                        nonlinear moment balance, not through SM alone.
      margin_m          how far forward of neutral the current CG sits (>0 stable).
      body_length_m     the length these stations are measured along.

    Note the trade the numbers will show: moving the CG forward buys stability
    but COSTS glide, because a stiffer airframe trims to a smaller angle of
    attack at the same deflection.  There is usually a window, not a maximum.
    """
    g = trim_gate(params, mach=mach)
    if g.get('error'):
        return dict(error=g['error'])
    d = float(g['diameter_m'])
    x_cp = float(g['x_cp_m'])
    last = glider_ld._last_stage(params)
    L = float(getattr(last, 'length_m', 0.0) or 0.0)

    cg_for_glide = None
    if L > 0 and find_glide_cg:
        # Scan forward from the neutral point.  Step is fine enough to land
        # within a few cm on a missile-scale body.
        steps = 400
        for i in range(1, steps + 1):
            x = x_cp - (i / steps) * x_cp
            if x <= 0:
                break
            r = trim_gate(params, mach=mach, x_cg_m=x)
            if r.get('error') or r.get('tumbles'):
                continue
            if r['LD_achievable'] >= g['LD_max'] - 1e-9:
                cg_for_glide = x
                break
    return dict(
        neutral_point_m=x_cp,
        x_cg_m=float(g['x_cg_m']),
        static_margin_cal=float(g['static_margin_cal']),
        margin_m=x_cp - float(g['x_cg_m']),
        cg_for_sm_m=x_cp - float(sm_target_cal) * d,
        sm_target_cal=float(sm_target_cal),
        cg_for_glide_m=cg_for_glide,
        body_length_m=L,
        diameter_m=d,
        LD_max=float(g['LD_max']),
        LD_achievable=float(g['LD_achievable']),
        tumbles=bool(g.get('tumbles')),
    )


def trim_gate(params, mach: float = 3.0, delta_max_deg: float = None,
              control_eff: float = None, x_cg_m: float = None,
              fin_station_m: float = None) -> dict:
    """Assess whether a no-sep airframe can trim to its best-glide AoA.

    delta_max_deg : max control-surface deflection (one-sided), deg.
    control_eff   : C_Nδ/C_Nα,fin (all-moving≈1.0, typical≈0.85, flap≈0.5).
    Returns a dict with the static margin, the L/D-max and trim-limited AoAs,
    the achievable L/D, and a verdict."""
    last = glider_ld._last_stage(params)
    d = float(last.diameter_m)
    if d <= 0:
        return dict(error="body diameter not set")

    # 1. L/D curve + peak (glider_ld) — also gives the body/fin slope split.
    ld = glider_ld.whole_booster_LD(params, mach=mach, return_curve=True)
    alpha_ldmax = ld["alpha_deg"]
    ld_max = ld["ld_max"]
    c_na_total = ld["c_na_pot"]
    c_na_fin = ld["c_na_fin"]

    # 2. Body normal force (incl. diameter transitions) + CG.
    c_na_body, x_body = gfs.body_normal_force(params)
    x_cg, L_total = gfs.estimate_cg(params)
    if x_cg_m is not None:
        x_cg = x_cg_m

    # 3. Fin station (control set near the aft of the finned body).
    if fin_station_m is not None:
        x_fin = fin_station_m
    else:
        x_fin = L_total - 0.5 * max(float(last.fin_root_chord_m or 0.0), 0.0)

    # 4. Combined CP and static margin (calibers).
    if c_na_total <= 0:
        return dict(error="no normal-force slope (check geometry/fins)")
    x_cp = (c_na_body * x_body + c_na_fin * x_fin) / c_na_total
    sm_cal = (x_cp - x_cg) / d

    # 5. Control authority -> max trimmable AoA, by NONLINEAR moment balance.
    #
    # The linearised relation this replaced,
    #     alpha_trim = (C_Nd/C_Na) * (x_fin - x_cg)/(x_cp - x_cg) * delta_max ,
    # assumes a constant normal-force slope and a FIXED centre of pressure.  Both
    # fail on a slender finned body: the build-up's own C_N is strongly nonlinear
    # (a sin(2a)/2 potential term plus a sin^2(a) viscous-crossflow term), and the
    # crossflow term acts on the body PLANFORM, well aft of the nose c.p., so the
    # centre of pressure migrates aft as alpha grows and the airframe stiffens.
    # With a fixed c.p. the (x_cp - x_cg) denominator is a small difference of two
    # large numbers, and near-neutral stability sent it to absurd values -- a
    # Scud-B returned alpha_trim = 144 deg, which the gate then read as "control
    # reaches best glide", handing back the unconstrained aerodynamic peak.  A
    # gate whose failure mode is to grant full L/D is not a gate.
    #
    # Instead, balance the actual moment about the CG using the SAME term-by-term
    # C_N the L/D sweep uses (glider_ld.cn_components), each term at its own
    # station: the slender-body potential term at the Barrowman body c.p., the
    # viscous crossflow at the planform centroid, and the fin term (plus the
    # commanded control increment) at the fin station.  Solved by bisection on
    # the build-up's own alpha sweep interval, so nothing is extrapolated.
    _r_fin = 0.5 * d
    _s_fin = _r_fin + float(getattr(last, 'fin_span_m', 0.0) or 0.0)
    ctrl = control_authority(getattr(params, 'ro', None), mach=mach,
                             fin_body_radius_m=_r_fin, fin_semispan_m=_s_fin)
    if delta_max_deg is None:
        delta_max_deg = ctrl['delta_max_deg']
        delta_assumed = ctrl['assumed']
    else:
        delta_assumed = False
    if control_eff is None:
        control_eff = ctrl['control_eff']
    x_p = float(ld.get('body_planform_centroid_m', 0.0) or 0.0)
    # The fin's POTENTIAL normal force acts at its aerodynamic centre (x_fin,
    # approximated here by the root mid-chord), but its VISCOUS crossflow part
    # acts further aft, at the PANEL CENTROID -- Simon & Blake, AIAA 99-4258
    # (AFRL), Eq. 6: C_m = (x_ac - x_cg)*C_N,p + (x_c - x_cg)*C_N,v, with "the
    # viscous normal force is assumed to act at the panel centroid".  Lumping
    # both at x_fin, as this gate did, understates the aft shift of the fin's
    # contribution as alpha grows.  x_fin is the root mid-chord, so the root
    # leading edge is x_fin - c_root/2 and the centroid sits the returned offset
    # aft of that.  For a swept, tapered panel the centroid is aft of the root
    # mid-chord; for an unswept rectangular panel the two coincide.
    _cr_fin = float(ld.get('fin_root_chord_m', 0.0) or 0.0)
    x_fin_panel = (x_fin - 0.5 * _cr_fin
                   + float(ld.get('fin_centroid_aft_le_m', 0.0) or 0.0))

    def _cm(alpha_rad, delta_rad):
        """Pitching moment about the CG (per diameter), positive nose-up.

        A component's normal force acting AFT of the CG pitches the nose DOWN, so
        each term enters as -(x_i - x_cg): the sum is the restoring moment.
        """
        c = glider_ld.cn_components(ld, alpha_rad)
        m = (c['body_potential'] * (x_body - x_cg)
             + c['crossflow_body'] * (x_p - x_cg)
             + c['fin_potential'] * (x_fin - x_cg)
             + c['crossflow_fin'] * (x_fin_panel - x_cg))
        # Commanded deflection adds normal force at the fin station.  A positive
        # (trailing-edge-down) deflection pitches the nose UP, hence the sign.
        m += (control_eff * c_na_fin * delta_rad) * (x_fin - x_cg)
        return -m / d

    def _solve_trim(delta_rad):
        """Largest alpha in (0, sweep_max] where the moment balances, or None."""
        lo, hi = 1e-6, math.radians(_ALPHA_SWEEP_MAX_DEG)
        f_lo, f_hi = _cm(lo, delta_rad), _cm(hi, delta_rad)
        if f_lo == 0.0:
            return 0.0
        if f_lo * f_hi > 0:
            return None            # no trim point on the interval
        # Bisect to 1e-4 deg, carrying f_lo rather than recomputing it: this runs
        # interactively behind the GUI's derived-L/D preview.  ~20 halvings cover
        # the 59 deg bracket; the cap is a guard, not the expected count.
        tol = math.radians(1e-4)
        for _ in range(80):
            if (hi - lo) < tol:
                break
            mid = 0.5 * (lo + hi)
            f_mid = _cm(mid, delta_rad)
            if f_lo * f_mid <= 0:
                hi = mid
            else:
                lo, f_lo = mid, f_mid
        return math.degrees(0.5 * (lo + hi))

    if sm_cal > 0 and c_na_fin > 0 and delta_max_deg > 0:
        # Deflect trailing-edge-down (nose-up) at full authority.
        _a = _solve_trim(-math.radians(delta_max_deg))
        alpha_trim_deg = 0.0 if _a is None else float(_a)
        trim_unbounded = _a is None
    else:
        # No commanded deflection (fixed surfaces), or no fin/stability at all:
        # a statically stable body trims at zero incidence and does not glide.
        alpha_trim_deg = 0.0
        trim_unbounded = False

    # 6. Achievable glide + verdict.
    #
    # Deflection is COMMANDABLE, so a vehicle that can reach alpha_trim_max can
    # also hold anything below it: the achievable glide is the BEST L/D over the
    # reachable band (0, alpha_trim_max], not the value at the endpoint.  That
    # matters because L/D is not monotonic in alpha -- it peaks at the best-glide
    # incidence and falls away beyond it, so a vehicle with lots of authority
    # would otherwise be scored on an over-rotated attitude it need not fly.
    def ld_best_upto(adeg):
        best_v, best_a = 0.0, 0.0
        for a, v in ld.get("curve", []):
            if a <= adeg + 1e-9 and v > best_v:
                best_v, best_a = v, a
        return best_v, best_a

    if sm_cal <= 0:
        verdict = "UNSTABLE (CP fwd of CG) -> tumbles -> ballistic reentry (no glide)"
        alpha_glide = 0.0
        ld_ach = 0.0
    elif delta_max_deg <= 0.0:
        # Fixed surfaces: nothing commands an incidence, so a stable body flies at
        # zero lift.  This is the branch a fin-stabilised ballistic missile body
        # belongs in, and the one the gate could not previously express.
        verdict = ("stable, but NO commanded control surfaces "
                   "(glider_control_surfaces = 'none') -> trims at zero "
                   "incidence -> ballistic reentry (no glide)")
        alpha_glide = 0.0
        ld_ach = 0.0
    elif trim_unbounded:
        # Full deflection never balances on the sweep: the control moment exceeds
        # the airframe's restoring moment everywhere, so there is no trimmed glide
        # to report.  Treated as a non-result, NOT as unlimited authority.
        verdict = (f"control moment exceeds the restoring moment at "
                   f"{delta_max_deg:.0f} deg -> no trimmed attitude below "
                   f"{_ALPHA_SWEEP_MAX_DEG:.0f} deg -> not a glide")
        alpha_glide = 0.0
        ld_ach = 0.0
    else:
        ld_ach, alpha_glide = ld_best_upto(alpha_trim_deg)
        if alpha_trim_deg >= alpha_ldmax:
            verdict = ("stable; control reaches best-glide AoA "
                       f"({alpha_ldmax:.0f} deg at {delta_max_deg:.0f} deg "
                       "deflection) -> full L/D available")
        elif alpha_trim_deg < 2.0:
            verdict = ("stable but very stiff/weak control (alpha_trim < 2 deg) "
                       "-> minimal maneuver, near-ballistic")
        else:
            verdict = (f"stable but CONTROL-LIMITED to ~{alpha_trim_deg:.0f} deg "
                       f"(< best-glide {alpha_ldmax:.0f} deg) -> reduced L/D")
        if alpha_trim_deg > _ALPHA_FIN_LINEAR_DEG:
            # Past the fin model's linear range the result is reported but marked:
            # a "glide" resolved by an extrapolated fin slope is not a screening
            # answer we should present as one.
            verdict += (f"; NOTE trim alpha {alpha_trim_deg:.0f} deg exceeds the "
                        f"{_ALPHA_FIN_LINEAR_DEG:.0f} deg linear-fin range - "
                        "high-incidence attitude, treat as indicative only")
    if delta_assumed and ld_ach > 0.0:
        verdict += ("; control authority ASSUMED "
                    f"({delta_max_deg:.0f} deg, tier '{ctrl['tier']}') - set the "
                    "reentry object's control surfaces to replace the assumption")

    # "No glide" and "tumbles" are DIFFERENT outcomes and must not be conflated.
    # Only a statically unstable body (SM <= 0) cannot hold its attitude and so
    # tumbles -- which changes its DRAG, because a tumbling body presents a large
    # mean projected area and its beta is re-derived as a tumbling cylinder
    # (METHODS 8.11).  A STABLE body with no commanded control still flies
    # nose-first; it simply flies at zero incidence and makes no lift.  Its
    # aeroshell beta is unchanged.  Callers must branch on `tumbles`, not on
    # LD_achievable == 0, or a fin-stabilised ballistic body would wrongly
    # inherit a tumbling drag coefficient.
    return dict(
        static_margin_cal=sm_cal, x_cp_m=x_cp, x_cg_m=x_cg, x_fin_m=x_fin,
        c_na_total=c_na_total, c_na_body=c_na_body, c_na_fin=c_na_fin,
        alpha_LDmax_deg=alpha_ldmax, LD_max=ld_max,
        alpha_trim_max_deg=alpha_trim_deg, delta_max_deg=delta_max_deg,
        control_eff=control_eff, control_tier=ctrl['tier'],
        control_assumed=bool(delta_assumed),
        trim_unbounded=bool(trim_unbounded),
        tumbles=bool(sm_cal <= 0),
        body_planform_centroid_m=x_p,
        alpha_glide_deg=alpha_glide, LD_achievable=ld_ach,
        verdict=verdict, mach=mach, diameter_m=d,
    )


if __name__ == "__main__":
    from booster_models import BoosterParams as MP

    def show(p, label, **kw):
        r = trim_gate(p, **kw)
        print(f"\n=== {label} (M{kw.get('mach',3.0)}, "
              f"delta_max={kw.get('delta_max_deg',25)} deg) ===")
        if "error" in r:
            print("  ", r["error"]); return
        print(f"  SM = {r['static_margin_cal']:+.2f} cal  (x_cp={r['x_cp_m']:.2f}, "
              f"x_cg={r['x_cg_m']:.2f}, x_fin={r['x_fin_m']:.2f})")
        print(f"  best-glide AoA = {r['alpha_LDmax_deg']:.0f} deg (L/D_max {r['LD_max']:.2f})"
              f" | trim-limited AoA = {r['alpha_trim_max_deg']:.0f} deg")
        print(f"  achievable: alpha {r['alpha_glide_deg']:.0f} deg, L/D = {r['LD_achievable']:.2f}")
        print(f"  -> {r['verdict']}")

    base = dict(diameter_m=0.5, length_m=4.0, nose_shape="tangent_ogive",
                nose_length_m=1.5, has_fins=True, n_fins=4, fin_span_m=0.3,
                fin_root_chord_m=0.4, fin_tip_chord_m=0.2, fin_sweep_deg=20.0,
                fin_thickness_m=0.02, mass_initial=500, mass_propellant=0,
                mass_final=500, burn_time_s=1, isp_s=1, thrust_N=1)
    finned = MP(name="no-sep finned glider", **base)
    show(finned, "finned glider, CG estimated", mach=3.0)
    # force CG aft of CP -> unstable
    show(finned, "same, CG forced near tail (unstable)", mach=3.0, x_cg_m=3.8)
    # force CG far forward -> very stable / control-limited
    show(finned, "same, CG forced far forward (over-stable)", mach=3.0, x_cg_m=0.6)
