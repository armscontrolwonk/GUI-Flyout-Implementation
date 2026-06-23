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

Pitching moment about the CG (per radian), linearised:
    C_mα = -SM · C_Nα,total                         (static restoring; <0 stable)
    C_mδ = -(x_fin - x_CG)/d · C_Nδ                  (control, fins aft)
Trim (C_m = 0) gives the AoA reachable at control deflection δ:
    α_trim(δ) = (C_Nδ/C_Nα,total) · (x_fin - x_CG)/(x_CP - x_CG) · δ
so the maximum trimmable AoA at full deflection δ_max is
    α_trim,max = (C_Nδ/C_Nα,total) · (x_fin - x_CG)/(x_CP - x_CG) · δ_max .

Control normal-force slope (all-moving-fin model, ref body base area):
    C_Nδ = control_eff · C_Nα,fin
where control_eff is the N-K-P deflection-vs-AoA effectiveness ratio
k_W(B)/K_W(B): ~1.0 for an ideal all-moving fin, ~0.85 typical, ~0.5 for a
trailing-edge flap (NACA 1307 deflection analysis, Appendix A / chart 1).

OUTCOMES:
  * SM <= 0  -> statically unstable -> tumbles -> reenters BALLISTICALLY (L/D≈0).
  * SM > 0, α_trim,max >= α_LDmax -> control reaches the best-glide AoA;
    achievable L/D = the L/D-max from glider_ld.
  * SM > 0, α_trim,max < α_LDmax -> control-limited; achievable L/D is the
    (lower) value of the L/D curve at α_trim,max.  Very stiff (over-stable)
    vehicles fall here with a small α_trim,max and a weak achievable glide.

This is a preliminary-design gate, not a 6-DOF trim solution: it uses linearised
small-AoA moment slopes, a single control set, and the constant-area-reference
build-up of glider_ld.  CG and fin station are estimates (overridable).
"""

from __future__ import annotations
import math
import glider_ld
import grid_fin_sizing as gfs


def trim_gate(params, mach: float = 3.0, delta_max_deg: float = 25.0,
              control_eff: float = 0.85, x_cg_m: float = None,
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
    ld = glider_ld.whole_missile_LD(params, mach=mach, return_curve=True)
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

    # 5. Control authority -> max trimmable AoA.
    c_n_delta = control_eff * c_na_fin
    delta_max = math.radians(delta_max_deg)
    if sm_cal > 0 and (x_cp - x_cg) > 1e-9 and c_na_fin > 0:
        alpha_trim = (c_n_delta / c_na_total) * (x_fin - x_cg) / (x_cp - x_cg) * delta_max
        alpha_trim_deg = math.degrees(alpha_trim)
    else:
        alpha_trim_deg = 0.0

    # 6. Achievable glide + verdict.
    def ld_at(adeg):
        curve = ld.get("curve", [])
        best = 0.0
        for a, v in curve:
            if a <= adeg + 1e-9:
                best = v
        return best

    if sm_cal <= 0:
        verdict = "UNSTABLE (CP fwd of CG) -> tumbles -> ballistic reentry (no glide)"
        alpha_glide = 0.0
        ld_ach = 0.0
    else:
        alpha_glide = min(alpha_ldmax, alpha_trim_deg)
        ld_ach = ld_max if alpha_trim_deg >= alpha_ldmax else ld_at(alpha_trim_deg)
        if alpha_trim_deg >= alpha_ldmax:
            verdict = "stable; control reaches best-glide AoA -> full L/D available"
        elif alpha_trim_deg < 2.0:
            verdict = ("stable but very stiff/weak control (alpha_trim < 2 deg) "
                       "-> minimal maneuver, near-ballistic")
        else:
            verdict = (f"stable but CONTROL-LIMITED to ~{alpha_trim_deg:.0f} deg "
                       f"(< best-glide {alpha_ldmax:.0f} deg) -> reduced L/D")

    return dict(
        static_margin_cal=sm_cal, x_cp_m=x_cp, x_cg_m=x_cg, x_fin_m=x_fin,
        c_na_total=c_na_total, c_na_body=c_na_body, c_na_fin=c_na_fin,
        alpha_LDmax_deg=alpha_ldmax, LD_max=ld_max,
        alpha_trim_max_deg=alpha_trim_deg, delta_max_deg=delta_max_deg,
        control_eff=control_eff,
        alpha_glide_deg=alpha_glide, LD_achievable=ld_ach,
        verdict=verdict, mach=mach, diameter_m=d,
    )


if __name__ == "__main__":
    from missile_models import MissileParams as MP

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
