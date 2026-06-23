"""
Grid-fin static-margin sizing check.

Purpose: tell the user whether a vehicle's grid fins are appropriately sized for
a typical static margin, given the body diameter.  Two uses:
  * SANITY-CHECK an OSINT fin estimate (does the measured fin give a sane margin?)
  * ESTIMATE a missing fin size (what frame area "should" a vehicle of this
    diameter carry for a typical margin?)

Method (Barrowman).  The static margin is
    SM = (x_CP - x_CG) / D            [calibers]
with the combined centre of pressure the normal-force-weighted average of the
body (nose) and grid-fin contributions,
    x_CP = (C_Na,body * x_body + C_Na,fin * x_fin) / (C_Na,body + C_Na,fin).
Body normal force is the Barrowman nose term (C_Na = 2 referenced to the body
base area, CP at a shape-dependent fraction of nose length).  Grid-fin normal
force is _cl_alpha_gridfins evaluated at a representative ascent Mach (the
grid-fin C_Na model is the one validated against the nine-paper set recorded in
data/ -- see METHODS.md, section 8.5).  CG is estimated from the stage mass
stack (mass-weighted longitudinal centroid) at liftoff/full unless supplied.

Scaling note (why this is a useful OSINT check): to hold the static margin in
*calibers* roughly constant across geometrically similar vehicles, fin AREA
scales as D^2 (tail-volume argument).  So a measured fin that gives a wildly
atypical margin for the vehicle's diameter is a flag that the OSINT geometry --
or the CG/station assumptions -- are off.

APPROXIMATIONS (first-order; all flagged in the output):
  * CG from the stage stack assumes uniform mass density within each stage and a
    simple nose->payload->upper-stages->stage-1 longitudinal layout.  Supply
    x_cg_m to override.
  * Body normal force is nose-only (classic Barrowman for a constant-diameter
    body); flares/boattails and viscous body-crossflow lift are not added.
  * The fin longitudinal station defaults to near the aft of the finned stage;
    supply fin_station_m to override.
  * The grid-fin normal-force slope is the APPROXIMATE _cl_alpha_gridfins lift
    model (less validated than the drag model; it likely over-estimates C_Na by
    treating the full web wetted area as lifting, and is insensitive to the
    solidity override).  This is the dominant uncertainty in the static-margin
    MAGNITUDE.  In practice the VERDICT is robust to it -- because the fins sit
    far aft of the CG, the combined CP (hence the margin band) changes little
    across a 3x swing in fin C_Na -- but a precise margin would want the lift
    slope calibrated against measured grid-fin C_Na (e.g. W&M AIAA 93-0035,
    Fig. 6).
This is a preliminary-design / OSINT sanity tool, not a substitute for
Mach-resolved wind-tunnel/CFD stability data.
"""

from __future__ import annotations
import math
from missile_models import (MissileParams, _cl_alpha_gridfins, effective_rv,
                             _SHAPE_ALIAS)

# Barrowman nose centre-of-pressure, as a fraction of nose length from the tip.
_NOSE_CP_FRACTION = {
    "cone":           0.666,
    "tangent_ogive":  0.466,
    "von_karman":     0.500,
    "lv_haack":       0.437,
    "parabola":       0.500,
    "blunt_cylinder": 0.500,
}

# Typical static-margin band (calibers) for a finned vehicle.
_SM_MARGINAL_LO = 0.5    # below this: marginal / insufficient
_SM_TYPICAL_HI  = 2.0    # above this: over-stabilised (over-finned)


def _nose_cp_fraction(shape: str) -> float:
    s = _SHAPE_ALIAS.get(shape, shape)
    return _NOSE_CP_FRACTION.get(s, 0.5)


def _front_nose(params: MissileParams):
    """(nose_shape, nose_length_m, body_diameter_m) for the as-flown front end.

    Prefers an attached shroud/fairing nose, then the stage nose, then the
    RV/payload nose.  Falls back to a 3-caliber tangent ogive if none is set.
    """
    d = params.diameter_m
    # shroud nose (fairing on during ascent)
    if params.shroud_nose_shape and params.shroud_nose_length_m > 0:
        return params.shroud_nose_shape, params.shroud_nose_length_m, d
    if params.nose_shape and params.nose_length_m > 0:
        return params.nose_shape, params.nose_length_m, d
    rv = effective_rv(params)
    if rv is not None and getattr(rv, 'shape', '') and getattr(rv, 'length_m', 0) > 0:
        # RV/HGB caps the stack and acts as the nose
        return rv.shape, rv.length_m, d
    return "tangent_ogive", 3.0 * d, d      # generic fallback


def estimate_cg(params: MissileParams):
    """Estimate (x_cg_m, total_length_m) from the stage stack at liftoff/full.

    x is measured aft from the nose tip.  The top-level ``length_m`` is taken as
    the FULL stack length (the convention in the shipped factories); the body
    region [nose, aft] is filled front->aft by the upper stages at their own
    ``length_m`` with the aft-most (first) stage taking the remainder, and each
    stage's OWN wet mass is m_i - m_{i+1} (cumulative-mass convention).  The
    payload/RV mass sits in the nose region when the RV caps the stack, else
    just behind a separate nose/shroud.  Approximate (see module docstring)."""
    nose_shape, nose_len, d = _front_nose(params)
    L_total = max(params.length_m, nose_len + 0.5)
    rv = effective_rv(params)
    payload = (params.payload_kg if params.payload_kg > 0
               else (rv.mass_kg if rv is not None else 0.0))
    nose_is_rv = not ((params.shroud_nose_shape and params.shroud_nose_length_m > 0)
                      or (params.nose_shape and params.nose_length_m > 0))

    chain = []
    s = params
    while s is not None:
        chain.append(s)
        s = s.stage2
    fwd_to_aft = list(reversed(chain))             # [stageN ... stage1]
    own = {}
    for i, st in enumerate(chain):
        nxt = chain[i + 1] if i + 1 < len(chain) else None
        upper = nxt.mass_initial if nxt is not None else payload
        own[id(st)] = max(st.mass_initial - upper, 0.0)

    body_start = nose_len
    body_len = L_total - nose_len
    upper_stages = fwd_to_aft[:-1]
    aft_stage = fwd_to_aft[-1]
    lens = {id(st): (st.length_m if st.length_m > 0 else 0.0)
            for st in upper_stages}
    upper_sum = sum(lens.values())
    if upper_sum >= body_len:                      # lengths inconsistent -> by mass
        tot = sum(own[id(st)] for st in fwd_to_aft) or 1.0
        lens = {id(st): body_len * own[id(st)] / tot for st in fwd_to_aft}
    else:
        lens[id(aft_stage)] = body_len - upper_sum

    seg = []                                       # (mass, x_centroid)
    if payload > 0:
        seg.append((payload, nose_len * 0.5 if nose_is_rv else nose_len))
    x = body_start
    for st in fwd_to_aft:
        L = max(lens[id(st)], 1e-6)
        seg.append((own[id(st)], x + 0.5 * L))
        x += L
    if params.n_boosters > 0:                      # strap-ons at the aft stage
        b = params.n_boosters * (params.booster_prop_kg + params.booster_inert_kg)
        seg.append((b, L_total - 0.5 * (lens[id(aft_stage)]
                                        if id(aft_stage) in lens else 1.0)))

    msum = sum(m for m, _ in seg)
    mx = sum(m * xx for m, xx in seg)
    x_cg = mx / msum if msum > 0 else 0.5 * L_total
    return x_cg, L_total


def body_normal_force(params: MissileParams):
    """(C_Na_body, x_cp_body_m) from the Barrowman nose term, referenced to the
    body base area.  Nose-only (constant-diameter body adds no potential lift)."""
    shape, nose_len, d = _front_nose(params)
    c_na = 2.0                                  # slender-body nose value
    x_cp = _nose_cp_fraction(shape) * nose_len
    return c_na, x_cp


def grid_fin_static_margin(params: MissileParams, mach: float = 1.5,
                           x_cg_m: float = None, fin_station_m: float = None,
                           solidity: float = None) -> dict:
    """Static margin (calibers) of the vehicle with its grid fins.

    Parameters
    ----------
    params        : MissileParams with has_grid_fins (on the finned stage)
    mach          : representative ascent Mach for the fin C_Na (default 1.5)
    x_cg_m        : override CG location (m aft of nose); else estimated
    fin_station_m : override fin longitudinal station (m aft of nose); else
                    near the aft of the finned stage
    solidity      : override grid_fin_solidity for the C_Na (else from params)

    Returns a dict with the margin, its components, a verdict, and the
    assumptions used.
    """
    d = params.diameter_m
    if not getattr(params, 'has_grid_fins', False) or params.n_grid_fins <= 0:
        raise ValueError("params has no grid fins (has_grid_fins / n_grid_fins).")

    c_na_body, x_body = body_normal_force(params)

    sig = params.grid_fin_solidity if solidity is None else solidity
    c_na_fin = _cl_alpha_gridfins(
        params.n_grid_fins, params.grid_fin_width_m, params.grid_fin_height_m,
        params.grid_fin_chord_m, params.grid_fin_web_thickness_m,
        params.grid_fin_cell_pitch_m, d, mach, solidity=sig)

    x_cg, total_len = estimate_cg(params)
    if x_cg_m is not None:
        x_cg = x_cg_m
    if fin_station_m is not None:
        x_fin = fin_station_m
    else:
        # grid fins sit near the aft of their stage; default to just forward of
        # the aft end by half the fin chord.
        x_fin = total_len - 0.5 * max(params.grid_fin_chord_m, 0.0)

    denom = c_na_body + c_na_fin
    x_cp = (c_na_body * x_body + c_na_fin * x_fin) / denom if denom > 0 else x_body
    sm_cal = (x_cp - x_cg) / d if d > 0 else float('nan')

    if sm_cal < 0:
        verdict = "UNSTABLE (CP ahead of CG)"
    elif sm_cal < _SM_MARGINAL_LO:
        verdict = "marginal / low stability"
    elif sm_cal <= _SM_TYPICAL_HI:
        verdict = "appropriate (typical 0.5-2 cal)"
    else:
        verdict = "over-stabilised (more fin than needed)"

    return dict(
        static_margin_cal=sm_cal,
        x_cp_m=x_cp, x_cg_m=x_cg, x_body_m=x_body, x_fin_m=x_fin,
        c_na_body=c_na_body, c_na_fin=c_na_fin,
        total_length_m=total_len, diameter_m=d, mach=mach, verdict=verdict,
        cg_estimated=(x_cg_m is None), fin_station_estimated=(fin_station_m is None),
    )


def grid_fin_area_for_margin(params: MissileParams, target_sm_cal: float = 1.5,
                             mach: float = 1.5, x_cg_m: float = None,
                             fin_station_m: float = None) -> dict:
    """Inverse: grid-fin frame area per fin needed to hit target_sm_cal.

    Because the grid-fin C_Na scales linearly with frame area in the model, the
    required area is the current area times the ratio of required to current
    C_Na.  Returns the per-fin frame area and the current value for comparison.
    """
    d = params.diameter_m
    cur = grid_fin_static_margin(params, mach=mach, x_cg_m=x_cg_m,
                                 fin_station_m=fin_station_m)
    c_na_body, x_body = cur['c_na_body'], cur['x_body_m']
    x_cg, x_fin = cur['x_cg_m'], cur['x_fin_m']

    x_cp_target = x_cg + target_sm_cal * d
    # C_Na,fin needed: C_Na,b*(x_cp - x_b) = C_Na,f*(x_f - x_cp)
    if x_fin <= x_cp_target:
        return dict(feasible=False,
                    reason="target CP is at/aft of the fin station -- "
                           "fins can't place it there; move target or station.",
                    **cur)
    c_na_fin_req = c_na_body * (x_cp_target - x_body) / (x_fin - x_cp_target)
    cur_area = params.grid_fin_width_m * params.grid_fin_height_m
    cur_c_na_fin = cur['c_na_fin']
    if cur_c_na_fin <= 0:
        return dict(feasible=False, reason="current fin C_Na is zero.", **cur)
    area_req = cur_area * (c_na_fin_req / cur_c_na_fin)
    return dict(
        feasible=True,
        target_sm_cal=target_sm_cal,
        frame_area_per_fin_m2_required=area_req,
        frame_area_per_fin_m2_current=cur_area,
        equivalent_square_side_m=math.sqrt(area_req) if area_req > 0 else 0.0,
        c_na_fin_required=c_na_fin_req, c_na_fin_current=cur_c_na_fin,
        current_static_margin_cal=cur['static_margin_cal'],
        diameter_m=d, mach=mach,
    )


if __name__ == "__main__":
    from missile_models import get_missile
    m = get_missile("STARS-1")
    print("=== STARS-1 grid-fin static-margin check (M=1.5, full/liftoff) ===")
    r = grid_fin_static_margin(m, mach=1.5)
    for k in ("static_margin_cal", "verdict", "x_cp_m", "x_cg_m", "x_body_m",
              "x_fin_m", "c_na_body", "c_na_fin", "total_length_m",
              "diameter_m", "cg_estimated", "fin_station_estimated"):
        v = r[k]
        print(f"  {k:24s}: {v:.3f}" if isinstance(v, float) else f"  {k:24s}: {v}")
    print("\n=== Inverse: frame area for SM = 1.5 cal ===")
    inv = grid_fin_area_for_margin(m, target_sm_cal=1.5, mach=1.5)
    for k, v in inv.items():
        print(f"  {k:32s}: {v:.4f}" if isinstance(v, float) else f"  {k:32s}: {v}")
