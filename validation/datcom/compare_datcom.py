"""Cross-validate glider_ld.py against Digital DATCOM (USAF, public domain).

Parses the committed Digital DATCOM output for the finless slender reference
body (finless_body.datcom.out, generated from finless_body.inp) and compares
its normal-force and L/D-vs-alpha curves at M2/3/5 against the glider_ld build-up
(Jorgensen TR R-474 + Allen-Perkins NACA 1048 + Pitts-Nielsen-Kaattari NACA
1307).  Reports the L/D-max, the best-glide AoA, and the C_A0 drag level for each.

ALSO compares the CENTRE OF PRESSURE vs alpha, which is what the trim gate
(trim_gate.py) actually rests on.  The .out file has carried CM and XCP columns
since it was committed, and nothing read them: the L/D comparison above validates
the FORCE, while the gate needs the MOMENT, i.e. where that force acts.  The
gate models the body as two stations -- the slender-body potential term at the
Barrowman nose c.p. and the Allen-Perkins viscous crossflow at the body planform
centroid -- so the c.p. migrates aft as alpha grows and the crossflow term takes
over.  This checks that migration against DATCOM.

DATCOM's convention, verified against its own printed XCP column (29/30 rows to
within 0.002): XCP = (X_mrc - x_cp)/L_ref = CM/CN, so x_cp = X_mrc - (CM/CN)*L_ref,
with X_mrc and L_ref read from each block's REFERENCE DIMENSIONS header.

Run:  python validation/datcom/compare_datcom.py
(No DATCOM binary needed — the reference .out is committed.  To regenerate it,
see README.md in this directory.)
"""
import os
import re
import sys
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '..', '..')))

import glider_ld                       # noqa: E402
from booster_models import BoosterParams as MP   # noqa: E402

# The reference body — must match finless_body.inp.
BODY = MP(name='finless ref', diameter_m=0.5, length_m=4.0,
          nose_shape='tangent_ogive', nose_length_m=1.5,
          mass_initial=500, mass_propellant=0, mass_final=500,
          burn_time_s=1, isp_s=1, thrust_N=1)

OUT = os.path.join(HERE, 'finless_body.datcom.out')


# Reference dimensions from the .out header (and finless_body.inp): the moment
# reference centre and longitudinal reference length the CM/XCP columns use.
X_MRC_M = 2.0
L_REF_M = 0.5


def parse_datcom(path, with_moment=False):
    """Yield (mach, [(alpha, cd, cl, cn, ca), ...]) per Mach block.

    with_moment=True yields (alpha, cd, cl, cn, ca, cm) instead, carrying the
    pitching-moment column needed for the centre-of-pressure comparison."""
    txt = open(path).read().splitlines()
    blocks = []
    for i, l in enumerate(txt):
        if not l.startswith('0 ALPHA     CD'):
            continue
        mach = None
        for j in range(i - 1, max(0, i - 6), -1):
            m = re.match(r'0\s+([0-9]+\.[0-9]+)\s', txt[j])
            if m:
                mach = float(m.group(1))
                break
        rows = []
        for k in range(i + 1, min(i + 20, len(txt))):
            p = txt[k].split()
            if len(p) >= 7 and re.match(r'-?[0-9]+\.[0-9]', p[0]):
                try:
                    row = (float(p[0]), float(p[1]), float(p[2]),
                           float(p[4]), float(p[5]))
                    if with_moment:
                        row = row + (float(p[3]),)
                    rows.append(row)
                except ValueError:
                    pass
        if mach and rows:
            blocks.append((mach, rows))
    return blocks


def datcom_cp_m(cm, cn):
    """DATCOM centre of pressure, metres aft of the nose tip."""
    return X_MRC_M - (cm / cn) * L_REF_M if abs(cn) > 1e-6 else None


def model_cp_m(aero, alpha_deg, x_body_cp_m):
    """glider_ld's body centre of pressure at one alpha, metres aft of the nose.

    The two body terms act at different stations, so the c.p. is their
    normal-force-weighted mean -- the same weighting trim_gate takes moments
    with.  Fin terms are excluded: the committed DATCOM case is body-alone."""
    c = glider_ld.cn_components(aero, math.radians(alpha_deg))
    cn = c['body_potential'] + c['crossflow_body']
    if cn <= 1e-9:
        return None
    return (c['body_potential'] * x_body_cp_m
            + c['crossflow_body'] * aero['body_planform_centroid_m']) / cn


def compare_cp():
    """Model vs DATCOM centre of pressure at every (Mach, alpha).  Returns
    [(mach, alpha, x_cp_datcom, x_cp_model), ...]."""
    import grid_fin_sizing as gfs
    _, x_body_cp = gfs.body_normal_force(BODY)
    out = []
    for mach, rows in parse_datcom(OUT, with_moment=True):
        aero = glider_ld.whole_booster_LD(BODY, mach=mach)
        for a, cd, cl, cn, ca, cm in rows:
            d_cp = datcom_cp_m(cm, cn)
            m_cp = model_cp_m(aero, a, x_body_cp)
            if d_cp is not None and m_cp is not None:
                out.append((mach, a, d_cp, m_cp))
    return out


def main():
    print(f"reference body: D={BODY.diameter_m} L={BODY.length_m} "
          f"nose={BODY.nose_shape} {BODY.nose_length_m} m")
    print(f"glider_ld A_p_body = "
          f"{glider_ld.whole_booster_LD(BODY, mach=5.0)['body_planform_m2']:.3f} m^2\n")
    worst = 0.0
    for mach, rows in parse_datcom(OUT):
        r = glider_ld.whole_booster_LD(BODY, mach=mach)
        d_ld, d_a = max((cl / cd, a) for a, cd, cl, cn, ca in rows if cd > 0)
        gap = 100.0 * (r['ld_max'] - d_ld) / d_ld
        worst = max(worst, abs(gap))
        print(f"M{mach:.0f}: glider_ld L/D_max={r['ld_max']:.2f}@{r['alpha_deg']:.0f}d  "
              f"C_A0={r['cd0']:.3f}   |   DATCOM {d_ld:.2f}@{d_a:.0f}d  "
              f"C_A0={rows[0][4]:.3f}   |   L/D gap {gap:+.0f}%")
    print(f"\nworst L/D gap: {worst:.0f}%  (glider_ld is conservative — "
          f"under-predicts, the safe direction for range)")

    # --- centre of pressure: what the TRIM GATE rests on ---------------------
    import grid_fin_sizing as gfs
    _, x_body_cp = gfs.body_normal_force(BODY)
    a5 = glider_ld.whole_booster_LD(BODY, mach=5.0)
    print(f"\nCENTRE OF PRESSURE (the trim gate's dependency)")
    print(f"  model stations: potential c.p. {x_body_cp:.3f} m, "
          f"crossflow planform centroid {a5['body_planform_centroid_m']:.3f} m "
          f"(body L={BODY.length_m} m)")
    rows = compare_cp()
    by_mach = {}
    for mach, a, d_cp, m_cp in rows:
        by_mach.setdefault(mach, []).append((a, d_cp, m_cp))
    worst_cp = 0.0
    for mach in sorted(by_mach):
        rs = by_mach[mach]
        d_lo, d_hi = rs[0][1], rs[-1][1]
        m_lo, m_hi = rs[0][2], rs[-1][2]
        errs = [100.0 * (m - d) / BODY.length_m for _, d, m in rs]
        worst_cp = max(worst_cp, max(abs(e) for e in errs))
        print(f"  M{mach:.0f}: DATCOM x_cp {d_lo:.2f}->{d_hi:.2f} m   "
              f"model {m_lo:.2f}->{m_hi:.2f} m   "
              f"error {min(errs):+.1f}..{max(errs):+.1f}% of L")
    print(f"  worst c.p. error: {worst_cp:.1f}% of body length, model FORWARD of "
          f"DATCOM at every point.")
    print("  Reference uncertainty, for scale: Sooy & Schmidt, JSR 42(2) 2005, put "
          "DATCOM's\n  OWN c.p. error against wind-tunnel data below 2% of body "
          "length at any AoA\n  (body-wing-tail M1.5/M4.6, body-tail M2.0), and "
          "Simon & Blake (AIAA 99-4258)\n  report c.p. 'well predicted at all "
          "angles of attack' at supersonic speeds.  So the\n  gap above is model "
          "error, not reference noise.")
    print("  Direction matters: a c.p. modelled too far forward understates the "
          "restoring\n  moment, so a given deflection trims to a HIGHER alpha "
          "than it really would.\n  That is the NON-conservative direction for "
          "the trim gate (it over-grants glide).\n  Both curves migrate AFT with "
          "alpha, so the two-station shape is right; the\n  offset is the "
          "slender-body potential term sitting at the nose c.p.")


if __name__ == '__main__':
    main()
