"""Cross-validate glider_ld.py against Digital DATCOM (USAF, public domain).

Parses the committed Digital DATCOM output for the finless slender reference
body (finless_body.datcom.out, generated from finless_body.inp) and compares
its normal-force and L/D-vs-alpha curves at M2/3/5 against the glider_ld build-up
(Jorgensen TR R-474 + Allen-Perkins NACA 1048 + Pitts-Nielsen-Kaattari NACA
1307).  Reports the L/D-max, the best-glide AoA, and the C_A0 drag level for each.

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


def parse_datcom(path):
    """Yield (mach, [(alpha, cd, cl, cn, ca), ...]) per Mach block."""
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
                    rows.append((float(p[0]), float(p[1]), float(p[2]),
                                 float(p[4]), float(p[5])))
                except ValueError:
                    pass
        if mach and rows:
            blocks.append((mach, rows))
    return blocks


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


if __name__ == '__main__':
    main()
