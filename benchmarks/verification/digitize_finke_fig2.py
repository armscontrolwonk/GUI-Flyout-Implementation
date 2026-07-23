"""Pixel-trace Finke IDA P-2395 Fig. 2 (T_eq vs altitude) from the page scan.

Replaces an earlier eyeball read (±3-4 km on a steep curve → ±6-8% in T) with
a tick-calibrated programmatic trace (~sub-km), mirroring the Reentry-F
`digitize_reentryf_qdot.py` method.  Emits `finke_fig2_teq.csv` (alt_km, Teq_K)
and a QC overlay.  Re-run:  python benchmarks/verification/digitize_finke_fig2.py

Calibration (this 300-dpi render, page 15 of ADA231552; pixels):
  x-axis (altitude): AX_X=852 px = 0 km, 1801 px = 150 km  → 6.327 px/km
      cross-checked against the 100/200/300/400-kft arrow ticks.
  y-axis (T_eq):     AX_Y=2283 px = 0 K, 396 px = 6000 K    → 0.3145 px/K
      cross-checked against the 1000/2000/.../5000 K gridline ticks.
The trace seeds at the curve peak (topmost thick run in the 10-25 km window)
and walks both directions with a nearest-run tracker; frame lines are masked.
"""
import numpy as np
from PIL import Image
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCAN = os.path.join(HERE, "finke_p2395_fig2.png")   # cropped page scan (in repo)

# Calibration is in FULL-PAGE pixels; the committed scan is cropped by (dx,dy).
CROP_DX, CROP_DY = 700, 300
AX_X, AX_Y = 852.0 - CROP_DX, 2283.0 - CROP_DY
PX_PER_KM = (1801 - 852) / 150.0
PX_PER_K = (2283 - 396) / 6000.0
to_km = lambda px: (px - AX_X) / PX_PER_KM
to_K = lambda py: (AX_Y - py) / PX_PER_K


def trace():
    im = np.array(Image.open(SCAN).convert("L"))
    dark = im < 140
    H, W = dark.shape
    X0, X1, Y0, Y1 = int(AX_X + 4), W - 5, 5, int(AX_Y - 4)
    sub = dark[Y0:Y1, X0:X1].copy()
    # mask frame lines (very long runs)
    for r in np.where(sub.sum(1) / sub.shape[1] > 0.55)[0]:
        sub[max(r - 8, 0):r + 9, :] = False
    for c in np.where(sub.sum(0) / sub.shape[0] > 0.55)[0]:
        sub[:, max(c - 4, 0):c + 5] = False

    def runs(x, min_len=1):
        idx = np.where(sub[:, x - X0])[0]
        if idx.size == 0:
            return []
        out, s, p = [], idx[0], idx[0]
        for i in idx[1:]:
            if i - p > 2:
                if p - s + 1 >= min_len:
                    out.append((s + p) / 2 + Y0)
                s = i
            p = i
        if p - s + 1 >= min_len:
            out.append((s + p) / 2 + Y0)
        return out

    seed = None
    for x in range(int(AX_X + 10 * PX_PER_KM), int(AX_X + 25 * PX_PER_KM)):
        for r in runs(x, min_len=4):
            if seed is None or r < seed[1]:
                seed = (x, r)
    sx, sy = seed
    MAXJ = 28

    def walk(d):
        path, y, gaps = [], float(sy), 0
        rng = range(sx, X1) if d > 0 else range(sx, X0 - 1, -1)
        for x in rng:
            c = [r for r in runs(x) if abs(r - y) <= MAXJ]
            if c:
                y = min(c, key=lambda r: abs(r - y))
                path.append((x, y)); gaps = 0
            else:
                gaps += 1
                if gaps > 25:
                    break
        return path

    path = sorted(walk(-1)[::-1] + [(sx, float(sy))] + walk(+1))
    xs = np.array([p[0] for p in path]); ys = np.array([p[1] for p in path])
    return im, path, to_km(xs), to_K(ys)


def main():
    im, path, h, T = trace()
    order = np.argsort(h)
    np.savetxt(os.path.join(HERE, "finke_fig2_teq.csv"),
               np.column_stack([h[order], T[order]]),
               delimiter=",", header="alt_km,Teq_K", comments="", fmt="%.3f")
    rgb = np.stack([im] * 3, -1)
    for x, y in path:
        yy = int(round(y))
        rgb[max(yy - 2, 0):yy + 3, int(x), 0] = 255
        rgb[max(yy - 2, 0):yy + 3, int(x), 1:] = 0
    Image.fromarray(rgb).save(os.path.join(HERE, "finke_fig2_trace_qc.png"))
    for tgt in (20.0, 37.0, 60.0, 80.0):
        m = np.abs(h - tgt) < 0.5
        if m.sum():
            print(f"T_eq({tgt:4.0f} km) = {np.median(T[m]):5.0f} K")


if __name__ == "__main__":
    main()
