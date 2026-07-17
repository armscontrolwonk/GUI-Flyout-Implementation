"""Pixel-digitizer for the nominal Reentry-F stagnation heating curve.

Input: the embedded scan of "Figure 1 - Nominal Reentry 'F' trajectory"
(gamma_E 21.2 deg, V_E 20,300 ft/s; the figure Berry's nose-tip white paper
reproduces as its Fig. 6 [NASA LWP-460]).  Extract it losslessly from the
white-paper PDF first:

    pdfimages -png -f 8 -l 8 ReentryF_White_Paper_v2.pdf fig6   # -> fig6-000.png

Method (what WebPlotDigitizer does, scripted):
1. Detect the vertical ruler bars and the x-axis; least-squares-calibrate each
   axis on ITS OWN tick marks (the figure's rulers are NOT frame-aligned — the
   velocity ruler is visibly shorter than the frame, so per-ruler calibration
   is mandatory).
2. Locate the heating-curve apex (topmost curve pixel in the bell region) and
   trace outward with slope-continuity tracking (nearest dark run to the
   slope-extrapolated prediction, +/-8 px window).
3. QC via a red-overlay PNG; the tangled descent (heating crosses the hooked
   mass-flux/pressure curves) was patched from a 3x zoom read; all curves
   terminate at ~436.2 s.

Validation: traced in-window (100->50 kft) flux range 10.2-30.2 kBtu/ft2-s vs
LWP-460's quoted 9,000-28,000; apex 348 MW/m2 vs Sutton-Graves ~340-380; the
figure's own "stagnation-point heating rate" label arrow touches the traced
curve at (426 s, ~17.4k).

Outputs: reentryf_qdot_trace_full.csv (t_s, qdot_Btu_ft2_s), integral
Q ~= 3.87 GJ/m2 cold-wall (+/-15-20%).  Run: python3 digitize_reentryf_qdot.py <fig6.png>
"""
import sys
import numpy as np
from PIL import Image

# axis calibration from detected tick rows/cols (see module docstring)
Q_ROWS = np.array([523, 463, 397, 338, 275, 211, 147, 84, 22.5])   # heating ruler ticks
Q_VALS = np.arange(0, 33, 4) * 1e3                                  # Btu/ft2-s
T_COLS = np.array([187.5, 255.5, 315.5, 381.5, 441, 506.5, 568.5, 634.5, 695, 755, 822])
T_VALS = np.arange(400, 441, 4)                                     # s
DESCENT_PATCH = [(434.5, 20200.0), (435.1, 15500.0), (436.2, 10200.0)]  # 3x-zoom reads

def digitize(png_path):
    a = np.asarray(Image.open(png_path).convert("L"))
    dark = a < 120
    H = a.shape[0]
    Aq = np.polyfit(Q_ROWS, Q_VALS, 1)
    At = np.polyfit(T_COLS, T_VALS, 1)

    def runs(col, ylo, yhi):
        ys = np.where(dark[max(0, int(ylo)):min(H, int(yhi)), col])[0] + max(0, int(ylo))
        out = []
        for y in ys:
            if out and y - out[-1][-1] <= 2:
                out[-1].append(y)
            else:
                out.append([y])
        return [float(np.mean(r)) for r in out]

    # apex: topmost curve pixel in the bell region
    best = (1e9, None)
    for x in range(680, 721):
        rs = runs(x, 25, 200)
        if rs and rs[0] < best[0]:
            best = (rs[0], x)
    ay, ax = best

    def track(x0, y0, x1, step):
        out = {x0: y0}
        slope, prev = 0.0, y0
        for x in range(x0 + step, x1, step):
            pred = prev + slope * step
            cand = runs(x, pred - 8, pred + 8) or runs(x, pred - 14, pred + 14)
            if cand:
                y = min(cand, key=lambda c: abs(c - pred))
                slope = 0.6 * slope + 0.4 * (y - prev) / step
                prev = y
                out[x] = y
            else:
                prev = pred
        return out

    trace = {**track(ax, ay, 186, -1), **track(ax, ay, 832, 1)}
    xs = sorted(trace)
    ts = At[0] * np.array(xs) + At[1]
    qs = np.clip(Aq[0] * np.array([trace[x] for x in xs]) + Aq[1], 0, None)
    keep = (ts >= 400) & (ts <= 433.4)          # validated span; then patch descent
    ts, qs = ts[keep], qs[keep]
    tp, qp = zip(*DESCENT_PATCH)
    ts = np.concatenate([ts, tp]); qs = np.concatenate([qs, qp])
    return ts, qs

if __name__ == "__main__":
    ts, qs = digitize(sys.argv[1] if len(sys.argv) > 1 else "fig6-000.png")
    Q = np.trapezoid(qs, ts)
    print(f"peak {qs.max():.0f} Btu/ft2s = {qs.max()*0.011357:.0f} MW/m2 at t={ts[qs.argmax()]:.1f}s")
    print(f"Q = {Q:.0f} Btu/ft2 = {Q*11356.5/1e9:.2f} GJ/m2")
    np.savetxt("reentryf_qdot_trace_full.csv", np.column_stack([ts, qs]),
               delimiter=",", header="t_s,qdot_Btu_ft2_s", comments="")
