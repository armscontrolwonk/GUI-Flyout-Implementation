"""Generate docs/grid_fin_solidity_diagram.png — a labelled frontal view of a
grid (lattice) fin showing cell pitch p, web (wall) thickness t, the open
window (p-t), and the solidity identity sigma = 1 - ((p-t)/p)^2.

Run:  python docs/make_grid_fin_diagram.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = os.path.join(os.path.dirname(__file__), "grid_fin_solidity_diagram.png")
p, t, n = 1.0, 0.22, 3
L = n * p + t
fig, ax = plt.subplots(figsize=(7.2, 7.2))
ax.add_patch(Rectangle((0, 0), L, L, facecolor="#9aa7b4", edgecolor="black", lw=1.5))
win = p - t
for i in range(n):
    for j in range(n):
        ax.add_patch(Rectangle((t + i * p, t + j * p), win, win,
                               facecolor="white", edgecolor="none"))
cy = t + win / 2.0
c0x, c1x = t + win / 2.0, t + win / 2.0 + p
ax.add_patch(FancyArrowPatch((c0x, -0.30), (c1x, -0.30), arrowstyle='<|-|>',
             mutation_scale=16, color='crimson', lw=1.8))
for cx in (c0x, c1x):
    ax.plot([cx, cx], [-0.42, t + win], ls=':', color='crimson', lw=1.0)
    ax.plot(cx, cy, 'o', color='crimson', ms=4)
ax.text((c0x + c1x) / 2, -0.52, "pitch  p  (centre-to-centre)", ha='center',
        va='top', color='crimson', fontsize=13, weight='bold')
wall_x0 = t + win
wy = t + 2 * p + win / 2.0
ax.add_patch(FancyArrowPatch((wall_x0, wy), (wall_x0 + t, wy), arrowstyle='<|-|>',
             mutation_scale=13, color='navy', lw=1.8))
ax.annotate("web (wall)\nthickness  t", xy=(wall_x0 + t / 2, wy),
            xytext=(L + 0.15, wy), color='navy', fontsize=13, weight='bold',
            va='center', arrowprops=dict(arrowstyle='->', color='navy', lw=1.4))
ax.add_patch(FancyArrowPatch((t, t), (t + win, t), arrowstyle='<|-|>',
             mutation_scale=12, color='#1a7f37', lw=1.6))
ax.text(t + win / 2, t - 0.16, "open window  (p - t)", ha='center', va='top',
        color='#1a7f37', fontsize=11.5, weight='bold')
ax.set_title("Grid (lattice) fin - frontal view", fontsize=15, weight='bold', pad=14)
ax.text(L / 2, L + 0.55,
        r"solidity  $\sigma = 1 - \left(\dfrac{p-t}{p}\right)^2$"
        "      (blocked frontal fraction)", ha='center', va='bottom', fontsize=14)
ax.text(L / 2, L + 0.18,
        "grey = solid material (webs)   .   white = open flow-through cells",
        ha='center', va='bottom', fontsize=10.5, color="#444")
ax.set_xlim(-0.9, L + 1.9); ax.set_ylim(-0.95, L + 1.25)
ax.set_aspect('equal'); ax.axis('off'); fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches='tight')
print("wrote", OUT)


def make_classes(out=None):
    """grid_fin_solidity_classes.png — open / typical / dense lattices side by
    side, sized to the mid-range σ of each visual class, for eyeball
    classification from imagery."""
    import math
    out = out or os.path.join(os.path.dirname(__file__),
                              "grid_fin_solidity_classes.png")
    tp = lambda sig: 1 - math.sqrt(1 - sig)        # t/p giving target σ
    # Anchored to REAL published fins (σ computed from their cited geometry).
    # Real measured hardware is dominated by very open, thin-web lattices.
    classes = [
        ("open", "σ ~ 0.04", 0.043,
         "Miller-Washington / Kretzschmar\n0.371 in pitch, 0.008 in web  (measured)"),
        ("typical", "σ ~ 0.09-0.12", 0.10,
         "Abate GTCM free-flight\n~0.11 cal pitch, 0.005-0.007 cal web  (measured)"),
        ("dense (rare)", "σ ~ 0.22", 0.225,
         "Chen DREV, thick web\n0.175 D pitch, 0.021 D web  (CFD; densest sourced)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 5.0))
    n, p = 4, 1.0
    for ax, (name, rng, sig, cite) in zip(axes, classes):
        t = tp(sig) * p; L = n * p + t; win = p - t
        ax.add_patch(Rectangle((0, 0), L, L, facecolor="#9aa7b4",
                               edgecolor="black", lw=1.2))
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((t + i * p, t + j * p), win, win,
                                       facecolor="white", edgecolor="none"))
        ax.set_title(f"{name}\n{rng}", fontsize=13, weight='bold')
        ax.text(L / 2, -0.55, cite, ha='center', va='top', fontsize=8.5,
                color="#333")
        ax.text(L / 2, -1.45, f"(shown: σ ~ {sig:.2f}, t/p ~ {tp(sig):.2f})",
                ha='center', va='top', fontsize=8.5, color="#777")
        ax.set_xlim(-0.3, L + 0.3); ax.set_ylim(-2.0, L + 0.3)
        ax.set_aspect('equal'); ax.axis('off')
    fig.suptitle("Grid-fin solidity σ — real-fin reference classes "
                 "(frontal view; grey = solid webs, white = open cells)",
                 fontsize=13, weight='bold', y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=130, bbox_inches='tight')
    print("wrote", out)


if __name__ == "__main__":
    make_classes()
