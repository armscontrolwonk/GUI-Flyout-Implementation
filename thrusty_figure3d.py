"""Procedural full-3-D Thrusty — the mascot as real geometry for the
Blender export, modeled to the turnaround sheet (front / 3-4 front /
side / back, 2026-08-17): a gently S-curved ruler slab with a stepped
lightning-bolt silhouette, 3-D googly eyes proud of the face, tube arms
(thumbs-up left, hand-on-hip right), striped socks, and chunky sneakers.

Pure geometry + a three-colour palette (white / black / red) — no texture
sidecars, so the figure survives any file shuffle and edits like any
other mesh.  Every part is its own named mesh so a modeler can pose him.

Coordinates: feet on z = 0, FACING -Y — Blender's front view (Numpad 1
looks from -Y) then shows his face, thumbs-up on screen-left exactly
like the sheet — centred on x = 0; `height_m` scales the whole figure
(the export uses 1.8 m — the schematic's scale-figure height).  Returns
[(name, verts, faces, colour)] with colour in {"white", "black", "red"}.
"""

import math

_SEG = 16          # tube / sphere segments (rough-draft resolution)


# ── mesh primitives ─────────────────────────────────────────────────────────
def _spheroid(c, rx, ry, rz, seg=_SEG):
    """Lat-long ellipsoid mesh centred at c."""
    verts, faces = [], []
    rings = []
    n_lat = max(4, seg // 2)
    for i in range(n_lat + 1):
        th = math.pi * i / n_lat            # 0 (top) → pi (bottom)
        z = math.cos(th)
        r = math.sin(th)
        if i in (0, n_lat):
            rings.append((len(verts),))
            verts.append((c[0], c[1], c[2] + rz * z))
            continue
        idx = []
        for k in range(seg):
            a = 2 * math.pi * k / seg
            idx.append(len(verts))
            verts.append((c[0] + rx * r * math.cos(a),
                          c[1] + ry * r * math.sin(a),
                          c[2] + rz * z))
        rings.append(tuple(idx))
    for j in range(len(rings) - 1):
        a, b = rings[j], rings[j + 1]
        for k in range(seg):
            k2 = (k + 1) % seg
            if len(a) == 1:
                faces.append((a[0], b[k2], b[k]))
            elif len(b) == 1:
                faces.append((a[k], a[k2], b[0]))
            else:
                faces.append((a[k], a[k2], b[k2], b[k]))
    return verts, faces


def _sphere(c, r, seg=_SEG):
    return _spheroid(c, r, r, r, seg)


def _tube(p0, p1, r, seg=_SEG):
    """Capped cylinder from p0 to p1 (flat n-gon caps)."""
    ax = tuple(b - a for a, b in zip(p0, p1))
    L = math.sqrt(sum(v * v for v in ax)) or 1e-9
    ax = tuple(v / L for v in ax)
    ref = (0.0, 0.0, 1.0) if abs(ax[2]) < 0.9 else (1.0, 0.0, 0.0)
    u = (ax[1] * ref[2] - ax[2] * ref[1],
         ax[2] * ref[0] - ax[0] * ref[2],
         ax[0] * ref[1] - ax[1] * ref[0])
    ul = math.sqrt(sum(v * v for v in u)) or 1e-9
    u = tuple(v / ul for v in u)
    w = (ax[1] * u[2] - ax[2] * u[1],
         ax[2] * u[0] - ax[0] * u[2],
         ax[0] * u[1] - ax[1] * u[0])
    verts, faces = [], []
    for base in (p0, p1):
        for k in range(seg):
            a = 2 * math.pi * k / seg
            verts.append(tuple(base[i] + r * (math.cos(a) * u[i]
                                              + math.sin(a) * w[i])
                               for i in range(3)))
    for k in range(seg):
        k2 = (k + 1) % seg
        faces.append((k, k2, seg + k2, seg + k))
    faces.append(tuple(range(seg - 1, -1, -1)))          # bottom cap
    faces.append(tuple(range(seg, 2 * seg)))             # top cap
    return verts, faces


def _ear_clip(poly):
    """Triangulate a simple (non-self-intersecting) polygon, CCW winding.
    Standard ear clipping — plenty for the ~20-point slab outline."""
    def cross(o, a, b):
        return ((a[0] - o[0]) * (b[1] - o[1])
                - (a[1] - o[1]) * (b[0] - o[0]))
    idx = list(range(len(poly)))
    area = sum(cross((0, 0), poly[i], poly[(i + 1) % len(poly)])
               for i in range(len(poly)))
    if area < 0:                                  # enforce CCW
        idx.reverse()
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        n = len(idx)
        for ii in range(n):
            i0, i1, i2 = idx[ii - 1], idx[ii], idx[(ii + 1) % n]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-12:           # reflex or degenerate
                continue
            ear = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = poly[j]
                if (cross(a, b, p) >= -1e-12 and cross(b, c, p) >= -1e-12
                        and cross(c, a, p) >= -1e-12):
                    ear = False
                    break
            if ear:
                tris.append((i0, i1, i2))
                idx.pop(ii)
                break
        else:
            break                                  # no ear found — bail
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


# ── the character ───────────────────────────────────────────────────────────
# Slab silhouette in the x-z front view (units of total height H, feet at
# z=0): a stepped double-ruler "bolt" — left edge bows out at the belly,
# stepped top (main ruler + the signed second ruler), stepped bottom.
_SLAB = [
    (-0.160, 0.170), (-0.030, 0.170), (-0.030, 0.140), (0.100, 0.140),
    (0.140, 0.400), (0.130, 0.600), (0.170, 0.880), (0.040, 0.900),
    (0.040, 0.980), (-0.080, 1.000), (-0.100, 0.920), (-0.160, 0.700),
    (-0.190, 0.450),
]
_SLAB_T = 0.090                       # slab thickness (side view)
_Z_LO, _Z_HI = 0.140, 1.000           # slab extent used by the bow curve


def _yoff(z):
    """The side view's gentle S: belly forward mid-slab, top leaning back
    (units of H)."""
    u = min(1.0, max(0.0, (z - _Z_LO) / (_Z_HI - _Z_LO)))
    return 0.030 * math.sin(math.pi * u) - 0.028 * u


def _slab_mesh():
    """The curved ruler-slab body: front/back faces are the triangulated
    silhouette, joined by a side band, each vertex bowed by _yoff(z)."""
    tris = _ear_clip(_SLAB)
    n = len(_SLAB)
    verts = []
    for side in (+1, -1):
        for (x, z) in _SLAB:
            verts.append((x, _yoff(z) + side * _SLAB_T / 2.0, z))
    faces = [tuple(t) for t in tris]                       # front (+y)
    faces += [tuple(n + i for i in reversed(t)) for t in tris]   # back
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((k, k2, n + k2, n + k))               # side band
    return verts, faces


def build(height_m=1.8, x0=0.0, y0=0.0):
    """The full figure at `height_m`, feet on z=0, centred on (x0, y0),
    facing +Y.  Returns [(name, verts, faces, colour)]."""
    H = float(height_m)
    parts = []

    def add(name, mesh, colour):
        # y is NEGATED: the part geometry below is authored facing +Y;
        # the shipped figure faces -Y (see module docstring).  A mirror
        # reverses orientation, so each face's winding is flipped too —
        # normals keep pointing outward.
        verts, faces = mesh
        parts.append((name, [(x0 + x * H, y0 - y * H, z * H)
                             for x, y, z in verts],
                      [tuple(reversed(f)) for f in faces], colour))

    def at(x, z, dy=0.0):
        """A point on/near the slab mid-plane at height z."""
        return (x, _yoff(z) + dy, z)

    add("Thrusty_Body", _slab_mesh(), "white")

    # googly eyes: overlapping spheres proud of the front face, black
    # pupils at their front
    fy = _SLAB_T / 2.0
    for tag, ex, ez, er in (("L", -0.050, 0.870, 0.065),
                            ("R", 0.045, 0.880, 0.058)):
        add(f"Thrusty_Eye_{tag}",
            _sphere(at(ex, ez, fy + 0.30 * er), er), "white")
        add(f"Thrusty_Pupil_{tag}",
            _sphere(at(ex, ez - 0.012, fy + 1.12 * er), 0.020), "black")
    # open smile: black mouth with a red tongue, embedded in the face
    add("Thrusty_Mouth",
        _spheroid(at(0.010, 0.760, fy + 0.010), 0.052, 0.028, 0.036),
        "black")
    add("Thrusty_Tongue",
        _spheroid(at(0.012, 0.742, fy + 0.022), 0.032, 0.024, 0.016),
        "red")

    # arms: white tubes with sphere joints.  Viewer-left gives the
    # thumbs-up; viewer-right rests on the hip (turnaround pose).
    ra = 0.024
    l_sh, l_el, l_wr = at(-0.165, 0.560), at(-0.275, 0.490), at(-0.310, 0.630)
    add("Thrusty_Arm_L_Upper", _tube(l_sh, l_el, ra), "white")
    add("Thrusty_Arm_L_Fore", _tube(l_el, l_wr, ra), "white")
    add("Thrusty_Elbow_L", _sphere(l_el, ra * 1.15), "white")
    fist = at(-0.320, 0.680)
    add("Thrusty_Fist_L", _sphere(fist, 0.050), "white")
    add("Thrusty_Thumb_L",
        _tube(fist, (fist[0] - 0.012, fist[1], fist[2] + 0.085), 0.021),
        "white")
    add("Thrusty_ThumbTip_L",
        _sphere((fist[0] - 0.012, fist[1], fist[2] + 0.085), 0.023), "white")
    r_sh, r_el, r_wr = at(0.140, 0.620), at(0.280, 0.510), at(0.185, 0.415)
    add("Thrusty_Arm_R_Upper", _tube(r_sh, r_el, ra), "white")
    add("Thrusty_Arm_R_Fore", _tube(r_el, r_wr, ra), "white")
    add("Thrusty_Elbow_R", _sphere(r_el, ra * 1.15), "white")
    add("Thrusty_Hand_R", _sphere(at(0.170, 0.405), 0.042), "white")

    # legs + striped socks + sneakers (both feet splayed a touch outward,
    # toes forward like the sheet)
    rl = 0.026
    for tag, lx, splay in (("L", -0.078, -14.0), ("R", 0.062, 14.0)):
        top = at(lx, _Z_LO + 0.005)
        ankle = (lx, 0.030, 0.045)
        add(f"Thrusty_Leg_{tag}", _tube(top, ankle, rl), "white")
        for i, zs in enumerate((0.095, 0.125)):
            t_ = (top[2] - zs) / (top[2] - ankle[2] or 1e-9)
            c = tuple(top[j] + (ankle[j] - top[j]) * t_ for j in range(3))
            add(f"Thrusty_Sock_{tag}{i + 1}",
                _tube((c[0], c[1], c[2] - 0.008),
                      (c[0], c[1], c[2] + 0.008), rl * 1.18), "black")
        sa = math.radians(splay)
        fwd = (math.sin(sa), math.cos(sa))
        sc = (lx + 0.045 * fwd[0], 0.040 + 0.045 * fwd[1], 0.0)
        add(f"Thrusty_Sole_{tag}",
            _spheroid((sc[0], sc[1], 0.020), 0.058, 0.115, 0.022), "white")
        add(f"Thrusty_Shoe_{tag}",
            _spheroid((sc[0] - 0.015 * fwd[0], sc[1] - 0.015 * fwd[1],
                       0.048), 0.050, 0.092, 0.038), "black")
        add(f"Thrusty_Toe_{tag}",
            _spheroid((sc[0] + 0.095 * fwd[0], sc[1] + 0.095 * fwd[1],
                       0.030), 0.038, 0.034, 0.028), "white")
    return parts


def bounds(parts):
    """(min, max) corner tuples over every part's vertices."""
    xs = [v[0] for _n, vs, _f, _c in parts for v in vs]
    ys = [v[1] for _n, vs, _f, _c in parts for v in vs]
    zs = [v[2] for _n, vs, _f, _c in parts for v in vs]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))
