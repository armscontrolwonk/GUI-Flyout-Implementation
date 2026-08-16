"""Blender export of the vehicle — a dimensionally correct 3-D rough draft
for modelers (TODO item 2, option (a): a generated bpy script).

Emits a self-contained **Blender Python script**: the modeler opens
Blender, loads it in the Scripting tab, and runs it — each vehicle element
lands as a DISCRETE, named, editable mesh object (S1, Interstage_1,
Fairing, Fin_1, Strapon_2, RO_Body, …) in its own collection, at true
metres, +Z up (the vehicle axis).

Real 3-D shapes, not extruded silhouettes:
- stages are true cylinders / truncated cones (conical stages), built as
  closed solids of revolution;
- interstages are true frustums with the same derived diameters the 2-D
  schematic uses (this stage's top → the next stage's base);
- noses and fairings revolve their real analytic profiles — tangent
  ogive, Von Kármán / Sears-Haack (Haack series), parabolic, blunt dome,
  straight cone — so a Sears-Haack fairing is a true Haack surface;
- an axisymmetric RO revolves its declared profile, with a true
  sphere-cone blend when a nose radius is stored on a cone;
- known non-revolved parts extrude: fins are swept plates placed around
  the body, the wedge RO is a prism across its span, the half-cone RO is
  a half-revolution closed by its flat deck.

Derive, don't invent: geometry comes from the SAME stored fields the 2-D
schematic draws, with the same fallbacks — and every fallback that stood
in for unset data is listed in the emitted script's header and returned
in flags, never silently guessed.  Quantities the model simply does not
store (fin/wing thickness, aerospike stalk ⌀) get thin nominal values,
each one flagged.

Pure module: no Tk, no bpy import here (bpy exists only inside Blender).
The emitted script is plain Python — tests compile() it."""

import math

from booster_schematic import stage_chain, _stage_top_diameter, fin_polygon

_PROFILE_N = 24        # points per curved nose profile


def _nose_profile(shape, R, L):
    """(r, z) nose profile from base (r=R, z=0) to tip (r=0, z=L) — the
    true analytic curve for the declared shape."""
    s = (shape or "cone").lower()
    n = _PROFILE_N
    if "karman" in s or "haack" in s:
        # Haack series, x measured from the tip: C=0 → Von Kármán (LD),
        # C=1/3 → LV-Haack (= Sears-Haack body nose in this code base).
        C = 1.0 / 3.0 if ("lv" in s or "sears" in s) else 0.0
        pts = []
        for i in range(n + 1):
            x = L * (1.0 - i / n)
            th = math.acos(max(-1.0, min(1.0, 1.0 - 2.0 * x / L)))
            r = R / math.sqrt(math.pi) * math.sqrt(
                max(0.0, th - math.sin(2.0 * th) / 2.0
                    + C * math.sin(th) ** 3))
            pts.append((r, L - x))
        return pts
    if "ogive" in s:
        rho = (R * R + L * L) / (2.0 * R)       # tangent-ogive radius
        return [(math.sqrt(max(0.0, rho * rho - (L - x) ** 2)) + R - rho,
                 L - x)
                for x in (L * (1.0 - i / n) for i in range(n + 1))]
    if "parabola" in s:
        # parabolic series K'=1: r = R(2ξ − ξ²), ξ measured from the tip
        return [(R * (2.0 * xi - xi * xi), L * (1.0 - xi))
                for xi in (1.0 - i / n for i in range(n + 1))]
    if "blunt" in s:
        return [(R * math.cos(math.pi / 2 * i / n),
                 L * math.sin(math.pi / 2 * i / n)) for i in range(n + 1)]
    return [(R, 0.0), (0.0, L)]                  # straight cone


def _sphere_cone_profile(R, L, rn):
    """Spherically-blunted cone, base (R, 0) → dome apex: straight flank to
    the sphere-cone tangency circle, then the true spherical cap.  Falls
    back to the sharp cone when rn is unset or oversize."""
    if rn <= 1e-9 or rn >= 0.9 * R or L <= rn:
        return [(R, 0.0), (0.0, L)]
    th = math.atan2(R, L)                        # cone half-angle
    zc = L - rn / math.sin(th)                   # sphere centre (on axis)
    pts = [(R, 0.0), (rn * math.cos(th), zc + rn * math.sin(th))]
    n = 10
    for i in range(1, n + 1):
        t = th + (math.pi / 2 - th) * i / n
        pts.append((rn * math.cos(t), zc + rn * math.sin(t)))
    return pts


def _f(v, default=0.0):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    return v


def vehicle_elements(p):
    """Geometry plan for the export: {'revolves': [(name, profile,
    (x, y, z), sweep)], 'plates': [(name, poly, thickness, (x, y, z),
    rot_z_deg)], 'flags': [...], 'total_height_m': float}.  Same stacking
    rules and fallbacks as booster_schematic.draw_booster."""
    stages = stage_chain(p)
    flags, revolves, plates = [], [], []
    z = 0.0
    finned = strap = grid_finned = None
    shroud_stage = next(
        (s for s in stages
         if (getattr(s, "shroud_length_m", 0.0) or 0.0) > 0.0), None)

    for i, s in enumerate(stages):
        d = _f(getattr(s, "diameter_m", 0.0)) or 0.6
        L = _f(getattr(s, "length_m", 0.0)) or 1.0
        if not _f(getattr(s, "diameter_m", 0.0)):
            flags.append(f"S{i+1} diameter unset — 0.6 m used")
        if not _f(getattr(s, "length_m", 0.0)):
            flags.append(f"S{i+1} length unset — 1 m used")
        d_top = _stage_top_diameter(s)
        revolves.append((f"S{i+1}",
                         [(0.0, 0.0), (d / 2, 0.0), (d_top / 2, L),
                          (0.0, L)],
                         (0.0, 0.0, z), "full"))
        if getattr(s, "has_fins", False) and _f(getattr(s, "fin_span_m", 0.0)) > 0:
            finned = (s, z)
        if getattr(s, "has_grid_fins", False) and int(getattr(s, "n_grid_fins", 0) or 0) > 0:
            grid_finned = (s, z)
        if int(getattr(s, "n_boosters", 0) or 0) > 0:
            strap = (s, z, L)
        z += L
        # a real adapter only when declared; diameters DERIVED, as in 2-D
        if getattr(s, "has_interstage", False) \
                and _f(getattr(s, "interstage_length_m", 0.0)) > 0:
            il = _f(s.interstage_length_m)
            nxt = stages[i + 1] if i + 1 < len(stages) else None
            d_is_top = (_f(getattr(nxt, "diameter_m", 0.0)) or d_top) if nxt \
                else d_top
            revolves.append((f"Interstage_{i+1}",
                             [(0.0, 0.0), (d_top / 2, 0.0),
                              (d_is_top / 2, il), (0.0, il)],
                             (0.0, 0.0, z), "full"))
            z += il

    top = stages[-1]
    top_surface_d = _stage_top_diameter(top)
    if shroud_stage is not None:
        sd = _f(getattr(shroud_stage, "shroud_diameter_m", 0.0)) \
            or top_surface_d
        sl = _f(getattr(shroud_stage, "shroud_length_m", 0.0)) or 2 * sd
        shape = getattr(shroud_stage, "shroud_nose_shape", "") or ""
        nose = _f(getattr(shroud_stage, "shroud_nose_length_m", 0.0))
        if not (0.0 < nose <= sl):
            nose = 0.45 * sl
            flags.append("fairing nose length unset — 0.45×length used")
        if not shape:
            flags.append("fairing nose shape unset — cone used")
        R = sd / 2.0
        cyl = sl - nose
        profile = [(0.0, 0.0)]
        if cyl > 0:
            profile += [(R, 0.0), (R, cyl)]
        profile += [(r, cyl + zz) for r, zz in
                    _nose_profile(shape or "cone", R, nose)]
        revolves.append(("Fairing", profile, (0.0, 0.0, z), "full"))
        nose_base_d = sd
        z += sl
    else:
        nd = top_surface_d or 1.0
        shape = getattr(top, "nose_shape", "") or ""
        nl = _f(getattr(top, "nose_length_m", 0.0))
        if nl <= 0.0:
            nl = 1.6 * nd
            flags.append("nose length unset — 1.6×⌀ used")
        if not shape:
            flags.append("nose shape unset — cone used")
        profile = [(0.0, 0.0)] + _nose_profile(shape or "cone", nd / 2, nl)
        revolves.append(("Payload_Nose", profile, (0.0, 0.0, z), "full"))
        nose_base_d = nd
        z += nl

    a_LD = _f(getattr(p, "aerospike_LD", 0.0))
    if a_LD > 0:
        L_spike = a_LD * nose_base_d
        r_stalk = 0.015 * nose_base_d
        flags.append("aerospike stalk ⌀ not stored — 0.03×⌀ nominal")
        revolves.append(("Aerospike",
                         [(0.0, 0.0), (r_stalk, 0.0), (r_stalk, L_spike),
                          (0.0, L_spike)],
                         (0.0, 0.0, z), "full"))
        a_dD = _f(getattr(p, "aerospike_dD", 0.0))
        if a_dD > 0:
            disk_d = a_dD * nose_base_d
            t_disk = 0.04 * nose_base_d
            revolves.append(("Aerodisk",
                             [(0.0, 0.0), (disk_d / 2, 0.0),
                              (disk_d / 2, t_disk), (0.0, t_disk)],
                             (0.0, 0.0, z + L_spike - t_disk), "full"))
        z += L_spike

    if finned:
        s, zb = finned
        R = _f(s.diameter_m) / 2.0
        span = _f(s.fin_span_m)
        root = _f(getattr(s, "fin_root_chord_m", 0.0)) or 0.8 * span
        tip = _f(getattr(s, "fin_tip_chord_m", 0.0)) or 0.4 * root
        sweep = _f(getattr(s, "fin_sweep_deg", 0.0))
        t = _f(getattr(s, "fin_thickness_m", 0.0))
        if t <= 0:
            t = max(0.01, 0.02 * span)
            flags.append("fin thickness unset — 0.02×span nominal")
        n_f = int(getattr(s, "n_fins", 0) or 4)
        poly = [(u, zz) for u, zz in
                fin_polygon(+1, R, 0.0, span, root, tip, sweep)]
        for k in range(n_f):
            plates.append((f"Fin_{k+1}", poly, t, (0.0, 0.0, zb),
                           360.0 * k / n_f))

    if grid_finned:
        s, zb = grid_finned
        R = _f(s.diameter_m) / 2.0
        gh = _f(getattr(s, "grid_fin_height_m", 0.0)) or 0.15 * 2 * R
        gc = _f(getattr(s, "grid_fin_chord_m", 0.0)) or gh
        n_g = int(s.n_grid_fins)
        t = max(0.01, 0.05 * gh)
        flags.append("grid fins exported as solid panels (lattice not "
                     "modeled) — thickness nominal")
        poly = [(R, 0.0), (R + gh, 0.0), (R + gh, gc), (R, gc)]
        for k in range(n_g):
            plates.append((f"GridFin_{k+1}", poly, t,
                           (0.0, 0.0, zb + 0.2), 360.0 * k / n_g))

    if strap:
        s, zb, Lc = strap
        n_b = int(s.n_boosters)
        bd = _f(getattr(s, "booster_diam_m", 0.0)) or 0.3
        bL = _f(getattr(s, "booster_length_m", 0.0))
        if bL <= 0:
            bL = min(0.45 * Lc, 18 * bd)
            flags.append("strap-on length unset — nominal used")
        cR = _f(s.diameter_m) / 2.0
        Rb = bd / 2.0
        c_dist = cR + Rb + 0.05
        profile = [(0.0, 0.0), (Rb, 0.0), (Rb, bL), (0.0, bL + 1.4 * bd)]
        for k in range(n_b):
            a = 2.0 * math.pi * k / n_b
            revolves.append((f"Strapon_{k+1}", profile,
                             (c_dist * math.cos(a), c_dist * math.sin(a),
                              zb), "full"))

    total = z
    ro = getattr(p, "ro", None)
    if ro is not None and _f(getattr(ro, "diameter_m", 0.0)) > 0:
        max_r = max((max(r for r, _ in prof) for _n, prof, _p, _s
                     in revolves), default=1.0)
        _ro_elements(ro, max_r + 1.0 + _f(ro.diameter_m),
                     revolves, plates, flags)

    return dict(revolves=revolves, plates=plates, flags=flags,
                total_height_m=total)


def _ro_elements(ro, x_off, revolves, plates, flags):
    """The reentry object beside the stack (base on z=0, like the 2-D
    schematic's corner drawing), from its own stored geometry."""
    D = _f(ro.diameter_m)
    L = _f(getattr(ro, "length_m", 0.0))
    if L <= 0:
        L = 1.6 * D
        flags.append("RO length unset — 1.6×⌀ used")
    rn = _f(getattr(ro, "nose_radius_m", 0.0))
    R = D / 2.0
    form = str(getattr(ro, "body_form", "") or "axisymmetric")
    if form not in ("wedge", "half_cone"):
        form = "axisymmetric"
    pos = (x_off, 0.0, 0.0)

    if form == "wedge":
        span = _f(getattr(ro, "body_span_m", 0.0))
        if span <= 0:
            span = D
            flags.append("wedge span unset — depth used as span (flagged)")
        h = D / 2.0                       # stored ⌀ IS the base depth
        plates.append(("RO_Body", [(-h, 0.0), (h, 0.0), (-h, L)],
                       span, pos, 0.0))
        return
    if form == "half_cone":
        # half-revolution of the cone, closed by its flat deck (the cut
        # is the diametral plane)
        revolves.append(("RO_Body", [(0.0, 0.0), (R, 0.0), (0.0, L)],
                        pos, "half"))
        return

    bic = None
    if getattr(ro, "biconic", False):
        Lf = _f(getattr(ro, "fore_length_m", 0.0))
        Dbrk = _f(getattr(ro, "break_diameter_m", 0.0))
        if 0 < Lf < L and 0 < Dbrk < D:
            bic = (Lf, Dbrk)
    shape = str(getattr(ro, "shape", "") or "cone").lower()
    if bic is not None:
        Lf, Dbrk = bic
        La, R1 = L - Lf, Dbrk / 2.0
        fore = _sphere_cone_profile(R1, Lf, min(rn, 0.9 * R1))
        profile = ([(0.0, 0.0), (R, 0.0), (R1, La)]
                   + [(r, La + zz) for r, zz in fore[1:]])
    elif "cone" in shape and "blunt" not in shape:
        profile = [(0.0, 0.0)] + _sphere_cone_profile(R, L, rn)
    else:
        if rn > 0:
            flags.append(f"RO nose radius not blended into '{shape}' "
                         "profile (drawn per the analytic shape)")
        profile = [(0.0, 0.0)] + _nose_profile(shape, R, L)
    revolves.append(("RO_Body", profile, pos, "full"))

    # faithful wing panels only from a stored planform (same rule as 2-D)
    w_rc = _f(getattr(ro, "wing_root_chord_m", 0.0))
    w_ss = _f(getattr(ro, "wing_span_exposed_m", 0.0))
    if w_rc > 0 and w_ss > 0:
        w_sw = _f(getattr(ro, "wing_sweep_deg", 0.0))

        def r_local(zz):
            if bic is not None:
                Lf, Dbrk = bic
                La, R1 = L - Lf, Dbrk / 2.0
                if zz <= La:
                    return R - zz * (R - R1) / La
                return max(0.0, R1 * (1.0 - (zz - La) / Lf))
            return max(0.0, R * (1.0 - zz / L))
        y_tip = min(max(w_rc - w_ss * math.tan(math.radians(w_sw)), 0.0),
                    w_rc)
        t = max(0.01, 0.02 * w_ss)
        flags.append("RO wing thickness not stored — 0.02×span nominal")
        poly = [(r_local(0.0), 0.0), (r_local(w_rc), w_rc),
                (r_local(0.0) + w_ss, y_tip), (r_local(0.0) + w_ss, 0.0)]
        plates.append(("RO_Wing_1", poly, t, pos, 0.0))
        plates.append(("RO_Wing_2", poly, t, pos, 180.0))


_SEG = 48                    # revolve segments (matches the emitted script)


def revolve_mesh(profile, pos, sweep_rad, seg=_SEG):
    """Tessellate an (r, z) profile revolved about local +Z at pos into
    (verts, faces) — 0-indexed.  The Python twin of the emitted script's
    _revolve, so the OBJ file and the in-Blender build are the same solid.
    r≈0 points become single apex vertices; a partial sweep (half-cone) is
    closed with a flat deck n-gon."""
    full = abs(sweep_rad - 2 * math.pi) < 1e-9
    S = seg if full else seg // 2
    cols = S if full else S + 1
    verts, rings = [], []
    for r, z in profile:
        if r < 1e-9:
            rings.append((len(verts),))
            verts.append((pos[0], pos[1], pos[2] + z))
        else:
            idx = []
            for k in range(cols):
                a = sweep_rad * k / S
                idx.append(len(verts))
                verts.append((pos[0] + r * math.cos(a),
                              pos[1] + r * math.sin(a), pos[2] + z))
            rings.append(tuple(idx))
    faces = []
    for j in range(len(rings) - 1):
        a, b = rings[j], rings[j + 1]
        if len(a) == 1 and len(b) == 1:
            continue
        for k in range(S):
            k2 = (k + 1) % S if full else k + 1
            if len(a) == 1:
                faces.append((a[0], b[k2], b[k]))
            elif len(b) == 1:
                faces.append((a[k], a[k2], b[0]))
            else:
                faces.append((a[k], a[k2], b[k2], b[k]))
    if not full:
        edge = [rr[0] for rr in rings] + [rr[-1] for rr in reversed(rings)]
        deck, seen = [], set()
        for i in edge:
            if i not in seen:
                seen.add(i)
                deck.append(i)
        if len(deck) >= 3:
            faces.append(tuple(deck))
    return verts, faces


def plate_mesh(poly, thickness, pos, rot_z_deg):
    """Tessellate a flat plate — 2-D polygon (u, z) extruded ±thickness/2 in
    v, rotated about +Z, placed at pos — into (verts, faces).  Twin of the
    emitted _plate."""
    t = thickness / 2.0
    a = math.radians(rot_z_deg)
    ca, sa = math.cos(a), math.sin(a)
    n = len(poly)
    verts = []
    for v in (-t, +t):
        for (u, z) in poly:
            verts.append((pos[0] + u * ca - v * sa,
                          pos[1] + u * sa + v * ca, pos[2] + z))
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((k, k2, n + k2, n + k))
    return verts, faces


def obj_export(p, title="vehicle"):
    """A Wavefront OBJ of the vehicle — the format Blender opens DIRECTLY
    (File → Import → Wavefront .obj), unlike the bpy script.  Each element
    is its own named `o` group, so stages/fairing/fins stay discrete and
    editable.  Metres, +Z up.  Returns (obj_text, info)."""
    els = vehicle_elements(p)
    lines = [f"# Thrusty rough-draft 3-D export — {title}",
             "# Wavefront OBJ.  Blender: File -> Import -> Wavefront (.obj).",
             "# Units: metres.  +Z up (vehicle axis).  One object per "
             "element."]
    for fl in els["flags"]:
        lines.append(f"# fallback: {fl}")
    base = 1                                   # OBJ vertices are 1-indexed
    for name, prof, pos, sweep in els["revolves"]:
        verts, faces = revolve_mesh(
            prof, pos, math.pi if sweep == "half" else 2 * math.pi)
        lines.append(f"o {name}")
        for x, y, z in verts:
            lines.append(f"v {x:.6g} {y:.6g} {z:.6g}")
        for f in faces:
            lines.append("f " + " ".join(str(i + base) for i in f))
        base += len(verts)
    for name, poly, t, pos, rot in els["plates"]:
        verts, faces = plate_mesh(poly, t, pos, rot)
        lines.append(f"o {name}")
        for x, y, z in verts:
            lines.append(f"v {x:.6g} {y:.6g} {z:.6g}")
        for f in faces:
            lines.append("f " + " ".join(str(i + base) for i in f))
        base += len(verts)
    n = len(els["revolves"]) + len(els["plates"])
    return "\n".join(lines) + "\n", dict(
        n_objects=n, flags=list(els["flags"]),
        total_height_m=els["total_height_m"])


def _fmt_pts(pts):
    return "[" + ", ".join(f"({r:.6g}, {zz:.6g})" for r, zz in pts) + "]"


def _fmt_pos(pos):
    return f"({pos[0]:.6g}, {pos[1]:.6g}, {pos[2]:.6g})"


def bpy_script(p, title="vehicle"):
    """The emitted Blender script (string) + a summary dict
    {'n_objects': int, 'flags': [...], 'total_height_m': float}."""
    import datetime
    els = vehicle_elements(p)
    coll = title.strip() or "vehicle"
    flag_lines = ("".join(f"#   - {fl}\n" for fl in els["flags"])
                  or "#   (none — every dimension came from stored data)\n")
    rev_lines = "".join(
        f"    ({name!r}, {_fmt_pts(prof)}, {_fmt_pos(pos)}, {sweep!r}),\n"
        for name, prof, pos, sweep in els["revolves"])
    plate_lines = "".join(
        f"    ({name!r}, {_fmt_pts(poly)}, {t:.6g}, {_fmt_pos(pos)}, "
        f"{rot:.6g}),\n"
        for name, poly, t, pos, rot in els["plates"])
    n = len(els["revolves"]) + len(els["plates"])
    header = (
        f"# Thrusty rough-draft 3-D export — {coll}\n"
        f"# Generated {datetime.date.today().isoformat()}.  Run inside "
        "Blender: Scripting tab -> Open -> Run Script.\n"
        "# Units: metres.  +Z is up (the vehicle axis).  Each element is a\n"
        f"# separate named, editable object in the {coll!r} collection.\n"
        "# Derive-don't-invent: only stored geometry is exported.\n"
        "# Fallbacks that stood in for unset data:\n" + flag_lines)
    body = '''
import bpy
import math

SEG = 48                     # revolve segments (mirror of blender_export._SEG)


def _mesh_obj(name, verts, faces, coll):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    coll.objects.link(obj)
    return obj


def _revolve(name, profile, pos, coll, sweep):
    """Solid of revolution of an (r, z) profile about local +Z at pos.
    Points with r ~ 0 become single apex vertices (caps close for free);
    a partial sweep (the half-cone) is closed with a flat deck n-gon."""
    full = abs(sweep - 2 * math.pi) < 1e-9
    S = SEG if full else SEG // 2
    cols = S if full else S + 1
    verts, rings = [], []
    for r, z in profile:
        if r < 1e-9:
            rings.append((len(verts),))
            verts.append((pos[0], pos[1], pos[2] + z))
        else:
            idx = []
            for k in range(cols):
                a = sweep * k / S
                idx.append(len(verts))
                verts.append((pos[0] + r * math.cos(a),
                              pos[1] + r * math.sin(a), pos[2] + z))
            rings.append(tuple(idx))
    faces = []
    for j in range(len(rings) - 1):
        a, b = rings[j], rings[j + 1]
        if len(a) == 1 and len(b) == 1:
            continue
        for k in range(S):
            k2 = (k + 1) % S if full else k + 1
            if len(a) == 1:
                faces.append((a[0], b[k2], b[k]))
            elif len(b) == 1:
                faces.append((a[k], a[k2], b[0]))
            else:
                faces.append((a[k], a[k2], b[k2], b[k]))
    if not full:                          # flat deck closes the half body
        edge = [rr[0] for rr in rings] + [rr[-1] for rr in reversed(rings)]
        deck, seen = [], set()
        for i in edge:
            if i not in seen:
                seen.add(i)
                deck.append(i)
        if len(deck) >= 3:
            faces.append(tuple(deck))
    return _mesh_obj(name, verts, faces, coll)


def _plate(name, poly, thickness, pos, rot_z_deg, coll):
    """A flat plate: 2-D polygon (u, z) extruded +-thickness/2 in v, then
    rotated about +Z and placed at pos.  u = outboard, z = up."""
    t = thickness / 2.0
    a = math.radians(rot_z_deg)
    ca, sa = math.cos(a), math.sin(a)
    n = len(poly)
    verts = []
    for v in (-t, +t):
        for (u, z) in poly:
            verts.append((pos[0] + u * ca - v * sa,
                          pos[1] + u * sa + v * ca, pos[2] + z))
    faces = [tuple(range(n - 1, -1, -1)), tuple(range(n, 2 * n))]
    for k in range(n):
        k2 = (k + 1) % n
        faces.append((k, k2, n + k2, n + k))
    return _mesh_obj(name, verts, faces, coll)


coll = bpy.data.collections.new(COLLECTION)
bpy.context.scene.collection.children.link(coll)
for _name, _profile, _pos, _sweep in REVOLVES:
    _revolve(_name, _profile, _pos, coll,
             sweep=(math.pi if _sweep == "half" else 2 * math.pi))
for _name, _poly, _t, _pos, _rot in PLATES:
    _plate(_name, _poly, _t, _pos, _rot, coll)
print("Thrusty export: %d objects in %r"
      % (len(REVOLVES) + len(PLATES), COLLECTION))
'''
    data = (f"\nCOLLECTION = {coll!r}\n\n"
            f"REVOLVES = [\n{rev_lines}]\n\n"
            f"PLATES = [\n{plate_lines}]\n")
    script = header + data + body
    return script, dict(n_objects=n, flags=list(els["flags"]),
                        total_height_m=els["total_height_m"])
