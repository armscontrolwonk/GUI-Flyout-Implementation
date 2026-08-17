"""blender_export: the rough-draft 3-D export (bpy script generator).

Assertions follow the house rule — identities and measured anchors on the
generated geometry (dimensions equal the stored fields, stacking is
contiguous, solids close), plus a compile() of the emitted script (plain
Python; bpy only exists inside Blender)."""
import json
import math
import os

import pytest

import blender_export as bx
from booster_models import BoosterParams, ro_from_dict


def _stage(name, dia, length, **kw):
    p = BoosterParams(name=name, mass_initial=1000.0, mass_propellant=800.0,
                      mass_final=200.0, diameter_m=dia, length_m=length,
                      thrust_N=1e5, burn_time_s=60.0, isp_s=250.0)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _demo_vehicle():
    """2 stages (S1 conical + interstage) + fairing + 4 fins + 2 strap-ons."""
    s2 = _stage("T Stage 2", 0.8, 3.0,
                shroud_length_m=2.0, shroud_diameter_m=1.0,
                shroud_nose_length_m=0.9, shroud_nose_shape="lv_haack")
    s1 = _stage("T", 1.2, 8.0,
                conical=True, top_diameter_m=0.9,
                has_interstage=True, interstage_length_m=0.5,
                stage2=s2,
                has_fins=True, n_fins=4, fin_span_m=0.6,
                fin_root_chord_m=1.0, fin_tip_chord_m=0.4,
                fin_sweep_deg=30.0, fin_thickness_m=0.03,
                n_boosters=2, booster_diam_m=0.5, booster_length_m=5.0)
    return s1


def test_elements_are_discrete_and_stack_contiguously():
    els = bx.vehicle_elements(_demo_vehicle())
    names = [n for n, *_ in els["revolves"]] + [n for n, *_ in els["plates"]]
    assert len(names) == len(set(names))            # every element discrete
    by = {n: (prof, pos, sw) for n, prof, pos, sw in els["revolves"]}
    assert {"S1", "Interstage_1", "S2", "Fairing",
            "Strapon_1", "Strapon_2"} <= set(by)
    assert {"Fin_1", "Fin_2", "Fin_3", "Fin_4"} <= set(names)
    # S1: true frustum — base r 0.6, top r 0.45, length 8
    prof, pos, _ = by["S1"]
    assert pos == (0.0, 0.0, 0.0)
    assert max(r for r, _ in prof) == pytest.approx(0.6)
    assert (0.45, 8.0) in prof                      # conical top
    # interstage sits ON S1's top, derived ⌀: S1 top 0.9 → S2 base 0.8
    prof, pos, _ = by["Interstage_1"]
    assert pos[2] == pytest.approx(8.0)
    assert (0.45, 0.0) in prof and (0.4, 0.5) in prof
    # S2 sits on the interstage; fairing on S2; total height closes
    assert by["S2"][1][2] == pytest.approx(8.5)
    assert by["Fairing"][1][2] == pytest.approx(11.5)
    assert els["total_height_m"] == pytest.approx(8.0 + 0.5 + 3.0 + 2.0)
    # every revolve profile has ≥2 points so it produces faces (closed
    # solids cap both ends, shells cap one, an interstage tube caps neither)
    for name, prof, _pos, _sw in els["revolves"]:
        assert len(prof) >= 2, name
    # strap-ons ring the core at equal angles, at the stored ⌀
    p1, p2 = by["Strapon_1"][1], by["Strapon_2"][1]
    r1 = math.hypot(p1[0], p1[1])
    assert r1 == pytest.approx(math.hypot(p2[0], p2[1]))
    assert r1 == pytest.approx(0.6 + 0.25 + 0.05)
    # fins: 4 plates at 90° spacing with the stored thickness
    rots = sorted(rot for n, _p, _t, _pos, rot in els["plates"]
                  if n.startswith("Fin_"))
    assert rots == pytest.approx([0.0, 90.0, 180.0, 270.0])
    assert all(t == pytest.approx(0.03) for n, _p, t, _pos, _r
               in els["plates"] if n.startswith("Fin_"))


def test_fairing_profile_is_a_real_haack_surface():
    """The Sears-Haack fairing nose revolves the true Haack curve: many
    profile points, monotone shrink to the axis, max radius = ⌀/2 — not a
    straight taper."""
    els = bx.vehicle_elements(_demo_vehicle())
    prof = next(p for n, p, _pos, _s in els["revolves"] if n == "Fairing")
    assert len(prof) > 20                            # a real curve
    assert max(r for r, _ in prof) == pytest.approx(0.5)
    zs = [z for _, z in prof]
    assert zs == sorted(zs)                          # base → tip
    assert prof[-1] == (0.0, pytest.approx(2.0))     # closes at length


def test_ro_shapes_are_real_3d():
    """RO beside the stack: a cone with a stored nose radius exports the
    TRUE sphere-cone (blunted apex shorter than the sharp cone, cap radius
    respected); a biconic carries its break radius; a wedge extrudes across
    its span; a half-cone is a half-revolve."""
    veh = _demo_vehicle()
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    veh.ro = ro                                      # cone, ⌀0.58×1.5, rn=.02
    els = bx.vehicle_elements(veh)
    prof, pos, sweep = next((p, q, s) for n, p, q, s in els["revolves"]
                            if n == "RO_Body")
    assert sweep == "full" and pos[0] > 0.6          # beside the stack
    assert max(r for r, _ in prof) == pytest.approx(0.29)
    # blunted apex height identity: L − rn/sin θ + rn (tangent-sphere
    # construction; the sphere sits deep on a slender cone)
    th = math.atan2(0.29, 1.5)
    assert max(z for _, z in prof) == pytest.approx(
        1.5 - 0.02 / math.sin(th) + 0.02)
    # biconic: break radius appears in the profile
    ro2 = ro_from_dict(dict(json.load(open("ro_library/C-HGB.ro.json")),
                            biconic=True, fore_length_m=0.6,
                            break_diameter_m=0.3))
    veh.ro = ro2
    els = bx.vehicle_elements(veh)
    prof = next(p for n, p, _q, _s in els["revolves"] if n == "RO_Body")
    assert (0.15, pytest.approx(0.9)) in [(r, z) for r, z in prof]
    # wedge: a prism whose extrusion is the stored span
    ro3 = ro_from_dict(dict(json.load(open("ro_library/C-HGB.ro.json")),
                            body_form="wedge", body_span_m=0.9))
    veh.ro = ro3
    els = bx.vehicle_elements(veh)
    plate = next((poly, t) for n, poly, t, _pos, _r in els["plates"]
                 if n == "RO_Body")
    assert plate[1] == pytest.approx(0.9)            # span = extrusion
    # half-cone: half revolve (closed by the deck in the emitted helper)
    ro4 = ro_from_dict(dict(json.load(open("ro_library/C-HGB.ro.json")),
                            body_form="half_cone"))
    veh.ro = ro4
    els = bx.vehicle_elements(veh)
    sweep = next(s for n, _p, _q, s in els["revolves"] if n == "RO_Body")
    assert sweep == "half"


def test_emitted_script_builds_valid_meshes_under_a_bpy_stub():
    """Execute the emitted script against a minimal bpy stub: every mesh
    must be created with in-range face indices (no dangling vertex refs),
    faces of ≥3 vertices, and one object per planned element — the closest
    thing to running Blender without Blender."""
    import sys
    import types

    made = []

    class _Mesh:
        def __init__(self, name):
            self.name = name

        def from_pydata(self, verts, edges, faces):
            assert len(verts) >= 3, self.name
            for f in faces:
                assert len(f) >= 3, self.name
                assert len(set(f)) == len(f), self.name   # no degenerate
                for i in f:
                    assert 0 <= i < len(verts), self.name
            made.append((self.name, len(verts), len(faces)))

        def update(self):
            pass

    bpy_stub = types.ModuleType("bpy")
    bpy_stub.data = types.SimpleNamespace(
        meshes=types.SimpleNamespace(new=_Mesh),
        collections=types.SimpleNamespace(
            new=lambda name: types.SimpleNamespace(
                name=name,
                objects=types.SimpleNamespace(link=lambda o: None))))
    bpy_stub.context = types.SimpleNamespace(
        scene=types.SimpleNamespace(
            collection=types.SimpleNamespace(
                children=types.SimpleNamespace(link=lambda c: None))))

    class _Obj:
        def __init__(self, name, mesh):
            self.name = name

    bpy_stub.data.objects = types.SimpleNamespace(new=_Obj)
    veh = _demo_vehicle()
    ro = ro_from_dict(dict(json.load(open("ro_library/C-HGB.ro.json")),
                           body_form="half_cone"))
    veh.ro = ro                        # exercise the half-revolve + deck
    script, info = bx.bpy_script(veh, title="StubRun")
    sys.modules["bpy"] = bpy_stub
    try:
        exec(compile(script, "<stub-run>", "exec"), {})
    finally:
        del sys.modules["bpy"]
    assert len(made) == info["n_objects"]
    assert all(nf > 0 for _n, _nv, nf in made)


def _parse_obj(text):
    """Minimal OBJ reader → {name: (verts, faces)}; faces as 0-indexed
    tuples resolved against the global vertex list (OBJ is 1-indexed)."""
    objs, verts, cur = {}, [], None
    for ln in text.splitlines():
        if ln.startswith("o "):
            cur = ln[2:].strip()
            objs[cur] = []
        elif ln.startswith("v "):
            verts.append(tuple(float(x) for x in ln.split()[1:4]))
        elif ln.startswith("f "):
            idx = [int(tok.split("/")[0]) - 1 for tok in ln.split()[1:]]
            objs[cur].append(idx)
    return objs, verts


def test_obj_export_is_valid_and_keeps_objects_discrete():
    """The direct-import path: a Wavefront OBJ Blender opens via File →
    Import.  Every element is its own `o` group, face indices are all in
    range (1-indexed, global), and the dimensions match the stored fields —
    S1 base ring radius = ⌀/2."""
    veh = _demo_vehicle()
    veh.ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    text, info = bx.obj_export(veh, title="Demo")
    objs, verts = _parse_obj(text)
    assert len(objs) == info["n_objects"]
    assert {"S1", "Interstage_1", "S2", "Fairing", "Fin_1", "Strapon_2",
            "RO_Body"} <= set(objs)
    for name, faces in objs.items():
        assert faces, name                          # every object has faces
        for f in faces:
            assert len(f) >= 3
            for i in f:
                assert 0 <= i < len(verts), name     # in range, no dangling
    # S1 base ring: max radius in XY equals the stored ⌀/2 = 0.6
    s1_faces = objs["S1"]
    s1_idx = {i for f in s1_faces for i in f}
    rmax = max(math.hypot(verts[i][0], verts[i][1]) for i in s1_idx)
    assert rmax == pytest.approx(0.6, abs=1e-3)


def test_thrusty_3d_figure_stands_beside_the_stack():
    """The mascot as FULL 3-D geometry (V-2-cutaway style): discrete
    named parts (body slab, googly eyes, thumbs-up fist, sneakers…),
    1.8 m top-of-slab over feet, feet on the stack-base ground line,
    parked CLEAR of every other object, palette colours bound per part
    (mtllib/usemtl — the pupils are black, the tongue red)."""
    veh = _demo_vehicle()
    veh.ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    text, info = bx.obj_export(veh, title="Demo")
    objs, verts = _parse_obj(text)
    fig_names = {n for n in objs if n.startswith("Thrusty_")}
    assert {"Thrusty_Body", "Thrusty_Eye_L", "Thrusty_Pupil_R",
            "Thrusty_Fist_L", "Thrusty_Thumb_L", "Thrusty_Hand_R",
            "Thrusty_Shoe_L", "Thrusty_Sole_R"} <= fig_names
    fig_idx = {i for n in fig_names for f in objs[n] for i in f}
    zs = [verts[i][2] for i in fig_idx]
    xs = [verts[i][0] for i in fig_idx]
    # feet on the ground line (stack base = -center_shift), slab top at
    # +1.8 m above it
    els = bx.vehicle_elements(veh)
    ground = -bx._center_shift(els)
    assert min(zs) == pytest.approx(ground, abs=1e-6)
    assert max(zs) - min(zs) == pytest.approx(bx._FIGURE_M, rel=0.02)
    # clear of everything already placed (fins, strap-ons, the RO)
    other = {i for nm, fs in objs.items() if nm not in fig_names
             for f in fs for i in f}
    ext = max(math.hypot(verts[i][0], verts[i][1]) for i in other)
    assert min(xs) > ext
    # palette plumbing: colours bound per part
    assert "mtllib" in text
    lines = text.splitlines()
    def mtl_of(name):
        i = lines.index(f"o {name}")
        return next(ln for ln in lines[i:] if ln.startswith("usemtl "))
    assert mtl_of("Thrusty_Body") == "usemtl Thrusty_white"
    assert mtl_of("Thrusty_Pupil_L") == "usemtl Thrusty_black"
    assert mtl_of("Thrusty_Tongue") == "usemtl Thrusty_red"
    assert mtl_of("Thrusty_Shoe_R") == "usemtl Thrusty_black"


def test_write_obj_bundle_ships_the_palette_sidecar(tmp_path):
    """The bundle is now a pair: <name>.obj + <name>.mtl carrying the
    figure's three flat palette colours.  No textures, no PNG."""
    path = str(tmp_path / "demo.obj")
    info = bx.write_obj_bundle(path, _demo_vehicle(), title="Demo")
    names = {os.path.basename(f) for f in info["files"]}
    assert names == {"demo.obj", "demo.mtl"}
    assert "mtllib demo.mtl" in open(path).read()
    mtl = open(tmp_path / "demo.mtl").read()
    for m in ("Thrusty_white", "Thrusty_black", "Thrusty_red"):
        assert f"newmtl {m}" in mtl
    assert "map_Kd" not in mtl                     # palette only


def test_obj_and_bpy_build_the_same_meshes():
    """The two export paths share one tessellation (revolve_mesh /
    plate_mesh), so the OBJ file and the in-Blender script produce the
    IDENTICAL solid — vertex counts per element match between the OBJ and
    a run of the emitted bpy script under the mesh-capturing stub."""
    import sys
    import types

    veh = _demo_vehicle()
    made = {}

    class _Mesh:
        def __init__(self, name):
            self.name = name

        def from_pydata(self, v, e, f):
            made[self.name] = (len(v), len(f))

        def update(self):
            pass

    stub = types.ModuleType("bpy")
    stub.data = types.SimpleNamespace(
        meshes=types.SimpleNamespace(new=_Mesh),
        objects=types.SimpleNamespace(new=lambda n, m: types.SimpleNamespace()),
        collections=types.SimpleNamespace(new=lambda n: types.SimpleNamespace(
            name=n, objects=types.SimpleNamespace(link=lambda o: None))))
    stub.context = types.SimpleNamespace(scene=types.SimpleNamespace(
        collection=types.SimpleNamespace(
            children=types.SimpleNamespace(link=lambda c: None))))
    script, _ = bx.bpy_script(veh, title="X")
    sys.modules["bpy"] = stub
    try:
        exec(compile(script, "<cmp>", "exec"), {})
    finally:
        del sys.modules["bpy"]
    obj_text, _ = bx.obj_export(veh, title="X")
    # per-object vertex counts straight from the OBJ text
    per_obj_v, cur = {}, None
    for ln in obj_text.splitlines():
        if ln.startswith("o "):
            cur = ln[2:].strip(); per_obj_v[cur] = 0
        elif ln.startswith("v "):
            per_obj_v[cur] += 1
    assert set(per_obj_v) == set(made)
    for name, (nv, _nf) in made.items():
        assert per_obj_v[name] == nv, name          # same vertex count


def test_gui_export_handler_runs_end_to_end(tmp_path, monkeypatch):
    """Field bug: the menu handler crashed with NameError (filedialog not
    imported in thrusty's module scope — the codebase imports it locally
    per handler).  Drive the REAL handler with a dummy self and stubbed
    dialogs: it must write a compilable script to the chosen path."""
    import types
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
    import thrusty

    shown = []
    monkeypatch.setattr(mb, "showinfo",
                        lambda *a, **k: shown.append(a), raising=True)
    dummy = types.SimpleNamespace(_schem_params=_demo_vehicle(),
                                  _schem_name="Test Vehicle")
    # default path: OBJ (the file Blender opens directly)
    obj_out = tmp_path / "veh.obj"
    monkeypatch.setattr(fd, "asksaveasfilename",
                        lambda **kw: str(obj_out), raising=True)
    thrusty.BoosterFlyoutApp._export_blender(dummy)
    otext = obj_out.read_text()
    assert otext.startswith("# Thrusty") and "\no S1\n" in otext
    assert "\nf " in otext                          # real faces, a mesh
    # .py path still available: emits the bpy script
    py_out = tmp_path / "veh.py"
    monkeypatch.setattr(fd, "asksaveasfilename",
                        lambda **kw: str(py_out), raising=True)
    thrusty.BoosterFlyoutApp._export_blender(dummy)
    ptext = py_out.read_text()
    compile(ptext, str(py_out), "exec")
    assert "import bpy" in ptext
    assert shown


def test_ro_wings_honor_the_fin_count():
    """Field bug: only 2 of a 4-fin C-HGB's wings exported (the count was
    hardcoded to a port/starboard pair).  The panels now come from the
    RO's stored n_fins — 4 flaps -> 4 wings, evenly clocked; n_fins=2
    stays a delta pair."""
    veh = _demo_vehicle()
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    ro.wing_root_chord_m = 0.4
    ro.wing_span_exposed_m = 0.2
    ro.wing_sweep_deg = 70.0
    ro.n_wings = 4
    veh.ro = ro
    wings = [(n, rot) for n, _poly, _t, _pos, rot
             in bx.vehicle_elements(veh)["plates"] if n.startswith("RO_Wing")]
    assert len(wings) == 4
    assert sorted(r for _n, r in wings) == pytest.approx([0.0, 90.0, 180.0,
                                                          270.0])
    ro.n_wings = 2
    wings2 = [n for n, *_ in bx.vehicle_elements(veh)["plates"]
              if n.startswith("RO_Wing")]
    assert len(wings2) == 2


def test_strapons_clock_into_the_gaps_between_fins():
    """Field bug (Strypi): strap-ons and booster fins both sat at k·(360/n),
    so 2 boosters landed on 2 of the 4 fins.  With fins present the strap-on
    ring is offset by half a fin spacing → boosters in the gaps.  No fins:
    no offset."""
    veh = _demo_vehicle()                        # 4 fins + 2 strap-ons
    els = bx.vehicle_elements(veh)
    fin_ang = sorted(rot for n, _p, _t, _pos, rot in els["plates"]
                     if n.startswith("Fin_"))
    strap_ang = sorted(
        (math.degrees(math.atan2(pos[1], pos[0])) % 360.0)
        for n, _pr, pos, _s in els["revolves"] if n.startswith("Strapon"))
    assert fin_ang == pytest.approx([0.0, 90.0, 180.0, 270.0])
    assert strap_ang == pytest.approx([45.0, 225.0])   # in the gaps
    # every strap-on is a half-gap (>= 45°/... ) clear of every fin
    for sa in strap_ang:
        assert min(min(abs(sa - fa), 360 - abs(sa - fa))
                   for fa in fin_ang) > 30.0
    # a stack with strap-ons but NO fins keeps the un-offset ring
    veh2 = _demo_vehicle()
    veh2.has_fins = False
    s2 = sorted((math.degrees(math.atan2(pos[1], pos[0])) % 360.0)
                for n, _pr, pos, _s in bx.vehicle_elements(veh2)["revolves"]
                if n.startswith("Strapon"))
    assert s2 == pytest.approx([0.0, 180.0])


def test_export_centers_booster_on_the_origin():
    """center=True (default) puts the booster's axis midpoint at z=0 — the
    stack spans −total/2 … +total/2 — so it drops into Blender centered.
    center=False keeps the base on z=0.  X=Y stay on the axis for the
    core."""
    veh = _demo_vehicle()
    total = bx.vehicle_elements(veh)["total_height_m"]

    cg_z = bx.vehicle_elements(veh)["cg_z"]
    assert 0.0 < cg_z < total

    def s1_base_z(**kw):
        """z of the S1 base ring — the stack's bottom reference, at z=0
        before centering.  Parse the OBJ, take S1's lowest vertex."""
        text, _ = bx.obj_export(veh, title="C", **kw)
        objs, verts = _parse_obj(text)
        s1 = {i for f in objs["S1"] for i in f}
        return min(verts[i][2] for i in s1)
    # S1 base sits at z=0 uncentered, and at −cg_z when centered on the CG
    assert s1_base_z(center=False) == pytest.approx(0.0, abs=1e-4)
    assert s1_base_z() == pytest.approx(-cg_z, abs=1e-4)
    # centering shifts EVERY vertex down by exactly cg_z (uniform), so the
    # CG lands on the origin
    zc = [v[2] for v in _parse_obj(bx.obj_export(veh, title="C")[0])[1]]
    z0 = [v[2] for v in _parse_obj(
        bx.obj_export(veh, title="C", center=False)[0])[1]]
    for a, b in zip(sorted(zc), sorted(z0)):
        assert a == pytest.approx(b - cg_z, abs=1e-4)
    assert "fuelled centre of gravity" in bx.obj_export(veh)[0]
    # the bpy script carries the same shift in its position literals
    assert f", {-cg_z:.6g})" in bx.bpy_script(veh, title="C")[0]


def test_ro_wing_thickness_from_stored_field():
    """The RO wing uses the stored wing_thickness_m (a real ROParams
    geometry field that JSON round-trips) — a modeler's entered thickness
    reaches the export; nominal only when it's unset."""
    veh = _demo_vehicle()
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    ro.wing_root_chord_m = 0.4
    ro.wing_span_exposed_m = 0.2
    ro.wing_thickness_m = 0.05
    veh.ro = ro
    t = next(t for n, _poly, t, _pos, _rot
             in bx.vehicle_elements(veh)["plates"] if n == "RO_Wing_1")
    assert t == pytest.approx(0.05)
    ro.wing_thickness_m = 0.0                    # unset → nominal + flag
    els = bx.vehicle_elements(veh)
    assert any("RO wing thickness unset" in fl for fl in els["flags"])
    # the geometry fields JSON round-trip
    from booster_models import ro_to_dict
    ro.wing_thickness_m = 0.05
    ro.n_wings = 6
    rt = ro_from_dict(ro_to_dict(ro))
    assert rt.wing_thickness_m == pytest.approx(0.05) and rt.n_wings == 6


def test_nozzles_drawn_per_stage_on_the_base():
    """Nozzles: n_nozzles flat disks on EACH stage's aft base (radius
    √(area_each/π)).  Per-nozzle drives the disk when set; a total-only
    stage (from Estimate) derives each = total/n.  Every stage's nozzles
    are drawn (upper ones hidden but SAVED) and named per stage so
    deleting stage 1 keeps the rest."""
    veh = _demo_vehicle()                        # 2 stages
    veh.n_nozzles = 4
    veh.nozzle_area_each_m2 = 0.05               # per-nozzle → disks
    veh.stage2.n_nozzles = 1
    veh.stage2.nozzle_exit_area_m2 = 0.3         # total only → derive each
    veh.stage2.nozzle_area_each_m2 = 0.0
    els = bx.vehicle_elements(veh)
    noz = {n: (prof, pos) for n, prof, pos, _s in els["revolves"]
           if "Nozzle" in n}
    assert {"S1_Nozzle_1", "S1_Nozzle_2", "S1_Nozzle_3", "S1_Nozzle_4",
            "S2_Nozzle_1"} == set(noz)
    # S1 disk radius = √(0.05/π); on a bolt circle (off-axis)
    r1 = noz["S1_Nozzle_1"][0][1][0]
    assert r1 == pytest.approx(math.sqrt(0.05 / math.pi), abs=1e-4)
    assert math.hypot(*noz["S1_Nozzle_1"][1][:2]) > 0    # ringed, not centered
    # S1 nozzles at the stage-1 base (z≈0), S2 up inside the stack
    assert noz["S1_Nozzle_1"][1][2] == pytest.approx(-0.005, abs=1e-6)
    assert noz["S2_Nozzle_1"][1][2] > 1.0
    # S2 single nozzle: radius from derived each = 0.3/1
    assert noz["S2_Nozzle_1"][0][1][0] == pytest.approx(
        math.sqrt(0.3 / math.pi), abs=1e-4)
    # the disk is a valid flat filled circle (fan)
    v, f = bx.revolve_mesh(noz["S1_Nozzle_1"][0], (0, 0, 0), 2 * math.pi)
    assert len(f) > 0 and all(len(face) == 3 for face in f)  # triangle fan


def test_nozzle_geometry_round_trips():
    """n_nozzles / nozzle_area_each_m2 survive the JSON round-trip."""
    from booster_models import booster_to_dict, booster_from_dict
    veh = _demo_vehicle()
    veh.n_nozzles = 4
    veh.nozzle_area_each_m2 = 0.05
    rt = booster_from_dict(booster_to_dict(veh))
    assert rt.n_nozzles == 4 and rt.nozzle_area_each_m2 == pytest.approx(0.05)


def test_nose_and_fairing_have_open_bases_stages_capped():
    """A nose/fairing is a SHELL sitting on the stack — its base is open
    (no cap), so its profile starts at the base RIM (r>0), while a stage is
    a closed solid whose profile starts at the axis (0,0) → base cap.  The
    RO beside the stack is a complete body and stays capped."""
    veh = _demo_vehicle()
    by = {n: prof for n, prof, _p, _s in bx.vehicle_elements(veh)["revolves"]}
    assert by["Fairing"][0][0] > 0.0          # open base rim
    assert by["S1"][0] == (0.0, 0.0)          # capped stage base
    assert by["S2"][0] == (0.0, 0.0)
    # the fairing still closes at its TIP (apex present)
    assert by["Fairing"][-1][0] == pytest.approx(0.0)
    # the interstage is a hollow tube — BOTH ends open (no apex either end)
    assert by["Interstage_1"][0][0] > 0.0 and by["Interstage_1"][-1][0] > 0.0
    # no-fairing vehicle: the payload nose is open-based too
    veh.stage2.shroud_length_m = 0.0
    by2 = {n: prof for n, prof, _p, _s
           in bx.vehicle_elements(veh)["revolves"]}
    assert by2["Payload_Nose"][0][0] > 0.0
    assert by2["Payload_Nose"][-1][0] == pytest.approx(0.0)   # closed tip
    # the RO body keeps its base cap (a complete body, not a shell)
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    veh.ro = ro
    by3 = {n: prof for n, prof, _p, _s
           in bx.vehicle_elements(veh)["revolves"]}
    assert by3["RO_Body"][0] == (0.0, 0.0)


def test_estimate_cg_uses_real_per_stage_lengths():
    """Regression: estimate_cg treated the root length_m as the WHOLE stack
    and squeezed a multi-stage vehicle, floating the CG up into stage 2.
    With the fix it lays stages out at their OWN lengths (+ interstages +
    nose), so a heavy fuelled first stage keeps the CG DOWN in stage 1, and
    the total matches the summed geometry."""
    from grid_fin_sizing import estimate_cg
    # 3 stages, heavy full first stage (TD-2-like proportions)
    s3 = _stage("S3", 0.6, 4.0)
    s3.mass_initial, s3.mass_propellant = 1600, 1400
    s2 = _stage("S2", 1.3, 8.0, stage2=s3,
                has_interstage=True, interstage_length_m=1.0)
    s2.mass_initial, s2.mass_propellant = 6500, 3700
    s1 = _stage("S1", 2.2, 16.0, stage2=s2,
                has_interstage=True, interstage_length_m=1.0)
    s1.mass_initial, s1.mass_propellant = 26400, 16100
    x_cg, total = estimate_cg(s1)
    assert total == pytest.approx(31.0, abs=1.0)      # 16+1+8+1+~nose
    h = total - x_cg                                  # height from base
    assert 8.0 < h < 16.0                             # inside stage 1
    assert h == pytest.approx(11.6, abs=0.6)


def _grid_fin_veh(sigma=None, web=0.0, pitch=0.0):
    p = _stage("S1", 1.37, 9.5)
    p.has_grid_fins = True
    p.n_grid_fins = 4
    p.grid_fin_height_m = 0.4     # radial span
    p.grid_fin_width_m = 0.5      # tangential width
    p.grid_fin_chord_m = 0.12     # streamwise depth (thin)
    if sigma is not None:
        p.grid_fin_solidity = sigma
    p.grid_fin_web_thickness_m = web
    p.grid_fin_cell_pitch_m = pitch
    return p


def test_grid_fins_face_the_flow_and_reflect_solidity():
    """A grid fin's broad face is perpendicular to the flow (streamwise
    depth = chord = the smallest dimension), and the mesh reflects solidity
    σ LITERALLY: σ→1 a solid panel (one box), σ→0 an empty mesh (no fin),
    between an open lattice (many webs).  Web+pitch derive σ from the real
    geometry (the STARS case)."""
    # σ = 1 → solid panel: one box per fin, and it faces the flow
    solid = bx.vehicle_elements(_grid_fin_veh(sigma=1.0))
    boxes = {n: v for n, v, _f in solid["meshes"] if n.startswith("GridFin")}
    assert len(boxes) == 4 and all(len(v) == 8 for v in boxes.values())
    zs = sorted({round(z, 4) for _x, _y, z in boxes["GridFin_1"]})
    assert len(zs) == 2 and zs[1] - zs[0] == pytest.approx(0.12)   # ⟂ flow
    # 0 < σ < 1 → an open lattice: many more verts (webs) than a box
    lat = bx.vehicle_elements(_grid_fin_veh(sigma=0.5))
    lv, lf = [(v, f) for n, v, f in lat["meshes"] if n == "GridFin_1"][0]
    assert len(lv) > 8 * 4                          # a real lattice
    # ...oriented at 45°: fin 1 sits at clock angle 0, so local u = x − R,
    # w = y; the webs' long edges run DIAGONALLY (|Δu| ≈ |Δw|), which the
    # old square lattice (axis-aligned edges) never had
    R = 1.37 / 2
    diag = False
    for face in lf:
        for k in range(len(face)):
            p, q = lv[face[k]], lv[face[(k + 1) % len(face)]]
            du, dw = q[0] - p[0], q[1] - p[1]
            if (math.hypot(du, dw) > 0.1
                    and abs(abs(du) - abs(dw)) < 0.02 * math.hypot(du, dw)):
                diag = True
    assert diag
    # σ = 0 (nothing set) → EMPTY: no grid-fin meshes at all
    empty = bx.vehicle_elements(_grid_fin_veh(sigma=0.0))
    assert not any(n.startswith("GridFin") for n, _v, _f in empty["meshes"])
    # web + pitch derive σ from geometry (STARS: 1 mm / 32 mm → ~0.06 lattice)
    geom = bx.vehicle_elements(_grid_fin_veh(sigma=0.0, web=0.001, pitch=0.032))
    assert any(n.startswith("GridFin") for n, _v, _f in geom["meshes"])
    assert any("σ=0.06" in f for f in geom["flags"])


def test_cg_marker_is_a_small_badge_on_the_surface():
    """The full-stack fuelled CG marker (classic symbol) is emitted as raw
    meshes — a small badge on the body SURFACE at the balance station,
    facing +X: all verts share x ≈ R_local (on the skin), lie in the Y-Z
    plane, straddle the CG station, and are small vs the body."""
    veh = _demo_vehicle()
    els = bx.vehicle_elements(veh)
    meshes = {n: (v, f) for n, v, f in els["meshes"]}
    assert {"CG_fuelled_ring", "CG_fuelled_fill1", "CG_fuelled_fill3"} \
        <= set(meshes)
    cg_z = els["cg_z"]
    # the badge is planar in Y-Z (all verts share one x = the surface station)
    xs = [x for _n, verts, _f in els["meshes"] for x, _y, _z in verts]
    assert max(xs) - min(xs) < 1e-6              # a flat badge, not a disk
    x0 = xs[0]
    # x0 sits on some stage's skin: within the stack's radii, not a giant disk
    R_max = max(0.5 * s.diameter_m for s in (veh, veh.stage2))
    assert 0.0 < x0 <= R_max + 0.05
    ring_z = [z for _x, _y, z in meshes["CG_fuelled_ring"][0]]
    assert min(ring_z) < cg_z < max(ring_z)      # straddles the station
    # small: the marker spans well under a body diameter
    ys = [y for _n, verts, _f in els["meshes"] for _x, y, _z in verts]
    assert max(ys) - min(ys) < R_max
    # it appears as OBJ objects and survives the bpy script
    obj = bx.obj_export(veh)[0]
    assert "\no CG_fuelled_ring\n" in obj
    script, info = bx.bpy_script(veh)
    compile(script, "<cg>", "exec")
    assert "MESHES = [" in script and "CG_fuelled_fill1" in script
    assert info["n_objects"] == (len(els["revolves"]) + len(els["plates"])
                                 + len(els["meshes"])
                                 + len(els["figures"]))


def test_bpy_script_compiles_and_names_everything():
    """The emitted script is plain Python (bpy resolves inside Blender):
    it must compile, carry every object name and the collection, and list
    the fallback flags in its header."""
    veh = _demo_vehicle()
    script, info = bx.bpy_script(veh, title="Test Vehicle")
    compile(script, "<blender-export>", "exec")      # syntax-valid
    _e = bx.vehicle_elements(veh)
    assert info["n_objects"] == (len(_e["revolves"]) + len(_e["plates"])
                                 + len(_e["meshes"])
                                 + len(_e["figures"]))
    for name in ("'S1'", "'Interstage_1'", "'S2'", "'Fairing'", "'Fin_4'",
                 "'Strapon_2'", "'Test Vehicle'"):
        assert name in script
    assert "import bpy" in script
    assert "def _revolve" in script and "def _plate" in script
    # a nominal fallback is declared, not silent (none in this vehicle's
    # dims except when we unset something):
    veh2 = _demo_vehicle()
    veh2.fin_thickness_m = 0.0
    script2, info2 = bx.bpy_script(veh2, title="T2")
    assert any("fin thickness" in fl for fl in info2["flags"])
    assert "fin thickness" in script2                # in the header comment
