"""blender_export: the rough-draft 3-D export (bpy script generator).

Assertions follow the house rule — identities and measured anchors on the
generated geometry (dimensions equal the stored fields, stacking is
contiguous, solids close), plus a compile() of the emitted script (plain
Python; bpy only exists inside Blender)."""
import json
import math

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
    # every revolve profile starts and ends on the axis → closed solids
    for name, prof, _pos, _sw in els["revolves"]:
        assert prof[0][0] == 0.0 and prof[-1][0] == 0.0, name
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


def test_gui_export_handler_runs_end_to_end(tmp_path, monkeypatch):
    """Field bug: the menu handler crashed with NameError (filedialog not
    imported in thrusty's module scope — the codebase imports it locally
    per handler).  Drive the REAL handler with a dummy self and stubbed
    dialogs: it must write a compilable script to the chosen path."""
    import types
    import tkinter.filedialog as fd
    import tkinter.messagebox as mb
    import thrusty

    out = tmp_path / "veh_blender.py"
    monkeypatch.setattr(fd, "asksaveasfilename",
                        lambda **kw: str(out), raising=True)
    shown = []
    monkeypatch.setattr(mb, "showinfo",
                        lambda *a, **k: shown.append(a), raising=True)
    dummy = types.SimpleNamespace(_schem_params=_demo_vehicle(),
                                  _schem_name="Test Vehicle")
    thrusty.BoosterFlyoutApp._export_blender(dummy)
    text = out.read_text()
    compile(text, str(out), "exec")
    assert "'S1'" in text and "Test Vehicle" in text
    assert shown


def test_bpy_script_compiles_and_names_everything():
    """The emitted script is plain Python (bpy resolves inside Blender):
    it must compile, carry every object name and the collection, and list
    the fallback flags in its header."""
    veh = _demo_vehicle()
    script, info = bx.bpy_script(veh, title="Test Vehicle")
    compile(script, "<blender-export>", "exec")      # syntax-valid
    assert info["n_objects"] == len(bx.vehicle_elements(veh)["revolves"]) \
        + len(bx.vehicle_elements(veh)["plates"])
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
