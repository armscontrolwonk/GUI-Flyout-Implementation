"""diagram_render: the matplotlib/Agg diagram renderer (no Tk needed).

Pixel-sampling tests: the renderer returns a PIL image, so the assertions
sample actual pixels — the same output the GUI displays — instead of
introspecting canvas items."""
import pytest

pytest.importorskip("PIL")
pytest.importorskip("matplotlib")

import diagram_render as dr
import image_measure as im


def _px(img, fx, fy):
    """Sample at fractional (unit-square) coordinates."""
    w, h = img.size
    return img.getpixel((round(fx * (w - 1)), round(fy * (h - 1))))


def _near(px, hexcolor, tol=12):
    want = tuple(int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return all(abs(a - b) <= tol for a, b in zip(px, want))


def test_subject_fills_white_rest_grey_arrow_red():
    """The reference-art contract, in pixels: the SUBJECT element (here the
    right fin, for a wing prompt) fills white, the body fills grey, the
    background stays white, and the 1→2 measure art is red."""
    st = im.DIAGRAM_STYLE
    spec = im.diagram_spec(dict(field="_wing_root_var"))
    img = dr.render_prompt_diagram(spec, im.DIAGRAM_BASES["ro_side"],
                                   232, 160)
    assert img.size == (232, 160)
    assert _near(_px(img, 0.50, 0.70), st["fill"])       # body: grey
    assert _near(_px(img, 0.70, 0.82), st["highlight"])  # fin_r: white
    assert _near(_px(img, 0.10, 0.20), st["bg"])         # background
    # a length prompt's arrow runs down the axis in measure red
    lspec = im.diagram_spec(dict(field="_len_var"))
    limg = dr.render_prompt_diagram(lspec, im.DIAGRAM_BASES["ro_side"],
                                    232, 160)
    assert _near(_px(limg, 0.50, 0.50), st["measure"], tol=60)
    # click 2's small dot sits exactly on the click point
    assert _near(_px(limg, 0.50, 0.90), st["measure"], tol=60)


def test_angle_prompt_renders_and_differs_from_length():
    spec = im.diagram_spec(dict(field=im.FLANK_UPPER_FIELD, angle=True))
    img = dr.render_prompt_diagram(spec, im.DIAGRAM_BASES["ro_side"],
                                   232, 160)
    lspec = im.diagram_spec(dict(field="_len_var"))
    limg = dr.render_prompt_diagram(lspec, im.DIAGRAM_BASES["ro_side"],
                                    232, 160)
    assert img.tobytes() != limg.tobytes()


def test_shape_aware_curve_shows_in_pixels():
    """A Sears-Haack body's curved flank passes through a point where the
    straight cone's flank does not: dark linework there for lv_haack,
    plain background for cone."""
    spec = im.diagram_spec(dict(field="_len_var"))
    cone = dr.render_prompt_diagram(spec, im.ro_side_base("cone"), 232, 160)
    haack = dr.render_prompt_diagram(spec, im.ro_side_base("lv_haack"),
                                     232, 160)
    # at y=0.5 the haack flank sits at x≈0.399; the cone flank at x≈0.415
    p_cone, p_haack = _px(cone, 0.399, 0.50), _px(haack, 0.399, 0.50)
    assert sum(p_cone) / 3 > 245                     # cone: background here
    assert sum(p_haack) / 3 < 200                    # haack: dark linework
