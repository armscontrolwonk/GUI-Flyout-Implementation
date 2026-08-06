"""Render every image-measure prompt diagram to one contact-sheet PNG.

The review workflow for the diagram art: the little what-to-click diagrams
are drawn by the live dialog from image_measure's DIAGRAM data through the
DIAGRAM_STYLE tokens, so restyling is a data edit — and THIS script is how
a restyle (or a new feature's art) gets reviewed: it opens the real dialogs
headlessly, walks every prompt, screenshots each diagram off the real
canvas (no parallel drawing path that could drift), and tiles them.

Usage (needs a display; headless: xvfb-run -a python3 diagram_contact_sheet.py):
    python3 diagram_contact_sheet.py [out.png]

Covers: RO axisymmetric (Sears-Haack, so the shape-aware curve is visible),
RO biconic break geometry, RO wedge (plan view), and a maxed-out booster
(2 stages + interstage + conical + fairing + fins + strap-ons).
"""
import json
import os
import sys
import textwrap
import time

import tkinter as tk

import thrusty
import image_measure as im
from booster_models import ro_from_dict
from PIL import Image, ImageDraw, ImageGrab


def _grab_dialog(parent, prompts, shots, seen):
    """Open the measure dialog off `parent`, walk `prompts`, screenshot the
    diagram canvas for each new field."""
    opened = []
    orig = tk.Toplevel

    def _capture(*a, **k):
        w = orig(*a, **k)
        opened.append(w)
        return w
    tk.Toplevel = _capture
    try:
        parent._measure_from_image()
    finally:
        tk.Toplevel = orig
    d = opened[-1]
    d.geometry("+0+0")
    for p in prompts:
        if p["field"] in seen:
            continue
        seen.add(p["field"])
        d._im_prompt_var.set(f"{p['field']}  —  {p['label']}")
        d._im_on_prompt()
        for _ in range(3):
            d.update_idletasks(); d.update(); time.sleep(0.03)
        c = d._im_diag
        x, y = c.winfo_rootx(), c.winfo_rooty()
        w, h = c.winfo_width(), c.winfo_height()
        img = ImageGrab.grab(xdisplay=os.environ.get("DISPLAY"))
        shots.append((p["field"], p["label"], img.crop((x, y, x + w, y + h))))
    d.destroy()
    parent.destroy()


def build_sheet(out_path="diagram_contact_sheet.png"):
    root = tk.Tk(); root.withdraw()
    ro = ro_from_dict(json.load(open("ro_library/C-HGB.ro.json")))
    shots, seen = [], set()

    def ro_editor(shape=None, biconic=False, form=None):
        dlg = thrusty.ROEditorDialog(root, ro=ro)
        dlg.withdraw()
        if form in ("wedge", "half_cone"):
            dlg._shape_var.set(dlg._BODY_FORM_LABELS[form])
        elif biconic:
            dlg._shape_var.set(dlg._BICONIC_LABEL)
        elif shape:
            dlg._shape_var.set(thrusty.NOSE_SHAPE_LABELS[shape])
        dlg._update_body_form_state()
        return dlg

    _grab_dialog(ro_editor(shape="lv_haack"),
                 im.ro_prompts("axisymmetric")
                 + im.ro_angle_prompts("axisymmetric"), shots, seen)
    _grab_dialog(ro_editor(biconic=True),
                 [p for p in im.ro_prompts("axisymmetric", biconic=True)
                  if p["field"] in ("_fore_len_var", "_break_dia_var")],
                 shots, seen)
    _grab_dialog(ro_editor(form="wedge"), im.ro_prompts("wedge"),
                 shots, seen)

    booster = thrusty.BoosterDialog(root, on_save=lambda *a, **k: None)
    booster.withdraw()
    booster._n_stages_var.set("2"); booster._update_stage_frames()
    booster._shroud_var.set(True); booster._update_shroud_state()
    booster._fins_var.set(True); booster._update_fins_state()
    booster._fin_n_var.set("4")
    booster._n_boosters_var.set("2")
    fr = booster._stage_frames[0]
    fr._interstage_var.set(True); fr._on_interstage()
    fr._conical_var.set(True); fr._on_conical()
    _grab_dialog(booster,
                 im.booster_prompts(n_stages=2, has_fairing=True,
                                    has_fins=True, n_fins=4, n_strapons=2,
                                    interstage_stages=(1,),
                                    conical_stages=(1,)), shots, seen)

    cols, cap, pad = 5, 56, 9
    # cells at the live canvas's NATIVE size — no resampling, so the sheet
    # shows exactly the pixels the dialog shows
    cw, ch = shots[0][2].size
    rows = (len(shots) + cols - 1) // cols
    W = cols * (cw + pad) + pad
    H = rows * (ch + cap + pad) + pad + 28
    sheet = Image.new("RGB", (W, H), im.DIAGRAM_STYLE["bg"])
    dr = ImageDraw.Draw(sheet)
    dr.text((pad, 8), "Thrusty image-measure diagrams — every prompt, "
            "rendered by the live dialog through DIAGRAM_STYLE. "
            "Red = clicks, numbered in order.", fill="black")
    for i, (field, label, img) in enumerate(shots):
        cx = pad + (i % cols) * (cw + pad)
        cy = 26 + (i // cols) * (ch + cap + pad)
        sheet.paste(img, (cx, cy))
        dr.rectangle((cx, cy, cx + cw, cy + ch), outline="#bbb")
        text = f"{field}\n" + "\n".join(
            textwrap.wrap(label.split(" — ")[0], 34)[:2])
        dr.multiline_text((cx, cy + ch + 3), text, fill="#333", spacing=1)
    sheet.save(out_path)
    root.destroy()
    return out_path, len(shots)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "diagram_contact_sheet.png"
    path, n = build_sheet(out)
    print(f"{n} diagrams -> {path}")
