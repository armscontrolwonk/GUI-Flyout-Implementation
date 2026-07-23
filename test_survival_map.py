"""Survival map — the one-glance station × question matrix between the lead
and the full analysis.

Rows are stations in the order the heat visits them (nose → body skin →
windward flank → interior); columns are the three ladder questions (surface
holds / endures duration / within flown record).  Each populated cell shows
the one number its tier was decided on; spans returned to the GUI colorize
cells in place.  These tests pin the block's placement, the per-material cell
logic, and that every span lands exactly on its cell text.
"""

import survivability_report as sr
from test_report_lead import _fly_ballistic
from test_thresholds import _fly_glider

_CACHE = {}


def _rep(key):
    if key not in _CACHE:
        if key == "mk21":
            _CACHE[key] = sr.build_report(_fly_ballistic("Mk21", "Generic ICBM"))
        elif key == "glider":
            _CACHE[key] = sr.build_report(_fly_glider())
    return _CACHE[key]


def _map_block(body):
    lead = body.split("═══ Full analysis")[0]
    assert "─── Survival map" in lead        # sits above the divider
    return lead[lead.index("─── Survival map"):].rstrip("\n").split("\n")


# ── Placement + row roster ───────────────────────────────────────────────────
def test_map_between_lead_and_divider():
    body = _rep("mk21")["body"]
    i_map = body.index("─── Survival map")
    assert body.index("What would change" if "What would change" in body
                      else "\n\n") < i_map < body.index("═══ Full analysis")


def test_mk21_rows_and_cells():
    lines = _map_block(_rep("mk21")["body"])
    txt = "\n".join(lines)
    # header row carries the three ladder questions
    assert "Surface holds?" in lines[1]
    assert "Endures duration?" in lines[1]
    assert "Within flown record?" in lines[1]
    # ablator nose: burn-through bound + load vs the family flight record
    nose = next(l for l in lines if l.startswith("  Nose"))
    assert "no burn-through" in nose
    assert "of Reentry-F graphite" in nose
    # ablator body skin: its own record comparison
    body_row = next(l for l in lines if l.startswith("  Body skin"))
    assert "of Pioneer Venus" in body_row
    # interior row: bondline temp vs the design limit
    interior = next(l for l in lines if l.startswith("  Interior"))
    assert "of 250 °C" in interior
    # empty cells render as the single glyph
    assert "—" in nose and "—" in interior
    # regime line names the transition gate state
    assert "boundary layer" in txt


def test_glider_uhtc_cells():
    lines = _map_block(_rep("glider")["body"])
    nose = next(l for l in lines if l.startswith("  Nose"))
    # UHTC nose: surface vs the P→A boundary, dwell vs the demonstrated
    # floor, coverage as the record cell
    assert " K of " in nose                    # T vs pa_K
    assert " s of 300 s" in nose               # dwell vs floor
    rec = next(l for l in lines if "of dwell in envelope" in l)
    assert rec is nose
    interior = next(l for l in lines if l.startswith("  Interior"))
    assert "of 250 °C" in interior


# ── Spans: every span lands exactly on its cell text ─────────────────────────
def test_spans_land_on_cells():
    for key in ("mk21", "glider"):
        rep = _rep(key)
        lines = _map_block(rep["body"])
        spans = rep["map_spans"]
        assert spans, key
        for ln, c0, c1, tier in spans:
            assert tier in sr.SURVIVAL_TIERS
            cell = lines[ln][c0:c1]
            assert cell == cell.strip() and cell   # exact, no padding bleed
            assert lines[ln][c0 - 1] == "\t"       # starts on a tab boundary


# ── Cell logic units (synthetic locations) ───────────────────────────────────
def _abl_loc(frac=0.2, burn=False, record=3870.0):
    load = (frac * record) if frac is not None else 500.0
    return dict(material="carbon_carbon", is_ablator=True, T_eq_peak_K=4500.0,
                criteria=dict(recession=dict(
                    load_MJ_m2=load,
                    demonstrated_load_MJ_m2=record if frac is not None else None,
                    load_fraction=frac, burnthrough_bound=burn,
                    demonstrated_load_source="Reentry-F graphite nosetip flew")))


def test_ablator_cells():
    surf, dur, rec = sr._map_skin_cells(_abl_loc(0.2))
    assert surf == ("no burn-through", "experience") and dur is None
    assert rec[1] == "experience" and "20%" in rec[0]
    # past the record → yellow; ≥2× shows the multiple
    assert sr._map_skin_cells(_abl_loc(1.4))[2][1] == "beyond"
    assert "2.5×" in sr._map_skin_cells(_abl_loc(2.5))[2][0]
    # burn-through bound → the only red an ablator can show
    assert sr._map_skin_cells(_abl_loc(0.2, burn=True))[0] == \
        ("burn-through bound", "fail")
    # no cited record → green refinement question
    assert sr._map_skin_cells(_abl_loc(None))[2] == \
        ("no cited flight record", "experience")


def test_reradiative_cells():
    L = dict(material="steel", is_ablator=False, T_eq_peak_K=2201.0,
             criteria=dict(peak_surface=dict(margin=2201 / 1700, limit_K=1700.0),
                           soak=dict(margin=0.3, time_above_s=38.0,
                                     dwell_s=120.0, floor=False)))
    surf, dur, rec = sr._map_skin_cells(L)
    assert surf[1] == "fail" and "1,700 K" in surf[0]
    assert dur[1] == "experience" and rec is None
    # beyond the 4,000 K no-ablation screen → yellow, single cell
    hot = dict(material="steel", is_ablator=False, T_eq_peak_K=4500.0,
               criteria={})
    assert sr._map_skin_cells(hot)[0] == ("past 4,000 K screen", "beyond")


def test_windward_row_only_with_criterion():
    nose = _abl_loc(0.2)
    # ablator body → windward computed flux-only (no criterion) → row dropped
    w_abl = dict(T_eq_windward_K=dict(lo=1900.0, hi=2300.0), criteria={})
    lines, _ = sr._survival_map(nose, None, None, w_abl, None, None)
    assert not any("Windward" in l for l in lines)
    # non-ablator body → the criterion exists → row present, tier from limits
    w = dict(T_eq_windward_K=dict(lo=1900.0, hi=2300.0),
             criteria=dict(windward_surface=dict(
                 limit_continuous_K=1811.0, limit_peak_K=1922.0,
                 T_lo_K=1900.0, T_hi_K=2300.0)),
             transition_state="turbulent", transition_factor_peak=3.4)
    lines, spans = sr._survival_map(nose, None, None, w, None, None)
    row = next(l for l in lines if "Windward flank" in l)
    assert "2,300 K of 1,811 K" in row
    assert any(t == "beyond" and lines[ln][c0:c1].startswith("2,300")
               for ln, c0, c1, t in spans)
    assert any("×3.4" in l for l in lines)     # regime names the factor


def test_transitional_regime_wording():
    # A barely-transitional factor (< 1.05) must not print "×1.0".
    nose = _abl_loc(0.2)
    bl = dict(evaluated=True, T_bond_peak_C=27.0, limit_C=250.0, crossed=False,
              transition_state="transitional", transition_factor_peak=1.02)
    lines, _ = sr._survival_map(nose, None, None, None, bl, None)
    txt = "\n".join(lines)
    assert "just past onset" in txt and "×1.0" not in txt
