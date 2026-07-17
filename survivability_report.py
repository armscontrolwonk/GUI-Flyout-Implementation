"""Reentry Survivability report — the down-leg Schilling/Townsend panel.

Assembles a mode-keyed report (SURVIVABILITY_REPORT_DESIGN.md) from a
trajectory result: inputs echoed → budget table → per-criterion margins →
a JUDGEMENT WITH CONSEQUENCES (accuracy band / time-to-failure / maneuver
envelope) → method + flight-anchor references.  Three forms, keyed
automatically from the reentry plan already baked into the result:

  Form A — Ballistic RV       : the δ/R_n accuracy ladder (accuracy fails
                                before survival; PANT / Lin 1982 / Reentry-F)
  Form B — Glider / HGV       : the stopwatch (survival-time vs glide-time,
                                NRC-2008 duration ladder; thermal-range cap)
  Form C — Maneuvering (MaRV) : Form B + the terminal-dive transient block

Everything here is PRESENTATION over numbers heating.py already computed
(result['heating_fom']) plus the stashed reentry arc (result['heating_arc']).
Screening tier throughout: Sutton-Graves cold-wall stagnation flux +
radiative-equilibrium wall temperature; consequence bands are qualitative and
flight-anchored (the user-benchmarkable part, design doc §10).
"""

from __future__ import annotations
import numpy as np

import heating
import tps_ladder
from booster_models import glide_family

# δ/R_n consequence ladder (design doc §3; crosscheck §10.2).  Flight anchors:
#   0.1  — Lin 1982 (TRW-SCATHE): 0.1 R_N at 67 kft already "mildly indented";
#          PANT: asymmetric recession → dispersion well below blunting levels.
#   0.5–1 — Reentry-F flew its full mission at ≈0.7 R_n radial blunting.
#   burn-through — solid-tip LENGTH governs (Reentry-F ~7.7 R_n axial);
#          heating.py's recession criterion supplies the depth-based margin.
SHAPE_CHANGE_ONSET = 0.10
SEVERE_BLUNTING    = 0.50
# INTERNAL INFERENCE, not a literature number (see BENCHMARKING.md "Threshold
# provenance audit"): derived from Murbach 1993/AEOLUS (SWERVE flew a
# carbon-carbon nose/LE) plus AHW's move to non-ablating tips — any meaningful
# recession on a glider tip/LE corrupts the aeroshape.  No source gives a
# tolerance number; 0.05 is a screening flag, not a cited threshold.
GLIDER_ABL_TIP_FLAG = 0.05

# ---------------------------------------------------------------------------
# UHTC anchor dataset (SURVIVABILITY_REPORT_DESIGN.md §11.2) — one record per
# flight / arc-jet / plasma-torch / furnace datum.  "A new flight strengthens
# the dataset" is a DATA EDIT here, not a code change.  Sources are exact
# (never paraphrased); BENCHMARKING.md §UHTC is the citation of record with the
# full per-datum discussion.  Fields None where the source does not give them.
# ---------------------------------------------------------------------------
UHTC_ANCHORS = [
    dict(id="Monteverde-2013-ZS", material_class="zrb2_sic", kind="arcjet",
         tip_radius_m=None, flux_MW_m2=None, flux_kind=None, stag_pressure_Pa=None,
         peak_T_K=1973, T_source="measured", dwell_s=300, recession_um=0,
         mass_change_pct=None, outcome="survived", failure_mode=None,
         source="Monteverde & Savino 2013, Corros. Sci. 75 (300 s at 1973 K, zero recession)"),
    dict(id="Monteverde-2012-ZS-sharp", material_class="zrb2_sic", kind="arcjet",
         tip_radius_m=0.002, flux_MW_m2=7.0, flux_kind="cold_wall", stag_pressure_Pa=None,
         peak_T_K=2723, T_source="cfd", dwell_s=575, recession_um=None,
         mass_change_pct=None, outcome="survived", failure_mode=None,
         source="Monteverde & Savino 2012 (sharp ZrB2-SiC tip, passive to ~2450 °C CFD at ~7 MW/m², ~575 s, measurable blunting)"),
    dict(id="Scatteia-2010-blunt", material_class="zrb2_sic", kind="arcjet",
         tip_radius_m=None, flux_MW_m2=26.0, flux_kind="cold_wall", stag_pressure_Pa=None,
         peak_T_K=None, T_source=None, dwell_s=None, recession_um=None,
         mass_change_pct=None, outcome="survived", failure_mode=None,
         source="Scatteia et al. 2010, DOI 10.2514/1.42834 (blunt; passive-band recession ~3.6 µm/s at 26 MW/m²)"),
    dict(id="Zhang-2008-passive", material_class="zrb2_sic", kind="arcjet",
         tip_radius_m=None, flux_MW_m2=1.7, flux_kind="cold_wall", stag_pressure_Pa=None,
         peak_T_K=None, T_source=None, dwell_s=None, recession_um=None,
         mass_change_pct=0.0, outcome="survived", failure_mode=None,
         source="Zhang et al. 2008, Compos. Sci. Technol. 68:1718 (passive at 1.7 MW/m², ~0% mass loss)"),
    dict(id="Zhang-2008-active", material_class="zrb2_sic", kind="arcjet",
         tip_radius_m=None, flux_MW_m2=5.4, flux_kind="cold_wall", stag_pressure_Pa=None,
         peak_T_K=None, T_source=None, dwell_s=None, recession_um=3000,
         mass_change_pct=-15.75, outcome="failed", failure_mode="active oxidation (~5 µm/s recession)",
         source="Zhang et al. 2008, Compos. Sci. Technol. 68:1718 (active at 5.4 MW/m²; 15.75% mass loss, ~3 mm)"),
    dict(id="Marschall-2012-PA", material_class="zrb2_sic", kind="arcjet",
         tip_radius_m=None, flux_MW_m2=2.02, flux_kind="cold_wall", stag_pressure_Pa=10000,
         peak_T_K=2215, T_source="measured", dwell_s=None, recession_um=None,
         mass_change_pct=None, outcome="degraded", failure_mode="passive→active transition (+400 K temperature jump)",
         source="Marschall et al. 2012, JTHT 26(4), DOI 10.2514/1.T3798 (flat face, ~2 MW/m² / 10 kPa)"),
    dict(id="SHARP-B1", material_class="zrb2_sic", kind="flight",
         tip_radius_m=None, flux_MW_m2=None, flux_kind=None, stag_pressure_Pa=None,
         peak_T_K=None, T_source=None, dwell_s=None, recession_um=None,
         mass_change_pct=None, outcome="degraded", failure_mode="flight corroboration of the PA/runaway threshold",
         source="SHARP-B1 (Kolodziej et al.)"),
    dict(id="Gasch-Johnson-2010-HS", material_class="hfb2_sic", kind="arcjet",
         tip_radius_m=None, flux_MW_m2=2.5, flux_kind="cold_wall", stag_pressure_Pa=None,
         peak_T_K=1963, T_source="measured", dwell_s=600, recession_um=None,
         mass_change_pct=None, outcome="survived", failure_mode=None,
         source="Gasch & Johnson 2010 (HfB2-SiC, ~1690 °C at ~2.5 MW/m², 600 s)"),
    dict(id="Sevastyanov-2014-HfB2-45SiC", material_class="hfb2_sic", kind="arcjet",
         tip_radius_m=None, flux_MW_m2=None, flux_kind=None, stag_pressure_Pa=None,
         peak_T_K=2973, T_source="measured", dwell_s=1080, recession_um=None,
         mass_change_pct=-1.5, outcome="survived", failure_mode=None,
         source="Sevastyanov et al. 2014, DOI 10.1134/S0036023614110217 (2500–2700 °C, 15–18 min, 1.5% mass loss, no cracking; high-SiC ~20%-porous variant, 10–30 kPa)"),
    dict(id="Savino-2008-1atm", material_class="hfb2_hfc_mosi2", kind="arcjet",
         tip_radius_m=None, flux_MW_m2=None, flux_kind=None, stag_pressure_Pa=118000,
         peak_T_K=2273, T_source="measured", dwell_s=40, recession_um=None,
         mass_change_pct=None, outcome="survived", failure_mode=None,
         source="Savino et al. 2008, DOI 10.1016/j.jeurceramsoc.2007.11.021 (~1 atm, >2000 °C, ~30–40 s)"),
    dict(id="DePrisco-2026-lowp", material_class="complex_boride", kind="plasma_torch",
         tip_radius_m=None, flux_MW_m2=None, flux_kind=None, stag_pressure_Pa=300,
         peak_T_K=1800, T_source="measured", dwell_s=None, recession_um=None,
         mass_change_pct=None, outcome="survived", failure_mode=None,
         source="De Prisco et al. 2026, JECS 46:118184 (ZrB2-TiB2-SiC hemisphere, 3×10⁻³ atm)"),
    dict(id="DePrisco-2026-highp", material_class="complex_boride", kind="plasma_torch",
         tip_radius_m=None, flux_MW_m2=None, flux_kind=None, stag_pressure_Pa=2300,
         peak_T_K=2700, T_source="measured", dwell_s=None, recession_um=None,
         mass_change_pct=None, outcome="failed", failure_mode="oxide-scale detachment",
         source="De Prisco et al. 2026, JECS 46:118184 (same specimens at 2.3×10⁻² atm — pressure-sensitivity evidence)"),
]

# Envelope constants consumed by the coverage verdict (§11.3/§11.4).  Values
# read from the anchors above; the sharp/blunt PA split follows §11.4: the PA
# edge is a flux/pressure surface, screened here by the two bounding anchors.
_UHTC_SHARP_RN_M = 0.05           # below this, use the sharp-conducting-tip anchor
_UHTC_PA_SHARP_K = 2723.0         # Monteverde-2012-ZS-sharp (passive to 2450 °C @ 7 MW/m²)
_UHTC_PA_BLUNT_K = 2215.0         # Marschall-2012-PA (flat face, ~2 MW/m² / 10 kPa)


def _uhtc_coverage(t, q, eps, nose_radius_m, mat):
    """Envelope-coverage classification for a UHTC hot-structure nose (§11.3).

    Returns dict(bands=[(t0,t1,'green'|'amber'|'red'),...] (absolute times),
    dwell_s, covered_s, coverage (fraction of above-ceiling dwell inside the
    demonstrated envelope; 1.0 when nothing exceeds the ceiling), exits
    (subset of {'too hot','too long'}), pa_K, pa_anchor, floor_s, lines
    (report text block)).
    """
    t = np.asarray(t, float); q = np.asarray(q, float)
    eps = max(float(eps or 0.85), 1e-3)
    T_eq = (q / (heating.SIGMA * eps)) ** 0.25
    ceiling = float(mat["continuous_K"])                    # 1923 K (1650 °C)
    floor_s = float(mat.get("oxidation_dwell_s") or 300.0)  # demonstrated floor
    sharp = float(nose_radius_m or 0.0) < _UHTC_SHARP_RN_M
    pa_K = _UHTC_PA_SHARP_K if sharp else _UHTC_PA_BLUNT_K
    pa_anchor = ("Monteverde-2012-ZS-sharp (sharp conducting tip: passive to "
                 "~2450 °C at ~7 MW/m²)" if sharp else
                 "Marschall-2012-PA (flat face: PA jump at ~2215 K / "
                 "~2 MW/m² / 10 kPa)")
    dt = np.diff(t, prepend=t[0])
    above = T_eq > ceiling
    cum = np.cumsum(np.where(above, dt, 0.0))
    # per-sample class
    cls = np.full(t.shape, 0, int)                          # 0 green
    cls[above] = 1                                          # amber candidate
    cls[(above) & (cum > floor_s)] = 2                      # red: too long
    cls[T_eq > pa_K] = 3                                    # red: too hot (wins)
    dwell = float(cum[-1]) if cum.size else 0.0
    covered = float(np.sum(dt[cls == 1])) if t.size else 0.0
    coverage = 1.0 if dwell <= 0 else covered / dwell
    exits = set()
    if np.any(cls == 3):
        exits.add("too hot")
    if np.any(cls == 2):
        exits.add("too long")
    # contiguous bands for the plot
    bands = []
    if t.size:
        colour = {0: 'green', 1: 'amber', 2: 'red', 3: 'red'}
        i0 = 0
        for i in range(1, len(cls) + 1):
            if i == len(cls) or colour[cls[i]] != colour[cls[i0]]:
                bands.append((float(t[i0]), float(t[min(i, len(cls) - 1)]),
                              colour[cls[i0]]))
                i0 = i
    # report text
    lines = []
    if dwell <= 0:
        lines.append(f"  Envelope coverage: GREEN — nose never exceeds the "
                     f"{ceiling - 273.15:.0f} °C glass ceiling; "
                     f"silica-protected, no dwell clock runs.")
    else:
        lines.append(f"  Nose above {ceiling - 273.15:.0f} °C for "
                     f"{dwell:,.0f} s: {covered:,.0f} s within the "
                     f"demonstrated ZrB₂-SiC envelope "
                     f"(floor {floor_s:.0f} s — Monteverde-2013-ZS, "
                     f"1973 K · 300 s; sharp-tip extension 575 s).")
        if "too long" in exits:
            lines.append(f"  RED (too long): dwell outruns the demonstrated "
                         f"floor — beyond validated dwell, extrapolation "
                         f"(not asserted failure).  Fix: shorten exposure.")
        if "too hot" in exits:
            lines.append(f"  RED (too hot): surface crosses the "
                         f"passive→active oxidation boundary — protective "
                         f"silica lost, heating runs away.  "
                         f"Anchor: {pa_anchor}.  Fix: loft / blunt tip / "
                         f"lower flux.")
        if not exits:
            lines.append("  AMBER: inside the demonstrated envelope, "
                         "consuming recession margin.")
        lines.append("  * Demonstrated at ground-facility pressure "
                     "(anchors span 3×10⁻³–1 atm, but long-dwell points are "
                     "low-pressure; the SiC active/passive transition is "
                     "pressure-sensitive — §11.6).")
    return dict(bands=bands, dwell_s=dwell, covered_s=covered,
                coverage=coverage, exits=exits, pa_K=pa_K,
                pa_anchor=pa_anchor, floor_s=floor_s, lines=lines)


def classify(result) -> str:
    """'A' (ballistic RV) | 'B' (glider) | 'C' (MaRV: glide + terminal dive)."""
    arc = result.get('heating_arc') or {}
    prof = arc.get('profile') or {}
    if not prof.get('glider'):
        return 'A'
    if (prof.get('terminal_alt_km', 0.0) > 0.0
            or prof.get('dive_target_radius_km', 0.0) > 0.0):
        return 'C'
    return 'B'


def _fwhm_s(t, q):
    """Width (s) of the flux pulse above half its peak."""
    q = np.asarray(q, float); t = np.asarray(t, float)
    if q.size < 2 or np.max(q) <= 0:
        return 0.0
    m = q >= 0.5 * np.max(q)
    return float(t[m][-1] - t[m][0]) if m.any() else 0.0


def _load_MJ(t, q):
    """Running integrated load Q(t), MJ/m² (trapezoid)."""
    t = np.asarray(t, float); q = np.asarray(q, float)
    if t.size < 2:
        return np.zeros_like(t)
    dQ = 0.5 * (q[1:] + q[:-1]) * np.diff(t)
    return np.concatenate([[0.0], np.cumsum(dQ)]) / 1e6


def _accuracy_band(d_over_rn, burn_margin):
    """(band, sentence, anchor) from the δ/R_n consequence ladder."""
    if burn_margin is not None and burn_margin >= 1.0:
        return ("BURN-THROUGH",
                "nosetip consumed — failure",
                "Reentry-F (solid-tip length governs, ~7.7 R_n axial)")
    if d_over_rn is None:
        return ("N/A", "no recession computed (non-ablating or no material)", "")
    if d_over_rn < SHAPE_CHANGE_ONSET:
        return ("NOMINAL",
                "shape change negligible — accuracy preserved",
                "Lin 1982 ('mildly indented' begins ≈0.1 R_n)")
    if d_over_rn < SEVERE_BLUNTING:
        return ("ACCURACY-DEGRADED",
                "shape-change onset — dispersion growth likely (CEP "
                "degradation); vehicle survives",
                "PANT (ADA019186); Lin 1982")
    return ("SEVERE BLUNTING",
            "large shape change — survivable (Reentry-F flew ≈0.7 R_n) but "
            "accuracy heavily degraded; β falls, range/dispersion shift",
            "Reentry-F (NASA CR-154044)")


def _loc_line(name, L):
    """One per-location margin line for the budget table."""
    if not L or not L.get('material'):
        return f"  {name:<5s} (no TPS material set)"
    mat = str(L.get('material'))
    T = float(L.get('T_eq_peak_K', 0.0) or 0.0)
    crit = L.get('criteria') or {}
    cmp_ = L.get('compromise')
    if L.get('is_ablator') and 'recession' in crit:
        rc = crit['recession']
        det = (f"recedes {rc['recession_m']*100:.1f} cm "
               f"({rc['margin']:.0%} of layer)")
    elif T >= heating.NOTHING_SURVIVES_K:
        det = f"T_eq {T:,.0f} K ≥ 4,000 K screen — beyond screening"
    elif cmp_:
        det = f"{cmp_['mode']} at t={cmp_['t_s']:.0f} s"
    else:
        worst = max((v.get('margin', 0.0) for v in crit.values()), default=0.0)
        det = f"worst margin {worst:.2f}"
    return f"  {name:<5s} {mat:<18s} T_eq {T:>7,.0f} K   {det}"


def build_report(result) -> dict:
    """Assemble the survivability report.

    Returns {'status', 'headline', 'body', 'form', 'plot'} — plot carries
    (t, q_MW, Q_MJ, t_fail, glide_s, tiers) for the flux/load axes.  status ∈
    'survive' | 'degraded' | 'fail' | 'analysis' | 'none' (tab colouring).
    """
    fom = result.get('heating_fom')
    arc = result.get('heating_arc')
    if not fom or not arc:
        return dict(status='none', headline="No reentry heating computed",
                    body="Fly a trajectory that reenters (and set TPS "
                         "materials on the reentry object).",
                    form=None, plot=None)
    prof = arc.get('profile') or {}
    form = classify(result)
    fam = glide_family(prof.get('guidance')) if prof.get('glider') else None

    t = np.asarray(arc['t'], float)
    q = np.asarray(arc['q_dot'], float)
    t0 = float(t[0]) if t.size else 0.0
    q_MW = q / 1e6
    Q_MJ = _load_MJ(t, q)
    dur = float(fom.get('duration_s') or (t[-1] - t0 if t.size else 0.0))

    # Locations (per-location FOM when present, else the single-location dict).
    locs = fom.get('locations') or {'nose': fom}
    nose = locs.get('nose') or fom
    body_loc = locs.get('body')

    # UHTC hot-structure nose → the envelope-coverage verdict (§11.3) replaces
    # the boolean dwell fail for the nose; its dwell-floor "compromise" is an
    # extrapolation flag, not a failure, so it must not drive t_fail.
    _nose_mat = heating.TPS_MATERIALS.get(str(prof.get('nose_material') or ""))
    uhtc_nose = bool(form in ('B', 'C') and _nose_mat
                     and not _nose_mat.get('is_ablator')
                     and _nose_mat.get('oxidation_dwell_s'))
    coverage = None
    if uhtc_nose:
        coverage = _uhtc_coverage(t, q, prof.get('emissivity', 0.85),
                                  prof.get('nose_radius_m', 0.0), _nose_mat)

    # t_fail: earliest compromise across locations (absolute mission time —
    # heating.py evaluated the arc on t_arr, so compromise t_s is absolute).
    t_fail = None
    for name_, L in locs.items():
        if uhtc_nose and name_ == 'nose':
            continue                      # coverage verdict owns the nose
        c = (L or {}).get('compromise')
        if c and (t_fail is None or c['t_s'] < t_fail):
            t_fail = float(c['t_s'])

    # ---- header ------------------------------------------------------------
    mode_str = ('ballistic' if form == 'A' else prof.get('guidance', ''))
    fam_str = f"   [{fam} family]" if fam else ""
    hdr = [
        f"Reentry object:  {prof.get('name') or '(unnamed)'}   "
        f"(R_n {prof.get('nose_radius_m', 0)*100:.1f} cm, "
        f"⌀ {prof.get('diameter_m', 0):.2f} m, "
        f"{prof.get('mass_kg', 0):,.0f} kg)",
        f"Reentry mode:    {mode_str}{fam_str}",
        f"Entry (arc start): {arc.get('entry_V_ms', 0)/1000:.2f} km/s at "
        f"γ = {arc.get('entry_gamma_deg', 0):+.1f}°",
        f"TPS:             nose {prof.get('nose_material') or '(none)'} · "
        f"body {prof.get('body_material') or '(none)'}"
        + (f" {prof.get('body_thickness_m', 0)*100:.1f} cm"
           if prof.get('body_thickness_m') else ""),
    ]

    # ---- budget ------------------------------------------------------------
    budget = [
        "─── Heating budget ─────────────────────────────────────────",
        f"  Peak stagnation flux:  {fom.get('q_peak_MW_m2', 0):.1f} MW/m²"
        f"   (pulse width {_fwhm_s(t, q):.0f} s)",
        f"  Integrated load:       {fom.get('integrated_load_MJ_m2', 0):,.0f} MJ/m²",
        f"  Peak T_eq:             {fom.get('T_eq_peak_K', 0):,.0f} K",
        f"  Reentry-arc duration:  {dur:,.0f} s",
        "─── Per-location margins ───────────────────────────────────",
        _loc_line("nose", nose),
    ]
    if body_loc is not None:
        budget.append(_loc_line("body", body_loc))

    # ---- judgement (mode-keyed) ---------------------------------------------
    j = ["─── Judgement ──────────────────────────────────────────────"]
    status = 'survive'

    # nose recession numbers (shared by A and the glider tip flag)
    rc = (nose.get('criteria') or {}).get('recession')
    d_over_rn = rc.get('recession_over_Rn') if rc else None
    burn_margin = rc.get('margin') if rc else None

    if form == 'A':
        band, sentence, anchor = _accuracy_band(d_over_rn, burn_margin)
        if rc:
            j.append(f"  Nose recession δ = {rc['recession_m']*100:.1f} cm "
                     f"→ δ/R_n = {d_over_rn:.2f}   band: {band}")
        if band == "BURN-THROUGH" or t_fail is not None:
            status = 'fail'
            when = f" at t≈{t_fail:,.0f} s" if t_fail is not None else ""
            j.append(f"  FAILS{when} — {sentence}.")
        elif float(nose.get('T_eq_peak_K', 0)) >= heating.NOTHING_SURVIVES_K \
                and not nose.get('is_ablator'):
            status = 'analysis'
            j.append("  BEYOND SCREENING — reradiative surface above the "
                     "4,000 K no-ablation screen; needs ablation analysis.")
        elif band in ("ACCURACY-DEGRADED", "SEVERE BLUNTING"):
            status = 'degraded'
            j.append(f"  SURVIVES, ACCURACY DEGRADED — {sentence}.")
        else:
            j.append(f"  SURVIVES, ACCURACY PRESERVED — {sentence}.")
        if anchor:
            j.append(f"  Anchor: {anchor}")
        j.append("  (Loft/depress trade: run a burnout-angle sweep to see "
                 "flux vs load across shaping.)")

    else:   # Forms B and C — the stopwatch
        glide_range_km = None
        if t_fail is not None and t.size:
            rng = np.asarray(arc['range'], float)
            r_fail = float(np.interp(t_fail, t, rng)) / 1000.0
            r_end = float(rng[-1]) / 1000.0
            glide_range_km = (r_fail, r_end)
            status = 'fail'
            j.append(f"  TPS LIKELY FAILS at t≈{t_fail - t0:,.0f} s of the "
                     f"{dur:,.0f}-s glide "
                     f"({(t_fail - t0)/max(dur, 1e-9):.0%} of the mission).")
            j.append(f"  Thermal range ≈ {r_fail:,.0f} km of the "
                     f"{r_end:,.0f}-km aero range → the vehicle is "
                     f"thermal-range capped (min(aero, thermal)).")
        elif coverage is not None:
            # UHTC hot-structure nose: envelope-coverage verdict (§11.3) —
            # green/amber inside the demonstrated record, red = extrapolation
            # (named exit), never an asserted failure.
            j += coverage['lines']
            if coverage['exits']:
                status = 'analysis'
            elif coverage['dwell_s'] > 0:
                j.append(f"  Coverage: {coverage['coverage']:.0%} of "
                         f"above-ceiling dwell inside the demonstrated "
                         f"envelope.")
            if body_loc is not None and body_loc.get('material') \
                    and not body_loc.get('compromise'):
                j.append(f"  Body holds the full {dur:,.0f}-s glide.")
        else:
            # survives — but distinguish honest survive from beyond-screening
            _worst_T = float(nose.get('T_eq_peak_K', 0) or 0)
            if _worst_T >= heating.NOTHING_SURVIVES_K and not nose.get('is_ablator'):
                status = 'analysis'
                j.append(f"  NOSE BEYOND SCREENING (T_eq "
                         f"{_worst_T:,.0f} K ≥ 4,000 K): a tip at this "
                         f"equilibrium needs ablation/oxidation-life analysis "
                         f"this tier cannot provide (R_n "
                         f"{prof.get('nose_radius_m', 0)*100:.0f} cm drives "
                         f"q̇ ∝ 1/√R_n).")
                if body_loc is not None and not body_loc.get('compromise'):
                    j.append(f"  Body holds the full {dur:,.0f}-s glide.")
            else:
                j.append(f"  TPS SURVIVES THE FULL {dur:,.0f}-s GLIDE.")
        # glider ablative-tip rule (SWERVE→AHW): meaningful recession on a
        # glider tip corrupts the aeroshape regardless of survival.
        if d_over_rn is not None and d_over_rn >= GLIDER_ABL_TIP_FLAG:
            if status == 'survive':
                status = 'degraded'
            j.append(f"  Ablative tip recedes δ/R_n = {d_over_rn:.2f} — on a "
                     f"glider any meaningful recession corrupts the aeroshape: "
                     f"needs a non-ablating tip (UHTC-class).  "
                     f"[SWERVE→AHW rule]")
        if fam == 'analytic':
            j.append("  Family note: analytic (idealized smooth capture) — "
                     "as-flown numerical modes typically read 2–4× higher "
                     "peak flux (phugoid troughs).")

        if form == 'C':
            # terminal-dive transient block (screening): the arc below the
            # commanded dive altitude (or 15 km for dive-at-target).
            _h_dive = (prof.get('terminal_alt_km', 0.0) or 15.0) * 1000.0
            alt = np.asarray(arc['alt'], float)
            m = alt <= _h_dive
            if m.any() and np.count_nonzero(m) > 1:
                q_d = q[m]; t_d = t[m]
                j.append("─── Terminal-dive transient (screening) ───────────────────")
                j.append(f"  Dive segment below {_h_dive/1000:.0f} km: peak "
                         f"{np.max(q_d)/1e6:.1f} MW/m² over "
                         f"{t_d[-1]-t_d[0]:.0f} s — heat-sink regime "
                         f"(windward flank/fin LE, not the nose; AoA probe is "
                         f"a later tier).")

    # ---- NRC ladder (gliders) + method line ---------------------------------
    tail = []
    if form in ('B', 'C') and dur > 60.0:
        tail += ["", tps_ladder.format_ladder(dur)]
    tail += [
        "",
        "Method: screening tier — Sutton-Graves cold-wall stagnation flux +",
        "radiative-equilibrium wall T; consequence bands are qualitative and",
        "flight-anchored (Reentry-F, PANT, Lin 1982, HTV-2, NRC-2008 tiers).",
        "Not a through-wall TPS design analysis.",
    ]

    headline = {
        'survive':  "LIKELY SURVIVES",
        'degraded': "SURVIVES — DEGRADED",
        'fail':     "LIKELY FAILS",
        'analysis': "NEEDS ANALYSIS (beyond screening)",
    }[status]
    if coverage is not None and coverage['exits'] and status == 'analysis':
        headline = ("BEYOND DEMONSTRATED ENVELOPE — EXTRAPOLATION ("
                    + " + ".join(sorted(coverage['exits'])) + ")")
    _form_name = {'A': "ballistic RV", 'B': "glider", 'C': "maneuvering (MaRV)"}[form]
    headline += f"   —   Form {form} ({_form_name})"

    body = "\n".join(hdr) + "\n\n" + "\n".join(budget) + "\n" \
           + "\n".join(j) + "\n" + "\n".join(tail) + "\n"

    plot = dict(
        t=t - t0, q_MW=q_MW, Q_MJ=Q_MJ,
        t_fail=(t_fail - t0) if t_fail is not None else None,
        glide_s=dur if form in ('B', 'C') else None,
        tiers=(tps_ladder.NAS_LINEAGE if form in ('B', 'C') else None),
        bands=([(b0 - t0, b1 - t0, c) for b0, b1, c in coverage['bands']]
               if coverage is not None else None),
    )
    return dict(status=status, headline=headline, body=body,
                form=form, plot=plot)
