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
import thresholds
from booster_models import glide_family

# ── Unified survival ladder ────────────────────────────────────────────────
# One 4-tier verdict for EVERY material (ablator, hot structure, metal, tile).
# The underlying test differs by material; the output your eye reads is always
# one of these four.  Colors carry the intended semantics: green = demonstrated
# experience; blue = permitted extrapolation (permission, NOT caution); yellow
# = caution / beyond design; red = reserved for a COMPUTED failure.
SURVIVAL_TIERS = {
    'experience': ("WITHIN EXPERIENCE",      "#2e8b57"),  # green
    'design':     ("WITHIN DESIGN ENVELOPE", "#2f6fb0"),  # blue (permission)
    'beyond':     ("BEYOND DESIGN ENVELOPE", "#c78a0a"),  # yellow (caution)
    'fail':       ("CANNOT SURVIVE",         "#c0392b"),  # red
}
_TIER_KEYS = ('experience', 'design', 'beyond', 'fail')
# Per-timestep coverage-band class → tier (for the plot shading).  0 below the
# glass ceiling and 1 above-but-within the demonstrated envelope are both
# demonstrated → experience; 2 (dwell past the demonstrated floor, still
# passive) is design-vouched extrapolation → design; 3 (passive→active
# transition) is beyond-design caution.
_BAND_CLASS_TIER = {'green': 'experience', 'amber': 'experience',
                    'red': 'design', 'redhot': 'beyond'}


def survival_tier(status, coverage=None):
    """Collapse the internal status (+ UHTC coverage) into one survival tier.

    RED ('cannot survive') is reserved for a COMPUTED failure — burn-through,
    melt, a t_fail crossing.  BLUE ('within design') is only reachable for a
    material with a demonstrated envelope to extrapolate past (the payoff of a
    curated anchor dataset); a material without one shows green/yellow/red only.
    """
    if status == 'fail':
        return 'fail'
    if coverage is not None and coverage.get('exits'):
        ex = coverage['exits']
        if 'too hot' in ex:        # passive→active oxidation: beyond design, less likely
            return 'beyond'
        if 'too long' in ex:       # past demonstrated dwell, still passive: design vouches
            return 'design'
    if status in ('analysis', 'degraded'):
        return 'beyond'
    return 'experience'

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
        # 2 = dwell past the demonstrated floor (still passive) → design tier;
        # 3 = passive→active transition → beyond tier (see _BAND_CLASS_TIER).
        colour = {0: 'green', 1: 'amber', 2: 'red', 3: 'redhot'}
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
            lines.append(f"  TOO LONG (blue 'within design' band): dwell "
                         f"outruns the demonstrated floor — the test record "
                         f"simply ENDS here.  The data is a floor, not a "
                         f"fence (nearly every anchor is a survival where "
                         f"the test stopped), so survival past the floor is "
                         f"plausible but undemonstrated — this is "
                         f"extrapolation, not a failure prediction.  "
                         f"Fix: shorten exposure.")
        if "too hot" in exits:
            lines.append(f"  TOO HOT (yellow 'beyond design' band): surface "
                         f"crosses the passive→active oxidation boundary — "
                         f"unlike 'too long', this exit has failure-side "
                         f"physics behind it: protective silica is lost, "
                         f"heating jumps (~+400 K, Marschall) and recession "
                         f"runs ~10× faster per unit flux (Zhang 2008, "
                         f"~5 µm/s active).  Rapid degradation expected, "
                         f"though not an asserted kill at screening tier.  "
                         f"Anchor: {pa_anchor}.  Fix: loft / blunt tip / "
                         f"lower flux.")
        if not exits:
            lines.append("  Within the demonstrated envelope (green band), "
                         "consuming recession margin.")
        lines.append("  * Demonstrated at ground-facility pressure "
                     "(anchors span 3×10⁻³–1 atm, but long-dwell points are "
                     "low-pressure; the SiC active/passive transition is "
                     "pressure-sensitive — §11.6).")
    return dict(bands=bands, dwell_s=dwell, covered_s=covered,
                coverage=coverage, exits=exits, pa_K=pa_K,
                pa_anchor=pa_anchor, floor_s=floor_s, lines=lines)


# ---------------------------------------------------------------------------
# Form C maneuver-load anchor dataset — one record per demonstrated (or
# published-representative) MaRV maneuver load.  Same philosophy as
# UHTC_ANCHORS: a new flight datum is a DATA EDIT, not a code change; sources
# are exact and BENCHMARKING.md §Form C is the citation of record.  These are
# structural/guidance survived-the-maneuver demonstrations, NOT thermal
# limits — the context block below is a demonstrated-envelope comparison,
# never a pass/fail verdict.
# ---------------------------------------------------------------------------
MANEUVER_ANCHORS = [
    dict(id="Regan-1984-worked-4g", vehicle="Regan 1984 worked case", g=4.0,
         kind="textbook",
         note="fixed L/D=1.5, β=10⁴ kg/m² MaRV hits the 4-g transverse limit "
              "at ≈45 km (gentle accuracy-maneuver class)",
         source="Regan 1984, Re-Entry Vehicle Dynamics (AIAA), Tables 6.7/6.8"),
    dict(id="Pershing-II-pullout", vehicle="Pershing II", g=25.0,
         kind="operational_flight",
         note="~25-g pullout below ~50 kft after ~Mach-8 reentry (RADAG "
              "map-match segment); velocity-control pullup/pulldown, fielded "
              "Dec 1983",
         source="Yengst 2010, Lightning Bolts; maneuver corroborated by "
                "Lund 1984 (Martin Marietta/AIAA)"),
    dict(id="BGRV-qual-25g", vehicle="BGRV", g=25.0, kind="qualification",
         note="components qualified to 25 g (Atlas-boosted, >Mach-15 "
              "separation onto low-altitude glide)",
         source="Yengst 2010, Lightning Bolts"),
    dict(id="AMaRV-flight-100g", vehicle="AMaRV", g=100.0,
         kind="flight_measured",
         note="Bell XI accelerometers measured >100-g reentry-maneuver "
              "levels; guidance held accuracy through ~100-g maneuvers "
              "(3 flights, 1979–81)",
         source="Yengst 2010, Lightning Bolts"),
    dict(id="Regan-1993-evader-cap", vehicle="Regan 1993 evader", g=100.0,
         kind="textbook",
         note="representative evader max side acceleration 100 g (140 kg, "
              "⌀ 0.4 m, (L/D)max 2.5, β ≈ 1.1×10⁴ kg/m²)",
         source="Regan & Anandakrishnan 1993, Dynamics of Atmospheric "
                "Re-Entry (AIAA), Table D.1"),
    dict(id="Wang-2019-PII-overload", vehicle="Pershing II-modeled HGRV",
         g=25.0, kind="textbook",
         note="modern guidance-simulation overload constraint ±25 g for a "
              "Pershing II-modeled vehicle (Table 1); glide/bleed phase at "
              "~16 km matches Yengst's sub-50-kft band — independent "
              "corroboration of the 25-g class, not flight data",
         source="Wang, Tang & Zhang 2019, IEEE Access 7:47437, "
                "DOI 10.1109/ACCESS.2019.2909589"),
]

# Envelope constants read from the anchors above.
_MARV_G_OPERATIONAL = 25.0    # Pershing-II-pullout (fielded system)
_MARV_G_DEMONSTRATED = 100.0  # AMaRV-flight-100g (flight-measured ceiling)


def _maneuver_context(g_cmd):
    """Demonstrated maneuver-load envelope context for Form C (text block).

    Compares the plan's commanded lift cap (glider_pullup_g_max, in g) to the
    flight-demonstrated ladder.  Context only — the anchors are structural/
    guidance demonstrations, not thermal limits, so this never changes the
    survivability status.
    """
    g_cmd = float(g_cmd or 0.0)
    if g_cmd <= 0.0:
        return []
    lines = ["─── Maneuver-load anchors (demonstrated envelope) ──────────",
             f"  Commanded lift cap {g_cmd:g} g vs the open flight record: "
             f"4 g (Regan textbook gentle) · 25 g (Pershing II operational "
             f"pullout; BGRV qual) · ~100 g (AMaRV, flight-measured)."]
    if g_cmd <= _MARV_G_OPERATIONAL:
        lines.append("  Within the operational-MaRV class (Pershing II "
                     "~25-g pullout, fielded Dec 1983).")
    elif g_cmd <= _MARV_G_DEMONSTRATED:
        lines.append("  Above the operational Pershing II class but inside "
                     "the AMaRV flight-demonstrated ~100-g ceiling "
                     "(3 flights, 1979–81).")
    else:
        lines.append(f"  EXCEEDS every flight-demonstrated maneuver load in "
                     f"the open record (~{_MARV_G_DEMONSTRATED:.0f} g, "
                     f"AMaRV) — structural/guidance extrapolation beyond the "
                     f"anchor dataset.")
    lines.append("  * Load anchors are survived-the-maneuver structural/"
                 "guidance demonstrations, not thermal limits; windward/AoA "
                 "heating during the pull-up is a later-tier probe "
                 "(engineering-code uncertainty ~15–40% at AoA, "
                 "Thompson 1989).")
    return lines


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
    _fail_loc = _fail_mode = None
    for name_, L in locs.items():
        if uhtc_nose and name_ == 'nose':
            continue                      # coverage verdict owns the nose
        c = (L or {}).get('compromise')
        if c and (t_fail is None or c['t_s'] < t_fail):
            t_fail = float(c['t_s'])
            _fail_loc, _fail_mode = name_, c.get('mode')

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
            # Windward-flank heating (screening AoA probe): a lifting vehicle
            # flies its glide at AoA, so the windward generator — not the nose —
            # carries the off-nose acreage heat.  The α=0 acreage flux scaled by
            # the modified-Newtonian amplification A(α)=sin(δ+α)/sin(δ), over the
            # glide sub-arc (heating.windward_flank_flux).
            _w = (fom or {}).get('windward')
            if _w and _w.get('T_eq_windward_K'):
                _T = _w['T_eq_windward_K']; _qw = _w['q_windward_MW_m2']
                _amp = _w['amplification']; _ab = _w['alpha_band_deg']
                j.append("─── Windward-flank heating (screening) ─────────────────────")
                _opstr = (f", {_T['op']:.0f} K at trim α={_w['alpha_op_deg']:.0f}°"
                          if _w.get('alpha_op_deg') is not None and _T.get('op') else "")
                j.append(f"  Windward T_eq {_T['lo']:.0f}–{_T['hi']:.0f} K across "
                         f"α {_ab[0]:.0f}–{_ab[1]:.0f}°{_opstr}  "
                         f"(δ={_w['delta_deg']:.0f}° flank; "
                         f"{_qw['lo']:.1f}–{_qw['hi']:.1f} MW/m², "
                         f"{_amp['lo']:.1f}–{_amp['hi']:.1f}× the α=0 acreage flux).")
                _wc = (_w.get('criteria') or {}).get('windward_surface')
                if _wc:
                    j.append(f"  vs body {_w['body_material']}: soak "
                             f"{_wc['limit_continuous_K']:.0f} K / peak "
                             f"{_wc['limit_peak_K']:.0f} K — {_w['verdict']}.")
                    if heating.WINDWARD_DRIVES_VERDICT:
                        if _wc['T_lo_K'] > _wc['limit_continuous_K'] and status == 'survive':
                            status = 'degraded'
                        elif _wc['T_hi_K'] > _wc['limit_peak_K'] and status == 'survive':
                            status = 'analysis'
                elif _w.get('verdict'):
                    j.append(f"  {_w['verdict']}.")
                j.append(f"  {_w.get('thompson_band', '')}; laminar — turbulent "
                         f"flank ~3–5× higher; control-fin gap interference "
                         f"10–80× at reattachment (Alviani 2022) — flags, not "
                         f"computed at screening tier.")

            # Terminal-dive transient block (screening): the low-AoA arc below
            # the commanded dive altitude (or 15 km for dive-at-target) — the
            # nose-stagnation complement to the windward glide flank above.
            _h_dive = (prof.get('terminal_alt_km', 0.0) or 15.0) * 1000.0
            alt = np.asarray(arc['alt'], float)
            m = alt <= _h_dive
            if m.any() and np.count_nonzero(m) > 1:
                q_d = q[m]; t_d = t[m]
                j.append("─── Terminal-dive transient (screening) ───────────────────")
                j.append(f"  Dive segment below {_h_dive/1000:.0f} km: peak "
                         f"{np.max(q_d)/1e6:.1f} MW/m² over "
                         f"{t_d[-1]-t_d[0]:.0f} s — heat-sink regime "
                         f"(nose-stagnation; the windward flank/fin LE is the "
                         f"block above).")
            j += _maneuver_context(prof.get('pullup_g_max', 0.0))

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

    # ── Modified-benchmark self-disclosure ───────────────────────────────────
    # A user-edited screening threshold rides on NONE of the shipped citations,
    # so the report says so plainly: an asterisk on the headline and a block
    # naming each changed number, its shipped default, and the default's source.
    _mods = thresholds.modified()
    if _mods:
        tail += ["", "Modified benchmarks (screening thresholds changed from the",
                 "shipped defaults — this verdict does not carry the citation of",
                 "record for the numbers below):"]
        for m in _mods:
            _u = (" " + m["units"]) if m["units"] else ""
            tail.append(f"  • {m['label']}: {m['value']:g}{_u} "
                        f"(default {m['default']:g}{_u}, {m['source']})")

    # ── Unified 4-tier verdict (the headline every material now shares) ──────
    tier = survival_tier(status, coverage)
    tier_label, tier_color = SURVIVAL_TIERS[tier]
    if coverage is not None and coverage.get('exits') and tier in ('design', 'beyond'):
        tier_label += "  (" + " + ".join(sorted(coverage['exits'])) + ")"
    _form_name = {'A': "ballistic RV", 'B': "glider", 'C': "maneuvering (MaRV)"}[form]
    headline = f"{tier_label}   —   Form {form} ({_form_name})"
    if _mods:
        headline += "  *"

    # ── Plain-language lead (inverted pyramid) ───────────────────────────────
    # Three layers a policy reader needs, in order: what was flown, why the
    # verdict is what it is (binding location + mechanism), and what would
    # change it.  The full engineering analysis — citations intact — follows
    # below the divider, unchanged.
    _name = prof.get('name') or '(unnamed)'
    _mode_phrase = ('ballistic reentry' if form == 'A'
                    else f"{prof.get('guidance', 'glide')} glide")
    lead = [f"{_name} — {_mode_phrase}, {dur:,.0f}-s reentry arc."]
    because, fix = [], None
    if form == 'A':
        _band, _, _ = _accuracy_band(d_over_rn, burn_margin)
        _dr = (f" (δ/R_n = {d_over_rn:.2f})"
               if d_over_rn is not None else "")
        if status == 'fail':
            _when = f" at t≈{t_fail:,.0f} s" if t_fail is not None else ""
            if _band == "BURN-THROUGH":
                because.append(f"The nose tip is consumed{_when} — the "
                               f"ablator burns through before impact.")
                fix = ("a thicker ablator, a blunter nose, or a less "
                       "demanding trajectory")
            else:
                because.append(f"The {_fail_loc or 'nose'} TPS "
                               f"fails{_when}: "
                               f"{_fail_mode or 'thermal limit exceeded'}.")
                fix = ("more capable TPS, a blunter nose, or a less "
                       "demanding trajectory")
        elif status == 'analysis':
            because.append(
                f"The nose reaches {float(nose.get('T_eq_peak_K', 0) or 0):,.0f} K — "
                f"hotter than any non-ablating surface can reradiate away, so "
                f"this screen cannot assess it; it needs a dedicated ablation "
                f"analysis.")
            fix = "a blunter nose (flux falls as 1/√R_n) or an ablative tip"
        elif status == 'degraded':
            if _band == "SEVERE BLUNTING":
                because.append(
                    f"The vehicle survives, but the nose blunts "
                    f"heavily{_dr} — survivable (Reentry-F flew ≈0.7 R_n) "
                    f"but accuracy is heavily degraded.")
            else:
                because.append(
                    f"The vehicle survives, but the nose recedes "
                    f"enough{_dr} to change its shape — dispersion grows "
                    f"and accuracy degrades.")
            fix = "a blunter or thicker tip, or a gentler trajectory"
        else:
            because.append(f"The vehicle survives with negligible shape "
                           f"change{_dr} — accuracy preserved.")
    else:
        _body_holds = bool(body_loc is not None and body_loc.get('material')
                           and not body_loc.get('compromise'))
        if status == 'fail' and t_fail is not None:
            _rng = np.asarray(arc['range'], float)
            _rf = float(np.interp(t_fail, t, _rng)) / 1000.0
            _re = float(_rng[-1]) / 1000.0
            because.append(
                f"The {_fail_loc or 'nose'} TPS likely fails at "
                f"t≈{t_fail - t0:,.0f} s "
                f"({_fail_mode or 'thermal limit exceeded'}) — "
                f"{(t_fail - t0)/max(dur, 1e-9):.0%} into the glide; range is "
                f"thermally capped at ≈{_rf:,.0f} of the {_re:,.0f}-km aero "
                f"range.")
            fix = "a shorter or steeper profile, or more capable TPS"
        elif coverage is not None and coverage.get('exits'):
            _ceil_C = (float(_nose_mat['continuous_K']) - 273.15) if _nose_mat else 1650.0
            _nose_lbl = (_nose_mat.get('label') if _nose_mat else None) \
                or prof.get('nose_material') or 'hot-structure'
            _dwell_cl = (
                f"the demonstrated test record ends at "
                f"{coverage['floor_s']:.0f} s, so most of this dwell is "
                f"extrapolation, not demonstration"
                if 'too long' in coverage['exits'] else
                f"within the {coverage['floor_s']:.0f}-s demonstrated dwell "
                f"record")
            because.append(
                f"The {_nose_lbl} nose runs above {_ceil_C:.0f} °C for "
                f"{coverage['dwell_s']:,.0f} s — {_dwell_cl}.")
            if 'too hot' in coverage['exits']:
                because.append(
                    "The surface crosses into active oxidation — the "
                    "protective silica layer is lost and degradation "
                    "accelerates; this is the failure-side edge of the "
                    "envelope, not just missing data.")
            if _body_holds:
                because.append(f"The body TPS holds the full glide; the nose "
                               f"is the binding problem.")
            fix = ("loft, blunt the tip, or fly a lower-flux profile"
                   if 'too hot' in coverage['exits']
                   else "shorten the hot dwell (a steeper or shorter glide)")
        elif status == 'analysis':
            because.append(
                f"The nose reaches {float(nose.get('T_eq_peak_K', 0) or 0):,.0f} K — "
                f"hotter than any non-ablating surface can reradiate away, so "
                f"this screen cannot assess it; it needs a dedicated ablation "
                f"analysis." + ("  The body TPS holds the full glide."
                                if _body_holds else ""))
            fix = "a blunter nose (flux falls as 1/√R_n) or an ablative tip"
        elif status == 'degraded':
            if d_over_rn is not None and d_over_rn >= GLIDER_ABL_TIP_FLAG:
                because.append(
                    f"The TPS survives the heat, but the ablative tip recedes "
                    f"(δ/R_n = {d_over_rn:.2f}) — on a glider any meaningful "
                    f"recession corrupts the aeroshape it steers with.")
                fix = "a non-ablating (UHTC-class) tip"
            else:
                because.append("The TPS survives, but a screening overlay "
                               "degrades the verdict — see the full analysis "
                               "below.")
        else:
            because.append(f"Nose and body both hold the full {dur:,.0f}-s "
                           f"glide within the demonstrated record.")
    lead += ["", " ".join(because)]
    if fix:
        lead += ["", f"What would change the verdict: {fix}."]
    # NRC lineage context, written to keep the two ladders distinct: the NRC
    # rungs are a DESIGN LINEAGE (what TPS class each mission duration
    # historically required), not a demonstration of the flown material.
    if form in ('B', 'C') and dur > 60.0:
        _ctx = (f"For context: gliders of this class historically used "
                f"ablative carbon-phenolic to ~800 s (NRC 2008, AMaRV "
                f"lineage); past ~{tps_ladder.CROSSOVER_S:,.0f} s the record "
                f"steps to advanced carbon-carbon.")
        if uhtc_nose and coverage is not None:
            _ctx += (f"  This vehicle instead flies a reradiating UHTC nose — "
                     f"a different material with a different failure mode, "
                     f"whose own demonstrated record ends at "
                     f"{coverage['floor_s']:.0f} s.")
        elif _nose_mat and _nose_mat.get('is_ablator'):
            _ctx += (f"  This vehicle flies an ablative "
                     f"{_nose_mat.get('label', 'nose')}, judged here by "
                     f"recession, not dwell.")
        lead += ["", _ctx]
    if _mods:
        lead += ["", "* Screening benchmarks modified from the shipped "
                     "defaults — see Modified benchmarks below."]

    _divider = "═══ Full analysis " + "═" * 42
    body = "\n".join(lead) + "\n\n" + _divider + "\n\n" \
           + "\n".join(hdr) + "\n\n" + "\n".join(budget) + "\n" \
           + "\n".join(j) + "\n" + "\n".join(tail) + "\n"

    plot = dict(
        t=t - t0, q_MW=q_MW, Q_MJ=Q_MJ,
        t_fail=(t_fail - t0) if t_fail is not None else None,
        glide_s=dur if form in ('B', 'C') else None,
        tiers=(tps_ladder.NAS_LINEAGE if form in ('B', 'C') else None),
        tier=tier, tier_color=tier_color,
        # bands recolored from per-timestep coverage class → survival tier
        bands=([(b0 - t0, b1 - t0, _BAND_CLASS_TIER.get(c, c))
                for b0, b1, c in coverage['bands']]
               if coverage is not None else None),
    )
    return dict(status=status, tier=tier, headline=headline, body=body,
                form=form, plot=plot)
