"""Build the JL-2 SLBM + RV workbooks from parade-photo photogrammetry.

Produces two files in models/:
  JL-2.xlsx          — detailed single-missile workbook (missile_xlsx format;
                       RV parameters live in its PAYLOAD & REENTRY VEHICLE block)
  JL-2_catalog.xlsx  — catalog long-format (docs/ingestion format: one `rockets`
                       row per stage + a `payloads` row for the RV)

Measurement chain (Jeffrey Lewis, WebPlotDigitizer on 2019-parade JL-3
canister photo — proxy geometry for the JL-2 family):
  Dist3 = 154.6746 px vertical = canister radius 1.1 m  → scale 7.112 mm/px
  Dist1 = 620.8316 px = stage-1 raceway (motor cylinder) = 4.415 m
  Dist0 = 324.1810 px = stage-2 raceway                  = 2.306 m
  Dist2 =  44.4954 px = motor dome height                = 0.316 m
Each motor = cylinder + two half-ellipsoid domes of height Dist2.

Assumption set (P35-anchored; Wei & Bai, "Design and Development of P35
Solid Rocket Motor for Long March 11 Launch Vehicle", GLEX 2017, #36288 —
same steel-case HTPB technology family):
  missile diameter    2.0 m   (canister OD 2.2 m discounted to the missile;
                               P35 is likewise 2.0 m)
  grain density       1800 kg/m³ (HTPB/AP composite; mass_estimator.py APCP)
  volumetric loading  0.85
  mass ratio          0.88    (prop/wet; P35 flight value → dry/wet = 0.12)
  Isp (vacuum)        272 s   (P35 ground Isp 248 s + P_a·A_e/(mdot·g0))
  stage-1 burn        48 s    (P35-class average thrust ~1176 kN)
  stage-2 burn        48 s    placeholder (same diameter -> similar web
                              thickness; detailed workbook only — the
                              catalog leaves it blank for the resolver)
  stage-1 nozzle A_e  sized so sea-level Isp comes out at exactly 248 s
Stage 3 was not measurable in the photo: its row/column is left blank for
the ingest resolver (or a later measurement) to fill.  RV mass/beta are
likewise left blank (estimate-with-VERIFY).

CAVEAT: per-stage `mass_initial` in the detailed workbook is the PARTIAL
stack (stages 1-2 wet only) — it excludes the unmeasured stage 3 and
payload, so update it when those numbers land.

Run from the repo root:  python models/make_jl2_xlsx.py
"""
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'docs', 'ingestion'))

G0 = 9.80665
PA_SL = 101_325.0            # Pa, sea level

# ── WebPlotDigitizer measurements (px) ──────────────────────────────────────
PX_RADIUS  = 154.6745886654478   # Dist3: canister radius
PX_STAGE1  = 620.8316474783319   # Dist1: stage-1 raceway
PX_STAGE2  = 324.18098720292505  # Dist0: stage-2 raceway
PX_DOME    = 44.49542961608768   # Dist2: dome height
CANISTER_RADIUS_M = 1.1

# ── Assumptions (see module docstring for provenance) ───────────────────────
DIAMETER_M     = 2.0
RHO_GRAIN      = 1800.0          # kg/m³
VOL_LOADING    = 0.85
MASS_RATIO     = 0.88            # propellant / wet  (P35)
ISP_VAC_S      = 272.0
ISP_GROUND_S   = 248.0           # P35 ground Isp — used only to size A_e
BURN1_S        = 48.0

SCALE = CANISTER_RADIUS_M / PX_RADIUS          # m per px
DOME_H = PX_DOME * SCALE

def stage_masses(px_raceway: float):
    """(length_m, prop_kg, wet_kg, dry_kg) for one motor."""
    r = DIAMETER_M / 2.0
    l_cyl = px_raceway * SCALE
    vol = math.pi * r * r * l_cyl + 2.0 * (2.0 / 3.0) * math.pi * r * r * DOME_H
    prop = vol * RHO_GRAIN * VOL_LOADING
    wet = prop / MASS_RATIO
    return l_cyl + 2.0 * DOME_H, prop, wet, wet - prop

L1, PROP1, WET1, DRY1 = stage_masses(PX_STAGE1)
L2, PROP2, WET2, DRY2 = stage_masses(PX_STAGE2)

MDOT1     = PROP1 / BURN1_S
THRUST1_N = ISP_VAC_S * G0 * MDOT1                       # vacuum
AE1_M2    = (ISP_VAC_S - ISP_GROUND_S) * MDOT1 * G0 / PA_SL

CITATION = ('Photogrammetry of 2019-parade JL-3 canister (proxy geometry); '
            'motor anchors from P35 (Wei & Bai, GLEX 2017 #36288): '
            'steel case HTPB, mass ratio 0.88, ground Isp 248 s')


BURN2_S   = 48.0                                         # placeholder (see docstring)
THRUST2_N = ISP_VAC_S * G0 * PROP2 / BURN2_S             # vacuum


def build_detailed(path: str) -> None:
    from missile_models import MissileParams
    from missile_xlsx import export_missile_xlsx, _R

    def mk(name, mass_init, prop, wet_dry, length, thrust, burn, ae):
        p = MissileParams(
            name=name, mass_initial=round(mass_init), mass_propellant=round(prop),
            mass_final=round(wet_dry), diameter_m=DIAMETER_M, length_m=round(length, 2),
            thrust_N=round(thrust), burn_time_s=burn, isp_s=ISP_VAC_S,
        )
        p.solid_motor = True
        p.nozzle_exit_area_m2 = ae
        return p

    # mass_initial = stack at ignition; stage 3 + payload NOT yet included.
    s1 = mk('JL-2', WET1 + WET2, PROP1, DRY1, L1, THRUST1_N, BURN1_S, round(AE1_M2, 2))
    s2 = mk('JL-2 S2', WET2, PROP2, DRY2, L2, THRUST2_N, BURN2_S, 0.0)
    s1.stage2 = s2
    # RV block — mass/beta/geometry to be filled from a future measurement
    s1.num_rvs = 1
    s1.rv_separates = True
    export_missile_xlsx(path, s1)

    # rv_shape only serialises alongside a positive beta, which we don't
    # have yet — write the shape cell directly so the intent is recorded.
    import openpyxl
    wb = openpyxl.load_workbook(path)
    ws = wb['Missile']
    ws.cell(row=_R['rv_shape'], column=4, value='Cone')
    # Body nose shape: blunted SLBM ogive, approximated as tangent ogive.
    # With a shape set, the Chin (1961) component drag model governs.
    ws.cell(row=_R['nose_shape'], column=4, value='Tangent Ogive')
    # The exporter still pre-fills the legacy Forden Cd table, which the
    # importer would treat as an explicit override — clear it so the
    # nose-shape (Chin) model is what the workbook actually encodes.
    wc = wb['Cd Table']
    for row in wc.iter_rows(min_row=4, max_col=2):
        for cell in row:
            cell.value = None
    wb.save(path)


def build_catalog(path: str) -> None:
    import make_catalog_template as cat

    cat.ROCKETS_ROWS = [
        ['jl2', 'JL-2 (SLBM)', 'icbm', 'China', 1, 'solid_composite',
         '', round(PROP1), '', DIAMETER_M, round(L1, 2), BURN1_S, ISP_VAC_S,
         round(THRUST1_N), 'YES', '', 'true_gravity_turn',
         '', '', 0, 'jl2_rv', 'reentry', '', CITATION, '', ''],
        ['jl2', 'JL-2 (SLBM)', '', '', 2, 'solid_composite',
         '', round(PROP2), '', DIAMETER_M, round(L2, 2), '', ISP_VAC_S,
         '', 'YES', '', 'true_gravity_turn', '', '', 0, '', '', '', '', '', ''],
        # Stage 3 not measurable in the parade photo — blanks are deliberate
        # (ingest resolver estimates and flags VERIFY).
        ['jl2', 'JL-2 (SLBM)', '', '', 3, 'solid_composite',
         '', '', '', '', '', '', '', '', 'YES', '', 'true_gravity_turn',
         '', '', 0, '', '', '', '', '', ''],
    ]
    cat.PAYLOADS_ROWS = [
        ['jl2_rv', 'JL-2 RV', 'ballistic', '', '', 'cone', '', '', '',
         '', '', '', '', 'NO', '', '', '', '', CITATION, '', ''],
    ]
    cat.OUT = path
    cat.main()


if __name__ == '__main__':
    print(f'scale {SCALE*1000:.3f} mm/px, dome h {DOME_H:.3f} m, dia {DIAMETER_M} m')
    for tag, L, p, w, d in (('S1', L1, PROP1, WET1, DRY1), ('S2', L2, PROP2, WET2, DRY2)):
        print(f'  {tag}: length {L:.2f} m, prop {p/1000:.2f} t, wet {w/1000:.2f} t, dry {d/1000:.2f} t')
    print(f'  S1 thrust(vac) {THRUST1_N/1000:.0f} kN, A_e {AE1_M2:.2f} m² '
          f'(ground Isp check: {ISP_VAC_S - PA_SL*AE1_M2/(MDOT1*G0):.1f} s)')
    out = os.path.dirname(os.path.abspath(__file__))
    build_detailed(os.path.join(out, 'JL-2.xlsx'))
    build_catalog(os.path.join(out, 'JL-2_catalog.xlsx'))
    print('wrote JL-2.xlsx and JL-2_catalog.xlsx')
