"""β + (L/D)max pairing consistency — `booster_models.check_beta_ld_pairing`.

β and `glider_LD` are entered independently and `trajectory._aero_polar`
back-solves whatever `k` reconciles them, so the polar can never disagree with
the user (docs/aero_polar_calibration.md §2).  The check supplies that missing
disagreement, one-sided: the swept geometry's C_D is a FLOOR, so it can say
"you cannot have less drag than this" and never "you must have this much".

The library test below is a CHARACTERISATION test, not a pass/fail gate on the
shipped data.  A flag means the stored geometry *description* cannot support
the stored pair — which is as often an incomplete description (fins with no
planform) as a wrong pair.  Recording the expected verdict per object means a
change to any β, L/D or dimension surfaces in review instead of silently
altering what the objects claim.
"""
import glob
import json
import math
import os

import pytest

import booster_models as bm

HERE = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# synthetic sweeps — exercise each verdict without depending on library data
# --------------------------------------------------------------------------
def _sweep(C_D0=0.05, k=0.7, cla=2.0, a_max_deg=25.0, n=51):
    """A clean parabolic sweep: C_L = cla·α, C_D = C_D0 + k·C_L²."""
    rows = []
    for i in range(n):
        a = math.radians(a_max_deg) * i / (n - 1)
        cl = cla * a
        rows.append(dict(alpha_deg=math.degrees(a), C_L=cl,
                         C_D=C_D0 + k * cl * cl, L_D=0.0, beta=0.0))
    star = max(rows, key=lambda r: r['C_L'] / r['C_D'])
    return dict(conditions=dict(mach=10.0, cf=0.0),
                alpha=rows,
                trim=dict(C_D0=C_D0, C_L_star=star['C_L'],
                          C_D_star=star['C_D'],
                          LD_max=star['C_L'] / star['C_D'],
                          alpha_star_deg=star['alpha_deg'], C_L0=0.0))


def _pair(sweep, mass=300.0, a_ref=0.25, beta=None, ld=2.0, C_D0=None, **kw):
    """Enter a pair by naming the C_D0 you want, rather than a β."""
    if beta is None:
        beta = mass / (float(C_D0) * a_ref)
    return bm.check_beta_ld_pairing(sweep, mass, a_ref, beta, ld, **kw)


def test_consistent_pair_is_ok_and_reports_the_implied_parasite():
    # Geometry peaks at 1/(2·sqrt(0.05·0.7)) = 2.67.  Ask for less, at the
    # geometry's own C_D0: the shortfall must show up as parasite surplus.
    r = _pair(_sweep(), C_D0=0.05, ld=2.0)
    assert r['verdict'] == 'ok'
    assert r['beta_verdict'] == 'ok' and r['trim_verdict'] == 'ok'
    assert r['implied_parasite'] > 0.0
    # C_L* and C_D* are identities of the back-solve, not fits.
    assert r['C_L_star'] == pytest.approx(2 * 0.05 * 2.0)
    assert r['C_D_star'] == pytest.approx(2 * 0.05)


def test_ld_above_what_the_shape_supports_is_flagged():
    r = _pair(_sweep(), C_D0=0.05, ld=6.0)      # geometry peaks at 2.67
    assert r['verdict'] == 'drag_below_shape'
    assert r['trim_verdict'] == 'drag_below_shape'
    assert r['C_D_geo_at_C_L_star'] > r['C_D_star']
    assert '%' in r['reason']


def test_beta_alone_can_fail_while_the_trim_point_passes():
    # Claim much less zero-lift drag than the shape has, but ask for so little
    # lift that C_D* still clears the shape's drag at that C_L.
    r = _pair(_sweep(C_D0=0.10, k=0.7), C_D0=0.05, ld=0.6)
    assert r['beta_verdict'] == 'beta_below_shape'
    assert r['trim_verdict'] in ('ok', 'marginal')
    assert r['verdict'] == 'beta_below_shape'   # the more severe of the two
    assert 'zero-lift' in r['reason']


def test_lift_beyond_the_sweep_is_its_own_verdict():
    r = _pair(_sweep(cla=0.2), C_D0=0.05, ld=8.0)
    assert r['verdict'] == 'lift_unreachable'
    assert r['C_L_star'] > r['C_L_max_geo']


def test_tolerance_band_separates_marginal_from_impossible():
    sw = _sweep()
    # Just inside the band is 'marginal'; far outside is a hard flag.
    near = _pair(sw, C_D0=0.05, ld=2.80, tol=0.15)
    far = _pair(sw, C_D0=0.05, ld=6.00, tol=0.15)
    assert near['verdict'] == 'marginal'
    assert far['verdict'] == 'drag_below_shape'
    # A zero tolerance must be at least as strict as the default.
    strict = _pair(sw, C_D0=0.05, ld=2.80, tol=0.0)
    assert strict['verdict'] == 'drag_below_shape'


def test_the_check_is_one_sided():
    """Lowering the entered L/D must never turn a pass into a failure.

    The floor omits only drag-ADDING terms, so claiming a worse glide than the
    shape supports is always physically available.
    """
    sw = _sweep()
    verdicts = [_pair(sw, C_D0=0.05, ld=ld)['trim_verdict']
                for ld in (0.2, 0.5, 1.0, 1.5, 2.0)]
    assert set(verdicts) == {'ok'}, verdicts


def test_missing_inputs_never_raise():
    sw = _sweep()
    # (mass, a_ref, beta, glider_LD) — each row zeroes one required input.
    for args in ((0.0, 0.25, 24000.0, 2.0),
                 (300.0, 0.0, 24000.0, 2.0),
                 (300.0, 0.25, 0.0, 2.0),
                 (300.0, 0.25, 24000.0, 0.0),
                 (300.0, 0.25, None, 2.0),
                 (300.0, 0.25, 'not a number', 2.0)):
        r = bm.check_beta_ld_pairing(sw, *args)
        assert r['verdict'] == 'insufficient', (args, r)
        assert r['reason']
    # A sweep with no usable rows is also survivable.
    empty = bm.check_beta_ld_pairing(dict(conditions={}, alpha=[], trim={}),
                                     300.0, 0.25, 24000.0, 2.0)
    assert empty['verdict'] == 'insufficient' 


# --------------------------------------------------------------------------
# the Mach band
# --------------------------------------------------------------------------
def test_band_is_never_stricter_than_any_single_mach():
    """The band verdict is the most permissive one, by construction.

    β is a schema constant with no stated Mach, while a shape's own β varies
    several-fold across a glide band — so a single-Mach verdict is partly an
    artefact of the Mach chosen.  Only "no Mach supports this" is defensible.
    """
    ro = bm.ro_from_dict(json.load(
        open(os.path.join(HERE, 'ro_library', 'SWERVE.ro.json'))))
    band = bm.check_ro_pairing_band(ro)
    ranks = [bm._PAIRING_RANK[v] for v in band['by_mach'].values()]
    assert bm._PAIRING_RANK[band['verdict']] == min(ranks)
    assert band['mach_best'] in band['by_mach']
    for m in band['mach_ok']:
        assert band['by_mach'][m] in ('ok', 'marginal')


def test_a_flagged_pairing_names_the_band_it_failed_over():
    ro = bm.ro_from_dict(json.load(
        open(os.path.join(HERE, 'ro_library', 'C-HGB.ro.json'))))
    band = bm.check_ro_pairing_band(ro)
    assert band['verdict'] == 'drag_below_shape'
    assert 'no Mach in' in band['reason']


# --------------------------------------------------------------------------
# shipped library — characterisation
# --------------------------------------------------------------------------
#
# Expected band verdict per shipped object, with the reason it is not 'ok'.
# Update deliberately, with the reason, when an object's β, L/D or dimensions
# change — that is the point of the table.
EXPECTED = {
    'AHW': 'marginal',
    # A 8.5° cone with 4 fins whose planform is not in the file.  The pair sits
    # just inside the band from M8 up; the bare-cone floor has no fin lift.
    'C-HGB': 'drag_below_shape',
    # A stubby 10.9° cone: as a bare cone it peaks at L/D 1.07 against the
    # stored 2.00, and its β is ~2× the bare-cone ceiling.  4 fins are declared
    # with no planform, so the modelled shape is not the described vehicle.
    # Flagged as a description gap, not a corrected value.
    'Generic-Maneuvering-Body': 'ok',
    'HTV-2': 'ok',
    'SWERVE': 'ok',
}


def _library_objects():
    for path in sorted(glob.glob(os.path.join(HERE, 'ro_library', '*.ro.json'))):
        name = os.path.basename(path)[:-len('.ro.json')]
        yield name, bm.ro_from_dict(json.load(open(path)))


def test_every_library_object_is_checkable_or_explicitly_not():
    """No object may crash the check, and any object with the geometry and the
    pair to test must have a recorded expectation."""
    for name, ro in _library_objects():
        band = bm.check_ro_pairing_band(ro)
        assert band['verdict'] in bm._PAIRING_RANK, (name, band)
        if band['verdict'] == 'insufficient':
            assert name not in EXPECTED, (
                f"{name}: has an expectation but is no longer checkable "
                f"({band['reason']})")
            assert band['reason']
        else:
            assert name in EXPECTED, (
                f"{name}: newly checkable, verdict {band['verdict']!r} — add it "
                f"to EXPECTED with the reason. {band['reason']}")


def test_library_pairings_match_their_recorded_verdicts():
    for name, ro in _library_objects():
        if name not in EXPECTED:
            continue
        band = bm.check_ro_pairing_band(ro)
        assert band['verdict'] == EXPECTED[name], (
            f"{name}: verdict changed to {band['verdict']!r} "
            f"(expected {EXPECTED[name]!r}) — {band['reason']}")


# --------------------------------------------------------------------------
# the user-facing note (pure, so the GUI needs no display to be tested)
# --------------------------------------------------------------------------
def test_note_severity_tracks_the_verdict():
    seen = {}
    for name, ro in _library_objects():
        text, sev = bm.pairing_note(ro)
        assert sev in ('none', 'ok', 'warn', 'bad')
        band = bm.check_ro_pairing_band(ro)
        if band['verdict'] == 'insufficient':
            assert (text, sev) == ('', 'none'), name
        else:
            assert text, name
            seen[name] = sev
    assert seen.get('C-HGB') == 'bad'          # flagged, and stays flagged
    assert seen.get('SWERVE') == 'ok'


def test_note_never_raises_on_junk():
    class _Junk:
        diameter_m = 'x'
        length_m = None
    text, sev = bm.pairing_note(_Junk())
    assert sev in ('none', 'bad')
    assert isinstance(text, str)


def test_note_is_advisory_only():
    """The note reports; it must not mutate the object it inspects."""
    ro = bm.ro_from_dict(json.load(
        open(os.path.join(HERE, 'ro_library', 'C-HGB.ro.json'))))
    before = (ro.beta_kg_m2, ro.glider_LD, ro.diameter_m, ro.length_m)
    bm.pairing_note(ro)
    assert (ro.beta_kg_m2, ro.glider_LD, ro.diameter_m, ro.length_m) == before
