"""Form C maneuver-load anchor dataset + demonstrated-envelope context.

The anchors are citations (BENCHMARKING.md §Form C is the record); these
tests pin the dataset's integrity and the context block's classification —
they must never soften the standing rule that every number traces to a real
source.
"""

import survivability_report as sr


def test_anchor_dataset_integrity():
    # Every record fully sourced — no bare numbers.
    assert len(sr.MANEUVER_ANCHORS) >= 4
    ids = [a['id'] for a in sr.MANEUVER_ANCHORS]
    assert len(ids) == len(set(ids))
    for a in sr.MANEUVER_ANCHORS:
        assert a['g'] > 0
        assert a['kind'] in ('textbook', 'operational_flight',
                             'qualification', 'flight_measured')
        assert a['source'].strip() and a['note'].strip()
    # The two envelope constants must be readable from the dataset itself.
    gs = {a['id']: a['g'] for a in sr.MANEUVER_ANCHORS}
    assert sr._MARV_G_OPERATIONAL == gs['Pershing-II-pullout'] == 25.0
    assert sr._MARV_G_DEMONSTRATED == gs['AMaRV-flight-100g'] == 100.0


def test_context_classification():
    def blob(g):
        return "\n".join(sr._maneuver_context(g))

    assert sr._maneuver_context(0.0) == []       # no cap → no block
    assert "operational-MaRV class" in blob(10.0)
    assert "operational-MaRV class" in blob(25.0)         # boundary inclusive
    assert "AMaRV flight-demonstrated" in blob(40.0)
    assert "AMaRV flight-demonstrated" in blob(100.0)     # ceiling inclusive
    assert "EXCEEDS every flight-demonstrated" in blob(150.0)
    # Context, not a verdict: the honesty caveat rides every block.
    for g in (10.0, 40.0, 150.0):
        assert "not thermal limits" in blob(g)


def test_regan_1984_worked_case():
    """Regan 1984 Tables 6.7/6.8: the fixed-L/D=1.5 MaRV hits its 4-g
    transverse limit at ≈45 km.  Verifies the atmosphere + glide force-law
    chain against the published table — and that Regan's β=10⁴ must be read
    in his Pa (weight) convention: taken as kg/m² it misses by ~15 km."""
    from regan1984_marv_check import fly, BETA_REGAN_PA, G0

    z_ok, _ = fly(BETA_REGAN_PA / G0, dt=0.1)
    assert z_ok is not None and abs(z_ok - 45.0e3) < 5.0e3
    z_bad, _ = fly(BETA_REGAN_PA, dt=0.1)
    assert z_bad is not None and z_bad < 35.0e3
