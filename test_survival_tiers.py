"""Unified 4-tier survival ladder (survivability_report.survival_tier).

One verdict for every material: within experience (green) / within design
envelope (blue, permitted extrapolation) / beyond design envelope (yellow,
caution) / cannot survive (red, computed failure). Red is reserved for a
computed failure; blue is only reachable via a demonstrated envelope (the
payoff of a curated anchor dataset).
"""

import survivability_report as sr


def test_tier_palette_is_four():
    assert list(sr.SURVIVAL_TIERS) == list(sr._TIER_KEYS) == [
        'experience', 'design', 'beyond', 'fail']
    # each tier carries a (label, hex-color) pair
    for k, (label, col) in sr.SURVIVAL_TIERS.items():
        assert label and col.startswith("#") and len(col) == 7


def test_classifier_all_four_reachable():
    long_ = {'exits': {'too long'}}
    hot = {'exits': {'too hot'}}
    # green — plain survive, demonstrated
    assert sr.survival_tier('survive', None) == 'experience'
    # green — degraded means SURVIVES with a flight-demonstrated consequence
    # (recession-accuracy, aeroshape); the consequence is a headline
    # annotation, not a survival-envelope exit (Mk21 on an easy trajectory
    # must not read 'beyond design envelope').
    assert sr.survival_tier('degraded', None) == 'experience'
    # blue — past the demonstrated dwell floor but still passive (design vouches)
    assert sr.survival_tier('survive', long_) == 'design'
    # yellow — passive→active transition, or the screen cannot assess
    assert sr.survival_tier('survive', hot) == 'beyond'
    assert sr.survival_tier('analysis', None) == 'beyond'
    # red — computed failure always wins, even with coverage exits present
    assert sr.survival_tier('fail', None) == 'fail'
    assert sr.survival_tier('fail', hot) == 'fail'


def test_blue_needs_a_demonstrated_envelope():
    # 'within design' (blue) is only reachable through coverage exits — a
    # material with no curated envelope (coverage None) can never land on blue,
    # only green / yellow / red.  This is the honest tie to the data campaign.
    reachable_without_coverage = {sr.survival_tier(s, None)
                                  for s in ('survive', 'degraded', 'analysis', 'fail')}
    assert 'design' not in reachable_without_coverage
    assert reachable_without_coverage == {'experience', 'beyond', 'fail'}


def test_band_class_remap_covers_coverage_colours():
    # every per-timestep coverage-band colour maps to a survival tier
    for c in ('green', 'amber', 'red', 'redhot'):
        assert sr._BAND_CLASS_TIER[c] in sr._TIER_KEYS
    # too-long (red) → design/blue; too-hot (redhot) → beyond/yellow
    assert sr._BAND_CLASS_TIER['red'] == 'design'
    assert sr._BAND_CLASS_TIER['redhot'] == 'beyond'
