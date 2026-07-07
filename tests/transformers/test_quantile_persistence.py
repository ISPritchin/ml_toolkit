import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('quantile_persistence', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_above_med_from_docstring():
    # [10,20,30,40,50,60] w=6: честная медиана = (30+40)/2 = 35
    # values > 35: {40,50,60} → above_med=3/6=0.5
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'above_med_w6')[-1] == pytest.approx(3 / 6, abs=0.01)


def test_monotone_rank_trend_positive():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'rank_trend_w6')[-1] > 0


def test_constant_series_high_q_stability():
    # All values equal → CV of ranks≈0 → q_stability≈1
    arrs, sfxs = _run([30] * 12, {'windows': [12]})
    assert _get(arrs, sfxs, 'q_stability_w12')[-1] > 0.5


def test_bot_q_high_for_always_low_values():
    # All same value → each ≤ p25 → bot_q=1.0 when all are equal to p25
    arrs, sfxs = _run([10] * 6, {'windows': [6]})
    assert _get(arrs, sfxs, 'bot_q_w6')[-1] == pytest.approx(1.0, abs=0.1)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # окно [10,0,60,0,0,35]: честная медиана = (0+10)/2 = 5 → выше: {10,60,35} = 0.5
    assert math.isfinite(_get(arrs, sfxs, 'above_med_w6')[-1]), 'above_med_w6 must be finite'
    assert _get(arrs, sfxs, 'above_med_w6')[-1] == pytest.approx(0.5, rel=1e-4)
    # p75 = sorted[int(0.75*5)] = sorted[3] = 10 → >= 10: {10,60,35} = 0.5
    assert math.isfinite(_get(arrs, sfxs, 'top_q_w6')[-1]), 'top_q_w6 must be finite'
    assert _get(arrs, sfxs, 'top_q_w6')[-1] == pytest.approx(0.5, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'bot_q_w6')[-1]), 'bot_q_w6 must be finite'
    assert _get(arrs, sfxs, 'bot_q_w6')[-1] == pytest.approx(0.5, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'rank_trend_w6')[-1]), 'rank_trend_w6 must be finite'
    assert _get(arrs, sfxs, 'rank_trend_w6')[-1] == pytest.approx(0.16666666658333334, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'q_stability_w6')[-1]), 'q_stability_w6 must be finite'
    assert _get(arrs, sfxs, 'q_stability_w6')[-1] == pytest.approx(0.7113248658381998, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'above_ewma_w6')[-1]), 'above_ewma_w6 must be finite'
    assert _get(arrs, sfxs, 'above_ewma_w6')[-1] == pytest.approx(0.3333333333333333, rel=1e-4)
