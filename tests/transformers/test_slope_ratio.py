import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('slope_ratio', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_acceleration_ratio_greater_than_one():
    # Rapid growth at end → short slope > long slope → ratio > 1
    arrs, sfxs = _run([10, 12, 14, 20, 30, 45], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(12.5 / 6.714, abs=0.05)


def test_constant_series_ratio_near_zero():
    # slope_short=0, slope_long=0 → ratio = 0/(0+EPS) ≈ 0
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'pairs': [[3, 6]]})
    assert abs(_get(arrs, sfxs, 'w3_w6')[-1]) < 1e-3


def test_uniform_slope_ratio_near_one():
    # Same slope in both windows
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(1.0, abs=0.05)


def test_negative_ratio_when_trends_diverge():
    # Short window declining, long window flat → ratio < 0
    arrs, sfxs = _run([30, 30, 30, 50, 40, 30], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] < 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert math.isfinite(_get(arrs, sfxs, 'w3_w6')[-1]), 'w3_w6 must be finite'
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(9.423076918002957, rel=1e-4)


def test_full_output_vector():
    # 10 значений, params={'pairs': [[3, 6]]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6') == pytest.approx([0.0, -1.0, 1.0, 2.142857, -20.0, 2.5, 3.5, -4.375, 6.363636, 2.916667], abs=1e-6)
