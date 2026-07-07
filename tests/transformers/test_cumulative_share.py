import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('cumulative_share', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_share():
    # [10,20,30,40]: cum_sum=100, last share=40/100=0.4
    arrs, sfxs = _run([10, 20, 30, 40])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(0.4, abs=1e-4)


def test_single_spike_after_zeros():
    # [0,0,0,10]: cum_sum=10, share=10/10=1.0
    arrs, sfxs = _run([0, 0, 0, 10])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(1.0, abs=1e-4)


def test_uniform_series_decreasing_share():
    # Uniform [10,10,10,10]: shares = 1, 1/2, 1/3, 1/4
    arrs, sfxs = _run([10, 10, 10, 10])
    result = _get(arrs, sfxs, '')
    assert result[0] == pytest.approx(1.0, abs=1e-4)
    assert result[1] == pytest.approx(0.5, abs=1e-4)
    assert result[2] == pytest.approx(1 / 3, abs=1e-4)
    assert result[3] == pytest.approx(0.25, abs=1e-4)


def test_all_zeros_share_near_zero():
    arrs, sfxs = _run([0, 0, 0, 0])
    # 0/(|0|+EPS) ≈ 0
    assert abs(_get(arrs, sfxs, '')[-1]) < 1e-3

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, '')[-1]), ' must be finite'
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(0.10769230769197634, rel=1e-4)
