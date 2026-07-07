import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('trend_flip', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_flip_from_docstring():
    # [10,20,30,40,50,60,55,45,35,25,15,5] pair(lag=6,w=6)
    # slope_now(last 6: 60→5) = -10, slope_ago(first 6: 10→60) = +10 → flag=1, change=-20
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60, 55, 45, 35, 25, 15, 5],
                      {'lag_window_pairs': [[6, 6]]})
    assert _get(arrs, sfxs, 'flag')[-1] == pytest.approx(1.0)
    assert _get(arrs, sfxs, 'slope_change_lag6_w6')[-1] == pytest.approx(-20.0, abs=1e-3)


def test_no_flip_for_monotone_series():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120],
                      {'lag_window_pairs': [[6, 6]]})
    assert _get(arrs, sfxs, 'flag')[-1] == pytest.approx(0.0)


def test_no_flip_before_lag_available():
    # pos < lag=6 → flag=0
    arrs, sfxs = _run([10, 20, 30, 40], {'lag_window_pairs': [[6, 6]]})
    for v in _get(arrs, sfxs, 'flag'):
        assert v == pytest.approx(0.0)


def test_positive_slope_change_when_trend_accelerates():
    # Growing faster recently → slope_now > slope_ago → change > 0
    values = [10, 10, 10, 10, 10, 10, 20, 40, 80, 160, 320, 640]
    arrs, sfxs = _run(values, {'lag_window_pairs': [[6, 6]]})
    assert _get(arrs, sfxs, 'slope_change_lag6_w6')[-1] > 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'lag_window_pairs': [[6, 6]]})
    assert math.isfinite(_get(arrs, sfxs, 'flag')[-1]), 'flag must be finite'
    assert _get(arrs, sfxs, 'flag')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'slope_change_lag6_w6')[-1]), 'slope_change_lag6_w6 must be finite'
    assert _get(arrs, sfxs, 'slope_change_lag6_w6')[-1] == pytest.approx(9.285714285714286, rel=1e-4)
