import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('distance_to_global_max', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_distance():
    # [10,30,20,25]: running_max=30, last=(25-30)/30≈-0.1667
    arrs, sfxs = _run([10, 30, 20, 25])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(-1 / 6, abs=1e-4)


def test_new_max_distance_zero():
    # [10,30,40,50]: last value is all-time max → distance=0
    arrs, sfxs = _run([10, 30, 40, 50])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(0.0, abs=1e-4)


def test_distance_always_non_positive():
    values = [50, 10, 30, 20, 40, 15]
    arrs, sfxs = _run(values)
    assert all(v <= 0 for v in _get(arrs, sfxs, ''))


def test_all_zeros_distance_near_zero():
    arrs, sfxs = _run([0, 0, 0, 0])
    # (0-0)/(0+EPS)=0
    assert abs(_get(arrs, sfxs, '')[-1]) < 1e-3

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, '')[-1]), ' must be finite'
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(-0.5624999999929687, rel=1e-4)
