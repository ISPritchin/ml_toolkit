import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("window_mean", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_mean():
    # [10,20,30,40] w=3: last 3 = [20,30,40], mean=30
    arrs, sfxs = _run([10, 20, 30, 40], {"windows": [3]})
    assert _get(arrs, sfxs, "w3")[-1] == pytest.approx(30.0)


def test_all_zeros_mean_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0], {"windows": [3]})
    assert _get(arrs, sfxs, "w3")[-1] == pytest.approx(0.0)


def test_constant_series_mean_equals_value():
    arrs, sfxs = _run([42, 42, 42, 42], {"windows": [3]})
    assert _get(arrs, sfxs, "w3")[-1] == pytest.approx(42.0)


def test_partial_window_at_start():
    # Row 0: only 1 value available → mean=v[0]
    arrs, sfxs = _run([100, 200, 300], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[0] == pytest.approx(100.0)


def test_longer_window_includes_zeros():
    # [0,0,0,30] w=4: mean=(0+0+0+30)/4=7.5
    arrs, sfxs = _run([0, 0, 0, 30], {"windows": [4]})
    assert _get(arrs, sfxs, "w4")[-1] == pytest.approx(7.5)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # mean of [10,0,60,0,0,35]=17.5
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(17.5, abs=0.0001)
