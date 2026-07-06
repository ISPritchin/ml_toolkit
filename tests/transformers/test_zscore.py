import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("zscore", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_zscore():
    # [10,10,10,10,10,40] w=6: mean=15, std=sqrt(125), zscore=(40-15)/sqrt(125)
    expected = 25 / math.sqrt(125)
    arrs, sfxs = _run([10, 10, 10, 10, 10, 40], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(expected, abs=1e-4)


def test_constant_series_zscore_zero():
    # std=0, all values equal mean → (v-mean)/(0+EPS)≈0
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {"windows": [6]})
    assert abs(_get(arrs, sfxs, "w6")[-1]) < 1e-4


def test_below_mean_zscore_negative():
    # [40,40,40,40,40,10] w=6: last value below mean → zscore<0
    arrs, sfxs = _run([40, 40, 40, 40, 40, 10], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] < 0


def test_zscore_at_mean_is_zero():
    # If last value == mean, zscore=0
    # [10,20,30,40,50,30] w=6: mean=30, last=30 → zscore=0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0, abs=1e-4)


def test_all_zeros_zscore_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert abs(_get(arrs, sfxs, "w6")[-1]) < 1e-4

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.7714542762551692, rel=1e-4)
