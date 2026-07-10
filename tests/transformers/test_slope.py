import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('slope', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_value_from_docstring():
    # [10,20,30,40,50] w=5: slope = 10.0 (documented example)
    arrs, sfxs = _run([10, 20, 30, 40, 50], {'windows': [5]})
    assert _get(arrs, sfxs, 'w5')[-1] == pytest.approx(10.0, abs=1e-4)


def test_constant_series_slope_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_linear_ascending_slope_positive():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] > 0


def test_linear_descending_slope_negative():
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] < 0


def test_all_zeros_slope_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_single_point_slope_zero():
    # ws=1 → n<2 → slope=0
    arrs, sfxs = _run([100], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[0] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.8571428571428572, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, -6.0, 3.0, 2.1, -0.3, 0.0, 0.0, -1.1, 1.1], abs=1e-6)
