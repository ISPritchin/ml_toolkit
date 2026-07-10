import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('log_level', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_value():
    # [10,20,30,40,50,60] w=6: mean=35, log_level=log1p(35)=ln(36)
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(math.log1p(35), abs=1e-4)


def test_all_zeros_log_level_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_constant_series_log_level():
    # [50]*6: mean=50, log_level=log1p(50)
    arrs, sfxs = _run([50, 50, 50, 50, 50, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(math.log1p(50), abs=1e-4)


def test_log_level_increases_with_mean():
    # Higher mean → higher log_level
    arrs1, sfxs1 = _run([10] * 6, {'windows': [6]})
    arrs2, sfxs2 = _run([100] * 6, {'windows': [6]})
    assert _get(arrs1, sfxs1, 'w6')[-1] < _get(arrs2, sfxs2, 'w6')[-1]

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(2.917770732084279, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([1.94591, 1.386294, 1.94591, 2.047693, 1.832581, 2.302585, 2.079442, 1.7492, 2.374906], abs=1e-6)
