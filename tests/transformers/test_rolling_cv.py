import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('rolling_cv', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_cv_from_docstring():
    # [10,10,10,10,10,40] w=6: mean=15, std=sqrt(125)≈11.18, CV=11.18/15≈0.745
    arrs, sfxs = _run([10, 10, 10, 10, 10, 40], {'windows': [6]})
    expected_std = math.sqrt(125)
    expected_cv = expected_std / 15
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(expected_cv, abs=1e-3)


def test_constant_series_cv_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_all_zeros_cv_zero():
    # mean=0, std=0 → CV=0/(0+EPS)≈0
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-3)


def test_cv_nonneg():
    # CV is always non-negative
    arrs, sfxs = _run([10, 50, 5, 100, 3, 80], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] >= 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # cv=std/mean
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.296253103696781, abs=0.001)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 1.0, 0.816497, 0.657342, 1.020204, 0.62361, 0.801784, 1.292424, 0.82809], abs=1e-6)
