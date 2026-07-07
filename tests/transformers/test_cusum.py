import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('cusum', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_cusum():
    # [10,40,20,30] w=4: mean=25
    # pos: max(0,10-25)=0, max(0,40-25)=15, max(0,20-25)=0, max(0,30-25)=5 → sum=20
    # neg: min(0,10-25)=-15, min(0,40-25)=0, min(0,20-25)=-5, min(0,30-25)=0 → sum=-20
    arrs, sfxs = _run([10, 40, 20, 30], {'windows': [4]})
    assert _get(arrs, sfxs, 'pos_w4')[-1] == pytest.approx(20.0, abs=1e-4)
    assert _get(arrs, sfxs, 'neg_w4')[-1] == pytest.approx(-20.0, abs=1e-4)


def test_constant_series_both_zero():
    arrs, sfxs = _run([25, 25, 25, 25], {'windows': [4]})
    assert _get(arrs, sfxs, 'pos_w4')[-1] == pytest.approx(0.0, abs=1e-6)
    assert _get(arrs, sfxs, 'neg_w4')[-1] == pytest.approx(0.0, abs=1e-6)


def test_all_zeros_cusum_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'pos_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert _get(arrs, sfxs, 'neg_w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_pos_always_nonneg_neg_always_nonpos():
    values = [0, 50, 0, 100, 0, 30]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert _get(arrs, sfxs, 'pos_w6')[-1] >= 0
    assert _get(arrs, sfxs, 'neg_w6')[-1] <= 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'pos_w6')[-1]), 'pos_w6 must be finite'
    assert _get(arrs, sfxs, 'pos_w6')[-1] == pytest.approx(60.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'neg_w6')[-1]), 'neg_w6 must be finite'
    assert _get(arrs, sfxs, 'neg_w6')[-1] == pytest.approx(-60.0, rel=1e-4)
