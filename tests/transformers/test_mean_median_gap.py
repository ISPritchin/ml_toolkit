import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('mean_median_gap', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_right_skew():
    # [10,10,10,10,10,40] w=6: sorted=[10,10,10,10,10,40], median=(10+10)/2=10
    # mean=15; gap=(15-10)/15=1/3
    arrs, sfxs = _run([10, 10, 10, 10, 10, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1 / 3, abs=1e-4)


def test_symmetric_gap_zero():
    # Symmetric series → mean==median → gap=0
    arrs, sfxs = _run([10, 20, 30, 40, 50], {'windows': [5]})
    # sorted=[10,20,30,40,50], ws=5 (odd), median=sorted[2]=30; mean=30 → gap=0
    assert _get(arrs, sfxs, 'w5')[-1] == pytest.approx(0.0, abs=1e-4)


def test_all_zeros_gap_zero():
    # mean=0, median=0, gap=(0-0)/(0+EPS)≈0
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_left_skew_negative_gap():
    # [10,40,40,40,40,40] w=6: sorted=[10,40,40,40,40,40], median=(40+40)/2=40
    # mean=(10+200)/6=210/6=35; gap=(35-40)/35=-1/7≈-0.1429
    arrs, sfxs = _run([10, 40, 40, 40, 40, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(-1 / 7, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.714285714244898, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 0.0, 0.0, -0.111111, 0.142857, -0.166667, 0.071429, 0.578947, 0.025641], abs=1e-6)
