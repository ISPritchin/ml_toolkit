import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('skew_proxy', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_right_skew():
    # [10,10,10,10,10,40] w=6: lo=10, hi=40, mean=15
    # skew = (15-10)/(40-10) = 5/30 ≈ 0.1667
    arrs, sfxs = _run([10, 10, 10, 10, 10, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(5 / 30, abs=1e-4)


def test_symmetric_value_half():
    # Symmetric around midpoint → skew≈0.5
    # [0,10,20,30,40,50]: lo=0,hi=50,mean=25 → (25-0)/50=0.5
    arrs, sfxs = _run([0, 10, 20, 30, 40, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.5, abs=1e-4)


def test_constant_series_value_near_zero():
    # lo=hi=mean=20 → (20-20)/(20-20+EPS)≈0
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {'windows': [6]})
    assert abs(_get(arrs, sfxs, 'w6')[-1]) < 1e-4


def test_all_zeros_value_near_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert abs(_get(arrs, sfxs, 'w6')[-1]) < 1e-4


def test_left_skew_value_above_half():
    # [0,40,40,40,40,40] w=6: lo=0,hi=40,mean=200/6≈33.33
    # skew=(33.33-0)/40≈0.833
    arrs, sfxs = _run([0, 40, 40, 40, 40, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(200 / 6 / 40, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.2916666666618056, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 0.5, 0.5, 0.5625, 0.4375, 0.6, 0.466667, 0.316667, 0.4875], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [10, 10, 10, 10, 10, 40]
    arrs_plain, sfxs_plain = _run(values, {'windows': [6]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [6], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['w6']
    assert arrs_plain[0].tolist() == arrs_explicit[0].tolist()


def test_dilation_2_reads_every_other_point():
    # row_idx=5 (last), ws=3, dilation=2 -> idx 1,3,5 -> [10,10,40], mean=20, lo=10, hi=40
    values = [999, 10, 999, 10, 999, 40]
    arrs, sfxs = _run(values, {'windows': [3], 'dilations': [1, 2]})
    assert sfxs == ['w3', 'w3_d2']
    assert _get(arrs, sfxs, 'w3_d2')[-1] == pytest.approx((20 - 10) / (40 - 10), abs=1e-6)
