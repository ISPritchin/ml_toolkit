import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('rolling_std', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_value():
    # [10,10,10,10,10,40] w=6: mean=15, sum_sq_dev=750, std=sqrt(125)
    arrs, sfxs = _run([10, 10, 10, 10, 10, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(math.sqrt(125), abs=1e-4)


def test_constant_series_std_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-10)


def test_all_zeros_std_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-10)


def test_partial_window_at_start():
    # At row index 0, window=1, std must be 0
    arrs, sfxs = _run([100, 200, 300], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[0] == pytest.approx(0.0, abs=1e-10)


def test_multiple_windows():
    values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]
    arrs, sfxs = _run(values, {'windows': [6, 12]})
    # std_w6 at last row uses last 6 values [70..120], mean=95
    # biased std of arithmetic progression with step 10, length 6:
    # mean=95, deviations: -25,-15,-5,5,15,25 → sum_sq=875*2/2=1750, std=sqrt(1750/6)
    expected_w6 = math.sqrt(sum((v - 95) ** 2 for v in [70, 80, 90, 100, 110, 120]) / 6)
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(expected_w6, abs=1e-4)
    # For uniform arithmetic growth, a wider window captures a larger spread → std_w6 < std_w12
    assert _get(arrs, sfxs, 'w6')[-1] < _get(arrs, sfxs, 'w12')[-1]

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # biased std of [10,0,60,0,0,35]
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(22.684429314693666, abs=0.001)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 3.0, 4.898979, 4.43706, 5.356071, 5.612486, 5.612486, 6.139015, 8.073878], abs=1e-6)
