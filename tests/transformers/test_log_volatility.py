import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('log_volatility', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_constant_log_diffs_zero_vol():
    # [0, e-1, e²-1, e³-1]: log1p values = 0,1,2,3 → all diffs=1, std=0
    e1, e2, e3 = math.e - 1, math.e ** 2 - 1, math.e ** 3 - 1
    arrs, sfxs = _run([0, e1, e2, e3], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(0.0, abs=1e-6)


def test_constant_series_zero_vol():
    arrs, sfxs = _run([50, 50, 50, 50, 50, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_all_zeros_zero_vol():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_volatile_series_positive_vol():
    # Large alternating log-jumps → high volatility
    arrs, sfxs = _run([10, 1000, 10, 1000, 10, 1000], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] > 1.0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(3.228279321265595, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 0.0, 2.25543, 1.861179, 1.995804, 2.085155, 2.174237, 1.968964, 2.096638], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs_plain, sfxs_plain = _run(values, {'windows': [6]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [6], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['w6']
    assert arrs_plain[0].tolist() == arrs_explicit[0].tolist()


def test_dilation_2_reads_every_other_point():
    # even indices carry log1p values 0,1,2,3 (constant diff=1 -> vol=0), odd indices are
    # noise the dilation=2 step must skip entirely
    e1, e2, e3 = math.e - 1, math.e**2 - 1, math.e**3 - 1
    values = [0, 999, e1, 999, e2, 999, e3]
    arrs, sfxs = _run(values, {'windows': [4], 'dilations': [1, 2]})
    assert sfxs == ['w4', 'w4_d2']
    assert _get(arrs, sfxs, 'w4_d2')[-1] == pytest.approx(0.0, abs=1e-6)
