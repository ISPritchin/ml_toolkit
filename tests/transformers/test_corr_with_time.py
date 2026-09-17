import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('corr_with_time', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_perfect_linear_corr_one():
    # Perfectly linear series → Pearson(time_indices, values) = 1.0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_perfect_negative_linear_corr_minus_one():
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(-1.0, abs=1e-4)


def test_constant_series_corr_zero():
    # std=0 → correlation undefined → 0 by convention
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert abs(_get(arrs, sfxs, 'w6')[-1]) < 1e-4


def test_all_zeros_corr_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert abs(_get(arrs, sfxs, 'w6')[-1]) < 1e-4


def test_requires_min_window_of_three():
    # ws < 3 → result should be 0 (not enough for Pearson)
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    # first 2 rows: ws=1 and ws=2 → 0
    assert _get(arrs, sfxs, 'w6')[0] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'w6')[1] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.13981728140845512, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 0.0, 0.5, 0.52915, -0.062622, 0.0, 0.0, -0.200331, 0.152323], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [10, 20, 30, 40, 50, 60]
    arrs_plain, sfxs_plain = _run(values, {'windows': [6]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [6], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['w6']
    assert arrs_plain[0].tolist() == arrs_explicit[0].tolist()


def test_dilation_2_picks_up_strided_linear_trend():
    # every-other-point sub-series [10,30,50,70] is perfectly linear -> r=1.0, even though
    # the contiguous window (with the interleaved noise) is not
    values = [10, -999, 30, -999, 50, -999, 70]
    arrs, sfxs = _run(values, {'windows': [4], 'dilations': [1, 2]})
    assert sfxs == ['w4', 'w4_d2']
    assert _get(arrs, sfxs, 'w4_d2')[-1] == pytest.approx(1.0, abs=1e-4)
