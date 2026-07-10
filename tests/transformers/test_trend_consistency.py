import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('trend_consistency', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_perfect_linear_trend_all_ones_from_docstring():
    # [10,20,30,40,50,60]: all diffs same sign as slope → dir_consistency=1.0, r²=1.0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'dir_consistency_w6')[-1] == pytest.approx(1.0, abs=1e-4)
    assert _get(arrs, sfxs, 'r_squared_w6')[-1] == pytest.approx(1.0, abs=1e-4)
    assert _get(arrs, sfxs, 'clean_streak_w6')[-1] == pytest.approx(5.0)


def test_constant_series_dir_consistency_zero():
    # slope=0 → no direction → dir_consistency=0
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'dir_consistency_w6')[-1] == pytest.approx(0.0)


def test_descending_trend_dir_consistency_one():
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'dir_consistency_w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_r_squared_near_zero_for_noisy_series():
    # Alternating series: slope≈0, residuals large → r²≈0 or negative
    arrs, sfxs = _run([10, 90, 10, 90, 10, 90], {'windows': [6]})
    assert _get(arrs, sfxs, 'r_squared_w6')[-1] < 0.1

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'dir_consistency_w6')[-1]), 'dir_consistency_w6 must be finite'
    assert _get(arrs, sfxs, 'dir_consistency_w6')[-1] == pytest.approx(0.4, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'noise_signal_w6')[-1]), 'noise_signal_w6 must be finite'
    assert _get(arrs, sfxs, 'noise_signal_w6')[-1] == pytest.approx(2.015785284364629, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'clean_streak_w6')[-1]), 'clean_streak_w6 must be finite'
    assert _get(arrs, sfxs, 'clean_streak_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'sub_sign_consist_w6')[-1]), 'sub_sign_consist_w6 must be finite'
    assert _get(arrs, sfxs, 'sub_sign_consist_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'r_squared_w6')[-1]), 'r_squared_w6 must be finite'
    assert _get(arrs, sfxs, 'r_squared_w6')[-1] == pytest.approx(0.019548872180768728, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'dir_consistency_w4') == pytest.approx([0.0, 1.0, 0.5, 0.333333, 0.666667, 0.0, 0.0, 0.666667, 0.333333], abs=1e-6)
    assert _get(arrs, sfxs, 'noise_signal_w4') == pytest.approx([0.0, 0.0, 0.471405, 0.448211, 4.454632, 0.0, 0.0, 1.366947, 1.813559], abs=1e-6)
    assert _get(arrs, sfxs, 'clean_streak_w4') == pytest.approx([0.0, 1.0, 1.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'sub_sign_consist_w4') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'r_squared_w4') == pytest.approx([1.0, 1.0, 0.25, 0.28, 0.003922, 0.0, 0.0, 0.040133, 0.023202], abs=1e-6)
