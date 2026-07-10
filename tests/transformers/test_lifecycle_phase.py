import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('lifecycle_phase', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_ascending_phase_maturity_and_new_peak():
    # [10,20,30,40,50,60]: always new peak → at end: completeness=1.0, phase=1, is_new_peak=1
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'completeness')[-1] == pytest.approx(1.0, abs=1e-4)
    assert _get(arrs, sfxs, 'phase_flag')[-1] == pytest.approx(1.0)
    assert _get(arrs, sfxs, 'is_new_peak')[-1] == pytest.approx(1.0)


def test_decline_after_peak_phase_2():
    # [10,20,50,40,30,20]: peak at pos=2 → after that, v < max, completeness<0.8 → phase=2
    arrs, sfxs = _run([10, 20, 50, 40, 30, 20], {'windows': [6]})
    assert _get(arrs, sfxs, 'phase_flag')[-1] == pytest.approx(2.0)
    assert _get(arrs, sfxs, 'completeness')[-1] == pytest.approx(20 / 50, abs=1e-4)


def test_all_zeros_completeness_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'completeness')[-1] == pytest.approx(0.0, abs=1e-4)


def test_is_new_peak_only_at_peaks():
    # [10,5,15,8,12,20]: new peaks at pos 0,2,5
    arrs, sfxs = _run([10, 5, 15, 8, 12, 20], {'windows': [6]})
    flags = _get(arrs, sfxs, 'is_new_peak')
    assert flags[0] == pytest.approx(1.0)
    assert flags[1] == pytest.approx(0.0)
    assert flags[2] == pytest.approx(1.0)
    assert flags[5] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'peak_age_share')[-1]), 'peak_age_share must be finite'
    assert _get(arrs, sfxs, 'peak_age_share')[-1] == pytest.approx(0.2, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'post_peak_share')[-1]), 'post_peak_share must be finite'
    assert _get(arrs, sfxs, 'post_peak_share')[-1] == pytest.approx(0.8, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'completeness')[-1]), 'completeness must be finite'
    assert _get(arrs, sfxs, 'completeness')[-1] == pytest.approx(0.4374999999945312, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'ramp_norm')[-1]), 'ramp_norm must be finite'
    assert _get(arrs, sfxs, 'ramp_norm')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'is_new_peak')[-1]), 'is_new_peak must be finite'
    assert _get(arrs, sfxs, 'is_new_peak')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'phase_flag')[-1]), 'phase_flag must be finite'
    assert _get(arrs, sfxs, 'phase_flag')[-1] == pytest.approx(2.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'post_peak_slope_w6')[-1]), 'post_peak_slope_w6 must be finite'
    assert _get(arrs, sfxs, 'post_peak_slope_w6')[-1] == pytest.approx(-1.8571428571428572, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'peak_age_share') == pytest.approx([0.0, 0.0, 0.666667, 0.5, 0.4, 0.833333, 0.714286, 0.625, 0.888889], abs=1e-6)
    assert _get(arrs, sfxs, 'post_peak_share') == pytest.approx([1.0, 1.0, 0.333333, 0.5, 0.6, 0.166667, 0.285714, 0.375, 0.111111], abs=1e-6)
    assert _get(arrs, sfxs, 'completeness') == pytest.approx([1.0, 0.0, 1.0, 0.75, 0.0, 1.0, 0.266667, 0.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'ramp_norm') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'is_new_peak') == pytest.approx([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'phase_flag') == pytest.approx([1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'post_peak_slope_w4') == pytest.approx([0.0, 6.0, 3.0, -2.1, 0.3, 0.0, -0.0, 1.1, 1.1], abs=1e-6)
