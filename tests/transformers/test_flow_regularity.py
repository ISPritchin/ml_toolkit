import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("flow_regularity", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_gap_mean_and_cv():
    # [5,0,0,8,0,0] w=6: one gap of length 2 → gap_mean=2.0, gap_std=0.0, gap_cv=0.0
    arrs, sfxs = _run([5, 0, 0, 8, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "gap_mean_w6")[-1] == pytest.approx(2.0, abs=1e-4)
    assert _get(arrs, sfxs, "gap_cv_w6")[-1] == pytest.approx(0.0, abs=1e-4)


def test_continuous_stream_is_monthly():
    # All nonzero → is_monthly=1
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120], {"windows": [12]})
    assert _get(arrs, sfxs, "is_monthly_w12")[-1] == pytest.approx(1.0)


def test_all_zeros_no_gaps_no_bursts():
    # No active months → gap_mean=0 (no gaps recorded)
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "gap_mean_w6")[-1] == pytest.approx(0.0)


def test_variable_gaps_positive_cv():
    # [10,0,10,0,0,0,10,0] w=8: gaps of length 1 and 3 → cv > 0
    arrs, sfxs = _run([10, 0, 10, 0, 0, 0, 10, 0], {"windows": [8]})
    assert _get(arrs, sfxs, "gap_cv_w8")[-1] > 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'gap_mean_w6')[-1]), 'gap_mean_w6 must be finite'
    assert _get(arrs, sfxs, 'gap_mean_w6')[-1] == pytest.approx(1.5, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'gap_std_w6')[-1]), 'gap_std_w6 must be finite'
    assert _get(arrs, sfxs, 'gap_std_w6')[-1] == pytest.approx(0.5, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'gap_cv_w6')[-1]), 'gap_cv_w6 must be finite'
    assert _get(arrs, sfxs, 'gap_cv_w6')[-1] == pytest.approx(0.3333333331111111, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'is_monthly_w6')[-1]), 'is_monthly_w6 must be finite'
    assert _get(arrs, sfxs, 'is_monthly_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'cadence_shift_w6')[-1]), 'cadence_shift_w6 must be finite'
    assert _get(arrs, sfxs, 'cadence_shift_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'active_len_cv_w6')[-1]), 'active_len_cv_w6 must be finite'
    assert _get(arrs, sfxs, 'active_len_cv_w6')[-1] == pytest.approx(0.0, abs=1e-6)
