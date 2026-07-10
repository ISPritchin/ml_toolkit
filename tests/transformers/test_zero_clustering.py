import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('zero_clustering', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_values_from_docstring():
    # [10,0,0,10,0,10] w=6: runs idx1-2(len2), idx4(len1) → max_run=2, run_count=2
    # last zero at offset4 → last_zero_rec=5-1-4=0 (offset 4 means w-1-offset=4)
    arrs, sfxs = _run([10, 0, 0, 10, 0, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'max_zero_run_w6')[-1] == pytest.approx(2.0)
    assert _get(arrs, sfxs, 'zero_run_count_w6')[-1] == pytest.approx(2.0)


def test_no_zeros_last_zero_rec_equals_window():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'max_zero_run_w6')[-1] == pytest.approx(0.0)
    # No zeros → last_zero_ago = ws = 6
    assert _get(arrs, sfxs, 'last_zero_rec_w6')[-1] == pytest.approx(6.0)


def test_zero_after_active_flag():
    # [10,0]: v[0]=10, v[1]=0 → zero_after_active=1 at row 1
    arrs, sfxs = _run([10, 0], {'windows': [2]})
    assert _get(arrs, sfxs, 'zero_after_active')[-1] == pytest.approx(1.0)


def test_zero_after_active_not_set_for_active():
    # [0,10]: v[1]=10 ≠ 0 → flag=0
    arrs, sfxs = _run([0, 10], {'windows': [2]})
    assert _get(arrs, sfxs, 'zero_after_active')[-1] == pytest.approx(0.0)


def test_long_run_detected():
    arrs, sfxs = _run([10, 0, 0, 0, 0, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'max_zero_run_w6')[-1] == pytest.approx(4.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'max_zero_run_w6')[-1]), 'max_zero_run_w6 must be finite'
    assert _get(arrs, sfxs, 'max_zero_run_w6')[-1] == pytest.approx(2.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'zero_run_count_w6')[-1]), 'zero_run_count_w6 must be finite'
    assert _get(arrs, sfxs, 'zero_run_count_w6')[-1] == pytest.approx(2.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'recent_vs_long_w6')[-1]), 'recent_vs_long_w6 must be finite'
    assert _get(arrs, sfxs, 'recent_vs_long_w6')[-1] == pytest.approx(1.3333333306666666, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'last_zero_rec_w6')[-1]), 'last_zero_rec_w6 must be finite'
    assert _get(arrs, sfxs, 'last_zero_rec_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'front_back_w6')[-1]), 'front_back_w6 must be finite'
    assert _get(arrs, sfxs, 'front_back_w6')[-1] == pytest.approx(0.49999999925000005, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'zero_after_active')[-1]), 'zero_after_active must be finite'
    assert _get(arrs, sfxs, 'zero_after_active')[-1] == pytest.approx(0.0, abs=1e-6)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'max_zero_run_w4') == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'zero_run_count_w4') == pytest.approx([0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'recent_vs_long_w4') == pytest.approx([0.0, 1.0, 1.0, 1.333333, 0.666667, 1.333333, 1.333333, 0.666667, 1.333333], abs=1e-6)
    assert _get(arrs, sfxs, 'last_zero_rec_w4') == pytest.approx([1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'front_back_w4') == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'zero_after_active') == pytest.approx([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], abs=1e-6)
