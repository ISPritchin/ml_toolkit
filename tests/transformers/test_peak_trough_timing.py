import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('peak_trough_timing', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_peak_and_trough_timing():
    # [10,80,40,20,5,30] w=6: peak at offset=1(80), trough at offset=4(5)
    # months_since_peak = (6-1)-1 = 4; months_since_trough = (6-1)-4 = 1
    arrs, sfxs = _run([10, 80, 40, 20, 5, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'peak_w6')[-1] == pytest.approx(4.0)
    assert _get(arrs, sfxs, 'trough_w6')[-1] == pytest.approx(1.0)


def test_current_is_peak_months_since_zero():
    # Monotone ascending: last value is peak → months_since_peak=0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'peak_w6')[-1] == pytest.approx(0.0)


def test_current_is_trough_months_since_zero():
    # Monotone descending: last value is trough → months_since_trough=0
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'trough_w6')[-1] == pytest.approx(0.0)


def test_peak_at_start_of_window():
    # [80,70,60,50,40,30] w=6: peak at offset=0 → months_since=(6-1)-0=5
    arrs, sfxs = _run([80, 70, 60, 50, 40, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'peak_w6')[-1] == pytest.approx(5.0)


def test_zeros_window():
    # All zeros: peak and trough both at first element → months_since = ws-1
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    # peak/trough both at offset=0 → months_since = 5
    assert _get(arrs, sfxs, 'peak_w6')[-1] == pytest.approx(5.0)
    assert _get(arrs, sfxs, 'trough_w6')[-1] == pytest.approx(5.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'peak_w6')[-1]), 'peak_w6 must be finite'
    assert _get(arrs, sfxs, 'peak_w6')[-1] == pytest.approx(3.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'trough_w6')[-1]), 'trough_w6 must be finite'
    assert _get(arrs, sfxs, 'trough_w6')[-1] == pytest.approx(4.0, rel=1e-4)
