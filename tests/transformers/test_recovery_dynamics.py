import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('recovery_dynamics', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_values_from_docstring():
    # [10,80,40,20,5,30] w=6: min=5, max=80, v=30
    # completeness=(30-5)/(80-5)=25/75=1/3
    # trough at offset 4 → months_since_trough=5-1-4=0 → speed=25/(0+1)=25
    arrs, sfxs = _run([10, 80, 40, 20, 5, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'completeness_w6')[-1] == pytest.approx(1 / 3, abs=1e-3)
    assert _get(arrs, sfxs, 'speed_w6')[-1] == pytest.approx(12.5, abs=0.5)


def test_at_max_completeness_one():
    # Monotone ascending: current = max, min = first → completeness=1
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'completeness_w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_constant_series_completeness_zero_to_small():
    # All same value → max=min=value → completeness=0/(0+EPS)≈0
    arrs, sfxs = _run([50] * 6, {'windows': [6]})
    assert _get(arrs, sfxs, 'completeness_w6')[-1] == pytest.approx(0.0, abs=1e-3)


def test_is_recovering_now_flag():
    # v[t]>v[t-1]>v[t-2] AND v[t] < mean_12 → flag=1
    # [1,50,40,30,20,10,20,30]: v grows for last 3, but mean is high
    arrs, sfxs = _run([100, 100, 100, 100, 100, 100, 10, 20, 30], {'windows': [9]})
    assert _get(arrs, sfxs, 'is_recovering_now')[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'completeness_w6')[-1]), 'completeness_w6 must be finite'
    assert _get(arrs, sfxs, 'completeness_w6')[-1] == pytest.approx(0.5833333333236111, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'drawdown_dur_w6')[-1]), 'drawdown_dur_w6 must be finite'
    assert _get(arrs, sfxs, 'drawdown_dur_w6')[-1] == pytest.approx(5.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'post_trough_gain_w6')[-1]), 'post_trough_gain_w6 must be finite'
    assert _get(arrs, sfxs, 'post_trough_gain_w6')[-1] == pytest.approx(2.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'trough_is_recent_w6')[-1]), 'trough_is_recent_w6 must be finite'
    assert _get(arrs, sfxs, 'trough_is_recent_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'speed_w6')[-1]), 'speed_w6 must be finite'
    assert _get(arrs, sfxs, 'speed_w6')[-1] == pytest.approx(7.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'is_recovering_now')[-1]), 'is_recovering_now must be finite'
    assert _get(arrs, sfxs, 'is_recovering_now')[-1] == pytest.approx(0.0, abs=1e-6)
