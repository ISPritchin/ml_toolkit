import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('extreme_events', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_spike_detected_from_docstring():
    # [10,10,10,10,10,10,100] w=7: z[6]=(100-22.857)/31.493≈2.449 → spike_count=1, is_spike_now=1
    values = [10, 10, 10, 10, 10, 10, 100]
    arrs, sfxs = _run(values, {'windows': [7]})
    assert _get(arrs, sfxs, 'spike_count_w7')[-1] == pytest.approx(1.0)
    assert _get(arrs, sfxs, 'is_spike_now')[-1] == pytest.approx(1.0)


def test_no_spikes_in_constant_series():
    arrs, sfxs = _run([30] * 6, {'windows': [6]})
    assert _get(arrs, sfxs, 'spike_count_w6')[-1] == pytest.approx(0.0)


def test_crash_detected_on_large_drop():
    # [100,10]: drop=(100-10)/100=0.9 > 0.5 → crash_count=1
    arrs, sfxs = _run([100, 10], {'windows': [2]})
    assert _get(arrs, sfxs, 'crash_count_w2')[-1] == pytest.approx(1.0)


def test_no_crash_for_moderate_drop():
    # [100,60]: drop=40/100=0.4 < 0.5 → no crash
    arrs, sfxs = _run([100, 60], {'windows': [2]})
    assert _get(arrs, sfxs, 'crash_count_w2')[-1] == pytest.approx(0.0)


def test_balance_positive_when_more_spikes():
    arrs, sfxs = _run([10, 10, 10, 10, 10, 10, 100], {'windows': [7]})
    assert _get(arrs, sfxs, 'balance_w7')[-1] >= 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'spike_count_w6')[-1]), 'spike_count_w6 must be finite'
    assert _get(arrs, sfxs, 'spike_count_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'max_spike_z_w6')[-1]), 'max_spike_z_w6 must be finite'
    assert _get(arrs, sfxs, 'max_spike_z_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'crash_count_w6')[-1]), 'crash_count_w6 must be finite'
    assert _get(arrs, sfxs, 'crash_count_w6')[-1] == pytest.approx(2.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'max_drop_w6')[-1]), 'max_drop_w6 must be finite'
    assert _get(arrs, sfxs, 'max_drop_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'recency_w6')[-1]), 'recency_w6 must be finite'
    assert _get(arrs, sfxs, 'recency_w6')[-1] == pytest.approx(2.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'balance_w6')[-1]), 'balance_w6 must be finite'
    assert _get(arrs, sfxs, 'balance_w6')[-1] == pytest.approx(-2.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'is_spike_now')[-1]), 'is_spike_now must be finite'
    assert _get(arrs, sfxs, 'is_spike_now')[-1] == pytest.approx(0.0, abs=1e-6)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'spike_count_w4') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'max_spike_z_w4') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'crash_count_w4') == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0], abs=1e-6)
    assert _get(arrs, sfxs, 'max_drop_w4') == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'recency_w4') == pytest.approx([1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0], abs=1e-6)
    assert _get(arrs, sfxs, 'balance_w4') == pytest.approx([0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -2.0, -2.0, -2.0], abs=1e-6)
    assert _get(arrs, sfxs, 'is_spike_now') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
