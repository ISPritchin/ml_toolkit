import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('burstiness', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_example_from_docstring():
    # [0,40,30,0,20,0] w=6: mean=15, max=40, burst_count=2, burst_dur=1.5, zero_count=3
    # peak_mean=40/15≈2.667; burst_count=2; burst_dur=1.5; gap_mean=3/2=1.5; calm_share=0.5
    arrs, sfxs = _run([0, 40, 30, 0, 20, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'peak_mean_w6')[-1] == pytest.approx(40 / 15, abs=1e-3)
    assert _get(arrs, sfxs, 'burst_count_w6')[-1] == pytest.approx(2.0)
    assert _get(arrs, sfxs, 'burst_dur_w6')[-1] == pytest.approx(1.5, abs=1e-4)
    assert _get(arrs, sfxs, 'gap_mean_w6')[-1] == pytest.approx(1.5, abs=1e-4)
    assert _get(arrs, sfxs, 'calm_share_w6')[-1] == pytest.approx(0.5, abs=1e-4)


def test_all_zeros_all_outputs_defined():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'burst_count_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'calm_share_w6')[-1] == pytest.approx(1.0)
    assert _get(arrs, sfxs, 'gap_mean_w6')[-1] == pytest.approx(6.0)  # 6 zeros / max(0,1)=1


def test_continuous_activity_no_bursts_pattern():
    # All nonzero → one continuous "burst", zero_count=0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'calm_share_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'burst_count_w6')[-1] == pytest.approx(1.0)


def test_peak_mean_large_for_single_spike():
    # [0,0,0,0,0,100]: mean≈16.67, max=100 → peak_mean≈6
    arrs, sfxs = _run([0, 0, 0, 0, 0, 100], {'windows': [6]})
    assert _get(arrs, sfxs, 'peak_mean_w6')[-1] > 5.0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'peak_mean_w6')[-1]), 'peak_mean_w6 must be finite'
    assert _get(arrs, sfxs, 'peak_mean_w6')[-1] == pytest.approx(3.42857142837551, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'peak_med_w6')[-1]), 'peak_med_w6 must be finite'
    # честная медиана [0,0,0,10,35,60] = (0+10)/2 = 5 → 60/5 = 12
    assert _get(arrs, sfxs, 'peak_med_w6')[-1] == pytest.approx(12.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'gap_mean_w6')[-1]), 'gap_mean_w6 must be finite'
    assert _get(arrs, sfxs, 'gap_mean_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'burst_count_w6')[-1]), 'burst_count_w6 must be finite'
    assert _get(arrs, sfxs, 'burst_count_w6')[-1] == pytest.approx(3.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'burst_dur_w6')[-1]), 'burst_dur_w6 must be finite'
    assert _get(arrs, sfxs, 'burst_dur_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'burst_cv_w6')[-1]), 'burst_cv_w6 must be finite'
    assert _get(arrs, sfxs, 'burst_cv_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'calm_share_w6')[-1]), 'calm_share_w6 must be finite'
    assert _get(arrs, sfxs, 'calm_share_w6')[-1] == pytest.approx(0.5, rel=1e-4)
