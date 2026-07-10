import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('cross_window_momentum', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_all_accel_for_geometric_growth():
    # 12 values of geometric growth: each level strictly above prior windows
    values = [2 ** i for i in range(12)]  # 1,2,4,...,2048
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'all_accel')[-1] == pytest.approx(1.0)


def test_all_decel_for_geometric_decay():
    # 12 values of geometric decay
    values = [2 ** (11 - i) for i in range(12)]  # 2048,...,1
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'all_decel')[-1] == pytest.approx(1.0)


def test_ratio_w1_w3_known():
    # [10,20,30,40,50,60]: v=60, mean_w3=50 → ratio=60/50=1.2
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60])
    assert _get(arrs, sfxs, 'ratio_w1_w3')[-1] == pytest.approx(1.2, abs=1e-4)


def test_constant_series_ratios_near_one():
    arrs, sfxs = _run([30] * 12)
    assert _get(arrs, sfxs, 'ratio_w1_w3')[-1] == pytest.approx(1.0, abs=1e-3)
    assert _get(arrs, sfxs, 'ratio_w3_w6')[-1] == pytest.approx(1.0, abs=1e-3)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert math.isfinite(_get(arrs, sfxs, 'ratio_w1_w3')[-1]), 'ratio_w1_w3 must be finite'
    assert _get(arrs, sfxs, 'ratio_w1_w3')[-1] == pytest.approx(3.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'ratio_w3_w6')[-1]), 'ratio_w3_w6 must be finite'
    assert _get(arrs, sfxs, 'ratio_w3_w6')[-1] == pytest.approx(0.6666666666285714, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'ratio_w6_w24')[-1]), 'ratio_w6_w24 must be finite'
    assert _get(arrs, sfxs, 'ratio_w6_w24')[-1] == pytest.approx(0.8076923076550295, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'all_accel')[-1]), 'all_accel must be finite'
    assert _get(arrs, sfxs, 'all_accel')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'all_decel')[-1]), 'all_decel must be finite'
    assert _get(arrs, sfxs, 'all_decel')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'horizon_spread')[-1]), 'horizon_spread must be finite'
    assert _get(arrs, sfxs, 'horizon_spread')[-1] == pytest.approx(-0.6190392065952346, rel=1e-4)


def test_full_output_vector():
    # 26 значений, params={}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11, 0, 18, 7, 25, 0, 0, 14, 30, 5, 0, 22, 16, 0, 9, 28, 3]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'ratio_w1_w3') == pytest.approx([1.0, 0.0, 2.0, 1.285714, 0.0, 1.875, 0.631579, 0.0, 2.5, 1.064516, 0.0, 1.862069, 0.84, 1.5, 0.0, 0.0, 3.0, 2.045455, 0.306122, 0.0, 2.444444, 1.263158, 0.0, 1.08, 2.27027, 0.225], abs=1e-6)
    assert _get(arrs, sfxs, 'ratio_w3_w6') == pytest.approx([1.0, 1.0, 1.0, 1.037037, 1.296296, 1.142857, 0.95, 0.95, 1.0, 1.24, 1.24, 1.09434, 0.892857, 1.234568, 1.04918, 1.0, 0.4375, 1.157895, 1.324324, 1.428571, 0.760563, 0.873563, 1.041096, 0.961538, 0.986667, 1.025641], abs=1e-6)
    assert _get(arrs, sfxs, 'ratio_w6_w24') == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.014493, 1.15942, 1.090909, 1.082251, 1.190476, 1.115789, 1.189542, 1.488189, 1.200787, 1.049869, 1.286052, 1.333333, 1.331439, 0.92803, 1.255051, 1.490654, 1.307632, 0.932735, 1.22449, 1.258065], abs=1e-6)
    assert _get(arrs, sfxs, 'all_accel') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'all_decel') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'horizon_spread') == pytest.approx([0.0, 0.0, 0.0, 0.036368, 0.259511, 0.133531, -0.036905, 0.096627, 0.087011, 0.294155, 0.389465, 0.199713, 0.06024, 0.608281, 0.230987, 0.048665, -0.575102, 0.434286, 0.567163, 0.281984, -0.04652, 0.26404, 0.308492, -0.108854, 0.189101, 0.254892], abs=1e-6)
