import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("cross_window_momentum", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_all_accel_for_geometric_growth():
    # 12 values of geometric growth: each level strictly above prior windows
    values = [2 ** i for i in range(12)]  # 1,2,4,...,2048
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, "all_accel")[-1] == pytest.approx(1.0)


def test_all_decel_for_geometric_decay():
    # 12 values of geometric decay
    values = [2 ** (11 - i) for i in range(12)]  # 2048,...,1
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, "all_decel")[-1] == pytest.approx(1.0)


def test_ratio_w1_w3_known():
    # [10,20,30,40,50,60]: v=60, mean_w3=50 → ratio=60/50=1.2
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60])
    assert _get(arrs, sfxs, "ratio_w1_w3")[-1] == pytest.approx(1.2, abs=1e-4)


def test_constant_series_ratios_near_one():
    arrs, sfxs = _run([30] * 12)
    assert _get(arrs, sfxs, "ratio_w1_w3")[-1] == pytest.approx(1.0, abs=1e-3)
    assert _get(arrs, sfxs, "ratio_w3_w6")[-1] == pytest.approx(1.0, abs=1e-3)

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
