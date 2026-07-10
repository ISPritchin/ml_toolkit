import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('lag_comparison', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_lag3_from_docstring():
    # [10,20,30,40]: lag3_ratio = 40/10 - 1 = 3.0
    arrs, sfxs = _run([10, 20, 30, 40])
    assert _get(arrs, sfxs, 'lag3_ratio')[-1] == pytest.approx(3.0, abs=1e-4)


def test_lag9_zero_before_9_periods():
    arrs, sfxs = _run([10] * 9)
    assert _get(arrs, sfxs, 'lag9_ratio')[7] == pytest.approx(0.0)


def test_constant_series_all_ratios_zero():
    arrs, sfxs = _run([50] * 15)
    assert _get(arrs, sfxs, 'lag3_ratio')[-1] == pytest.approx(0.0, abs=1e-3)
    assert _get(arrs, sfxs, 'lag12_ratio')[-1] == pytest.approx(0.0, abs=1e-3)


def test_lag12_yoy_doubling():
    # [10]*12 then [20]: last row v=20, v[t-12]=10 → lag12_ratio=20/10-1=1.0
    values = [10] * 12 + [20]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'lag12_ratio')[-1] == pytest.approx(1.0, abs=1e-4)


def test_yoy_accel_positive_when_speeding_up():
    # lag12 at current > lag12 six months ago → yoy_accel > 0
    # need 19 values (pos>=18)
    values = [10] * 7 + [20] * 6 + [40] * 6
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'yoy_accel')[-1] > 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'lag3_ratio')[-1]), 'lag3_ratio must be finite'
    assert _get(arrs, sfxs, 'lag3_ratio')[-1] == pytest.approx(-0.41666666667638885, rel=1e-4)
    # v[t-9] = 0 и v[t-12] = 0 → рост не определён → 0 (раньше ~3.5e10)
    assert math.isfinite(_get(arrs, sfxs, 'lag9_ratio')[-1]), 'lag9_ratio must be finite'
    assert _get(arrs, sfxs, 'lag9_ratio')[-1] == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(_get(arrs, sfxs, 'lag12_ratio')[-1]), 'lag12_ratio must be finite'
    assert _get(arrs, sfxs, 'lag12_ratio')[-1] == pytest.approx(0.0, abs=1e-9)
    # последние три lag3_ratio: -1 (0/10), 0 (база 0), -0.41667 (35/60) → среднее
    assert math.isfinite(_get(arrs, sfxs, 'lag3_trend')[-1]), 'lag3_trend must be finite'
    assert _get(arrs, sfxs, 'lag3_trend')[-1] == pytest.approx((-1.0 + 0.0 - 5 / 12) / 3, rel=1e-4)
    # последние три lag12_ratio: -1, -1, 0 → популяционное std = sqrt(2/9)
    assert math.isfinite(_get(arrs, sfxs, 'lag12_consistency')[-1]), 'lag12_consistency must be finite'
    assert _get(arrs, sfxs, 'lag12_consistency')[-1] == pytest.approx((2 / 9) ** 0.5, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'yoy_accel')[-1]), 'yoy_accel must be finite'
    assert _get(arrs, sfxs, 'yoy_accel')[-1] == pytest.approx(0.0, abs=1e-6)


def test_full_output_vector():
    # 14 значений, params={}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11, 0, 18, 7, 25]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'lag3_ratio') == pytest.approx([0.0, 0.0, 0.0, 0.5, 0.0, 0.25, -0.555556, 0.0, 0.333333, 1.75, 0.0, -0.1, -0.363636, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'lag9_ratio') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.833333, 0.0, 0.5, -0.222222, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'lag12_ratio') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.166667, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'lag3_trend') == pytest.approx([0.0, 0.0, 0.0, 0.166667, 0.166667, 0.25, -0.101852, -0.101852, -0.074074, 0.694444, 0.694444, 0.55, -0.154545, -0.154545], abs=1e-6)
    assert _get(arrs, sfxs, 'lag12_consistency') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.083333], abs=1e-6)
    assert _get(arrs, sfxs, 'yoy_accel') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
