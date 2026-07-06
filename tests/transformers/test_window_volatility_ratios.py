import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("window_volatility_ratios", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_constant_series_regime_flag_zero():
    # CV=0 for all windows → regime_flag=0
    arrs, sfxs = _run([40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40])
    assert _get(arrs, sfxs, "regime_flag")[-1] == pytest.approx(0.0)


def test_regime_flag_when_short_very_volatile():
    # Stable long history, volatile short: CV_3 >> CV_12 → flag=1
    values = [50] * 9 + [10, 90, 10]  # 12 values; last 3 are volatile
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, "regime_flag")[-1] == pytest.approx(1.0)


def test_cv_ratio_equal_when_same_volatility():
    # For a uniform constant series, all CVs are equal → ratios≈1.0 (but =0/0 → 0/EPS)
    # In practice both CVs=0 → cv_ratio = 0/(0+EPS) ≈ 0
    arrs, sfxs = _run([20] * 12)
    assert _get(arrs, sfxs, "cv_ratio_w3_w6")[-1] == pytest.approx(0.0, abs=1e-4)


def test_vol_accel_positive_when_nesting_volatile():
    # std_3 > std_6 > std_12 → (std_3-std_6) > (std_6-std_12) → vol_accel > 0
    values = [50] * 9 + [10, 90, 10]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, "vol_accel")[-1] > 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert math.isfinite(_get(arrs, sfxs, 'cv_ratio_w3_w6')[-1]), 'cv_ratio_w3_w6 must be finite'
    assert _get(arrs, sfxs, 'cv_ratio_w3_w6')[-1] == pytest.approx(1.0910010994060462, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'cv_ratio_w3_w12')[-1]), 'cv_ratio_w3_w12 must be finite'
    assert _get(arrs, sfxs, 'cv_ratio_w3_w12')[-1] == pytest.approx(1.0963587465683455, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'cv_ratio_w6_w24')[-1]), 'cv_ratio_w6_w24 must be finite'
    assert _get(arrs, sfxs, 'cv_ratio_w6_w24')[-1] == pytest.approx(1.1053889003104158, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'vol_accel')[-1]), 'vol_accel must be finite'
    assert _get(arrs, sfxs, 'vol_accel')[-1] == pytest.approx(-2.5338619132670885, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'short_excess')[-1]), 'short_excess must be finite'
    assert _get(arrs, sfxs, 'short_excess')[-1] == pytest.approx(0.09635874734358826, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'regime_flag')[-1]), 'regime_flag must be finite'
    assert _get(arrs, sfxs, 'regime_flag')[-1] == pytest.approx(0.0, abs=1e-6)
