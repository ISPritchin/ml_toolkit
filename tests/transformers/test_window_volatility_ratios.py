import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('window_volatility_ratios', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_constant_series_regime_flag_zero():
    # CV=0 for all windows → regime_flag=0
    arrs, sfxs = _run([40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40])
    assert _get(arrs, sfxs, 'regime_flag')[-1] == pytest.approx(0.0)


def test_regime_flag_when_short_very_volatile():
    # Stable long history, volatile short: CV_3 >> CV_12 → flag=1
    values = [50] * 9 + [10, 90, 10]  # 12 values; last 3 are volatile
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'regime_flag')[-1] == pytest.approx(1.0)


def test_cv_ratio_equal_when_same_volatility():
    # For a uniform constant series, all CVs are equal → ratios≈1.0 (but =0/0 → 0/EPS)
    # In practice both CVs=0 → cv_ratio = 0/(0+EPS) ≈ 0
    arrs, sfxs = _run([20] * 12)
    assert _get(arrs, sfxs, 'cv_ratio_w3_w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_vol_accel_positive_when_nesting_volatile():
    # std_3 > std_6 > std_12 → (std_3-std_6) > (std_6-std_12) → vol_accel > 0
    values = [50] * 9 + [10, 90, 10]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'vol_accel')[-1] > 0

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


def test_full_output_vector():
    # 26 значений, params={}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11, 0, 18, 7, 25, 0, 0, 14, 30, 5, 0, 22, 16, 0, 9, 28, 3]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'cv_ratio_w3_w6') == pytest.approx([0.0, 1.0, 1.0, 1.108146, 0.819485, 0.953509, 1.158231, 1.158231, 1.151279, 0.869384, 0.869384, 0.836684, 1.054761, 0.712612, 1.100758, 1.197283, 1.634057, 0.91253, 0.662427, 0.83811, 1.096572, 1.063399, 0.783846, 0.839073, 1.121511, 1.038984], abs=1e-6)
    assert _get(arrs, sfxs, 'cv_ratio_w3_w12') == pytest.approx([0.0, 1.0, 1.0, 1.108146, 0.819485, 0.953509, 1.231984, 1.056773, 1.162272, 0.929271, 0.835234, 0.861728, 1.012612, 0.561411, 1.064897, 1.341104, 1.575151, 0.882228, 0.676858, 1.202829, 1.117834, 0.805668, 0.805668, 0.821112, 1.050657, 0.785648], abs=1e-6)
    assert _get(arrs, sfxs, 'cv_ratio_w6_w24') == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.063677, 0.912403, 1.009549, 1.068885, 0.960719, 1.029933, 0.976983, 0.718271, 0.956307, 1.17834, 0.914448, 0.955413, 0.992336, 1.323304, 0.982892, 0.741265, 0.959701, 0.98042, 0.897653, 0.838386], abs=1e-6)
    assert _get(arrs, sfxs, 'vol_accel') == pytest.approx([0.0, 0.0, 0.0, 0.66196, 0.29902, 0.50756, 0.15576, 0.262979, 0.444897, -0.436312, -0.361263, -1.733228, -1.301426, -1.449778, 0.715065, 0.8863, -3.334152, -0.760393, -3.100432, 1.332854, -2.868791, -0.318112, -3.08362, 0.523495, 1.76016, 1.422383], abs=1e-6)
    assert _get(arrs, sfxs, 'short_excess') == pytest.approx([0.0, 0.0, 0.0, 0.108146, -0.180515, -0.046491, 0.231984, 0.056773, 0.162272, -0.070729, -0.164766, -0.138272, 0.012612, -0.438589, 0.064897, 0.341104, 0.575151, -0.117772, -0.323142, 0.202829, 0.117834, -0.194332, -0.194332, -0.178888, 0.050657, -0.214352], abs=1e-6)
    assert _get(arrs, sfxs, 'regime_flag') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
