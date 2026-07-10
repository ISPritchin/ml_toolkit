import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('seasonal_autocorr', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_bimonthly_pattern_even_odd():
    # [10,30,10,30,10,30]: even positions=10, odd positions=30
    # even_odd = 10/30 ≈ 0.333
    arrs, sfxs = _run([10, 30, 10, 30, 10, 30])
    assert _get(arrs, sfxs, 'even_odd_w12')[-1] == pytest.approx(10 / 30, abs=1e-4)


def test_constant_series_even_odd_one():
    # All equal → even_mean = odd_mean → even_odd ≈ 1.0
    arrs, sfxs = _run([20] * 12)
    assert _get(arrs, sfxs, 'even_odd_w12')[-1] == pytest.approx(1.0, abs=1e-3)


def test_lag6_positive_for_semiannual_pattern():
    # Perfect semi-annual: same values every 6 months → high lag6 autocorr
    series = [10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60]
    arrs, sfxs = _run(series)
    assert _get(arrs, sfxs, 'lag6')[-1] == pytest.approx(1.0, abs=1e-3)


def test_lag12_positive_for_annual_pattern():
    # Repeating annual pattern → lag12 autocorr high
    series = list(range(1, 13)) + list(range(1, 13))
    arrs, sfxs = _run(series)
    assert _get(arrs, sfxs, 'lag12')[-1] == pytest.approx(1.0, abs=1e-3)


def test_no_lag12_before_12_periods():
    # lag12 is 0 before pos=12
    arrs, sfxs = _run([10] * 24)
    assert _get(arrs, sfxs, 'lag12')[11] == pytest.approx(0.0, abs=1e-6)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    assert math.isfinite(_get(arrs, sfxs, 'lag6')[-1]), 'lag6 must be finite'
    assert _get(arrs, sfxs, 'lag6')[-1] == pytest.approx(-0.20647404740283806, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'lag6_w24')[-1]), 'lag6_w24 must be finite'
    assert _get(arrs, sfxs, 'lag6_w24')[-1] == pytest.approx(-0.20647404740283806, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'lag12')[-1]), 'lag12 must be finite'
    assert _get(arrs, sfxs, 'lag12')[-1] == pytest.approx(-0.9176629354822471, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'lag12_w24')[-1]), 'lag12_w24 must be finite'
    assert _get(arrs, sfxs, 'lag12_w24')[-1] == pytest.approx(-0.9176629354822471, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'quarter_cv_w12')[-1]), 'quarter_cv_w12 must be finite'
    assert _get(arrs, sfxs, 'quarter_cv_w12')[-1] == pytest.approx(0.27304261551550313, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'even_odd_w12')[-1]), 'even_odd_w12 must be finite'
    # чётность считается от позиции внутри сущности: чётные позиции (4,6,8,10,12,14)
    # дают mean 55/6, нечётные (3,5,...,13) — 190/6 → 55/190
    assert _get(arrs, sfxs, 'even_odd_w12')[-1] == pytest.approx(55 / 190, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'amplitude_w12')[-1]), 'amplitude_w12 must be finite'
    assert _get(arrs, sfxs, 'amplitude_w12')[-1] == pytest.approx(0.7346938775510204, rel=1e-4)


def test_full_output_vector():
    # 26 значений, params={}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11, 0, 18, 7, 25, 0, 0, 14, 30, 5, 0, 22, 16, 0, 9, 28, 3]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'lag6') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.944911, 0.940268, 0.948706, 0.953996, 0.948748, 0.444643, 0.068084, 0.010996, -0.050187, 0.208923, 0.211856, 0.0187, -0.070784, -0.103251, -0.144715, -0.135765, -0.168391, -0.122327], abs=1e-6)
    assert _get(arrs, sfxs, 'lag6_w24') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.944911, 0.940268, 0.948706, 0.953996, 0.948748, 0.444643, 0.068084, 0.010996, -0.050187, 0.208923, 0.211856, 0.0187, -0.070784, -0.103251, -0.144715, -0.135765, -0.185535, -0.205514], abs=1e-6)
    assert _get(arrs, sfxs, 'lag12') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -0.969216, -0.959038, -0.90976, -0.101597, -0.049013, 0.096118, 0.290003, 0.305169, 0.379625, 0.320744, 0.274618, 0.10052], abs=1e-6)
    assert _get(arrs, sfxs, 'lag12_w24') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, -0.969216, -0.959038, -0.90976, -0.101597, -0.049013, 0.096118, 0.290003, 0.305169, 0.379625, 0.320744, 0.267338, 0.231142], abs=1e-6)
    assert _get(arrs, sfxs, 'quarter_cv_w12') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.164089, 0.190941, 0.405801, 0.125457, 0.169706, 0.486506, 0.228255, 0.302651, 0.394339, 0.199862, 0.292973, 0.378649, 0.2307, 0.228089, 0.327601], abs=1e-6)
    assert _get(arrs, sfxs, 'even_odd_w12') == pytest.approx([0.0, 0.0, 0.0, 2.0, 1.333333, 0.75, 0.6875, 0.916667, 1.4, 1.2, 1.0, 0.792453, 0.811321, 0.551282, 0.397436, 0.449275, 0.652174, 0.535714, 0.547619, 0.547619, 0.571429, 0.539326, 0.539326, 0.6, 0.8625, 1.189655], abs=1e-6)
    assert _get(arrs, sfxs, 'amplitude_w12') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.463158, 0.5, 1.024793, 0.293578, 0.48, 1.263158, 0.620155, 0.738462, 1.107692, 0.515152, 0.70073, 1.051095, 0.59375, 0.644295, 0.818898], abs=1e-6)
