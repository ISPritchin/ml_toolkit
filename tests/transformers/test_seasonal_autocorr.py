import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("seasonal_autocorr", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_bimonthly_pattern_even_odd():
    # [10,30,10,30,10,30]: even positions=10, odd positions=30
    # even_odd = 10/30 ≈ 0.333
    arrs, sfxs = _run([10, 30, 10, 30, 10, 30])
    assert _get(arrs, sfxs, "even_odd_w12")[-1] == pytest.approx(10 / 30, abs=1e-4)


def test_constant_series_even_odd_one():
    # All equal → even_mean = odd_mean → even_odd ≈ 1.0
    arrs, sfxs = _run([20] * 12)
    assert _get(arrs, sfxs, "even_odd_w12")[-1] == pytest.approx(1.0, abs=1e-3)


def test_lag6_positive_for_semiannual_pattern():
    # Perfect semi-annual: same values every 6 months → high lag6 autocorr
    series = [10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 50, 60]
    arrs, sfxs = _run(series)
    assert _get(arrs, sfxs, "lag6")[-1] == pytest.approx(1.0, abs=1e-3)


def test_lag12_positive_for_annual_pattern():
    # Repeating annual pattern → lag12 autocorr high
    series = list(range(1, 13)) + list(range(1, 13))
    arrs, sfxs = _run(series)
    assert _get(arrs, sfxs, "lag12")[-1] == pytest.approx(1.0, abs=1e-3)


def test_no_lag12_before_12_periods():
    # lag12 is 0 before pos=12
    arrs, sfxs = _run([10] * 24)
    assert _get(arrs, sfxs, "lag12")[11] == pytest.approx(0.0, abs=1e-6)

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
