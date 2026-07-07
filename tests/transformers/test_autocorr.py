import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('autocorr', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_perfect_positive_lag1_autocorr():
    # Monotone series: each step perfectly predicts next → r≈1
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60])
    assert _get(arrs, sfxs, 'lag1')[-1] == pytest.approx(1.0, abs=1e-4)


def test_perfect_negative_lag1_autocorr():
    # Alternating [10,30,10,30,10,30]: strong negative autocorr
    arrs, sfxs = _run([10, 30, 10, 30, 10, 30])
    assert _get(arrs, sfxs, 'lag1')[-1] == pytest.approx(-1.0, abs=1e-4)


def test_constant_series_autocorr_zero():
    # std=0 → autocorr=0 by convention
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20])
    assert abs(_get(arrs, sfxs, 'lag1')[-1]) < 1e-4


def test_known_lag1_from_docstring():
    # [10,20,15,25,20]: lag1 = -100/316.23 ≈ -0.316
    arrs, sfxs = _run([10, 20, 15, 25, 20])
    assert _get(arrs, sfxs, 'lag1')[-1] == pytest.approx(-0.316, abs=0.01)


def test_all_zeros_autocorr_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0])
    assert abs(_get(arrs, sfxs, 'lag1')[-1]) < 1e-4


def test_lag2_not_computed_before_pos_2():
    arrs, sfxs = _run([10, 20, 30, 40])
    assert _get(arrs, sfxs, 'lag2')[0] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'lag2')[1] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'lags': [3]})
    assert math.isfinite(_get(arrs, sfxs, 'lag1')[-1]), 'lag1 must be finite'
    assert _get(arrs, sfxs, 'lag1')[-1] == pytest.approx(-0.3711167869699702, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'lag2')[-1]), 'lag2 must be finite'
    assert _get(arrs, sfxs, 'lag2')[-1] == pytest.approx(-0.23290940619767977, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'lag3')[-1]), 'lag3 must be finite'
    assert _get(arrs, sfxs, 'lag3')[-1] == pytest.approx(0.23542258594302717, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'lag1_w12')[-1]), 'lag1_w12 must be finite'
    assert _get(arrs, sfxs, 'lag1_w12')[-1] == pytest.approx(-0.391641942576962, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'lag2_w12')[-1]), 'lag2_w12 must be finite'
    assert _get(arrs, sfxs, 'lag2_w12')[-1] == pytest.approx(-0.43361060227498427, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'partial_lag2')[-1]), 'partial_lag2 must be finite'
    assert _get(arrs, sfxs, 'partial_lag2')[-1] == pytest.approx(-0.4298376072839067, rel=1e-4)
