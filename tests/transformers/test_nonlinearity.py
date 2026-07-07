import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('nonlinearity', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_quad_proxy_from_docstring():
    # [10,20,30,20,10,5] w=6: mean=95/6≈15.833, third=2
    # Q1=(10+20)/2=15, Q2=(30+20)/2=25, Q3=(10+5)/2=7.5
    # quad=(15-50+7.5)/15.833=-27.5/15.833≈-1.737
    arrs, sfxs = _run([10, 20, 30, 20, 10, 5], {'windows': [6]})
    assert _get(arrs, sfxs, 'quad_proxy_w6')[-1] == pytest.approx(-1.737, abs=0.02)
    assert _get(arrs, sfxs, 'convexity_sign_w6')[-1] == pytest.approx(-1.0)


def test_linear_series_quad_proxy_near_zero():
    # Linear → Q1≈mean of first third, Q3≈mean of last third
    # For perfect linear the quad proxy is very small
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    # Q1=(10+20)/2=15, Q2=(30+40)/2=35, Q3=(50+60)/2=55
    # quad=(15-70+55)/35=0/35=0
    assert abs(_get(arrs, sfxs, 'quad_proxy_w6')[-1]) < 0.01


def test_constant_series_quad_proxy_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'quad_proxy_w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_all_zeros_quad_proxy_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'quad_proxy_w6')[-1] == pytest.approx(0.0, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'quad_proxy_w6')[-1]), 'quad_proxy_w6 must be finite'
    assert _get(arrs, sfxs, 'quad_proxy_w6')[-1] == pytest.approx(-2.142857142734694, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'convexity_sign_w6')[-1]), 'convexity_sign_w6 must be finite'
    assert _get(arrs, sfxs, 'convexity_sign_w6')[-1] == pytest.approx(-1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'mean_accel_w6')[-1]), 'mean_accel_w6 must be finite'
    assert _get(arrs, sfxs, 'mean_accel_w6')[-1] == pytest.approx(11.25, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'accel_std_w6')[-1]), 'accel_std_w6 must be finite'
    assert _get(arrs, sfxs, 'accel_std_w6')[-1] == pytest.approx(76.84196444651842, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'frac_concave_w6')[-1]), 'frac_concave_w6 must be finite'
    assert _get(arrs, sfxs, 'frac_concave_w6')[-1] == pytest.approx(0.25, rel=1e-4)
