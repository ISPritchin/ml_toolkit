import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('ewma', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_ewma_and_diff():
    # [100,80,120,90] alpha=0.3:
    # t0=100; t1=0.3*80+0.7*100=94; t2=0.3*120+0.7*94=101.8; t3=0.3*90+0.7*101.8=98.26
    arrs, sfxs = _run([100, 80, 120, 90], {'alphas': [0.3]})
    assert _get(arrs, sfxs, 'a30')[-1] == pytest.approx(98.26, abs=1e-2)
    assert _get(arrs, sfxs, 'diff_a30')[-1] == pytest.approx(90 - 98.26, abs=1e-2)


def test_constant_series_ewma_equals_value():
    # Constant series: EWMA stays at initial value
    arrs, sfxs = _run([50, 50, 50, 50], {'alphas': [0.3]})
    assert _get(arrs, sfxs, 'a30')[-1] == pytest.approx(50.0, abs=1e-6)
    assert _get(arrs, sfxs, 'diff_a30')[-1] == pytest.approx(0.0, abs=1e-6)


def test_initial_value_equals_first_observation():
    arrs, sfxs = _run([200, 100], {'alphas': [0.3]})
    assert _get(arrs, sfxs, 'a30')[0] == pytest.approx(200.0)


def test_zeros_series_ewma_zero():
    arrs, sfxs = _run([0, 0, 0, 0], {'alphas': [0.3]})
    assert _get(arrs, sfxs, 'a30')[-1] == pytest.approx(0.0)


def test_spike_after_zeros():
    # [0,0,0,10] alpha=0.3: EWMA starts at 0, last=0.3*10=3.0
    arrs, sfxs = _run([0, 0, 0, 10], {'alphas': [0.3]})
    assert _get(arrs, sfxs, 'a30')[-1] == pytest.approx(3.0, abs=1e-6)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'alphas': [0.3]})
    assert math.isfinite(_get(arrs, sfxs, 'a30')[-1]), 'a30 must be finite'
    assert _get(arrs, sfxs, 'a30')[-1] == pytest.approx(19.413219724110796, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'diff_a30')[-1]), 'diff_a30 must be finite'
    assert _get(arrs, sfxs, 'diff_a30')[-1] == pytest.approx(15.586780275889204, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'alphas': [0.3, 0.5]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'alphas': [0.3, 0.5]})
    assert _get(arrs, sfxs, 'a30') == pytest.approx([6.0, 4.2, 6.54, 7.278, 5.0946, 8.06622, 6.846354, 4.792448, 9.354713], abs=1e-6)
    assert _get(arrs, sfxs, 'diff_a30') == pytest.approx([0.0, -4.2, 5.46, 1.722, -5.0946, 6.93378, -2.846354, -4.792448, 10.645287], abs=1e-6)
    assert _get(arrs, sfxs, 'a50') == pytest.approx([6.0, 3.0, 7.5, 8.25, 4.125, 9.5625, 6.78125, 3.390625, 11.695312], abs=1e-6)
    assert _get(arrs, sfxs, 'diff_a50') == pytest.approx([0.0, -3.0, 4.5, 0.75, -4.125, 5.4375, -2.78125, -3.390625, 8.304688], abs=1e-6)
