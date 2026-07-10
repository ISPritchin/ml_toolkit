import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('max_drawdown', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_drawdown():
    # [10,80,40,20,5,30] w=6: peak=80, trough=5 → (80-5)/80=0.9375
    arrs, sfxs = _run([10, 80, 40, 20, 5, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.9375, abs=1e-4)


def test_monotone_ascending_no_drawdown():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_all_zeros_drawdown_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_complete_crash_drawdown_near_one():
    # [100,0,0,0,0,0]: peak=100, trough=0 → (100-0)/100=1.0
    arrs, sfxs = _run([100, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_half_drawdown():
    # [100,80,60,50,50,50] w=6: peak=100 at start, min=50 → (100-50)/100=0.5
    arrs, sfxs = _run([100, 80, 60, 50, 50, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.5, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], abs=1e-6)
