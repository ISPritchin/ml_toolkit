import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('accel', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_deceleration():
    # [10,15,25,30]: 30-2*25+15 = -5 (decelerating)
    arrs, sfxs = _run([10, 15, 25, 30])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(-5.0)


def test_linear_series_zero_acceleration():
    # Constant increments → second difference = 0
    arrs, sfxs = _run([10, 20, 30, 40])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(0.0)


def test_known_acceleration():
    # [10,12,17,25]: 25-2*17+12 = 3 (accelerating)
    arrs, sfxs = _run([10, 12, 17, 25])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(3.0)


def test_zero_before_history_available():
    # At pos=0 and pos=1, accel=0 (not enough history)
    arrs, sfxs = _run([10, 20, 30])
    assert _get(arrs, sfxs, '')[0] == pytest.approx(0.0)
    assert _get(arrs, sfxs, '')[1] == pytest.approx(0.0)


def test_zeros_in_series():
    # [0,0,10]: 10-2*0+0 = 10 (big acceleration from zeros)
    arrs, sfxs = _run([0, 0, 10])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(10.0)

def test_full_output_vector():
    # 6 значений: [5, 10, 20, 15, 15, 30]
    # pos=0,1 -> 0 (недостаточно истории)
    # pos=2: 20 - 2*10 + 5   = 5
    # pos=3: 15 - 2*20 + 10  = -15
    # pos=4: 15 - 2*15 + 20  = 5
    # pos=5: 30 - 2*15 + 15  = 15
    arrs, sfxs = _run([5, 10, 20, 15, 15, 30])
    assert _get(arrs, sfxs, '') == pytest.approx([0.0, 0.0, 5.0, -15.0, 5.0, 15.0])


def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    assert math.isfinite(_get(arrs, sfxs, '')[-1]), ' must be finite'
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(35.0, rel=1e-4)
