import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("volatility_trend", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_value():
    # [20,20,20,10,40,10] pair(3,6):
    # short window [10,40,10]: mean=20, sq_devs=100+400+100=600, std=sqrt(200)≈14.142
    # long window [20,20,20,10,40,10]: mean=20, sq_devs=0+0+0+100+400+100=600, std=sqrt(100)=10
    # diff = sqrt(200)-10 ≈ 4.142
    arrs, sfxs = _run([20, 20, 20, 10, 40, 10], {"pairs": [[3, 6]]})
    expected = math.sqrt(200) - 10
    assert _get(arrs, sfxs, "w3_w6")[-1] == pytest.approx(expected, abs=1e-4)


def test_constant_series_diff_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {"pairs": [[3, 6]]})
    assert _get(arrs, sfxs, "w3_w6")[-1] == pytest.approx(0.0, abs=1e-6)


def test_stabilizing_negative_diff():
    # Volatile at start, stable at end → std_short < std_long → diff < 0
    arrs, sfxs = _run([10, 50, 10, 50, 30, 30], {"pairs": [[3, 6]]})
    assert _get(arrs, sfxs, "w3_w6")[-1] < 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'pairs': [[6, 12]]})
    assert math.isfinite(_get(arrs, sfxs, 'w6_w12')[-1]), 'w6_w12 must be finite'
    assert _get(arrs, sfxs, 'w6_w12')[-1] == pytest.approx(-3.6514091737404684, rel=1e-4)
