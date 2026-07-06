import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("half_ratio", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_ratio():
    # [10,10,10,20,20,20] w=6: first_half=30, second_half=60 → ratio=2.0
    arrs, sfxs = _run([10, 10, 10, 20, 20, 20], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(2.0, abs=1e-4)


def test_equal_halves_ratio_one():
    # [20,20,20,20,20,20] w=6: both halves sum=60 → ratio=1.0
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(1.0, abs=1e-4)


def test_declining_ratio_less_than_one():
    # [20,20,20,10,10,10] w=6: first_half=60, second_half=30 → ratio=0.5
    arrs, sfxs = _run([20, 20, 20, 10, 10, 10], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.5, abs=1e-4)


def test_incomplete_window_yields_zero():
    # Only 4 rows for w=6: window not full yet → ratio=0
    arrs, sfxs = _run([10, 20, 30, 40], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0, abs=1e-4)


def test_zero_first_half_undefined_ratio_zero():
    # [0,0,0,10,10,10] w=6: first_half_sum=0 → отношение не определено → 0
    # (раньше 30/eps ~ 3e10 — взрывной выброс)
    arrs, sfxs = _run([0, 0, 0, 10, 10, 10], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0, abs=1e-9)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.4999999999928571, rel=1e-4)
