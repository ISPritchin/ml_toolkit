import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("sign_change_count", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_oscillating_count():
    # [10,30,20,40,30,50] w=6: diffs +20,-10,+20,-10,+20 → all 4 pairs change sign
    arrs, sfxs = _run([10, 30, 20, 40, 30, 50], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(4.0)


def test_monotone_ascending_zero_changes():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_monotone_descending_zero_changes():
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_constant_series_zero_changes():
    # All diffs=0, sign=0, zero signs ignored → count=0
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_single_reversal():
    # [10,20,30,40,30,20] w=6: goes up then down → 1 sign change
    arrs, sfxs = _run([10, 20, 30, 40, 30, 20], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # sign changes in [10,0,60,0,0,35]: 10→0(+→0), 0→60(0→+), 60→0(+→0) → 3
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(3.0, abs=1e-06)
