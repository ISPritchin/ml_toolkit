import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("direction_flag", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_ascending_trend_returns_plus_one():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(1.0)


def test_descending_trend_returns_minus_one():
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(-1.0)


def test_constant_series_returns_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_all_zeros_returns_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_spike_at_end_returns_plus_one():
    # Mean trend is upward due to spike at end
    arrs, sfxs = _run([10, 10, 10, 10, 10, 100], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, rel=1e-4)
