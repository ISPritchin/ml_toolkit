import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("rolling_min_max", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_min_max():
    # [10,80,40,20,5,30] w=6 → min=5, max=80
    arrs, sfxs = _run([10, 80, 40, 20, 5, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "min_w6")[-1] == pytest.approx(5.0)
    assert _get(arrs, sfxs, "max_w6")[-1] == pytest.approx(80.0)


def test_all_zeros_min_max_both_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "min_w6")[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, "max_w6")[-1] == pytest.approx(0.0)


def test_partial_window_uses_available_rows():
    # At row 2 only 3 values [10,80,40] are in window
    arrs, sfxs = _run([10, 80, 40, 20, 5, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "min_w6")[2] == pytest.approx(10.0)
    assert _get(arrs, sfxs, "max_w6")[2] == pytest.approx(80.0)


def test_monotone_ascending_max_equals_current():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {"windows": [6]})
    # max is always the last (current) value
    assert _get(arrs, sfxs, "max_w6")[-1] == pytest.approx(60.0)
    assert _get(arrs, sfxs, "min_w6")[-1] == pytest.approx(10.0)


def test_zero_in_window_sets_min_to_zero():
    # Any zero in window → min=0
    arrs, sfxs = _run([100, 50, 0, 80, 70, 90], {"windows": [6]})
    assert _get(arrs, sfxs, "min_w6")[-1] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # min of [10,0,60,0,0,35]=0
    assert _get(arrs, sfxs, 'min_w6')[-1] == pytest.approx(0.0, abs=1e-06)
    # max of [10,0,60,0,0,35]=60
    assert _get(arrs, sfxs, 'max_w6')[-1] == pytest.approx(60.0, abs=1e-06)
