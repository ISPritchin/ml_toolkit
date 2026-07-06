import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("max_abs_jump", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_value_from_docstring():
    # [10,20,15,60,55,50] w=6: |jumps|=10,5,45,5,5 → max=45
    arrs, sfxs = _run([10, 20, 15, 60, 55, 50], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(45.0)


def test_constant_series_jump_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_all_zeros_jump_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_single_large_spike():
    # [10,10,10,100,10,10] w=6: max jump = 90 (10→100) or 90 (100→10)
    arrs, sfxs = _run([10, 10, 10, 100, 10, 10], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(90.0)


def test_before_window_ready_is_zero():
    # ws=1 → no jump possible → 0
    arrs, sfxs = _run([100], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[0] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(60.0, rel=1e-4)
