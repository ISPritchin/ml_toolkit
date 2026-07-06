import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("active_run_count", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_run_count():
    # [10,0,5,0,8,3] w=6: transitions 0→nonzero at idx 0,2,4 → count=3
    arrs, sfxs = _run([10, 0, 5, 0, 8, 3], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(3.0)


def test_all_zeros_no_runs():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_single_continuous_run_counts_one():
    # [10,20,30,40,50,60]: one contiguous run → count=1
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(1.0)


def test_alternating_nonzero_zero():
    # [10,0,5,0,8,0]: three runs starting from positions 0,2,4
    arrs, sfxs = _run([10, 0, 5, 0, 8, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(3.0)


def test_zero_then_one_run():
    # [0,0,0,10,20,30]: one run starting at idx 3
    arrs, sfxs = _run([0, 0, 0, 10, 20, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # 3 isolated non-zero runs in [10,0,60,0,0,35]
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(3.0, abs=1e-06)
