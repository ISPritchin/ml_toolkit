import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('rank_in_window', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_rank():
    # [10,30,20,50,40,45] w=6, v=45: values ≤45 are 10,30,20,40,45 → 5 out of 6
    arrs, sfxs = _run([10, 30, 20, 50, 40, 45], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(5 / 6, abs=1e-4)


def test_all_equal_rank_one():
    # All values equal → all values ≤ v → rank = ws/ws = 1.0
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_minimum_value_rank_small():
    # [50,40,30,20,10,5] w=6, v=5: only v itself is ≤5 → 1/6
    arrs, sfxs = _run([50, 40, 30, 20, 10, 5], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1 / 6, abs=1e-4)


def test_maximum_value_rank_one():
    # [5,10,20,30,40,50] w=6, v=50: all ≤50 → 6/6=1.0
    arrs, sfxs = _run([5, 10, 20, 30, 40, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_all_zeros_rank_one():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.8333333333333334, rel=1e-4)
