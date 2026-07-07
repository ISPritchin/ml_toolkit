import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('run_above_mean', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_run():
    # [10,20,30,40,50,60] w=6: at each step v > rolling mean → run=5 at last
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'window': 6})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(5.0)


def test_all_equal_run_zero():
    # v==mean at each step → not >, run resets → 0
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'window': 6})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0)


def test_single_spike_after_zeros():
    # [0,0,0,0,0,10] w=6: mean_w6=10/6≈1.67, v=10>1.67 → run=1
    arrs, sfxs = _run([0, 0, 0, 0, 0, 10], {'window': 6})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0)


def test_run_resets_on_dip_below_mean():
    # [10,20,30,40,5,60] w=6: v=5 < mean → run resets; then v=60 > mean → run=1
    arrs, sfxs = _run([10, 20, 30, 40, 5, 60], {'window': 6})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'window': 12})
    assert math.isfinite(_get(arrs, sfxs, 'w12')[-1]), 'w12 must be finite'
    assert _get(arrs, sfxs, 'w12')[-1] == pytest.approx(1.0, rel=1e-4)
