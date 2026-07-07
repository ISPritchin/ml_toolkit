import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('alternation_rate', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_alternation_rate():
    # [10,30,20,40,30,50] w=6: diffs +20,-10,+20,-10,+20 (5 diffs)
    # all 4 adjacent pairs change sign → alt_rate=4/4=1.0
    arrs, sfxs = _run([10, 30, 20, 40, 30, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'alt_rate_w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_known_max_jump_share():
    # TV=80, max_jump=20 → max_jump_share=20/80=0.25
    arrs, sfxs = _run([10, 30, 20, 40, 30, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'max_jump_share_w6')[-1] == pytest.approx(0.25, abs=1e-4)


def test_known_mean_abs_jump():
    # TV=80, n_diffs=5 → mean_abs_jump=80/5=16.0
    arrs, sfxs = _run([10, 30, 20, 40, 30, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'mean_abs_jump_w6')[-1] == pytest.approx(16.0, abs=1e-4)


def test_monotone_alt_rate_zero():
    # [10,20,30,40,50,60]: all diffs positive → no alternation
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'alt_rate_w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_constant_series():
    # All diffs=0, signs=0, no nonzero signs → alternation=0, TV=0, mean_jump=0
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {'windows': [6]})
    assert _get(arrs, sfxs, 'alt_rate_w6')[-1] == pytest.approx(0.0, abs=1e-4)
    assert _get(arrs, sfxs, 'mean_abs_jump_w6')[-1] == pytest.approx(0.0, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'alt_rate_w6')[-1]), 'alt_rate_w6 must be finite'
    assert _get(arrs, sfxs, 'alt_rate_w6')[-1] == pytest.approx(0.75, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'max_jump_share_w6')[-1]), 'max_jump_share_w6 must be finite'
    assert _get(arrs, sfxs, 'max_jump_share_w6')[-1] == pytest.approx(0.3636363636341598, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'mean_abs_jump_w6')[-1]), 'mean_abs_jump_w6 must be finite'
    assert _get(arrs, sfxs, 'mean_abs_jump_w6')[-1] == pytest.approx(33.0, rel=1e-4)
