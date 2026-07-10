import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('roughness_ratio', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_ratio():
    # [40,40,40,40,10,40] pair(3,6):
    # short [10,40,40] wait, last 3 are [40,10,40]: TV=30+30=60, mean=30, TV_norm=2.0
    # Actually last 3 of [40,40,40,40,10,40] = [40,10,40]
    # long [40,40,40,40,10,40]: TV=0+0+0+30+30=60, mean=35, TV_norm=60/35≈1.714
    # ratio = 2.0/1.714 = 7/6 ≈ 1.1667
    arrs, sfxs = _run([40, 40, 40, 40, 10, 40], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(7 / 6, abs=1e-3)


def test_constant_series_ratio_near_one():
    # Both TVs=0, both means=same → ratio = 0/(0+EPS) / (0/(0+EPS)+EPS) → 0/EPS
    # Actually both TV_norms=0 → ratio = 0/(0+EPS) ≈ 0 (not 1)
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'pairs': [[3, 6]]})
    # TV=0, TV_norm=0 for both, ratio≈0
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_volatile_short_window_ratio_above_one():
    # Short [1,100,1]: TV=198, mean=34, TV_norm≈5.82
    # Long [1000,1000,1000,1,100,1]: TV≈1197, mean≈517, TV_norm≈2.31 → ratio≈2.52 > 1
    arrs, sfxs = _run([1000, 1000, 1000, 1, 100, 1], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] > 1.0


def test_all_zeros_ratio_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(0.0, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert math.isfinite(_get(arrs, sfxs, 'w3_w6')[-1]), 'w3_w6 must be finite'
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(0.3181818181389807, rel=1e-4)


def test_full_output_vector():
    # 10 значений, params={'pairs': [[3, 6]]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6') == pytest.approx([0.0, 1.0, 1.0, 0.688776, 0.308571, 0.466667, 0.547368, 0.37594, 0.40678, 0.396391], abs=1e-6)
