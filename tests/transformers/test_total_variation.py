import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("total_variation", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_tv_from_docstring():
    # [10,30,20,40,30,50] w=6: TV=80, mean=30, TV_norm=80/30≈2.667
    arrs, sfxs = _run([10, 30, 20, 40, 30, 50], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(80.0, abs=1e-4)
    assert _get(arrs, sfxs, "norm_w6")[-1] == pytest.approx(80 / 30, abs=1e-4)


def test_constant_series_tv_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, "norm_w6")[-1] == pytest.approx(0.0, abs=1e-4)


def test_all_zeros_tv_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(0.0)


def test_monotone_ascending_tv_is_range():
    # [10,20,30,40,50,60]: TV=sum(10*5)=50, mean=35, TV_norm=50/35≈1.429
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(50.0)


def test_tv_always_nonneg():
    arrs, sfxs = _run([100, 20, 80, 10, 90, 5], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] >= 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # |0-10|+|60-0|+|0-60|+|0-0|+|35-0| = 10+60+60+0+35=165
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(165.0, abs=0.0001)
    assert math.isfinite(_get(arrs, sfxs, 'norm_w6')[-1]), 'norm_w6 must be finite'
    assert _get(arrs, sfxs, 'norm_w6')[-1] == pytest.approx(9.428571428032653, rel=1e-4)
