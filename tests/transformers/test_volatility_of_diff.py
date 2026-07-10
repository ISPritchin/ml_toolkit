import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('volatility_of_diff', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_constant_diffs_vol_zero():
    # [10,20,30,40] w=4: diffs=[10,10,10], mean_d=10, all deviations=0 → std=0
    arrs, sfxs = _run([10, 20, 30, 40], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(0.0, abs=1e-8)


def test_known_vol():
    # [10,30,20,40,30,50] w=6: diffs=+20,-10,+20,-10,+20 (5 diffs)
    # mean_d=40/5=8; deviations: 12,-18,12,-18,12; sq_devs: 144,324,144,324,144 → sum=1080
    # std=sqrt(1080/5)=sqrt(216)≈14.70
    arrs, sfxs = _run([10, 30, 20, 40, 30, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(math.sqrt(216), abs=1e-4)


def test_all_zeros_vol_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-8)


def test_alternating_gives_positive_vol():
    # [10,30,10,30,10,30]: diffs +20,-20,+20,-20,+20 (5 diffs), mean=4
    # sum_sq = 16²+24²+16²+24²+16² = 1920, biased std = sqrt(1920/5) = sqrt(384) ≈ 19.596
    import math
    arrs, sfxs = _run([10, 30, 10, 30, 10, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(math.sqrt(384), abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(40.98780306383839, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 0.0, 9.0, 7.874008, 8.831761, 10.198039, 11.813363, 10.984838, 13.274872], abs=1e-6)
