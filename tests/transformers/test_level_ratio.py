import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('level_ratio', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_ratio():
    # [10,20,30,40,50,60] pair(3,6): mean_short=(40+50+60)/3=50, mean_long=35
    # ratio=50/35≈1.4286
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(50 / 35, abs=1e-4)


def test_constant_series_ratio_one():
    arrs, sfxs = _run([25, 25, 25, 25, 25, 25], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_declining_ratio_less_than_one():
    # [60,50,40,30,20,10]: recent mean < long mean → ratio < 1
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] < 1.0


def test_all_zeros_ratio_near_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'pairs': [[3, 6]]})
    # 0/(0+EPS) ≈ 0
    assert abs(_get(arrs, sfxs, 'w3_w6')[-1]) < 1e-3

# test_with_mixed_zeros skipped for level_ratio: 'pairs'


def test_full_output_vector():
    # 10 значений, params={'pairs': [[3, 6]]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6') == pytest.approx([1.0, 1.0, 1.0, 1.037037, 1.296296, 1.142857, 0.95, 0.95, 1.0, 1.24], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_pairs():
    values = [10, 20, 30, 40, 50, 60]
    arrs_plain, sfxs_plain = _run(values, {'pairs': [[3, 6]]})
    arrs_explicit, sfxs_explicit = _run(values, {'pairs': [[3, 6]], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['w3_w6']
    assert arrs_plain[0].tolist() == arrs_explicit[0].tolist()


def test_dilation_2_produces_a_distinct_suffix():
    values = [10, 999, 20, 999, 30, 999, 40, 999, 50, 999, 60]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]], 'dilations': [1, 2]})
    assert sfxs == ['w3_w6', 'w3_w6_d2']
    # doubling-noise-skipped strided sub-series matches the docstring's known example
    assert _get(arrs, sfxs, 'w3_w6_d2')[-1] == pytest.approx(1.429, abs=0.01)
