import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('log_slope_ratio', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_steady_exponential_ratio_near_one():
    # [10,20,40,80,160,320]: uniform doubling → short and long log-slopes ≈ equal → ratio≈1
    arrs, sfxs = _run([10, 20, 40, 80, 160, 320], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] == pytest.approx(1.019, abs=0.05)


def test_accelerating_series_ratio_above_one():
    # Rapid growth at end → short slope > long slope → ratio > 1
    arrs, sfxs = _run([1, 2, 3, 10, 50, 300], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] > 1.0


def test_decelerating_series_ratio_below_one():
    # Fast growth earlier, slow recently → short slope < long slope → ratio < 1
    arrs, sfxs = _run([1, 10, 100, 150, 160, 165], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6')[-1] < 1.0

# test_with_mixed_zeros skipped for log_slope_ratio: 'pairs'


def test_full_output_vector():
    # 10 значений, params={'pairs': [[3, 6]]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'w3_w6') == pytest.approx([0.0, -1.0, 1.0, 3.167265, -8.06976, 2.124775, 4.423253, -3.999498, 9.862016, 3.738823], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_pairs():
    values = [10, 20, 40, 80, 160, 320]
    arrs_plain, sfxs_plain = _run(values, {'pairs': [[3, 6]]})
    arrs_explicit, sfxs_explicit = _run(values, {'pairs': [[3, 6]], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['w3_w6']
    assert arrs_plain[0].tolist() == arrs_explicit[0].tolist()


def test_dilation_2_produces_a_distinct_suffix():
    values = [10, 999, 20, 999, 40, 999, 80, 999, 160, 999, 320]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]], 'dilations': [1, 2]})
    assert sfxs == ['w3_w6', 'w3_w6_d2']
    # doubling series on the strided sub-series -> ratio close to 1, same shape as the base case
    assert _get(arrs, sfxs, 'w3_w6_d2')[-1] == pytest.approx(1.019, abs=0.05)
