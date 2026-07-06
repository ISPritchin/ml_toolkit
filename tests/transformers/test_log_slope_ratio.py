import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("log_slope_ratio", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_steady_exponential_ratio_near_one():
    # [10,20,40,80,160,320]: uniform doubling → short and long log-slopes ≈ equal → ratio≈1
    arrs, sfxs = _run([10, 20, 40, 80, 160, 320], {"pairs": [[3, 6]]})
    assert _get(arrs, sfxs, "w3_w6")[-1] == pytest.approx(1.019, abs=0.05)


def test_accelerating_series_ratio_above_one():
    # Rapid growth at end → short slope > long slope → ratio > 1
    arrs, sfxs = _run([1, 2, 3, 10, 50, 300], {"pairs": [[3, 6]]})
    assert _get(arrs, sfxs, "w3_w6")[-1] > 1.0


def test_decelerating_series_ratio_below_one():
    # Fast growth earlier, slow recently → short slope < long slope → ratio < 1
    arrs, sfxs = _run([1, 10, 100, 150, 160, 165], {"pairs": [[3, 6]]})
    assert _get(arrs, sfxs, "w3_w6")[-1] < 1.0

# test_with_mixed_zeros skipped for log_slope_ratio: 'pairs'
