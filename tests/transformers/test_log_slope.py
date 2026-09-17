import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('log_slope', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_value_from_docstring():
    # [10,20,40,80] w=4: doubling → log_slope≈0.666/month
    arrs, sfxs = _run([10, 20, 40, 80], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(0.666, abs=0.01)


def test_constant_series_log_slope_zero():
    arrs, sfxs = _run([50, 50, 50, 50, 50, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_all_zeros_log_slope_zero():
    # log1p(0)=0 for all → constant log-series → slope=0
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_declining_series_negative_log_slope():
    arrs, sfxs = _run([80, 40, 20, 10, 5, 2], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] < 0


def test_log_slope_positive_for_growth():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] > 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.05192127040329686, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, -1.94591, 0.30952, 0.363497, -0.026236, -0.167967, 0.069315, -0.116315, -0.079364], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [10, 20, 40, 80]
    arrs_plain, sfxs_plain = _run(values, {'windows': [4]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [4], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['w4']
    assert arrs_plain[0].tolist() == arrs_explicit[0].tolist()


def test_dilation_2_reads_every_other_point():
    # doubling series [10,20,40,80] at even indices -> same known-value slope 0.666
    values = [10, 999, 20, 999, 40, 999, 80]
    arrs, sfxs = _run(values, {'windows': [4], 'dilations': [1, 2]})
    assert sfxs == ['w4', 'w4_d2']
    assert _get(arrs, sfxs, 'w4_d2')[-1] == pytest.approx(0.666, abs=0.01)
