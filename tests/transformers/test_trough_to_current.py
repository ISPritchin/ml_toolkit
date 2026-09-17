import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('trough_to_current', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_ratio():
    # [10,80,40,20,5,30] w=6: lo=5, last=30 → ratio=30/5=6.0
    arrs, sfxs = _run([10, 80, 40, 20, 5, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(6.0, abs=1e-4)


def test_current_is_trough_ratio_one():
    # [60,50,40,30,20,10] w=6: lo=10, last=10 → ratio=1.0
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, abs=1e-4)


def test_zero_trough_undefined_ratio_zero():
    # lo=0 → кратность к дну не определена → 0 (раньше v/eps ~ 3e10)
    arrs, sfxs = _run([0, 80, 40, 20, 0, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-9)


def test_constant_series_ratio_one():
    arrs, sfxs = _run([40, 40, 40, 40, 40, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    # в окне есть нулевой месяц (lo=0) → кратность не определена → 0 (раньше 3.5e10)
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-9)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [10, 80, 40, 20, 5, 30]
    arrs_plain, sfxs_plain = _run(values, {'windows': [6]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [6], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['w6']
    assert arrs_plain[0].tolist() == arrs_explicit[0].tolist()


def test_dilation_2_reads_every_other_point():
    # even indices carry [10,80,40,20,5,30] -> known trough_to_current 6.0
    values = [10, 999, 80, 999, 40, 999, 20, 999, 5, 999, 30]
    arrs, sfxs = _run(values, {'windows': [6], 'dilations': [1, 2]})
    assert sfxs == ['w6', 'w6_d2']
    assert _get(arrs, sfxs, 'w6_d2')[-1] == pytest.approx(6.0, abs=1e-4)
