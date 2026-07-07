import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('regime_change', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_magnitude_and_flag_from_docstring():
    # [0,0,100,100,100,100] w=6: optimal split at k=2, mag=|0-100|/47.14≈2.121 > 2 → flag=1
    import math
    values = [0, 0, 100, 100, 100, 100]
    mean = sum(values) / 6
    var = sum((v - mean) ** 2 for v in values) / 6
    std = math.sqrt(var)
    expected_mag = abs(0 - 100) / std
    arrs, sfxs = _run(values, {'windows': [6]})
    assert _get(arrs, sfxs, 'magnitude_w6')[-1] == pytest.approx(expected_mag, abs=0.1)
    assert _get(arrs, sfxs, 'flag_w6')[-1] == pytest.approx(1.0)


def test_constant_series_no_regime_change():
    arrs, sfxs = _run([50] * 12, {'windows': [12]})
    assert _get(arrs, sfxs, 'flag_w12')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'magnitude_w12')[-1] == pytest.approx(0.0, abs=1e-3)


def test_late_vs_early_positive_for_increasing_series():
    # [10,20,30,40,50,60]: late half is higher
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'late_vs_early_w6')[-1] > 0


def test_asymmetry_greater_than_one_when_last_above_first():
    # [10,10,10,100,100,100]: last 3 avg=100, first 3 avg=10 → asym=100/10=10
    arrs, sfxs = _run([10, 10, 10, 100, 100, 100], {'windows': [6]})
    assert _get(arrs, sfxs, 'asymmetry_w6')[-1] == pytest.approx(10.0, abs=0.1)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'magnitude_w6')[-1]), 'magnitude_w6 must be finite'
    assert _get(arrs, sfxs, 'magnitude_w6')[-1] == pytest.approx(0.925745131506203, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'split_pos_w6')[-1]), 'split_pos_w6 must be finite'
    assert _get(arrs, sfxs, 'split_pos_w6')[-1] == pytest.approx(5.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'flag_w6')[-1]), 'flag_w6 must be finite'
    assert _get(arrs, sfxs, 'flag_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'late_vs_early_w6')[-1]), 'late_vs_early_w6 must be finite'
    assert _get(arrs, sfxs, 'late_vs_early_w6')[-1] == pytest.approx(-0.5143028508367794, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'asymmetry_w6')[-1]), 'asymmetry_w6 must be finite'
    assert _get(arrs, sfxs, 'asymmetry_w6')[-1] == pytest.approx(0.4999999999785714, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'current_regime_len')[-1]), 'current_regime_len must be finite'
    assert _get(arrs, sfxs, 'current_regime_len')[-1] == pytest.approx(3.0, rel=1e-4)
