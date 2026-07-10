import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('microstructure', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_surprise_from_docstring():
    # [20,20,20,20,20,30] w=6: mean=130/6≈21.667, std=3.727
    # surprise = |30-21.667|/3.727 ≈ 2.236
    import math
    values = [20, 20, 20, 20, 20, 30]
    mean = sum(values) / 6
    sq_dev = sum((v - mean) ** 2 for v in values) / 6
    std = math.sqrt(sq_dev)
    expected_surprise = abs(30 - mean) / std
    arrs, sfxs = _run(values, {'windows': [6]})
    assert _get(arrs, sfxs, 'surprise_w6')[-1] == pytest.approx(expected_surprise, abs=0.01)


def test_constant_series_predictability_high():
    # CV=0 → predictability=1/(1+0)=1.0
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'predictability_w6')[-1] == pytest.approx(1.0, abs=1e-3)


def test_constant_series_surprise_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'surprise_w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_surprise_dir_positive_when_above_mean():
    arrs, sfxs = _run([10, 10, 10, 10, 10, 100], {'windows': [6]})
    assert _get(arrs, sfxs, 'surprise_dir')[-1] == pytest.approx(1.0)


def test_surprise_dir_negative_when_below_mean():
    arrs, sfxs = _run([100, 100, 100, 100, 100, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'surprise_dir')[-1] == pytest.approx(-1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'snr_w6')[-1]), 'snr_w6 must be finite'
    assert _get(arrs, sfxs, 'snr_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'surprise_w6')[-1]), 'surprise_w6 must be finite'
    assert _get(arrs, sfxs, 'surprise_w6')[-1] == pytest.approx(0.7714542762551692, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'predictability_w6')[-1]), 'predictability_w6 must be finite'
    assert _get(arrs, sfxs, 'predictability_w6')[-1] == pytest.approx(0.4354920624483158, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'cond_mean_w6')[-1]), 'cond_mean_w6 must be finite'
    assert _get(arrs, sfxs, 'cond_mean_w6')[-1] == pytest.approx(34.99999993, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'vs_cond_mean_w6')[-1]), 'vs_cond_mean_w6 must be finite'
    assert _get(arrs, sfxs, 'vs_cond_mean_w6')[-1] == pytest.approx(1.0000000019714286, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'surprise_dir')[-1]), 'surprise_dir must be finite'
    assert _get(arrs, sfxs, 'surprise_dir')[-1] == pytest.approx(1.0, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'snr_w4') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'surprise_w4') == pytest.approx([0.0, 1.0, 1.224745, 0.507093, 0.980196, 1.069045, 0.534522, 0.77374, 1.269526], abs=1e-6)
    assert _get(arrs, sfxs, 'predictability_w4') == pytest.approx([1.0, 0.5, 0.55051, 0.603376, 0.494999, 0.615912, 0.555006, 0.436219, 0.547019], abs=1e-6)
    assert _get(arrs, sfxs, 'cond_mean_w4') == pytest.approx([6.0, 6.0, 9.0, 9.0, 10.5, 12.0, 9.333333, 9.5, 13.0], abs=1e-6)
    assert _get(arrs, sfxs, 'vs_cond_mean_w4') == pytest.approx([1.0, 0.0, 1.333333, 1.0, 0.0, 1.25, 0.428571, 0.0, 1.538462], abs=1e-6)
    assert _get(arrs, sfxs, 'surprise_dir') == pytest.approx([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0], abs=1e-6)
