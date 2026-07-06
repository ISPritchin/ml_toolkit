import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("microstructure", values, params)


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
    arrs, sfxs = _run(values, {"windows": [6]})
    assert _get(arrs, sfxs, "surprise_w6")[-1] == pytest.approx(expected_surprise, abs=0.01)


def test_constant_series_predictability_high():
    # CV=0 → predictability=1/(1+0)=1.0
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "predictability_w6")[-1] == pytest.approx(1.0, abs=1e-3)


def test_constant_series_surprise_zero():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "surprise_w6")[-1] == pytest.approx(0.0, abs=1e-4)


def test_surprise_dir_positive_when_above_mean():
    arrs, sfxs = _run([10, 10, 10, 10, 10, 100], {"windows": [6]})
    assert _get(arrs, sfxs, "surprise_dir")[-1] == pytest.approx(1.0)


def test_surprise_dir_negative_when_below_mean():
    arrs, sfxs = _run([100, 100, 100, 100, 100, 10], {"windows": [6]})
    assert _get(arrs, sfxs, "surprise_dir")[-1] == pytest.approx(-1.0)

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
