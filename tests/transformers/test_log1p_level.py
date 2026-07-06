import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("log1p_level", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_value():
    # v=100: sign(100)*log1p(100)=ln(101)≈4.6151
    arrs, sfxs = _run([100])
    assert _get(arrs, sfxs, "")[-1] == pytest.approx(math.log1p(100), abs=1e-6)


def test_zero_maps_to_zero():
    arrs, sfxs = _run([0])
    assert _get(arrs, sfxs, "")[-1] == pytest.approx(0.0)


def test_large_value_compresses_scale():
    # log1p(1000) should be much less than 1000
    arrs, sfxs = _run([1000])
    val = _get(arrs, sfxs, "")[-1]
    assert val == pytest.approx(math.log1p(1000), abs=1e-6)
    assert val < 10  # compressed from 1000


def test_series_monotone_positive():
    arrs, sfxs = _run([0, 10, 100, 1000])
    result = _get(arrs, sfxs, "")
    for i in range(len(result) - 1):
        assert result[i] <= result[i + 1]


def test_zeros_in_series_give_zero():
    arrs, sfxs = _run([10, 0, 50, 0])
    result = _get(arrs, sfxs, "")
    assert result[1] == pytest.approx(0.0)
    assert result[3] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, '')[-1]), ' must be finite'
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(3.58351893845611, rel=1e-4)
