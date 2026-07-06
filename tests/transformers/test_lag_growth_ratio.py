import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("lag_growth_ratio", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_growth_ratio():
    # [10,20,30,40] lag=3: ratio = 40/10 - 1 = 3.0
    arrs, sfxs = _run([10, 20, 30, 40], {"lags": [3]})
    assert _get(arrs, sfxs, "lag3")[-1] == pytest.approx(3.0, abs=1e-4)


def test_no_change_ratio_zero():
    # [20,20,20,20] lag=3: ratio = 20/20 - 1 = 0
    arrs, sfxs = _run([20, 20, 20, 20], {"lags": [3]})
    assert _get(arrs, sfxs, "lag3")[-1] == pytest.approx(0.0, abs=1e-4)


def test_zero_before_lag_available():
    # lag=3 requires pos>=3; at pos=0,1,2 → ratio=0
    arrs, sfxs = _run([10, 20, 30, 40], {"lags": [3]})
    result = _get(arrs, sfxs, "lag3")
    assert result[0] == pytest.approx(0.0)
    assert result[2] == pytest.approx(0.0)


def test_zero_lag_value_undefined_ratio_zero():
    # v_lag=0 → рост не определён → 0 (раньше v/eps ~ 5e10)
    arrs, sfxs = _run([0, 0, 0, 50], {"lags": [3]})
    assert _get(arrs, sfxs, "lag3")[-1] == pytest.approx(0.0, abs=1e-9)


def test_decline_gives_negative_ratio():
    # [100,80,60,40] lag=3: 40/100-1=-0.6
    arrs, sfxs = _run([100, 80, 60, 40], {"lags": [3]})
    assert _get(arrs, sfxs, "lag3")[-1] == pytest.approx(-0.6, abs=1e-4)

# test_with_mixed_zeros skipped for lag_growth_ratio: 'lags'
