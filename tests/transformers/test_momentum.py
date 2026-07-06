import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("momentum", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_value_from_docstring():
    # [10,20,30,40,50,60] h=3: recent_mean=50, prior_mean=20 → 50/20-1=1.5
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {"half_windows": [3]})
    assert _get(arrs, sfxs, "h3")[-1] == pytest.approx(1.5, abs=1e-4)


def test_equal_halves_momentum_zero():
    # recent_mean = prior_mean → 1 - 1 = 0
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {"half_windows": [3]})
    assert _get(arrs, sfxs, "h3")[-1] == pytest.approx(0.0, abs=1e-4)


def test_all_zeros_momentum_undefined_zero():
    # prior_mean=0 → моментум не определён → 0 (раньше давал -1,
    # неотличимую от реального «полного прекращения» при живой базе)
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"half_windows": [3]})
    assert _get(arrs, sfxs, "h3")[-1] == pytest.approx(0.0, abs=1e-9)


def test_declining_series_negative_momentum():
    # [60,50,40,30,20,10] h=3: recent=20, prior=50 → 20/50-1=-0.6
    arrs, sfxs = _run([60, 50, 40, 30, 20, 10], {"half_windows": [3]})
    assert _get(arrs, sfxs, "h3")[-1] == pytest.approx(-0.6, abs=1e-4)


def test_not_enough_history_stays_zero():
    # h=3 requires pos>=5; first 5 rows → 0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {"half_windows": [3]})
    for i in range(5):
        assert _get(arrs, sfxs, "h3")[i] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'half_windows': [3]})
    assert math.isfinite(_get(arrs, sfxs, 'h3')[-1]), 'h3 must be finite'
    assert _get(arrs, sfxs, 'h3')[-1] == pytest.approx(-0.5000000000214286, rel=1e-4)
