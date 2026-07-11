import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('dfa', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    arrs, sfxs = _run([10, 12, 11, 13, 12, 14, 13, 15, 14, 16, 15, 17], {'windows': [12]})
    assert _get(arrs, sfxs, 'alpha_w12')[-1] == pytest.approx(0.9487585, abs=1e-5)


def test_pure_linear_trend_gives_nonstationary_high_alpha():
    arrs, sfxs = _run([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21], {'windows': [12]})
    assert _get(arrs, sfxs, 'alpha_w12')[-1] > 1.2


def test_alternating_series_gives_low_alpha():
    arrs, sfxs = _run([10, 20, 10, 20, 10, 20, 10, 20, 10, 20, 10, 20], {'windows': [12]})
    assert _get(arrs, sfxs, 'alpha_w12')[-1] < 0.1


def test_constant_series_is_zero():
    arrs, sfxs = _run([10] * 12, {'windows': [12]})
    assert _get(arrs, sfxs, 'alpha_w12')[-1] == pytest.approx(0.0)


def test_insufficient_history_is_zero():
    # ws < 8 -> недостаточно для 2 масштабов -> 0
    arrs, sfxs = _run([5, 10, 15, 20, 25], {'windows': [12]})
    assert _get(arrs, sfxs, 'alpha_w12')[-1] == pytest.approx(0.0)
