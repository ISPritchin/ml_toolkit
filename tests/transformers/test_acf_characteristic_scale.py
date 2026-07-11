import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('acf_characteristic_scale', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    arrs, sfxs = _run([10, 9, 7, 4, 3, 4, 7, 9], {'windows': [8], 'max_lag': 4})
    assert _get(arrs, sfxs, 'f1ecac_w8')[-1] == pytest.approx(2.0)
    assert _get(arrs, sfxs, 'first_min_ac_w8')[-1] == pytest.approx(4.0)


def test_insufficient_history_is_zero():
    # ws < 3 -> lag=1 не измерим -> 0 (не censored, а "недостаточно истории")
    arrs, sfxs = _run([5, 10], {'windows': [8], 'max_lag': 4})
    assert _get(arrs, sfxs, 'f1ecac_w8')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'first_min_ac_w8')[-1] == pytest.approx(0.0)


def test_persistent_linear_series_is_censored_at_max_lag():
    # строго монотонный рост: r_lag остаётся высоким на всех лагах -> f1ecac censored
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60, 70, 80], {'windows': [8], 'max_lag': 3})
    assert _get(arrs, sfxs, 'f1ecac_w8')[-1] == pytest.approx(3.0)
