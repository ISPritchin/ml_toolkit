import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('auto_period', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    values = [10, 30, 10, 30, 10, 30, 10, 30, 10, 30, 10, 30]
    arrs, sfxs = _run(values, {'windows': [12], 'min_lag': 2, 'max_lag': 4})
    assert _get(arrs, sfxs, 'period_w12')[-1] == pytest.approx(2.0)
    assert _get(arrs, sfxs, 'strength_w12')[-1] == pytest.approx(1.0)


def test_ties_prefer_smallest_lag_not_harmonic():
    # период 2 и его гармоника (лаг 4) дают одинаковую r=1.0 -> должен выбраться лаг 2
    values = [10, 30, 10, 30, 10, 30, 10, 30, 10, 30, 10, 30]
    arrs, sfxs = _run(values, {'windows': [12], 'min_lag': 2, 'max_lag': 6})
    assert _get(arrs, sfxs, 'period_w12')[-1] == pytest.approx(2.0)


def test_insufficient_history_is_zero():
    arrs, sfxs = _run([5, 10], {'windows': [12], 'max_lag': 4})
    assert _get(arrs, sfxs, 'period_w12')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'strength_w12')[-1] == pytest.approx(0.0)


def test_default_min_lag_used_when_omitted():
    values = [10, 30, 10, 30, 10, 30, 10, 30, 10, 30, 10, 30]
    arrs, sfxs = _run(values, {'windows': [12], 'max_lag': 4})
    assert _get(arrs, sfxs, 'period_w12')[-1] == pytest.approx(2.0)
