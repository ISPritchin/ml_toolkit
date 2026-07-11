import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('automutual_info', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    arrs, sfxs = _run([10, 10, 50, 50, 90, 90], {'lag_window_pairs': [[1, 6]]})
    assert _get(arrs, sfxs, 'lag1_w6')[-1] == pytest.approx(0.455486, abs=1e-5)


def test_constant_series_has_zero_mutual_information():
    # единственное состояние -> все маргиналы вырождены, MI не определена -> 0
    arrs, sfxs = _run([10, 10, 10, 10, 10, 10], {'lag_window_pairs': [[1, 6]]})
    assert _get(arrs, sfxs, 'lag1_w6')[-1] == pytest.approx(0.0)


def test_insufficient_history_is_zero():
    arrs, sfxs = _run([5, 10, 15], {'lag_window_pairs': [[1, 6]]})
    assert _get(arrs, sfxs, 'lag1_w6')[-1] == pytest.approx(0.0)


def test_multiple_pairs_produce_distinct_suffixes():
    values = [1, 2, 4, 2, 1, 3, 5, 2, 1, 4, 6, 3]
    arrs, sfxs = _run(values, {'lag_window_pairs': [[1, 6], [1, 12]]})
    assert set(sfxs) == {'lag1_w6', 'lag1_w12'}
    assert len(_get(arrs, sfxs, 'lag1_w6')) == 12
