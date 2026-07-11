import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('energy_ratio_by_chunks', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    # [5,5,10,10,30,30], w=6, third=2
    # e1=50, e2=200, e3=1800, total=2050
    arrs, sfxs = _run([5, 5, 10, 10, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'first_w6')[-1] == pytest.approx(50 / 2050, abs=1e-6)
    assert _get(arrs, sfxs, 'mid_w6')[-1] == pytest.approx(200 / 2050, abs=1e-6)
    assert _get(arrs, sfxs, 'last_w6')[-1] == pytest.approx(1800 / 2050, abs=1e-6)


def test_shares_are_symmetric_for_uniform_series():
    arrs, sfxs = _run([10, 10, 10, 10, 10, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'first_w6')[-1] == pytest.approx(1 / 3, abs=1e-6)
    assert _get(arrs, sfxs, 'mid_w6')[-1] == pytest.approx(1 / 3, abs=1e-6)
    assert _get(arrs, sfxs, 'last_w6')[-1] == pytest.approx(1 / 3, abs=1e-6)


def test_insufficient_history_is_zero():
    # ws < 3 -> third < 1 -> недостаточно для деления на трети -> 0
    arrs, sfxs = _run([5, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'first_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'mid_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'last_w6')[-1] == pytest.approx(0.0)


def test_all_zero_window_is_zero_via_safe_ratio():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'last_w6')[-1] == pytest.approx(0.0)
