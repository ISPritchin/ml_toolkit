import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('permutation_entropy', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_monotonic_series_has_zero_entropy():
    # строго монотонный рост -> все тройки одного ordinal-паттерна -> h=0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-9)


def test_two_pattern_alternation_matches_independent_entropy():
    # [10,20,10,20,10,20]: тройки чередуют ровно 2 паттерна поровну (2/2)
    # h = -(0.5*ln0.5 + 0.5*ln0.5) = ln2, normalized = ln2/ln6 = 0.386853
    arrs, sfxs = _run([10, 20, 10, 20, 10, 20], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.386853, abs=1e-5)


def test_insufficient_history_is_zero():
    # ws < 3 -> недостаточно для одной тройки -> 0
    arrs, sfxs = _run([5, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0)


def test_output_bounded_in_unit_interval():
    arrs, sfxs = _run([3, 7, 2, 9, 1, 8, 4, 6, 5, 10, 0, 12], {'windows': [12]})
    out = _get(arrs, sfxs, 'w12')
    assert (out >= -1e-9).all()
    assert (out <= 1.0 + 1e-9).all()


def test_ties_are_handled_deterministically():
    # повторяющиеся/нулевые значения не должны падать или давать NaN
    arrs, sfxs = _run([0, 0, 0, 5, 0, 0], {'windows': [6]})
    out = _get(arrs, sfxs, 'w6')
    assert all(v == v for v in out)  # no NaN
