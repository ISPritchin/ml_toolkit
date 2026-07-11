import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('c3', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    # [1,2,4,2,1], lag=1, w=5: triples (1,2,4)->8,(2,4,2)->16,(4,2,1)->8, mean=10.667
    # mean_w=2.0, std_w=sqrt(6/5)=1.095 -> c3 = 10.667/1.095**3 = 8.114...
    arrs, sfxs = _run([1, 2, 4, 2, 1], {'lag_window_pairs': [[1, 5]]})
    assert _get(arrs, sfxs, 'lag1_w5')[-1] == pytest.approx(8.114408, abs=1e-4)


def test_insufficient_history_is_zero():
    # ws <= 2*lag на ранних позициях -> недостаточно троек -> 0
    arrs, sfxs = _run([5, 10, 15], {'lag_window_pairs': [[1, 12]]})
    out = _get(arrs, sfxs, 'lag1_w12')
    # позиция 0 и 1: ws=1,2 <= 2*1=2 -> 0 троек -> 0
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(0.0)


def test_constant_series_gives_zero_via_safe_ratio():
    # std_w = 0 -> safe_ratio возвращает 0 (не взрыв на нулевом std)
    arrs, sfxs = _run([7, 7, 7, 7, 7], {'lag_window_pairs': [[1, 5]]})
    assert _get(arrs, sfxs, 'lag1_w5')[-1] == pytest.approx(0.0)


def test_multiple_pairs_produce_distinct_suffixes():
    arrs, sfxs = _run([1, 2, 4, 2, 1, 3, 5, 2, 1, 4, 6, 3], {'lag_window_pairs': [[1, 6], [1, 12]]})
    assert set(sfxs) == {'lag1_w6', 'lag1_w12'}
    assert len(_get(arrs, sfxs, 'lag1_w6')) == 12
    assert len(_get(arrs, sfxs, 'lag1_w12')) == 12
