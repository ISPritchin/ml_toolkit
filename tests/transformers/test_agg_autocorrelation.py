import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('agg_autocorrelation', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    # строго линейный ряд: r_lag1=r_lag2=1.0 -> mean=abs_mean=1.0, std=0.0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6], 'max_lag': 2})
    assert _get(arrs, sfxs, 'mean_w6')[-1] == pytest.approx(1.0, abs=1e-6)
    assert _get(arrs, sfxs, 'abs_mean_w6')[-1] == pytest.approx(1.0, abs=1e-6)
    assert _get(arrs, sfxs, 'std_w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_oscillation_cancels_in_mean_but_not_in_abs_mean():
    # [10,20,10,20,10,20]: r_lag1=-1.0 (строго противофазно), r_lag2=+1.0
    # mean=(−1+1)/2=0, abs_mean=(1+1)/2=1, std population = 1.0
    arrs, sfxs = _run([10, 20, 10, 20, 10, 20], {'windows': [6], 'max_lag': 2})
    assert _get(arrs, sfxs, 'mean_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert _get(arrs, sfxs, 'abs_mean_w6')[-1] == pytest.approx(1.0, abs=1e-6)
    assert _get(arrs, sfxs, 'std_w6')[-1] == pytest.approx(1.0, abs=1e-6)


def test_insufficient_history_is_zero():
    # ws < 3 -> ни один лаг (>=1) не даёт ws>=lag+2=3 -> все три выхода 0
    arrs, sfxs = _run([5, 10], {'windows': [6], 'max_lag': 3})
    assert _get(arrs, sfxs, 'mean_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'abs_mean_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'std_w6')[-1] == pytest.approx(0.0)


def test_single_valid_lag_gives_zero_std():
    # ws=4, max_lag=3: только lag=1 валиден (ws>=1+2=3), lag=2 требует ws>=4 (ок, valid тоже!)
    # lag=1: ws>=3 ok; lag=2: ws>=4 ok; lag=3: ws>=5 не ok -> 2 валидных лага, не 1.
    # Возьмём ws=3, max_lag=3: lag=1 valid (ws>=3), lag=2 invalid (ws>=4 нет), lag=3 invalid.
    arrs, sfxs = _run([10, 20, 30], {'windows': [3], 'max_lag': 3})
    assert _get(arrs, sfxs, 'std_w3')[-1] == pytest.approx(0.0)
