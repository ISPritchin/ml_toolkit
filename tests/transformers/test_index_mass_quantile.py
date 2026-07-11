import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('index_mass_quantile', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    # [0,0,10,10,10,10], total=40, running=[0,0,10,20,30,40]
    # q25 порог 10 -> i=2 -> 2/5=0.4; q50 порог 20 -> i=3 -> 0.6; q75 порог 30 -> i=4 -> 0.8
    arrs, sfxs = _run([0, 0, 10, 10, 10, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'q25_w6')[-1] == pytest.approx(0.4, abs=1e-6)
    assert _get(arrs, sfxs, 'q50_w6')[-1] == pytest.approx(0.6, abs=1e-6)
    assert _get(arrs, sfxs, 'q75_w6')[-1] == pytest.approx(0.8, abs=1e-6)


def test_uniform_series_mass_centered():
    # равномерный ряд: масса набирается линейно -> q50 около середины окна
    arrs, sfxs = _run([10, 10, 10, 10, 10, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'q50_w6')[-1] == pytest.approx(0.4, abs=1e-6)


def test_all_zero_window_is_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'q25_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'q50_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'q75_w6')[-1] == pytest.approx(0.0)


def test_mass_front_loaded_gives_low_q50():
    # почти вся масса в начале окна -> q50 маленький
    arrs, sfxs = _run([100, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'q50_w6')[-1] == pytest.approx(0.0, abs=1e-6)
