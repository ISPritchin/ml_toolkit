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


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [0, 0, 10, 10, 10, 10]
    arrs_plain, sfxs_plain = _run(values, {'windows': [6]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [6], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['q25_w6', 'q50_w6', 'q75_w6']
    for a, b in zip(arrs_plain, arrs_explicit, strict=True):
        assert a.tolist() == b.tolist()


def test_dilation_2_reads_every_other_point():
    # even indices carry [0,0,10,10,10,10] -> known q25=0.4, q50=0.6, q75=0.8
    values = [0, 999, 0, 999, 10, 999, 10, 999, 10, 999, 10]
    arrs, sfxs = _run(values, {'windows': [6], 'dilations': [1, 2]})
    assert sfxs == ['q25_w6', 'q50_w6', 'q75_w6', 'q25_w6_d2', 'q50_w6_d2', 'q75_w6_d2']
    assert _get(arrs, sfxs, 'q25_w6_d2')[-1] == pytest.approx(0.4, abs=1e-6)
    assert _get(arrs, sfxs, 'q50_w6_d2')[-1] == pytest.approx(0.6, abs=1e-6)
    assert _get(arrs, sfxs, 'q75_w6_d2')[-1] == pytest.approx(0.8, abs=1e-6)
