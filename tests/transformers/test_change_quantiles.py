import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('change_quantiles', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    # [5,20,22,21,23,5], ql=0.2, qh=0.8 -> корридор [5,22]
    # квалифицирующие пары: (5,20)->15, (20,22)->2, (22,21)->1
    # mean=6.0, std=sqrt(76.667-36)=6.377
    arrs, sfxs = _run([5, 20, 22, 21, 23, 5], {'windows': [6], 'ql': 0.2, 'qh': 0.8})
    assert _get(arrs, sfxs, 'mean_w6')[-1] == pytest.approx(6.0, abs=1e-6)
    assert _get(arrs, sfxs, 'std_w6')[-1] == pytest.approx(6.377042, abs=1e-4)


def test_constant_series_has_zero_change():
    arrs, sfxs = _run([10, 10, 10, 10, 10, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'mean_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'std_w6')[-1] == pytest.approx(0.0)


def test_defaults_used_when_ql_qh_omitted():
    # ql/qh не переданы -> дефолты 0.2/0.8, не должно упасть с KeyError
    arrs, sfxs = _run([5, 20, 22, 21, 23, 5], {'windows': [6]})
    assert _get(arrs, sfxs, 'mean_w6')[-1] == pytest.approx(6.0, abs=1e-6)


def test_no_qualifying_pairs_gives_zero():
    # окно из двух точек: корридор дан по этим же двум точкам, но нужно >=2 точки
    # в корридоре ОДНОВРЕМЕННО в паре; проверяем корректность на вырожденном ws=1
    arrs, sfxs = _run([42], {'windows': [6]})
    assert _get(arrs, sfxs, 'mean_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'std_w6')[-1] == pytest.approx(0.0)
