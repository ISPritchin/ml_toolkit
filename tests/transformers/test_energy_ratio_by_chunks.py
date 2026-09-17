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


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [5, 5, 10, 10, 30, 30]
    arrs_plain, sfxs_plain = _run(values, {'windows': [6]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [6], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['first_w6', 'mid_w6', 'last_w6']
    for a, b in zip(arrs_plain, arrs_explicit, strict=True):
        assert a.tolist() == b.tolist()


def test_dilation_2_reads_every_other_point():
    # even indices carry [5,5,10,10,30,30] -> known last_share 0.878
    values = [5, 999, 5, 999, 10, 999, 10, 999, 30, 999, 30]
    arrs, sfxs = _run(values, {'windows': [6], 'dilations': [1, 2]})
    assert sfxs == ['first_w6', 'mid_w6', 'last_w6', 'first_w6_d2', 'mid_w6_d2', 'last_w6_d2']
    assert _get(arrs, sfxs, 'last_w6_d2')[-1] == pytest.approx(0.878, abs=0.01)
