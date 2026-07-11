import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('transition_matrix', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    arrs, sfxs = _run([10, 10, 50, 50, 90, 90], {'windows': [6]})
    assert _get(arrs, sfxs, 'stickiness_w6')[-1] == pytest.approx(0.6, abs=1e-6)
    assert _get(arrs, sfxs, 'trans_entropy_w6')[-1] == pytest.approx(0.7324868, abs=1e-5)


def test_constant_series_is_maximally_sticky():
    arrs, sfxs = _run([10, 10, 10, 10, 10, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'stickiness_w6')[-1] == pytest.approx(1.0, abs=1e-6)
    assert _get(arrs, sfxs, 'trans_entropy_w6')[-1] == pytest.approx(0.0, abs=1e-6)


def test_insufficient_history_is_zero():
    arrs, sfxs = _run([5, 10, 15], {'windows': [6]})
    assert _get(arrs, sfxs, 'stickiness_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'trans_entropy_w6')[-1] == pytest.approx(0.0)
