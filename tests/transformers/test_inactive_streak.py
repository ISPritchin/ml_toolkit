import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('inactive_streak', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_example_from_docstring():
    # [10,0,0,10,0]: current=1 (last zero), max=2 (prev streak of 2)
    arrs, sfxs = _run([10, 0, 0, 10, 0])
    assert _get(arrs, sfxs, 'current')[-1] == pytest.approx(1.0)
    assert _get(arrs, sfxs, 'max')[-1] == pytest.approx(2.0)


def test_active_current_streak_zero():
    # [10,20,30]: always active → current=0 always
    arrs, sfxs = _run([10, 20, 30])
    assert _get(arrs, sfxs, 'current')[-1] == pytest.approx(0.0)


def test_all_zeros_current_grows():
    # [0,0,0,0,0]: current=5, max=5 at end
    arrs, sfxs = _run([0, 0, 0, 0, 0])
    assert _get(arrs, sfxs, 'current')[-1] == pytest.approx(5.0)
    assert _get(arrs, sfxs, 'max')[-1] == pytest.approx(5.0)


def test_max_remembers_historical_maximum():
    # [0,0,0,10,0]: max=3 (first run), current=1 (last zero)
    arrs, sfxs = _run([0, 0, 0, 10, 0])
    assert _get(arrs, sfxs, 'max')[-1] == pytest.approx(3.0)
    assert _get(arrs, sfxs, 'current')[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    # last value=35≠0 → no streak
    assert _get(arrs, sfxs, 'current')[-1] == pytest.approx(0.0, abs=1e-06)
    # longest zero run is 2 ({4,5} or {12,13})
    assert _get(arrs, sfxs, 'max')[-1] == pytest.approx(2.0, abs=1e-06)


def test_full_output_vector():
    # 9 значений, params={}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'current') == pytest.approx([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'max') == pytest.approx([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], abs=1e-6)
