import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('growth_since_start', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_value_from_docstring():
    # [0,10,30]: first_nonzero=10, at t=2: (30-10)/10=2.0
    arrs, sfxs = _run([0, 10, 30])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(2.0, abs=1e-4)


def test_back_to_start_level_returns_zero():
    # [10,20,10]: first=10, at end (10-10)/10=0
    arrs, sfxs = _run([10, 20, 10])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(0.0, abs=1e-4)


def test_before_first_activity_zero():
    # [0,0,10]: first two rows are 0 → growth=0
    arrs, sfxs = _run([0, 0, 10])
    assert _get(arrs, sfxs, '')[0] == pytest.approx(0.0)
    assert _get(arrs, sfxs, '')[1] == pytest.approx(0.0)


def test_all_zeros_growth_zero():
    arrs, sfxs = _run([0, 0, 0, 0])
    for v in _get(arrs, sfxs, ''):
        assert v == pytest.approx(0.0)


def test_decline_below_start_negative():
    # [50,20,10]: first=50, at end (10-50)/50=-0.8
    arrs, sfxs = _run([50, 20, 10])
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(-0.8, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    # (35-50)/50=-0.3, first nonzero=50 at pos=0
    assert _get(arrs, sfxs, '')[-1] == pytest.approx(-0.3, abs=0.001)


def test_full_output_vector():
    # 9 значений, params={}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, '') == pytest.approx([0.0, -1.0, 1.0, 0.5, -1.0, 1.5, -0.333333, -1.0, 2.333333], abs=1e-6)
