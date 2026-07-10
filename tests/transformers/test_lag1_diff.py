import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('lag1_diff', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_values_from_docstring():
    # [50,60,65]: diff=5, log_diff≈0.079, pct_change≈0.0833
    arrs, sfxs = _run([50, 60, 65])
    assert _get(arrs, sfxs, 'diff')[-1] == pytest.approx(5.0)
    assert _get(arrs, sfxs, 'log_diff')[-1] == pytest.approx(math.log1p(65) - math.log1p(60), abs=1e-6)
    assert _get(arrs, sfxs, 'pct_change')[-1] == pytest.approx(5 / 60, abs=1e-6)


def test_constant_series_all_zeros():
    arrs, sfxs = _run([30, 30, 30])
    assert _get(arrs, sfxs, 'diff')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'log_diff')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'pct_change')[-1] == pytest.approx(0.0)


def test_first_row_always_zero():
    arrs, sfxs = _run([100, 200])
    assert _get(arrs, sfxs, 'diff')[0] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'log_diff')[0] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'pct_change')[0] == pytest.approx(0.0)


def test_transition_from_zero_to_nonzero():
    # [0,100]: diff=100, log_diff=log1p(100)-log1p(0)=log1p(100)
    # pct_change при нулевой базе не определён → 0 (раньше 100/eps ~ 1e11)
    arrs, sfxs = _run([0, 100])
    assert _get(arrs, sfxs, 'diff')[-1] == pytest.approx(100.0)
    assert _get(arrs, sfxs, 'log_diff')[-1] == pytest.approx(math.log1p(100), abs=1e-6)
    assert _get(arrs, sfxs, 'pct_change')[-1] == pytest.approx(0.0, abs=1e-9)


def test_declining_series_negative_diff():
    arrs, sfxs = _run([100, 60, 30])
    assert _get(arrs, sfxs, 'diff')[-1] == pytest.approx(-30.0)
    assert _get(arrs, sfxs, 'pct_change')[-1] < 0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    assert math.isfinite(_get(arrs, sfxs, 'diff')[-1]), 'diff must be finite'
    assert _get(arrs, sfxs, 'diff')[-1] == pytest.approx(35.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'log_diff')[-1]), 'log_diff must be finite'
    assert _get(arrs, sfxs, 'log_diff')[-1] == pytest.approx(3.58351893845611, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'pct_change')[-1]), 'pct_change must be finite'
    # v_prev = 0 → изменение в % не определено → 0 (раньше 3.5e10)
    assert _get(arrs, sfxs, 'pct_change')[-1] == pytest.approx(0.0, abs=1e-9)


def test_full_output_vector():
    # 9 значений, params={}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'diff') == pytest.approx([0.0, -6.0, 12.0, -3.0, -9.0, 15.0, -11.0, -4.0, 20.0], abs=1e-6)
    assert _get(arrs, sfxs, 'log_diff') == pytest.approx([0.0, -1.94591, 2.564949, -0.262364, -2.302585, 2.772589, -1.163151, -1.609438, 3.044522], abs=1e-6)
    assert _get(arrs, sfxs, 'pct_change') == pytest.approx([0.0, -1.0, 0.0, -0.25, -1.0, 0.0, -0.733333, -1.0, 0.0], abs=1e-6)
