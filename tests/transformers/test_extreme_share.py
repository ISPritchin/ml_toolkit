import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('extreme_share', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_extreme_and_balance():
    # [10,10,10,10,10,40] w=6: mean=15, std=sqrt(125)≈11.18
    # 1.5*std≈16.77; only 40 is extreme (|40-15|=25>16.77)
    # extreme = 1/6; balance = 1/6 - 0.5 = -1/3
    arrs, sfxs = _run([10, 10, 10, 10, 10, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'extreme_w6')[-1] == pytest.approx(1 / 6, abs=1e-4)
    assert _get(arrs, sfxs, 'balance_w6')[-1] == pytest.approx(-1 / 3, abs=1e-4)


def test_all_zeros_no_extremes():
    # std=0 → threshold=0; |0-0|=0, not >0 → extreme=0
    # no values above mean=0 → balance=-0.5
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'extreme_w6')[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, 'balance_w6')[-1] == pytest.approx(-0.5)


def test_uniform_series_no_extremes():
    # constant → std=0, no deviations, extreme=0, balance=-0.5 (none strictly above mean)
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {'windows': [6]})
    assert _get(arrs, sfxs, 'extreme_w6')[-1] == pytest.approx(0.0)


def test_symmetric_oscillation_balance_near_zero():
    # [10,50,10,50,10,50]: half above, half below → balance=0
    arrs, sfxs = _run([10, 50, 10, 50, 10, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'balance_w6')[-1] == pytest.approx(0.0, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'extreme_w6')[-1]), 'extreme_w6 must be finite'
    assert _get(arrs, sfxs, 'extreme_w6')[-1] == pytest.approx(0.16666666666666666, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'balance_w6')[-1]), 'balance_w6 must be finite'
    assert _get(arrs, sfxs, 'balance_w6')[-1] == pytest.approx(-0.16666666666666669, rel=1e-4)
