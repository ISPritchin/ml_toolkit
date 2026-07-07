import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('zero_share', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_share():
    # [10,0,0,10,0,10] w=6: zeros at idx 1,2,4 → 3/6=0.5
    arrs, sfxs = _run([10, 0, 0, 10, 0, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.5)


def test_all_zeros_share_one():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0)


def test_no_zeros_share_zero():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0)


def test_short_window_counts_only_recent():
    # [10,0,0,10,0,10]: last 3 values [10,0,10] → 1/3 zeros for w3
    arrs, sfxs = _run([10, 0, 0, 10, 0, 10], {'windows': [3]})
    assert _get(arrs, sfxs, 'w3')[-1] == pytest.approx(1 / 3, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # 3 zeros in last 6 = 0.5
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.5, abs=1e-06)
