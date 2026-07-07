import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('rolling_sum', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_sum():
    # [10,20,30,40] w=3: last 3 → 20+30+40=90
    arrs, sfxs = _run([10, 20, 30, 40], {'windows': [3]})
    assert _get(arrs, sfxs, 'w3')[-1] == pytest.approx(90.0)


def test_all_zeros_sum_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0)


def test_single_spike_in_window():
    # [0,0,0,0,0,100] w=6: sum=100
    arrs, sfxs = _run([0, 0, 0, 0, 0, 100], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(100.0)


def test_partial_window_at_second_row():
    # Row 1: window clips to 2 → [10,20], sum=30
    arrs, sfxs = _run([10, 20, 30, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[1] == pytest.approx(30.0)


def test_sum_additive_across_windows():
    values = [10, 20, 30, 40, 50, 60]
    arrs, sfxs = _run(values, {'windows': [3, 6]})
    # sum_w6 at last row = sum of all 6
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(210.0)
    # sum_w3 at last row = last 3
    assert _get(arrs, sfxs, 'w3')[-1] == pytest.approx(150.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # sum of [10,0,60,0,0,35]
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(105.0, abs=1e-06)
