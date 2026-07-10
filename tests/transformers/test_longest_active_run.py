import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('longest_active_run', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_longest_run():
    # [10,10,0,10,10,10] w=6: runs of length 2 and 3 → max=3
    arrs, sfxs = _run([10, 10, 0, 10, 10, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(3.0)


def test_all_zeros_max_run_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0)


def test_continuous_active_run_equals_window():
    arrs, sfxs = _run([5, 10, 15, 20, 25, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(6.0)


def test_alternating_zeros_max_run_one():
    # [10,0,5,0,8,0] → each active run has length 1
    arrs, sfxs = _run([10, 0, 5, 0, 8, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0)


def test_single_active_at_end():
    # [0,0,0,0,0,10]: only 1 active at end → run=1
    arrs, sfxs = _run([0, 0, 0, 0, 0, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # no two consecutive non-zeros in [10,0,60,0,0,35]
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0, abs=1e-06)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], abs=1e-6)
