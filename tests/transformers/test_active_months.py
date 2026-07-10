import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('active_months', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_count():
    # [10,0,5,0,8,3] w=6: nonzero count = 4
    arrs, sfxs = _run([10, 0, 5, 0, 8, 3], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(4.0)


def test_all_zeros_count_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0)


def test_all_active_count_equals_window():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(6.0)


def test_partial_window_at_start():
    # At row 0, only 1 value available, window clips to 1
    arrs, sfxs = _run([10, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[0] == pytest.approx(1.0)


def test_shorter_window_clips_to_available_history():
    # First 3 rows: [10,0,5], only 3 rows available for w6 → count nonzero = 2
    arrs, sfxs = _run([10, 0, 5, 0, 8, 3], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[2] == pytest.approx(2.0)

def test_full_output_vector():
    # 8 значений, чередование нулей и ненулей, окно w=4
    # values: [5, 0, 0, 8, 0, 3, 3, 0]
    # pos=0 (w_eff=1): [5]           -> 1
    # pos=1 (w_eff=2): [5,0]         -> 1
    # pos=2 (w_eff=3): [5,0,0]       -> 1
    # pos=3 (w_eff=4): [5,0,0,8]     -> 2
    # pos=4 (w_eff=4): [0,0,8,0]     -> 1
    # pos=5 (w_eff=4): [0,8,0,3]     -> 2
    # pos=6 (w_eff=4): [8,0,3,3]     -> 3
    # pos=7 (w_eff=4): [0,3,3,0]     -> 2
    arrs, sfxs = _run([5, 0, 0, 8, 0, 3, 3, 0], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([1.0, 1.0, 1.0, 2.0, 1.0, 2.0, 3.0, 2.0])


def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # 3 non-zeros in last 6: 10,60,35
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(3.0, abs=1e-06)
