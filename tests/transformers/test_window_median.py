import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("window_median", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_odd_window_middle_element():
    # [10,40,20,30] w=3: last 3=[40,20,30], sorted=[20,30,40], ws//2=1 → 30
    arrs, sfxs = _run([10, 40, 20, 30], {"windows": [3]})
    assert _get(arrs, sfxs, "w3")[-1] == pytest.approx(30.0)


def test_all_zeros_median_zero():
    arrs, sfxs = _run([0, 0, 0, 0], {"windows": [3]})
    assert _get(arrs, sfxs, "w3")[-1] == pytest.approx(0.0)


def test_constant_series_median_equals_value():
    arrs, sfxs = _run([15, 15, 15, 15], {"windows": [3]})
    assert _get(arrs, sfxs, "w3")[-1] == pytest.approx(15.0)


def test_zeros_dominate_median():
    # [0,0,0,50] w=4: sorted=[0,0,0,50], ws//2=2 → sorted[2]=0
    arrs, sfxs = _run([0, 0, 0, 50], {"windows": [4]})
    assert _get(arrs, sfxs, "w4")[-1] == pytest.approx(0.0)


def test_spike_at_end_does_not_shift_median_much():
    # [10,10,10,10,10,1000] w=6: sorted=[10,10,10,10,10,1000], ws//2=3 → 10
    arrs, sfxs = _run([10, 10, 10, 10, 10, 1000], {"windows": [6]})
    assert _get(arrs, sfxs, "w6")[-1] == pytest.approx(10.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # честная медиана чётного окна: sorted [0,0,0,10,35,60] → (0+10)/2 = 5
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(5.0, abs=0.0001)
