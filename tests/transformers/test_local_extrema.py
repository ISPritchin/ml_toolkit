import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('local_extrema', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_oscillating_count():
    # [10,30,20,40,30,50] w=6: internal offsets 1..4
    # offset1=30: peak (>10 and >20)
    # offset2=20: trough (<30 and <40)
    # offset3=40: peak (>20 and >30)
    # offset4=30: trough (<40 and <50)
    # → 4 extrema
    arrs, sfxs = _run([10, 30, 20, 40, 30, 50], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(4.0)


def test_monotone_no_extrema():
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0)


def test_constant_no_extrema():
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0)


def test_single_peak_in_middle():
    # [10,10,50,10,10,10] w=6: offset2=50 is peak → 1 extremum
    arrs, sfxs = _run([10, 10, 50, 10, 10, 10], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(2.0, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 0.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0], abs=1e-6)
