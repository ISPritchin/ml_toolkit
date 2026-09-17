import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('gini', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_gini():
    # [0,10,30,60] w=4: total=100
    # sum|v_i-v_j| all pairs: 10+30+60+10+20+50+30+20+30+60+50+30 counted twice
    # unique pairs: (0,10)=10,(0,30)=30,(0,60)=60,(10,30)=20,(10,60)=50,(30,60)=30 → 200
    # all pairs (both directions): 400
    # gini = 400 / (2*4*100) = 0.5
    arrs, sfxs = _run([0, 10, 30, 60], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(0.5, abs=1e-4)


def test_all_equal_gini_zero():
    # All values equal → all |v_i-v_j|=0 → gini=0
    arrs, sfxs = _run([20, 20, 20, 20], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(0.0, abs=1e-4)


def test_all_zeros_gini_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_concentrated_on_one_value():
    # [0,0,0,100] w=4: total=100
    # pairs with 100: (0,100)×3×2=600; pairs (0,0): 0
    # gini = 600 / (2*4*100) = 0.75
    arrs, sfxs = _run([0, 0, 0, 100], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(0.75, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.6587301587301587, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 0.5, 0.444444, 0.361111, 0.535714, 0.333333, 0.446429, 0.644737, 0.455128], abs=1e-6)


def test_dilations_defaults_to_1_and_matches_plain_windows():
    values = [0, 10, 30, 60]
    arrs_plain, sfxs_plain = _run(values, {'windows': [4]})
    arrs_explicit, sfxs_explicit = _run(values, {'windows': [4], 'dilations': [1]})
    assert sfxs_plain == sfxs_explicit == ['w4']
    assert arrs_plain[0].tolist() == arrs_explicit[0].tolist()


def test_dilation_2_reads_every_other_point():
    # row_idx=6 (last), ws=4, dilation=2 -> idx 0,2,4,6 -> [0,10,30,60] -> same known gini=0.5
    values = [0, 999, 10, 999, 30, 999, 60]
    arrs, sfxs = _run(values, {'windows': [4], 'dilations': [1, 2]})
    assert sfxs == ['w4', 'w4_d2']
    assert _get(arrs, sfxs, 'w4_d2')[-1] == pytest.approx(0.5, abs=1e-6)
