import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('entropy', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_uniform_distribution_max_entropy():
    # [10,10,10,10] w=4: all equal shares → entropy=1.0 (maximum)
    arrs, sfxs = _run([10, 10, 10, 10], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(1.0, abs=1e-4)


def test_single_nonzero_entropy_zero():
    # [0,0,0,10] w=4: only one nonzero → p=1 → -p*ln(p)=0 → entropy=0
    arrs, sfxs = _run([0, 0, 0, 10], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(0.0, abs=1e-4)


def test_all_zeros_entropy_zero():
    arrs, sfxs = _run([0, 0, 0, 0], {'windows': [4]})
    assert _get(arrs, sfxs, 'w4')[-1] == pytest.approx(0.0, abs=1e-4)


def test_nonuniform_entropy_between_zero_and_one():
    # [10,10,10,70] w=4: non-uniform → 0 < entropy < 1
    arrs, sfxs = _run([10, 10, 10, 70], {'windows': [4]})
    val = _get(arrs, sfxs, 'w4')[-1]
    assert 0.0 < val < 1.0

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'w6')[-1]), 'w6 must be finite'
    assert _get(arrs, sfxs, 'w6')[-1] == pytest.approx(0.5078388381816539, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'w4') == pytest.approx([0.0, 0.0, 0.57938, 0.765247, 0.492614, 0.777293, 0.70488, 0.371244, 0.680625], abs=1e-6)
