import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('client_age', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_young_client_flag_set():
    # new_client_flag depends only on position_within_entity (< new_client_months=3),
    # not on the values themselves — a single entity of length 6 gets position 0..5.
    arrs, sfxs = _run([0, 0, 1, 1, 0, 0])
    flag = _get(arrs, sfxs, 'new_client_flag')
    assert flag[0] == pytest.approx(1.0)
    assert flag[1] == pytest.approx(1.0)
    assert flag[2] == pytest.approx(1.0)
    assert flag[3] == pytest.approx(0.0)
    assert flag[4] == pytest.approx(0.0)
    assert flag[5] == pytest.approx(0.0)


def test_normalized_age_formula():
    # months_since_start_norm = pos/(pos+12)
    arrs, sfxs = _run([10] * 13)
    norm = _get(arrs, sfxs, 'months_since_start_norm')
    assert norm[0] == pytest.approx(0.0)          # 0/(0+12)=0
    assert norm[12] == pytest.approx(12 / 24)     # 12/(12+12)=0.5


def test_norm_monotone_increases():
    arrs, sfxs = _run([1] * 24)
    norm = _get(arrs, sfxs, 'months_since_start_norm')
    assert all(norm[i] <= norm[i + 1] for i in range(len(norm) - 1))

def test_with_mixed_zeros():
    # client_age depends only on position_within_entity, not on the values — the zero/non-zero
    # pattern below is irrelevant to the result; only the series length (→ last position = 14)
    # determines new_client_flag[-1] and months_since_start_norm[-1].
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    assert math.isfinite(_get(arrs, sfxs, 'new_client_flag')[-1]), 'new_client_flag must be finite'
    assert _get(arrs, sfxs, 'new_client_flag')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'months_since_start_norm')[-1]), 'months_since_start_norm must be finite'
    assert _get(arrs, sfxs, 'months_since_start_norm')[-1] == pytest.approx(0.5384615384615384, rel=1e-4)


def test_full_output_vector():
    # 10 значений, params={}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11]
    arrs, sfxs = _run(values)
    assert _get(arrs, sfxs, 'new_client_flag') == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'months_since_start_norm') == pytest.approx([0.0, 0.076923, 0.142857, 0.2, 0.25, 0.294118, 0.333333, 0.368421, 0.4, 0.428571], abs=1e-6)
