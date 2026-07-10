import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('recent_share', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_share():
    # [10,20,30,40,50,60] pair(3,6): S_short=150, S_long=210 → 150/210≈0.7143
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'r3_w6')[-1] == pytest.approx(150 / 210, abs=1e-4)


def test_uniform_series_gives_ratio_equal_to_fraction():
    # Uniform constant series: recent_share = short_w / long_w
    arrs, sfxs = _run([20, 20, 20, 20, 20, 20], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'r3_w6')[-1] == pytest.approx(0.5, abs=1e-4)


def test_all_zeros_share_near_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'pairs': [[3, 6]]})
    assert abs(_get(arrs, sfxs, 'r3_w6')[-1]) < 1e-3


def test_activity_concentrated_at_start_low_share():
    # [100,100,100,0,0,0]: last 3 all zero, long sum=300
    arrs, sfxs = _run([100, 100, 100, 0, 0, 0], {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'r3_w6')[-1] == pytest.approx(0.0, abs=1e-4)

# test_with_mixed_zeros skipped for recent_share: 'pairs'


def test_full_output_vector():
    # 10 значений, params={'pairs': [[3, 6]]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20, 11]
    arrs, sfxs = _run(values, {'pairs': [[3, 6]]})
    assert _get(arrs, sfxs, 'r3_w6') == pytest.approx([1.0, 1.0, 1.0, 0.777778, 0.777778, 0.571429, 0.475, 0.475, 0.5, 0.62], abs=1e-6)
