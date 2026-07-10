import math

import pytest

from tests.transformers.conftest import get_feature_output, run_transformer


def _run(values, params=None):
    return run_transformer('mean_deviation_shape', values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_semi_ratio():
    # [10,10,10,10,10,40] w=6: mean=15
    # up: values>mean → 40; deviations: 40-15=25; up_semi=25/1=25
    # down: values<mean → 10,10,10,10,10; deviations: |10-15|=5 each; down_semi=25/5=5
    # semi_ratio = up_semi/down_semi = 25/5 = 5.0
    arrs, sfxs = _run([10, 10, 10, 10, 10, 40], {'windows': [6]})
    assert _get(arrs, sfxs, 'semi_ratio_w6')[-1] == pytest.approx(5.0, abs=1e-4)


def test_symmetric_distribution_semi_ratio_one():
    # [10,20,30,40]: mean=25; up: 30,40→deviations 5,15→mean_up=10; down: 10,20→deviations 15,5→mean_down=10
    # semi_ratio=1.0
    arrs, sfxs = _run([10, 20, 30, 40], {'windows': [4]})
    assert _get(arrs, sfxs, 'semi_ratio_w4')[-1] == pytest.approx(1.0, abs=1e-4)


def test_constant_series_semi_ratio_zero():
    # All values = mean → no up/down deviations → semi_ratio=0
    arrs, sfxs = _run([25, 25, 25, 25, 25, 25], {'windows': [6]})
    assert _get(arrs, sfxs, 'semi_ratio_w6')[-1] == pytest.approx(0.0, abs=1e-4)


def test_all_zeros_semi_ratio_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {'windows': [6]})
    assert _get(arrs, sfxs, 'semi_ratio_w6')[-1] == pytest.approx(0.0, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'up_semi_w6')[-1]), 'up_semi_w6 must be finite'
    assert _get(arrs, sfxs, 'up_semi_w6')[-1] == pytest.approx(32.5, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'down_semi_w6')[-1]), 'down_semi_w6 must be finite'
    assert _get(arrs, sfxs, 'down_semi_w6')[-1] == pytest.approx(15.612494995995995, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'semi_ratio_w6')[-1]), 'semi_ratio_w6 must be finite'
    assert _get(arrs, sfxs, 'semi_ratio_w6')[-1] == pytest.approx(2.0816659993327993, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'max_up_z_w6')[-1]), 'max_up_z_w6 must be finite'
    assert _get(arrs, sfxs, 'max_up_z_w6')[-1] == pytest.approx(1.8735318137625536, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'max_down_z_w6')[-1]), 'max_down_z_w6 must be finite'
    assert _get(arrs, sfxs, 'max_down_z_w6')[-1] == pytest.approx(0.7714542762551692, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'dev_asym_w6')[-1]), 'dev_asym_w6 must be finite'
    assert _get(arrs, sfxs, 'dev_asym_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'cross_count_w6')[-1]), 'cross_count_w6 must be finite'
    assert _get(arrs, sfxs, 'cross_count_w6')[-1] == pytest.approx(3.0, rel=1e-4)


def test_full_output_vector():
    # 9 значений, params={'windows': [4]}
    values = [6, 0, 12, 9, 0, 15, 4, 0, 20]
    arrs, sfxs = _run(values, {'windows': [4]})
    assert _get(arrs, sfxs, 'up_semi_w4') == pytest.approx([0.0, 3.0, 4.242641, 4.038874, 5.460082, 3.872983, 5.830952, 10.25, 8.143249], abs=1e-6)
    assert _get(arrs, sfxs, 'down_semi_w4') == pytest.approx([0.0, 3.0, 6.0, 4.802343, 5.25, 9.0, 5.385165, 3.902456, 8.003905], abs=1e-6)
    assert _get(arrs, sfxs, 'semi_ratio_w4') == pytest.approx([0.0, 1.0, 0.707107, 0.841021, 1.040016, 0.430331, 1.082781, 2.626551, 1.017409], abs=1e-6)
    assert _get(arrs, sfxs, 'max_up_z_w4') == pytest.approx([0.0, 1.0, 1.224745, 1.183216, 1.260252, 1.069045, 1.425393, 1.669649, 1.269526], abs=1e-6)
    assert _get(arrs, sfxs, 'max_down_z_w4') == pytest.approx([0.0, 1.0, 1.224745, 1.521278, 0.980196, 1.603567, 1.247219, 0.77374, 1.207598], abs=1e-6)
    assert _get(arrs, sfxs, 'dev_asym_w4') == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-6)
    assert _get(arrs, sfxs, 'cross_count_w4') == pytest.approx([0.0, 1.0, 2.0, 1.0, 2.0, 2.0, 3.0, 2.0, 2.0], abs=1e-6)
