import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("value_clustering", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_top1_share():
    # [10,10,10,10,10,50] w=6: total=100, top1=50 → share=0.5
    arrs, sfxs = _run([10, 10, 10, 10, 10, 50], {"windows": [6]})
    assert _get(arrs, sfxs, "top1_share_w6")[-1] == pytest.approx(0.5, abs=1e-4)


def test_known_herfindahl():
    # [10,10,10,10,10,50] w=6: total=100
    # herf = 5*(10/100)² + (50/100)² = 5*0.01 + 0.25 = 0.30
    arrs, sfxs = _run([10, 10, 10, 10, 10, 50], {"windows": [6]})
    assert _get(arrs, sfxs, "herfindahl_w6")[-1] == pytest.approx(0.3, abs=1e-4)


def test_uniform_distribution_minimum_herfindahl():
    # [10,10,10,10,10,10] w=6: each share=1/6 → herf=6*(1/6)²=1/6≈0.1667
    arrs, sfxs = _run([10, 10, 10, 10, 10, 10], {"windows": [6]})
    assert _get(arrs, sfxs, "herfindahl_w6")[-1] == pytest.approx(1 / 6, abs=1e-4)


def test_all_zeros_no_clustering():
    # total=0 → skip computation → all outputs=0
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "top1_share_w6")[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, "herfindahl_w6")[-1] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'top1_share_w6')[-1]), 'top1_share_w6 must be finite'
    assert _get(arrs, sfxs, 'top1_share_w6')[-1] == pytest.approx(0.5714285714231292, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'top3_share_w6')[-1]), 'top3_share_w6 must be finite'
    assert _get(arrs, sfxs, 'top3_share_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'bot3_share_w6')[-1]), 'bot3_share_w6 must be finite'
    assert _get(arrs, sfxs, 'bot3_share_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'concentration_w6')[-1]), 'concentration_w6 must be finite'
    # bot3_sum = 0 → полюсное отношение не определено → 0 (раньше 1.05e11)
    assert _get(arrs, sfxs, 'concentration_w6')[-1] == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(_get(arrs, sfxs, 'density_w6')[-1]), 'density_w6 must be finite'
    assert _get(arrs, sfxs, 'density_w6')[-1] == pytest.approx(0.5833333333300926, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'herfindahl_w6')[-1]), 'herfindahl_w6 must be finite'
    assert _get(arrs, sfxs, 'herfindahl_w6')[-1] == pytest.approx(0.44671201814058953, rel=1e-4)
