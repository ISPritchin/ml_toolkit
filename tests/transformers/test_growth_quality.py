import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("growth_quality", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_values_from_docstring():
    # [10,20,30,40,50,60] w=6: 5 equal pos diffs of +10, sum=50
    # best_share=10/50=0.2, organic=0.8, pos_count=5, consist=5/5=1.0, gini=0
    arrs, sfxs = _run([10, 20, 30, 40, 50, 60], {"windows": [6]})
    assert _get(arrs, sfxs, "best_share_w6")[-1] == pytest.approx(0.2, abs=1e-4)
    assert _get(arrs, sfxs, "organic_w6")[-1] == pytest.approx(0.8, abs=1e-4)
    assert _get(arrs, sfxs, "pos_count_w6")[-1] == pytest.approx(5.0)
    assert _get(arrs, sfxs, "consist_score_w6")[-1] == pytest.approx(1.0, abs=1e-4)
    assert _get(arrs, sfxs, "growth_gini_w6")[-1] == pytest.approx(0.0, abs=1e-4)


def test_single_spike_organic_near_zero():
    # [0,0,0,0,0,100]: one positive diff=+100, best_share=1.0, organic=0
    arrs, sfxs = _run([0, 0, 0, 0, 0, 100], {"windows": [6]})
    assert _get(arrs, sfxs, "organic_w6")[-1] == pytest.approx(0.0, abs=1e-3)
    assert _get(arrs, sfxs, "best_share_w6")[-1] == pytest.approx(1.0, abs=1e-3)


def test_constant_series_pos_count_zero():
    # All same → no positive diffs → pos_count=0
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "pos_count_w6")[-1] == pytest.approx(0.0)


def test_all_zeros_no_diffs():
    # max_pos=0, sum_pos=0 → best_share=0/(0+EPS)=0, organic=1-0=1.0 (degenerate: no growth at all)
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "pos_count_w6")[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, "best_share_w6")[-1] == pytest.approx(0.0, abs=1e-3)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'best_share_w6')[-1]), 'best_share_w6 must be finite'
    assert _get(arrs, sfxs, 'best_share_w6')[-1] == pytest.approx(0.6315789473617728, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'consist_score_w6')[-1]), 'consist_score_w6 must be finite'
    assert _get(arrs, sfxs, 'consist_score_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'pos_count_w6')[-1]), 'pos_count_w6 must be finite'
    assert _get(arrs, sfxs, 'pos_count_w6')[-1] == pytest.approx(2.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'growth_gini_w6')[-1]), 'growth_gini_w6 must be finite'
    assert _get(arrs, sfxs, 'growth_gini_w6')[-1] == pytest.approx(0.13157894736772854, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'organic_w6')[-1]), 'organic_w6 must be finite'
    assert _get(arrs, sfxs, 'organic_w6')[-1] == pytest.approx(0.3684210526382272, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'neg_sum_share_w6')[-1]), 'neg_sum_share_w6 must be finite'
    assert _get(arrs, sfxs, 'neg_sum_share_w6')[-1] == pytest.approx(0.6666666666603175, rel=1e-4)
