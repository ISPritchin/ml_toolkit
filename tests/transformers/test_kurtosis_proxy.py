import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("kurtosis_proxy", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_kurtosis():
    # [10,10,10,10,10,70] w=6: mean=20, std=sqrt(500)=10*sqrt(5)
    # z(10)=-1/sqrt(5), z^4=1/25=0.04; z(70)=sqrt(5), z^4=25
    # sum(z^4) = 5*0.04+25 = 25.2; kurt = 25.2/6-3 = 1.2
    arrs, sfxs = _run([10, 10, 10, 10, 10, 70], {"windows": [6]})
    assert _get(arrs, sfxs, "kurt_w6")[-1] == pytest.approx(1.2, abs=1e-3)


def test_constant_series_kurtosis_zero():
    # std=0 → kurtosis not computed → 0
    arrs, sfxs = _run([30, 30, 30, 30, 30, 30], {"windows": [6]})
    assert _get(arrs, sfxs, "kurt_w6")[-1] == pytest.approx(0.0, abs=1e-6)


def test_all_zeros_kurtosis_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0, 0], {"windows": [6]})
    assert _get(arrs, sfxs, "kurt_w6")[-1] == pytest.approx(0.0, abs=1e-6)


def test_top1_share_known():
    # [10,10,10,10,10,50] w=6: total=100, top1=50 → share=0.5
    arrs, sfxs = _run([10, 10, 10, 10, 10, 50], {"windows": [6]})
    # p75 index = int(6*0.75)=4, sorted=[10,10,10,10,10,50], p75=sorted[4]=10
    # p25 index = int(6*0.25)=1, p25=sorted[1]=10
    # p75/p25 ≈ 10/10 = 1.0
    assert _get(arrs, sfxs, "p75_p25_w6")[-1] == pytest.approx(1.0, abs=1e-2)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'kurt_w6')[-1]), 'kurt_w6 must be finite'
    assert _get(arrs, sfxs, 'kurt_w6')[-1] == pytest.approx(-0.7083872871215724, rel=1e-4)
    # p25 = sorted[int(0.25*5)] = 0 → отношение не определено → 0 (раньше 3.5e10)
    assert math.isfinite(_get(arrs, sfxs, 'p75_p25_w6')[-1]), 'p75_p25_w6 must be finite'
    assert _get(arrs, sfxs, 'p75_p25_w6')[-1] == pytest.approx(0.0, abs=1e-9)
    # p10 = sorted[0] = 0 → 0 (раньше 6e10)
    assert math.isfinite(_get(arrs, sfxs, 'p90_p10_w6')[-1]), 'p90_p10_w6 must be finite'
    assert _get(arrs, sfxs, 'p90_p10_w6')[-1] == pytest.approx(0.0, abs=1e-9)
    # p75 = sorted[int(0.75*5)] = sorted[3] = 10 → выше: {35, 60} → 95/105
    assert math.isfinite(_get(arrs, sfxs, 'upper_tail_w6')[-1]), 'upper_tail_w6 must be finite'
    assert _get(arrs, sfxs, 'upper_tail_w6')[-1] == pytest.approx(95 / 105, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'lower_tail_w6')[-1]), 'lower_tail_w6 must be finite'
    assert _get(arrs, sfxs, 'lower_tail_w6')[-1] == pytest.approx(0.0, abs=1e-6)
