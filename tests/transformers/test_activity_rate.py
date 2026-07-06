import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("activity_rate", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_rate():
    # [0,10,0,8]: first active at pos=1, total active=2, tenure=3 → rate=2/3
    arrs, sfxs = _run([0, 10, 0, 8])
    assert _get(arrs, sfxs, "share_of_tenure_active")[-1] == pytest.approx(2 / 3, abs=1e-4)


def test_all_zeros_rate_zero():
    arrs, sfxs = _run([0, 0, 0, 0])
    assert _get(arrs, sfxs, "share_of_tenure_active")[-1] == pytest.approx(0.0)


def test_always_active_rate_one():
    # Every month active: first_active=0, tenure=n, active_count=n → rate=1
    arrs, sfxs = _run([10, 20, 30, 40])
    assert _get(arrs, sfxs, "share_of_tenure_active")[-1] == pytest.approx(1.0, abs=1e-6)


def test_rate_before_first_activation_is_zero():
    # [0,0,0,10]: first 3 rows have no activation
    arrs, sfxs = _run([0, 0, 0, 10])
    assert _get(arrs, sfxs, "share_of_tenure_active")[0] == pytest.approx(0.0)
    assert _get(arrs, sfxs, "share_of_tenure_active")[2] == pytest.approx(0.0)


def test_sporadic_client():
    # [10,0,0,0,0,10]: first_active=0, tenure=6, active_count=2 → rate=2/6=1/3
    arrs, sfxs = _run([10, 0, 0, 0, 0, 10])
    assert _get(arrs, sfxs, "share_of_tenure_active")[-1] == pytest.approx(1 / 3, abs=1e-4)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    assert math.isfinite(_get(arrs, sfxs, 'share_of_tenure_active')[-1]), 'share_of_tenure_active must be finite'
    assert _get(arrs, sfxs, 'share_of_tenure_active')[-1] == pytest.approx(0.5333333333333333, rel=1e-4)
