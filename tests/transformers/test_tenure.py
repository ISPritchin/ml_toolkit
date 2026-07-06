import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("tenure", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_value_from_docstring():
    # [0,0,10,20,30]: first_active at pos=2, at pos=4 → tenure=4-2+1=3
    arrs, sfxs = _run([0, 0, 10, 20, 30])
    assert _get(arrs, sfxs, "tenure_months")[-1] == pytest.approx(3.0)


def test_first_active_flag_only_once():
    # [0,0,10,20,30]: flag=1 only at index 2, not elsewhere
    arrs, sfxs = _run([0, 0, 10, 20, 30])
    flags = _get(arrs, sfxs, "first_active_flag")
    assert flags[0] == pytest.approx(0.0)
    assert flags[1] == pytest.approx(0.0)
    assert flags[2] == pytest.approx(1.0)
    assert flags[3] == pytest.approx(0.0)
    assert flags[4] == pytest.approx(0.0)


def test_all_zeros_tenure_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0])
    assert _get(arrs, sfxs, "tenure_months")[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, "first_active_flag")[-1] == pytest.approx(0.0)


def test_active_from_start_tenure_counts_from_one():
    # [10,20,30]: first at pos=0 → tenure=pos-0+1=3 at pos=2
    arrs, sfxs = _run([10, 20, 30])
    assert _get(arrs, sfxs, "tenure_months")[-1] == pytest.approx(3.0)
    assert _get(arrs, sfxs, "first_active_flag")[0] == pytest.approx(1.0)


def test_activation_after_zeros_flag_and_tenure():
    # [0,0,0,50]: first at pos=3 → flag=1 there, tenure=1
    arrs, sfxs = _run([0, 0, 0, 50])
    assert _get(arrs, sfxs, "first_active_flag")[-1] == pytest.approx(1.0)
    assert _get(arrs, sfxs, "tenure_months")[-1] == pytest.approx(1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    # first nonzero at pos=0, last pos=14 → 15 months
    assert _get(arrs, sfxs, 'tenure_months')[-1] == pytest.approx(15.0, abs=1e-06)
    assert math.isfinite(_get(arrs, sfxs, 'first_active_flag')[-1]), 'first_active_flag must be finite'
    assert _get(arrs, sfxs, 'first_active_flag')[-1] == pytest.approx(0.0, abs=1e-6)
