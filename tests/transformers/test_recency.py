import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("recency", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_known_value_from_docstring():
    # [10,0,0,30,0] pos=4: last active at pos=3 → gap=4-3=1
    arrs, sfxs = _run([10, 0, 0, 30, 0])
    assert _get(arrs, sfxs, "recency_gap")[-1] == pytest.approx(1.0)


def test_before_any_activity_minus_one():
    # [0,0,0,10]: before pos=3, gap=-1
    arrs, sfxs = _run([0, 0, 0, 10])
    assert _get(arrs, sfxs, "recency_gap")[0] == pytest.approx(-1.0)
    assert _get(arrs, sfxs, "recency_gap")[1] == pytest.approx(-1.0)
    assert _get(arrs, sfxs, "recency_gap")[2] == pytest.approx(-1.0)


def test_currently_active_gap_zero():
    # [0,0,10]: at pos=2 v=10 → last_active обновляется ДО вычисления gap → gap=0
    arrs, sfxs = _run([0, 0, 10])
    assert _get(arrs, sfxs, "recency_gap")[-1] == pytest.approx(0.0)


def test_gap_grows_with_inactivity():
    # [10,0,0,0,0]: row0 активен → gap=0; далее gap растёт: 1, 2, 3, 4
    arrs, sfxs = _run([10, 0, 0, 0, 0])
    result = _get(arrs, sfxs, "recency_gap")
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(1.0)
    assert result[4] == pytest.approx(4.0)


def test_all_zeros_always_minus_one():
    arrs, sfxs = _run([0, 0, 0, 0, 0])
    for v in _get(arrs, sfxs, "recency_gap"):
        assert v == pytest.approx(-1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    assert math.isfinite(_get(arrs, sfxs, 'recency_gap')[-1]), 'recency_gap must be finite'
    # последний месяц активен (35) → gap = 0; предпоследний (0) → gap = 2 (последняя активность = 60)
    assert _get(arrs, sfxs, 'recency_gap')[-1] == pytest.approx(0.0, abs=1e-9)
    assert _get(arrs, sfxs, 'recency_gap')[-2] == pytest.approx(2.0, rel=1e-4)
