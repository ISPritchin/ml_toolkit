import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("streak", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)


def test_ascending_series_streak_up():
    # [10,20,30,40,50]: 4 consecutive rises → streak_up=4, streak_down=0
    arrs, sfxs = _run([10, 20, 30, 40, 50])
    assert _get(arrs, sfxs, "up")[-1] == pytest.approx(4.0)
    assert _get(arrs, sfxs, "down")[-1] == pytest.approx(0.0)


def test_descending_series_streak_down():
    # [50,40,30,20,10]: 4 consecutive falls → streak_down=4, streak_up=0
    arrs, sfxs = _run([50, 40, 30, 20, 10])
    assert _get(arrs, sfxs, "down")[-1] == pytest.approx(4.0)
    assert _get(arrs, sfxs, "up")[-1] == pytest.approx(0.0)


def test_flat_step_resets_both_streaks():
    # [10,20,20]: 20==20 → both streaks=0 at end
    arrs, sfxs = _run([10, 20, 20])
    assert _get(arrs, sfxs, "up")[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, "down")[-1] == pytest.approx(0.0)


def test_alternating_streaks_reset():
    # [10,20,10,20,10,20]: alternates → streak_up=1, streak_down=0 at last step
    arrs, sfxs = _run([10, 20, 10, 20, 10, 20])
    assert _get(arrs, sfxs, "up")[-1] == pytest.approx(1.0)
    assert _get(arrs, sfxs, "down")[-1] == pytest.approx(0.0)


def test_all_zeros_both_streaks_zero():
    arrs, sfxs = _run([0, 0, 0, 0, 0])
    assert _get(arrs, sfxs, "up")[-1] == pytest.approx(0.0)
    assert _get(arrs, sfxs, "down")[-1] == pytest.approx(0.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values)
    # v[14]=35 > v[13]=0 → up streak increments; v[13]=0=v[12]=0 so streak was 0 → up=1
    assert _get(arrs, sfxs, 'up')[-1] == pytest.approx(1.0, abs=1e-06)
    assert math.isfinite(_get(arrs, sfxs, 'down')[-1]), 'down must be finite'
    assert _get(arrs, sfxs, 'down')[-1] == pytest.approx(0.0, abs=1e-6)
