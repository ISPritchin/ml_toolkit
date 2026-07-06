import math
import pytest

from tests.transformers.conftest import run_transformer, get_feature_output


def _run(values, params=None):
    return run_transformer("plateau", values, params)


def _get(arrays, suffixes, suffix):
    return get_feature_output(arrays, suffixes, suffix)

def test_known_flat_series_from_docstring():
    # [100,101,100,101,100,101]: |diff|=1 < 5%*~100=5 → all flat
    # flat_share=5/5=1.0, longest_flat=5
    arrs, sfxs = _run([100, 101, 100, 101, 100, 101], {"windows": [6]})
    assert _get(arrs, sfxs, "flat_share_w6")[-1] == pytest.approx(1.0, abs=1e-4)
    assert _get(arrs, sfxs, "longest_flat_w6")[-1] == pytest.approx(5.0)


def test_dynamic_series_flat_share_zero():
    # [10,30,20,40,30,50]: large swings → no flat steps
    arrs, sfxs = _run([10, 30, 20, 40, 30, 50], {"windows": [6]})
    assert _get(arrs, sfxs, "flat_share_w6")[-1] == pytest.approx(0.0)


def test_current_flat_streak_running_state():
    # [100,200,201,202]: first step (100→200) is not flat (large jump), then 2 flat steps
    # 200→201: |diff|=1 < 5%*(200+201)/2≈10 → flat; same for 201→202 → streak=2
    arrs, sfxs = _run([100, 200, 201, 202], {"windows": [6]})
    assert _get(arrs, sfxs, "current_flat_streak")[-1] == pytest.approx(2.0)


def test_constant_series_all_flat():
    arrs, sfxs = _run([50, 50, 50, 50, 50, 50], {"windows": [6]})
    assert _get(arrs, sfxs, "flat_share_w6")[-1] == pytest.approx(1.0, abs=1e-4)


def test_approach_zero_and_exit():
    # Серия: [100, 10, 1, 0, 0, 0, 50]
    # Снижение к нулю идёт через крупные относительные скачки → НЕ плато:
    #   10→100: |diff|=90, порог=5%*55=2.75       → не плато
    #   1→10:   |diff|=9,  порог=5%*5.5=0.275     → не плато
    #   0→1:    |diff|=1,  порог=5%*0.5=0.025     → не плато (снижение 100%!)
    # Нули между собой — плато (|0-0|=0 < 5%*EPS ≈ 0):
    #   0→0 дважды → два плоских шага, серия длиной 2
    # Выход из нуля в 50:
    #   50→0:   |diff|=50, порог=5%*25=1.25       → не плато → серия обрывается
    values = [100, 10, 1, 0, 0, 0, 50]
    arrs, sfxs = _run(values, {"windows": [6]})

    # В разгаре нулевого плато (pos=5): streak=2
    assert _get(arrs, sfxs, "current_flat_streak")[5] == pytest.approx(2.0)

    # Сразу после выхода в 50 (pos=6): плато сброшено
    assert _get(arrs, sfxs, "current_flat_streak")[6] == pytest.approx(0.0)

    # Выход случился ровно на текущем шаге → recency=0 (месяцев с выхода)
    assert _get(arrs, sfxs, "plateau_exit_recency")[6] == pytest.approx(0.0)

    # flat_share окна w=6 на pos=6: окно [10,1,0,0,0,50], 5 переходов, 2 плоских → 2/5=0.4
    assert _get(arrs, sfxs, "flat_share_w6")[6] == pytest.approx(2 / 5, abs=1e-6)

    # longest_flat в том же окне = 2 (два 0→0 подряд)
    assert _get(arrs, sfxs, "longest_flat_w6")[6] == pytest.approx(2.0)

    # На pos=3 (только что упало в 0): окно [100,10,1,0], все переходы не плоские → flat_share=0
    assert _get(arrs, sfxs, "flat_share_w6")[3] == pytest.approx(0.0)

    # До выхода из плато: plateau_exit_recency=-1 (плато ещё не завершалось, на pos=3)
    assert _get(arrs, sfxs, "plateau_exit_recency")[3] == pytest.approx(-1.0)

def test_with_mixed_zeros():
    # Series with alternating zeros and non-zeros (economic domain):
    # [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    # zeros at idx 2,4,5,8,10,12,13 — two consecutive-zero runs ({4,5} and {12,13})
    # last 6 values: [10, 0, 60, 0, 0, 35]  (3 zeros, 3 non-zeros)
    values = [50, 30, 0, 80, 0, 0, 20, 40, 0, 10, 0, 60, 0, 0, 35]
    arrs, sfxs = _run(values, {'windows': [6]})
    # one 0→0 step flat out of 5 in [10,0,60,0,0,35]: only indices (12,13) are both zero
    assert _get(arrs, sfxs, 'flat_share_w6')[-1] == pytest.approx(0.2, abs=1e-06)
    assert math.isfinite(_get(arrs, sfxs, 'longest_flat_w6')[-1]), 'longest_flat_w6 must be finite'
    assert _get(arrs, sfxs, 'longest_flat_w6')[-1] == pytest.approx(1.0, rel=1e-4)
    assert math.isfinite(_get(arrs, sfxs, 'near_mean_w6')[-1]), 'near_mean_w6 must be finite'
    assert _get(arrs, sfxs, 'near_mean_w6')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'current_flat_streak')[-1]), 'current_flat_streak must be finite'
    assert _get(arrs, sfxs, 'current_flat_streak')[-1] == pytest.approx(0.0, abs=1e-6)
    assert math.isfinite(_get(arrs, sfxs, 'plateau_exit_recency')[-1]), 'plateau_exit_recency must be finite'
    # выход из плато (0→35) в текущем месяце → 0 месяцев с выхода
    assert _get(arrs, sfxs, 'plateau_exit_recency')[-1] == pytest.approx(0.0, abs=1e-9)
