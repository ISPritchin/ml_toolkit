"""Тесты новых модулей transformers/ — группа trend."""

import numpy as np
import pytest

from ml_toolkit.transformers import TRANSFORMERS
from ml_toolkit.transformers._windowing import compute_position_within_entity

# ── Helpers ───────────────────────────────────────────────────────────────────

def _pos(n):
    """position_within_entity для одной entity длиной n."""
    return compute_position_within_entity(np.zeros(n, dtype=np.int64))


def _run(name, values, params):
    mod = TRANSFORMERS[name]
    pos = _pos(len(values))
    return mod.compute(np.array(values, dtype=np.float64), pos, params)


# ── slope ─────────────────────────────────────────────────────────────────────

def test_slope_positive_for_linear_growth():
    arrays, suffixes = _run('slope', range(12), {'windows': [6, 12]})
    assert suffixes == ['w6', 'w12']
    assert arrays[0][-1] > 0.9   # наклон ≈ 1
    assert arrays[1][-1] > 0.9


def test_slope_negative_for_decline():
    arrays, _ = _run('slope', list(range(12, 0, -1)), {'windows': [6]})
    assert arrays[0][-1] < -0.9


def test_slope_zero_for_constant():
    arrays, _ = _run('slope', [5.0] * 12, {'windows': [6, 12]})
    assert abs(arrays[0][-1]) < 1e-6
    assert abs(arrays[1][-1]) < 1e-6


def test_slope_short_history_uses_available_rows():
    # Entity из 3 строк: window=12 должно сузиться до 3
    arrays, _ = _run('slope', [1.0, 2.0, 3.0], {'windows': [12]})
    assert arrays[0][-1] > 0.9


def test_slope_multiple_windows_independent():
    values = list(range(24))
    arrays, _suffixes = _run('slope', values, {'windows': [6, 12, 24]})
    assert len(arrays) == 3
    assert all(a[-1] > 0.9 for a in arrays)


# ── slope_ratio ───────────────────────────────────────────────────────────────

def test_slope_ratio_accelerating_trend():
    # Ускоряющийся рост: последние 6 месяцев круче, чем в целом за 12
    values = [0]*6 + [1, 3, 6, 10, 15, 21]  # нарастающие скачки
    arrays, suffixes = _run('slope_ratio', values, {'pairs': [[6, 12]]})
    assert suffixes == ['w6_w12']
    assert arrays[0][-1] > 1.0   # краткосрочный наклон > долгосрочного


def test_slope_ratio_constant_is_near_one():
    # Равномерный рост → оба наклона одинаковы → ratio ≈ 1
    values = list(range(12))
    arrays, _ = _run('slope_ratio', values, {'pairs': [[6, 12]]})
    assert abs(arrays[0][-1] - 1.0) < 0.2


# ── momentum ──────────────────────────────────────────────────────────────────

def test_momentum_positive_when_accelerating():
    # Первые 3 месяца низкие, последние 3 — высокие
    values = [1.0, 1.0, 1.0, 10.0, 10.0, 10.0]
    arrays, suffixes = _run('momentum', values, {'half_windows': [3]})
    assert suffixes == ['h3']
    assert arrays[0][-1] > 0   # recent > prior


def test_momentum_negative_when_declining():
    values = [10.0, 10.0, 10.0, 1.0, 1.0, 1.0]
    arrays, _ = _run('momentum', values, {'half_windows': [3]})
    assert arrays[0][-1] < 0


def test_momentum_zero_for_short_history():
    # При pos < 2*half_window−1 значение должно остаться 0
    arrays, _ = _run('momentum', [1.0, 2.0, 3.0], {'half_windows': [6]})
    assert all(v == 0.0 for v in arrays[0])


# ── direction_flag ────────────────────────────────────────────────────────────

def test_direction_flag_plus_one_for_growth():
    arrays, suffixes = _run('direction_flag', range(12), {'windows': [6]})
    assert suffixes == ['w6']
    assert arrays[0][-1] == 1.0


def test_direction_flag_minus_one_for_decline():
    arrays, _ = _run('direction_flag', list(range(12, 0, -1)), {'windows': [6]})
    assert arrays[0][-1] == -1.0


def test_direction_flag_zero_for_constant():
    arrays, _ = _run('direction_flag', [5.0] * 12, {'windows': [6]})
    assert arrays[0][-1] == 0.0


# ── max_abs_jump ──────────────────────────────────────────────────────────────

def test_max_abs_jump_detects_spike():
    values = [1.0, 1.0, 1.0, 100.0, 1.0, 1.0]
    arrays, _ = _run('max_abs_jump', values, {'windows': [6]})
    assert arrays[0][-1] == pytest.approx(99.0)


def test_max_abs_jump_zero_for_constant():
    arrays, _ = _run('max_abs_jump', [5.0] * 12, {'windows': [6]})
    assert arrays[0][-1] == 0.0


# ── streak ────────────────────────────────────────────────────────────────────

def test_streak_up_counts_consecutive_increases():
    arrays, suffixes = _run('streak', [1, 2, 3, 4, 5, 4], {})
    assert suffixes == ['up', 'down']
    up, down = arrays
    assert up[-2] == 4.0    # четыре подряд роста перед последним шагом
    assert down[-1] == 1.0  # последний шаг — падение


def test_streak_resets_on_entity_boundary():
    # Две entity: [1,2,3] и [10,5]
    values = np.array([1.0, 2.0, 3.0, 10.0, 5.0])
    entity = np.array([0, 0, 0, 1, 1], dtype=np.int64)
    pos = compute_position_within_entity(entity)
    up, down = streak._kernel(values, pos)
    # Streak не должна переходить через границу entity
    assert up[3] == 0.0  # первый период второй entity
    assert down[4] == 1.0


def test_streak_flat_resets_both():
    arrays, _ = _run('streak', [5.0, 5.0, 5.0, 5.0], {})
    assert all(v == 0.0 for v in arrays[0])   # streak_up
    assert all(v == 0.0 for v in arrays[1])   # streak_down


# ── growth_since_start ────────────────────────────────────────────────────────

def test_growth_since_start_zero_at_first_nonzero():
    arrays, suffixes = _run('growth_since_start', [0.0, 0.0, 5.0, 10.0], {})
    assert suffixes == ['']
    out = arrays[0]
    assert out[2] == pytest.approx(0.0)   # первое ненулевое → рост = 0
    assert out[3] == pytest.approx(1.0)   # 10 = 2×5 → рост 100%


def test_growth_since_start_resets_per_entity():
    values = np.array([5.0, 10.0, 3.0, 9.0])
    entity = np.array([0, 0, 1, 1], dtype=np.int64)
    pos = compute_position_within_entity(entity)
    out = growth_since_start._kernel(values, pos)
    assert out[0] == pytest.approx(0.0)  # первое значение entity 0
    assert out[2] == pytest.approx(0.0)  # первое значение entity 1


def test_growth_since_start_ignores_leading_zeros():
    arrays, _ = _run('growth_since_start', [0.0, 0.0, 0.0, 4.0, 8.0], {})
    out = arrays[0]
    assert out[0] == out[1] == out[2] == 0.0
    assert out[3] == pytest.approx(0.0)
    assert out[4] == pytest.approx(1.0)


# ── preset round-trip ─────────────────────────────────────────────────────────

def test_preset_minimum_loads_and_runs():
    """Пресет minimum.yaml должен грузиться и запускаться без ошибок."""
    from pathlib import Path

    import yaml

    from ml_toolkit.transformers import TRANSFORMERS

    preset_path = Path(__file__).parent.parent / 'ml_toolkit' / 'transformers' / 'presets' / 'minimum.yaml'
    preset = yaml.safe_load(preset_path.read_text())

    values = np.array(list(range(24)), dtype=np.float64)
    pos = _pos(24)

    results = {}
    for name, params in preset.items():
        mod = TRANSFORMERS[name]
        arrays, suffixes = mod.compute(values, pos, params or {})
        for arr, suffix in zip(arrays, suffixes, strict=False):
            col = f'{name}__{suffix}' if suffix else name
            results[col] = arr

    # minimum.yaml на сегодня содержит только slope - остальные ассерты были
    # актуальны для полного monthly.yaml, который больше не поставляется
    # (нет автоматического "полного" пресета по умолчанию, см. CLAUDE.md).
    assert 'slope__w6' in results
    assert 'slope__w12' in results
    assert 'slope__w24' in results


# Импорт для теста streak boundary (нужен прямой доступ к _kernel)
from ml_toolkit.transformers import growth_since_start, streak
