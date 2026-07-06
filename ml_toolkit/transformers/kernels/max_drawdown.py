"""Наибольшая относительная просадка от локального пика на скользящем окне.

Signal:
    Измеряет наихудшее падение «от пика к текущему минимуму» внутри окна. Высокое
    значение означает, что клиент пережил глубокую просадку: потенциально нестабильный
    или восстанавливающийся после шока. Ноль — просадки не было.

Formula:
    running_peak = max(v[i..j]) для i <= j in [t-w+1..t]
    dd[j] = (running_peak - v[j]) / (|running_peak| + eps)
    max_drawdown_w = max(dd[j])

Outputs:
    {product}__max_drawdown__w6   — макс. просадка в окне 6 мес
    {product}__max_drawdown__w12  — макс. просадка в окне 12 мес
    {product}__max_drawdown__w24  — макс. просадка в окне 24 мес

Preset (monthly.yaml):
    max_drawdown:
      windows: [6, 12, 24]

Interpretation:
    = 0 — значения только росли (или оставались стабильными) внутри окна.
    = 0.5 — в какой-то момент клиент потерял 50% от своего локального максимума.
    = 0.9375 — просадка ряда D: с 80 до 5 (классический V-образный пример).
    max_drawdown_w12 > 0.7 при recovery_completeness = 1 — клиент упал и полностью восстановился.

Example:
    Ряд (6 мес): [10, 80, 40, 20, 5, 30],  w=6

    running_peak растёт до 80, затем падение до 5
    наихудшая просадка в точке v=5: (80 − 5) / 80 = 0.9375
    → max_drawdown__w6 = 0.9375  (потеря ~94% от пика)
"""

import numba as nb
import numpy as np

from .._windowing import resolve_window_size, safe_ratio

FEATURE = "max_drawdown"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            running_peak = product_values[row_idx - ws + 1]
            largest_dd = 0.0
            for offset in range(ws):
                v = product_values[row_idx - ws + 1 + offset]
                if v > running_peak:
                    running_peak = v
                dd = safe_ratio(running_peak - v, running_peak)
                if dd > largest_dd:
                    largest_dd = dd
            out[j, row_idx] = largest_dd
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
