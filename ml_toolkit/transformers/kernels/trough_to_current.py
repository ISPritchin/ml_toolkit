"""Кратность текущего значения к минимуму окна: восстановление от локального дна.

Signal:
    Показывает, насколько текущий месяц «выше дна» за последние w месяцев. Значение > 1
    означает, что клиент работает выше исторического минимума окна. Резкий рост trough_to_current
    при одновременно положительном slope — подтверждение восстановления, а не случайный выброс.

Formula:
    lo_w     = min(v[t-w+1..t])
    trough_to_current_w = safe_ratio(v[t], lo_w) = v[t] / |lo_w|

    lo_w вычисляется включительно с текущим месяцем (входит в окно).
    При lo_w ~ 0 (в окне был нулевой месяц) кратность не определена -> 0;
    раньше здесь возникал взрыв v/eps ~ 1e11.

Outputs:
    {product}__trough_to_current__w6   — v[t] / min_6 (кратность к дну за 6 мес)
    {product}__trough_to_current__w12  — v[t] / min_12 (кратность к дну за 12 мес)

Preset (monthly.yaml):
    trough_to_current:
      windows: [6, 12]

Interpretation:
    = 1.0 — текущий месяц сам является минимумом окна (новое дно).
    > 2.0 — клиент удвоил оборот по сравнению с минимумом окна (сильное восстановление).
    >> 1 при distance_to_global_max близком к 0 — и к дну высоко, и к максимуму близко.
    trough_to_current_w6 >> w12 — быстрое восстановление именно в последние полгода.

Example:
    Ряд (6 мес): [10, 80, 40, 20, 5, 30],  w=6,  v[t]=30

    lo_w = min(окна) = 5
    trough_to_current = v[t] / lo_w = 30 / 5
    → trough_to_current__w6 = 6.0  (текущий мес. в 6× выше дна окна)
"""

import numba as nb
import numpy as np

from .._windowing import compute_window_min_and_max, resolve_window_size, safe_ratio

FEATURE = "trough_to_current"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        v = product_values[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            lo, _ = compute_window_min_and_max(product_values, row_idx, ws)
            out[j, row_idx] = safe_ratio(v, lo)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
