"""Знак OLS-наклона на окне: +1 растёт, -1 падает, 0 стабильно.

Signal:
    Дискретная характеристика направления тренда. Используется как категориальный
    признак или для фильтрации: клиенты с direction_flag_w12 = -1 находятся
    в снижающемся тренде независимо от абсолютного уровня.

Formula:
    slope_w = OLS_slope(v[t-w+1..t], positions=[0..w-1])
    direction_flag_w = +1 if slope_w > eps
                       -1 if slope_w < -eps
                        0 otherwise

Outputs:
    {product}__direction_flag__w6   — знак тренда за 6 мес
    {product}__direction_flag__w12  — знак тренда за 12 мес

Preset (monthly.yaml):
    direction_flag:
      windows: [6, 12]

Interpretation:
    w6 = +1, w12 = -1 — краткосрочный разворот вверх на фоне долгосрочного снижения.
    w6 = w12 = +1 — устойчивый рост на обоих горизонтах.
    w6 = 0 — текущий период стабилен (горизонтальный тренд).
    Расхождение w6 и w12 сигнализирует о возможном переломе тренда.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60]
    (t=5, w=6)

    OLS-наклон по окну = +10 (ровный рост +10 ед/мес)
    slope > eps → знак +1
    → direction_flag__w6 = +1.0  (восходящий тренд)

"""

import numba as nb
import numpy as np

from .._windowing import EPS, fit_linear_trend_slope, resolve_window_size

FEATURE = 'direction_flag'


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    windows: np.ndarray,
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            s = fit_linear_trend_slope(product_values, row_idx, ws)
            if s > EPS:
                out[j, row_idx] = 1.0
            elif s < -EPS:
                out[j, row_idx] = -1.0
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """Args:
    params: {"windows": [6, 12]}

    """
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    arrays = [out[j] for j in range(len(windows))]
    suffixes = [f'w{w}' for w in params['windows']]
    return arrays, suffixes
