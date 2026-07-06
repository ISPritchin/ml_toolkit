"""Сумма значений на скользящем окне.

Signal:
    Суммарный объём за период: ключевой показатель «веса» клиента в окне.
    Используется как знаменатель в nормировании других признаков (entropy, gini, recent_share)
    и как самостоятельный признак для сравнения периодов.

Formula:
    rolling_sum_w = sum(v[t-w+1..t])
    Эффективное окно: ws = min(pos+1, w).

Outputs:
    {product}__rolling_sum__w3   — сумма за 3 мес (квартал)
    {product}__rolling_sum__w6   — сумма за 6 мес
    {product}__rolling_sum__w12  — сумма за 12 мес

Preset (monthly.yaml):
    rolling_sum:
      windows: [3, 6, 12]

Interpretation:
    sum_w3 / sum_w12 ≈ recent_share__r3_w12 — доля последнего квартала в годовом объёме.
    sum_w12 = 450 для равномерного ряда G [10..65] — базовый годовой объём.
    sum_w3 растёт при sum_w12 стабильном — ускорение в конце года.
    Имеет размерность исходной колонки; не нормирован.

Example:
    Ряд (4 мес): [10, 20, 30, 40],  w=3

    rolling_sum = v[t−2] + v[t−1] + v[t] = 20 + 30 + 40
    → rolling_sum__w3 = 90  (объём последнего квартала)
"""

import numba as nb
import numpy as np

from .._windowing import compute_window_sum, resolve_window_size

FEATURE = "rolling_sum"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            out[j, row_idx] = compute_window_sum(product_values, row_idx, ws)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [3]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
