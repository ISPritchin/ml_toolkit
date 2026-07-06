"""Относительный разрыв между средним и медианой окна: прокси скошенности.

Signal:
    Если среднее выше медианы — распределение правосторонне скошено (редкие крупные
    месяцы). Если ниже — левосторонняя асимметрия (редкие глубокие провалы). Используется
    как быстрый прокси для формы распределения без полного вычисления эксцесса.

Formula:
    mean_w   = mean(v[t-w+1..t])
    median_w = медиана отсортированного буфера окна
    mean_median_gap_w = (mean_w - median_w) / (|mean_w| + eps)

Outputs:
    {product}__mean_median_gap__w6   — (mean-median)/|mean| за 6 мес
    {product}__mean_median_gap__w12  — (mean-median)/|mean| за 12 мес

Preset (monthly.yaml):
    mean_median_gap:
      windows: [6, 12]

Interpretation:
    > 0 — правосторонняя асимметрия: редкие крупные всплески тянут среднее вверх.
    < 0 — левосторонняя: редкие провалы (или нули) занижают среднее.
    ≈ 0 — симметричное распределение, среднее = медиане.
    Сильная правая асимметрия (> 0.3) типична для B2B-проектных клиентов с нулями.

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 40],  w=6

    mean = 90/6 = 15
    median (чётное окно) = 0.5·(10 + 10) = 10
    mean_median_gap = (15 − 10) / 15 = 0.333
    → mean_median_gap__w6 = 0.333  (правосторонняя асимметрия от всплеска 40)
"""

import numba as nb
import numpy as np

from .._windowing import (
    compute_window_mean,
    fill_window_sorted,
    resolve_window_size,
    safe_ratio,
    sorted_median,
)

FEATURE = "mean_median_gap"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    max_w = 1
    for j in range(n_w):
        if windows[j] > max_w:
            max_w = windows[j]
    sorted_buf = np.empty(max_w)
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean = compute_window_mean(product_values, row_idx, ws)
            fill_window_sorted(sorted_buf, product_values, row_idx, ws)
            median = sorted_median(sorted_buf, ws)
            out[j, row_idx] = safe_ratio(mean - median, mean)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
