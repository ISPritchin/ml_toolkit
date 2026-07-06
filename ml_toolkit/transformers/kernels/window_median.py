"""Медиана значений за окно.

Signal:
    Медиана (50-й персентиль) за последние w месяцев. Устойчива к выбросам.

Formula:
    median_w = percentile_50(v[t-w+1..t])
    Честная медиана: при чётном w — среднее двух центральных элементов.

Outputs:
    {product}__window_median__w3   — медиана за 3 месяца
    {product}__window_median__w6   — медиана за 6 месяцев
    {product}__window_median__w12  — медиана за 12 месяцев

Preset (monthly.yaml):
    window_median:
      windows: [3, 6, 12]

Interpretation:
    median_w12 = 80 — половина месяцев за год были ≤80, половина ≥80.
    (median_w12 - mean_w12) > 0 — тяжелый хвост в положительную сторону (часто высокие значения).
    (median_w12 - mean_w12) < 0 — редкие пики, больше низких значений.

Example:
    Ряд (4 мес): [10, 40, 20, 30],  w=3

    окно (посл. 3) = [40, 20, 30] → сортировка [20, 30, 40]
    median = sorted_buf[3//2] = sorted_buf[1] = 30
    → window_median__w3 = 30.0
"""

import numba as nb
import numpy as np

from .._windowing import fill_window_sorted, resolve_window_size, sorted_median

FEATURE = "window_median"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_median = np.zeros((n_w, n_rows))
    max_w = 1
    for j in range(n_w):
        if windows[j] > max_w:
            max_w = windows[j]
    sorted_buf = np.empty(max_w)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            fill_window_sorted(sorted_buf, product_values, row_idx, ws)
            out_median[j, row_idx] = sorted_median(sorted_buf, ws)

    return (out_median,)


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [3, 6, 12]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    (median,) = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params["windows"]):
        arrays.append(median[j])
        suffixes.append(f"w{w}")
    return arrays, suffixes
