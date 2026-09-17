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

Preset entry:
    window_median:
      windows: [3, 6, 12]
      dilations: [1, 2]   # optional, default [1]; d>1 spaces window taps d rows apart

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

from ml_toolkit.transformers._windowing import (
    FILL_NAN_UNBOUNDED_LOW,
    fill_window_sorted,
    resolve_window_size,
    sorted_median,
)

FEATURE = 'window_median'
# медиана сырых значений колонки — при знакопеременных данных произвольного знака и
# масштаба; "median >= 0" верно только при неотрицательных значениях.
FILL_NAN: dict[str | None, float] = {None: FILL_NAN_UNBOUNDED_LOW}


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray, dilations: np.ndarray
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    n_d = dilations.shape[0]
    out_median = np.zeros((n_w * n_d, n_rows))
    max_w = 1
    for j in range(n_w):
        max_w = max(max_w, windows[j])
    sorted_buf = np.empty(max_w)

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_w):
            for d in range(n_d):
                ws = resolve_window_size(pos, windows[j], dilations[d])
                fill_window_sorted(sorted_buf, product_values, row_idx, ws, dilations[d])
                out_median[k, row_idx] = sorted_median(sorted_buf, ws)
                k += 1

    return (out_median,)


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [3, 6, 12], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    windows = np.array(params['windows'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    (median,) = _kernel(values, position, windows, dilations)
    arrays = []
    suffixes = []
    k = 0
    for w in params['windows']:
        for d in params.get('dilations', [1]):
            arrays.append(median[k])
            suffixes.append(f'w{w}' if d == 1 else f'w{w}_d{d}')
            k += 1
    return arrays, suffixes
