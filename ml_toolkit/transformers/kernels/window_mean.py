"""Среднее значение за окно.

Signal:
    Простое среднее арифметическое за последние w месяцев.

Formula:
    mean_w = sum(v[t-w+1..t]) / w

Outputs:
    {product}__window_mean__w3   — среднее за 3 месяца
    {product}__window_mean__w6   — среднее за 6 месяцев
    {product}__window_mean__w12  — среднее за 12 месяцев

Preset entry:
    window_mean:
      windows: [3, 6, 12]
      dilations: [1, 2]   # optional, default [1]; d>1 spaces window taps d rows apart

Interpretation:
    mean_w12 = 100 — среднее ежемесячное значение 100 за последний год.
    (mean_w3 - mean_w12) > 0 — последний квартал выше среднегодового.

Example:
    Ряд (4 мес): [10, 20, 30, 40],  w=3

    mean_w = (v[t−2] + v[t−1] + v[t]) / 3 = (20 + 30 + 40) / 3
    → window_mean__w3 = 30.0

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import FILL_NAN_UNBOUNDED_LOW, compute_window_mean, resolve_window_size

FEATURE = 'window_mean'
# среднее сырых значений колонки — при знакопеременных данных произвольного знака и
# масштаба; "mean >= 0" верно только при неотрицательных значениях.
FILL_NAN: dict[str | None, float] = {None: FILL_NAN_UNBOUNDED_LOW}


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray, dilations: np.ndarray
):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    n_d = dilations.shape[0]
    out_mean = np.zeros((n_w * n_d, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_w):
            for d in range(n_d):
                ws = resolve_window_size(pos, windows[j], dilations[d])
                out_mean[k, row_idx] = compute_window_mean(product_values, row_idx, ws, dilations[d])
                k += 1

    return (out_mean,)


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [3, 6, 12], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    windows = np.array(params['windows'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    (mean,) = _kernel(values, position, windows, dilations)
    arrays = []
    suffixes = []
    k = 0
    for w in params['windows']:
        for d in params.get('dilations', [1]):
            arrays.append(mean[k])
            suffixes.append(f'w{w}' if d == 1 else f'w{w}_d{d}')
            k += 1
    return arrays, suffixes
