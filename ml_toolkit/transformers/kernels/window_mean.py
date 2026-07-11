"""Среднее значение за окно.

Signal:
    Простое среднее арифметическое за последние w месяцев.

Formula:
    mean_w = sum(v[t-w+1..t]) / w

Outputs:
    {product}__window_mean__w3   — среднее за 3 месяца
    {product}__window_mean__w6   — среднее за 6 месяцев
    {product}__window_mean__w12  — среднее за 12 месяцев

Preset (monthly.yaml):
    window_mean:
      windows: [3, 6, 12]

Interpretation:
    mean_w12 = 100 — средний ежемесячный оборот 100 за последний год.
    (mean_w3 - mean_w12) > 0 — последний квартал выше среднегодового.

Example:
    Ряд (4 мес): [10, 20, 30, 40],  w=3

    mean_w = (v[t−2] + v[t−1] + v[t]) / 3 = (20 + 30 + 40) / 3
    → window_mean__w3 = 30.0

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_mean, resolve_window_size

FEATURE = 'window_mean'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_mean = np.zeros((n_w, n_rows))

    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            out_mean[j, row_idx] = compute_window_mean(product_values, row_idx, ws)

    return (out_mean,)


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [3, 6, 12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    (mean,) = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params['windows']):
        arrays.append(mean[j])
        suffixes.append(f'w{w}')
    return arrays, suffixes
