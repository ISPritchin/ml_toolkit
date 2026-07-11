"""Ранг текущего значения внутри скользящего окна (доля точек ≤ v).

Signal:
    Показывает, где текущий месяц находится в распределении значений окна: 1.0 = на
    максимуме, 0 = на минимуме (или ниже всех). Устойчив к выбросам в отличие от zscore:
    не зависит от формы распределения, только от порядка.

Formula:
    rank_in_window_w = count(v[i] <= v[t], i in [t-w+1..t]) / ws

    При равных значениях считаются все точки <= v[t] (включая текущую).

Outputs:
    {product}__rank_in_window__w6   — перцентильный ранг в окне 6
    {product}__rank_in_window__w12  — перцентильный ранг в окне 12
    {product}__rank_in_window__w24  — перцентильный ранг в окне 24

Preset (monthly.yaml):
    rank_in_window:
      windows: [6, 12, 24]

Interpretation:
    = 1.0 — текущий месяц является максимумом окна (аналогично pct_of_max = 1).
    = 0.5 — текущий месяц медианный.
    rank_w6 = 1.0, rank_w24 = 0.5 — текущий месяц хорош в краткосроке, средний в долгосроке.
    Для линейно растущего ряда G ранг w12 последнего элемента = 12/12 = 1.0.

Example:
    Ряд (6 мес): [10, 30, 20, 50, 40, 45],  w=6,  v[t]=45

    значения ≤ 45: 10, 30, 20, 40, 45 → 5 шт. (50 исключено)
    rank_in_window = 5/6 = 0.833
    → rank_in_window__w6 = 0.833  (текущий мес. выше 5 из 6)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'rank_in_window'


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
            count = 0
            for offset in range(ws):
                if product_values[row_idx - ws + 1 + offset] <= v:
                    count += 1
            out[j, row_idx] = count / ws
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
