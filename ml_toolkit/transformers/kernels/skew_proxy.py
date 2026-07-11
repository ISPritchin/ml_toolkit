"""Прокси-мера скошенности: положение среднего между минимумом и максимумом окна.

Signal:
    Показывает, тяготеет ли распределение к нижней или верхней части диапазона.
    Значение > 0.5 — среднее «ближе к максимуму» (редкие низкие значения, большинство высокие);
    < 0.5 — «ближе к минимуму» (редкие всплески на фоне низкого основания).

Formula:
    lo_w  = min(v[t-w+1..t])
    hi_w  = max(v[t-w+1..t])
    mean_w = mean(v[t-w+1..t])
    skew_proxy_w = (mean_w - lo_w) / (hi_w - lo_w + eps)

    Значение в [0, 1] при lo <= mean <= hi.

Outputs:
    {product}__skew_proxy__w6   — прокси скошенности за 6 мес
    {product}__skew_proxy__w12  — прокси скошенности за 12 мес

Preset entry:
    skew_proxy:
      windows: [6, 12]

Interpretation:
    = 0.5 — среднее посередине между min и max (симметричное распределение).
    < 0.3 — правосторонняя асимметрия: пиковые значения редки, основной уровень низкий.
    > 0.7 — левосторонняя: редкие провалы на фоне высокого основного уровня.
    Для пульсирующего ряда B: mean = 24.2, min = 0, max = 100 → skew_proxy ≈ 0.24.

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 40],  w=6

    mean = 90/6 = 15,  lo = 10,  hi = 40
    skew_proxy = (mean − lo) / (hi − lo) = (15 − 10)/(40 − 10) = 5/30 = 0.167
    → skew_proxy__w6 = 0.167  (среднее ближе к минимуму — редкий всплеск)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import (
    compute_window_mean,
    compute_window_min_and_max,
    resolve_window_size,
    safe_ratio,
)

FEATURE = 'skew_proxy'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean = compute_window_mean(product_values, row_idx, ws)
            lo, hi = compute_window_min_and_max(product_values, row_idx, ws)
            out[j, row_idx] = safe_ratio(mean - lo, hi - lo)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
