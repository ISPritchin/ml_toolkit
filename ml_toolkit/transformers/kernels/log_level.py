"""log1p(|mean_w|): средний уровень в лог-шкале, устойчив к выбросам.

Signal:
    Логарифм среднего значения в окне — масштабированный индикатор «размера бизнеса»
    клиента за период. В отличие от log1p_level не чувствителен к разовым всплескам,
    поскольку усредняет весь период.

Formula:
    mean_w = mean(v[t-w+1..t])
    log_level_w = log1p(|mean_w|) = ln(1 + |mean_w|)

Outputs:
    {product}__log_level__w6   — log1p среднего за 6 мес
    {product}__log_level__w12  — log1p среднего за 12 мес

Preset (monthly.yaml):
    log_level:
      windows: [6, 12]

Interpretation:
    log_level_w12 = 6 ≈ mean ≈ 400 (среднемесячный доход 400 единиц).
    log_level_w12 = 10 ≈ mean ≈ 22000 (крупный клиент).
    log_level_w6 > log_level_w12 — недавний уровень выше годового среднего.
    Стабильное значение без изменений между w6 и w12 = плато или равномерный доход.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  w=6

    mean_w = 210/6 = 35
    log_level = ln(1 + 35) = ln(36) = 3.584
    → log_level__w6 = 3.584

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import compute_window_mean, resolve_window_size

FEATURE = 'log_level'


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
            out[j, row_idx] = np.log1p(abs(mean))
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
