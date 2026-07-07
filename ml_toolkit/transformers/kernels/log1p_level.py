"""Знаковый log1p текущего значения: сжатая шкала для экспоненциальных объёмов.

Signal:
    Логарифмически сжимает масштаб оборота: позволяет сравнивать клиентов с разными
    размерами бизнеса и работать с признаками в log-шкале без потери знака.
    Устойчив к экспоненциальному распределению оборотов.

Formula:
    log1p_level = sign(v[t]) * log1p(|v[t]|)
        = ln(1 + |v[t]|) если v[t] >= 0
        = -ln(1 + |v[t]|) если v[t] < 0

Outputs:
    {product}__log1p_level   — знаковый log1p текущего значения (без суффикса)

Preset (monthly.yaml):
    log1p_level: {}

Interpretation:
    v = 0    → log1p_level = 0 (неактивный месяц).
    v = 100  → log1p_level ≈ 4.61.
    v = 1000 → log1p_level ≈ 6.91 (не в 10× больше, а в 1.5×).
    Разница log1p между двумя клиентами примерно равна log-ratio их уровней.
    Используется как признак уровня для линейных моделей вместо сырого v.

Example:
    Ряд (4 мес): [0, 25, 50, 100]
    (t=3; преобразуется только текущее значение v[t]=100)

    log1p_level = sign(v)·ln(1 + |v|) = +ln(101)
    → log1p_level = 4.615

"""

import numba as nb
import numpy as np

FEATURE = 'log1p_level'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out = np.zeros(n_rows)
    for row_idx in range(n_rows):
        v = product_values[row_idx]
        sign = 1.0 if v >= 0.0 else -1.0
        out[row_idx] = sign * np.log1p(abs(v))
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    return [_kernel(values, position)], ['']
