"""Относительный прирост от первого ненулевого значения entity (running state).

Signal:
    Показывает, во сколько раз (в %) вырос клиент от момента первой активности
    до текущего месяца. Высокое значение — кратный рост за всю историю; отрицательное
    — клиент сейчас ниже стартового уровня (регресс).

Formula:
    first_nonzero = v[first_active_position]
    growth_since_start = (v[t] - first_nonzero) / (|first_nonzero| + eps)

    Running state: first_nonzero запоминается при первом v != 0 и сбрасывается при pos=0.
    До первой активности значение = 0.

Outputs:
    {product}__growth_since_start   — совокупный рост от старта (без суффикса)

Preset (monthly.yaml):
    growth_since_start: {}

Interpretation:
    = +1.0 — доход удвоился с момента первой активности.
    = 0 — вернулся на стартовый уровень (после роста и падения).
    < -0.5 — потерял более половины от стартового уровня: сильный регресс.
    Для молодых клиентов (tenure < 6 мес) значение нестабильно из-за малой базы.

Example:
    Ряд (3 мес): [0, 10, 30]
    (t=2; первое ненулевое значение = 10)

    growth_since_start = (v[t] − first_nonzero) / first_nonzero = (30 − 10)/10
    → growth_since_start = 2.0  (рост ×3 от стартового уровня)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import safe_ratio

FEATURE = 'growth_since_start'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out = np.zeros(n_rows)
    first_nonzero = 0.0
    for row_idx in range(n_rows):
        if position_within_entity[row_idx] == 0:
            first_nonzero = 0.0
        if first_nonzero == 0.0 and product_values[row_idx] != 0.0:
            first_nonzero = product_values[row_idx]
        if first_nonzero != 0.0:
            out[row_idx] = safe_ratio(product_values[row_idx] - first_nonzero, first_nonzero)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    return [_kernel(values, position)], ['']
