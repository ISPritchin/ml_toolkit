"""Доля текущего значения в накопленной сумме с начала истории (running state).

Signal:
    Показывает, какую долю совокупного исторического дохода составил текущий месяц.
    Высокое значение — текущий месяц является пиковым (возможно, контрактный платёж),
    низкое — клиент генерировал значительно больший оборот ранее.

Formula:
    cum_sum[t] = sum(v[i], i in [t0..t])    (running, сбрасывается при pos=0)
    cumulative_share = v[t] / (|cum_sum[t]| + eps)

Outputs:
    {product}__cumulative_share   — доля текущего мес. в накопленной сумме (без суффикса)

Preset (monthly.yaml):
    cumulative_share: {}

Interpretation:
    > 0.2 — текущий месяц даёт >20% всего накопленного дохода: аномально крупный.
    ≈ 1/n где n = tenure — клиент равномерно активен на протяжении всей истории.
    Снижается монотонно для равномерного ряда; резкий скачок вверх указывает на
    контрактный «выброс» или перезапуск активности.

Example:
    Ряд (4 мес): [10, 20, 30, 40]
    (t=3; running-сумма с начала истории)

    cum_sum = 10+20+30+40 = 100
    cumulative_share = v[t] / cum_sum = 40/100
    → cumulative_share = 0.40  (текущий мес. — 40% всего накопленного)
"""

import numba as nb
import numpy as np

from .._windowing import safe_ratio

FEATURE = "cumulative_share"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out = np.zeros(n_rows)
    cum_sum = 0.0
    for row_idx in range(n_rows):
        if position_within_entity[row_idx] == 0:
            cum_sum = 0.0
        cum_sum += product_values[row_idx]
        out[row_idx] = safe_ratio(product_values[row_idx], cum_sum)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    return [_kernel(values, position)], [""]
