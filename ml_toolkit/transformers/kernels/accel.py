"""Вторая разность: ускорение или торможение в текущем месяце.

Signal:
    Показывает, ускоряется ли динамика клиента (рост темпа роста) или тормозит
    (снижение темпа роста). Положительное значение — ускорение (тренд набирает силу),
    отрицательное — замедление или торможение.

Formula:
    accel = (v[t] - v[t-1]) - (v[t-1] - v[t-2])
           = v[t] - 2*v[t-1] + v[t-2]

    Равно нулю при p < 2 (недостаточно истории).

Outputs:
    {product}__accel   — вторая разность текущего месяца (без суффикса)

Preset (monthly.yaml):
    accel: {}

Interpretation:
    > 0 — тренд ускоряется (приросты нарастают), клиент в фазе разгона.
    < 0 — тренд тормозит (приросты уменьшаются), возможный разворот вниз.
    = 0 — равномерное движение, линейный рост/падение или плато.
    Полезно в паре со slope: большой slope + положительный accel = «разгон»,
    большой slope + отрицательный accel = «усталость тренда».

Example:
    Ряд (4 мес): [10, 15, 25, 30]
    (Результат на последнем шаге t=3)

    прирост t-1→t   = 30 − 25 = 5
    прирост t-2→t-1 = 25 − 15 = 10
    accel = 5 − 10 = −5   (или 30 − 2·25 + 15 = −5)
    → accel = −5.0  (тренд тормозит)
"""

import numba as nb
import numpy as np

FEATURE = "accel"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out = np.zeros(n_rows)
    for row_idx in range(n_rows):
        if position_within_entity[row_idx] >= 2:
            out[row_idx] = (
                product_values[row_idx]
                - 2.0 * product_values[row_idx - 1]
                + product_values[row_idx - 2]
            )
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    return [_kernel(values, position)], [""]
