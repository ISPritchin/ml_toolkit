"""Длина текущей серии последовательных ростов или падений (running state).

Signal:
    Фиксирует текущую «полосу» непрерывного роста или падения. streak_up > 0 при
    streak_down = 0 и наоборот (взаимно исключающие). Высокое значение streak_up
    — устойчивый восходящий импульс без единого откатного месяца.

Formula:
    streak_up[t]   = streak_up[t-1] + 1   if v[t] > v[t-1], else 0
    streak_down[t] = streak_down[t-1] + 1 if v[t] < v[t-1], else 0

    Сбрасывается при нарушении строгого неравенства (плоский шаг → обе серии = 0).
    При pos = 0 — обе серии = 0.

Outputs:
    {product}__streak__up   — длина текущей непрерывной серии роста
    {product}__streak__down — длина текущей непрерывной серии падения

Preset (monthly.yaml):
    streak: {}

Interpretation:
    streak_up = 6 — шесть месяцев подряд значения только растут.
    streak_down = 3 — три месяца снижения подряд (потенциальный сигнал риска).
    streak_up = streak_down = 0 — текущий месяц не продолжает ни одну из серий.
    Для ряда G все 11 шагов строго возрастающие: на последнем шаге streak_up = 11.

Example:
    Ряд (5 мес): [10, 20, 30, 40, 50]
    (t=4; каждый месяц строго выше предыдущего)

    streak_up растёт: t=1→1, t=2→2, t=3→3, t=4→4
    streak_down = 0 (ни одного снижения)
    → streak__up = 4,  streak__down = 0

"""

import numba as nb
import numpy as np

FEATURE = 'streak'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    streak_up = np.zeros(n_rows)
    streak_down = np.zeros(n_rows)
    for row_idx in range(n_rows):
        if position_within_entity[row_idx] == 0:
            continue
        if product_values[row_idx] > product_values[row_idx - 1]:
            streak_up[row_idx] = streak_up[row_idx - 1] + 1.0
        elif product_values[row_idx] < product_values[row_idx - 1]:
            streak_down[row_idx] = streak_down[row_idx - 1] + 1.0
    return streak_up, streak_down


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    up, down = _kernel(values, position)
    return [up, down], ['up', 'down']
