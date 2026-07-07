"""Серии неактивности: текущая и максимальная за всю историю (running state).

Signal:
    Показывает текущий «простой» клиента (current) и наибольшую паузу за всю историю
    (max). Если current > 0 — клиент не активен прямо сейчас. Высокий max_streak при
    низком current — в прошлом была долгая дормантность, сейчас восстанавливается.

Formula:
    current_streak: текущая длина непрерывной серии нулей (running state)
        сбрасывается при v[t] != 0
    max_streak: максимальная серия нулей за всё время (running state)
        обновляется при cur > longest

Outputs:
    {product}__inactive_streak__current  — текущая серия нулей
    {product}__inactive_streak__max      — исторический максимум серии нулей

Preset (monthly.yaml):
    inactive_streak: {}

Interpretation:
    current = 0 — клиент активен в текущем месяце.
    current > 3 — клиент молчит более квартала: потенциальный отток.
    max = 6, current = 0 — когда-то выпадал на полгода, сейчас восстановился.
    max > 12 — хроническая нестабильность в прошлом (дормантный клиент-возвращенец).

Example:
    Ряд (5 мес): [10, 0, 0, 10, 0]
    (t=4; running-счётчики серий нулей)

    серии нулей: idx1,2 (длина 2), затем idx4 (длина 1)
    в текущем мес. v=0 → current = 1
    исторический максимум серии нулей = 2
    → inactive_streak__current = 1,  inactive_streak__max = 2

"""

import numba as nb
import numpy as np

FEATURE = 'inactive_streak'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    current_streak = np.zeros(n_rows)
    max_streak = np.zeros(n_rows)
    cur = 0
    longest = 0
    for row_idx in range(n_rows):
        if position_within_entity[row_idx] == 0:
            cur = 0
            longest = 0
        if product_values[row_idx] != 0.0:
            cur = 0
        else:
            cur += 1
            longest = max(longest, cur)
        current_streak[row_idx] = cur
        max_streak[row_idx] = longest
    return current_streak, max_streak


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    cur, mx = _kernel(values, position)
    return [cur, mx], ['current', 'max']
