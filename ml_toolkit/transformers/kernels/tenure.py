"""Срок использования продукта и флаг первой активации (running state).

Signal:
    tenure_months показывает, сколько времени прошло с первого ненулевого значения ряда
    (первой активности). first_active_flag помечает сам момент первого появления,
    что полезно для разметки «вход» — «первое ненулевое значение».

Formula:
    Tracking: first_active_position — позиция при первом v != 0 (сбрасывается при pos=0)
    tenure_months[t] = pos - first_active_position + 1    если first активирован
                     = 0                                   до первой активации
    first_active_flag = 1 только в строке, где впервые v != 0 (единственный раз)

Outputs:
    {product}__tenure__tenure_months     — месяцев с первой активации
    {product}__tenure__first_active_flag — флаг строки первой активации (0 или 1)

Preset entry:
    tenure: {}

Interpretation:
    tenure_months = 0 — ещё не было ни одного ненулевого значения.
    tenure_months = 12 — ряд активен (или наблюдается) уже год с первого ненулевого значения.
    first_active_flag = 1 — этот конкретный месяц был первым (используется для когортного анализа).
    Ряд с tenure = 36 и activity_rate = 0.3 — долгая история, но нерегулярная активность.

Example:
    Ряд (5 мес): [0, 0, 10, 20, 30]
    (t=4; первая активность на pos=2)

    first_active_position = 2
    tenure_months = pos − first_active_position + 1 = 4 − 2 + 1 = 3
    first_active_flag = 0 (флаг=1 был только в строке pos=2)
    → tenure__tenure_months = 3,  first_active_flag = 0

"""

import numba as nb
import numpy as np

FEATURE = 'tenure'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    tenure_months = np.zeros(n_rows)
    first_active_flag = np.zeros(n_rows)
    first_active_position = -1
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            first_active_position = -1
        if product_values[row_idx] != 0.0:
            if first_active_position == -1:
                first_active_position = pos
                first_active_flag[row_idx] = 1.0
        if first_active_position >= 0:
            tenure_months[row_idx] = pos - first_active_position + 1
    return tenure_months, first_active_flag


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    t, f = _kernel(values, position)
    return [t, f], ['tenure_months', 'first_active_flag']
