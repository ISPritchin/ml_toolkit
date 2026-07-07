"""Доля активных месяцев от срока использования (running state).

Signal:
    Отражает, насколько интенсивно клиент использует продукт за весь период с первого
    ненулевого наблюдения. Близко к 1 — клиент активен практически каждый месяц.
    Ниже 0.3 — хроническая нестабильность или проектный характер доходов.

Formula:
    tenure = position - first_active_position + 1
    share_of_tenure_active = count_active_total / tenure

    Выход равен 0, если активности ещё не было (first_active_position == -1).

Outputs:
    {product}__activity_rate__share_of_tenure_active — доля активных месяцев за весь стаж

Preset (monthly.yaml):
    activity_rate: {}

Interpretation:
    > 0.9 — потоковый клиент, практически без пропусков.
    0.4–0.9 — смешанный паттерн, возможна сезонность или проектные провалы.
    < 0.3 — редкий/проектный клиент, большинство месяцев нулевые.
    Вместе с tenure позволяет отличить «молодого непрерывного» от «старого нерегулярного».

Example:
    Ряд (4 мес): [0, 10, 0, 8]
    (t=3; позиции внутри сущности pos=0..3)

    первая активность: pos=1  → first_active_pos = 1
    активных всего:    10, 8  → total_active = 2
    tenure = 3 − 1 + 1 = 3
    → activity_rate__share_of_tenure_active = 2/3 = 0.667

"""

import numba as nb
import numpy as np

FEATURE = 'activity_rate'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out = np.zeros(n_rows)
    total_active = 0
    first_active_pos = -1
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            total_active = 0
            first_active_pos = -1
        if product_values[row_idx] != 0.0:
            if first_active_pos == -1:
                first_active_pos = pos
            total_active += 1
        if first_active_pos >= 0:
            tenure = pos - first_active_pos + 1
            out[row_idx] = total_active / tenure
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    return [_kernel(values, position)], ['share_of_tenure_active']
