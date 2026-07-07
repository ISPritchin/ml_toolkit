"""Давность последней активности: месяцев с последнего ненулевого значения (running state).

Signal:
    Измеряет, как давно клиент проявлял активность. recency_gap = 0 означает активность
    прямо сейчас; большие значения — клиент давно молчит (риск оттока). -1 = активности
    ещё не было (новый клиент без первого платежа).

Formula:
    last_active_position — позиция последнего v[i] != 0, включая текущий месяц
    (running state; обновляется ДО вычисления gap)
    recency_gap = current_position - last_active_position

    Значение: 0 если текущий месяц активен, > 0 если последняя активность была
    ранее, -1 до первой активности.

Outputs:
    {product}__recency__recency_gap — месяцев с последней активности (-1 если не было)

Preset (monthly.yaml):
    recency: {}

Interpretation:
    = 0 — активен прямо сейчас (v[t] != 0).
    = 1 — прошлый месяц был активен, текущий нет.
    > 6 — 6+ месяцев молчания (серьёзный риск потери клиента).
    = -1 — клиент ещё не совершил ни одного платежа.

Example:
    Ряд (5 мес): [10, 0, 0, 30, 0]
    (t=4; позиции pos=0..4; по ряду gap = [0, 1, 2, 0, 1])

    последняя активность: pos=3 (значение 30)
    recency_gap = current_pos − last_active_pos = 4 − 3 = 1
    → recency__recency_gap = 1  (молчит 1 месяц)

"""

import numba as nb
import numpy as np

FEATURE = 'recency'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out = np.zeros(n_rows)
    last_active_position = -1
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        if pos == 0:
            last_active_position = -1
        if product_values[row_idx] != 0.0:
            last_active_position = pos
        if last_active_position >= 0:
            out[row_idx] = pos - last_active_position
        else:
            out[row_idx] = -1.0
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    return [_kernel(values, position)], ['recency_gap']
