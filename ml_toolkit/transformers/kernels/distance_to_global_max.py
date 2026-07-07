"""Относительное отстояние от all-time максимума сущности (running state).

Signal:
    Показывает, насколько текущий уровень ниже исторического пика клиента/холдинга.
    Значение 0 — клиент на историческом максимуме прямо сейчас. Большое отрицательное
    значение — клиент далеко от своего пика (глубокая просадка или стагнация после пика).

Formula:
    running_max[t] = max(v[0], v[1], ..., v[t])   (обновляется streaming)
    distance_to_global_max = (v[t] - running_max[t]) / (|running_max[t]| + eps)

    Всегда <= 0; равно 0 только в момент нового исторического максимума.

Outputs:
    {product}__distance_to_global_max   — отн. отстояние от all-time max (без суффикса)

Preset (monthly.yaml):
    distance_to_global_max: {}

Interpretation:
    = 0 — клиент устанавливает новый исторический максимум (is_new_peak = 1).
    = -0.1 — на 10% ниже исторического пика, небольшая коррекция.
    < -0.5 — клиент упал более чем на 50% от пика: критическое снижение.
    В паре с lifecycle_phase помогает идентифицировать фазу «снижения после зрелости».

Example:
    Ряд (4 мес): [10, 30, 20, 25]
    (t=3; running_max с начала истории)

    running_max = max(10, 30, 20, 25) = 30
    distance = (v[t] − running_max) / running_max = (25 − 30)/30 = −0.1667
    → distance_to_global_max = −0.167  (на ~17% ниже исторического пика)

"""

import numba as nb
import numpy as np

from .._windowing import safe_ratio

FEATURE = 'distance_to_global_max'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray):
    n_rows = product_values.shape[0]
    out = np.zeros(n_rows)
    running_max = 0.0
    for row_idx in range(n_rows):
        if position_within_entity[row_idx] == 0:
            running_max = product_values[row_idx]
        v = product_values[row_idx]
        running_max = max(running_max, v)
        out[row_idx] = safe_ratio(v - running_max, running_max)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {} — параметры не используются."""
    return [_kernel(values, position)], ['']
