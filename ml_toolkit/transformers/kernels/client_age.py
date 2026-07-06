"""Возраст клиента: флаг нового клиента и нормированное время с начала.

Signal:
    Показывает, насколько «зрелым» является клиент в наблюдаемом ряду.
    Новые клиенты (position < 3) имеют ограниченную историю и требуют других
    ожиданий от признаков, чем клиенты с длинной историей.

Formula:
    new_client_flag         = 1 if position < 3 else 0
    months_since_start_norm = position / (position + 12)

    Нормировка через (pos + 12) сжимает шкалу: для pos=0 → 0, pos=12 → 0.5,
    pos=60 → 0.83, что дает нелинейный «сигмоид» возраста.

Outputs:
    {product}__client_age__new_client_flag        — флаг молодого клиента (pos < 3)
    {product}__client_age__months_since_start_norm — нормированный возраст [0, 1)

Preset (monthly.yaml):
    client_age: {}

Interpretation:
    new_client_flag = 1 — клиент с историей менее 3 месяцев, признаки тренда ненадёжны.
    months_since_start_norm > 0.8 — клиент с историей > 5 лет (достаточно данных).
    Используется как вспомогательный признак для взвешивания других фич по надёжности.

Example:
    Ряд (5 мес): [10, 20, 30, 40, 50]
    (t=4; позиция внутри сущности pos=4)

    new_client_flag = 1 if pos<3 else 0 → pos=4 ≥ 3 → 0
    months_since_start_norm = pos/(pos+12) = 4/16 = 0.25
    → client_age__new_client_flag = 0.0
    → client_age__months_since_start_norm = 0.25
"""

import numba as nb
import numpy as np

FEATURE = "client_age"


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    new_client_months: int,
    norm_months: float,
):
    n_rows = product_values.shape[0]
    new_client_flag = np.zeros(n_rows)
    months_since_start_norm = np.zeros(n_rows)
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        new_client_flag[row_idx] = 1.0 if pos < new_client_months else 0.0
        months_since_start_norm[row_idx] = pos / (pos + norm_months)
    return new_client_flag, months_since_start_norm


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"new_client_months": 3, "norm_months": 12 (опционально)}"""
    new_client_months = int(params.get("new_client_months", 3))
    norm_months = float(params.get("norm_months", 12.0))
    ncf, msn = _kernel(values, position, new_client_months, norm_months)
    return [ncf, msn], ["new_client_flag", "months_since_start_norm"]
