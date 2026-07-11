"""Экспоненциально взвешенное скользящее среднее и отклонение от него.

Signal:
    EWMA сглаживает ряд с бо́льшим весом последних наблюдений, устойчив к выбросам.
    Отклонение (ewma_diff = v[t] - EWMA[t]) показывает, «выше» или «ниже» нормы
    текущий месяц: положительное — приятный сюрприз, отрицательное — провал.

Formula:
    EWMA[0] = v[0]
    EWMA[t] = alpha * v[t] + (1 - alpha) * EWMA[t-1]
    ewma_diff[t] = v[t] - EWMA[t]

    alpha = 0.3: примерно 3-4 месяца «эффективной памяти».
    Тег суффикса: a30 для alpha=0.30.

Outputs:
    {product}__ewma__a30      — EWMA с alpha=0.30
    {product}__ewma__diff_a30 — текущее минус EWMA (отклонение от нормы)

Preset (monthly.yaml):
    ewma:
      alphas: [0.3]

Interpretation:
    ewma_diff > 0 — текущий месяц выше сглаженного тренда (позитивный импульс).
    ewma_diff < 0 — текущий месяц ниже нормы (потенциальный провал).
    EWMA > скользящего среднего w12 — последние месяцы тянут уровень вверх.
    Стабильно нулевой ewma_diff при ненулевом ряде = идеально ровный поток.

Example:
    Ряд (4 мес): [100, 80, 120, 90],  alpha=0.3

    t=0: EWMA = 100
    t=1: EWMA = 0.3·80  + 0.7·100 = 24 + 70 = 94.0
    t=2: EWMA = 0.3·120 + 0.7·94  = 36 + 65.8 = 101.8
    t=3: EWMA = 0.3·90  + 0.7·101.8 = 27 + 71.26 = 98.26
    → ewma__a30 = 98.26,  ewma__diff_a30 = 90 − 98.26 = −8.26

"""

import numba as nb
import numpy as np

FEATURE = 'ewma'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, alpha: float):
    n_rows = product_values.shape[0]
    out_ewma = np.zeros(n_rows)
    out_diff = np.zeros(n_rows)
    running = 0.0
    for row_idx in range(n_rows):
        if position_within_entity[row_idx] == 0:
            running = product_values[row_idx]
        else:
            running = alpha * product_values[row_idx] + (1.0 - alpha) * running
        out_ewma[row_idx] = running
        out_diff[row_idx] = product_values[row_idx] - running
    return out_ewma, out_diff


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"alphas": [0.3]} — ключ обязателен, дефолты задаёт пресет."""
    alphas = params['alphas']
    arrays = []
    suffixes = []
    for alpha in alphas:
        tag = f'a{round(alpha * 100):02d}'
        ev, ed = _kernel(values, position, alpha)
        arrays.extend([ev, ed])
        suffixes.extend([tag, f'diff_{tag}'])
    return arrays, suffixes
