"""Коэффициент вариации Пирсона (std / |mean|) на скользящем окне.

Signal:
    Нормированная мера изменчивости: насколько велики отклонения относительно уровня.
    Позволяет сравнивать волатильность клиентов разного масштаба. Высокое CV —
    непредсказуемый клиент; низкое — стабильный поток.

Formula:
    mean_w, std_w — среднее и станд. отклонение (смещённое) окна
    rolling_cv_w = std_w / (|mean_w| + eps)

Outputs:
    {product}__rolling_cv__w6   — CV за 6 мес
    {product}__rolling_cv__w12  — CV за 12 мес
    {product}__rolling_cv__w24  — CV за 24 мес

Preset (monthly.yaml):
    rolling_cv:
      windows: [6, 12, 24]

Interpretation:
    = 0 — абсолютно стабильный ряд (все значения одинаковы).
    = 0.76 — высокая волатильность (ряд V, осциллирующий).
    > 1 — экстремальная нестабильность (нулевые месяцы при высоком среднем).
    cv_w6 > cv_w12 — нестабильность нарастает в последнее время.
    Используется в microstructure (predictability = 1/(1+CV)).

Example:
    Ряд (6 мес): [10, 10, 10, 10, 10, 40],  w=6

    mean = 90/6 = 15
    std  = sqrt(mean((v−15)²)) = 11.18
    rolling_cv = std / |mean| = 11.18 / 15 = 0.745
    → rolling_cv__w6 = 0.745  (заметная волатильность от всплеска 40)

"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean_and_std, resolve_window_size, safe_ratio

FEATURE = 'rolling_cv'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean, std = compute_window_mean_and_std(product_values, row_idx, ws)
            out[j, row_idx] = safe_ratio(std, mean)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12, 24]}"""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
