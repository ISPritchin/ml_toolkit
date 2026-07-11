"""Наибольшая непрерывная серия активных (ненулевых) месяцев в окне.

Signal:
    Показывает максимальную «полосу непрерывной активности» за период. Отличается от
    active_months: ряд может иметь 10 активных из 12, но longest_active_run = 3
    (все короткие серии с перерывами) — прерывистый паттерн.

Formula:
    longest_active_run_w = max длина непрерывной серии v[i] != 0 в окне [t-w+1..t]

Outputs:
    {product}__longest_active_run__w6   — наибольшая непрерыв. серия за 6 мес
    {product}__longest_active_run__w12  — наибольшая непрерыв. серия за 12 мес

Preset entry:
    longest_active_run:
      windows: [6, 12]

Interpretation:
    = w — ряд активен весь период без перерывов (непрерывный).
    = 1 — только одиночные активные месяцы, ни разу два подряд (пульсирующий).
    = 6 при w12 — минимум полугодовая непрерывная активность за год.
    Разница (active_months - longest_active_run) ≈ число «прерванных» серий.

Example:
    Ряд (6 мес): [10, 10, 0, 10, 10, 10],  w=6

    непрерывные активные серии: [10,10] (длина 2), [10,10,10] (длина 3)
    наибольшая = 3
    → longest_active_run__w6 = 3

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'longest_active_run'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            cur_run = 0
            best_run = 0
            for offset in range(ws):
                if product_values[row_idx - ws + 1 + offset] != 0.0:
                    cur_run += 1
                    best_run = max(best_run, cur_run)
                else:
                    cur_run = 0
            out[j, row_idx] = best_run
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}."""
    windows = np.array(params['windows'], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f'w{w}' for w in params['windows']]
