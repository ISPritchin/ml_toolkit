"""Среднемесячный геометрический возврат: exp(mean(log_diff)) - 1.

Signal:
    Оценивает среднемесячный мультипликативный темп роста в log-шкале, устойчив к
    экспоненциальному распределению оборотов. Положительный — доходы растут, отрицательный
    — снижаются; интерпретируется как «средний % изменения в месяц».

Formula:
    log_diff[i] = log1p(|v[i]|) - log1p(|v[i-1]|)   для i in [t-w+2..t]
    geometric_return_w = exp(mean(log_diff)) - 1

    Требует n_diffs >= 1 (ws >= 2).

Outputs:
    {product}__geometric_return__w6   — геом. возврат за 6 мес
    {product}__geometric_return__w12  — геом. возврат за 12 мес

Preset (monthly.yaml):
    geometric_return:
      windows: [6, 12]

Interpretation:
    ≈ +0.19 — рост ≈ 19% в месяц (экспоненциальный разгон, как в лог-примере).
    ≈ 0 — доходы стагнируют в log-шкале.
    < 0 — систематическое снижение.
    В паре с log_volatility: высокий возврат + высокая log_vol = рост с высоким риском.

Example:
    Ряд (4 мес): [10, 20, 40, 80],  w=4  (удвоение каждый месяц)

    log_diff = log1p(20)−log1p(10), log1p(40)−log1p(20), log1p(80)−log1p(40)
             ≈ 0.647, 0.669, 0.681   (n_diffs = 3)
    mean(log_diff) = 0.665
    geometric_return = exp(0.665) − 1 = 0.946
    → geometric_return__w4 = 0.946  (≈ +95% в месяц)
"""

import numba as nb
import numpy as np

from .._windowing import resolve_window_size

FEATURE = "geometric_return"


@nb.njit(cache=True)
def _kernel(log_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = log_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            n_diffs = ws - 1
            if n_diffs >= 1:
                # сумма лог-разностей телескопируется: lv[t] - lv[t-ws+1] — O(1) на окно
                ld_sum = log_values[row_idx] - log_values[row_idx - ws + 1]
                out[j, row_idx] = np.exp(ld_sum / n_diffs) - 1.0
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    # log1p считается один раз на колонку (векторно)
    log_values = np.log1p(np.abs(values))
    out = _kernel(log_values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
