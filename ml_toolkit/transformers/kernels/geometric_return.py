"""Среднемесячный геометрический возврат: exp(mean(log_diff)) - 1.

Signal:
    Оценивает среднемесячный мультипликативный темп роста в log-шкале, устойчив к
    экспоненциальному распределению значений. Положительный — ряд растёт, отрицательный
    — снижается; интерпретируется как «средний % изменения в месяц».

Formula:
    log_diff[i] = log1p(|v[i]|) - log1p(|v[i-1]|)   для i in [t-w+2..t]
    geometric_return_w = exp(mean(log_diff)) - 1

    Требует n_diffs >= 1 (ws >= 2).

Outputs:
    {product}__geometric_return__w6   — геом. возврат за 6 мес
    {product}__geometric_return__w12  — геом. возврат за 12 мес

Preset entry:
    geometric_return:
      windows: [6, 12]
      dilations: [1, 2]   # optional, default [1]; d>1 measures d-step geometric growth

Interpretation:
    ≈ +0.19 — рост ≈ 19% в месяц (экспоненциальный разгон, как в лог-примере).
    ≈ 0 — ряд стагнирует в log-шкале.
    < 0 — систематическое снижение.
    В паре с log_volatility: высокий возврат + высокая log_vol = рост с высоким риском.

Example:
    Ряд (4 мес): [10, 20, 40, 80],  w=4  (удвоение каждый месяц)

    log_diff = log1p(20)−log1p(10), log1p(40)−log1p(20), log1p(80)−log1p(40)
             ≈ 0.647, 0.669, 0.681   (n_diffs = 3)
    mean(log_diff) = 0.665
    geometric_return = exp(0.665) − 1 = 0.945
    → geometric_return__w4 = 0.945  (≈ +95% в месяц)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import resolve_window_size

FEATURE = 'geometric_return'
FILL_NAN: dict[str | None, float] = {None: -2.0}  # exp(mean(log_diff))-1 > -1 всегда (exp строго положителен) — -2 недостижим


@nb.njit(cache=True)
def _kernel(
    log_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray, dilations: np.ndarray
):
    n_rows = log_values.shape[0]
    n_w = windows.shape[0]
    n_d = dilations.shape[0]
    out = np.zeros((n_w * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_w):
            for d in range(n_d):
                dilation = dilations[d]
                ws = resolve_window_size(pos, windows[j], dilation)
                n_diffs = ws - 1
                if n_diffs >= 1:
                    base = row_idx - (ws - 1) * dilation
                    # сумма лог-разностей телескопируется: lv[t] - lv[base] — O(1) на окно
                    ld_sum = log_values[row_idx] - log_values[base]
                    out[k, row_idx] = np.exp(ld_sum / n_diffs) - 1.0
                k += 1
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    windows = np.array(params['windows'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    # log1p считается один раз на колонку (векторно)
    log_values = np.log1p(np.abs(values))
    out = _kernel(log_values, position, windows, dilations)
    suffixes = [f'w{w}' if d == 1 else f'w{w}_d{d}' for w in params['windows'] for d in params.get('dilations', [1])]
    return [out[k] for k in range(len(suffixes))], suffixes
