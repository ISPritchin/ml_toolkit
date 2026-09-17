"""Симметричный моментум: mean(последние N мес) / |mean(предыдущие N мес)| - 1.

Signal:
    Сравнивает два смежных периода одинаковой длины: «насколько лучше (хуже) последнее
    полугодие/квартал относительно предыдущего?» Более робастен к выбросам, чем pct_change,
    поскольку использует средние, а не точечные значения.

Formula:
    recent_mean = mean(v[t], v[t-1], ..., v[t-h+1])      последние h мес
    prior_mean  = mean(v[t-h], v[t-h-1], ..., v[t-2h+1]) предыдущие h мес
    momentum_h  = recent_mean / (|prior_mean| + eps) - 1

    Требует position >= 2*h - 1.

Outputs:
    {product}__momentum__h3  — моментум 3/3 мес (квартальный)
    {product}__momentum__h6  — моментум 6/6 мес (полугодовой)
    {product}__momentum__h3_d2 — моментум 3/3 с шагом 2 (при dilations=[1,2])

Preset entry:
    momentum:
      half_windows: [3, 6]
      dilations: [1, 2]   # optional, default [1]; d>1 spaces the recent/prior blocks' taps
                          # d rows apart (each block's own span becomes (h-1)*d+1)

Interpretation:
    momentum_h6 = +1.33 — последнее полугодие в 2.33× выше предыдущего (пример ряда G).
    momentum_h3 > 0, h6 < 0 — краткосрочный отскок при среднесрочном спаде.
    Оба > 0 — устойчивое ускорение на двух горизонтах.
    = -1 при prior_mean > 0 и recent_mean = 0 — полное прекращение активности.

Example:
    Ряд (6 мес): [10, 20, 30, 40, 50, 60],  h=3

    recent_mean = (40+50+60)/3 = 50   (последние 3 мес)
    prior_mean  = (10+20+30)/3 = 20   (предыдущие 3 мес)
    momentum = 50/20 − 1 = 1.5
    → momentum__h3 = 1.5  (последний квартал в 2.5× выше предыдущего)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import EPS, FILL_NAN_UNBOUNDED_LOW, safe_ratio

FEATURE = 'momentum'
FILL_NAN: dict[str | None, float] = {None: FILL_NAN_UNBOUNDED_LOW}


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    half_windows: np.ndarray,
    dilations: np.ndarray,
):
    n_rows = product_values.shape[0]
    n_h = half_windows.shape[0]
    n_d = dilations.shape[0]
    out = np.zeros((n_h * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_h):
            h = half_windows[j]
            for d_idx in range(n_d):
                dilation = dilations[d_idx]
                if pos >= (2 * h - 1) * dilation:
                    recent_sum = 0.0
                    prior_sum = 0.0
                    for offset in range(h):
                        recent_sum += product_values[row_idx - offset * dilation]
                        prior_sum += product_values[row_idx - h * dilation - offset * dilation]
                    prior_mean = prior_sum / h
                    # при нулевой базе (prior_mean ~ 0) моментум не определён -> 0, а не -1
                    if abs(prior_mean) > EPS:
                        out[k, row_idx] = safe_ratio(recent_sum / h, prior_mean) - 1.0
                k += 1
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"half_windows": [3, 6], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    half_windows = np.array(params['half_windows'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    out = _kernel(values, position, half_windows, dilations)
    suffixes = [
        f'h{h}' if d == 1 else f'h{h}_d{d}' for h in params['half_windows'] for d in params.get('dilations', [1])
    ]
    return [out[k] for k in range(len(suffixes))], suffixes
