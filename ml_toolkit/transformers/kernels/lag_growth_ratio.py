"""Относительный рост текущего значения к значению k месяцев назад.

Signal:
    Измеряет, насколько текущий доход превышает доход k месяцев назад (QoQ, HoH, YoY).
    Используется в structural_signals для оценки динамики на нескольких горизонтах.
    Отличие от lag_comparison: поддерживает произвольные лаги через параметр.

Formula:
    lag_growth_ratio_lagK = v[t] / (|v[t-k]| + eps) - 1
    Равно 0 при position < k.

Outputs:
    {product}__lag_growth_ratio__lag3   — рост vs 3 мес назад
    {product}__lag_growth_ratio__lag6   — рост vs 6 мес назад
    {product}__lag_growth_ratio__lag12  — рост vs 12 мес назад

Preset (monthly.yaml):
    lag_growth_ratio:
      lags: [3, 6, 12]

Interpretation:
    lag12 = +0.5 — доход вырос на 50% год к году.
    lag6 = +0.3, lag12 = 0 — рост произошёл в последние полгода.
    lag3 > lag6 > lag12 — ускорение роста (разные горизонты нарастают).
    lag3 < lag12 < 0 — системный спад на всех горизонтах.

Example:
    Ряд (4 мес): [10, 20, 30, 40],  lag=3
    (t=3; сравнение с v[t-3]=10)

    lag_growth_ratio = v[t] / v[t-3] − 1 = 40/10 − 1 = 3.0
    → lag_growth_ratio__lag3 = 3.0  (рост ×4 за 3 мес)

"""

import numba as nb
import numpy as np

from .._windowing import EPS, safe_ratio

FEATURE = 'lag_growth_ratio'


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, lags: np.ndarray):
    n_rows = product_values.shape[0]
    n_l = lags.shape[0]
    out = np.zeros((n_l, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        v = product_values[row_idx]
        for j in range(n_l):
            lag = lags[j]
            if pos >= lag:
                v_lag = product_values[row_idx - lag]
                # при нулевой базе рост не определён -> 0, а не v/eps
                if abs(v_lag) > EPS:
                    out[j, row_idx] = safe_ratio(v, v_lag) - 1.0
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"lags": [6, 12]}"""
    lags = np.array(params['lags'], dtype=np.int64)
    out = _kernel(values, position, lags)
    return [out[j] for j in range(len(lags))], [f'lag{l}' for l in params['lags']]
