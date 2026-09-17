"""Отношение log-наклонов на коротком и длинном окне: ускорение в лог-шкале.

Signal:
    Показывает, ускоряется ли темп роста (в log-шкале) в последнее время относительно
    длинного горизонта. Значение > 1 — краткосрочный log-рост быстрее долгосрочного
    (разгон); < 1 — замедление темпа.

Formula:
    log_slope_short = OLS_slope(log1p(|v|), окно ws_short)
    log_slope_long  = OLS_slope(log1p(|v|), окно ws_long)
    log_slope_ratio_wS_wL = log_slope_short / (|log_slope_long| + eps)

Outputs:
    {product}__log_slope_ratio__w6_w12  — log_slope_6 / |log_slope_12|

Preset entry:
    log_slope_ratio:
      pairs:
        - [6, 12]
      dilations: [1, 2]   # optional, default [1]; applies to both windows of every pair

Interpretation:
    > 1 — краткосрочный log-темп роста выше долгосрочного (ускорение).
    ≈ 1 — темп стабилен на обоих горизонтах.
    < 0 — знаки наклонов разошлись: краткосрочный разворот тренда.
    Используется в паре с log_slope для диагностики «разгон vs стагнация».

Example:
    Ряд (6 мес): [10, 20, 40, 80, 160, 320],  пара (3, 6)
    (t=5; удвоение каждый месяц)

    log_slope_short (окно 3, посл. [80,160,320]) = 0.689
    log_slope_long  (окно 6, весь ряд)           = 0.676
    log_slope_ratio = 0.689 / 0.676 = 1.019
    → log_slope_ratio__w3_w6 = 1.019  (краткосрочный log-темп чуть выше)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import (
    FILL_NAN_UNBOUNDED_LOW,
    fit_linear_trend_slope,
    resolve_window_size,
    safe_ratio,
)

FEATURE = 'log_slope_ratio'
FILL_NAN: dict[str | None, float] = {None: FILL_NAN_UNBOUNDED_LOW}


@nb.njit(cache=True)
def _kernel(
    log_values: np.ndarray, position_within_entity: np.ndarray, pairs: np.ndarray, dilations: np.ndarray
):
    n_rows = log_values.shape[0]
    n_p = pairs.shape[0]
    n_d = dilations.shape[0]
    out = np.zeros((n_p * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_p):
            for d in range(n_d):
                dilation = dilations[d]
                ws_short = resolve_window_size(pos, pairs[j, 0], dilation)
                ws_long = resolve_window_size(pos, pairs[j, 1], dilation)
                s_short = fit_linear_trend_slope(log_values, row_idx, ws_short, dilation)
                s_long = fit_linear_trend_slope(log_values, row_idx, ws_long, dilation)
                out[k, row_idx] = safe_ratio(s_short, s_long)
                k += 1
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"pairs": [[6, 12]], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    pairs = np.array(params['pairs'], dtype=np.int64)
    dilations = np.array(params.get('dilations', [1]), dtype=np.int64)
    # log1p считается один раз на колонку, без буфера на каждое окно
    log_values = np.log1p(np.abs(values))
    out = _kernel(log_values, position, pairs, dilations)
    suffixes = [
        f'w{a}_w{b}' if d == 1 else f'w{a}_w{b}_d{d}' for a, b in params['pairs'] for d in params.get('dilations', [1])
    ]
    return [out[k] for k in range(len(suffixes))], suffixes
