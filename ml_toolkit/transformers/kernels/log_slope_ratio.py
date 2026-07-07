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

Preset (monthly.yaml):
    log_slope_ratio:
      pairs:
        - [6, 12]

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

from .._windowing import fit_linear_trend_slope, resolve_window_size, safe_ratio

FEATURE = 'log_slope_ratio'


@nb.njit(cache=True)
def _kernel(log_values: np.ndarray, position_within_entity: np.ndarray, pairs: np.ndarray):
    n_rows = log_values.shape[0]
    n_p = pairs.shape[0]
    out = np.zeros((n_p, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_p):
            ws_short = resolve_window_size(pos, pairs[j, 0])
            ws_long = resolve_window_size(pos, pairs[j, 1])
            s_short = fit_linear_trend_slope(log_values, row_idx, ws_short)
            s_long = fit_linear_trend_slope(log_values, row_idx, ws_long)
            out[j, row_idx] = safe_ratio(s_short, s_long)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"pairs": [[6, 12]]}"""
    pairs = np.array(params['pairs'], dtype=np.int64)
    # log1p считается один раз на колонку, без буфера на каждое окно
    log_values = np.log1p(np.abs(values))
    out = _kernel(log_values, position, pairs)
    p = params['pairs']
    return [out[j] for j in range(len(p))], [f'w{a}_w{b}' for a, b in p]
