"""Отношение OLS-наклонов: slope(short) / |slope(long)| — ускорение тренда.

Signal:
    Сравнивает скорость изменения на коротком и длинном горизонте. Значение > 1 —
    краткосрочный тренд быстрее долгосрочного (ускорение). Отрицательное — смена знака:
    тренды разнонаправленны. Полезен для обнаружения разворотов и всплесков роста.

Formula:
    slope_short = OLS_slope(v[t-ws_short+1..t])
    slope_long  = OLS_slope(v[t-ws_long+1..t])
    slope_ratio_wS_wL = slope_short / (|slope_long| + eps)

Outputs:
    {product}__slope_ratio__w6_w12   — slope_6 / |slope_12|
    {product}__slope_ratio__w12_w24  — slope_12 / |slope_24|

Preset entry:
    slope_ratio:
      pairs:
        - [6, 12]
        - [12, 24]
      dilations: [1, 2]   # optional, default [1]; applies to both windows of every pair

Interpretation:
    > 1 — краткосрочный наклон круче долгосрочного (ускорение роста или падения).
    ≈ 1 — темп стабилен на обоих горизонтах.
    < 0 — разнонаправленные тренды: краткосрочный разворот.
    w6_w12 > 2 и w12_w24 > 1 — ускорение нарастает на нескольких горизонтах.

Example:
    Ряд (6 мес): [10, 12, 14, 20, 30, 45],  пара (3, 6)

    slope_short (окно 3, посл. [20,30,45]) = 12.5
    slope_long  (окно 6, весь ряд)         = 6.714
    slope_ratio = 12.5 / 6.714 = 1.862
    → slope_ratio__w3_w6 = 1.862  (краткосрочный тренд почти вдвое круче)

"""

import numba as nb
import numpy as np

from ml_toolkit.transformers._windowing import (
    FILL_NAN_UNBOUNDED_LOW,
    fit_linear_trend_slope,
    resolve_window_size,
    safe_ratio,
)

FEATURE = 'slope_ratio'
FILL_NAN: dict[str | None, float] = {None: FILL_NAN_UNBOUNDED_LOW}


@nb.njit(cache=True)
def _kernel(
    product_values: np.ndarray,
    position_within_entity: np.ndarray,
    short_windows: np.ndarray,
    long_windows: np.ndarray,
    dilations: np.ndarray,
):
    n_rows = product_values.shape[0]
    n_pairs = short_windows.shape[0]
    n_d = dilations.shape[0]
    out = np.zeros((n_pairs * n_d, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        k = 0
        for j in range(n_pairs):
            for d in range(n_d):
                dilation = dilations[d]
                ws_s = resolve_window_size(pos, short_windows[j], dilation)
                ws_l = resolve_window_size(pos, long_windows[j], dilation)
                s_short = fit_linear_trend_slope(product_values, row_idx, ws_s, dilation)
                s_long = fit_linear_trend_slope(product_values, row_idx, ws_l, dilation)
                out[k, row_idx] = safe_ratio(s_short, s_long)
                k += 1
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"pairs": [[6, 12], [12, 24]], "dilations": [1, 2]}. "dilations" optional, default [1]."""
    pairs = params['pairs']
    dilations_list = params.get('dilations', [1])
    short_windows = np.array([p[0] for p in pairs], dtype=np.int64)
    long_windows = np.array([p[1] for p in pairs], dtype=np.int64)
    dilations = np.array(dilations_list, dtype=np.int64)
    out = _kernel(values, position, short_windows, long_windows, dilations)
    suffixes = [f'w{p[0]}_w{p[1]}' if d == 1 else f'w{p[0]}_w{p[1]}_d{d}' for p in pairs for d in dilations_list]
    return [out[k] for k in range(len(suffixes))], suffixes
