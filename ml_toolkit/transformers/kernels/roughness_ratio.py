"""Отношение нормированной TV на коротком окне к нормированной TV на длинном.

Signal:
    Показывает, нарастает ли «шероховатость» ряда в последнее время. Значение > 1
    — краткосрочный ряд более рваный, чем долгосрочный (нестабильность нарастает).
    < 1 — недавние месяцы спокойнее исторического фона.

Formula:
    TV_w = sum(|v[i] - v[i-1]|, i in [t-w+2..t])
    TV_norm_w = TV_w / (|mean_w| + eps)
    roughness_ratio_wS_wL = TV_norm_short / (TV_norm_long + eps)

Outputs:
    {product}__roughness_ratio__w6_w12  — TV_norm_6 / TV_norm_12

Preset (monthly.yaml):
    roughness_ratio:
      pairs:
        - [6, 12]

Interpretation:
    = 1.0 — шероховатость стабильна на обоих горизонтах.
    > 2.0 — краткосрочный период вдвое «рваней» долгосрочного (нестабильность нарастает).
    < 0.5 — последние месяцы значительно спокойнее: стабилизация.
    Для гладкого ряда G: TV_norm низкое и стабильное, ratio ≈ 1.
    Для ряда, резко ставшего волатильным: ratio >> 1 — предупреждение о нестабильности.

Example:
    Ряд (6 мес): [40, 40, 40, 40, 10, 40],  пара (3, 6)

    короткое окно (посл. 3 [40,10,40]): TV = 30+30 = 60, mean = 30 → TV_norm = 2.0
    длинное окно (все 6): TV = 0+0+0+30+30 = 60, mean = 35 → TV_norm = 1.714
    roughness_ratio = 2.0 / 1.714 = 1.167
    → roughness_ratio__w3_w6 = 1.167  (краткосрочно чуть «рваней»)

"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean, resolve_window_size, safe_ratio

FEATURE = 'roughness_ratio'


@nb.njit(cache=True)
def _tv_norm(product_values: np.ndarray, row_idx: int, ws: int, mean: float) -> float:
    tv = 0.0
    for offset in range(1, ws):
        abs_idx = row_idx - ws + 1 + offset
        tv += abs(product_values[abs_idx] - product_values[abs_idx - 1])
    return safe_ratio(tv, mean)


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, pairs: np.ndarray):
    n_rows = product_values.shape[0]
    n_p = pairs.shape[0]
    out = np.zeros((n_p, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_p):
            ws_short = resolve_window_size(pos, pairs[j, 0])
            ws_long = resolve_window_size(pos, pairs[j, 1])
            mean_short = compute_window_mean(product_values, row_idx, ws_short)
            mean_long = compute_window_mean(product_values, row_idx, ws_long)
            tv_short = _tv_norm(product_values, row_idx, ws_short, mean_short)
            tv_long = _tv_norm(product_values, row_idx, ws_long, mean_long)
            out[j, row_idx] = safe_ratio(tv_short, tv_long)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"pairs": [[6, 12]]}"""
    pairs = np.array(params['pairs'], dtype=np.int64)
    out = _kernel(values, position, pairs)
    p = params['pairs']
    return [out[j] for j in range(len(p))], [f'w{a}_w{b}' for a, b in p]
