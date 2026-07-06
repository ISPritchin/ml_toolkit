"""Суммарная вариация (шероховатость): ненормированная и нормированная на уровень.

Signal:
    Суммарная вариация (TV) — суммарный абсолютный путь ряда за период. Высокое TV
    означает «рваный», нестабильный ряд. TV_norm нормирует на уровень, что позволяет
    сравнивать клиентов разного масштаба. В отличие от std: TV накапливает все движения.

Formula:
    TV_w     = sum(|v[i] - v[i-1]|, i in [t-w+2..t])
    mean_w   = mean(v[t-w+1..t])
    TV_norm_w = TV_w / (|mean_w| + eps)

    TV имеет размерность исходной колонки; TV_norm безразмерна.

Outputs:
    {product}__total_variation__w6        — TV за 6 мес (в единицах продукта)
    {product}__total_variation__norm_w6   — TV_norm за 6 мес
    {product}__total_variation__w12       — TV за 12 мес
    {product}__total_variation__norm_w12  — TV_norm за 12 мес

Preset (monthly.yaml):
    total_variation:
      windows: [6, 12]

Interpretation:
    TV_norm = 0 — абсолютно гладкий ряд (плоский или нет изменений).
    TV_norm_w12 = 1.47 для линейного ряда G (TV=55 при mean=37.5).
    TV_norm_w12 = 16.7 для осциллирующего ряда V (TV≈836 при mean=50).
    TV_norm_w6 >> TV_norm_w12 — нарастающая нестабильность в последнее время.

Example:
    Ряд (6 мес): [10, 30, 20, 40, 30, 50],  w=6

    |приращения|: 20, 10, 20, 10, 20 → TV = 80
    mean = 180/6 = 30
    TV_norm = 80 / 30 = 2.667
    → total_variation__w6 = 80,  norm_w6 = 2.667
"""

import numba as nb
import numpy as np

from .._windowing import compute_window_mean, resolve_window_size, safe_ratio

FEATURE = "total_variation"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out_tv = np.zeros((n_w, n_rows))
    out_tv_norm = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            mean = compute_window_mean(product_values, row_idx, ws)
            tv = 0.0
            for offset in range(1, ws):
                abs_idx = row_idx - ws + 1 + offset
                tv += abs(product_values[abs_idx] - product_values[abs_idx - 1])
            out_tv[j, row_idx] = tv
            out_tv_norm[j, row_idx] = safe_ratio(tv, mean)
    return out_tv, out_tv_norm


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [6, 12]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out_tv, out_tv_norm = _kernel(values, position, windows)
    arrays = []
    suffixes = []
    for j, w in enumerate(params["windows"]):
        arrays.append(out_tv[j])
        suffixes.append(f"w{w}")
        arrays.append(out_tv_norm[j])
        suffixes.append(f"norm_w{w}")
    return arrays, suffixes
