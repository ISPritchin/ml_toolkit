"""Текущее значение как доля максимума скользящего окна.

Signal:
    Показывает, где текущий месяц находится относительно лучшего месяца окна.
    Значение = 1.0 — текущий месяц является максимальным внутри окна. Снижение
    говорит об откате от недавнего пика.

Formula:
    hi_w = max(v[t-w+1..t])
    pct_of_max_w = safe_ratio(v[t], hi_w) = v[t] / |hi_w|

    При hi_w ~ 0 (все месяцы окна нулевые) результат = 0 — «доля не определена».

Outputs:
    {product}__pct_of_max__w6   — текущее / max за 6 мес
    {product}__pct_of_max__w12  — текущее / max за 12 мес
    {product}__pct_of_max__w24  — текущее / max за 24 мес

Preset (monthly.yaml):
    pct_of_max:
      windows: [6, 12, 24]

Interpretation:
    = 1.0 — текущий месяц является пиком окна (is_new_peak ↔ distance_to_global_max = 0).
    = 0.5 — вдвое ниже наилучшего за период.
    pct_of_max_w6 = 1, pct_of_max_w24 = 0.4 — краткосрочный пик, долгосрочно ниже.
    Для ряда D на последнем шаге: pct_of_max_w12 = 90/90 = 1.0 (полное восстановление).

Example:
    Ряд (6 мес): [10, 20, 90, 40, 30, 45],  w=6

    hi = max(окна) = 90
    pct_of_max = v[t] / hi = 45 / 90
    → pct_of_max__w6 = 0.5  (текущий мес. — половина пикового)
"""

import numba as nb
import numpy as np

from .._windowing import compute_window_min_and_max, resolve_window_size, safe_ratio

FEATURE = "pct_of_max"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        v = product_values[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            _, hi = compute_window_min_and_max(product_values, row_idx, ws)
            out[j, row_idx] = safe_ratio(v, hi)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
