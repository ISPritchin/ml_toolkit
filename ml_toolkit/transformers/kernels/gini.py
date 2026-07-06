"""Коэффициент Джини неравномерности объёмов внутри скользящего окна.

Signal:
    Измеряет неравенство между месяцами: 0 — идеально равномерное распределение,
    1 — весь объём в одном месяце. Высокий gini при высоком mean — клиент с редкими
    крупными платежами (проектный B2B), низкий — равномерный поток.

Formula:
    S_w = sum(v[i], i in окне)
    gini_w = sum(|v[i] - v[j]|, i,j in окне) / (2 * ws * S_w)
          = sum((2i - ws - 1) * v_sorted[i], i=1..ws) / (ws * S_w)

    Вычисляется через сортировку окна за O(ws log ws) (эквивалентная формула
    по отсортированному буферу вместо перебора пар O(ws²)). При S_w <= eps → 0.

Outputs:
    {product}__gini__w6   — коэф. Джини за 6 мес
    {product}__gini__w12  — коэф. Джини за 12 мес

Preset (monthly.yaml):
    gini:
      windows: [6, 12]

Interpretation:
    gini_w12 ≈ 0.28 для пульсирующего клиента [50,0,0,80,0,0,100,0,0,60,0,0].
    gini_w12 ≈ 0 для равномерного потока (плоский ряд).
    gini_w12 > 0.6 — экстремальная концентрация (1-2 месяца дают почти весь доход).
    В паре с entropy: оба меряют концентрацию, gini чувствителен к ненулевому разбросу.

Example:
    Ряд (4 мес): [0, 10, 30, 60],  w=4
    S = 100

    Все |v[i]−v[j]| для i,j ∈ {0,1,2,3}:
      (0,10)=10, (0,30)=30, (0,60)=60, (10,30)=20, (10,60)=50, (30,60)=30
      Сумма попарных = 2·(10+30+60+20+50+30) = 400
    gini = 400 / (2·4·100) = 400/800 = 0.5
    → gini__w4 = 0.50
"""

import numba as nb
import numpy as np

from .._windowing import EPS, compute_window_sum, fill_window_sorted, resolve_window_size

FEATURE = "gini"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    max_w = 1
    for j in range(n_w):
        if windows[j] > max_w:
            max_w = windows[j]
    sorted_buf = np.empty(max_w)
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            win_sum = compute_window_sum(product_values, row_idx, ws)
            if win_sum > EPS:
                fill_window_sorted(sorted_buf, product_values, row_idx, ws)
                gini_num = 0.0
                for i in range(ws):
                    gini_num += (2 * (i + 1) - ws - 1) * sorted_buf[i]
                out[j, row_idx] = gini_num / (ws * win_sum)
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12, 24]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
