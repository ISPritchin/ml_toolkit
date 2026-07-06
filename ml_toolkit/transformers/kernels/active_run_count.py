"""Число отдельных вспышек активности (переходов 0→ненулевое) в окне.

Signal:
    Считает, сколько раз клиент «включился» после паузы в течение окна.
    Высокое значение — дробный пульсирующий паттерн (много отдельных сделок/проектов).
    Низкое при ненулевых active_months — непрерывный поток.

Formula:
    active_run_count_w = count(transitions 0 -> nonzero in [t-w+1..t])
    Переход: v[i] != 0 and v[i-1] == 0

Outputs:
    {product}__active_run_count__w6   — число вспышек за 6 мес
    {product}__active_run_count__w12  — число вспышек за 12 мес

Preset (monthly.yaml):
    active_run_count:
      windows: [6, 12]

Interpretation:
    active_run_count_w12 = 1 при active_months_w12 = 12 — непрерывный клиент.
    active_run_count_w12 = 4 при active_months_w12 = 4 — одиночные всплески (B2B-проекты).
    active_run_count_w12 = 8–10 при active_months_w12 = 10 — почти каждый месяц активен,
    но есть короткие паузы, типичные для сезонного ритма.

Example:
    Ряд (6 мес): [10, 0, 5, 0, 8, 3]
    (t=5, w=6; окно охватывает весь ряд)

    переходы 0→ненулевое:
      i=0: 10≠0, prev=нет        → вспышка #1
      i=2: 5≠0, prev(i=1)=0      → вспышка #2
      i=4: 8≠0, prev(i=3)=0      → вспышка #3
    → active_run_count__w6 = 3
"""

import numba as nb
import numpy as np

from .._windowing import resolve_window_size

FEATURE = "active_run_count"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            count = 0
            prev_active = False
            for offset in range(ws):
                is_active = product_values[row_idx - ws + 1 + offset] != 0.0
                if is_active and not prev_active:
                    count += 1
                prev_active = is_active
            out[j, row_idx] = count
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
