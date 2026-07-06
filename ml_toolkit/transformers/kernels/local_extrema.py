"""Число локальных экстремумов (пиков и впадин) в скользящем окне.

Signal:
    Подсчитывает, сколько раз ряд менял направление внутри окна. Высокое значение
    — хаотический, осциллирующий ряд (много разворотов). Низкое — монотонный тренд
    или плато. Дополняет alternation_rate: та считает долю смен, эта — абсолютное число.

Formula:
    Для каждой внутренней позиции offset in [1..ws-2]:
        локальный максимум: v[t-ws+1+offset] > v[t-ws+offset] AND v[t-ws+1+offset] > v[t-ws+2+offset]
        локальный минимум:  v[t-ws+1+offset] < v[t-ws+offset] AND v[t-ws+1+offset] < v[t-ws+2+offset]
    n_local_extrema_w = count(peaks + troughs)

    Требует ws >= 3.

Outputs:
    {product}__local_extrema__w6   — число локальных экстремумов за 6 мес
    {product}__local_extrema__w12  — число локальных экстремумов за 12 мес

Preset (monthly.yaml):
    local_extrema:
      windows: [6, 12]

Interpretation:
    n_local_extrema_w12 = 0 — монотонный тренд, ни одного разворота.
    n_local_extrema_w12 ≈ 5–6 — сильная осцилляция (похоже на V-ряд чётные/нечётные).
    n_local_extrema_w12 = 1–2 — один разворот: возможно U-образный или одиночный пик.
    Полезно вместе с peak_trough_timing для датировки последнего разворота.

Example:
    Ряд (6 мес): [10, 30, 20, 40, 30, 50],  w=6
    (t=5; проверяются внутренние позиции offset=1..4)

    idx1 (30): >10 и >20 → пик
    idx2 (20): <30 и <40 → впадина
    idx3 (40): >20 и >30 → пик
    idx4 (30): <40 и <50 → впадина
    → local_extrema__w6 = 4  (сильно осциллирующий ряд)
"""

import numba as nb
import numpy as np

from .._windowing import resolve_window_size

FEATURE = "local_extrema"


@nb.njit(cache=True)
def _kernel(product_values: np.ndarray, position_within_entity: np.ndarray, windows: np.ndarray):
    n_rows = product_values.shape[0]
    n_w = windows.shape[0]
    out = np.zeros((n_w, n_rows))
    for row_idx in range(n_rows):
        pos = position_within_entity[row_idx]
        for j in range(n_w):
            ws = resolve_window_size(pos, windows[j])
            if ws >= 3:
                count = 0
                for offset in range(1, ws - 1):
                    prev = product_values[row_idx - ws + offset]
                    cur = product_values[row_idx - ws + 1 + offset]
                    nxt = product_values[row_idx - ws + 2 + offset]
                    if (cur > prev and cur > nxt) or (cur < prev and cur < nxt):
                        count += 1
                out[j, row_idx] = count
    return out


def compute(values: np.ndarray, position: np.ndarray, params: dict):
    """params: {"windows": [12]}"""
    windows = np.array(params["windows"], dtype=np.int64)
    out = _kernel(values, position, windows)
    return [out[j] for j in range(len(windows))], [f"w{w}" for w in params["windows"]]
